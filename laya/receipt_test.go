//go:build unix

package laya

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/UnicoLab/ul_layago/model"
	"github.com/UnicoLab/ul_layago/tensor"
)

type latencyStats struct {
	Runs      int     `json:"runs"`
	P50Ms     float64 `json:"p50_ms"`
	P95Ms     float64 `json:"p95_ms"`
	MeanMs    float64 `json:"mean_ms"`
	MinMs     float64 `json:"min_ms"`
	MaxMs     float64 `json:"max_ms"`
	CPUMsMean float64 `json:"process_cpu_ms_per_run"`
}

type receiptRow struct {
	Tokens       int                `json:"tokens"`
	Path         string             `json:"path"`
	Workers      int                `json:"intra_op_workers"`
	Scope        string             `json:"scope"`
	GFLOP        float64            `json:"gflop_per_pass,omitempty"`
	GFLOPSAtP50  float64            `json:"gflop_per_second_at_p50,omitempty"`
	Latency      latencyStats       `json:"latency"`
	PhaseMsMean  map[string]float64 `json:"phase_ms_mean,omitempty"`
	LoadAvgStart string             `json:"load_average_at_start"`
}

type concurrencyRow struct {
	Tokens             int          `json:"tokens"`
	Path               string       `json:"path"`
	ConcurrentRequests int          `json:"concurrent_requests"`
	WorkersPerRequest  int          `json:"intra_op_workers_per_request"`
	Latency            latencyStats `json:"per_request_latency"`
	ThroughputPerSec   float64      `json:"passes_per_second"`
	LoadAvgStart       string       `json:"load_average_at_start"`
}

func percentile(sorted []float64, q float64) float64 {
	// Nearest rank, as scripts/evaluation/evaluate_laya.py reports p95.
	idx := int(math.Ceil(q*float64(len(sorted)))) - 1
	return sorted[max(0, min(idx, len(sorted)-1))]
}

func summarize(ms []float64, cpuMs float64) latencyStats {
	s := append([]float64(nil), ms...)
	sort.Float64s(s)
	var sum float64
	for _, v := range s {
		sum += v
	}
	round := func(v float64) float64 { return math.Round(v*1000) / 1000 }
	return latencyStats{
		Runs: len(s), P50Ms: round(percentile(s, 0.5)), P95Ms: round(percentile(s, 0.95)),
		MeanMs: round(sum / float64(len(s))), MinMs: round(s[0]), MaxMs: round(s[len(s)-1]),
		CPUMsMean: round(cpuMs / float64(len(s))),
	}
}

func processCPU() time.Duration {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		return 0
	}
	return time.Duration(ru.Utime.Nano() + ru.Stime.Nano())
}

func command(name string, args ...string) string {
	out, err := exec.Command(name, args...).Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func loadAverage() string {
	if b, err := os.ReadFile("/proc/loadavg"); err == nil {
		return strings.Join(strings.Fields(string(b))[:3], " ")
	}
	return strings.Trim(command("sysctl", "-n", "vm.loadavg"), "{} ")
}

func cpuModel() string {
	if m := command("sysctl", "-n", "machdep.cpu.brand_string"); m != "" {
		return m
	}
	if b, err := os.ReadFile("/proc/cpuinfo"); err == nil {
		for _, line := range strings.Split(string(b), "\n") {
			if k, v, ok := strings.Cut(line, ":"); ok && strings.TrimSpace(k) == "model name" {
				return strings.TrimSpace(v)
			}
		}
	}
	return "unknown"
}

// TestWriteM0LatencyReceipt measures the full-size mmBERT-base encoder (22
// layers, d=768, 12 heads, GeGLU 1152, random weights) and the Laya head at
// 32/64/128/256 tokens and writes a receipt. It is a measurement tool, not a
// gate: set LAYAGO_RECEIPT_OUT to the output path to run it, and optionally
// LAYAGO_RECEIPT_RUN_SCALE (default 1) to scale the run counts.
func TestWriteM0LatencyReceipt(t *testing.T) {
	out := os.Getenv("LAYAGO_RECEIPT_OUT")
	if out == "" {
		t.Skip("set LAYAGO_RECEIPT_OUT to write the M0 latency receipt")
	}
	scale := 1.0
	if v, err := strconv.ParseFloat(os.Getenv("LAYAGO_RECEIPT_RUN_SCALE"), 64); err == nil && v > 0 {
		scale = v
	}
	loadStart := loadAverage()
	cfg := model.MMBERTBase()
	built := time.Now()
	fp, err := model.NewEncoder(cfg, model.RandomWeights(cfg, 1))
	if err != nil {
		t.Fatal(err)
	}
	qw := model.RandomWeights(cfg, 1)
	qw.Quantize()
	q8, err := model.NewEncoder(cfg, qw)
	if err != nil {
		t.Fatal(err)
	}
	head := RandomHead(cfg.HiddenSize, 2, 2, 7)
	buildTime := time.Since(built)
	var ms runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&ms)

	ids := func(n int) []int32 {
		out := make([]int32, n)
		for i := range out {
			out[i] = int32(5 + (i*7919)%250000)
		}
		return out
	}
	runsFor := func(tokens, workers int) int {
		base := map[int]int{32: 30, 64: 30, 128: 16, 256: 10}[tokens]
		if workers == 1 {
			base = map[int]int{32: 8, 64: 6, 128: 4, 256: 3}[tokens]
		}
		return max(3, int(float64(base)*scale))
	}
	ctx := context.Background()
	var rows []receiptRow
	for _, tokens := range []int{32, 64, 128, 256} {
		seq := [][]int32{ids(tokens)}
		for _, path := range []string{"fp32", "int8"} {
			enc := fp
			if path == "int8" {
				enc = q8
			}
			for _, workers := range []int{1, 8, runtime.GOMAXPROCS(0)} {
				par := tensor.Parallel{Workers: workers}
				runs := runsFor(tokens, workers)
				for w := 0; w < 2; w++ {
					if _, err := enc.Forward(ctx, seq, model.Options{Parallel: par}); err != nil {
						t.Fatal(err)
					}
				}
				la := loadAverage()
				var wall []float64
				var phases model.Timings
				cpu0 := processCPU()
				for r := 0; r < runs; r++ {
					start := time.Now()
					if _, err := enc.Forward(ctx, seq, model.Options{Parallel: par, Timings: &phases}); err != nil {
						t.Fatal(err)
					}
					wall = append(wall, float64(time.Since(start).Microseconds())/1000)
				}
				cpu := float64((processCPU() - cpu0).Microseconds()) / 1000
				st := summarize(wall, cpu)
				phaseMs := map[string]float64{}
				for name, d := range phases.Map() {
					phaseMs[name] = math.Round(float64(d.Microseconds())/float64(runs)) / 1000
				}
				gflop := cfg.FLOPs(tokens) / 1e9
				rows = append(rows, receiptRow{
					Tokens: tokens, Path: path, Workers: par.N(), Scope: "encoder",
					GFLOP: math.Round(gflop*1000) / 1000, GFLOPSAtP50: math.Round(gflop/(st.P50Ms/1000)*10) / 10,
					Latency: st, PhaseMsMean: phaseMs, LoadAvgStart: la,
				})
				t.Logf("tokens=%d %s workers=%d: p50 %.1f ms p95 %.1f ms (cpu %.1f ms/run, load %s)",
					tokens, path, par.N(), st.P50Ms, st.P95Ms, st.CPUMsMean, la)
			}
		}
		// Encoder plus Laya decision head for one four-option question.
		par := tensor.Parallel{Workers: 8}
		markers := []int{1, 3, 5, 7}
		runs := runsFor(tokens, 8)
		la := loadAverage()
		var wall []float64
		cpu0 := processCPU()
		for r := 0; r < runs+2; r++ {
			start := time.Now()
			res, err := fp.Forward(ctx, seq, model.Options{Parallel: par})
			if err != nil {
				t.Fatal(err)
			}
			head.forward([]headInput{{hidden: res.Hidden[0], qtype: Choice, markers: markers}}, par)
			if r >= 2 {
				wall = append(wall, float64(time.Since(start).Microseconds())/1000)
			} else {
				cpu0 = processCPU()
			}
		}
		st := summarize(wall, float64((processCPU()-cpu0).Microseconds())/1000)
		rows = append(rows, receiptRow{
			Tokens: tokens, Path: "fp32", Workers: 8, Scope: "encoder+laya_head(1 question, 4 options)",
			Latency: st, LoadAvgStart: la,
		})
	}

	// Request-level parallelism: eight single-worker passes at once.
	var conc []concurrencyRow
	for _, tokens := range []int{32, 64} {
		const requests = 8
		per := max(2, int(4*scale))
		seq := [][]int32{ids(tokens)}
		la := loadAverage()
		var mu sync.Mutex
		var wall []float64
		var wg sync.WaitGroup
		start := time.Now()
		cpu0 := processCPU()
		for g := 0; g < requests; g++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for r := 0; r < per; r++ {
					s := time.Now()
					if _, err := fp.Forward(ctx, seq, model.Options{Parallel: tensor.Parallel{Workers: 1}}); err != nil {
						t.Error(err)
						return
					}
					mu.Lock()
					wall = append(wall, float64(time.Since(s).Microseconds())/1000)
					mu.Unlock()
				}
			}()
		}
		wg.Wait()
		elapsed := time.Since(start).Seconds()
		st := summarize(wall, float64((processCPU()-cpu0).Microseconds())/1000)
		conc = append(conc, concurrencyRow{
			Tokens: tokens, Path: "fp32", ConcurrentRequests: requests, WorkersPerRequest: 1,
			Latency: st, ThroughputPerSec: math.Round(float64(len(wall))/elapsed*100) / 100, LoadAvgStart: la,
		})
	}

	receipt := map[string]any{
		"contract":     "hive.layago-m0-latency-receipt/v1",
		"generated_at": time.Now().UTC().Format(time.RFC3339),
		"label": "First M0-LAYA latency numbers for the pure-Go layago encoder on this development machine. " +
			"This is NOT the production gate, which is p50/p95 at 32/64/128/256 tokens on 8 vCPU AVX2 " +
			"(Cloud Run) under open-loop load and has not been measured.",
		"host": map[string]any{
			"cpu":                    cpuModel(),
			"goos":                   runtime.GOOS,
			"goarch":                 runtime.GOARCH,
			"logical_cpus":           runtime.NumCPU(),
			"gomaxprocs":             runtime.GOMAXPROCS(0),
			"go_version":             runtime.Version(),
			"kernel_class":           "generic (portable Go, no assembly or SIMD intrinsics; the compiler fuses multiply-add on arm64)",
			"load_average_start":     loadStart,
			"load_average_end":       loadAverage(),
			"shared_host":            "other build and test workloads ran concurrently; wall time includes that contention, process CPU time per run is the less contended signal",
			"model_build_seconds":    math.Round(buildTime.Seconds()*100) / 100,
			"go_heap_after_build_mb": math.Round(float64(ms.HeapAlloc) / (1 << 20)),
		},
		"model": map[string]any{
			"shape":            "mmBERT-base: 22 layers, hidden 768, 12 heads, GeGLU intermediate 1152, vocab 256000, global attention every 3rd layer, sliding window 128 (|i-j|<=64) elsewhere, RoPE theta 160000",
			"weights":          "deterministic random N(0, 1/in) (latency depends on shape, not values); int8 path quantizes embeddings and all encoder linears per row",
			"laya_head":        "2 norm-first transformer layers (d=768, 12 heads, ReLU FF 3072), marker scorer and act head, random weights",
			"flops_convention": "2 FLOPs per multiply-add; dense layers plus windowed attention products",
		},
		"protocol": map[string]any{
			"single_request":   "one sequence per pass, 2 warm-up passes, then N timed passes; wall clock around Encoder.Forward; nearest-rank percentiles",
			"intra_op_workers": "goroutines per GEMM/attention call; 8 mirrors the 8 vCPU production shape only in thread count",
			"concurrency":      "8 goroutines each running single-worker passes back to back (request-level parallelism)",
		},
		"results":     rows,
		"concurrency": conc,
		"not_measured": []string{
			"8 vCPU AVX2 Cloud Run shape and open-loop Poisson load at 5/10/20 rps",
			"cold start to ready, cgroup memory.current",
			"an AVX2 or NEON assembly microkernel (only the portable Go kernel exists)",
			"the 6-layer and 4-layer student shapes",
		},
	}
	data, err := json.MarshalIndent(receipt, "", " ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(out, append(data, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
	fmt.Println("wrote", out)
}
