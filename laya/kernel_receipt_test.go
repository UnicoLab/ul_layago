//go:build unix

package laya

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"runtime"
	"runtime/metrics"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/UnicoLab/ul_layago/model"
	"github.com/UnicoLab/ul_layago/tensor"
)

type gemmReceiptRow struct {
	Linear      string  `json:"linear"`
	M           int     `json:"m_tokens"`
	N           int     `json:"n_out"`
	K           int     `json:"k_in"`
	Path        string  `json:"path"`
	Kernel      string  `json:"kernel"`
	Runs        int     `json:"runs"`
	MinMs       float64 `json:"min_ms"`
	MedianMs    float64 `json:"median_ms"`
	GFLOPSAtMin float64 `json:"gflop_per_second_at_min"`
}

type encoderKernelRow struct {
	Tokens      int                `json:"tokens"`
	Path        string             `json:"path"`
	Kernel      string             `json:"kernel"`
	Workers     int                `json:"intra_op_workers"`
	GFLOP       float64            `json:"gflop_per_pass"`
	Latency     latencyStats       `json:"latency"`
	CPUMsP50    float64            `json:"process_cpu_ms_p50"`
	GFLOPPerCPU float64            `json:"gflop_per_cpu_second_at_p50"`
	PhaseMsMean map[string]float64 `json:"phase_ms_mean"`
}

type speedupRow struct {
	Tokens     int     `json:"tokens"`
	Path       string  `json:"path"`
	Workers    int     `json:"intra_op_workers"`
	Kernel     string  `json:"kernel"`
	WallP50    float64 `json:"wall_p50_speedup_vs_generic"`
	CPUP50     float64 `json:"process_cpu_p50_speedup_vs_generic"`
	GEMMShare  float64 `json:"generic_gemm_phase_share"`
	LoadAvgRun string  `json:"load_average_at_start"`
}

type quantile struct {
	P50, P99, Max float64
}

func quantiles(v []float64) quantile {
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	r := func(x float64) float64 { return math.Round(x*1000) / 1000 }
	return quantile{P50: r(percentile(s, 0.5)), P99: r(percentile(s, 0.99)), Max: r(s[len(s)-1])}
}

// tickerLateness runs work repeatedly for d with GOMAXPROCS=1 while a second
// goroutine sleeps 1 ms at a time, and returns how late (ms) it woke up. Go
// assembly is not asynchronously preemptible, so an assembly call longer
// than the scheduler's preemption quantum would show up here as lateness
// beyond that of the portable kernel.
func tickerLateness(work func(), d time.Duration) quantile {
	prev := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(prev)
	var lat []float64
	stop := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			select {
			case <-stop:
				return
			default:
			}
			want := time.Now().Add(time.Millisecond)
			time.Sleep(time.Millisecond)
			lat = append(lat, float64(time.Since(want).Microseconds())/1000)
		}
	}()
	for end := time.Now().Add(d); time.Now().Before(end); {
		work()
	}
	close(stop)
	<-done
	return quantiles(lat)
}

// stopCounts accumulates the runtime's stop-the-world stopping latency
// histogram (time from deciding to stop the world until every P has
// stopped) over several measurement windows.
type stopCounts struct {
	buckets []float64
	counts  []uint64
}

func readStopping() *metrics.Float64Histogram {
	s := []metrics.Sample{{Name: "/sched/pauses/stopping/gc:seconds"}}
	metrics.Read(s)
	if s[0].Value.Kind() != metrics.KindFloat64Histogram {
		return nil
	}
	return s[0].Value.Float64Histogram()
}

// measure forces a GC every gcEvery while work runs for d and adds the
// stopping latencies observed to s.
func (s *stopCounts) measure(work func(), d, gcEvery time.Duration) {
	before := readStopping()
	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		t := time.NewTicker(gcEvery)
		defer t.Stop()
		for {
			select {
			case <-stop:
				return
			case <-t.C:
				runtime.GC()
			}
		}
	}()
	for end := time.Now().Add(d); time.Now().Before(end); {
		work()
	}
	close(stop)
	wg.Wait()
	after := readStopping()
	if before == nil || after == nil {
		return
	}
	if s.counts == nil {
		s.buckets = after.Buckets
		s.counts = make([]uint64, len(after.Counts))
	}
	for i := range s.counts {
		s.counts[i] += after.Counts[i] - before.Counts[i]
	}
}

// quantiles returns p50, p99 and max in ms (bucket upper bounds) and the
// number of stops.
func (s *stopCounts) quantiles() (quantile, uint64) {
	var total uint64
	for _, c := range s.counts {
		total += c
	}
	if total == 0 {
		return quantile{}, 0
	}
	at := func(q float64) float64 {
		need := max(1, uint64(math.Ceil(q*float64(total))))
		var seen uint64
		for i, c := range s.counts {
			seen += c
			if seen >= need {
				hi := s.buckets[i+1]
				if math.IsInf(hi, 1) {
					hi = s.buckets[i]
				}
				return math.Round(hi*1e6) / 1e3
			}
		}
		return math.NaN()
	}
	return quantile{P50: at(0.5), P99: at(0.99), Max: at(1)}, total
}

func kernelNames(ks []tensor.Kernel) []string {
	out := make([]string, len(ks))
	for i, k := range ks {
		out[i] = k.String()
	}
	return out
}

// marshalReceipt indents like the M0 receipt and keeps "<" and ">" readable
// in verdict strings.
func marshalReceipt(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", " ")
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func envScale() float64 {
	if v, err := strconv.ParseFloat(os.Getenv("LAYAGO_RECEIPT_RUN_SCALE"), 64); err == nil && v > 0 {
		return v
	}
	return 1
}

// TestWriteM1KernelReceipt compares the portable Go GEMM kernel with the
// assembly kernel of this CPU on the full mmBERT-base shape (22 layers,
// d=768, GeGLU 1152, random weights) and writes a receipt: single-core GEMM
// throughput per projection, encoder latency and CPU time at 32/64/128/256
// tokens (float32 and int8, 1 and 8 workers, kernels interleaved run by run
// so host contention hits both alike), full-size parity between the
// kernels, and preemption evidence (the longest single assembly call, ticker
// lateness at GOMAXPROCS=1 and GC stop-the-world stopping latency). It is a
// measurement tool, not a gate: set LAYAGO_KERNEL_RECEIPT_OUT (a path, or -
// for stdout) to run it.
//
//	LAYAGO_KERNEL_RECEIPT_OUT=/tmp/m1.json go test -timeout 120m -run TestWriteM1KernelReceipt ./internal/layago/laya
func TestWriteM1KernelReceipt(t *testing.T) {
	out := os.Getenv("LAYAGO_KERNEL_RECEIPT_OUT")
	if out == "" {
		t.Skip("set LAYAGO_KERNEL_RECEIPT_OUT to write the M1 kernel receipt")
	}
	kernels := tensor.Kernels()
	if len(kernels) < 2 {
		t.Skipf("only %v is available here; there is no assembly kernel to compare", kernels)
	}
	best := kernels[len(kernels)-1]
	prov := kernelReceiptProvenance(best, cpuModel(), runtime.GOOS, runtime.GOARCH, runtime.GOMAXPROCS(0), translatedByRosetta(), os.Getenv("LAYAGO_RECEIPT_HOST_NOTE"))
	scale := envScale()
	ctx := context.Background()
	loadStart := loadAverage()

	cfg := model.MMBERTBase()
	built := time.Now()
	fpW := model.RandomWeights(cfg, 1)
	fp, err := model.NewEncoder(cfg, fpW)
	if err != nil {
		t.Fatal(err)
	}
	qW := model.RandomWeights(cfg, 1)
	qW.Quantize()
	q8, err := model.NewEncoder(cfg, qW)
	if err != nil {
		t.Fatal(err)
	}
	buildTime := time.Since(built)
	packed := time.Now()
	fpW.Prepack(best)
	qW.Prepack(best)
	packTime := time.Since(packed)
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

	// 1. Single-core GEMM throughput on layer 0's projections.
	var gemm []gemmReceiptRow
	r := rand.New(rand.NewPCG(21, 22)) //nolint:gosec // G404: benchmark inputs
	layer := func(w *model.Weights) map[string]*tensor.Linear {
		l := w.Layers[0]
		return map[string]*tensor.Linear{"Wqkv": l.Wqkv, "Wo": l.Wo, "Wi": l.Wi, "WoM": l.WoM}
	}
	for _, name := range []string{"Wqkv", "Wo", "Wi", "WoM"} {
		for _, m := range []int{64, 256} {
			for _, path := range []string{"fp32", "int8"} {
				lin := layer(fpW)[name]
				if path == "int8" {
					lin = layer(qW)[name]
				}
				x := make([]float32, m*lin.In)
				for i := range x {
					x[i] = float32(r.NormFloat64())
				}
				dst := make([]float32, m*lin.Out)
				runs := max(5, int(15*scale))
				times := map[tensor.Kernel][]float64{}
				for run := -2; run < runs; run++ {
					for _, k := range kernels {
						start := time.Now()
						lin.Forward(dst, x, m, tensor.Parallel{Workers: 1, Kernel: k})
						if run >= 0 {
							times[k] = append(times[k], float64(time.Since(start).Microseconds())/1000)
						}
					}
				}
				for _, k := range kernels {
					s := append([]float64(nil), times[k]...)
					sort.Float64s(s)
					flop := 2 * float64(m*lin.Out*lin.In)
					gemm = append(gemm, gemmReceiptRow{
						Linear: name, M: m, N: lin.Out, K: lin.In, Path: path, Kernel: k.String(), Runs: len(s),
						MinMs: s[0], MedianMs: percentile(s, 0.5),
						GFLOPSAtMin: math.Round(flop/(s[0]/1000)/1e9*10) / 10,
					})
				}
			}
		}
	}

	// 2. Encoder latency, kernels interleaved run by run.
	var rows []encoderKernelRow
	var speedups []speedupRow
	for _, tokens := range []int{32, 64, 128, 256} {
		seq := [][]int32{ids(tokens)}
		gflop := cfg.FLOPs(tokens) / 1e9
		for _, path := range []string{"fp32", "int8"} {
			enc := fp
			if path == "int8" {
				enc = q8
			}
			for _, workers := range []int{1, 8} {
				runs := max(3, int(float64(map[int]int{32: 30, 64: 30, 128: 16, 256: 10}[tokens])*scale))
				if workers == 1 {
					runs = max(3, int(float64(map[int]int{32: 8, 64: 6, 128: 4, 256: 3}[tokens])*scale))
				}
				la := loadAverage()
				wall := map[tensor.Kernel][]float64{}
				cpu := map[tensor.Kernel][]float64{}
				phases := map[tensor.Kernel]*model.Timings{}
				for _, k := range kernels {
					phases[k] = &model.Timings{}
				}
				for run := -2; run < runs; run++ {
					for _, k := range kernels {
						opt := model.Options{Parallel: tensor.Parallel{Workers: workers, Kernel: k}}
						if run >= 0 {
							opt.Timings = phases[k]
						}
						c0, start := processCPU(), time.Now()
						if _, err := enc.Forward(ctx, seq, opt); err != nil {
							t.Fatal(err)
						}
						if run >= 0 {
							wall[k] = append(wall[k], float64(time.Since(start).Microseconds())/1000)
							cpu[k] = append(cpu[k], float64((processCPU()-c0).Microseconds())/1000)
						}
					}
				}
				p50 := map[tensor.Kernel][2]float64{}
				var gemmShare float64
				for _, k := range kernels {
					var cpuSum float64
					for _, v := range cpu[k] {
						cpuSum += v
					}
					st := summarize(wall[k], cpuSum)
					cs := append([]float64(nil), cpu[k]...)
					sort.Float64s(cs)
					cpuP50 := percentile(cs, 0.5)
					phaseMs := map[string]float64{}
					var total, gemmMs float64
					for name, d := range phases[k].Map() {
						v := math.Round(float64(d.Microseconds())/float64(runs)) / 1000
						phaseMs[name] = v
						total += v
						if name == "qkv" || name == "attn_out" || name == "mlp_in" || name == "mlp_out" {
							gemmMs += v
						}
					}
					if k == tensor.KernelGeneric && total > 0 {
						gemmShare = math.Round(gemmMs/total*1000) / 1000
					}
					rows = append(rows, encoderKernelRow{
						Tokens: tokens, Path: path, Kernel: k.String(), Workers: workers,
						GFLOP: math.Round(gflop*1000) / 1000, Latency: st,
						CPUMsP50: math.Round(cpuP50*1000) / 1000, GFLOPPerCPU: math.Round(gflop/(cpuP50/1000)*10) / 10,
						PhaseMsMean: phaseMs,
					})
					p50[k] = [2]float64{st.P50Ms, cpuP50}
					t.Logf("tokens=%d %s workers=%d %s: p50 %.1f ms p95 %.1f ms, cpu p50 %.1f ms (load %s)",
						tokens, path, workers, k, st.P50Ms, st.P95Ms, cpuP50, la)
				}
				g := p50[tensor.KernelGeneric]
				a := p50[best]
				speedups = append(speedups, speedupRow{
					Tokens: tokens, Path: path, Workers: workers, Kernel: best.String(),
					WallP50: math.Round(g[0]/a[0]*100) / 100, CPUP50: math.Round(g[1]/a[1]*100) / 100,
					GEMMShare: gemmShare, LoadAvgRun: la,
				})
			}
		}
	}

	// 3. Full-size parity between the kernels at 64 tokens.
	parity := map[string]any{}
	for _, path := range []string{"fp32", "int8"} {
		enc := fp
		if path == "int8" {
			enc = q8
		}
		ref, err := enc.Forward(ctx, [][]int32{ids(64)}, model.Options{Parallel: tensor.Parallel{Workers: 8, Kernel: tensor.KernelGeneric}})
		if err != nil {
			t.Fatal(err)
		}
		got, err := enc.Forward(ctx, [][]int32{ids(64)}, model.Options{Parallel: tensor.Parallel{Workers: 8, Kernel: best}})
		if err != nil {
			t.Fatal(err)
		}
		var worst float64
		identical := 0
		for i, v := range got.Hidden[0] {
			w := ref.Hidden[0][i]
			if math.Float32bits(v) == math.Float32bits(w) {
				identical++
			}
			worst = math.Max(worst, math.Abs(float64(v-w)))
		}
		parity[path] = map[string]any{
			"elements": len(got.Hidden[0]), "bit_identical": identical, "max_abs_diff": worst,
		}
	}

	// 4. Preemption: the longest single assembly call, ticker lateness at
	// GOMAXPROCS=1 and GC stop-the-world stopping latency.
	mr, nr := best.Tile()
	callK := tensor.MaxCallKSteps()
	one, err := tensor.NewLinear(nr, callK, make([]float32, nr*callK), nil)
	if err != nil {
		t.Fatal(err)
	}
	oneX := make([]float32, mr*callK)
	oneDst := make([]float32, mr*nr)
	var callUs []float64
	for i := 0; i < 2000; i++ {
		start := time.Now()
		one.Forward(oneDst, oneX, mr, tensor.Parallel{Workers: 1, Kernel: best})
		callUs = append(callUs, float64(time.Since(start).Nanoseconds())/1000)
	}
	preempt := map[string]any{
		"longest_call": map[string]any{
			"what":      fmt.Sprintf("one %dx%d tile over K=%d (the per-call cap), timed around Linear.Forward so it includes the Go wrapper: an upper bound on the non-preemptible window", mr, nr, callK),
			"calls":     len(callUs),
			"microsecs": quantiles(callUs),
		},
	}
	// A single-tile pass 512 calls long: only a preemption point at every
	// kernel call keeps the stopping latency far below the tile time.
	longK := 512 * callK
	longW := make([]float32, nr*longK)
	for i := range longW {
		longW[i] = float32(r.NormFloat64())
	}
	long, err := tensor.NewLinear(nr, longK, longW, nil)
	if err != nil {
		t.Fatal(err)
	}
	longX := make([]float32, mr*longK)
	for i := range longX {
		longX[i] = float32(r.NormFloat64())
	}
	longStops := map[tensor.Kernel]*stopCounts{}
	longTileMs := map[tensor.Kernel]float64{}
	seq64 := [][]int32{ids(64)}
	encStops := map[tensor.Kernel]*stopCounts{}
	lateness := map[tensor.Kernel]quantile{}
	prevProcs := runtime.GOMAXPROCS(2)
	for round := 0; round < 3; round++ {
		for _, k := range kernels {
			par := tensor.Parallel{Workers: 1, Kernel: k}
			start := time.Now()
			long.Forward(oneDst, longX, mr, par)
			longTileMs[k] = math.Max(longTileMs[k], float64(time.Since(start).Microseconds())/1000)
			if longStops[k] == nil {
				longStops[k] = &stopCounts{}
			}
			longStops[k].measure(func() { long.Forward(oneDst, longX, mr, par) }, time.Second, 5*time.Millisecond)
		}
	}
	runtime.GOMAXPROCS(prevProcs)
	for round := 0; round < 3; round++ {
		for _, k := range kernels {
			if encStops[k] == nil {
				encStops[k] = &stopCounts{}
			}
			encStops[k].measure(func() {
				if _, err := fp.Forward(ctx, seq64, model.Options{Parallel: tensor.Parallel{Workers: 8, Kernel: k}}); err != nil {
					t.Error(err)
				}
			}, time.Duration(scale*float64(time.Second)), 10*time.Millisecond)
		}
	}
	for _, k := range kernels {
		lateness[k] = tickerLateness(func() {
			if _, err := fp.Forward(ctx, seq64, model.Options{Parallel: tensor.Parallel{Workers: 1, Kernel: k}}); err != nil {
				t.Error(err)
			}
		}, time.Duration(3*scale*float64(time.Second)))
	}
	for _, k := range kernels {
		longQ, longN := longStops[k].quantiles()
		encQ, encN := encStops[k].quantiles()
		preempt[k.String()] = map[string]any{
			"long_tile": map[string]any{
				"what":               fmt.Sprintf("GOMAXPROCS=2; one %dx%d output tile with K=%d (%d assembly calls) per pass, passes back to back, runtime.GC every 5 ms, 3 rounds of 1 s interleaved with the other kernel", mr, nr, longK, longK/callK),
				"tile_ms_max":        longTileMs[k],
				"stw_stopping_ms":    longQ,
				"stops":              longN,
				"no_preemption_hint": "without a preemption point per assembly call the median stopping latency is about half a tile or more (6.3 ms measured with the wrappers removed, TestAssemblyKernelsStopPromptlyForTheRuntime)",
			},
			"encoder_64_tokens_8_workers": map[string]any{
				"stw_stopping_ms": encQ,
				"stops":           encN,
			},
			"ticker_lateness_ms_gomaxprocs_1": lateness[k],
		}
	}

	receipt := map[string]any{
		"contract":     "hive.layago-m1-kernel-receipt/v1",
		"generated_at": time.Now().UTC().Format(time.RFC3339),
		"label":        prov.Label,
		"host": map[string]any{
			"cpu":                        cpuModel(),
			"goos":                       runtime.GOOS,
			"goarch":                     runtime.GOARCH,
			"logical_cpus":               runtime.NumCPU(),
			"gomaxprocs":                 runtime.GOMAXPROCS(0),
			"go_version":                 runtime.Version(),
			"kernels":                    kernelNames(kernels),
			"load_average_start":         loadStart,
			"load_average_end":           loadAverage(),
			"translated":                 prov.Translated,
			"host_note":                  prov.HostNote,
			"contention":                 "wall time includes whatever else ran on the host (see the load averages); process CPU time is the less contended signal, and kernels were interleaved run by run so contention hits both alike",
			"model_build_seconds":        math.Round(buildTime.Seconds()*100) / 100,
			"prepack_seconds":            math.Round(packTime.Seconds()*1000) / 1000,
			"packed_weight_mb":           math.Round(float64(fpW.PackedBytes()+qW.PackedBytes())/(1<<20)*10) / 10,
			"go_heap_after_build_mb":     math.Round(float64(ms.HeapAlloc) / (1 << 20)),
			"tile":                       fmt.Sprintf("%dx%d, at most %d K steps per assembly call", mr, nr, callK),
			"generic_kernel_description": "portable Go 4x4 dot-product tiles; the compiler fuses multiply-add on arm64",
		},
		"model": map[string]any{
			"shape":            "mmBERT-base: 22 layers, hidden 768, 12 heads, GeGLU intermediate 1152, vocab 256000, global attention every 3rd layer, sliding window 128 (|i-j|<=64) elsewhere",
			"weights":          "deterministic random N(0, 1/in); int8 quantizes embeddings and all encoder linears per row",
			"flops_convention": "2 FLOPs per multiply-add; dense layers plus windowed attention products",
		},
		"protocol": map[string]any{
			"gemm":     "one worker, layer-0 projections, 2 warm-up rounds then N rounds, kernels alternating within each round; min and median wall time",
			"encoder":  "one sequence per pass, 2 warm-up rounds, then N rounds with the kernels alternating within each round; wall clock and process CPU time around Encoder.Forward; nearest-rank percentiles",
			"parity":   "final hidden state of one 64-token pass, generic versus assembly, 8 workers",
			"preempt":  "long_tile and encoder rows: /sched/pauses/stopping/gc:seconds (time from deciding to stop the world until every P has stopped) while runtime.GC is forced, kernels alternating over 3 rounds; ticker: a goroutine sleeping 1 ms at a time while single-worker passes run back to back at GOMAXPROCS=1, lateness past its deadline (Parallel.Run yields every 0.5 ms of tasks, so on a dedicated host this stays near 1 ms instead of the runtime's 10 ms preemption; on a shared host the OS's own scheduling adds to it)",
			"speedups": "generic p50 divided by assembly p50, per (tokens, path, workers); generic_gemm_phase_share is the share of the generic pass spent in the four projections, which bounds what any GEMM kernel can save",
		},
		"gemm":         gemm,
		"encoder":      rows,
		"speedups":     speedups,
		"parity":       parity,
		"preemption":   preempt,
		"not_measured": prov.NotMeasured,
	}
	writeReceipt(t, out, "LAYAGO_KERNEL_RECEIPT", receipt)
}

// receiptProvenance is what a kernel receipt says about where it was made.
type receiptProvenance struct {
	Label       string
	HostNote    string
	Translated  bool
	NotMeasured []string
}

// kernelReceiptProvenance derives the receipt's label, host note and gaps
// from the run itself, so a receipt made on the gate's AVX2 host (README
// procedure, run 7) does not carry the development machine's caveats and a
// translated run says so. note is LAYAGO_RECEIPT_HOST_NOTE, the operator's
// statement about the host (shared or dedicated, Cloud Run job, ...).
func kernelReceiptProvenance(best tensor.Kernel, cpu, goos, goarch string, gomaxprocs int, translated bool, note string) receiptProvenance {
	p := receiptProvenance{Translated: translated, HostNote: note}
	where := fmt.Sprintf("%s (%s/%s, GOMAXPROCS %d)", cpu, goos, goarch, gomaxprocs)
	if translated {
		where += " under Rosetta 2"
	}
	if p.HostNote == "" {
		p.HostNote = "not stated (set LAYAGO_RECEIPT_HOST_NOTE to say whether the host was dedicated or shared)"
	}
	p.Label = "Portable Go versus " + best.String() + " GEMM kernel for the layago encoder on " + where + ". " +
		"This is a kernel comparison, NOT the M0-LAYA gate, which is TestM0LayaGate's p50/p95 under open-loop load on 8 vCPU AVX2 " +
		"(internal/layago/README.md, 'M0-LAYA gate procedure')."
	p.NotMeasured = []string{"the M0-LAYA gate: open-loop Poisson load (TestM0LayaGate)"}
	switch {
	case translated:
		p.Label += " Rosetta 2 translates the x86 instructions, so these timings say nothing about native speed."
		p.NotMeasured = append(p.NotMeasured, "native timings: this process was translated by Rosetta 2")
	case best != tensor.KernelAVX2:
		p.NotMeasured = append(p.NotMeasured, "AVX2 timings: this host runs "+best.String()+", not the gate's amd64_avx2 kernel")
	case gomaxprocs != 8:
		p.NotMeasured = append(p.NotMeasured, fmt.Sprintf("the gate's 8 vCPU shape: GOMAXPROCS was %d", gomaxprocs))
	}
	p.NotMeasured = append(p.NotMeasured, "cold start to ready and cgroup memory.current (TestM0LayaGate records them)", "the 6-layer and 4-layer student shapes")
	return p
}

// A kernel receipt describes the host it ran on: the development machine's
// Rosetta caveat must not follow the receipt onto the AVX2 gate host, and a
// translated or non-AVX2 run must say what it cannot show.
func TestKernelReceiptProvenanceFollowsTheHost(t *testing.T) {
	gate := kernelReceiptProvenance(tensor.KernelAVX2, "AMD EPYC 7B13", "linux", "amd64", 8, false, "Cloud Run job, dedicated")
	for _, s := range append([]string{gate.Label, gate.HostNote}, gate.NotMeasured...) {
		if strings.Contains(s, "Rosetta") || strings.Contains(s, "development machine") || strings.Contains(s, "AVX2 timings") {
			t.Errorf("gate-host receipt carries another host's caveat: %q", s)
		}
	}
	if !strings.Contains(gate.Label, "AMD EPYC 7B13 (linux/amd64, GOMAXPROCS 8)") || gate.HostNote != "Cloud Run job, dedicated" {
		t.Errorf("gate-host provenance: %+v", gate)
	}
	neon := kernelReceiptProvenance(tensor.KernelNEON, "Apple M3 Max", "darwin", "arm64", 16, false, "")
	if !strings.Contains(strings.Join(neon.NotMeasured, "|"), "AVX2 timings: this host runs arm64_neon") || !strings.Contains(neon.HostNote, "not stated") {
		t.Errorf("NEON-host provenance: %+v", neon)
	}
	rosetta := kernelReceiptProvenance(tensor.KernelAVX2, "Apple M3 Max", "darwin", "amd64", 16, true, "")
	if !rosetta.Translated || !strings.Contains(rosetta.Label, "under Rosetta 2") || !strings.Contains(strings.Join(rosetta.NotMeasured, "|"), "Rosetta 2") {
		t.Errorf("translated provenance: %+v", rosetta)
	}
	if got := kernelReceiptProvenance(tensor.KernelAVX2, "x", "linux", "amd64", 4, false, ""); !strings.Contains(strings.Join(got.NotMeasured, "|"), "GOMAXPROCS was 4") {
		t.Errorf("4-vCPU AVX2 provenance: %+v", got)
	}
	if translatedByRosetta() && runtime.GOARCH != "amd64" {
		t.Error("a native arm64 process reported itself translated")
	}
	t.Logf("this process is translated by Rosetta 2: %v", translatedByRosetta())
}

// writeReceipt writes the indented receipt to path or, when path is "-"
// (a container whose files are discarded), prints it compact on one stdout
// line prefixed by marker, so one log entry holds the whole receipt.
func writeReceipt(t *testing.T, path, marker string, receipt any) {
	t.Helper()
	if path == "-" {
		var buf bytes.Buffer
		enc := json.NewEncoder(&buf)
		enc.SetEscapeHTML(false)
		if err := enc.Encode(receipt); err != nil {
			t.Fatal(err)
		}
		fmt.Print(marker + " " + buf.String())
		return
	}
	data, err := marshalReceipt(receipt)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	fmt.Println("wrote", path)
}
