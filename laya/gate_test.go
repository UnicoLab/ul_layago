//go:build unix

package laya

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"runtime"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/UnicoLab/ul_layago/model"
	"github.com/UnicoLab/ul_layago/tensor"
)

// gateConfig is the M0-LAYA gate run, read from LAYAGO_GATE_* variables.
type gateConfig struct {
	Out          string    `json:"-"`
	Tokens       []int     `json:"tokens"`
	RPS          []float64 `json:"offered_rps"`
	Seconds      float64   `json:"seconds_per_cell"`
	Workers      int       `json:"intra_op_workers"`
	Inflight     int       `json:"max_inflight_passes"`
	Kernel       string    `json:"kernel_requested"`
	Path         string    `json:"path"`
	Layers       int       `json:"layers"`
	BudgetMs     float64   `json:"budget_p95_ms"`
	BudgetTokens int       `json:"budget_tokens"`
	Seed         uint64    `json:"seed"`
}

func gateEnv(name, def string) string {
	if v := strings.TrimSpace(os.Getenv(name)); v != "" {
		return v
	}
	return def
}

func parseGateConfig() (gateConfig, error) {
	c := gateConfig{Out: os.Getenv("LAYAGO_GATE_OUT")}
	for _, f := range strings.Split(gateEnv("LAYAGO_GATE_TOKENS", "32,64,128,256"), ",") {
		n, err := strconv.Atoi(strings.TrimSpace(f))
		if err != nil || n <= 0 || n > 8192 {
			return c, fmt.Errorf("LAYAGO_GATE_TOKENS: bad value %q", f)
		}
		c.Tokens = append(c.Tokens, n)
	}
	for _, f := range strings.Split(gateEnv("LAYAGO_GATE_RPS", "5,10,20"), ",") {
		v, err := strconv.ParseFloat(strings.TrimSpace(f), 64)
		if err != nil || v <= 0 {
			return c, fmt.Errorf("LAYAGO_GATE_RPS: bad value %q", f)
		}
		c.RPS = append(c.RPS, v)
	}
	ints := []struct {
		dst  *int
		name string
		def  int
	}{
		{&c.Workers, "LAYAGO_GATE_WORKERS", runtime.GOMAXPROCS(0)},
		{&c.Inflight, "LAYAGO_GATE_INFLIGHT", 1},
		{&c.Layers, "LAYAGO_GATE_LAYERS", 22},
		{&c.BudgetTokens, "LAYAGO_GATE_BUDGET_TOKENS", 64},
	}
	for _, v := range ints {
		n, err := strconv.Atoi(gateEnv(v.name, strconv.Itoa(v.def)))
		if err != nil || n <= 0 {
			return c, fmt.Errorf("%s must be a positive integer", v.name)
		}
		*v.dst = n
	}
	var err error
	if c.Seconds, err = strconv.ParseFloat(gateEnv("LAYAGO_GATE_SECONDS", "120"), 64); err != nil || c.Seconds <= 0 {
		return c, fmt.Errorf("LAYAGO_GATE_SECONDS must be positive")
	}
	if c.BudgetMs, err = strconv.ParseFloat(gateEnv("LAYAGO_GATE_BUDGET_MS", "120"), 64); err != nil || c.BudgetMs <= 0 {
		return c, fmt.Errorf("LAYAGO_GATE_BUDGET_MS must be positive")
	}
	if c.Seed, err = strconv.ParseUint(gateEnv("LAYAGO_GATE_SEED", "1"), 10, 64); err != nil {
		return c, fmt.Errorf("LAYAGO_GATE_SEED must be an unsigned integer")
	}
	c.Kernel = gateEnv("LAYAGO_GATE_KERNEL", "auto")
	if _, err := tensor.ParseKernel(c.Kernel); err != nil {
		return c, err
	}
	c.Path = gateEnv("LAYAGO_GATE_PATH", "fp32")
	if c.Path != "fp32" && c.Path != "int8" {
		return c, fmt.Errorf("LAYAGO_GATE_PATH must be fp32 or int8")
	}
	if c.Layers > 22 {
		return c, fmt.Errorf("LAYAGO_GATE_LAYERS must be at most 22 (the mmBERT-base depth)")
	}
	return c, nil
}

type gateCell struct {
	Tokens         int                `json:"tokens"`
	OfferedRPS     float64            `json:"offered_rps"`
	Requests       int                `json:"requests"`
	Completed      int                `json:"completed"`
	Abandoned      int                `json:"abandoned_after_drain_deadline"`
	AchievedRPS    float64            `json:"achieved_rps"`
	LatencyMs      map[string]float64 `json:"latency_ms_from_scheduled_arrival"`
	ServiceMs      map[string]float64 `json:"service_ms"`
	Utilization    float64            `json:"busy_fraction_of_inflight_slots"`
	CPUSeconds     float64            `json:"process_cpu_seconds"`
	LoadAvgStart   string             `json:"load_average_at_start"`
	BudgetVerdict  string             `json:"budget_verdict,omitempty"`
	ElapsedSeconds float64            `json:"elapsed_seconds"`
}

func msQuantiles(d []time.Duration) map[string]float64 {
	if len(d) == 0 {
		return nil
	}
	s := make([]float64, len(d))
	for i, v := range d {
		s[i] = float64(v.Microseconds()) / 1000
	}
	sort.Float64s(s)
	r := func(x float64) float64 { return math.Round(x*100) / 100 }
	return map[string]float64{
		"p50": r(percentile(s, 0.5)), "p90": r(percentile(s, 0.9)), "p95": r(percentile(s, 0.95)),
		"p99": r(percentile(s, 0.99)), "max": r(s[len(s)-1]),
	}
}

// runGateCell offers Poisson arrivals at rps for the configured duration
// and serves them with at most Inflight concurrent passes. Latency runs
// from each request's scheduled arrival, not from when a server picked it
// up, so queueing counts (no coordinated omission). Requests still queued
// once the drain deadline (twice the offered duration) passes are abandoned
// and fail the cell.
func runGateCell(c gateConfig, tokens int, rps float64, pass func()) gateCell {
	rng := rand.New(rand.NewPCG(c.Seed, uint64(tokens)<<20|uint64(rps*1000))) //nolint:gosec // G404: load schedule, not a security use
	dur := time.Duration(c.Seconds * float64(time.Second))
	var arrivals []time.Duration
	for at := 0.0; ; {
		at += rng.ExpFloat64() / rps
		if at >= c.Seconds {
			break
		}
		arrivals = append(arrivals, time.Duration(at*float64(time.Second)))
	}
	cell := gateCell{Tokens: tokens, OfferedRPS: rps, Requests: len(arrivals), LoadAvgStart: loadAverage()}
	queue := make(chan time.Time, len(arrivals))
	var mu sync.Mutex
	var latency, service []time.Duration
	var busy atomic.Int64
	var abandoned atomic.Int64
	start := time.Now()
	deadline := start.Add(2 * dur)
	cpu0 := processCPU()
	var wg sync.WaitGroup
	for i := 0; i < c.Inflight; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for due := range queue {
				if time.Now().After(deadline) {
					abandoned.Add(1)
					continue
				}
				s := time.Now()
				pass()
				end := time.Now()
				busy.Add(int64(end.Sub(s)))
				mu.Lock()
				latency = append(latency, end.Sub(due))
				service = append(service, end.Sub(s))
				mu.Unlock()
			}
		}()
	}
	for _, at := range arrivals {
		due := start.Add(at)
		time.Sleep(time.Until(due))
		queue <- due
	}
	close(queue)
	wg.Wait()
	elapsed := time.Since(start)
	cell.ElapsedSeconds = math.Round(elapsed.Seconds()*100) / 100
	cell.Completed = len(latency)
	cell.Abandoned = int(abandoned.Load())
	cell.AchievedRPS = math.Round(float64(cell.Completed)/elapsed.Seconds()*100) / 100
	cell.LatencyMs = msQuantiles(latency)
	cell.ServiceMs = msQuantiles(service)
	cell.Utilization = math.Round(float64(busy.Load())/(float64(elapsed)*float64(c.Inflight))*1000) / 1000
	cell.CPUSeconds = math.Round((processCPU()-cpu0).Seconds()*100) / 100
	if tokens == c.BudgetTokens {
		switch {
		case cell.Abandoned > 0 || cell.Completed == 0:
			cell.BudgetVerdict = "fail: saturated (the queue did not drain)"
		case cell.LatencyMs["p95"] <= c.BudgetMs:
			cell.BudgetVerdict = fmt.Sprintf("pass: p95 %.1f ms <= %.0f ms", cell.LatencyMs["p95"], c.BudgetMs)
		default:
			cell.BudgetVerdict = fmt.Sprintf("fail: p95 %.1f ms > %.0f ms", cell.LatencyMs["p95"], c.BudgetMs)
		}
	}
	return cell
}

func readTrim(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
}

func buildSetting(key string) string {
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, s := range info.Settings {
			if s.Key == key {
				return s.Value
			}
		}
	}
	return ""
}

// TestM0LayaGate is the M0-LAYA latency gate harness: the Laya encoder at
// the mmBERT-base shape (or its first LAYAGO_GATE_LAYERS layers, for the
// fewer-layer student decision) plus the Laya decision head on one
// four-option question, under open-loop Poisson load, with p50/p95/p99 per
// (tokens, offered rps) and a verdict against the fast-lane budget (p95 <=
// 120 ms at 64 tokens). Random weights: latency depends on shape, not
// values. It runs only when LAYAGO_GATE_OUT is set (a path, or - for
// stdout); internal/layago/README.md gives the exact procedure for the
// production shape (8 vCPU AVX2). Numbers from any other host are not the
// gate.
func TestM0LayaGate(t *testing.T) {
	c, err := parseGateConfig()
	if err != nil {
		t.Fatal(err)
	}
	if c.Out == "" {
		t.Skip("set LAYAGO_GATE_OUT (a path or -) to run the M0-LAYA gate harness")
	}
	kern, _ := tensor.ParseKernel(c.Kernel)
	par := tensor.Parallel{Workers: c.Workers, Kernel: kern}
	cfg := model.MMBERTBase()
	cfg.NumLayers = c.Layers
	cfg.Layers = cfg.Layers[:c.Layers]
	memBefore := readTrim("/sys/fs/cgroup/memory.current")

	coldStart := time.Now()
	w := model.RandomWeights(cfg, c.Seed)
	if c.Path == "int8" {
		w.Quantize()
	}
	enc, err := model.NewEncoder(cfg, w)
	if err != nil {
		t.Fatal(err)
	}
	head := RandomHead(cfg.HiddenSize, 2, 2, c.Seed+6)
	built := time.Since(coldStart)
	packStart := time.Now()
	w.Prepack(par.EffectiveKernel())
	head.Prepack(par.EffectiveKernel())
	packed := time.Since(packStart)

	ctx := context.Background()
	eff := par.EffectiveKernel()
	if err := checkGateKernel(kern, eff, runtime.GOARCH); err != nil {
		t.Fatal(err)
	}
	// Self-check on this CPU before timing anything: the timed kernel
	// against the portable one on one 64-token pass, held to the design's
	// per-layer tolerance.
	selfCheck := map[string]any{"kernel": eff.String()}
	if eff != tensor.KernelGeneric {
		ids := gateIDs(64)
		ref, err := enc.Forward(ctx, [][]int32{ids}, model.Options{Parallel: tensor.Parallel{Workers: c.Workers, Kernel: tensor.KernelGeneric}})
		if err != nil {
			t.Fatal(err)
		}
		got, err := enc.Forward(ctx, [][]int32{ids}, model.Options{Parallel: par})
		if err != nil {
			t.Fatal(err)
		}
		var worst float64
		identical := 0
		for i, v := range got.Hidden[0] {
			if math.Float32bits(v) == math.Float32bits(ref.Hidden[0][i]) {
				identical++
			}
			worst = math.Max(worst, math.Abs(float64(v-ref.Hidden[0][i])))
		}
		selfCheck["elements"] = len(got.Hidden[0])
		selfCheck["bit_identical_to_generic"] = identical
		selfCheck["max_abs_diff_vs_generic"] = worst
		if worst > 1e-3 {
			t.Fatalf("%s differs from the portable kernel by %.3g on this CPU (tolerance 1e-3): not timing it", eff, worst)
		}
	}
	markers := []int{1, 3, 5, 7}
	pass := func(ids []int32) func() {
		return func() {
			res, err := enc.Forward(ctx, [][]int32{ids}, model.Options{Parallel: par})
			if err != nil {
				t.Error(err)
				return
			}
			head.forward([]headInput{{hidden: res.Hidden[0], qtype: Choice, markers: markers}}, par)
		}
	}
	firstStart := time.Now()
	pass(gateIDs(c.Tokens[0]))()
	firstPass := time.Since(firstStart)

	var cells []gateCell
	for _, tokens := range c.Tokens {
		run := pass(gateIDs(tokens))
		for i := 0; i < 3; i++ {
			run()
		}
		for _, rps := range c.RPS {
			cell := runGateCell(c, tokens, rps, run)
			cells = append(cells, cell)
			t.Logf("tokens=%d rps=%.1f: completed %d/%d, p50 %.1f p95 %.1f p99 %.1f ms, busy %.2f %s",
				tokens, rps, cell.Completed, cell.Requests, cell.LatencyMs["p50"], cell.LatencyMs["p95"],
				cell.LatencyMs["p99"], cell.Utilization, cell.BudgetVerdict)
		}
	}
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	gomaxprocs := runtime.GOMAXPROCS(0)
	decision := judgeGate(c, cells, gateHost{Kernel: eff.String(), GOMAXPROCS: gomaxprocs, GoVersion: runtime.Version()})
	t.Logf("M0-LAYA rule: %s (candidate %q, %d layers); deviations %v; failing %v; %s",
		decision.Verdict, decision.Candidate, decision.Layers, decision.Deviations, decision.FailingCells, decision.ParityPrerequisite)
	receipt := map[string]any{
		"contract":     "hive.layago-m0-gate-receipt/v1",
		"generated_at": time.Now().UTC().Format(time.RFC3339),
		"gate": fmt.Sprintf("M0-LAYA: fast-lane p95 <= %.0f ms at %d tokens under open-loop load on the production shape (8 vCPU AVX2); "+
			"decision is this run read against the pre-registered rule, the parity prerequisite still applies, and the ship/student decision is recorded separately", c.BudgetMs, c.BudgetTokens),
		"config":            c,
		"preregistration":   gatePreregistration,
		"decision":          decision,
		"kernel_self_check": selfCheck,
		"host": map[string]any{
			"cpu":              cpuModel(),
			"goos":             runtime.GOOS,
			"goarch":           runtime.GOARCH,
			"goamd64":          buildSetting("GOAMD64"),
			"logical_cpus":     runtime.NumCPU(),
			"gomaxprocs":       gomaxprocs,
			"go_version":       runtime.Version(),
			"kernel_effective": par.EffectiveKernel().String(),
			"kernels":          kernelNames(tensor.Kernels()),
			"cgroup_cpu_max":   readTrim("/sys/fs/cgroup/cpu.max"),
			"cloud_run":        strings.TrimSpace(os.Getenv("K_SERVICE") + " " + os.Getenv("CLOUD_RUN_JOB") + " " + os.Getenv("K_REVISION") + " " + os.Getenv("CLOUD_RUN_EXECUTION")),
			"load_avg_end":     loadAverage(),
		},
		"startup": map[string]any{
			"random_model_build_seconds": math.Round(built.Seconds()*100) / 100,
			"prepack_seconds":            math.Round(packed.Seconds()*1000) / 1000,
			"first_pass_ms":              math.Round(float64(firstPass.Microseconds())) / 1000,
			"note":                       "random weights are generated, not read: a real artifact's load and digest time must be measured separately",
		},
		"memory": map[string]any{
			"cgroup_memory_current_before": memBefore,
			"cgroup_memory_current_after":  readTrim("/sys/fs/cgroup/memory.current"),
			"cgroup_memory_peak":           readTrim("/sys/fs/cgroup/memory.peak"),
			"go_heap_mb":                   math.Round(float64(ms.HeapAlloc) / (1 << 20)),
			"go_sys_mb":                    math.Round(float64(ms.Sys) / (1 << 20)),
			"packed_weight_mb":             math.Round(float64(w.PackedBytes())/(1<<20)*10) / 10,
		},
		"model": fmt.Sprintf("mmBERT-base shape truncated to %d layers (hidden 768, 12 heads, GeGLU 1152, vocab 256000, global attention on layers 0, 3, 6, ...), %s weights, plus the Laya head (2 layers) on one 4-option choice question", c.Layers, c.Path),
		"cells": cells,
	}
	writeReceipt(t, c.Out, "LAYAGO_GATE_RECEIPT", receipt)
}

// checkGateKernel refuses a run that would time a different kernel from
// the one asked for: auto on amd64 must find the AVX2 kernel, and a named
// assembly kernel this CPU cannot run (which EffectiveKernel quietly turns
// into generic) is an error, not a portable-kernel timing.
func checkGateKernel(requested, effective tensor.Kernel, goarch string) error {
	switch {
	case requested == tensor.KernelAuto && effective == tensor.KernelGeneric && goarch == "amd64":
		return fmt.Errorf("no AVX2+FMA kernel on this CPU, so the run would time the portable kernel; set LAYAGO_GATE_KERNEL=generic to do that on purpose")
	case requested != tensor.KernelAuto && requested != effective:
		return fmt.Errorf("LAYAGO_GATE_KERNEL=%s is not available on this CPU/build (it would run %s): not timing it", requested, effective)
	}
	return nil
}

func gateIDs(n int) []int32 {
	out := make([]int32, n)
	for i := range out {
		out[i] = int32(5 + (i*7919)%250000)
	}
	return out
}

// The harness's own logic: arrivals are Poisson at the offered rate,
// latency counts queueing from the scheduled arrival, and an overloaded
// cell is marked saturated instead of reporting only the requests it
// finished.
func TestGateCellCountsQueueingAndSaturation(t *testing.T) {
	c := gateConfig{Seconds: 0.5, Inflight: 1, BudgetMs: 500, BudgetTokens: 8, Seed: 3}
	fast := runGateCell(c, 8, 40, func() { time.Sleep(time.Millisecond) })
	if fast.Requests < 8 || fast.Requests > 40 || fast.Completed != fast.Requests || fast.Abandoned != 0 {
		t.Fatalf("light load: %+v", fast)
	}
	if !strings.HasPrefix(fast.BudgetVerdict, "pass") || fast.LatencyMs["p50"] < 1 {
		t.Fatalf("light load verdict %q, latency %v", fast.BudgetVerdict, fast.LatencyMs)
	}
	// Offered 40 rps against a 100 ms service (capacity 10 rps): the queue
	// grows, latency from arrival far exceeds service time, and the drain
	// deadline cuts the backlog.
	slow := runGateCell(c, 8, 40, func() { time.Sleep(100 * time.Millisecond) })
	if slow.Abandoned == 0 || !strings.HasPrefix(slow.BudgetVerdict, "fail: saturated") {
		t.Fatalf("overload not detected: %+v", slow)
	}
	if slow.LatencyMs["p95"] < 3*slow.ServiceMs["p95"] {
		t.Fatalf("queueing not counted: latency %v service %v", slow.LatencyMs, slow.ServiceMs)
	}
	if other := runGateCell(c, 16, 40, func() {}); other.BudgetVerdict != "" {
		t.Fatalf("only the budget token count gets a verdict, got %q", other.BudgetVerdict)
	}
}

// The harness never times a kernel other than the one requested.
func TestGateRefusesAKernelItCannotRun(t *testing.T) {
	for _, c := range []struct {
		requested, effective tensor.Kernel
		goarch               string
		ok                   bool
	}{
		{tensor.KernelAuto, tensor.KernelAVX2, "amd64", true},
		{tensor.KernelAuto, tensor.KernelGeneric, "amd64", false},
		{tensor.KernelAuto, tensor.KernelNEON, "arm64", true},
		{tensor.KernelGeneric, tensor.KernelGeneric, "amd64", true},
		{tensor.KernelAVX2, tensor.KernelAVX2, "amd64", true},
		{tensor.KernelAVX2, tensor.KernelGeneric, "amd64", false},
		{tensor.KernelNEON, tensor.KernelGeneric, "amd64", false},
	} {
		if err := checkGateKernel(c.requested, c.effective, c.goarch); (err == nil) != c.ok {
			t.Errorf("requested %s, effective %s on %s: err %v, want ok=%v", c.requested, c.effective, c.goarch, err, c.ok)
		}
	}
	// And the real resolution: a kernel this build cannot run is refused.
	for _, k := range []tensor.Kernel{tensor.KernelNEON, tensor.KernelAVX2} {
		if eff := (tensor.Parallel{Kernel: k}).EffectiveKernel(); eff != k {
			if checkGateKernel(k, eff, runtime.GOARCH) == nil {
				t.Errorf("%s resolves to %s here and was not refused", k, eff)
			}
		}
	}
}
