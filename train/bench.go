package train

import (
	"context"
	"fmt"
	"math"
	"os"
	"os/exec"
	"runtime"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/UnicoLab/ul_layago/model"
	"github.com/UnicoLab/ul_layago/tensor"
)

// BenchSpec is an M0-LAYA latency run on a loaded trunk.
type BenchSpec struct {
	// Tokens are the sequence lengths (32, 64, 128, 256).
	Tokens []int
	// Layers are trunk depths to time: the first n layers of the same
	// engine (all layers when empty).
	Layers []int
	// Workers are the intra-op worker counts of single-request passes (8).
	Workers []int
	// Batch sequences of one length are stacked in the batched passes (8),
	// run with BatchWorkers intra-op workers (the first of Workers).
	Batch, BatchWorkers int
	// Runs timed passes per cell after Warmup untimed ones (100, 3). A cell
	// with fewer than MinGateRuns runs cannot carry the M0-LAYA verdict.
	Runs, Warmup int
	Kernel       tensor.Kernel
	// Progress, when set, receives one line per finished cell.
	Progress func(string)
}

func (s BenchSpec) withDefaults(layers int) BenchSpec {
	if len(s.Tokens) == 0 {
		s.Tokens = []int{32, 64, 128, 256}
	}
	if len(s.Layers) == 0 {
		s.Layers = []int{layers}
	}
	if len(s.Workers) == 0 {
		s.Workers = []int{8}
	}
	if s.Batch <= 0 {
		s.Batch = 8
	}
	if s.BatchWorkers <= 0 {
		s.BatchWorkers = s.Workers[0]
	}
	if s.Runs <= 0 {
		s.Runs = MinGateRuns
	}
	if s.Warmup < 0 {
		s.Warmup = 0
	} else if s.Warmup == 0 {
		s.Warmup = 3
	}
	return s
}

// LatencyStats summarises timed passes in milliseconds (nearest-rank
// percentiles).
type LatencyStats struct {
	Runs int     `json:"runs"`
	P50  float64 `json:"p50_ms"`
	P95  float64 `json:"p95_ms"`
	Mean float64 `json:"mean_ms"`
	Min  float64 `json:"min_ms"`
	Max  float64 `json:"max_ms"`
}

func summarise(ms []float64) LatencyStats {
	s := append([]float64(nil), ms...)
	sort.Float64s(s)
	var sum float64
	for _, v := range s {
		sum += v
	}
	r := func(v float64) float64 { return math.Round(v*1000) / 1000 }
	return LatencyStats{Runs: len(s), P50: r(s[percentileIndex(len(s), 0.5)]), P95: r(s[percentileIndex(len(s), 0.95)]),
		Mean: r(sum / float64(len(s))), Min: r(s[0]), Max: r(s[len(s)-1])}
}

// BenchRow is one timed cell.
type BenchRow struct {
	Tokens  int    `json:"tokens"`
	Layers  int    `json:"layers"`
	Path    string `json:"path"`
	Mode    string `json:"mode"`
	Workers int    `json:"intra_op_workers"`
	Batch   int    `json:"sequences_per_pass"`
	// Latency is per pass; PerSequence divides a batched pass by Batch.
	Latency          LatencyStats `json:"latency_per_pass"`
	PerSequenceP50Ms float64      `json:"per_sequence_p50_ms"`
	PerSequenceP95Ms float64      `json:"per_sequence_p95_ms"`
	CPUMsPerPass     float64      `json:"process_cpu_ms_per_pass"`
	GFLOPPerPass     float64      `json:"gflop_per_pass"`
	LoadAvgStart     string       `json:"load_average_at_start"`
}

// BenchHost describes where a bench ran.
type BenchHost struct {
	GOOS        string `json:"goos"`
	GOARCH      string `json:"goarch"`
	CPU         string `json:"cpu"`
	NumCPU      int    `json:"num_cpu"`
	GOMAXPROCS  int    `json:"gomaxprocs"`
	GoVersion   string `json:"go_version"`
	Kernel      string `json:"gemm_kernel"`
	Translated  bool   `json:"rosetta_translated"`
	LoadAvgPre  string `json:"load_average_at_start"`
	LoadAvgPost string `json:"load_average_at_end"`
}

// Bench times the trunk forward pass (the part of a Laya decision whose cost
// depends on the trunk) over tokens x layers x workers, single and batched,
// for each loaded numeric path. trunks maps "fp32"/"int8" to a loaded
// trunk of the same weights.
func Bench(ctx context.Context, trunks map[string]*Trunk, spec BenchSpec) ([]BenchRow, BenchHost, error) {
	paths := sortedKeys(trunks)
	if len(paths) == 0 {
		return nil, BenchHost{}, fmt.Errorf("train: no trunk to bench")
	}
	first := trunks[paths[0]]
	spec = spec.withDefaults(first.Encoder.Config.NumLayers)
	par0 := tensor.Parallel{Kernel: spec.Kernel}
	host := BenchHost{GOOS: runtime.GOOS, GOARCH: runtime.GOARCH, CPU: cpuModel(), NumCPU: runtime.NumCPU(),
		GOMAXPROCS: runtime.GOMAXPROCS(0), GoVersion: runtime.Version(), Kernel: par0.EffectiveKernel().String(),
		Translated: rosettaTranslated(), LoadAvgPre: LoadAverage()}
	ids := benchTokens(first, slices.Max(spec.Tokens))
	var rows []BenchRow
	for _, path := range sortedKeys(trunks) {
		t := trunks[path]
		t.Encoder.Weights.Prepack(par0.EffectiveKernel())
		for _, layers := range spec.Layers {
			enc, err := truncatedEncoder(t.Encoder, layers)
			if err != nil {
				return nil, host, err
			}
			for _, n := range spec.Tokens {
				seq := ids[:n]
				var cells []benchCell
				for _, w := range spec.Workers {
					cells = append(cells, benchCell{"single", w, 1})
				}
				cells = append(cells, benchCell{"batched", spec.BatchWorkers, spec.Batch})
				for _, c := range cells {
					seqs := make([][]int32, c.batch)
					for i := range seqs {
						seqs[i] = seq
					}
					par := tensor.Parallel{Workers: c.workers, Kernel: spec.Kernel}
					row := BenchRow{Tokens: n, Layers: layers, Path: path, Mode: c.mode, Workers: c.workers, Batch: c.batch,
						GFLOPPerPass: math.Round(enc.Config.FLOPs(n)*float64(c.batch)/1e6) / 1e3, LoadAvgStart: LoadAverage()}
					var ms []float64
					cpu0 := processCPU()
					for r := 0; r < spec.Warmup+spec.Runs; r++ {
						if r == spec.Warmup {
							cpu0 = processCPU()
						}
						start := time.Now()
						if _, err := enc.Forward(ctx, seqs, model.Options{Parallel: par}); err != nil {
							return nil, host, err
						}
						if r >= spec.Warmup {
							ms = append(ms, float64(time.Since(start).Microseconds())/1000)
						}
					}
					row.CPUMsPerPass = math.Round(float64((processCPU()-cpu0).Microseconds())/float64(spec.Runs)) / 1000
					row.Latency = summarise(ms)
					row.PerSequenceP50Ms = math.Round(row.Latency.P50/float64(c.batch)*1000) / 1000
					row.PerSequenceP95Ms = math.Round(row.Latency.P95/float64(c.batch)*1000) / 1000
					rows = append(rows, row)
					if spec.Progress != nil {
						spec.Progress(fmt.Sprintf("%s %2d layers %3d tokens %-7s w=%-2d b=%d: p50 %.1f ms p95 %.1f ms (cpu %.0f ms/pass, load %s)",
							path, layers, n, c.mode, c.workers, c.batch, row.Latency.P50, row.Latency.P95, row.CPUMsPerPass, row.LoadAvgStart))
					}
				}
			}
		}
	}
	host.LoadAvgPost = LoadAverage()
	return rows, host, nil
}

type benchCell struct {
	mode           string
	workers, batch int
}

// truncatedEncoder shares the weights of the first n layers.
func truncatedEncoder(e *model.Encoder, n int) (*model.Encoder, error) {
	if n == e.Config.NumLayers {
		return e, nil
	}
	if n <= 0 || n > e.Config.NumLayers {
		return nil, fmt.Errorf("train: cannot time %d of %d layers", n, e.Config.NumLayers)
	}
	cfg := e.Config
	cfg.NumLayers = n
	cfg.Layers = cfg.Layers[:n]
	w := *e.Weights
	w.Layers = w.Layers[:n]
	return model.NewEncoder(cfg, &w)
}

// benchTokens tokenises a mixed-language Laya-layout prompt and repeats it
// to n tokens behind a CLS token: latency depends on length, not content,
// but real ids keep the embedding reads realistic.
func benchTokens(t *Trunk, n int) []int32 {
	text := "choice question: Which operation does the question ask for? lookup compare explain forecast " +
		"Pourquoi le chiffre d'affaires hebdomadaire a-t-il baissé dans la région nord ? " +
		"Warum ist der Wochenumsatz gesunken? 北部地域の週次売上が下がったのはなぜですか？ لماذا انخفضت الإيرادات؟ "
	enc, err := t.Tokenizer.Encode(text, false)
	ids := []int32{t.Special.CLS}
	if err != nil || len(enc.IDs) == 0 {
		enc.IDs = []int32{t.Special.Mask}
	}
	for len(ids) < n {
		ids = append(ids, enc.IDs...)
	}
	return ids[:n]
}

// MinGateRuns is the fewest timed runs a cell needs to carry the M0-LAYA
// verdict: with fewer, the nearest-rank p95 is one of the last few runs (the
// maximum at 20 runs or fewer), so a single slow or fast pass decides it.
const MinGateRuns = 100

// BenchDecision applies the M0-LAYA rule to bench rows: ship all layers if
// the single-request fp32 p95 at the evaluated length is within the budget,
// otherwise recommend the deepest timed student that meets it. The verdict
// is the rule's mechanical output on this host; GateRun says whether the
// cells it read had enough runs to count as a gate reading at all.
type BenchDecision struct {
	Rule           string  `json:"rule"`
	BudgetMs       float64 `json:"budget_ms"`
	FastLaneTokens int     `json:"fast_lane_tokens"`
	// TypicalTokens is the dataset slice's median sequence length when
	// known; the verdict is read at min(TypicalTokens, FastLaneTokens),
	// because the fast lane caps its input.
	TypicalTokens  int     `json:"typical_tokens,omitempty"`
	EvaluatedAt    int     `json:"evaluated_tokens"`
	Workers        int     `json:"intra_op_workers"`
	FullLayers     int     `json:"full_layers"`
	FullP95Ms      float64 `json:"full_depth_p95_ms"`
	FullRuns       int     `json:"full_depth_runs"`
	Verdict        string  `json:"verdict"`
	Recommendation string  `json:"recommendation"`
	// GateRun is false when a cell the verdict read had fewer than
	// MinGateRuns timed runs; GateNote says why.
	GateRun  bool   `json:"gate_run"`
	GateNote string `json:"gate_note,omitempty"`
	Scope    string `json:"scope"`
}

// DecideM0 reads the rule off the rows.
func DecideM0(rows []BenchRow, fullLayers, typical, workers int) BenchDecision {
	d := BenchDecision{
		Rule:     "ship the full-depth trunk if the fast-lane single-request fp32 p95 is at most 120 ms, else a fewer-layer student of the same engine",
		BudgetMs: 120, FastLaneTokens: 64, TypicalTokens: typical, Workers: workers, FullLayers: fullLayers, GateRun: true,
		Scope: "this-machine evidence only; the pre-registered M0-LAYA gate is 8 vCPU AVX2 on Cloud Run under open-loop Poisson load (README)",
	}
	d.EvaluatedAt = 64
	if typical > 0 && typical < 64 {
		d.EvaluatedAt = typical
	}
	// cell reads the fp32 single-request row at the shortest timed length
	// that is at least the evaluated one.
	cell := func(layers int) *BenchRow {
		var best *BenchRow
		for i := range rows {
			r := &rows[i]
			if r.Mode == "single" && r.Path == "fp32" && r.Layers == layers && r.Workers == workers &&
				r.Tokens >= d.EvaluatedAt && (best == nil || r.Tokens < best.Tokens) {
				best = r
			}
		}
		return best
	}
	read := func(r *BenchRow) {
		if r.Latency.Runs < MinGateRuns && d.GateRun {
			d.GateRun = false
			d.GateNote = fmt.Sprintf("%d layers at %d tokens has %d timed runs, fewer than %d: the p95 is not a gate reading",
				r.Layers, r.Tokens, r.Latency.Runs, MinGateRuns)
		}
	}
	full := cell(fullLayers)
	if full == nil {
		d.Verdict, d.GateRun = "not_measured", false
		return d
	}
	read(full)
	d.FullP95Ms, d.FullRuns = full.Latency.P95, full.Latency.Runs
	if full.Latency.P95 <= d.BudgetMs {
		d.Verdict = "pass"
		d.Recommendation = fmt.Sprintf("ship %d layers on this machine's evidence", fullLayers)
		return d
	}
	d.Verdict = "fail"
	var depths []int
	for _, r := range rows {
		if r.Layers < fullLayers && !slices.Contains(depths, r.Layers) {
			depths = append(depths, r.Layers)
		}
	}
	sort.Sort(sort.Reverse(sort.IntSlice(depths)))
	for _, l := range depths {
		if r := cell(l); r != nil && r.Latency.P95 <= d.BudgetMs {
			read(r)
			d.Recommendation = fmt.Sprintf("a %d-layer student of the same engine (p95 %.1f ms here)", l, r.Latency.P95)
			return d
		}
	}
	d.Recommendation = "no timed depth meets the budget on this machine"
	return d
}

// command runs a fixed system probe (sysctl) with a short deadline.
func command(name string, args ...string) string {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, name, args...).Output() //nolint:gosec // G204: fixed system probes (sysctl)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// LoadAverage returns the 1/5/15-minute load averages as text.
func LoadAverage() string {
	if b, err := os.ReadFile("/proc/loadavg"); err == nil {
		f := strings.Fields(string(b))
		if len(f) >= 3 {
			return strings.Join(f[:3], " ")
		}
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

// rosettaTranslated reports an amd64 process running under Rosetta 2.
func rosettaTranslated() bool {
	return runtime.GOOS == "darwin" && command("sysctl", "-n", "sysctl.proc_translated") == "1"
}
