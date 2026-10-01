package train

import (
	"context"
	"fmt"
	"runtime"
	"slices"
	"time"

	"github.com/UnicoLab/ul_layago/tensor"
)

// LatencyCase is one served-decision setup to time: an artifact over a
// loaded trunk (float32, or the per-row int8 trunk path).
type LatencyCase struct {
	Name     string
	Artifact *Artifact
	Trunk    *Trunk
}

// LatencySpec times single served decisions, one request at a time.
type LatencySpec struct {
	// Tokens are the sequence lengths timed (32, 64, 128). A 0 in the list
	// times the items' own sequences end to end, state rendering and
	// tokenization included.
	Tokens []int
	// Runs timed decisions per cell after Warmup untimed ones (100, 5).
	Runs, Warmup int
	// Workers are the intra-op workers of each trunk pass (8).
	Workers int
	Kernel  tensor.Kernel
	// Progress, when set, receives one line per finished cell.
	Progress func(string)
}

// LatencyRow is one timed cell. A decision is the trunk pass, the head
// over every option marker, the temperature and the act decision.
type LatencyRow struct {
	Case           string `json:"case"`
	ArtifactFormat string `json:"artifact_format"`
	TrunkPath      string `json:"trunk_numeric_path"`
	// Tokens is the sequence length, or 0 for the items' own sequences.
	Tokens      int     `json:"tokens"`
	MeanTokens  float64 `json:"mean_tokens"`
	Items       int     `json:"distinct_items"`
	MeanOptions float64 `json:"mean_options"`
	Workers     int     `json:"intra_op_workers"`
	// Latency is wall time per decision; HeadP50Ms and LayoutP50Ms (state
	// rendering and tokenization, Tokens 0 only) are its parts.
	Latency     LatencyStats `json:"latency_per_decision"`
	HeadP50Ms   float64      `json:"head_p50_ms"`
	LayoutP50Ms float64      `json:"layout_p50_ms,omitempty"`
	// ActedShare is the share of timed decisions that acted (Tokens 0
	// only: a cut sequence's decision means nothing).
	ActedShare       *float64 `json:"acted_share,omitempty"`
	CPUMsPerDecision float64  `json:"process_cpu_ms_per_decision"`
	LoadAvgStart     string   `json:"load_average_at_start"`
	LoadAvgEnd       string   `json:"load_average_at_end"`
}

func (s LatencySpec) withDefaults() LatencySpec {
	if len(s.Tokens) == 0 {
		s.Tokens = []int{32, 64, 128}
	}
	if s.Runs <= 0 {
		s.Runs = MinGateRuns
	}
	if s.Warmup < 0 {
		s.Warmup = 0
	} else if s.Warmup == 0 {
		s.Warmup = 5
	}
	if s.Workers <= 0 {
		s.Workers = 8
	}
	return s
}

// ArtifactLatency times served decisions of each case over items, cycled in
// order. At a fixed length L every item's sequence is cut to L-1 tokens plus
// [SEP] and markers past the cut are moved to the last state position: the
// work (a trunk pass over L tokens and the head over the item's K options)
// is a real decision's, the outputs of a cut sequence are not decisions.
// Items shorter than L are skipped at that length.
func ArtifactLatency(ctx context.Context, cases []LatencyCase, items []*Item, spec LatencySpec) ([]LatencyRow, BenchHost, error) {
	spec = spec.withDefaults()
	par := tensor.Parallel{Workers: spec.Workers, Kernel: spec.Kernel}
	host := BenchHost{GOOS: runtime.GOOS, GOARCH: runtime.GOARCH, CPU: cpuModel(), NumCPU: runtime.NumCPU(),
		GOMAXPROCS: runtime.GOMAXPROCS(0), GoVersion: runtime.Version(), Kernel: par.EffectiveKernel().String(),
		Translated: rosettaTranslated(), LoadAvgPre: LoadAverage()}
	var scorable []*Item
	for _, it := range items {
		if it.Scorable {
			scorable = append(scorable, it)
		}
	}
	if len(scorable) == 0 {
		return nil, host, fmt.Errorf("train: no scorable item to time")
	}
	var rows []LatencyRow
	for _, c := range cases {
		if c.Artifact == nil || c.Trunk == nil {
			return nil, host, fmt.Errorf("train: latency case %q needs an artifact and a trunk", c.Name)
		}
		c.Trunk.Encoder.Weights.Prepack(par.EffectiveKernel())
		for _, n := range spec.Tokens {
			row, err := timeCell(ctx, c, scorable, n, spec, par)
			if err != nil {
				return nil, host, err
			}
			rows = append(rows, row)
			if spec.Progress != nil {
				spec.Progress(fmt.Sprintf("%s, %d tokens: p50 %.1f ms, p95 %.1f ms (head %.2f ms), load %s", c.Name, n,
					row.Latency.P50, row.Latency.P95, row.HeadP50Ms, row.LoadAvgEnd))
			}
		}
	}
	host.LoadAvgPost = LoadAverage()
	return rows, host, nil
}

// latencyInput is one decision's input.
type latencyInput struct {
	it      *Item
	ids     []int32
	markers []int
}

func cutSequence(it *Item, n int, sep int32) (latencyInput, bool) {
	if len(it.IDs) < n {
		return latencyInput{}, false
	}
	ids := append(slices.Clone(it.IDs[:n-1]), sep)
	mk := make([]int, len(it.Markers))
	for i, m := range it.Markers {
		mk[i] = min(m, n-2)
	}
	return latencyInput{it: it, ids: ids, markers: mk}, true
}

func timeCell(ctx context.Context, c LatencyCase, items []*Item, n int, spec LatencySpec, par tensor.Parallel) (LatencyRow, error) {
	m := &c.Artifact.Manifest
	row := LatencyRow{Case: c.Name, ArtifactFormat: m.Weights.Format, TrunkPath: c.Trunk.Identity.NumericPath, Tokens: n,
		Workers: spec.Workers, LoadAvgStart: LoadAverage()}
	var inputs []latencyInput
	for _, it := range items {
		if n == 0 {
			inputs = append(inputs, latencyInput{it: it})
			continue
		}
		if in, ok := cutSequence(it, n, c.Trunk.Special.SEP); ok {
			inputs = append(inputs, in)
		}
	}
	if len(inputs) == 0 {
		return row, fmt.Errorf("train: no item has %d tokens to time", n)
	}
	row.Items = min(len(inputs), spec.Runs)
	total := spec.Warmup + spec.Runs
	wall, head, layout := make([]float64, 0, spec.Runs), make([]float64, 0, spec.Runs), make([]float64, 0, spec.Runs)
	var tokens, options, acted int
	var cpu time.Duration
	for r := 0; r < total; r++ {
		if err := ctx.Err(); err != nil {
			return row, err
		}
		in := inputs[r%len(inputs)]
		ex := in.it.Ex
		ts := m.Tasks[ex.Task]
		if ts == nil {
			return row, fmt.Errorf("train: the artifact has no task %q", ex.Task)
		}
		c0, t0 := processCPU(), time.Now()
		ids, markers := in.ids, in.markers
		if n == 0 {
			it, err := layOut(in.it.source(), ts, c.Trunk, m.Sequence.Shortlist)
			if err != nil {
				return row, err
			}
			if !it.Scorable {
				return row, fmt.Errorf("train: %s is unscorable on this trunk", ex.ID)
			}
			ids, markers = it.IDs, it.Markers
		}
		t1 := time.Now()
		feats, err := c.Trunk.Features(ctx, [][]int32{ids}, [][]int{markers}, par)
		if err != nil {
			return row, err
		}
		t2 := time.Now()
		qt := ex.Type.QType()
		logits := c.Artifact.Head.Logits(feats[0].Markers, feats[0].K, qt)
		act := c.Artifact.Head.ActLogits(feats[0].Pooled, qt, logits)
		temp, _ := m.Calibration.Temperatures.Lookup(ex.Task, feats[0].K)
		probs := ServedProbabilities(logits, temp)
		score := slices.Max(probs)
		th, ok := m.Act.Thresholds[ex.Task]
		if ok && th.Score == ScoreActProbability {
			score = ActProbability(act)
		}
		decidedAct := ok && th.Acts(score)
		t3 := time.Now()
		dc := processCPU() - c0
		if r < spec.Warmup {
			continue
		}
		wall = append(wall, float64(t3.Sub(t0).Microseconds())/1000)
		head = append(head, float64(t3.Sub(t2).Microseconds())/1000)
		layout = append(layout, float64(t1.Sub(t0).Microseconds())/1000)
		cpu += dc
		tokens += len(ids)
		options += feats[0].K
		if decidedAct {
			acted++
		}
	}
	row.Latency = summarise(wall)
	row.HeadP50Ms = summarise(head).P50
	if n == 0 {
		row.LayoutP50Ms = summarise(layout).P50
		share := float64(acted) / float64(spec.Runs)
		row.ActedShare = &share
	}
	row.MeanTokens = float64(tokens) / float64(spec.Runs)
	row.MeanOptions = float64(options) / float64(spec.Runs)
	row.CPUMsPerDecision = float64(cpu.Microseconds()) / 1000 / float64(spec.Runs)
	row.LoadAvgEnd = LoadAverage()
	return row, nil
}
