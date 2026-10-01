package train

import (
	"context"
	"math/rand/v2"
	"strings"
	"testing"
)

func TestBenchRunsEveryCell(t *testing.T) {
	trunk := tinyTrunk(t, TrunkSpec{})
	rows, host, err := Bench(context.Background(), map[string]*Trunk{"fp32": trunk},
		BenchSpec{Tokens: []int{16, 40}, Layers: []int{5, 3}, Workers: []int{1, 2}, Batch: 3, Runs: 2, Warmup: 1})
	if err != nil {
		t.Fatal(err)
	}
	// 2 lengths x 2 depths x (2 single + 1 batched).
	if len(rows) != 12 || host.GOARCH == "" || host.Kernel == "" {
		t.Fatalf("%d rows, host %+v", len(rows), host)
	}
	for _, r := range rows {
		if r.Latency.Runs != 2 || r.Latency.P95 < r.Latency.P50 || r.GFLOPPerPass <= 0 {
			t.Fatalf("row %+v", r)
		}
		if r.Mode == "batched" && (r.Batch != 3 || r.PerSequenceP50Ms > r.Latency.P50) {
			t.Fatalf("batched row %+v", r)
		}
	}
}

// BenchmarkHeadMMBERTBase times the served head (four options, act head)
// at the mmBERT-base width, the cost a decision adds to the trunk pass.
func BenchmarkHeadMMBERTBase(b *testing.B) {
	h, err := NewHead(HeadConfig{D: 768, Hidden: 768, ActHidden: 256, NAct: 2}, 1)
	if err != nil {
		b.Fatal(err)
	}
	r := rand.New(rand.NewPCG(1, 1)) //nolint:gosec // G404: benchmark data
	markers, pooled := randFloats(r, 4*768, 1), randFloats(r, 768, 1)
	for b.Loop() {
		l := h.Logits(markers, 4, TaskChoice.QType())
		_ = h.ActLogits(pooled, TaskChoice.QType(), l)
	}
}

func TestDecideM0AppliesTheRule(t *testing.T) {
	row := func(layers, tokens int, p95 float64) BenchRow {
		return BenchRow{Tokens: tokens, Layers: layers, Path: "fp32", Mode: "single", Workers: 8, Latency: LatencyStats{P95: p95, Runs: 100}}
	}
	pass := DecideM0([]BenchRow{row(22, 32, 60), row(22, 64, 110), row(22, 128, 300)}, 22, 0, 8)
	if pass.Verdict != "pass" || pass.EvaluatedAt != 64 || pass.FullP95Ms != 110 || !pass.GateRun || pass.FullRuns != 100 {
		t.Fatalf("%+v", pass)
	}
	// Fewer than MinGateRuns runs: the rule still reads the cell, but the
	// reading is not a gate run.
	thin := row(22, 64, 96)
	thin.Latency.Runs = 40
	if d := DecideM0([]BenchRow{thin}, 22, 0, 8); d.Verdict != "pass" || d.GateRun || !strings.Contains(d.GateNote, "40 timed runs") {
		t.Fatalf("%+v", d)
	}
	fail := DecideM0([]BenchRow{row(22, 64, 180), row(6, 64, 55), row(4, 64, 40)}, 22, 0, 8)
	if fail.Verdict != "fail" || !strings.Contains(fail.Recommendation, "6-layer") {
		t.Fatalf("%+v", fail)
	}
	// A slice whose typical sequence is 48 tokens is read at the next timed
	// length, 64.
	typical := DecideM0([]BenchRow{row(22, 32, 50), row(22, 64, 130), row(6, 64, 100)}, 22, 48, 8)
	if typical.EvaluatedAt != 48 || typical.Verdict != "fail" || !strings.Contains(typical.Recommendation, "6-layer") {
		t.Fatalf("%+v", typical)
	}
	if none := DecideM0([]BenchRow{row(22, 64, 500), row(6, 64, 200)}, 22, 0, 8); !strings.Contains(none.Recommendation, "no timed depth") {
		t.Fatalf("%+v", none)
	}
}
