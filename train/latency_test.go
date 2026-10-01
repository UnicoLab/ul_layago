package train

import (
	"context"
	"testing"
)

// TestArtifactLatencyTimesEveryCell: each case is timed at each length with
// the requested runs; a fixed length cuts sequences to that many tokens
// ending in [SEP], and 0 times the items' own sequences end to end.
func TestArtifactLatencyTimesEveryCell(t *testing.T) {
	l := qLoop(t)
	var cases []LatencyCase
	for i, name := range []string{"float32", "int8"} {
		art, err := LoadArtifact(l.dirs[i])
		if err != nil {
			t.Fatal(err)
		}
		cases = append(cases, LatencyCase{Name: name, Artifact: art, Trunk: l.trunk})
	}
	var items []*Item
	for _, it := range l.p.Items {
		if it.Role == RoleDevConf {
			items = append(items, it)
		}
	}
	var lines int
	rows, host, err := ArtifactLatency(context.Background(), cases, items, LatencySpec{Tokens: []int{24, 48, 0}, Runs: 4, Warmup: 1,
		Workers: 2, Progress: func(string) { lines++ }})
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 6 || lines != 6 || host.LoadAvgPre == "" || host.GOMAXPROCS < 1 {
		t.Fatalf("%d rows, %d progress lines, host %+v", len(rows), lines, host)
	}
	for _, r := range rows {
		if r.Latency.Runs != 4 || r.Latency.P50 <= 0 || r.HeadP50Ms <= 0 || r.MeanOptions < 2 || r.Workers != 2 || r.TrunkPath != "fp32" {
			t.Fatalf("row %+v", r)
		}
		switch r.Tokens {
		case 0:
			if r.LayoutP50Ms <= 0 || r.ActedShare == nil || r.MeanTokens < 24 {
				t.Fatalf("end-to-end row %+v", r)
			}
		default:
			if r.MeanTokens != float64(r.Tokens) || r.ActedShare != nil {
				t.Fatalf("fixed-length row %+v", r)
			}
		}
	}
	if rows[0].ArtifactFormat != "float32" || rows[3].ArtifactFormat != "int8" {
		t.Fatalf("formats %s %s", rows[0].ArtifactFormat, rows[3].ArtifactFormat)
	}
	if _, _, err := ArtifactLatency(context.Background(), cases, items, LatencySpec{Tokens: []int{100000}, Runs: 1}); err == nil {
		t.Fatal("a length no item reaches was timed")
	}
	in, ok := cutSequence(items[0], 10, l.trunk.Special.SEP)
	if !ok || len(in.ids) != 10 || in.ids[9] != l.trunk.Special.SEP {
		t.Fatalf("cut %v", in.ids)
	}
	for _, m := range in.markers {
		if m > 8 {
			t.Fatalf("marker %d past the cut", m)
		}
	}
}
