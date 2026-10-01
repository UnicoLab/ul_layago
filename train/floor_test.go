package train

import (
	"context"
	"slices"
	"testing"
)

// TestHashedFloorTrainsDecodesAndIsDeterministic: the floor fits every
// train item, decodes every item (scorable, never acting), fits its
// temperatures on the fit languages' dev_cal items only, learns the
// fixture's train split and repeats bit for bit.
func TestHashedFloorTrainsDecodesAndIsDeterministic(t *testing.T) {
	ctx := context.Background()
	trunk := tinyTrunk(t, tinySpec)
	p, err := Prepare(fixtureDataset(t, InstanceQ), trunk, PrepareOptions{})
	if err != nil {
		t.Fatal(err)
	}
	cfg := DefaultFloorConfig()
	cfg.Bits, cfg.MinCell, cfg.FitLangs = 16, 1, []string{"fr"}
	a, err := FitFloor(ctx, p, cfg)
	if err != nil {
		t.Fatal(err)
	}
	b, err := FitFloor(ctx, p, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(a.Rows) != len(p.Items) || !slices.Equal(a.Weights, b.Weights) {
		t.Fatalf("%d rows for %d items, deterministic %v", len(a.Rows), len(p.Items), slices.Equal(a.Weights, b.Weights))
	}
	cal, trainN, right := 0, 0, 0
	for i, it := range p.Items {
		r := a.Rows[i]
		if !r.Scorable || r.Acted || len(r.Probs) != it.Ex.NumOptions() {
			t.Fatalf("row %+v", r)
		}
		if it.Role == RoleDevCal && primaryLanguage(it.Ex.Lang) == "fr" {
			cal++
		}
		if it.Role == SplitTrain {
			trainN++
			if r.Correct {
				right++
			}
		}
	}
	if a.Temperatures["*"].N != cal || a.TrainItems != trainN {
		t.Fatalf("temperature on %d items (dev_cal fr %d), trained on %d of %d", a.Temperatures["*"].N, cal, a.TrainItems, trainN)
	}
	if float64(right) < 0.6*float64(trainN) {
		t.Fatalf("the floor fits %d of %d train items", right, trainN)
	}
}
