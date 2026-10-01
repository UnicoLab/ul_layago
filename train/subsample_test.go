package train

import (
	"math"
	"reflect"
	"slices"
	"strings"
	"testing"
)

// TestPrepareTrainSubsample: a train fraction keeps ceil(fraction·n) train
// items per (task, language, domain) stratum, chosen by a seeded hash of the
// id, leaves every evaluated split and the task definitions whole, changes
// the sequence digest (so the feature cache key) and is recorded.
func TestPrepareTrainSubsample(t *testing.T) {
	trunk := tinyTrunk(t, tinySpec)
	ds := fixtureDataset(t, InstanceQ)
	full, err := Prepare(ds, trunk, PrepareOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if full.TrainSubsample != nil {
		t.Fatalf("no fraction, record %+v", full.TrainSubsample)
	}
	if one, err := Prepare(ds, trunk, PrepareOptions{TrainFraction: 1, SubsampleSeed: 9}); err != nil || one.TrainSubsample != nil ||
		one.SequenceDigest != full.SequenceDigest {
		t.Fatalf("fraction 1 must keep everything: %v %+v", err, one)
	}
	half, err := Prepare(ds, trunk, PrepareOptions{TrainFraction: 0.5, SubsampleSeed: 1})
	if err != nil {
		t.Fatal(err)
	}
	stratum := func(it *Item) string { return it.Ex.Task + "/" + it.Ex.Lang + "/" + it.Ex.DomainID }
	want, got := map[string]int{}, map[string]int{}
	var fullEval, halfEval, kept []string
	for _, it := range full.Items {
		if it.Role == SplitTrain {
			want[stratum(it)]++
		} else {
			fullEval = append(fullEval, it.Ex.ID)
		}
	}
	for _, it := range half.Items {
		if it.Role == SplitTrain {
			got[stratum(it)]++
			kept = append(kept, it.Ex.ID)
		} else {
			halfEval = append(halfEval, it.Ex.ID)
		}
	}
	if !slices.Equal(fullEval, halfEval) {
		t.Fatalf("evaluated items changed: %d vs %d", len(fullEval), len(halfEval))
	}
	source := 0
	for k, n := range want {
		source += n
		if got[k] != int(math.Ceil(0.5*float64(n))) {
			t.Errorf("stratum %s: kept %d of %d", k, got[k], n)
		}
	}
	s := half.TrainSubsample
	if s == nil || s.SourceExamples != source || s.KeptExamples != len(kept) || s.Strata != len(want) || s.Fraction != 0.5 ||
		s.Seed != 1 || s.Rule != SubsampleRule || s.KeptIDsSHA256 != idDigest(kept) || len(kept) >= source {
		t.Fatalf("record %+v, kept %d of %d", s, len(kept), source)
	}
	var byTask int
	for _, n := range s.KeptByTask {
		byTask += n
	}
	if byTask != len(kept) || len(s.KeptByLang) == 0 || len(s.KeptByDomain) == 0 {
		t.Fatalf("kept counts %+v", s)
	}
	if half.SequenceDigest == full.SequenceDigest {
		t.Fatal("the sequence digest must cover the subsample")
	}
	if !reflect.DeepEqual(half.Tasks, full.Tasks) {
		t.Fatal("task definitions must come from every visible example")
	}
	if lim := strings.Join(limitations(half, DevHalfReport{}), "\n"); !strings.Contains(lim, "stratified subsample of train") {
		t.Fatalf("limitations %s", lim)
	}

	again, err := Prepare(ds, trunk, PrepareOptions{TrainFraction: 0.5, SubsampleSeed: 1})
	if err != nil || again.TrainSubsample.KeptIDsSHA256 != s.KeptIDsSHA256 || again.SequenceDigest != half.SequenceDigest {
		t.Fatalf("not deterministic: %v", err)
	}
	other, err := Prepare(ds, trunk, PrepareOptions{TrainFraction: 0.5, SubsampleSeed: 2})
	if err != nil || other.TrainSubsample.KeptIDsSHA256 == s.KeptIDsSHA256 {
		t.Fatalf("the seed must change the subsample: %v", err)
	}
	for _, bad := range []float64{-0.1, 1.5, math.NaN()} {
		if _, err := Prepare(ds, trunk, PrepareOptions{TrainFraction: bad}); err == nil {
			t.Errorf("fraction %v accepted", bad)
		}
	}
}
