package train

import (
	"context"
	"reflect"
	"testing"
)

func TestWarmStartClonesHeadAndKeepsPinnedProvenance(t *testing.T) {
	l := qLoop(t)
	initial := l.res.Float32.Head
	before := initial.Clone()
	cfg := loopConfig(1)
	cfg.Epochs = 1
	cfg.ActEpochs = 1
	cfg.Int8 = false
	cfg.InitHead = initial
	cfg.InitWeightsSHA256 = "test-pinned-weights"
	result, err := Train(context.Background(), l.p, l.recs, l.trunk.Identity.HiddenSize, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(initial, before) {
		t.Fatal("training changed caller's initial head")
	}
	if reflect.DeepEqual(result.Float32.Head, initial) {
		t.Fatal("warm start did not train")
	}
	if result.Config.InitWeightsSHA256 != "test-pinned-weights" {
		t.Fatal("initialization provenance lost")
	}
	cfg.InitHead, _ = NewHead(HeadConfig{D: 32, Hidden: 32, ActHidden: 256, NAct: 2}, 1)
	if _, err := Train(context.Background(), l.p, l.recs, l.trunk.Identity.HiddenSize, cfg); err == nil {
		t.Fatal("accepted incompatible initial head")
	}
}
