package model

import (
	"context"
	"os"
	"slices"
	"testing"

	"github.com/UnicoLab/ul_layago/safetensors"
	"github.com/UnicoLab/ul_layago/tensor"
)

// A truncated trunk is the full checkpoint's first n layers followed by its
// final norm: exactly the full pass's hidden state n, normalised.
func TestTruncatedEncoderIsTheFirstLayersOfTheFullOne(t *testing.T) {
	raw, err := os.ReadFile(parityDir + "checkpoint/encoder/config.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := ParseHFConfig(raw)
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range []int{0, cfg.NumLayers + 1} {
		if _, err := cfg.Truncate(n); err == nil {
			t.Fatalf("Truncate(%d) of %d layers accepted", n, cfg.NumLayers)
		}
	}
	short, err := cfg.Truncate(3)
	if err != nil {
		t.Fatal(err)
	}
	if short.NumLayers != 3 || !slices.Equal(short.Layers, cfg.Layers[:3]) || len(cfg.Layers) != 5 {
		t.Fatalf("Truncate(3) = %+v (full config changed: %d layers)", short, len(cfg.Layers))
	}
	f, err := safetensors.Open(parityDir + "checkpoint/model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	if _, err := LoadWeights(f, short, LoadOptions{Prefix: "encoder."}); err == nil {
		t.Fatal("deeper layers must still be rejected unless SkipDeeperLayers is set")
	}
	w, err := LoadWeights(f, short, LoadOptions{Prefix: "encoder.", SkipDeeperLayers: true})
	if err != nil {
		t.Fatal(err)
	}
	truncated, err := NewEncoder(short, w)
	if err != nil {
		t.Fatal(err)
	}
	full := loadParityEncoder(t, false)
	seqs := [][]int32{{2, 40, 41, 42, 4, 99, 1}, {2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1}}
	par := tensor.Parallel{Workers: 1}
	got, err := truncated.Forward(context.Background(), seqs, Options{Parallel: par})
	if err != nil {
		t.Fatal(err)
	}
	ref, err := full.Forward(context.Background(), seqs, Options{Parallel: par, CaptureLayers: true})
	if err != nil {
		t.Fatal(err)
	}
	d := cfg.HiddenSize
	for s, seq := range seqs {
		want := make([]float32, len(seq)*d)
		fn := full.Weights.FinalNorm
		tensor.LayerNorm(want, ref.Layers[3][s], len(seq), d, fn.Weight, fn.Bias, cfg.NormEps, par)
		if !slices.Equal(got.Hidden[s], want) {
			t.Fatalf("sequence %d: truncated output differs from normalised hidden state 3", s)
		}
	}
}

func TestDeeperLayerNames(t *testing.T) {
	for _, tc := range []struct {
		name string
		n    int
		want bool
	}{
		{"layers.3.attn.Wqkv.weight", 3, true},
		{"layers.2.attn.Wqkv.weight", 3, false},
		{"layers.10.mlp.Wi.weight", 3, true},
		{"layers.03.mlp.Wi.weight", 3, false},
		{"layers.x.mlp.Wi.weight", 3, false},
		{"layers.4", 3, false},
		{"final_norm.weight", 0, false},
		{"embeddings.norm.weight", 0, false},
	} {
		if got := deeperLayer(tc.name, tc.n); got != tc.want {
			t.Errorf("deeperLayer(%q, %d) = %v, want %v", tc.name, tc.n, got, tc.want)
		}
	}
}
