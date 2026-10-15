package model

import (
	"context"
	"math"
	"testing"

	"github.com/UnicoLab/ul_layago/tensor"
)

func TestParseHFConfigUltraBERTAndSwiGLU(t *testing.T) {
	cases := []struct {
		name    string
		json    string
		wantAct string
		wantErr bool
	}{
		{
			name: "modernbert_gelu_default",
			json: `{
				"model_type": "modernbert",
				"vocab_size": 100,
				"hidden_size": 16,
				"num_hidden_layers": 4,
				"num_attention_heads": 2,
				"intermediate_size": 32,
				"norm_eps": 1e-5,
				"local_attention": 128,
				"global_attn_every_n_layers": 3,
				"global_rope_theta": 160000.0,
				"local_rope_theta": 10000.0
			}`,
			wantAct: "gelu",
		},
		{
			name: "ultrabert_silu",
			json: `{
				"model_type": "ultrabert",
				"vocab_size": 100,
				"hidden_size": 16,
				"num_hidden_layers": 4,
				"num_attention_heads": 2,
				"intermediate_size": 32,
				"norm_eps": 1e-5,
				"local_attention": 128,
				"global_attn_every_n_layers": 3,
				"global_rope_theta": 160000.0,
				"local_rope_theta": 10000.0,
				"hidden_activation": "silu"
			}`,
			wantAct: "silu",
		},
		{
			name: "ultrabert_swiglu",
			json: `{
				"model_type": "ultrabert",
				"vocab_size": 100,
				"hidden_size": 16,
				"num_hidden_layers": 4,
				"num_attention_heads": 2,
				"intermediate_size": 32,
				"norm_eps": 1e-5,
				"local_attention": 128,
				"global_attn_every_n_layers": 3,
				"global_rope_theta": 160000.0,
				"local_rope_theta": 10000.0,
				"hidden_activation": "swiglu"
			}`,
			wantAct: "swiglu",
		},
		{
			name: "invalid_activation",
			json: `{
				"model_type": "ultrabert",
				"vocab_size": 100,
				"hidden_size": 16,
				"num_hidden_layers": 4,
				"num_attention_heads": 2,
				"intermediate_size": 32,
				"hidden_activation": "relu"
			}`,
			wantErr: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg, err := ParseHFConfig([]byte(tc.json))
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if cfg.HiddenActivation != tc.wantAct {
				t.Fatalf("HiddenActivation = %q, want %q", cfg.HiddenActivation, tc.wantAct)
			}
		})
	}
}

func TestTruncateToGlobalBoundary(t *testing.T) {
	base := MMBERTBase()
	for l := 0; l < base.NumLayers; l++ {
		wantGlobal := l%3 == 0
		if got := base.IsGlobalLayer(l); got != wantGlobal {
			t.Fatalf("IsGlobalLayer(%d) = %v, want %v", l, got, wantGlobal)
		}
	}
	if base.IsGlobalLayer(-1) || base.IsGlobalLayer(base.NumLayers) {
		t.Fatalf("out-of-range layer reported as global")
	}

	cases := []struct {
		target     int
		wantLayers int
		wantLast   int
	}{
		{target: 13, wantLayers: 13, wantLast: 12},
		{target: 12, wantLayers: 10, wantLast: 9},
		{target: 11, wantLayers: 10, wantLast: 9},
		{target: 10, wantLayers: 10, wantLast: 9},
		{target: 6, wantLayers: 4, wantLast: 3},
		{target: 4, wantLayers: 4, wantLast: 3},
		{target: 2, wantLayers: 1, wantLast: 0},
		{target: 0, wantLayers: 22, wantLast: 21},
		{target: 25, wantLayers: 22, wantLast: 21},
	}

	for _, tc := range cases {
		got := base.TruncateToGlobalBoundary(tc.target)
		if got.NumLayers != tc.wantLayers || len(got.Layers) != tc.wantLayers {
			t.Fatalf("TruncateToGlobalBoundary(%d): got %d layers, want %d", tc.target, got.NumLayers, tc.wantLayers)
		}
		if tc.target > 0 && tc.target < base.NumLayers {
			if !got.IsGlobalLayer(got.NumLayers - 1) {
				t.Fatalf("TruncateToGlobalBoundary(%d): final layer %d is not a global attention layer", tc.target, got.NumLayers-1)
			}
			if got.NumLayers-1 != tc.wantLast {
				t.Fatalf("TruncateToGlobalBoundary(%d): last layer index = %d, want %d", tc.target, got.NumLayers-1, tc.wantLast)
			}
		}
	}
}

func makeSyntheticWeights(t *testing.T, cfg Config) *Weights {
	t.Helper()
	d, inter := cfg.HiddenSize, cfg.IntermediateSize
	ones := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = 1
		}
		return out
	}
	seqFill := func(n int, scale float32) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = float32((i%11)-5) * scale
		}
		return out
	}
	lin := func(outDim, inDim int, scale float32) *tensor.Linear {
		l, err := tensor.NewLinear(outDim, inDim, seqFill(outDim*inDim, scale), nil)
		if err != nil {
			t.Fatalf("NewLinear(%d, %d): %v", outDim, inDim, err)
		}
		return l
	}
	w := &Weights{
		Embed:     Embedding{Rows: cfg.VocabSize, Dim: d, W: seqFill(cfg.VocabSize*d, 0.1)},
		EmbedNorm: Norm{Weight: ones(d)},
		FinalNorm: Norm{Weight: ones(d)},
	}
	for i := 0; i < cfg.NumLayers; i++ {
		var attnNorm *Norm
		if i > 0 {
			n := Norm{Weight: ones(d)}
			attnNorm = &n
		}
		w.Layers = append(w.Layers, Layer{
			AttnNorm: attnNorm,
			Wqkv:     lin(3*d, d, 0.05),
			Wo:       lin(d, d, 0.05),
			MLPNorm:  Norm{Weight: ones(d)},
			Wi:       lin(2*inter, d, 0.07),
			WoM:      lin(d, inter, 0.07),
		})
	}
	return w
}

func TestEncoderForwardSwiGLUVsGeGLU(t *testing.T) {
	baseCfg := Config{
		VocabSize:        16,
		HiddenSize:       8,
		NumLayers:        2,
		NumHeads:         2,
		IntermediateSize: 12,
		NormEps:          1e-5,
		Layers: []LayerSpec{
			{Window: -1, RopeTheta: 10000},
			{Window: 4, RopeTheta: 10000},
		},
	}
	w := makeSyntheticWeights(t, baseCfg)

	cfgGeGLU := baseCfg
	cfgGeGLU.HiddenActivation = "gelu"
	encGeGLU, err := NewEncoder(cfgGeGLU, w)
	if err != nil {
		t.Fatalf("NewEncoder GeGLU: %v", err)
	}

	cfgSiLU := baseCfg
	cfgSiLU.HiddenActivation = "silu"
	encSiLU, err := NewEncoder(cfgSiLU, w)
	if err != nil {
		t.Fatalf("NewEncoder SiLU: %v", err)
	}

	cfgSwiGLU := baseCfg
	cfgSwiGLU.HiddenActivation = "swiglu"
	encSwiGLU, err := NewEncoder(cfgSwiGLU, w)
	if err != nil {
		t.Fatalf("NewEncoder SwiGLU: %v", err)
	}

	seqs := [][]int32{{1, 2, 3, 4}}
	resGeGLU, err := encGeGLU.Forward(context.Background(), seqs, Options{})
	if err != nil {
		t.Fatalf("Forward GeGLU: %v", err)
	}
	resSiLU, err := encSiLU.Forward(context.Background(), seqs, Options{})
	if err != nil {
		t.Fatalf("Forward SiLU: %v", err)
	}
	resSwiGLU, err := encSwiGLU.Forward(context.Background(), seqs, Options{})
	if err != nil {
		t.Fatalf("Forward SwiGLU: %v", err)
	}

	var maxDiffGeGLUSiLU float64
	for i := range resSiLU.Hidden[0] {
		if resSiLU.Hidden[0][i] != resSwiGLU.Hidden[0][i] {
			t.Fatalf("silu and swiglu differed at %d: %v vs %v", i, resSiLU.Hidden[0][i], resSwiGLU.Hidden[0][i])
		}
		d := math.Abs(float64(resGeGLU.Hidden[0][i] - resSiLU.Hidden[0][i]))
		if d > maxDiffGeGLUSiLU {
			maxDiffGeGLUSiLU = d
		}
	}
	if maxDiffGeGLUSiLU < 1e-4 {
		t.Fatalf("expected SwiGLU forward pass to differ from GeGLU, maxDiff=%v", maxDiffGeGLUSiLU)
	}
}
