package model

import (
	"math"
	"math/rand/v2"

	"github.com/UnicoLab/ul_layago/tensor"
)

// MMBERTBase is the mmBERT-base / Laya multilingual backbone shape: 22
// layers, 768 hidden, 12 heads, GeGLU intermediate 1152, a 256k Gemma-2
// vocabulary, global attention every third layer with a 128-token sliding
// window elsewhere, and RoPE theta 160000 for both layer types.
func MMBERTBase() Config {
	cfg := Config{
		VocabSize: 256000, HiddenSize: 768, IntermediateSize: 1152, NumLayers: 22, NumHeads: 12,
		NormEps: 1e-5,
	}
	for i := 0; i < cfg.NumLayers; i++ {
		if i%3 == 0 {
			cfg.Layers = append(cfg.Layers, LayerSpec{Global: true, Window: -1, RopeTheta: 160000})
		} else {
			cfg.Layers = append(cfg.Layers, LayerSpec{Window: 64, RopeTheta: 160000})
		}
	}
	return cfg
}

// RandomWeights builds deterministic random weights for cfg, for latency
// benchmarks and load tests only. Linear weights are N(0, 1/in) so
// activations stay in a normal range (no denormals) through every layer.
func RandomWeights(cfg Config, seed uint64) *Weights {
	r := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15)) //nolint:gosec // G404: deterministic benchmark weights, not a security use
	fill := func(n int, std float64) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = float32(r.NormFloat64() * std)
		}
		return out
	}
	ones := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = 1
		}
		return out
	}
	linear := func(out, in int) *tensor.Linear {
		l, err := tensor.NewLinear(out, in, fill(out*in, 1/math.Sqrt(float64(in))), nil)
		if err != nil {
			panic(err) // sizes come from cfg and are always consistent
		}
		return l
	}
	d, inter := cfg.HiddenSize, cfg.IntermediateSize
	w := &Weights{
		Embed:     Embedding{Rows: cfg.VocabSize, Dim: d, W: fill(cfg.VocabSize*d, 1)},
		EmbedNorm: Norm{Weight: ones(d)},
		FinalNorm: Norm{Weight: ones(d)},
	}
	for i := 0; i < cfg.NumLayers; i++ {
		layer := Layer{
			Wqkv: linear(3*d, d), Wo: linear(d, d),
			MLPNorm: Norm{Weight: ones(d)},
			Wi:      linear(2*inter, d), WoM: linear(d, inter),
		}
		if i > 0 {
			layer.AttnNorm = &Norm{Weight: ones(d)}
		}
		w.Layers = append(w.Layers, layer)
	}
	return w
}

// FLOPs returns the multiply-add work of one forward pass over one sequence
// of n tokens, counting 2 FLOPs per multiply-add: the dense layers plus the
// attention score and value products within each layer's window.
func (c Config) FLOPs(n int) float64 {
	d, inter := float64(c.HiddenSize), float64(c.IntermediateSize)
	dense := 2 * float64(n) * float64(c.NumLayers) * (3*d*d + d*d + 2*inter*d + inter*d)
	var attn float64
	for _, l := range c.Layers {
		keys := 0
		for i := 0; i < n; i++ {
			lo, hi := 0, n-1
			if !l.Global {
				lo, hi = max(0, i-l.Window), min(n-1, i+l.Window)
			}
			keys += hi - lo + 1
		}
		attn += 2 * 2 * float64(keys) * d
	}
	return dense + attn
}
