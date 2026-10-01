package laya

import (
	"math"
	"math/rand/v2"

	"github.com/UnicoLab/ul_layago/model"
	"github.com/UnicoLab/ul_layago/tensor"
)

// RandomHead builds a decision head of Laya's shape with deterministic random
// weights, for latency benchmarks only.
func RandomHead(d, headLayers, nAct int, seed uint64) *Head {
	r := rand.New(rand.NewPCG(seed, seed^0x5851f42d4c957f2d)) //nolint:gosec // G404: deterministic benchmark weights, not a security use
	fill := func(n int, std float64) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = float32(r.NormFloat64() * std)
		}
		return out
	}
	linear := func(out, in int) *tensor.Linear {
		l, err := tensor.NewLinear(out, in, fill(out*in, 1/math.Sqrt(float64(in))), fill(out, 0.1))
		if err != nil {
			panic(err) // sizes are derived from d and always consistent
		}
		return l
	}
	norm := func() model.Norm {
		w := fill(d, 0.1)
		for i := range w {
			w[i]++
		}
		return model.Norm{Weight: w, Bias: fill(d, 0.1)}
	}
	h := &Head{D: d, Heads: max(1, d/64), NAct: nAct, typeEmb: fill(3*d, 0.5)}
	for i := 0; i < headLayers; i++ {
		h.layers = append(h.layers, headLayer{
			inProj: linear(3*d, d), outProj: linear(d, d),
			norm1: norm(), norm2: norm(),
			linear1: linear(4*d, d), linear2: linear(d, 4*d),
		})
	}
	h.scorerLN = norm()
	h.scorer1 = linear(d, d)
	h.scorer2 = linear(1, d)
	h.act1 = linear(256, d+4)
	h.act2 = linear(nAct, 256)
	return h
}
