package laya

import (
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/UnicoLab/ul_layago/model"
	"github.com/UnicoLab/ul_layago/safetensors"
	"github.com/UnicoLab/ul_layago/tensor"
)

// headLayer is a torch.nn.TransformerEncoderLayer with norm_first=True, ReLU
// feed-forward of width 4d and biases everywhere, evaluated without dropout.
type headLayer struct {
	inProj, outProj  *tensor.Linear
	norm1, norm2     model.Norm
	linear1, linear2 *tensor.Linear
}

// Head is Laya's DecisionModel minus the encoder: a question-type embedding
// added to every position, the head transformer layers, the marker scorer
// LN→Linear→GELU→Linear→1 and the act head Linear(d+4→256)→GELU→Linear.
type Head struct {
	D, Heads int
	typeEmb  []float32 // 3 x D
	layers   []headLayer
	scorerLN model.Norm
	scorer1  *tensor.Linear
	scorer2  *tensor.Linear
	act1     *tensor.Linear
	act2     *tensor.Linear
	NAct     int
}

const torchLayerNormEps = 1e-5

// loadHead reads the non-encoder DecisionModel parameters and verifies that
// no other tensor outside encoderPrefix is present.
func loadHead(f *safetensors.File, d, headLayers, nAct int, encoderPrefix string) (*Head, error) {
	used := map[string]bool{}
	get := func(name string, shape ...int) ([]float32, error) {
		t, err := f.Tensor(name)
		if err != nil {
			return nil, fmt.Errorf("laya: missing weight %s", name)
		}
		if !slices.Equal(t.Shape, shape) {
			return nil, fmt.Errorf("laya: weight %s has shape %v, want %v", name, t.Shape, shape)
		}
		used[name] = true
		return t.Float32()
	}
	// A linear weight is float (F32/F16/BF16/F64) or, in an int8
	// hive.laya-artifact/v1 head, I8 with a per-row F32 sidecar
	// "<weight name>_scale" (tensor.QuantizeRows layout).
	linear := func(prefix, wName, bName string, out, in int) (*tensor.Linear, error) {
		name := prefix + wName
		t, err := f.Tensor(name)
		if err != nil {
			return nil, fmt.Errorf("laya: missing weight %s", name)
		}
		if !slices.Equal(t.Shape, []int{out, in}) {
			return nil, fmt.Errorf("laya: weight %s has shape %v, want %v", name, t.Shape, []int{out, in})
		}
		used[name] = true
		b, err := get(prefix+bName, out)
		if err != nil {
			return nil, err
		}
		if t.DType == safetensors.I8 {
			q, err := t.Int8()
			if err != nil {
				return nil, err
			}
			scale, err := get(name+"_scale", out)
			if err != nil {
				return nil, err
			}
			return tensor.NewQuantizedLinear(out, in, q, scale, b)
		}
		w, err := t.Float32()
		if err != nil {
			return nil, err
		}
		return tensor.NewLinear(out, in, w, b)
	}
	norm := func(name string) (model.Norm, error) {
		w, err := get(name+".weight", d)
		if err != nil {
			return model.Norm{}, err
		}
		b, err := get(name+".bias", d)
		return model.Norm{Weight: w, Bias: b}, err
	}
	h := &Head{D: d, Heads: max(1, d/64), NAct: nAct}
	if d%h.Heads != 0 {
		return nil, fmt.Errorf("laya: hidden size %d is not divisible by %d heads", d, h.Heads)
	}
	var err error
	if h.typeEmb, err = get("type_emb.weight", 3, d); err != nil {
		return nil, err
	}
	for i := 0; i < headLayers; i++ {
		p := fmt.Sprintf("head.layers.%d.", i)
		var l headLayer
		if l.inProj, err = linear(p, "self_attn.in_proj_weight", "self_attn.in_proj_bias", 3*d, d); err != nil {
			return nil, err
		}
		if l.outProj, err = linear(p, "self_attn.out_proj.weight", "self_attn.out_proj.bias", d, d); err != nil {
			return nil, err
		}
		if l.linear1, err = linear(p, "linear1.weight", "linear1.bias", 4*d, d); err != nil {
			return nil, err
		}
		if l.linear2, err = linear(p, "linear2.weight", "linear2.bias", d, 4*d); err != nil {
			return nil, err
		}
		if l.norm1, err = norm(p + "norm1"); err != nil {
			return nil, err
		}
		if l.norm2, err = norm(p + "norm2"); err != nil {
			return nil, err
		}
		h.layers = append(h.layers, l)
	}
	if h.scorerLN, err = norm("scorer.0"); err != nil {
		return nil, err
	}
	if h.scorer1, err = linear("scorer.1.", "weight", "bias", d, d); err != nil {
		return nil, err
	}
	if h.scorer2, err = linear("scorer.3.", "weight", "bias", 1, d); err != nil {
		return nil, err
	}
	if h.act1, err = linear("act_head.0.", "weight", "bias", 256, d+4); err != nil {
		return nil, err
	}
	if h.act2, err = linear("act_head.2.", "weight", "bias", nAct, 256); err != nil {
		return nil, err
	}
	// The temperature buffer is saved with the weights but inference uses
	// the config's temperatures, as laya/agent.py does.
	if _, ok := f.Header.Lookup("temperature"); ok {
		used["temperature"] = true
	}
	for _, name := range f.Header.Names() {
		// An empty prefix means the file holds the head alone (a
		// hive.laya-artifact/v1 export), where every tensor must be used.
		if encoderPrefix != "" && strings.HasPrefix(name, encoderPrefix) {
			continue
		}
		if !used[name] {
			return nil, fmt.Errorf("laya: unexpected weight %s", name)
		}
	}
	return h, nil
}

// headOutput is the head's raw result for one question sequence.
type headOutput struct {
	logits    []float32 // one per marker, before temperature
	actLogits []float32
}

// Prepack builds the head linears' packed GEMM panels for kernel k now
// instead of on first use (see tensor.Linear.Prepack).
func (h *Head) Prepack(k tensor.Kernel) {
	for _, l := range h.layers {
		for _, lin := range []*tensor.Linear{l.inProj, l.outProj, l.linear1, l.linear2} {
			lin.Prepack(k)
		}
	}
	for _, lin := range []*tensor.Linear{h.scorer1, h.scorer2, h.act1, h.act2} {
		lin.Prepack(k)
	}
}

// headInput is one question sequence for the head.
type headInput struct {
	hidden  []float32 // len x D encoder output
	qtype   QType
	markers []int
}

// forward runs the decision head over the encoder outputs of each question.
func (h *Head) forward(items []headInput, par tensor.Parallel) []headOutput {
	d := h.D
	starts := make([]int, len(items)+1)
	for s, it := range items {
		starts[s+1] = starts[s] + len(it.hidden)/d
	}
	total := starts[len(items)]
	x := make([]float32, total*d)
	for s, it := range items {
		copy(x[starts[s]*d:], it.hidden)
		q := int(it.qtype)
		te := h.typeEmb[q*d : (q+1)*d]
		for r := starts[s]; r < starts[s+1]; r++ {
			tensor.AddInPlace(x[r*d:(r+1)*d], te)
		}
	}
	a := make([]float32, total*d)
	qkv := make([]float32, total*3*d)
	attn := make([]float32, total*d)
	tmp := make([]float32, total*d)
	ff := make([]float32, total*4*d)
	hd := d / h.Heads
	scale := float32(1 / math.Sqrt(float64(hd)))
	for _, l := range h.layers {
		tensor.LayerNorm(a, x, total, d, l.norm1.Weight, l.norm1.Bias, torchLayerNormEps, par)
		l.inProj.Forward(qkv, a, total, par)
		selfAttention(attn, qkv, starts, d, h.Heads, hd, scale, par)
		l.outProj.Forward(tmp, attn, total, par)
		tensor.AddInPlace(x, tmp)
		tensor.LayerNorm(a, x, total, d, l.norm2.Weight, l.norm2.Bias, torchLayerNormEps, par)
		l.linear1.Forward(ff, a, total, par)
		tensor.ReLUInPlace(ff)
		l.linear2.Forward(tmp, ff, total, par)
		tensor.AddInPlace(x, tmp)
	}

	out := make([]headOutput, len(items))
	for s, it := range items {
		k := len(it.markers)
		rows := make([]float32, k*d)
		for i, m := range it.markers {
			copy(rows[i*d:(i+1)*d], x[(starts[s]+m)*d:(starts[s]+m+1)*d])
		}
		normed := make([]float32, k*d)
		tensor.LayerNorm(normed, rows, k, d, h.scorerLN.Weight, h.scorerLN.Bias, torchLayerNormEps, tensor.Parallel{Workers: 1})
		hid := make([]float32, k*d)
		h.scorer1.Forward(hid, normed, k, tensor.Parallel{Workers: 1})
		tensor.GELUInPlace(hid, tensor.Parallel{Workers: 1})
		logits := make([]float32, k)
		h.scorer2.Forward(logits, hid, k, tensor.Parallel{Workers: 1})

		feats := ActFeatures(logits)
		in := make([]float32, d+4)
		copy(in, x[starts[s]*d:(starts[s]+1)*d]) // pooled = h[:, 0]
		copy(in[d:], feats[:])
		mid := make([]float32, 256)
		h.act1.Forward(mid, in, 1, tensor.Parallel{Workers: 1})
		tensor.GELUInPlace(mid, tensor.Parallel{Workers: 1})
		act := make([]float32, h.NAct)
		h.act2.Forward(act, mid, 1, tensor.Parallel{Workers: 1})
		out[s] = headOutput{logits: logits, actLogits: act}
	}
	return out
}

// ActFeatures is DecisionModel.forward's confidence vector: [top1,
// top1-top2, normalized entropy, k/255] over softmax(logits), with k clamped
// to at least 2 and top2 = 0 for a single option.
func ActFeatures(logits []float32) [4]float32 {
	p := append([]float32(nil), logits...)
	tensor.SoftmaxInPlace(p)
	top1, top2 := float32(-1), float32(-1)
	for _, v := range p {
		if v > top1 {
			top1, top2 = v, top1
		} else if v > top2 {
			top2 = v
		}
	}
	if top2 < 0 {
		top2 = 0
	}
	k := float64(max(2, len(p)))
	var ent float64
	for _, v := range p {
		ent -= float64(v) * math.Log(math.Max(float64(v), 1e-9))
	}
	ent /= math.Log(k)
	return [4]float32{top1, top1 - top2, float32(ent), float32(k / 255)}
}

// selfAttention is nn.MultiheadAttention without masking across sequences:
// each sequence attends only to itself (the padding mask of a batch).
func selfAttention(out, qkv []float32, starts []int, d, heads, hd int, scale float32, par tensor.Parallel) {
	nseq := len(starts) - 1
	par.Run(nseq*heads, func(task int) {
		s, hh := task/heads, task%heads
		start, n := starts[s], starts[s+1]-starts[s]
		scores := make([]float32, n)
		acc := make([]float32, hd)
		for i := 0; i < n; i++ {
			q := qkv[(start+i)*3*d+hh*hd : (start+i)*3*d+(hh+1)*hd]
			for j := 0; j < n; j++ {
				k := qkv[(start+j)*3*d+d+hh*hd : (start+j)*3*d+d+(hh+1)*hd]
				scores[j] = tensor.Dot(q, k) * scale
			}
			tensor.SoftmaxInPlace(scores)
			clear(acc)
			for j := 0; j < n; j++ {
				p := scores[j]
				v := qkv[(start+j)*3*d+2*d+hh*hd : (start+j)*3*d+2*d+(hh+1)*hd]
				for t := range acc {
					acc[t] += p * v[t]
				}
			}
			copy(out[(start+i)*d+hh*hd:(start+i)*d+(hh+1)*hd], acc)
		}
	})
}
