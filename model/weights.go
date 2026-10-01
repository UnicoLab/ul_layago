package model

import (
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/UnicoLab/ul_layago/safetensors"
	"github.com/UnicoLab/ul_layago/tensor"
)

// Norm is a LayerNorm's affine parameters; Bias is nil for ModernBERT.
type Norm struct {
	Weight []float32
	Bias   []float32
}

// Embedding is a token table held as float32 rows or as int8 rows with one
// scale per row.
type Embedding struct {
	Rows, Dim int
	W         []float32
	Q         []int8
	Scale     []float32
}

// Lookup writes row id into dst.
func (e *Embedding) Lookup(dst []float32, id int) {
	if e.Q == nil {
		copy(dst[:e.Dim], e.W[id*e.Dim:(id+1)*e.Dim])
		return
	}
	s := e.Scale[id]
	for i, q := range e.Q[id*e.Dim : (id+1)*e.Dim] {
		dst[i] = float32(q) * s
	}
}

// Quantize converts the table to symmetric per-row int8.
func (e *Embedding) Quantize() {
	if e.Q != nil {
		return
	}
	e.Q, e.Scale = tensor.QuantizeRows(e.W, e.Rows, e.Dim)
	e.W = nil
}

// Layer holds one ModernBERT encoder layer. AttnNorm is nil for layer 0,
// whose attention input is the embedding output itself.
type Layer struct {
	AttnNorm *Norm
	Wqkv, Wo *tensor.Linear
	MLPNorm  Norm
	Wi, WoM  *tensor.Linear
}

// Weights are the encoder parameters in float32 (or int8 where quantized).
type Weights struct {
	Embed     Embedding
	EmbedNorm Norm
	Layers    []Layer
	FinalNorm Norm
}

// LoadOptions selects the tensor-name prefix and the weight format.
type LoadOptions struct {
	// Prefix is prepended to every ModernBertModel parameter name, e.g.
	// "encoder." inside a Laya checkpoint or "model." in ModernBertForMaskedLM.
	Prefix string
	// Int8 quantizes the embedding table and every encoder linear weight to
	// symmetric per-row int8 after loading. Norms stay float32.
	Int8 bool
	// SkipDeeperLayers accepts tensors of layers at or beyond cfg.NumLayers
	// under Prefix and leaves them unread, so a config cut with
	// Config.Truncate loads the first layers of a deeper checkpoint. Any
	// other unexpected tensor is still an error.
	SkipDeeperLayers bool
}

type loader struct {
	f      *safetensors.File
	prefix string
	used   map[string]bool
}

func (l *loader) tensor(name string, shape ...int) (safetensors.Tensor, error) {
	full := l.prefix + name
	t, err := l.f.Tensor(full)
	if err != nil {
		return safetensors.Tensor{}, fmt.Errorf("model: missing weight %s", full)
	}
	if !slices.Equal(t.Shape, shape) {
		return safetensors.Tensor{}, fmt.Errorf("model: weight %s has shape %v, want %v", full, t.Shape, shape)
	}
	l.used[full] = true
	return t, nil
}

func (l *loader) floats(name string, shape ...int) ([]float32, error) {
	t, err := l.tensor(name, shape...)
	if err != nil {
		return nil, err
	}
	return t.Float32()
}

func (l *loader) norm(name string, dim int, bias bool) (Norm, error) {
	w, err := l.floats(name+".weight", dim)
	if err != nil {
		return Norm{}, err
	}
	n := Norm{Weight: w}
	if bias {
		if n.Bias, err = l.floats(name+".bias", dim); err != nil {
			return Norm{}, err
		}
	}
	return n, nil
}

// linear loads name.weight ([out, in]) as float32, or as int8 when the file
// stores I8 with a per-row F32 sidecar name.weight_scale ([out]).
func (l *loader) linear(name string, out, in int, bias bool) (*tensor.Linear, error) {
	t, err := l.tensor(name+".weight", out, in)
	if err != nil {
		return nil, err
	}
	var b []float32
	if bias {
		if b, err = l.floats(name+".bias", out); err != nil {
			return nil, err
		}
	}
	if t.DType == safetensors.I8 {
		q, err := t.Int8()
		if err != nil {
			return nil, err
		}
		scale, err := l.floats(name+".weight_scale", out)
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

// LoadWeights reads every ModernBertModel parameter under opts.Prefix,
// checking shapes against cfg, and rejects unexpected tensors under the
// prefix so a mismatched checkpoint fails loudly.
func LoadWeights(f *safetensors.File, cfg Config, opts LoadOptions) (*Weights, error) {
	l := &loader{f: f, prefix: opts.Prefix, used: map[string]bool{}}
	d, inter := cfg.HiddenSize, cfg.IntermediateSize
	w := &Weights{}
	emb, err := l.tensor("embeddings.tok_embeddings.weight", cfg.VocabSize, d)
	if err != nil {
		return nil, err
	}
	w.Embed = Embedding{Rows: cfg.VocabSize, Dim: d}
	if w.Embed.W, err = emb.Float32(); err != nil {
		return nil, err
	}
	if w.EmbedNorm, err = l.norm("embeddings.norm", d, cfg.NormBias); err != nil {
		return nil, err
	}
	for i := 0; i < cfg.NumLayers; i++ {
		p := fmt.Sprintf("layers.%d.", i)
		var layer Layer
		if i > 0 {
			n, err := l.norm(p+"attn_norm", d, cfg.NormBias)
			if err != nil {
				return nil, err
			}
			layer.AttnNorm = &n
		}
		if layer.Wqkv, err = l.linear(p+"attn.Wqkv", 3*d, d, cfg.AttentionBias); err != nil {
			return nil, err
		}
		if layer.Wo, err = l.linear(p+"attn.Wo", d, d, cfg.AttentionBias); err != nil {
			return nil, err
		}
		if layer.MLPNorm, err = l.norm(p+"mlp_norm", d, cfg.NormBias); err != nil {
			return nil, err
		}
		if layer.Wi, err = l.linear(p+"mlp.Wi", 2*inter, d, cfg.MLPBias); err != nil {
			return nil, err
		}
		if layer.WoM, err = l.linear(p+"mlp.Wo", d, inter, cfg.MLPBias); err != nil {
			return nil, err
		}
		w.Layers = append(w.Layers, layer)
	}
	if w.FinalNorm, err = l.norm("final_norm", d, cfg.NormBias); err != nil {
		return nil, err
	}
	for _, name := range f.Header.Names() {
		if !strings.HasPrefix(name, opts.Prefix) || l.used[name] {
			continue
		}
		if opts.SkipDeeperLayers && deeperLayer(strings.TrimPrefix(name, opts.Prefix), cfg.NumLayers) {
			continue
		}
		return nil, fmt.Errorf("model: unexpected weight %s", name)
	}
	if opts.Int8 {
		w.Quantize()
	}
	return w, nil
}

// deeperLayer reports whether a tensor name (without prefix) belongs to a
// layer index >= n: "layers.<i>." with i a plain decimal.
func deeperLayer(name string, n int) bool {
	rest, ok := strings.CutPrefix(name, "layers.")
	if !ok {
		return false
	}
	num, _, ok := strings.Cut(rest, ".")
	if !ok || num == "" || (len(num) > 1 && num[0] == '0') {
		return false
	}
	i, err := strconv.Atoi(num)
	return err == nil && i >= n
}

// Prepack builds every encoder linear's packed panels for GEMM kernel k now
// (see tensor.Linear.Prepack), so a loader pays the one-time re-layout, and
// the second copy of the linear weights it keeps, before the first request.
// It is a no-op for the generic kernel. Quantize after Prepack discards the
// panels.
func (w *Weights) Prepack(k tensor.Kernel) {
	for _, layer := range w.Layers {
		for _, lin := range []*tensor.Linear{layer.Wqkv, layer.Wo, layer.Wi, layer.WoM} {
			lin.Prepack(k)
		}
	}
}

// PackedBytes reports the memory held by packed panels across the encoder.
func (w *Weights) PackedBytes() int {
	total := 0
	for _, layer := range w.Layers {
		for _, lin := range []*tensor.Linear{layer.Wqkv, layer.Wo, layer.Wi, layer.WoM} {
			total += lin.PackedBytes()
		}
	}
	return total
}

// Quantize converts the embedding table and every linear weight to int8.
func (w *Weights) Quantize() {
	w.Embed.Quantize()
	for _, layer := range w.Layers {
		for _, lin := range []*tensor.Linear{layer.Wqkv, layer.Wo, layer.Wi, layer.WoM} {
			lin.Quantize()
		}
	}
}
