// Package model implements the ModernBERT encoder forward pass (the mmBERT
// backbone Laya uses) in portable Go: token embeddings and LayerNorm, then per
// layer a pre-norm attention block with a fused Wqkv projection, rotary
// position embeddings and either global or sliding-window attention, and a
// pre-norm GeGLU MLP, then a final LayerNorm. It matches Hugging Face's
// ModernBertModel (transformers 4.57) for inference with the SDPA attention
// path.
package model

import (
	"encoding/json"
	"fmt"
)

// LayerSpec is one layer's attention layout, taken from the exported config
// rather than recomputed from the layer index.
type LayerSpec struct {
	Global bool
	// Window is the largest |i-j| a sliding layer may attend across
	// (local_attention/2, inclusive, as in transformers 4.57). -1 for global.
	Window    int
	RopeTheta float64
}

// Config is the subset of a Hugging Face ModernBERT config.json the forward
// pass depends on.
type Config struct {
	VocabSize        int
	HiddenSize       int
	IntermediateSize int
	NumLayers        int
	NumHeads         int
	NormEps          float64
	NormBias         bool
	AttentionBias    bool
	MLPBias          bool
	HiddenActivation string
	PadTokenID       int
	Layers           []LayerSpec
}

// HeadDim returns HiddenSize / NumHeads.
func (c Config) HeadDim() int { return c.HiddenSize / c.NumHeads }

// IsGlobalLayer reports whether layer l (0-indexed) uses global full attention.
func (c Config) IsGlobalLayer(l int) bool {
	return l >= 0 && l < len(c.Layers) && c.Layers[l].Global
}

// Truncate returns the config of the first n layers: a student that keeps
// the checkpoint's first layers with their own attention types and rope
// thetas, the final norm then following layer n-1. Load it with
// LoadOptions.SkipDeeperLayers.
func (c Config) Truncate(n int) (Config, error) {
	if n < 1 || n > c.NumLayers {
		return Config{}, fmt.Errorf("model: cannot keep %d of %d layers", n, c.NumLayers)
	}
	out := c
	out.NumLayers = n
	out.Layers = append([]LayerSpec(nil), c.Layers[:n]...)
	return out, nil
}

// TruncateToGlobalBoundary returns a truncated Config with the largest layer
// count <= targetLayers whose final layer (count - 1) is a global attention
// layer (UltraBERT Appendix A.2), falling back to Truncate(targetLayers) when
// no global layer exists at or below targetLayers.
func (c Config) TruncateToGlobalBoundary(targetLayers int) Config {
	if c.NumLayers <= 0 || targetLayers <= 0 || targetLayers >= c.NumLayers {
		return c
	}
	for count := targetLayers; count >= 1; count-- {
		if c.IsGlobalLayer(count - 1) {
			out, err := c.Truncate(count)
			if err == nil {
				return out
			}
		}
	}
	out, err := c.Truncate(targetLayers)
	if err != nil {
		return c
	}
	return out
}

type hfConfig struct {
	ModelType              string   `json:"model_type"`
	VocabSize              int      `json:"vocab_size"`
	HiddenSize             int      `json:"hidden_size"`
	IntermediateSize       int      `json:"intermediate_size"`
	NumHiddenLayers        int      `json:"num_hidden_layers"`
	NumAttentionHeads      int      `json:"num_attention_heads"`
	NormEps                *float64 `json:"norm_eps"`
	LayerNormEps           *float64 `json:"layer_norm_eps"`
	NormBias               bool     `json:"norm_bias"`
	AttentionBias          bool     `json:"attention_bias"`
	MLPBias                bool     `json:"mlp_bias"`
	HiddenActivation       string   `json:"hidden_activation"`
	LocalAttention         int      `json:"local_attention"`
	GlobalAttnEveryNLayers int      `json:"global_attn_every_n_layers"`
	LayerTypes             []string `json:"layer_types"`
	GlobalRopeTheta        *float64 `json:"global_rope_theta"`
	LocalRopeTheta         *float64 `json:"local_rope_theta"`
	RopeParameters         map[string]struct {
		RopeTheta *float64 `json:"rope_theta"`
		RopeType  string   `json:"rope_type"`
	} `json:"rope_parameters"`
	PadTokenID int `json:"pad_token_id"`
}

// ParseHFConfig reads a ModernBERT/UltraBERT config.json in either the transformers 4
// layout (global_attn_every_n_layers, global/local_rope_theta) or the
// transformers 5 layout (layer_types, rope_parameters). Anything the forward
// pass cannot reproduce exactly is rejected.
func ParseHFConfig(data []byte) (Config, error) {
	var h hfConfig
	if err := json.Unmarshal(data, &h); err != nil {
		return Config{}, fmt.Errorf("model: parse config: %w", err)
	}
	if h.ModelType != "modernbert" && h.ModelType != "ultrabert" {
		return Config{}, fmt.Errorf("model: model_type %q is not modernbert or ultrabert", h.ModelType)
	}
	var hiddenAct string
	switch h.HiddenActivation {
	case "", "gelu":
		hiddenAct = "gelu"
	case "silu", "swiglu":
		hiddenAct = h.HiddenActivation
	default:
		return Config{}, fmt.Errorf("model: hidden_activation %q is not supported (want exact-erf gelu or silu/swiglu)", h.HiddenActivation)
	}
	c := Config{
		VocabSize: h.VocabSize, HiddenSize: h.HiddenSize, IntermediateSize: h.IntermediateSize,
		NumLayers: h.NumHiddenLayers, NumHeads: h.NumAttentionHeads,
		NormBias: h.NormBias, AttentionBias: h.AttentionBias, MLPBias: h.MLPBias,
		HiddenActivation: hiddenAct, PadTokenID: h.PadTokenID,
	}
	switch {
	case h.NormEps != nil:
		c.NormEps = *h.NormEps
	case h.LayerNormEps != nil:
		c.NormEps = *h.LayerNormEps
	default:
		return Config{}, fmt.Errorf("model: config declares no norm_eps")
	}
	if c.VocabSize <= 0 || c.HiddenSize <= 0 || c.IntermediateSize <= 0 || c.NumLayers <= 0 || c.NumHeads <= 0 {
		return Config{}, fmt.Errorf("model: config sizes must be positive")
	}
	if c.HiddenSize%c.NumHeads != 0 || c.HeadDim()%2 != 0 {
		return Config{}, fmt.Errorf("model: hidden_size %d is not an even multiple of %d heads", c.HiddenSize, c.NumHeads)
	}
	globalTheta, localTheta := h.GlobalRopeTheta, h.LocalRopeTheta
	if p, ok := h.RopeParameters["full_attention"]; ok && p.RopeTheta != nil {
		if p.RopeType != "" && p.RopeType != "default" {
			return Config{}, fmt.Errorf("model: rope_type %q is not supported", p.RopeType)
		}
		globalTheta = p.RopeTheta
	}
	if p, ok := h.RopeParameters["sliding_attention"]; ok && p.RopeTheta != nil {
		if p.RopeType != "" && p.RopeType != "default" {
			return Config{}, fmt.Errorf("model: rope_type %q is not supported", p.RopeType)
		}
		localTheta = p.RopeTheta
	}
	if globalTheta == nil {
		return Config{}, fmt.Errorf("model: config declares no global rope theta")
	}
	types := h.LayerTypes
	if len(types) == 0 {
		if h.GlobalAttnEveryNLayers <= 0 {
			return Config{}, fmt.Errorf("model: config declares neither layer_types nor global_attn_every_n_layers")
		}
		for i := 0; i < c.NumLayers; i++ {
			if i%h.GlobalAttnEveryNLayers == 0 {
				types = append(types, "full_attention")
			} else {
				types = append(types, "sliding_attention")
			}
		}
	}
	if len(types) != c.NumLayers {
		return Config{}, fmt.Errorf("model: %d layer_types for %d layers", len(types), c.NumLayers)
	}
	for i, t := range types {
		switch t {
		case "full_attention":
			c.Layers = append(c.Layers, LayerSpec{Global: true, Window: -1, RopeTheta: *globalTheta})
		case "sliding_attention":
			if localTheta == nil || h.LocalAttention <= 0 {
				return Config{}, fmt.Errorf("model: sliding layer %d needs local_attention and a local rope theta", i)
			}
			c.Layers = append(c.Layers, LayerSpec{Window: h.LocalAttention / 2, RopeTheta: *localTheta})
		default:
			return Config{}, fmt.Errorf("model: layer %d has unsupported type %q", i, t)
		}
	}
	return c, nil
}
