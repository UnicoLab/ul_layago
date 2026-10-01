package train

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"

	"github.com/UnicoLab/ul_layago/laya"
	"github.com/UnicoLab/ul_layago/model"
	"github.com/UnicoLab/ul_layago/safetensors"
	"github.com/UnicoLab/ul_layago/tensor"
	"github.com/UnicoLab/ul_layago/tokenizer"
)

// TrunkSpec selects the frozen encoder and the sequence layout.
type TrunkSpec struct {
	// Dir is a Hugging Face ModernBERT directory (config.json,
	// model.safetensors with "model." or unprefixed tensors, tokenizer.json,
	// tokenizer_config.json) or a Laya checkpoint directory (encoder/,
	// tokenizer/, model.safetensors with "encoder." tensors).
	Dir string
	// Layers keeps the first Layers encoder layers (0 keeps all): a
	// fewer-layer trunk of the same engine.
	Layers int
	// Int8 runs the trunk on the per-row int8 path (benchmarks only: a head
	// is trained and served on float32 trunk states).
	Int8 bool
	// MaxLen and HeadMaxLen are Laya's sequence and head budgets
	// (build_sequence); zero selects 256 and 192.
	MaxLen, HeadMaxLen int
	// ModelID names the trunk in manifests; empty reads MANIFEST.json's
	// model_id, else the directory name.
	ModelID string
}

func (s TrunkSpec) withDefaults() TrunkSpec {
	if s.MaxLen == 0 {
		s.MaxLen = 256
	}
	if s.HeadMaxLen == 0 {
		s.HeadMaxLen = 192
	}
	return s
}

// TrunkIdentity pins the exact trunk features were computed with. Its base
// fields are the manifest's "base" section of hive.laya-artifact/v1: every
// file name is relative to the base directory, which laya.LoadArtifact and
// routermodel.Load are given at serving time.
type TrunkIdentity struct {
	ModelID         string   `json:"model_id"`
	ConfigFile      string   `json:"config_file"`
	ConfigSHA256    string   `json:"config_sha256"`
	WeightsFile     string   `json:"weights_file"`
	WeightsSHA256   string   `json:"weights_sha256"`
	TensorPrefix    string   `json:"tensor_prefix"`
	LayersTotal     int      `json:"layers_total"`
	LayersUsed      int      `json:"layers_used"`
	LayerTypes      []string `json:"layer_types"`
	HiddenSize      int      `json:"hidden_size"`
	TokenizerFile   string   `json:"-"`
	TokenizerSHA256 string   `json:"-"`
	// Conversion records how the weights file was produced when it was
	// converted (MANIFEST.json of a converted checkpoint), else nil.
	Conversion map[string]any `json:"conversion"`
	// NumericPath, MaxLen and HeadMaxLen are not base fields; they key the
	// feature cache.
	NumericPath string `json:"-"`
	MaxLen      int    `json:"-"`
	HeadMaxLen  int    `json:"-"`
}

// Trunk is a loaded frozen encoder with its tokenizer.
type Trunk struct {
	Spec      TrunkSpec
	Identity  TrunkIdentity
	Encoder   *model.Encoder
	Tokenizer *tokenizer.Tokenizer
	Special   laya.SpecialTokens
}

type trunkFiles struct {
	config, weights, tokenizer, tokenizerConfig, prefix string
}

func locate(dir string) (trunkFiles, error) {
	f := trunkFiles{config: "config.json", weights: "model.safetensors", tokenizer: "tokenizer.json", //nolint:gosec // G101: file names, not credentials
		tokenizerConfig: "tokenizer_config.json"}
	if _, err := os.Stat(filepath.Join(dir, "rl_agent_config.json")); err == nil {
		f = trunkFiles{config: "encoder/config.json", weights: "model.safetensors", tokenizer: "tokenizer/tokenizer.json",
			tokenizerConfig: "tokenizer/tokenizer_config.json", prefix: "encoder."}
	}
	for _, p := range []string{f.config, f.weights, f.tokenizer, f.tokenizerConfig} {
		if _, err := os.Stat(filepath.Join(dir, p)); err != nil {
			return trunkFiles{}, fmt.Errorf("train: trunk directory %s: %w", dir, err)
		}
	}
	return f, nil
}

// FileSHA256 hashes a file in 1 MiB reads.
func FileSHA256(path string) (string, error) {
	f, err := os.Open(path) //nolint:gosec // G304: model files chosen by the operator
	if err != nil {
		return "", err
	}
	defer func() { _ = f.Close() }()
	h := sha256.New()
	if _, err := io.CopyBuffer(h, f, make([]byte, 1<<20)); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// OpenTrunk loads and fingerprints a frozen trunk.
func OpenTrunk(spec TrunkSpec) (*Trunk, error) {
	spec = spec.withDefaults()
	if spec.HeadMaxLen >= spec.MaxLen {
		return nil, fmt.Errorf("train: head_max_len %d must be below max_len %d", spec.HeadMaxLen, spec.MaxLen)
	}
	files, err := locate(spec.Dir)
	if err != nil {
		return nil, err
	}
	path := func(rel string) string { return filepath.Join(spec.Dir, rel) }
	rawCfg, err := os.ReadFile(path(files.config))
	if err != nil {
		return nil, err
	}
	cfg, err := model.ParseHFConfig(rawCfg)
	if err != nil {
		return nil, err
	}
	id := TrunkIdentity{ModelID: spec.ModelID, ConfigFile: files.config, WeightsFile: files.weights, TokenizerFile: files.tokenizer,
		LayersTotal: cfg.NumLayers, HiddenSize: cfg.HiddenSize, MaxLen: spec.MaxLen, HeadMaxLen: spec.HeadMaxLen, NumericPath: "fp32"}
	if spec.Int8 {
		id.NumericPath = "int8"
	}
	var manifest struct {
		ModelID  string `json:"model_id"`
		Revision string `json:"revision"`
		Source   string `json:"source_bin_sha256"`
	}
	if raw, err := os.ReadFile(path("MANIFEST.json")); err == nil && json.Unmarshal(raw, &manifest) == nil {
		if id.ModelID == "" {
			id.ModelID = manifest.ModelID
		}
		if manifest.Source != "" {
			id.Conversion = map[string]any{"source": "pytorch_model.bin", "source_sha256": manifest.Source,
				"revision": manifest.Revision, "output": files.weights}
		}
	}
	var checkpoint struct {
		Encoder string `json:"encoder"`
	}
	if raw, err := os.ReadFile(path("rl_agent_config.json")); err == nil && json.Unmarshal(raw, &checkpoint) == nil && id.ModelID == "" {
		id.ModelID = checkpoint.Encoder
	}
	if id.ModelID == "" {
		id.ModelID = filepath.Base(filepath.Clean(spec.Dir))
	}
	for _, item := range []struct {
		rel string
		dst *string
	}{{files.weights, &id.WeightsSHA256}, {files.config, &id.ConfigSHA256}, {files.tokenizer, &id.TokenizerSHA256}} {
		if *item.dst, err = FileSHA256(path(item.rel)); err != nil {
			return nil, err
		}
	}
	st, err := safetensors.Open(path(files.weights))
	if err != nil {
		return nil, err
	}
	defer func() { _ = st.Close() }()
	prefix := files.prefix
	if prefix == "" {
		if _, ok := st.Header.Lookup("model.embeddings.tok_embeddings.weight"); ok {
			prefix = "model."
		}
	}
	id.TensorPrefix = prefix
	w, err := model.LoadWeights(st, cfg, model.LoadOptions{Prefix: prefix, Int8: spec.Int8})
	if err != nil {
		return nil, err
	}
	if spec.Layers < 0 || spec.Layers > cfg.NumLayers {
		return nil, fmt.Errorf("train: cannot keep %d of %d layers", spec.Layers, cfg.NumLayers)
	}
	if spec.Layers > 0 && spec.Layers < cfg.NumLayers {
		// The first n layers with their own attention layouts; the final
		// norm follows layer n-1 (model.Config.Truncate on the loader side).
		cfg.NumLayers = spec.Layers
		cfg.Layers = append([]model.LayerSpec(nil), cfg.Layers[:spec.Layers]...)
		w.Layers = w.Layers[:spec.Layers]
	}
	id.LayersUsed = cfg.NumLayers
	id.LayerTypes = layerTypes(cfg)
	enc, err := model.NewEncoder(cfg, w)
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.LoadFile(path(files.tokenizer))
	if err != nil {
		return nil, err
	}
	sp, err := laya.LoadSpecialTokens(path(files.tokenizerConfig), tok)
	if err != nil {
		return nil, err
	}
	return &Trunk{Spec: spec, Identity: id, Encoder: enc, Tokenizer: tok, Special: sp}, nil
}

// layerTypes names each used layer's attention as transformers does.
func layerTypes(cfg model.Config) []string {
	out := make([]string, len(cfg.Layers))
	for i, l := range cfg.Layers {
		out[i] = "sliding_attention"
		if l.Global {
			out[i] = "full_attention"
		}
	}
	return out
}

// NewTrunk wraps an already loaded encoder (tests, random configs). The
// identity digests and file names are whatever the caller states.
func NewTrunk(spec TrunkSpec, id TrunkIdentity, enc *model.Encoder, tok *tokenizer.Tokenizer, sp laya.SpecialTokens) (*Trunk, error) {
	spec = spec.withDefaults()
	if spec.HeadMaxLen >= spec.MaxLen {
		return nil, fmt.Errorf("train: head_max_len %d must be below max_len %d", spec.HeadMaxLen, spec.MaxLen)
	}
	id.MaxLen, id.HeadMaxLen = spec.MaxLen, spec.HeadMaxLen
	id.LayersUsed, id.HiddenSize, id.LayerTypes = enc.Config.NumLayers, enc.Config.HiddenSize, layerTypes(enc.Config)
	if id.LayersTotal == 0 {
		id.LayersTotal = enc.Config.NumLayers
	}
	if id.NumericPath == "" {
		id.NumericPath = "fp32"
	}
	return &Trunk{Spec: spec, Identity: id, Encoder: enc, Tokenizer: tok, Special: sp}, nil
}

// SpecialTokenManifest is the tokenizer's special tokens with their ids, as
// the manifest's tokenizer.special_tokens records them.
func (t *Trunk) SpecialTokenManifest() map[string]SpecialToken {
	sp := t.Special
	return map[string]SpecialToken{
		"cls": {Token: t.Tokenizer.Token(sp.CLS), ID: sp.CLS}, "sep": {Token: t.Tokenizer.Token(sp.SEP), ID: sp.SEP},
		"mask": {Token: t.Tokenizer.Token(sp.Mask), ID: sp.Mask}, "pad": {Token: t.Tokenizer.Token(sp.Pad), ID: sp.Pad},
	}
}

// SpecialToken is one special token with its id.
type SpecialToken struct {
	Token string `json:"token"`
	ID    int32  `json:"id"`
}

// Features is the frozen trunk's output for one sequence: the final hidden
// state at position 0 and at each option marker.
type Features struct {
	Pooled  []float32 // D
	Markers []float32 // K x D
	K       int
}

// Features runs the trunk over sequences (stacked, each attending only to
// itself) and gathers the pooled and marker states. A sequence's result
// does not depend on what it is stacked with.
func (t *Trunk) Features(ctx context.Context, seqs [][]int32, markers [][]int, par tensor.Parallel) ([]Features, error) {
	res, err := t.Encoder.Forward(ctx, seqs, model.Options{Parallel: par})
	if err != nil {
		return nil, err
	}
	if len(markers) != len(res.Hidden) {
		return nil, fmt.Errorf("train: %d marker lists for %d sequences", len(markers), len(res.Hidden))
	}
	d := t.Encoder.Config.HiddenSize
	out := make([]Features, len(markers))
	for i, mk := range markers {
		h := res.Hidden[i]
		f := Features{Pooled: append([]float32(nil), h[:d]...), K: len(mk), Markers: make([]float32, len(mk)*d)}
		for j, m := range mk {
			copy(f.Markers[j*d:(j+1)*d], h[m*d:(m+1)*d])
		}
		out[i] = f
	}
	return out, nil
}

// kernelName records the GEMM kernel features were computed with: the
// assembly kernels are bit-identical to the portable one on arm64 but not on
// amd64, so the cache key carries the architecture and kernel.
func kernelName(par tensor.Parallel) string {
	return runtime.GOARCH + "/" + par.EffectiveKernel().String()
}
