package laya

import (
	"context"
	"fmt"
	"maps"
	"reflect"

	"github.com/UnicoLab/ul_layago/model"
	"github.com/UnicoLab/ul_layago/tokenizer"
)

// VerifiedBase owns one verified frozen encoder and tokenizer. Its pins and
// weights remain private; attaching heads never changes them or calibration.
// It can be shared concurrently. Do not mutate loaded Artifacts' Agent weights.
type VerifiedBase struct {
	base      ArtifactBase
	tokens    ArtifactTokenizer
	sequence  ArtifactSequenceSpec
	encoder   *model.Encoder
	tokenizer *tokenizer.Tokenizer
	special   SpecialTokens
	cache     *featureCache
}

// LoadArtifactBase verifies and loads the encoder pinned by an existing head
// manifest. No head is loaded, and no files are re-opened after verification.
func LoadArtifactBase(artifactDir, baseDir string, opts ArtifactOptions) (*VerifiedBase, error) {
	art, err := readArtifactManifest(artifactDir)
	if err != nil {
		return nil, err
	}
	return loadArtifactBase(art.Manifest, baseDir, opts)
}

// LoadArtifactWithBase verifies an independent head and attaches it to a base
// already loaded once. Base/tokenizer/sequence-format pins must match exactly. Each head retains
// its own independently validated sequence and head length budgets.
func LoadArtifactWithBase(dir string, base *VerifiedBase) (*Artifact, error) {
	art, err := readArtifactManifest(dir)
	if err != nil {
		return nil, err
	}
	return attachArtifact(dir, art, base)
}

func (b *VerifiedBase) compatible(m ArtifactManifest) error {
	if b == nil {
		return fmt.Errorf("laya: a verified base is required")
	}
	if b.base != m.Base || b.sequence.Format != m.Sequence.Format || b.sequence.PromptFormat != m.Sequence.PromptFormat || b.sequence.TruncateLeft != m.Sequence.TruncateLeft || !reflect.DeepEqual(b.tokens, m.Tokenizer) {
		return fmt.Errorf("laya: artifact base, tokenizer or sequence-format pins differ from the shared verified base")
	}
	return nil
}

func loadArtifactBase(m ArtifactManifest, dir string, opts ArtifactOptions) (*VerifiedBase, error) {
	if opts.FeatureCacheBytes < 0 {
		return nil, fmt.Errorf("laya: feature cache budget must be nonnegative")
	}
	cfgRaw, err := readPinned(dir, m.Base.ConfigFile, m.Base.ConfigSHA256, maxArtifactConfigBytes)
	if err != nil {
		return nil, err
	}
	tokRaw, err := readPinned(dir, m.Tokenizer.File, m.Tokenizer.SHA256, maxArtifactTokenizerBytes)
	if err != nil {
		return nil, err
	}
	f, size, err := openPinned(dir, m.Base.WeightsFile, m.Base.WeightsSHA256)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	full, err := model.ParseHFConfig(cfgRaw)
	if err != nil {
		return nil, err
	}
	if full.NumLayers != m.Base.LayersTotal || full.HiddenSize != m.Base.HiddenSize {
		return nil, fmt.Errorf("laya: base config shape disagrees with manifest")
	}
	cfg, err := full.Truncate(m.Base.LayersUsed)
	if err != nil {
		return nil, err
	}
	enc, err := loadArtifactEncoder(f, size, cfg, m.Base.TensorPrefix, opts.Int8Encoder)
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.Load(tokRaw)
	if err != nil {
		return nil, err
	}
	sp, err := m.Tokenizer.special(tok)
	if err != nil {
		return nil, err
	}
	tokens := m.Tokenizer
	tokens.SpecialTokens = maps.Clone(m.Tokenizer.SpecialTokens)
	return &VerifiedBase{base: m.Base, tokens: tokens, sequence: m.Sequence, encoder: enc, tokenizer: tok, special: sp, cache: newFeatureCache(opts.FeatureCacheBytes)}, nil
}

// CacheStats reports retained full encoder states, not head decisions. Bytes
// counts float32 payload and a conservative per-entry overhead, not total RSS.
func (b *VerifiedBase) CacheStats() CacheStats {
	if b == nil || b.cache == nil {
		return CacheStats{}
	}
	return b.cache.stats()
}

func (b *VerifiedBase) forward(ctx context.Context, seqs [][]int32, opt model.Options) (*model.Result, error) {
	if b.cache == nil || opt.CaptureLayers || opt.Timings != nil {
		return b.encoder.Forward(ctx, seqs, opt)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	result := &model.Result{Hidden: make([][]float32, len(seqs))}
	keys := make([]featureKey, len(seqs))
	var missing [][]int32
	var positions []int
	for i, seq := range seqs {
		keys[i] = sequenceKey(seq, opt.Parallel)
		if h, ok := b.cache.get(keys[i]); ok {
			result.Hidden[i] = h
		} else {
			missing = append(missing, seq)
			positions = append(positions, i)
		}
	}
	if len(missing) > 0 {
		fresh, err := b.encoder.Forward(ctx, missing, opt)
		if err != nil {
			return nil, err
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		for j, i := range positions {
			result.Hidden[i] = fresh.Hidden[j]
			b.cache.put(keys[i], fresh.Hidden[j])
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return result, nil
}
