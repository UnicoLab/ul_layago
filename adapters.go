package layago

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/UnicoLab/ul_layago/laya"
	"github.com/UnicoLab/ul_layago/tensor"
)

var ErrUnknownAdapter = errors.New("layago: explicit registered adapter ID required")

// AdapterSetOptions loads independent existing trained head artifacts over one
// verified frozen base. A head adapter is not encoder LoRA. CacheBytes is opt-in
// retained whole-sequence float32 state storage, not a process memory limit.
type AdapterSetOptions struct {
	Base       string
	Artifacts  map[string]string
	Workers    int
	Kernel     tensor.Kernel
	CacheBytes int64
}

// AdapterRequest explicitly selects an immutable head/calibration bundle.
// Ordinary single-model Request remains unchanged.
type AdapterRequest struct {
	Request
	AdapterID string `json:"adapter_id"`
}

// AdapterSet keeps one encoder/tokenizer and independent calibrated heads.
// Selection is per request, never global mutable state. Requests share a
// cancellation-aware compute gate, bounding concurrent encoder work.
type AdapterSet struct {
	predictors map[string]*Predictor
	base       *laya.VerifiedBase
}

func OpenAdapters(opts AdapterSetOptions) (*AdapterSet, error) {
	if opts.Base == "" || len(opts.Artifacts) == 0 {
		return nil, errors.New("layago: adapters require a base and at least one artifact")
	}
	if opts.Workers < 0 || opts.CacheBytes < 0 {
		return nil, errors.New("layago: workers and cache bytes must be nonnegative")
	}
	if _, err := tensor.ParseKernel(opts.Kernel.String()); err != nil {
		return nil, err
	}
	ids := make([]string, 0, len(opts.Artifacts))
	for id, dir := range opts.Artifacts {
		if strings.TrimSpace(id) == "" || dir == "" {
			return nil, errors.New("layago: adapter IDs and artifact paths must be nonempty")
		}
		ids = append(ids, id)
	}
	sort.Strings(ids)
	base, err := laya.LoadArtifactBase(opts.Artifacts[ids[0]], opts.Base, laya.ArtifactOptions{FeatureCacheBytes: opts.CacheBytes})
	if err != nil {
		return nil, err
	}
	out := &AdapterSet{base: base, predictors: map[string]*Predictor{}}
	gate := make(chan struct{}, 1)
	par := tensor.Parallel{Workers: opts.Workers, Kernel: opts.Kernel}
	for _, id := range ids {
		art, err := laya.LoadArtifactWithBase(opts.Artifacts[id], base)
		if err != nil {
			return nil, fmt.Errorf("layago: adapter %q: %w", id, err)
		}
		art.Agent.Prepack(par.Kernel)
		out.predictors[id] = &Predictor{agent: art.Agent, artifact: art, parallel: par, gate: gate, metadata: Metadata{Kind: "artifact", SHA256: art.ManifestSHA256, Kernel: par.EffectiveKernel().String(), Workers: par.N(), MaxLen: art.Agent.Config.MaxLen, AdapterID: id}}
	}
	return out, nil
}

func (s *AdapterSet) Predict(ctx context.Context, adapterID string, r Request) (*Response, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if s == nil {
		return nil, ErrUnknownAdapter
	}
	p, ok := s.predictors[adapterID]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrUnknownAdapter, adapterID)
	}
	return p.Predict(ctx, r)
}

// Metadata returns a copy; caller changes cannot alter the loaded registry.
func (s *AdapterSet) Metadata() map[string]Metadata {
	out := map[string]Metadata{}
	if s != nil {
		for id, p := range s.predictors {
			out[id] = p.Metadata()
		}
	}
	return out
}
func (s *AdapterSet) CacheStats() laya.CacheStats {
	if s == nil {
		return laya.CacheStats{}
	}
	return s.base.CacheStats()
}
