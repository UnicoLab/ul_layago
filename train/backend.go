package train

import (
	"context"
	"fmt"
	"math"
)

// FitSample contains only a fitting split's frozen marker states and labels.
// Held-out/test data and dev_conf are never given to an accelerated fitter.
// Fitters must treat these borrowed features/targets as read-only.
type FitSample struct {
	ID      string    `json:"id"`
	Markers []float32 `json:"markers"`
	K       int       `json:"k"`
	QType   int       `json:"qtype"`
	Type    TaskType  `json:"type"`
	Target  []float64 `json:"target"`
	Label   int       `json:"label"`
	Weight  float64   `json:"weight"`
}
type ScorerFit struct {
	Head      *Head
	History   []EpochStats
	BestEpoch int
	Info      map[string]any
}
type ScorerFitFunc func(context.Context, *Head, []FitSample, []FitSample, Config) (*ScorerFit, error)

func fitSamples(samples []*sample) []FitSample {
	out := make([]FitSample, len(samples))
	for i, s := range samples {
		out[i] = FitSample{ID: s.it.Ex.ID, Markers: s.rec.Markers, K: s.rec.K, QType: int(s.it.Ex.Type.QType()), Type: s.it.Ex.Type, Target: s.target, Label: s.it.Ex.Label, Weight: s.weight}
	}
	return out
}
func advance(cfg Config, stage string, step int) {
	if cfg.Progress != nil {
		cfg.Progress(stage, step)
	}
}
func validateScorerFit(initial *Head, fit *ScorerFit) error {
	if fit == nil || fit.Head == nil || fit.Head.Config != initial.Config || len(fit.History) == 0 || fit.BestEpoch < 1 || fit.BestEpoch > len(fit.History) {
		return fmt.Errorf("train: invalid accelerated scorer result")
	}
	for i, p := range fit.Head.Params() {
		if len(p.Data) != len(initial.Params()[i].Data) {
			return fmt.Errorf("train: invalid backend tensor shape %s", p.Name)
		}
		for _, v := range p.Data {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return fmt.Errorf("train: non-finite backend parameter %s", p.Name)
			}
		}
	}
	for i, st := range fit.History {
		if st.Epoch != i+1 || math.IsNaN(st.TrainLoss) || math.IsInf(st.TrainLoss, 0) || math.IsNaN(st.DevCalNLL) || math.IsInf(st.DevCalNLL, 0) || math.IsNaN(st.DevCalAccuracy) || st.Seconds < 0 || math.IsNaN(st.Seconds) || math.IsInf(st.Seconds, 0) || st.DevCalAccuracy < 0 || st.DevCalAccuracy > 1 {
			return fmt.Errorf("train: invalid backend history epoch %d", st.Epoch)
		}
	}
	return nil
}

func (cfg Config) scorerScheduleEpochs() int {
	if cfg.ScheduleEpochs > 0 {
		return cfg.ScheduleEpochs
	}
	return cfg.Epochs
}
