package train

import (
	"fmt"

	"github.com/UnicoLab/ul_layago/tensor"
)

// ScoredItem is one prepared item decoded by an artifact: the served
// logits and act logits from its cached trunk features, and the row under
// the artifact's own temperatures and thresholds.
type ScoredItem struct {
	Item      *Item
	Row       Row
	Logits    []float32
	ActLogits []float32
}

// ScoreCached decodes every prepared item with the artifact's head,
// temperatures and act thresholds from cached trunk features, exactly as
// Train decodes them for the manifest's evaluation and Predictor serves
// them: on the artifact's own features, Evaluate over the rows reproduces
// the manifest's evaluation.splits. recs must hold the features of every
// scorable item; unscorable items escalate. p must be prepared as the
// artifact was (same data, options and trunk).
func (a *Artifact) ScoreCached(p *Prepared, recs map[string]*Record, workers int) ([]ScoredItem, error) {
	m := &a.Manifest
	if p.Instance != m.Instance {
		return nil, fmt.Errorf("train: items of instance %s, the artifact serves %s", p.Instance, m.Instance)
	}
	d := m.Base.HiddenSize
	for _, it := range p.Items {
		if _, ok := m.Tasks[it.Ex.Task]; !ok {
			return nil, fmt.Errorf("train: %s: the artifact has no task %q", it.Ex.ID, it.Ex.Task)
		}
		if !it.Scorable {
			continue
		}
		rec := recs[it.Ex.ID]
		if rec == nil {
			return nil, fmt.Errorf("train: no cached features for %s", it.Ex.ID)
		}
		if rec.K != it.Ex.NumOptions() || len(rec.Pooled) != d || len(rec.Markers) != rec.K*d {
			return nil, fmt.Errorf("train: features of %s do not match its %d options at width %d", it.Ex.ID, it.Ex.NumOptions(), d)
		}
	}
	temps, thresholds := m.Calibration.Temperatures, m.Act.Thresholds
	out := make([]ScoredItem, len(p.Items))
	tensor.Parallel{Workers: workers}.Run(len(p.Items), func(i int) {
		it := p.Items[i]
		out[i].Item = it
		if !it.Scorable {
			out[i].Row = decodeRow(it, nil, nil, temps, thresholds)
			return
		}
		rec := recs[it.Ex.ID]
		qt := it.Ex.Type.QType()
		logits := a.Head.Logits(rec.Markers, rec.K, qt)
		act := a.Head.ActLogits(rec.Pooled, qt, logits)
		out[i].Row, out[i].Logits, out[i].ActLogits = decodeRow(it, logits, act, temps, thresholds), logits, act
	})
	return out, nil
}

// Evaluate summarises rows per split, language, task and language × task
// with the manifest's metric definitions (evaluation.splits).
func Evaluate(rows []Row) map[string]SplitEval { return evaluate(rows) }
