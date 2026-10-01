// Package evalreport computes the evaluation receipt of trained Laya heads
// from their per-item predictions (laya-train eval): per task accuracy and
// macro-F1 with 95 % bootstrap intervals that resample whole stories, per
// language accuracy, top-1 and per-label expected calibration error before
// and after temperature scaling, act-threshold coverage and selective risk,
// confusion, majority-class baselines, and paired comparisons of two models
// on the same items (story-paired bootstrap and exact McNemar, with Holm's
// adjustment over each family of tests). With focus languages it also
// reports every task per focus language on the focus splits (accuracy,
// macro-F1, the majority baseline on the same items, calibration including
// the worst label, and the act thresholds as served) and the ratio of the
// second focus language's accuracy to the first's against a target.
// Standard library only.
//
// Examples of one story share their question, catalog and wording, so they
// are not independent: every interval here resamples stories, not items.
// McNemar's test treats items as independent pairs and is reported beside
// the story-paired interval, which is the one to read when they disagree.
package evalreport

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"sort"
)

// Contract names the report layout.
const Contract = "hive.laya-eval-report/v1"

// Prediction is one evaluated item of one model, as laya-train eval writes
// it. Probs are the served probabilities under the artifact's temperature,
// ProbsT1 the same logits at temperature 1.
type Prediction struct {
	Model    string   `json:"model"`
	ID       string   `json:"id"`
	Split    string   `json:"split"`
	Story    string   `json:"story_id"`
	Lang     string   `json:"lang"`
	Domain   string   `json:"domain_id"`
	Task     string   `json:"task"`
	TaskType string   `json:"task_type"`
	Options  string   `json:"options"`
	Keys     []string `json:"keys"`
	Label    int      `json:"label"`
	// Gold is the gold key when Label is -1: a catalog shortlist offered
	// options without it, so the item cannot be answered correctly.
	Gold           string    `json:"gold,omitempty"`
	Scorable       bool      `json:"scorable"`
	Probs          []float64 `json:"probs,omitempty"`
	ProbsT1        []float64 `json:"probs_t1,omitempty"`
	Temperature    float64   `json:"temperature"`
	TemperatureKey string    `json:"temperature_key"`
	Pred           int       `json:"pred"`
	ScoreKind      string    `json:"score_kind"`
	Lambda         *float64  `json:"lambda"`
	Certified      bool      `json:"certified"`
	DecisionScore  float64   `json:"decision_score"`
	ActProbability float64   `json:"act_probability"`
	Acted          bool      `json:"acted"`
	// Slices are the dataset's evaluation slice names for the item (for
	// example unseen_template).
	Slices []string `json:"slices,omitempty"`
}

// Correct reports whether a scorable item's prediction is its label.
func (p *Prediction) Correct() bool { return p.Scorable && p.Label >= 0 && p.Pred == p.Label }

// GoldKey is the label's class key.
func (p *Prediction) GoldKey() string {
	if p.Label < 0 {
		return p.Gold
	}
	return p.Keys[p.Label]
}

// PredKey is the prediction's class key ("" when unscorable).
func (p *Prediction) PredKey() string {
	if !p.Scorable {
		return ""
	}
	return p.Keys[p.Pred]
}

// Baseline is a majority-class predictor of one task, fitted on train
// labels: the most frequent class key of a fixed-option task, or the most
// frequent label position of a task whose options vary per example.
type Baseline struct {
	Options string  `json:"options"`
	Key     string  `json:"key,omitempty"`
	Index   int     `json:"index"`
	TrainN  int     `json:"train_examples"`
	Share   float64 `json:"train_share"`
}

// predict is the baseline's class key for an item.
func (b Baseline) predict(p *Prediction) string {
	if b.Options == "fixed" {
		return b.Key
	}
	return p.Keys[min(b.Index, len(p.Keys)-1)]
}

// Config controls a report.
type Config struct {
	// Splits are the evaluated splits, in report order.
	Splits []string `json:"splits"`
	// Boot is the number of bootstrap resamples per interval; Seed salts
	// every interval's generator (each interval also hashes its own cell,
	// so results do not depend on the order cells are computed in).
	Boot int    `json:"bootstrap_resamples"`
	Seed uint64 `json:"bootstrap_seed"`
	// Bins is the number of equal-width calibration bins (the manifest's 10).
	Bins int `json:"ece_bins"`
	// MinLabelSupport is the fewest items offering a label for its
	// one-vs-rest calibration error to enter the worst-label figure.
	MinLabelSupport int `json:"min_label_support"`
	// ConfusionTasks get a confusion record; TopPairs bounds the listed
	// off-diagonal pairs of a task whose options vary per example.
	ConfusionTasks []string `json:"confusion_tasks"`
	TopPairs       int      `json:"top_confusion_pairs"`
	// Alpha is the act thresholds' risk bound, for the risk-bound check.
	Alpha float64 `json:"alpha"`
	// FocusLangs are the languages reported per task on FocusSplits (the
	// primary results when some languages matter most); with two or more,
	// the second's accuracy over the first's is reported per task against
	// RatioTarget.
	FocusLangs  []string `json:"focus_langs,omitempty"`
	FocusSplits []string `json:"focus_splits,omitempty"`
	RatioTarget float64  `json:"ratio_target,omitempty"`
	// PairedLangs, when set, restricts every paired comparison to items of
	// these languages.
	PairedLangs []string `json:"paired_langs,omitempty"`
}

// DefaultConfig is the configuration laya-train report defaults to.
func DefaultConfig() Config {
	return Config{Splits: []string{"dev_conf", "heldout_domain", "heldout_language"}, Boot: 2000, Seed: 20260928, Bins: 10,
		MinLabelSupport: 50, ConfusionTasks: []string{"q_operation", "q_metric_choice"}, TopPairs: 15, Alpha: 0.05}
}

// Estimate is a statistic with its 95 % story-bootstrap percentile interval.
// Lo and Hi are absent when the statistic is undefined on the cell.
type Estimate struct {
	Value   *float64 `json:"value"`
	Lo      *float64 `json:"ci95_lo"`
	Hi      *float64 `json:"ci95_hi"`
	N       int      `json:"n"`
	Stories int      `json:"stories"`
}

// Calibration compares expected calibration errors before (T = 1) and
// after the artifact's temperatures.
type Calibration struct {
	Temperatures map[string]float64 `json:"temperatures_used"`
	Top1T1       float64            `json:"ece_top1_t1"`
	Top1         float64            `json:"ece_top1"`
	// WorstLabel is the largest one-vs-rest ECE over labels offered by at
	// least MinLabelSupport items; nil when no label has that support
	// (typical of tasks whose options vary per example).
	WorstLabelT1 *LabelECE `json:"worst_label_t1"`
	WorstLabel   *LabelECE `json:"worst_label"`
	// EligibleLabels counts the labels with enough support.
	EligibleLabels int `json:"eligible_labels"`
	// Pooled is one-vs-rest ECE over every (item, option) pair.
	PooledT1 float64 `json:"ece_pooled_ovr_t1"`
	Pooled   float64 `json:"ece_pooled_ovr"`
}

// LabelECE is one label's one-vs-rest calibration error.
type LabelECE struct {
	Label   string  `json:"label"`
	ECE     float64 `json:"ece"`
	Support int     `json:"support"`
}

// Act is a task's act-threshold outcome on a split: the threshold
// certified on dev_conf, applied as served.
type Act struct {
	ScoreKind string   `json:"score_kind"`
	Lambda    *float64 `json:"lambda"`
	Certified bool     `json:"certified"`
	N         int      `json:"n"`
	Acted     int      `json:"acted"`
	Errors    int      `json:"errors"`
	Coverage  Estimate `json:"coverage"`
	// Risk is the error rate among acted items; RiskUpper95 its one-sided
	// 95 % Clopper-Pearson bound, and WithinAlpha whether that bound is at
	// most Alpha (nil when nothing acted).
	Risk        Estimate `json:"selective_risk"`
	RiskUpper95 *float64 `json:"risk_upper95_clopper_pearson"`
	WithinAlpha *bool    `json:"risk_upper95_within_alpha"`
}

// Confusion is a task's confusion on a split. Fixed-option tasks carry the
// full matrix (Matrix[gold][pred]); tasks whose options vary per example
// carry their most frequent off-diagonal pairs.
type Confusion struct {
	Labels []string         `json:"labels,omitempty"`
	Matrix [][]int          `json:"matrix,omitempty"`
	Pairs  []ConfusionPair  `json:"top_pairs,omitempty"`
	Errors int              `json:"errors"`
	ByGold map[string][]int `json:"errors_by_gold,omitempty"`
}

// ConfusionPair is one off-diagonal (gold, predicted) pair.
type ConfusionPair struct {
	Gold  string `json:"gold"`
	Pred  string `json:"pred"`
	Count int    `json:"count"`
}

// TaskReport is one task of one model on one split.
type TaskReport struct {
	N           int         `json:"n"`
	Stories     int         `json:"stories"`
	Unscorable  int         `json:"unscorable"`
	Accuracy    Estimate    `json:"accuracy"`
	MacroF1     Estimate    `json:"macro_f1"`
	Calibration Calibration `json:"calibration"`
	Act         Act         `json:"act"`
	Confusion   *Confusion  `json:"confusion,omitempty"`
}

// LangReport is one language of one model on one split: accuracy over every
// task, and accuracy and macro-F1 per task (points only).
type LangReport struct {
	Accuracy Estimate             `json:"accuracy"`
	Tasks    map[string]PointCell `json:"tasks"`
}

// PointCell is a cell reported without an interval.
type PointCell struct {
	N        int     `json:"n"`
	Accuracy float64 `json:"accuracy"`
	MacroF1  float64 `json:"macro_f1"`
}

// SplitReport is one model on one split.
type SplitReport struct {
	N         int                    `json:"n"`
	Stories   int                    `json:"stories"`
	Accuracy  Estimate               `json:"accuracy"`
	Tasks     map[string]*TaskReport `json:"tasks"`
	Languages map[string]*LangReport `json:"languages"`
}

// ModelReport is one model on every evaluated split, and on the focus
// languages of the focus splits.
type ModelReport struct {
	Model  string                  `json:"model"`
	Splits map[string]*SplitReport `json:"splits"`
	Focus  map[string]*FocusSplit  `json:"focus,omitempty"`
}

// BaselineCell is a majority baseline on one split and task.
type BaselineCell struct {
	Accuracy Estimate `json:"accuracy"`
	MacroF1  Estimate `json:"macro_f1"`
}

// Paired compares model A with model B on the items both scored, per split
// and task ("*" pools every task of the split).
type Paired struct {
	A     string              `json:"model_a"`
	B     string              `json:"model_b"`
	Cells []PairedCell        `json:"cells"`
	Holm  map[string][]string `json:"holm_families"`
}

// PairedCell is one paired comparison. DeltaAccuracy and DeltaMacroF1 are
// A minus B with story-paired bootstrap intervals; B01 counts items A got
// right and B wrong, B10 the reverse; P is the exact two-sided McNemar p,
// PHolm its Holm adjustment within the cell's family.
type PairedCell struct {
	Split         string   `json:"split"`
	Task          string   `json:"task"`
	N             int      `json:"n"`
	Stories       int      `json:"stories"`
	AccuracyA     float64  `json:"accuracy_a"`
	AccuracyB     float64  `json:"accuracy_b"`
	DeltaAccuracy Estimate `json:"delta_accuracy"`
	DeltaMacroF1  Estimate `json:"delta_macro_f1"`
	AOnlyRight    int      `json:"a_right_b_wrong"`
	BOnlyRight    int      `json:"a_wrong_b_right"`
	PredFlips     int      `json:"prediction_differs"`
	ActFlips      int      `json:"act_decision_differs"`
	P             float64  `json:"mcnemar_exact_p"`
	PHolm         float64  `json:"mcnemar_p_holm"`
}

// Report is the whole evaluation receipt body.
type Report struct {
	Contract  string                             `json:"contract"`
	Config    Config                             `json:"config"`
	Models    []*ModelReport                     `json:"models"`
	Baselines map[string]Baseline                `json:"majority_baseline_definition"`
	Majority  map[string]map[string]BaselineCell `json:"majority_baseline"`
	Paired    []*Paired                          `json:"paired"`
}

// Build computes the report. preds holds each model's predictions (every
// split; items outside cfg.Splits are ignored); baselines are fitted on
// train by the caller; pairs names the (A, B) model comparisons.
func Build(preds map[string][]*Prediction, baselines map[string]Baseline, pairs [][2]string, cfg Config) (*Report, error) {
	if cfg.Boot < 1 || cfg.Bins < 1 || len(cfg.Splits) == 0 {
		return nil, fmt.Errorf("evalreport: bootstrap %d, bins %d and splits %v must be set", cfg.Boot, cfg.Bins, cfg.Splits)
	}
	r := &Report{Contract: Contract, Config: cfg, Baselines: baselines, Majority: map[string]map[string]BaselineCell{}}
	names := make([]string, 0, len(preds))
	for name := range preds {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		mr, err := buildModel(name, preds[name], cfg)
		if err != nil {
			return nil, err
		}
		mr.Focus = buildFocus(name, preds[name], baselines, cfg)
		r.Models = append(r.Models, mr)
	}
	if len(names) > 0 && len(baselines) > 0 {
		for _, split := range cfg.Splits {
			byTask := groupBy(inSplit(preds[names[0]], split), func(p *Prediction) string { return p.Task })
			r.Majority[split] = map[string]BaselineCell{}
			for _, task := range sortedKeys(byTask) {
				b, ok := baselines[task]
				if !ok {
					continue
				}
				// The baseline is scored on the scorable items, the items
				// every model accuracy is computed on.
				var items []*Prediction
				for _, p := range byTask[task] {
					if p.Scorable {
						items = append(items, p)
					}
				}
				hit := func(sel []*Prediction) (float64, bool) { return baselineAccuracy(b, sel) }
				f1 := func(sel []*Prediction) (float64, bool) {
					gold, pred := make([]string, len(sel)), make([]string, len(sel))
					for i, p := range sel {
						gold[i], pred[i] = p.GoldKey(), b.predict(p)
					}
					return macroF1(gold, pred), len(sel) > 0
				}
				key := "majority/" + split + "/" + task
				r.Majority[split][task] = BaselineCell{Accuracy: estimate(items, hit, cfg, key+"/acc"),
					MacroF1: estimate(items, f1, cfg, key+"/f1")}
			}
		}
	}
	for _, pair := range pairs {
		a, b := preds[pair[0]], preds[pair[1]]
		if a == nil || b == nil {
			return nil, fmt.Errorf("evalreport: no predictions for %s or %s", pair[0], pair[1])
		}
		pr, err := buildPaired(pair[0], pair[1], a, b, cfg)
		if err != nil {
			return nil, err
		}
		r.Paired = append(r.Paired, pr)
	}
	return r, nil
}

func inSplit(preds []*Prediction, split string) []*Prediction {
	var out []*Prediction
	for _, p := range preds {
		if p.Split == split {
			out = append(out, p)
		}
	}
	return out
}

func groupBy(preds []*Prediction, key func(*Prediction) string) map[string][]*Prediction {
	out := map[string][]*Prediction{}
	for _, p := range preds {
		out[key(p)] = append(out[key(p)], p)
	}
	return out
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	return keys
}

func stories(preds []*Prediction) int {
	s := map[string]bool{}
	for _, p := range preds {
		s[p.Story] = true
	}
	return len(s)
}

func accuracyStat(sel []*Prediction) (float64, bool) {
	n, c := 0, 0
	for _, p := range sel {
		if !p.Scorable {
			continue
		}
		n++
		if p.Correct() {
			c++
		}
	}
	if n == 0 {
		return 0, false
	}
	return float64(c) / float64(n), true
}

func macroF1Stat(sel []*Prediction) (float64, bool) {
	var gold, pred []string
	for _, p := range sel {
		if p.Scorable {
			gold, pred = append(gold, p.GoldKey()), append(pred, p.PredKey())
		}
	}
	if len(gold) == 0 {
		return 0, false
	}
	return macroF1(gold, pred), true
}

func buildModel(name string, preds []*Prediction, cfg Config) (*ModelReport, error) {
	mr := &ModelReport{Model: name, Splits: map[string]*SplitReport{}}
	for _, p := range preds {
		if (p.Label < 0 && (p.Label != -1 || p.Gold == "")) || p.Label >= len(p.Keys) || (p.Scorable && (p.Pred < 0 || p.Pred >= len(p.Keys) ||
			len(p.Probs) != len(p.Keys) || len(p.ProbsT1) != len(p.Keys))) {
			return nil, fmt.Errorf("evalreport: %s %s: label, prediction or probabilities do not match its %d options", name, p.ID, len(p.Keys))
		}
	}
	for _, split := range cfg.Splits {
		items := inSplit(preds, split)
		if len(items) == 0 {
			continue
		}
		sr := &SplitReport{N: len(items), Stories: stories(items), Tasks: map[string]*TaskReport{}, Languages: map[string]*LangReport{}}
		sr.Accuracy = estimate(items, accuracyStat, cfg, name+"/"+split+"/acc")
		byTask := groupBy(items, func(p *Prediction) string { return p.Task })
		for _, task := range sortedKeys(byTask) {
			ti := byTask[task]
			key := name + "/" + split + "/" + task
			tr := &TaskReport{N: len(ti), Stories: stories(ti), Accuracy: estimate(ti, accuracyStat, cfg, key+"/acc"),
				MacroF1: estimate(ti, macroF1Stat, cfg, key+"/f1"), Calibration: calibration(ti, cfg), Act: actReport(ti, cfg, key)}
			for _, p := range ti {
				if !p.Scorable {
					tr.Unscorable++
				}
			}
			if slices.Contains(cfg.ConfusionTasks, task) {
				tr.Confusion = confusion(ti, cfg.TopPairs)
			}
			sr.Tasks[task] = tr
		}
		byLang := groupBy(items, func(p *Prediction) string { return p.Lang })
		for _, lang := range sortedKeys(byLang) {
			li := byLang[lang]
			lr := &LangReport{Accuracy: estimate(li, accuracyStat, cfg, name+"/"+split+"/lang/"+lang), Tasks: map[string]PointCell{}}
			for task, lt := range groupBy(li, func(p *Prediction) string { return p.Task }) {
				acc, _ := accuracyStat(lt)
				f1, _ := macroF1Stat(lt)
				lr.Tasks[task] = PointCell{N: len(lt), Accuracy: acc, MacroF1: f1}
			}
			sr.Languages[lang] = lr
		}
		mr.Splits[split] = sr
	}
	return mr, nil
}

// cellRNG seeds one interval's generator from the report seed and the
// cell's name.
func cellRNG(seed uint64, cell string) *rand.Rand {
	h := sha256.Sum256([]byte(cell))
	return rand.New(rand.NewPCG(seed, binary.LittleEndian.Uint64(h[:8]))) //nolint:gosec // G404: reproducible resampling, not a security use
}

// clusters groups item indices by story, in first-appearance order.
func clusters(preds []*Prediction) [][]int {
	idx := map[string]int{}
	var out [][]int
	for i, p := range preds {
		j, ok := idx[p.Story]
		if !ok {
			j = len(out)
			idx[p.Story] = j
			out = append(out, nil)
		}
		out[j] = append(out[j], i)
	}
	return out
}

// estimate computes stat on the cell and its percentile interval over
// cfg.Boot resamples of whole stories; resamples where stat is undefined
// are left out.
func estimate(preds []*Prediction, stat func([]*Prediction) (float64, bool), cfg Config, cell string) Estimate {
	e := Estimate{N: len(preds), Stories: stories(preds)}
	v, ok := stat(preds)
	if !ok {
		return e
	}
	e.Value = ptr(v)
	cl := clusters(preds)
	rng := cellRNG(cfg.Seed, cell)
	vals := make([]float64, 0, cfg.Boot)
	sel := make([]*Prediction, 0, len(preds))
	for b := 0; b < cfg.Boot; b++ {
		sel = sel[:0]
		for range cl {
			for _, i := range cl[rng.IntN(len(cl))] {
				sel = append(sel, preds[i])
			}
		}
		if x, ok := stat(sel); ok {
			vals = append(vals, x)
		}
	}
	if len(vals) > 0 {
		sort.Float64s(vals)
		e.Lo, e.Hi = ptr(quantile(vals, 0.025)), ptr(quantile(vals, 0.975))
	}
	return e
}

func ptr[T any](v T) *T { return &v }

// quantile is the linear-interpolation quantile of sorted values (numpy's
// default).
func quantile(sorted []float64, q float64) float64 {
	h := q * float64(len(sorted)-1)
	lo := int(math.Floor(h))
	if lo >= len(sorted)-1 {
		return sorted[len(sorted)-1]
	}
	return sorted[lo] + (h-float64(lo))*(sorted[lo+1]-sorted[lo])
}

// macroF1 is the unweighted mean F1 over the classes that occur as gold or
// prediction: the trainer's (and the manifest's) definition.
func macroF1(gold, pred []string) float64 {
	type counts struct{ tp, fp, fn int }
	cs := map[string]*counts{}
	get := func(k string) *counts {
		c := cs[k]
		if c == nil {
			c = &counts{}
			cs[k] = c
		}
		return c
	}
	for i := range gold {
		if gold[i] == pred[i] {
			get(gold[i]).tp++
			continue
		}
		get(pred[i]).fp++
		get(gold[i]).fn++
	}
	var sum float64
	for _, k := range sortedKeys(cs) {
		c := cs[k]
		sum += 2 * float64(c.tp) / float64(2*c.tp+c.fp+c.fn)
	}
	return sum / float64(len(cs))
}

// ece is the expected calibration error of confidences against outcomes
// over equal-width bins, computed exactly as the trainer's (and so the
// manifest's) ECE: edges i·(1/bins) with the last at 1, the first bin
// closed [0, e1], the others (lo, hi], each bin's items summed in order.
func ece(conf []float64, hit []bool, bins int) float64 {
	if len(conf) == 0 {
		return 0
	}
	edges := make([]float64, bins+1)
	for i := range edges {
		edges[i] = float64(i) * (1 / float64(bins))
	}
	edges[bins] = 1
	var total float64
	n := float64(len(conf))
	for b := 0; b < bins; b++ {
		lo, hi := edges[b], edges[b+1]
		var cnt, sumConf, sumHit float64
		for i, c := range conf {
			if !(c <= hi && (c > lo || (b == 0 && c >= lo))) {
				continue
			}
			cnt++
			sumConf += c
			if hit[i] {
				sumHit++
			}
		}
		if cnt > 0 {
			total += cnt / n * math.Abs(sumConf/cnt-sumHit/cnt)
		}
	}
	return total
}

func calibration(preds []*Prediction, cfg Config) Calibration {
	c := Calibration{Temperatures: map[string]float64{}}
	var topT1, top []float64
	var topHit []bool
	type pairs struct {
		conf, confT1 []float64
		hit          []bool
	}
	byLabel := map[string]*pairs{}
	pooled := &pairs{}
	for _, p := range preds {
		if !p.Scorable {
			continue
		}
		c.Temperatures[p.TemperatureKey] = p.Temperature
		topT1 = append(topT1, p.ProbsT1[argmax(p.ProbsT1)])
		top = append(top, p.Probs[p.Pred])
		topHit = append(topHit, p.Correct())
		for j, k := range p.Keys {
			lp := byLabel[k]
			if lp == nil {
				lp = &pairs{}
				byLabel[k] = lp
			}
			y := j == p.Label
			for _, q := range []*pairs{lp, pooled} {
				q.conf, q.confT1, q.hit = append(q.conf, p.Probs[j]), append(q.confT1, p.ProbsT1[j]), append(q.hit, y)
			}
		}
	}
	// ProbsT1's argmax equals Probs' (a positive temperature keeps the
	// order), so the same outcome is scored before and after.
	c.Top1T1, c.Top1 = ece(topT1, topHit, cfg.Bins), ece(top, topHit, cfg.Bins)
	c.PooledT1, c.Pooled = ece(pooled.confT1, pooled.hit, cfg.Bins), ece(pooled.conf, pooled.hit, cfg.Bins)
	for _, k := range sortedKeys(byLabel) {
		lp := byLabel[k]
		if len(lp.hit) < cfg.MinLabelSupport {
			continue
		}
		c.EligibleLabels++
		if e := ece(lp.confT1, lp.hit, cfg.Bins); c.WorstLabelT1 == nil || e > c.WorstLabelT1.ECE {
			c.WorstLabelT1 = &LabelECE{Label: k, ECE: e, Support: len(lp.hit)}
		}
		if e := ece(lp.conf, lp.hit, cfg.Bins); c.WorstLabel == nil || e > c.WorstLabel.ECE {
			c.WorstLabel = &LabelECE{Label: k, ECE: e, Support: len(lp.hit)}
		}
	}
	return c
}

func argmax(p []float64) int {
	best := 0
	for i, v := range p {
		if v > p[best] {
			best = i
		}
	}
	return best
}

func actReport(preds []*Prediction, cfg Config, cell string) Act {
	a := Act{N: len(preds)}
	for _, p := range preds {
		a.ScoreKind, a.Lambda, a.Certified = p.ScoreKind, p.Lambda, p.Certified
		if p.Acted {
			a.Acted++
			if !p.Correct() {
				a.Errors++
			}
		}
	}
	cov := func(sel []*Prediction) (float64, bool) {
		n := 0
		for _, p := range sel {
			if p.Acted {
				n++
			}
		}
		return float64(n) / float64(max(1, len(sel))), len(sel) > 0
	}
	risk := func(sel []*Prediction) (float64, bool) {
		n, e := 0, 0
		for _, p := range sel {
			if p.Acted {
				n++
				if !p.Correct() {
					e++
				}
			}
		}
		return float64(e) / float64(max(1, n)), n > 0
	}
	a.Coverage = estimate(preds, cov, cfg, cell+"/coverage")
	a.Risk = estimate(preds, risk, cfg, cell+"/risk")
	if a.Acted > 0 {
		u := clopperPearsonUpper(a.Errors, a.Acted, 0.05)
		a.RiskUpper95 = &u
		within := u <= cfg.Alpha
		a.WithinAlpha = &within
	}
	return a
}

func confusion(preds []*Prediction, top int) *Confusion {
	c := &Confusion{}
	fixed := len(preds) > 0 && preds[0].Options == "fixed"
	if fixed {
		c.Labels = slices.Clone(preds[0].Keys)
		c.Matrix = make([][]int, len(c.Labels))
		for i := range c.Matrix {
			c.Matrix[i] = make([]int, len(c.Labels))
		}
	}
	pairs := map[[2]string]int{}
	byGold := map[string][]int{}
	for _, p := range preds {
		if !p.Scorable {
			continue
		}
		g, q := p.GoldKey(), p.PredKey()
		if fixed && p.Label >= 0 {
			c.Matrix[p.Label][p.Pred]++
		}
		counts := byGold[g]
		if counts == nil {
			counts = []int{0, 0}
		}
		counts[0]++
		if g != q {
			c.Errors++
			counts[1]++
			pairs[[2]string{g, q}]++
		}
		byGold[g] = counts
	}
	if !fixed {
		for k, n := range pairs {
			c.Pairs = append(c.Pairs, ConfusionPair{Gold: k[0], Pred: k[1], Count: n})
		}
		sort.Slice(c.Pairs, func(i, j int) bool {
			a, b := c.Pairs[i], c.Pairs[j]
			if a.Count != b.Count {
				return a.Count > b.Count
			}
			return a.Gold+"\x00"+a.Pred < b.Gold+"\x00"+b.Pred
		})
		if len(c.Pairs) > top {
			c.Pairs = c.Pairs[:top]
		}
		// errors_by_gold lists [items, errors] for gold labels with an
		// error, the fixed matrix already holding them.
		c.ByGold = map[string][]int{}
		for g, v := range byGold {
			if v[1] > 0 {
				c.ByGold[g] = v
			}
		}
	}
	return c
}

// pairedItem joins one item's predictions from both models.
type pairedItem struct {
	a, b *Prediction
}

func buildPaired(nameA, nameB string, a, b []*Prediction, cfg Config) (*Paired, error) {
	byID := map[string]*Prediction{}
	for _, p := range b {
		byID[p.ID] = p
	}
	out := &Paired{A: nameA, B: nameB, Holm: map[string][]string{}}
	var perTask, pooled []int
	for _, split := range cfg.Splits {
		var joined []pairedItem
		for _, p := range inSplit(a, split) {
			if len(cfg.PairedLangs) > 0 && !slices.Contains(cfg.PairedLangs, primaryLang(p.Lang)) {
				continue
			}
			q := byID[p.ID]
			if q == nil {
				return nil, fmt.Errorf("evalreport: %s has no prediction for %s", nameB, p.ID)
			}
			if q.GoldKey() != p.GoldKey() || q.Task != p.Task || q.Story != p.Story || q.Split != p.Split {
				return nil, fmt.Errorf("evalreport: %s differs between %s and %s", p.ID, nameA, nameB)
			}
			if p.Scorable && q.Scorable {
				joined = append(joined, pairedItem{p, q})
			}
		}
		if len(joined) == 0 {
			continue
		}
		byTask := map[string][]pairedItem{"*": joined}
		for _, j := range joined {
			byTask[j.a.Task] = append(byTask[j.a.Task], j)
		}
		for _, task := range sortedKeys(byTask) {
			cell := pairedCell(split, task, byTask[task], nameA+" vs "+nameB, cfg)
			if task == "*" {
				pooled = append(pooled, len(out.Cells))
			} else {
				perTask = append(perTask, len(out.Cells))
			}
			out.Cells = append(out.Cells, cell)
		}
	}
	for family, idx := range map[string][]int{"per_task": perTask, "pooled_per_split": pooled} {
		ps := make([]float64, len(idx))
		for i, j := range idx {
			ps[i] = out.Cells[j].P
		}
		adj := Holm(ps)
		var members []string
		for i, j := range idx {
			out.Cells[j].PHolm = adj[i]
			members = append(members, out.Cells[j].Split+"/"+out.Cells[j].Task)
		}
		out.Holm[family] = members
	}
	return out, nil
}

func pairedCell(split, task string, items []pairedItem, name string, cfg Config) PairedCell {
	c := PairedCell{Split: split, Task: task, N: len(items)}
	preds := make([]*Prediction, len(items))
	index := map[*Prediction]int{}
	for i, it := range items {
		preds[i] = it.a
		index[it.a] = i
		ra, rb := it.a.Correct(), it.b.Correct()
		switch {
		case ra && !rb:
			c.AOnlyRight++
		case !ra && rb:
			c.BOnlyRight++
		}
		if it.a.Pred != it.b.Pred {
			c.PredFlips++
		}
		if it.a.Acted != it.b.Acted {
			c.ActFlips++
		}
	}
	c.Stories = stories(preds)
	side := func(sel []*Prediction, useB bool) []*Prediction {
		out := make([]*Prediction, len(sel))
		for i, p := range sel {
			out[i] = p
			if useB {
				out[i] = items[index[p]].b
			}
		}
		return out
	}
	c.AccuracyA, _ = accuracyStat(side(preds, false))
	c.AccuracyB, _ = accuracyStat(side(preds, true))
	delta := func(stat func([]*Prediction) (float64, bool)) func([]*Prediction) (float64, bool) {
		return func(sel []*Prediction) (float64, bool) {
			x, ok1 := stat(side(sel, false))
			y, ok2 := stat(side(sel, true))
			return x - y, ok1 && ok2
		}
	}
	key := "paired/" + name + "/" + split + "/" + task
	c.DeltaAccuracy = estimate(preds, delta(accuracyStat), cfg, key+"/acc")
	c.DeltaMacroF1 = estimate(preds, delta(macroF1Stat), cfg, key+"/f1")
	c.P = McNemarExact(c.AOnlyRight, c.BOnlyRight)
	return c
}

// McNemarExact is the exact two-sided McNemar p-value for b and c
// discordant pairs: 2 · P(X <= min(b, c)) for X ~ Binomial(b+c, 1/2),
// capped at 1.
func McNemarExact(b, c int) float64 {
	n := b + c
	if n == 0 {
		return 1
	}
	return math.Min(1, 2*binomCDF(min(b, c), n, 0.5))
}

// Holm adjusts p-values for a family of tests (Holm-Bonferroni step-down),
// returned in the input order.
func Holm(ps []float64) []float64 {
	m := len(ps)
	order := make([]int, m)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool { return ps[order[i]] < ps[order[j]] })
	out := make([]float64, m)
	running := 0.0
	for rank, i := range order {
		v := math.Min(1, float64(m-rank)*ps[i])
		running = math.Max(running, v)
		out[i] = running
	}
	return out
}

// binomCDF is P(X <= k) for X ~ Binomial(n, p), summed in log space.
func binomCDF(k, n int, p float64) float64 {
	switch {
	case k < 0:
		return 0
	case k >= n:
		return 1
	case p <= 0:
		return 1
	case p >= 1:
		return 0
	}
	lgN, _ := math.Lgamma(float64(n + 1))
	lp, lq := math.Log(p), math.Log1p(-p)
	terms := make([]float64, k+1)
	m := math.Inf(-1)
	for i := 0; i <= k; i++ {
		a, _ := math.Lgamma(float64(i + 1))
		b, _ := math.Lgamma(float64(n - i + 1))
		terms[i] = lgN - a - b + float64(i)*lp + float64(n-i)*lq
		m = math.Max(m, terms[i])
	}
	var s float64
	for _, t := range terms {
		s += math.Exp(t - m)
	}
	return math.Min(1, math.Exp(m)*s)
}

// clopperPearsonUpper is the one-sided (1-delta) upper confidence bound on a
// binomial proportion after k events in n trials.
func clopperPearsonUpper(k, n int, delta float64) float64 {
	if n == 0 || k >= n {
		return 1
	}
	lo, hi := float64(k)/float64(n), 1.0
	for i := 0; i < 100; i++ {
		mid := (lo + hi) / 2
		if binomCDF(k, n, mid) > delta {
			lo = mid
		} else {
			hi = mid
		}
	}
	return hi
}

// FitBaselines fits a majority-class baseline per task from train labels:
// for each task, examples as (class key, label index, options mode).
func FitBaselines(train map[string][]LabelledExample) map[string]Baseline {
	out := map[string]Baseline{}
	for task, exs := range train {
		if len(exs) == 0 {
			continue
		}
		mode := exs[0].Options
		counts := map[string]int{}
		for _, e := range exs {
			k := e.Key
			if mode != "fixed" {
				k = fmt.Sprint(e.Index)
			}
			counts[k]++
		}
		best := ""
		for _, k := range sortedKeys(counts) {
			if best == "" || counts[k] > counts[best] {
				best = k
			}
		}
		b := Baseline{Options: mode, TrainN: len(exs), Share: float64(counts[best]) / float64(len(exs))}
		if mode == "fixed" {
			b.Key = best
			for _, e := range exs {
				if e.Key == best {
					b.Index = e.Index
					break
				}
			}
		} else {
			_, _ = fmt.Sscan(best, &b.Index)
		}
		out[task] = b
	}
	return out
}

// LabelledExample is one train example's label for FitBaselines.
type LabelledExample struct {
	Key     string
	Index   int
	Options string
}

// ModelNames lists the report's models.
func (r *Report) ModelNames() []string {
	out := make([]string, len(r.Models))
	for i, m := range r.Models {
		out[i] = m.Model
	}
	return out
}
