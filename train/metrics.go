package train

import "math"

// Row is one evaluated item under a head's temperatures and thresholds
// (train.py decode_row): the calibrated probabilities, the prediction and
// the act decision. An unscorable item has no probabilities and always
// escalates.
type Row struct {
	ID       string
	Split    string
	Lang     string
	Task     string
	Domain   string
	Label    int
	GoldKey  string
	Scorable bool
	Correct  bool
	Acted    bool
	// Probs are softmax(logits / temperature) in float64.
	Probs          []float64
	Pred           int
	PredKey        string
	TopProb        float64
	ProbLabel      float64
	Temperature    float64
	TemperatureKey string
	ActProbability float64
	DecisionScore  float64
}

// decodeRow decodes one item's served logits with temps and thresholds, as
// the Go runtime (laya.Artifact.Predict) and the Python trainer do.
func decodeRow(it *Item, logits, act []float32, temps Temperatures, thresholds map[string]Threshold) Row {
	ex := it.Ex
	keys := ex.ClassKeys()
	r := Row{ID: ex.ID, Split: it.Role, Lang: ex.Lang, Task: ex.Task, Domain: ex.DomainID, Label: ex.Label, GoldKey: ex.GoldKey()}
	if !it.Scorable {
		return r
	}
	r.Scorable = true
	r.Temperature, r.TemperatureKey = temps.Lookup(ex.Task, ex.NumOptions())
	r.Probs = ServedProbabilities(logits, r.Temperature)
	r.ActProbability = ActProbability(act)
	for i, p := range r.Probs {
		if p > r.Probs[r.Pred] {
			r.Pred = i
		}
	}
	r.PredKey, r.TopProb = keys[r.Pred], r.Probs[r.Pred]
	if ex.HasGold() {
		r.ProbLabel = r.Probs[ex.Label]
	}
	r.Correct = ex.HasGold() && r.Pred == ex.Label
	r.DecisionScore = r.TopProb
	if th, ok := thresholds[ex.Task]; ok {
		if th.Score == ScoreActProbability {
			r.DecisionScore = r.ActProbability
		}
		r.Acted = th.Acts(r.DecisionScore)
	}
	return r
}

// ServedProbabilities is softmax(logits/T) in float64 from served float32
// logits, as laya's decodeLogits computes it.
func ServedProbabilities(logits []float32, t float64) []float64 {
	s := make([]float64, len(logits))
	for i, v := range logits {
		s[i] = float64(v)
	}
	return softmax64(s, t)
}

// ECEBins is the number of equal-width bins of the reported ECE.
const ECEBins = 10

// Summary is one evaluation cell (metrics.py summarize). Optional metrics
// are omitted when the cell has no scorable item; SelectiveAccuracy is null
// when nothing acted.
type Summary struct {
	N                 int      `json:"n"`
	UnscorableRate    float64  `json:"unscorable_rate"`
	Coverage          float64  `json:"coverage"`
	AbstentionRate    float64  `json:"abstention_rate"`
	SelectiveAccuracy *float64 `json:"selective_accuracy"`
	Accuracy          *float64 `json:"accuracy,omitempty"`
	MacroF1           *float64 `json:"macro_f1,omitempty"`
	ECE               *float64 `json:"ece,omitempty"`
	NLL               *float64 `json:"nll,omitempty"`
	Brier             *float64 `json:"brier,omitempty"`
}

func ptr(v float64) *float64 { return &v }

// summarize computes one cell's metrics:
//
//   - accuracy: argmax equals the label, over scorable items;
//   - macro_f1: unweighted mean of per-class F1 over the class keys that
//     occur as a label or a prediction in the cell;
//   - ece: 10 equal-width bins over the top probability (first bin closed),
//     |mean confidence - accuracy| weighted by bin mass. Every label is a
//     class label, so the top probability and correctness are one event;
//   - nll and brier of the calibrated distribution against the label;
//   - coverage: share of all items acted on; abstention_rate the rest;
//     unscorable_rate the share whose markers did not fit (always escalated
//     and counted in abstention_rate); selective_accuracy: accuracy among
//     acted items.
func summarize(rows []Row) Summary {
	n := len(rows)
	s := Summary{N: n}
	if n == 0 {
		return s
	}
	var scored []Row
	acted, actedCorrect := 0, 0
	for _, r := range rows {
		if r.Scorable {
			scored = append(scored, r)
		}
		if r.Acted {
			acted++
			if r.Correct {
				actedCorrect++
			}
		}
	}
	s.UnscorableRate = float64(n-len(scored)) / float64(n)
	s.Coverage = float64(acted) / float64(n)
	s.AbstentionRate = 1 - s.Coverage
	if acted > 0 {
		s.SelectiveAccuracy = ptr(float64(actedCorrect) / float64(acted))
	}
	if len(scored) == 0 {
		return s
	}
	var correct, nllSum, brier float64
	gold, pred := make([]string, len(scored)), make([]string, len(scored))
	for i, r := range scored {
		if r.Correct {
			correct++
		}
		nllSum -= math.Log(math.Max(r.ProbLabel, 1e-12))
		for j, p := range r.Probs {
			e := 0.0
			if j == r.Label {
				e = 1
			}
			brier += (p - e) * (p - e)
		}
		if r.Label < 0 {
			brier++ // the gold option was not offered: probability 0 on it
		}
		gold[i], pred[i] = r.GoldKey, r.PredKey
	}
	m := float64(len(scored))
	s.Accuracy = ptr(correct / m)
	s.MacroF1 = ptr(macroF1(gold, pred))
	s.ECE = ptr(ece(scored, ECEBins))
	s.NLL = ptr(nllSum / m)
	s.Brier = ptr(brier / m)
	return s
}

// macroF1 is the unweighted mean F1 over the classes that occur as gold or
// prediction.
func macroF1(gold, pred []string) float64 {
	classes := map[string]bool{}
	for i := range gold {
		classes[gold[i]], classes[pred[i]] = true, true
	}
	var sum float64
	for _, c := range sortedKeys(classes) {
		var tp, fp, fn int
		for i := range gold {
			switch {
			case gold[i] == c && pred[i] == c:
				tp++
			case gold[i] != c && pred[i] == c:
				fp++
			case gold[i] == c && pred[i] != c:
				fn++
			}
		}
		if d := 2*tp + fp + fn; d > 0 {
			sum += 2 * float64(tp) / float64(d)
		}
	}
	return sum / float64(len(classes))
}

// ece is the top-1 expected calibration error over equal-width bins, with
// numpy's linspace edges: the first bin is [0, 0.1], the others (lo, hi].
func ece(rows []Row, bins int) float64 {
	edges := make([]float64, bins+1)
	for i := range edges {
		edges[i] = float64(i) * (1 / float64(bins))
	}
	edges[bins] = 1
	var total float64
	n := float64(len(rows))
	for b := 0; b < bins; b++ {
		lo, hi := edges[b], edges[b+1]
		var cnt, conf, acc float64
		for _, r := range rows {
			in := r.TopProb <= hi && (r.TopProb > lo || (b == 0 && r.TopProb >= lo))
			if !in {
				continue
			}
			cnt++
			conf += r.TopProb
			if r.Correct {
				acc++
			}
		}
		if cnt > 0 {
			total += cnt / n * math.Abs(conf/cnt-acc/cnt)
		}
	}
	return total
}

// SplitEval is one split's evaluation: overall, by language, by task and by
// language × task ("<lang>/<task>").
type SplitEval struct {
	All        Summary            `json:"all"`
	ByLang     map[string]Summary `json:"by_lang"`
	ByTask     map[string]Summary `json:"by_task"`
	ByLangTask map[string]Summary `json:"by_lang_task"`
}

// evaluate summarises rows per split (metrics.py evaluate).
func evaluate(rows []Row) map[string]SplitEval {
	bySplit := map[string][]Row{}
	for _, r := range rows {
		bySplit[r.Split] = append(bySplit[r.Split], r)
	}
	out := map[string]SplitEval{}
	for split, rs := range bySplit {
		langs, tasks, lt := map[string][]Row{}, map[string][]Row{}, map[string][]Row{}
		for _, r := range rs {
			langs[r.Lang] = append(langs[r.Lang], r)
			tasks[r.Task] = append(tasks[r.Task], r)
			lt[r.Lang+"/"+r.Task] = append(lt[r.Lang+"/"+r.Task], r)
		}
		e := SplitEval{All: summarize(rs), ByLang: map[string]Summary{}, ByTask: map[string]Summary{}, ByLangTask: map[string]Summary{}}
		for k, v := range langs {
			e.ByLang[k] = summarize(v)
		}
		for k, v := range tasks {
			e.ByTask[k] = summarize(v)
		}
		for k, v := range lt {
			e.ByLangTask[k] = summarize(v)
		}
		out[split] = e
	}
	return out
}

// generalisation splits every evaluated split's rows into items whose
// domain (language) occurs in train and items whose domain (language) does
// not: new businesses and new languages. Languages compare by primary
// subtag, as the split invariants do.
func generalisation(rows []Row) map[string]map[string]Summary {
	trainDomains, trainLangs := map[string]bool{}, map[string]bool{}
	for _, r := range rows {
		if r.Split == SplitTrain {
			trainDomains[r.Domain], trainLangs[primaryLanguage(r.Lang)] = true, true
		}
	}
	groups := map[string]map[string][]Row{}
	for _, r := range rows {
		if r.Split == SplitTrain {
			continue
		}
		if groups[r.Split] == nil {
			groups[r.Split] = map[string][]Row{}
		}
		dom, lang := "seen_domain", "seen_language"
		if !trainDomains[r.Domain] {
			dom = "unseen_domain"
		}
		if !trainLangs[primaryLanguage(r.Lang)] {
			lang = "unseen_language"
		}
		groups[r.Split][dom] = append(groups[r.Split][dom], r)
		groups[r.Split][lang] = append(groups[r.Split][lang], r)
	}
	out := map[string]map[string]Summary{}
	for split, g := range groups {
		out[split] = map[string]Summary{}
		for k, v := range g {
			out[split][k] = summarize(v)
		}
	}
	return out
}

// argmax returns the first index of the largest value.
func argmax(p []float64) int {
	best := 0
	for i, v := range p {
		if v > p[best] {
			best = i
		}
	}
	return best
}
