package train

import (
	"context"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand/v2"
	"strconv"
	"strings"
)

// FloorVersion identifies the hashed-feature floor in predictions.
const FloorVersion = "laya-go-hashed-floor/1"

// FloorConfig controls the hashed-feature floor. Its values are fixed
// before any dev result: the floor is a baseline, never tuned.
type FloorConfig struct {
	// Bits is log2 of the hashed feature space.
	Bits int `json:"hash_bits"`
	// Epochs of Adagrad at learning rate LR over train, in a seeded order.
	Epochs int     `json:"epochs"`
	LR     float64 `json:"learning_rate"`
	Seed   uint64  `json:"seed"`
	// FitLangs and MinCell select the dev_cal items the temperatures are
	// fitted on and the dev_conf items the act thresholds are certified
	// on, as the head's; Alpha, Delta, NMin and LTTStartErrors are the
	// head's Learn-then-Test settings, on the calibrated top probability.
	FitLangs       []string `json:"fit_langs,omitempty"`
	MinCell        int      `json:"min_cell"`
	Alpha          float64  `json:"alpha"`
	Delta          float64  `json:"delta"`
	NMin           int      `json:"n_min"`
	LTTStartErrors int      `json:"ltt_start_errors"`
}

// DefaultFloorConfig is the floor laya-train floor runs.
func DefaultFloorConfig() FloorConfig {
	return FloorConfig{Bits: 20, Epochs: 5, LR: 0.1, Seed: 1, MinCell: 30, Alpha: 0.05, Delta: 0.1, NMin: 20}
}

// Floor is a trained hashed-feature floor: one linear scorer per option
// over hashed features of the question, the catalog and the option, with
// a softmax per question.
type Floor struct {
	Config       FloorConfig
	Weights      []float64
	Temperatures Temperatures
	Thresholds   map[string]Threshold
	// Logits are every item's option scores, by id.
	Logits map[string][]float32
	Rows   []Row
	// TrainItems counts the items fitted.
	TrainItems int
}

type floorFeature struct {
	idx uint32
	val float64
}

// floorFeatures are an item's features per option. Every feature is
// namespaced by task. A fixed-option task (q_operation, the yes_no tasks)
// crosses each option key with the question's folded words and bigrams and
// with bucketed catalog-match signals (how well the best metric and the
// best scope value match the question, how many scope values it names, the
// catalog size): a multinomial logistic regression. A task whose options
// vary per example (q_metric_choice, q_scope_choice) scores each option by
// its lexical match with the question (bucketed and raw), the option words
// the question contains and the option's position. The floor reads the
// example's input, never the trunk's cut sequence.
func floorFeatures(it *Item, fixed bool, mask uint32) [][]floorFeature {
	ex := it.Ex
	question, _ := ex.Input["question"].(string)
	q := newLexicon(question)
	words := foldWords(question)
	h := func(parts ...string) uint32 {
		f := fnv.New32a()
		_, _ = f.Write([]byte(ex.Task))
		for _, p := range parts {
			_, _ = f.Write([]byte{0})
			_, _ = f.Write([]byte(p))
		}
		return f.Sum32() & mask
	}
	bucket := func(v float64) string { return strconv.Itoa(int(math.Floor(v * 4))) }
	k := ex.NumOptions()
	out := make([][]floorFeature, k)
	keys := ex.ClassKeys()
	if ex.Type == TaskYesNo || fixed {
		catalog, _ := catalogList(ex.Input["catalog"])
		var bestMetric, bestScope float64
		named := 0
		for _, m := range catalog {
			obj, ok := m.(map[string]any)
			if !ok {
				continue
			}
			key, _ := obj["key"].(string)
			var labels any
			if v, ok := obj["labels"]; ok {
				labels = v
			} else {
				labels = obj["label"]
			}
			for _, t := range append([]string{strings.ReplaceAll(key, "_", " ")}, promptLabels(labels)...) {
				bestMetric = max(bestMetric, q.similarity(t))
			}
			scope, _ := asMap(obj["scope_values"])
			for _, d := range sortedKeys(scope) {
				vals, _ := asList(scope[d])
				for _, v := range vals {
					s, err := pyStr(v)
					if err != nil {
						continue
					}
					sim := q.similarity(s)
					bestScope = max(bestScope, sim)
					if sim >= 1.5 {
						named++
					}
				}
			}
		}
		signals := []string{"metric=" + bucket(bestMetric), "scope=" + bucket(bestScope), "named=" + strconv.Itoa(min(named, 4)),
			"catalog=" + strconv.Itoa(min(len(catalog), 12)), "qlen=" + strconv.Itoa(min(len(words)/4, 8))}
		for j := 0; j < k; j++ {
			f := []floorFeature{{h(keys[j], "bias"), 1}}
			for _, s := range signals {
				f = append(f, floorFeature{h(keys[j], s), 1})
			}
			for i, w := range words {
				f = append(f, floorFeature{h(keys[j], "w", w), 1})
				if i > 0 {
					f = append(f, floorFeature{h(keys[j], "b", words[i-1], w), 1})
				}
			}
			out[j] = f
		}
		return out
	}
	qwords := map[string]bool{}
	for _, w := range words {
		qwords[w] = true
	}
	for j, o := range ex.Options {
		text := o.Key
		if o.Text != "" {
			text = o.Text
		}
		sim := max(q.similarity(text), q.similarity(strings.NewReplacer("_", " ", "=", " ").Replace(o.Key)))
		f := []floorFeature{{h("sim", bucket(sim)), 1}, {h("simraw"), sim}, {h("pos", strconv.Itoa(min(j, 15))), 1}}
		for _, w := range foldWords(text + " " + strings.NewReplacer("_", " ", "=", " ").Replace(o.Key)) {
			if qwords[w] {
				f = append(f, floorFeature{h("in", w), 1})
			} else {
				f = append(f, floorFeature{h("out", w), 1})
			}
		}
		out[j] = f
	}
	return out
}

func floorScores(w []float64, feats [][]floorFeature) []float64 {
	s := make([]float64, len(feats))
	for j, f := range feats {
		for _, x := range f {
			s[j] += w[x.idx] * x.val
		}
	}
	return s
}

// FitFloor fits the hashed-feature floor on every train item whose gold
// is offered (softmax cross-entropy, Adagrad, fixed epochs, no early
// stopping), fits its temperatures on dev_cal items of the fit languages,
// certifies act thresholds on their dev_conf items by the head's
// Learn-then-Test rule and decodes every item. It never runs the trunk.
func FitFloor(ctx context.Context, p *Prepared, cfg FloorConfig) (*Floor, error) {
	if cfg.Bits < 8 || cfg.Bits > 26 || cfg.Epochs < 1 || !(cfg.LR > 0) || cfg.MinCell < 1 || cfg.NMin < 1 ||
		!(cfg.Alpha > 0 && cfg.Alpha < 1) || !(cfg.Delta > 0 && cfg.Delta < 1) || cfg.LTTStartErrors < 0 {
		return nil, fmt.Errorf("train: floor config %+v is out of range", cfg)
	}
	mask := uint32(1)<<cfg.Bits - 1
	fl := &Floor{Config: cfg, Weights: make([]float64, 1<<cfg.Bits), Logits: map[string][]float32{}}
	g2 := make([]float64, len(fl.Weights))
	fits := Config{FitLangs: cfg.FitLangs}
	feats := make([][][]floorFeature, len(p.Items))
	var train []int
	for i, it := range p.Items {
		feats[i] = floorFeatures(it, p.Tasks[it.Ex.Task] != nil && p.Tasks[it.Ex.Task].Options == "fixed", mask)
		if it.Role == SplitTrain && it.Ex.HasGold() {
			train = append(train, i)
		}
	}
	if len(train) == 0 {
		return nil, fmt.Errorf("train: the floor has no train item")
	}
	fl.TrainItems = len(train)
	rng := rand.New(rand.NewPCG(cfg.Seed, cfg.Seed^0x5851f42d4c957f2d)) //nolint:gosec // G404: reproducible order, not a security use
	for epoch := 0; epoch < cfg.Epochs; epoch++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		for _, n := range rng.Perm(len(train)) {
			i := train[n]
			it := p.Items[i]
			s := floorScores(fl.Weights, feats[i])
			pr := softmax64(s, 1)
			tgt := it.Ex.Target()
			for j, f := range feats[i] {
				g := pr[j] - tgt[j]
				if g == 0 {
					continue
				}
				for _, x := range f {
					gx := g * x.val
					g2[x.idx] += gx * gx
					fl.Weights[x.idx] -= cfg.LR * gx / math.Sqrt(g2[x.idx]+1e-8)
				}
			}
		}
	}
	var temps []tempRecord
	for i, it := range p.Items {
		s := floorScores(fl.Weights, feats[i])
		l := make([]float32, len(s))
		for j, v := range s {
			l[j] = float32(v)
			s[j] = float64(l[j])
		}
		fl.Logits[it.Ex.ID] = l
		if it.Role == RoleDevCal && fits.fits(it) && it.Ex.HasGold() {
			temps = append(temps, tempRecord{task: it.Ex.Task, k: it.Ex.NumOptions(), scores: s, label: it.Ex.Label})
		}
	}
	fl.Temperatures = fitTemperatures(temps, cfg.MinCell)
	// The act head's probability is a constant 0.5 here, so its AUROC is
	// 0.5 and the thresholds read the calibrated top probability, under
	// the head's rule (delta still split between both score families).
	noAct := []float32{0, 0}
	var conf []thresholdRow
	for _, it := range p.Items {
		if it.Role != RoleDevConf || !fits.fits(it) {
			continue
		}
		scorable := *it
		scorable.Scorable = true
		r := decodeRow(&scorable, fl.Logits[it.Ex.ID], noAct, fl.Temperatures, nil)
		conf = append(conf, thresholdRow{task: r.Task, correct: r.Correct, maxProb: r.TopProb, actProb: r.ActProbability})
	}
	fl.Thresholds = chooseThresholds(conf, cfg.Alpha, cfg.Delta, cfg.NMin, cfg.LTTStartErrors)
	for _, it := range p.Items {
		scorable := *it
		scorable.Scorable = true
		fl.Rows = append(fl.Rows, decodeRow(&scorable, fl.Logits[it.Ex.ID], noAct, fl.Temperatures, fl.Thresholds))
	}
	return fl, nil
}
