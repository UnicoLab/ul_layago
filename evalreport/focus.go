package evalreport

import (
	"slices"
	"sort"
)

// FocusCell is one model's results on one task (or "*", every task) in
// one focus language of one split: accuracy and macro-F1 with story
// intervals, the majority baseline on the same items, calibration before
// and after temperature (top-1, worst label and pooled one-vs-rest), and
// the act thresholds as served on these items.
type FocusCell struct {
	N          int         `json:"n"`
	Stories    int         `json:"stories"`
	Unscorable int         `json:"unscorable"`
	Accuracy   Estimate    `json:"accuracy"`
	MacroF1    Estimate    `json:"macro_f1"`
	Majority   *float64    `json:"majority_accuracy,omitempty"`
	Calibrated Calibration `json:"calibration"`
	Act        Act         `json:"act"`
}

// FocusRatio compares a task's accuracy in two focus languages:
// Num's accuracy divided by Den's, with a 95 % interval that resamples
// each language's stories on its own (the two languages come from
// different stories). MeetsTarget compares the point with Target and
// LowerMeetsTarget the interval's lower bound.
type FocusRatio struct {
	Num              string   `json:"numerator"`
	Den              string   `json:"denominator"`
	Ratio            Estimate `json:"ratio"`
	Target           float64  `json:"target"`
	MeetsTarget      *bool    `json:"meets_target"`
	LowerMeetsTarget *bool    `json:"lower_bound_meets_target"`
}

// FocusSplit is one model's focus-language results on one split: cells by
// task and language, and the ratio of the second focus language to the
// first per task.
type FocusSplit struct {
	Cells map[string]map[string]*FocusCell `json:"cells"`
	Ratio map[string]*FocusRatio           `json:"ratio,omitempty"`
	// Templates splits each focus language's accuracy over every task by
	// whether the item's question template also occurs in train (the
	// dataset's unseen_template slice), which separates template recall
	// from generalisation.
	Templates map[string]*TemplateNovelty `json:"template_novelty,omitempty"`
}

// TemplateNovelty is accuracy on items whose question template occurs in
// train (Seen) and on items whose template does not (Unseen).
type TemplateNovelty struct {
	Seen   Estimate `json:"seen_template"`
	Unseen Estimate `json:"unseen_template"`
}

// UnseenTemplate is the dataset slice of items whose template is not in
// train.
const UnseenTemplate = "unseen_template"

// buildFocus computes the focus-language cells of one model.
func buildFocus(name string, preds []*Prediction, baselines map[string]Baseline, cfg Config) map[string]*FocusSplit {
	if len(cfg.FocusLangs) == 0 {
		return nil
	}
	out := map[string]*FocusSplit{}
	for _, split := range cfg.FocusSplits {
		items := inSplit(preds, split)
		if len(items) == 0 {
			continue
		}
		fs := &FocusSplit{Cells: map[string]map[string]*FocusCell{}, Ratio: map[string]*FocusRatio{}, Templates: map[string]*TemplateNovelty{}}
		byLangAll := groupBy(items, func(p *Prediction) string { return primaryLang(p.Lang) })
		for _, lang := range cfg.FocusLangs {
			var seen, unseen []*Prediction
			for _, p := range byLangAll[lang] {
				if slices.Contains(p.Slices, UnseenTemplate) {
					unseen = append(unseen, p)
				} else {
					seen = append(seen, p)
				}
			}
			if len(unseen) == 0 {
				continue
			}
			key := name + "/templates/" + split + "/" + lang
			fs.Templates[lang] = &TemplateNovelty{Seen: estimate(seen, accuracyStat, cfg, key+"/seen"),
				Unseen: estimate(unseen, accuracyStat, cfg, key+"/unseen")}
		}
		byTask := groupBy(items, func(p *Prediction) string { return p.Task })
		byTask["*"] = items
		for _, task := range sortedKeys(byTask) {
			byLang := groupBy(byTask[task], func(p *Prediction) string { return primaryLang(p.Lang) })
			fs.Cells[task] = map[string]*FocusCell{}
			for _, lang := range cfg.FocusLangs {
				li := byLang[lang]
				if len(li) == 0 {
					continue
				}
				key := name + "/focus/" + split + "/" + task + "/" + lang
				c := &FocusCell{N: len(li), Stories: stories(li), Accuracy: estimate(li, accuracyStat, cfg, key+"/acc"),
					MacroF1: estimate(li, macroF1Stat, cfg, key+"/f1"), Calibrated: calibration(li, cfg), Act: actReport(li, cfg, key)}
				for _, p := range li {
					if !p.Scorable {
						c.Unscorable++
					}
				}
				if task == "*" {
					// Class keys of different tasks do not form one label set.
					c.MacroF1 = Estimate{N: c.N, Stories: c.Stories}
				}
				if b, ok := baselines[task]; ok && task != "*" {
					if acc, ok := baselineAccuracy(b, li); ok {
						c.Majority = &acc
					}
				}
				fs.Cells[task][lang] = c
			}
			if len(cfg.FocusLangs) >= 2 {
				den, num := cfg.FocusLangs[0], cfg.FocusLangs[1]
				if r := focusRatio(byLang[num], byLang[den], cfg, name+"/ratio/"+split+"/"+task); r != nil {
					r.Num, r.Den = num, den
					fs.Ratio[task] = r
				}
			}
		}
		out[split] = fs
	}
	return out
}

// baselineAccuracy is a majority baseline's accuracy on the scorable items
// of a cell, the items the model's accuracy is computed on.
func baselineAccuracy(b Baseline, items []*Prediction) (float64, bool) {
	n, c := 0, 0
	for _, p := range items {
		if !p.Scorable {
			continue
		}
		n++
		if b.predict(p) == p.GoldKey() {
			c++
		}
	}
	return float64(c) / float64(max(1, n)), n > 0
}

// focusRatio is accuracy(num) / accuracy(den) with a stratified story
// bootstrap: each resample draws the numerator's stories and the
// denominator's stories separately.
func focusRatio(num, den []*Prediction, cfg Config, cell string) *FocusRatio {
	a, okA := accuracyStat(num)
	b, okB := accuracyStat(den)
	if !okA || !okB || b == 0 {
		return nil
	}
	r := &FocusRatio{Target: cfg.RatioTarget, Ratio: Estimate{N: len(num) + len(den), Stories: stories(num) + stories(den)}}
	r.Ratio.Value = ptr(a / b)
	cn, cd := clusters(num), clusters(den)
	rng := cellRNG(cfg.Seed, cell)
	vals := make([]float64, 0, cfg.Boot)
	var sn, sd []*Prediction
	for i := 0; i < cfg.Boot; i++ {
		sn, sd = sn[:0], sd[:0]
		for range cn {
			for _, j := range cn[rng.IntN(len(cn))] {
				sn = append(sn, num[j])
			}
		}
		for range cd {
			for _, j := range cd[rng.IntN(len(cd))] {
				sd = append(sd, den[j])
			}
		}
		x, ok1 := accuracyStat(sn)
		y, ok2 := accuracyStat(sd)
		if ok1 && ok2 && y > 0 {
			vals = append(vals, x/y)
		}
	}
	if len(vals) > 0 {
		sort.Float64s(vals)
		r.Ratio.Lo, r.Ratio.Hi = ptr(quantile(vals, 0.025)), ptr(quantile(vals, 0.975))
	}
	if r.Target > 0 {
		r.MeetsTarget = ptr(*r.Ratio.Value >= r.Target)
		if r.Ratio.Lo != nil {
			r.LowerMeetsTarget = ptr(*r.Ratio.Lo >= r.Target)
		}
	}
	return r
}

// primaryLang is a tag's primary subtag.
func primaryLang(tag string) string {
	for i := 0; i < len(tag); i++ {
		if tag[i] == '-' || tag[i] == '_' {
			return tag[:i]
		}
	}
	return tag
}
