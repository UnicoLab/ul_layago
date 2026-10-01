package evalreport

import (
	"fmt"
	"slices"
	"strings"
)

// renderFocus writes the primary tables (per task and focus language on
// each focus split: accuracy, macro-F1 and the majority baseline on the
// same items, the language ratio against its target, calibration and the
// act thresholds as served) and one secondary table of every other
// language's pooled accuracy.
func renderFocus(b *strings.Builder, r *Report, primary []string, models map[string]*ModelReport) {
	w := func(format string, args ...any) { _, _ = fmt.Fprintf(b, format, args...) }
	langs := r.Config.FocusLangs
	for _, split := range r.Config.FocusSplits {
		for _, m := range primary {
			fs := models[m].Focus[split]
			if fs == nil {
				continue
			}
			tasks := focusTasks(fs)
			w("### %s, %s: accuracy by task and language (%s)\n\n", split, m, strings.Join(langs, ", "))
			w("95 %% intervals resample whole stories (%d resamples). Majority: the train majority class on the same items.", r.Config.Boot)
			if len(langs) >= 2 && r.Config.RatioTarget > 0 {
				w(" Ratio: %s accuracy over %s accuracy, each language's stories resampled on their own; target ≥ %.2f.",
					langs[1], langs[0], r.Config.RatioTarget)
			}
			w("\n\n")
			head := []string{"Task"}
			for _, l := range langs {
				head = append(head, l+" n", l+" acc", l+" macro-F1", l+" majority")
			}
			if len(langs) >= 2 {
				head = append(head, langs[1]+"/"+langs[0], "≥ target")
			}
			table(b, head, func(add func(...string)) {
				for _, task := range tasks {
					row := []string{taskName(task)}
					for _, l := range langs {
						c := fs.Cells[task][l]
						if c == nil {
							row = append(row, "0", "-", "-", "-")
							continue
						}
						row = append(row, fmt.Sprint(c.N), est(c.Accuracy), est(c.MacroF1), val(c.Majority))
					}
					if len(langs) >= 2 {
						ratio, meets := "-", "-"
						if rt := fs.Ratio[task]; rt != nil {
							ratio = est(rt.Ratio)
							if rt.MeetsTarget != nil {
								meets = yesNo(*rt.MeetsTarget)
								if rt.LowerMeetsTarget != nil {
									meets += " (lower bound " + yesNo(*rt.LowerMeetsTarget) + ")"
								}
							}
						}
						row = append(row, ratio, meets)
					}
					add(row...)
				}
			})
			if len(fs.Templates) > 0 {
				w("\n#### %s, %s: template seen in train or not (all tasks)\n\n", split, m)
				table(b, []string{"Language", "seen template n", "seen template acc", "unseen template n", "unseen template acc"},
					func(add func(...string)) {
						for _, l := range langs {
							if tn := fs.Templates[l]; tn != nil {
								add(l, fmt.Sprint(tn.Seen.N), est(tn.Seen), fmt.Sprint(tn.Unseen.N), est(tn.Unseen))
							}
						}
					})
			}
			w("\n#### %s, %s: calibration by language (ECE, 10 bins, T = 1 → fitted T)\n\n", split, m)
			head = []string{"Task"}
			for _, l := range langs {
				head = append(head, l+" top-1", l+" worst label", l+" pooled one-vs-rest")
			}
			table(b, head, func(add func(...string)) {
				for _, task := range tasks {
					if task == "*" {
						continue
					}
					row := []string{task}
					for _, l := range langs {
						c := fs.Cells[task][l]
						if c == nil {
							row = append(row, "-", "-", "-")
							continue
						}
						k := c.Calibrated
						row = append(row, fmt.Sprintf("%.3f → %.3f", k.Top1T1, k.Top1), worst(k), fmt.Sprintf("%.3f → %.3f", k.PooledT1, k.Pooled))
					}
					add(row...)
				}
			})
			w("\n#### %s, %s: act thresholds as served, by language (alpha %.2f)\n\n", split, m, r.Config.Alpha)
			head = []string{"Task", "threshold"}
			for _, l := range langs {
				head = append(head, l+" coverage", l+" risk (acted / errors)", l+" risk CP95")
			}
			table(b, head, func(add func(...string)) {
				for _, task := range tasks {
					if task == "*" {
						continue
					}
					th := "-"
					for _, l := range langs {
						if c := fs.Cells[task][l]; c != nil {
							th = "uncertified (escalate)"
							if c.Act.Certified && c.Act.Lambda != nil {
								th = fmt.Sprintf("%s ≥ %.4f", short(c.Act.ScoreKind), *c.Act.Lambda)
							}
							break
						}
					}
					row := []string{task, th}
					for _, l := range langs {
						c := fs.Cells[task][l]
						if c == nil {
							row = append(row, "-", "-", "-")
							continue
						}
						risk, bound := "-", "-"
						if c.Act.Acted > 0 {
							risk = fmt.Sprintf("%s (%d / %d)", est(c.Act.Risk), c.Act.Acted, c.Act.Errors)
							bound = fmt.Sprintf("%.3f", *c.Act.RiskUpper95)
						}
						row = append(row, est(c.Act.Coverage), risk, bound)
					}
					add(row...)
				}
			})
			w("\n")
		}
	}
	w("### Secondary: other languages (accuracy over all tasks)\n\n")
	w("Not used for any choice, fit or threshold. Story-bootstrap 95 %% intervals.\n\n")
	head := []string{"Split", "Language", "n"}
	head = append(head, primary...)
	table(b, head, func(add func(...string)) {
		for _, split := range r.Config.Splits {
			ls := map[string]bool{}
			for _, m := range primary {
				if sr := models[m].Splits[split]; sr != nil {
					for l := range sr.Languages {
						if !slices.Contains(langs, primaryLang(l)) {
							ls[l] = true
						}
					}
				}
			}
			for _, l := range sortedKeys(ls) {
				n := "-"
				row := []string{split, l}
				var cells []string
				for _, m := range primary {
					sr := models[m].Splits[split]
					if sr == nil || sr.Languages[l] == nil {
						cells = append(cells, "-")
						continue
					}
					n = fmt.Sprint(sr.Languages[l].Accuracy.N)
					cells = append(cells, est(sr.Languages[l].Accuracy))
				}
				add(append(append(row, n), cells...)...)
			}
		}
	})
	w("\n")
}

// focusTasks lists a focus split's tasks with "*" (all tasks) last.
func focusTasks(fs *FocusSplit) []string {
	var tasks []string
	for t := range fs.Cells {
		if t != "*" {
			tasks = append(tasks, t)
		}
	}
	slices.Sort(tasks)
	if _, ok := fs.Cells["*"]; ok {
		tasks = append(tasks, "*")
	}
	return tasks
}

func taskName(t string) string {
	if t == "*" {
		return "all tasks"
	}
	return t
}
