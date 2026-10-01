package evalreport

import (
	"fmt"
	"slices"
	"strings"
)

// RenderMarkdown renders the report's tables for the models in primary (all
// models when empty): accuracy and macro-F1 with the majority baseline,
// calibration, act thresholds, per-language accuracy, the paired
// comparisons and the confusion records.
func RenderMarkdown(r *Report, primary []string) string {
	if len(primary) == 0 {
		primary = r.ModelNames()
	}
	models := map[string]*ModelReport{}
	for _, m := range r.Models {
		models[m.Model] = m
	}
	var b strings.Builder
	w := func(format string, args ...any) { _, _ = fmt.Fprintf(&b, format, args...) }
	if len(r.Config.FocusLangs) > 0 {
		renderFocus(&b, r, primary, models)
	}
	for _, split := range r.Config.Splits {
		if len(r.Config.FocusLangs) > 0 {
			break // the focus tables and the secondary summary replace the all-language tables
		}
		tasks := splitTasks(r, primary, split)
		if len(tasks) == 0 {
			continue
		}
		w("### %s: accuracy and macro-F1\n\n", split)
		w("95 %% intervals resample whole stories (%d resamples).\n\n", r.Config.Boot)
		head := []string{"Task", "n (stories)", "Majority acc / F1"}
		for _, m := range primary {
			head = append(head, m+" acc", m+" macro-F1")
		}
		table(&b, head, func(add func(...string)) {
			for _, task := range append(tasks, "*") {
				var row []string
				if task == "*" {
					sr := models[primary[0]].Splits[split]
					row = []string{"all tasks", fmt.Sprintf("%d (%d)", sr.N, sr.Stories), "-"}
					for _, m := range primary {
						row = append(row, est(models[m].Splits[split].Accuracy), "-")
					}
					add(row...)
					continue
				}
				tr := models[primary[0]].Splits[split].Tasks[task]
				maj := "-"
				if c, ok := r.Majority[split][task]; ok {
					maj = fmt.Sprintf("%s / %s", val(c.Accuracy.Value), val(c.MacroF1.Value))
				}
				row = []string{task, fmt.Sprintf("%d (%d)", tr.N, tr.Stories), maj}
				for _, m := range primary {
					t := models[m].Splits[split].Tasks[task]
					row = append(row, est(t.Accuracy), est(t.MacroF1))
				}
				add(row...)
			}
		})
		w("\n### %s: calibration (ECE, 10 bins, T = 1 → fitted T)\n\n", split)
		head = []string{"Task"}
		for _, m := range primary {
			head = append(head, m+" top-1", m+" worst label", m+" pooled one-vs-rest")
		}
		table(&b, head, func(add func(...string)) {
			for _, task := range tasks {
				row := []string{task}
				for _, m := range primary {
					c := models[m].Splits[split].Tasks[task].Calibration
					row = append(row, fmt.Sprintf("%.3f → %.3f", c.Top1T1, c.Top1), worst(c), fmt.Sprintf("%.3f → %.3f", c.PooledT1, c.Pooled))
				}
				add(row...)
			}
		})
		w("\n### %s: act thresholds (Learn-then-Test on dev_conf, alpha %.2f)\n\n", split, r.Config.Alpha)
		head = []string{"Task"}
		for _, m := range primary {
			head = append(head, m+" threshold", m+" coverage", m+" risk (acted / errors)", m+" risk CP95 ≤ alpha")
		}
		table(&b, head, func(add func(...string)) {
			for _, task := range tasks {
				row := []string{task}
				for _, m := range primary {
					a := models[m].Splits[split].Tasks[task].Act
					th := "uncertified (escalate)"
					if a.Certified && a.Lambda != nil {
						th = fmt.Sprintf("%s ≥ %.4f", short(a.ScoreKind), *a.Lambda)
					}
					risk, bound := "-", "-"
					if a.Acted > 0 {
						risk = fmt.Sprintf("%s (%d / %d)", est(a.Risk), a.Acted, a.Errors)
						bound = fmt.Sprintf("%.3f %s", *a.RiskUpper95, yesNo(*a.WithinAlpha))
					}
					row = append(row, th, est(a.Coverage), risk, bound)
				}
				add(row...)
			}
		})
		w("\n### %s: accuracy by language (all tasks)\n\n", split)
		langs := map[string]bool{}
		for _, m := range primary {
			for l := range models[m].Splits[split].Languages {
				langs[l] = true
			}
		}
		head = []string{"Language", "n"}
		head = append(head, primary...)
		table(&b, head, func(add func(...string)) {
			for _, l := range sortedKeys(langs) {
				lr := models[primary[0]].Splits[split].Languages[l]
				row := []string{l, fmt.Sprint(lr.Accuracy.N)}
				for _, m := range primary {
					row = append(row, est(models[m].Splits[split].Languages[l].Accuracy))
				}
				add(row...)
			}
		})
		w("\n")
	}
	for _, p := range r.Paired {
		w("### Paired: %s minus %s\n\n", p.A, p.B)
		if len(r.Config.PairedLangs) > 0 {
			w("Items in %s only. ", strings.Join(r.Config.PairedLangs, ", "))
		}
		w("Δ with 95 %% story-paired bootstrap intervals; a difference counts as significant when that interval excludes 0. "+
			"McNemar exact p, Holm-adjusted within the per-task family (%d tests) and within the pooled family (%d tests), is shown "+
			"beside it: it treats the items of one story as independent pairs, so it is anti-conservative, and the interval decides "+
			"when the two disagree.\n\n", len(p.Holm["per_task"]), len(p.Holm["pooled_per_split"]))
		table(&b, []string{"Split", "Task", "n", "acc A", "acc B", "Δ acc", "Δ macro-F1", "A only / B only right", "p", "p (Holm)",
			"prediction differs", "act differs"}, func(add func(...string)) {
			for _, c := range p.Cells {
				task := c.Task
				if task == "*" {
					task = "all tasks"
				}
				add(c.Split, task, fmt.Sprint(c.N), fmt.Sprintf("%.3f", c.AccuracyA), fmt.Sprintf("%.3f", c.AccuracyB),
					est(c.DeltaAccuracy), est(c.DeltaMacroF1), fmt.Sprintf("%d / %d", c.AOnlyRight, c.BOnlyRight), pval(c.P),
					pval(c.PHolm), fmt.Sprint(c.PredFlips), fmt.Sprint(c.ActFlips))
			}
		})
		w("\n")
	}
	for _, m := range primary {
		for _, split := range r.Config.Splits {
			sr := models[m].Splits[split]
			if sr == nil {
				continue
			}
			for _, task := range r.Config.ConfusionTasks {
				tr := sr.Tasks[task]
				if tr == nil || tr.Confusion == nil {
					continue
				}
				c := tr.Confusion
				w("#### Confusion, %s, %s, %s (%d errors of %d)\n\n", m, split, task, c.Errors, tr.N)
				if c.Matrix != nil {
					table(&b, append([]string{"gold \\ predicted"}, c.Labels...), func(add func(...string)) {
						for i, row := range c.Matrix {
							cells := []string{c.Labels[i]}
							for _, v := range row {
								cells = append(cells, fmt.Sprint(v))
							}
							add(cells...)
						}
					})
				} else {
					table(&b, []string{"gold", "predicted", "count"}, func(add func(...string)) {
						for _, p := range c.Pairs {
							add(p.Gold, p.Pred, fmt.Sprint(p.Count))
						}
					})
				}
				w("\n")
			}
		}
	}
	return b.String()
}

func splitTasks(r *Report, models []string, split string) []string {
	tasks := map[string]bool{}
	for _, m := range r.Models {
		if !slices.Contains(models, m.Model) || m.Splits[split] == nil {
			continue
		}
		for t := range m.Splits[split].Tasks {
			tasks[t] = true
		}
	}
	return sortedKeys(tasks)
}

func table(b *strings.Builder, head []string, rows func(add func(...string))) {
	b.WriteString("| " + strings.Join(head, " | ") + " |\n")
	b.WriteString("|" + strings.Repeat("---|", len(head)) + "\n")
	rows(func(cells ...string) {
		for i, c := range cells {
			cells[i] = strings.ReplaceAll(c, "|", "\\|")
		}
		b.WriteString("| " + strings.Join(cells, " | ") + " |\n")
	})
}

func val(v *float64) string {
	if v == nil {
		return "-"
	}
	return fmt.Sprintf("%.3f", *v)
}

// est renders an estimate to three decimals with its interval.
func est(e Estimate) string {
	if e.Value == nil {
		return "-"
	}
	if e.Lo == nil || e.Hi == nil {
		return val(e.Value)
	}
	return fmt.Sprintf("%.3f [%.3f, %.3f]", *e.Value, *e.Lo, *e.Hi)
}

func worst(c Calibration) string {
	if c.WorstLabelT1 == nil || c.WorstLabel == nil {
		return fmt.Sprintf("n/a (no label offered ≥ min support; %d eligible)", c.EligibleLabels)
	}
	return fmt.Sprintf("%.3f (%s) → %.3f (%s)", c.WorstLabelT1.ECE, c.WorstLabelT1.Label, c.WorstLabel.ECE, c.WorstLabel.Label)
}

func short(kind string) string {
	switch kind {
	case "max_probability":
		return "max p"
	case "act_probability":
		return "P(act)"
	}
	return kind
}

func yesNo(v bool) string {
	if v {
		return "yes"
	}
	return "no"
}

func pval(p float64) string {
	if p < 1e-4 {
		return fmt.Sprintf("%.1e", p)
	}
	return fmt.Sprintf("%.4f", p)
}
