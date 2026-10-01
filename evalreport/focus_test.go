package evalreport

import (
	"strings"
	"testing"
)

// TestFocusLanguagesAndRatio: the focus cells split each task by language
// with the majority baseline on the same items, the ratio is the second
// language's accuracy over the first's with an interval around it, paired
// comparisons can be restricted to the focus languages, and the Markdown
// leads with the focus tables and one secondary table.
func TestFocusLanguagesAndRatio(t *testing.T) {
	preds := fixture()
	unseen, unseenRight := 0, 0
	for _, m := range []string{"A", "B"} {
		for i, p := range preds[m] {
			if p.Split == "dev_conf" && p.Lang == "en" && i%3 == 0 {
				p.Slices = []string{"escalate_scope_absent", UnseenTemplate}
				if m == "A" {
					unseen++
					if p.Correct() {
						unseenRight++
					}
				}
			}
		}
	}
	cfg := DefaultConfig()
	cfg.Boot, cfg.MinLabelSupport = 300, 20
	cfg.FocusLangs, cfg.FocusSplits, cfg.RatioTarget, cfg.PairedLangs = []string{"en", "fr"}, []string{"dev_conf"}, 0.85, []string{"en"}
	baselines := FitBaselines(map[string][]LabelledExample{"q_operation": {{Key: "lookup", Index: 0, Options: "fixed"}}})
	r, err := Build(preds, baselines, [][2]string{{"A", "B"}}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	fs := r.Models[0].Focus["dev_conf"]
	if fs == nil || r.Models[0].Focus["heldout_domain"] != nil {
		t.Fatalf("focus splits %v", r.Models[0].Focus)
	}
	acc := func(lang string) (float64, float64) {
		n, c, maj := 0, 0, 0
		for _, p := range preds["A"] {
			if p.Split == "dev_conf" && p.Lang == lang {
				n++
				if p.Correct() {
					c++
				}
				if p.GoldKey() == "lookup" {
					maj++
				}
			}
		}
		return float64(c) / float64(n), float64(maj) / float64(n)
	}
	en, fr := fs.Cells["q_operation"]["en"], fs.Cells["q_operation"]["fr"]
	aEN, mEN := acc("en")
	aFR, mFR := acc("fr")
	if en.N != 30 || fr.N != 30 || !near(*en.Accuracy.Value, aEN) || !near(*fr.Accuracy.Value, aFR) || !near(*en.Majority, mEN) ||
		!near(*fr.Majority, mFR) || fs.Cells["*"]["en"].N != 30 || en.Act.N != 30 {
		t.Fatalf("cells en %+v fr %+v", en, fr)
	}
	if tn := fs.Templates["en"]; tn == nil || tn.Unseen.N != unseen || tn.Seen.N != 30-unseen ||
		!near(*tn.Unseen.Value, float64(unseenRight)/float64(unseen)) || fs.Templates["fr"] != nil {
		t.Fatalf("template novelty %+v (fr %+v), want %d unseen", fs.Templates["en"], fs.Templates["fr"], unseen)
	}
	ratio := fs.Ratio["q_operation"]
	if ratio == nil || ratio.Num != "fr" || ratio.Den != "en" || !near(*ratio.Ratio.Value, aFR/aEN) || ratio.Ratio.Lo == nil ||
		*ratio.Ratio.Lo > *ratio.Ratio.Value || *ratio.MeetsTarget != (aFR/aEN >= 0.85) {
		t.Fatalf("ratio %+v", ratio)
	}
	for _, c := range r.Paired[0].Cells {
		if c.Split == "dev_conf" && c.N != 30 {
			t.Fatalf("paired restricted to en has %d items", c.N)
		}
		if c.Split == "heldout_domain" {
			t.Fatalf("paired kept a de item: %+v", c)
		}
	}
	md := RenderMarkdown(r, []string{"A"})
	for _, want := range []string{"### dev_conf, A: accuracy by task and language (en, fr)", "fr/en", "#### dev_conf, A: calibration by language",
		"#### dev_conf, A: act thresholds as served, by language", "### Secondary: other languages", "| heldout_domain | de | 60 |",
		"#### dev_conf, A: template seen in train or not (all tasks)",
		"Items in en only."} {
		if !strings.Contains(md, want) {
			t.Errorf("markdown lacks %q", want)
		}
	}
	if strings.Contains(md, "### dev_conf: accuracy and macro-F1") {
		t.Error("the all-language tables were rendered beside the focus tables")
	}
}

// TestGoldOutsideTheOfferedOptions: a prediction whose gold a shortlist
// dropped (label -1 with its gold key) is scorable, always wrong, and pairs
// with a model that offered the gold.
func TestGoldOutsideTheOfferedOptions(t *testing.T) {
	preds := fixture()
	short := *preds["B"][0]
	short.Label, short.Gold, short.Keys = -1, short.Keys[short.Label], []string{"x", "y", "z"}
	preds["B"][0] = &short
	if short.Correct() || short.GoldKey() != preds["A"][0].GoldKey() {
		t.Fatalf("prediction %+v", short)
	}
	cfg := DefaultConfig()
	cfg.Boot = 50
	if _, err := Build(preds, nil, [][2]string{{"A", "B"}}, cfg); err != nil {
		t.Fatal(err)
	}
	bad := short
	bad.Gold = ""
	preds["B"][0] = &bad
	if _, err := Build(preds, nil, nil, cfg); err == nil {
		t.Fatal("label -1 without a gold key was accepted")
	}
}
