package evalreport

import (
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"
)

func near(a, b float64) bool { return math.Abs(a-b) < 1e-12 }

func TestMcNemarExactAndHolm(t *testing.T) {
	for _, c := range []struct {
		b, c int
		p    float64
	}{
		{0, 0, 1},
		{10, 0, 2 * math.Pow(0.5, 10)},
		{5, 3, 2 * 93.0 / 256},
		{3, 5, 2 * 93.0 / 256},
		{4, 4, 1},
	} {
		if got := McNemarExact(c.b, c.c); !near(got, c.p) {
			t.Errorf("McNemar(%d, %d) = %v, want %v", c.b, c.c, got, c.p)
		}
	}
	got := Holm([]float64{0.01, 0.04, 0.03})
	if want := []float64{0.03, 0.06, 0.06}; !near(got[0], want[0]) || !near(got[1], want[1]) || !near(got[2], want[2]) {
		t.Fatalf("Holm %v, want %v", got, want)
	}
	if got := Holm([]float64{0.5, 0.9}); !near(got[0], 1) || !near(got[1], 1) {
		t.Fatalf("Holm caps at 1: %v", got)
	}
}

// TestECEMatchesTheTrainersBins: the trainer's edges put 0.3 in (0.2, 0.3]
// (its edge is 3·0.1 = 0.30000000000000004), 0 and 0.1 in the first bin.
func TestECEMatchesTheTrainersBins(t *testing.T) {
	if e := ece([]float64{0.8, 0.8, 0.8, 0.8, 0.8}, []bool{true, true, true, true, false}, 10); !near(e, 0) {
		t.Fatalf("calibrated bin: %v", e)
	}
	// 0.3 and 0.25 share a bin: mean confidence 0.275, accuracy 0.5.
	if e := ece([]float64{0.3, 0.25}, []bool{true, false}, 10); !near(e, 0.225) {
		t.Fatalf("edge 0.3: %v", e)
	}
	if e := ece([]float64{0, 0.1}, []bool{false, false}, 10); !near(e, 0.05) {
		t.Fatalf("first bin: %v", e)
	}
}

func TestMacroF1(t *testing.T) {
	// a: tp 1, fn 1 -> 2/3; b: tp 1, fp 1 -> 2/3; c: fp... gold c pred a.
	gold := []string{"a", "a", "b", "c"}
	pred := []string{"a", "b", "b", "a"}
	// a: tp1 fp1 fn1 -> 0.5; b: tp1 fp1 -> 2/3; c: fn1 -> 0.
	if got, want := macroF1(gold, pred), (0.5+2.0/3+0)/3; !near(got, want) {
		t.Fatalf("macroF1 %v, want %v", got, want)
	}
}

// fixture builds predictions for two models over stories of a fixed-option
// task and a yes/no task.
func fixture() map[string][]*Prediction {
	out := map[string][]*Prediction{}
	lambda := 0.9
	for m, model := range []string{"A", "B"} {
		for s := 0; s < 12; s++ {
			for i := 0; i < 5; i++ {
				label := (s + i) % 3
				pred := label
				if (s*5+i)%(4+m) == 0 {
					pred = (label + 1) % 3
				}
				probs := []float64{0.1, 0.1, 0.1}
				probs[pred] = 0.8
				t1 := []float64{0.2, 0.2, 0.2}
				t1[pred] = 0.6
				out[model] = append(out[model], &Prediction{Model: model, ID: fmt.Sprintf("s%d-%d-op", s, i), Split: "dev_conf",
					Story: fmt.Sprintf("s%d", s), Lang: []string{"en", "fr"}[s%2], Task: "q_operation", Options: "fixed",
					Keys: []string{"lookup", "change", "rank"}, Label: label, Scorable: true, Probs: probs, ProbsT1: t1,
					Temperature: 1.5, TemperatureKey: "q_operation", Pred: pred, ScoreKind: "max_probability", Lambda: &lambda,
					Certified: true, DecisionScore: 0.8, Acted: i%2 == 0})
				yl := i % 2
				yp := yl
				if (s+i+m)%3 == 0 {
					yp = 1 - yl
				}
				pp := []float64{0.3, 0.7}
				if yp == 0 {
					pp = []float64{0.7, 0.3}
				}
				out[model] = append(out[model], &Prediction{Model: model, ID: fmt.Sprintf("s%d-%d-yn", s, i), Split: "heldout_domain",
					Story: fmt.Sprintf("h%d", s), Lang: "de", Task: "q_single_series", Options: "fixed", Keys: []string{"false", "true"},
					Label: yl, Scorable: true, Probs: pp, ProbsT1: pp, Temperature: 1, TemperatureKey: "*", Pred: yp,
					ScoreKind: "max_probability", Certified: false})
			}
		}
	}
	return out
}

func TestBuildIsDeterministicAndConsistent(t *testing.T) {
	preds := fixture()
	cfg := DefaultConfig()
	cfg.Boot, cfg.MinLabelSupport = 300, 20
	baselines := FitBaselines(map[string][]LabelledExample{
		"q_operation":     {{Key: "lookup", Index: 0, Options: "fixed"}, {Key: "lookup", Index: 0, Options: "fixed"}, {Key: "rank", Index: 2, Options: "fixed"}},
		"q_single_series": {{Key: "true", Index: 1, Options: "fixed"}},
	})
	if b := baselines["q_operation"]; b.Key != "lookup" || b.Index != 0 || b.TrainN != 3 || !near(b.Share, 2.0/3) {
		t.Fatalf("baseline %+v", b)
	}
	r1, err := Build(preds, baselines, [][2]string{{"A", "B"}}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := Build(preds, baselines, [][2]string{{"A", "B"}}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	j1, _ := json.Marshal(r1)
	j2, _ := json.Marshal(r2)
	if string(j1) != string(j2) {
		t.Fatal("the report is not deterministic")
	}
	if !reflect.DeepEqual(r1.ModelNames(), []string{"A", "B"}) {
		t.Fatalf("models %v", r1.ModelNames())
	}
	op := r1.Models[0].Splits["dev_conf"].Tasks["q_operation"]
	// A errs where (5s+i) % 4 == 0: 15 of 60.
	if op.N != 60 || op.Stories != 12 || !near(*op.Accuracy.Value, 45.0/60) {
		t.Fatalf("accuracy %+v", op.Accuracy)
	}
	if *op.Accuracy.Lo > *op.Accuracy.Value || *op.Accuracy.Hi < *op.Accuracy.Value || *op.Accuracy.Lo >= *op.Accuracy.Hi {
		t.Fatalf("interval %v [%v, %v]", *op.Accuracy.Value, *op.Accuracy.Lo, *op.Accuracy.Hi)
	}
	c := op.Calibration
	// Top-1 confidence 0.8 at T, 0.6 at T=1, accuracy 0.75.
	if !near(c.Top1, 0.05) || !near(c.Top1T1, 0.15) || c.EligibleLabels != 3 || c.WorstLabel == nil || c.Temperatures["q_operation"] != 1.5 {
		t.Fatalf("calibration %+v", c)
	}
	if op.Act.N != 60 || op.Act.Acted != 36 || !op.Act.Certified || op.Act.RiskUpper95 == nil || op.Act.WithinAlpha == nil {
		t.Fatalf("act %+v", op.Act)
	}
	if op.Confusion == nil || len(op.Confusion.Matrix) != 3 || op.Confusion.Errors != 15 {
		t.Fatalf("confusion %+v", op.Confusion)
	}
	var diag int
	for i := range op.Confusion.Matrix {
		diag += op.Confusion.Matrix[i][i]
	}
	if diag != 45 {
		t.Fatalf("diagonal %d", diag)
	}
	if lang := r1.Models[0].Splits["dev_conf"].Languages["en"]; lang == nil || lang.Tasks["q_operation"].N != 30 {
		t.Fatalf("languages %+v", r1.Models[0].Splits["dev_conf"].Languages)
	}
	maj := r1.Majority["dev_conf"]["q_operation"]
	if !near(*maj.Accuracy.Value, 20.0/60) {
		t.Fatalf("majority %+v", maj.Accuracy)
	}
	p := r1.Paired[0]
	var cell *PairedCell
	for i := range p.Cells {
		if p.Cells[i].Split == "dev_conf" && p.Cells[i].Task == "q_operation" {
			cell = &p.Cells[i]
		}
	}
	if cell == nil || cell.N != 60 || !near(cell.AccuracyA-cell.AccuracyB, *cell.DeltaAccuracy.Value) ||
		!near(cell.P, McNemarExact(cell.AOnlyRight, cell.BOnlyRight)) || cell.PHolm < cell.P {
		t.Fatalf("paired %+v", cell)
	}
	if len(p.Holm["per_task"]) != 2 || len(p.Holm["pooled_per_split"]) != 2 {
		t.Fatalf("families %v", p.Holm)
	}

	bad := fixture()
	bad["A"][0].Pred = 7
	if _, err := Build(bad, nil, nil, cfg); err == nil {
		t.Fatal("an out-of-range prediction was accepted")
	}
	if _, err := Build(preds, nil, [][2]string{{"A", "C"}}, cfg); err == nil {
		t.Fatal("a missing model was accepted")
	}
}

func TestPerExampleConfusionListsTopPairs(t *testing.T) {
	var preds []*Prediction
	for i := 0; i < 10; i++ {
		keys := []string{"orders", "revenue", fmt.Sprintf("m%d", i)}
		pred := 0
		if i < 4 {
			pred = 1
		}
		preds = append(preds, &Prediction{ID: fmt.Sprint(i), Story: fmt.Sprint(i), Task: "q_metric_choice", Options: "per_example",
			Keys: keys, Label: 0, Pred: pred, Scorable: true, Probs: []float64{0.5, 0.3, 0.2}, ProbsT1: []float64{0.5, 0.3, 0.2}})
	}
	c := confusion(preds, 5)
	if c.Errors != 4 || len(c.Pairs) != 1 || c.Pairs[0] != (ConfusionPair{Gold: "orders", Pred: "revenue", Count: 4}) ||
		!reflect.DeepEqual(c.ByGold["orders"], []int{10, 4}) || c.Matrix != nil {
		t.Fatalf("confusion %+v", c)
	}
}

func TestRenderMarkdownHasEverySection(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Boot, cfg.MinLabelSupport = 50, 20
	r, err := Build(fixture(), FitBaselines(map[string][]LabelledExample{"q_operation": {{Key: "lookup", Options: "fixed"}}}),
		[][2]string{{"A", "B"}}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	md := RenderMarkdown(r, []string{"A", "B"})
	for _, want := range []string{"### dev_conf: accuracy and macro-F1", "### heldout_domain: calibration", "### dev_conf: act thresholds",
		"### heldout_domain: accuracy by language", "### Paired: A minus B", "#### Confusion, A, dev_conf, q_operation (15 errors of 60)",
		"| q_operation | 60 (12) | 0.333 / ", "max p ≥ 0.9000", "uncertified (escalate)"} {
		if !strings.Contains(md, want) {
			t.Errorf("markdown lacks %q", want)
		}
	}
}
