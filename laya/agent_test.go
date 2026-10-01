package laya

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"slices"
	"testing"

	"github.com/UnicoLab/ul_layago/model"
	"github.com/UnicoLab/ul_layago/tensor"
)

const parityDir = "../testdata/parity/"

type goldenQuestion struct {
	ID           string `json:"id"`
	Type         string `json:"type"`
	Instructions string `json:"instructions"`
	Options      []struct {
		Key         string `json:"key"`
		Description string `json:"description"`
	} `json:"options"`
	Levels []string `json:"levels"`
	False  string   `json:"false"`
	True   string   `json:"true"`
}

type goldenAnswer struct {
	ID             string    `json:"id"`
	IDs            []int32   `json:"ids"`
	Markers        []int     `json:"markers"`
	Logits         []float64 `json:"logits"`
	Temperature    float64   `json:"temperature"`
	Probs          []float64 `json:"probs"`
	ActProbability float64   `json:"act_probability"`
	Choice         *string   `json:"choice"`
	Score          *float64  `json:"score"`
	Noul           *float64  `json:"noul"`
	Confidence     float64   `json:"confidence"`
}

type layaGolden struct {
	Questions []goldenQuestion `json:"questions"`
	Cases     []struct {
		State       string         `json:"state"`
		Answers     []goldenAnswer `json:"answers"`
		AnswersInt8 []goldenAnswer `json:"answers_int8"`
	} `json:"cases"`
}

func loadLayaGolden(t *testing.T) (layaGolden, []Question) {
	t.Helper()
	raw, err := os.ReadFile(parityDir + "laya_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g layaGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	var qs []Question
	for _, gq := range g.Questions {
		q := Question{ID: gq.ID, Instructions: gq.Instructions, Levels: gq.Levels, False: gq.False, True: gq.True}
		switch gq.Type {
		case "choice":
			q.Type = Choice
		case "score":
			q.Type = Score
		case "noul":
			q.Type = Noul
		default:
			t.Fatalf("unknown question type %q", gq.Type)
		}
		for _, o := range gq.Options {
			q.Options = append(q.Options, Option{Key: o.Key, Description: o.Description})
		}
		qs = append(qs, q)
	}
	return g, qs
}

func compareAnswers(t *testing.T, label string, got *Prediction, want []goldenAnswer, logitTol, probTol float64) (worstLogit, worstProb float64) {
	t.Helper()
	for i, w := range want {
		a := got.Answers[i]
		if a.ID != w.ID {
			t.Fatalf("%s: answer %d is %q, want %q", label, i, a.ID, w.ID)
		}
		if !slices.Equal(got.Sequences[i], w.IDs) || !slices.Equal(got.Markers[i], w.Markers) {
			t.Fatalf("%s/%s: sequence or markers differ\n got %v %v\nwant %v %v", label, w.ID, got.Sequences[i], got.Markers[i], w.IDs, w.Markers)
		}
		if a.Temperature != w.Temperature {
			t.Fatalf("%s/%s: temperature %v, want %v", label, w.ID, a.Temperature, w.Temperature)
		}
		for j := range w.Logits {
			worstLogit = math.Max(worstLogit, math.Abs(float64(a.Logits[j])-w.Logits[j]))
			worstProb = math.Max(worstProb, math.Abs(a.Probs[j]-w.Probs[j]))
		}
		worstProb = math.Max(worstProb, math.Abs(a.ActProbability-w.ActProbability))
		worstProb = math.Max(worstProb, math.Abs(a.Confidence-w.Confidence))
		switch {
		case w.Choice != nil && a.Choice != *w.Choice:
			t.Errorf("%s/%s: choice %q, want %q", label, w.ID, a.Choice, *w.Choice)
		case w.Score != nil:
			worstProb = math.Max(worstProb, math.Abs(a.Score-*w.Score)/float64(len(w.Probs)))
		case w.Noul != nil:
			worstProb = math.Max(worstProb, math.Abs(a.Noul-*w.Noul))
		}
	}
	if worstLogit > logitTol || worstProb > probTol {
		t.Fatalf("%s: max |Δ logit| %.3g (tol %g), max |Δ p| %.3g (tol %g)", label, worstLogit, logitTol, worstProb, probTol)
	}
	return worstLogit, worstProb
}

// The goldens are the numpy reference, which crosscheck_torch.py matched
// against the laya 0.3.5 package itself (same sequences, logits within 6e-6).
func TestAgentMatchesNumpyReference(t *testing.T) {
	g, qs := loadLayaGolden(t)
	agent, err := LoadCheckpoint(parityDir+"checkpoint", LoadOptions{})
	if err != nil {
		t.Fatal(err)
	}
	for _, kern := range tensor.Kernels() {
		for _, workers := range []int{1, 4} {
			var worstLogit, worstProb float64
			for _, c := range g.Cases {
				pred, err := agent.Predict(context.Background(), c.State, qs, model.Options{Parallel: tensor.Parallel{Workers: workers, Kernel: kern}})
				if err != nil {
					t.Fatal(err)
				}
				l, p := compareAnswers(t, "fp32/"+kern.String(), pred, c.Answers, 1e-4, 1e-5)
				worstLogit, worstProb = math.Max(worstLogit, l), math.Max(worstProb, p)
			}
			t.Logf("fp32 %s workers=%d: max |Δ logit| %.3g, max |Δ p| %.3g over %d cases x %d questions",
				kern, workers, worstLogit, worstProb, len(g.Cases), len(qs))
		}
	}
}

func TestInt8AgentMatchesDequantizedReference(t *testing.T) {
	g, qs := loadLayaGolden(t)
	agent, err := LoadCheckpoint(parityDir+"checkpoint", LoadOptions{Int8: true})
	if err != nil {
		t.Fatal(err)
	}
	for _, kern := range tensor.Kernels() {
		var worstLogit, worstProb, quantProb float64
		for _, c := range g.Cases {
			pred, err := agent.Predict(context.Background(), c.State, qs, model.Options{Parallel: tensor.Parallel{Kernel: kern}})
			if err != nil {
				t.Fatal(err)
			}
			l, p := compareAnswers(t, "int8/"+kern.String(), pred, c.AnswersInt8, 1e-4, 1e-5)
			worstLogit, worstProb = math.Max(worstLogit, l), math.Max(worstProb, p)
			for i, w := range c.Answers {
				for j := range w.Probs {
					quantProb = math.Max(quantProb, math.Abs(pred.Answers[i].Probs[j]-w.Probs[j]))
				}
			}
		}
		t.Logf("int8 %s vs dequantized reference: max |Δ logit| %.3g, max |Δ p| %.3g; int8 vs fp32 reference max |Δ p| %.3g (random weights, reported not gated)",
			kern, worstLogit, worstProb, quantProb)
	}
}

func TestBuildSequenceQuirks(t *testing.T) {
	agent, err := LoadCheckpoint(parityDir+"checkpoint", LoadOptions{})
	if err != nil {
		t.Fatal(err)
	}
	q := Question{Type: Choice, Instructions: "pick", Options: []Option{{Key: "a"}, {Key: "b"}}}
	head, headMarkers, err := BuildSequence(agent.Tokenizer, agent.Special, "", q, 1000, 40, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	prefix := len(head) - 1 // everything before the state and its closing [SEP]
	state := "one two three four five six"
	ids, markers, err := BuildSequence(agent.Tokenizer, agent.Special, state, q, prefix+3, 40, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != prefix+3 || ids[len(ids)-1] != agent.Special.SEP || len(markers) != 2 {
		t.Fatalf("right truncation keeps two state tokens and the [SEP]: ids %v markers %v", ids, markers)
	}
	// With no room left, Python's st[-0:] keeps the whole state and the final
	// ids[:max_len] cut applies, so the closing [SEP] is lost.
	left, _, err := BuildSequence(agent.Tokenizer, agent.Special, state, q, prefix+1, 40, nil, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(left) != prefix+1 || left[len(left)-1] == agent.Special.SEP {
		t.Fatalf("truncate_left with no room: %v", left)
	}
	// A max_len that cuts through the options drops the markers beyond it.
	_, cut, err := BuildSequence(agent.Tokenizer, agent.Special, state, q, headMarkers[1], 40, nil, false)
	if err != nil || len(cut) != 1 {
		t.Fatalf("markers beyond max_len must be dropped: %v %v", cut, err)
	}
	if _, _, err := BuildSequence(agent.Tokenizer, agent.Special, "", q, 100, 40, []int{0, 5}, false); err == nil {
		t.Fatal("an out-of-range option order must be rejected")
	}
	rev, rmarkers, err := BuildSequence(agent.Tokenizer, agent.Special, "", q, 100, 40, []int{1, 0}, false)
	if err != nil || len(rmarkers) != 2 || slices.Equal(rev, head) {
		t.Fatalf("option order not applied: %v %v", rev, err)
	}
}

func TestTemperatureAndDecodingRules(t *testing.T) {
	for in, want := range map[float64]float64{0.1: 0.5, 7.5: 5, 1.7: 1.7, math.NaN(): 1, math.Inf(1): 1} {
		if got := ClampTemperature(in); got != want {
			t.Fatalf("ClampTemperature(%v) = %v, want %v", in, got, want)
		}
	}
	for k, want := range map[int]string{1: "choice:2", 2: "choice:2", 3: "choice:3-5", 5: "choice:3-5", 6: "choice:6-10", 11: "choice:11+"} {
		if got := TempBucket(Choice, k); got != want {
			t.Fatalf("TempBucket(%d) = %q, want %q", k, got, want)
		}
	}
	if c := entropyConfidence([]float64{1}); c != 1 {
		t.Fatalf("single-option confidence = %v", c)
	}
	if c := entropyConfidence([]float64{0.5, 0.5}); math.Abs(c) > 1e-12 {
		t.Fatalf("uniform confidence = %v", c)
	}
	f := ActFeatures([]float32{3})
	if f[0] != 1 || f[1] != 1 || f[3] != float32(2.0/255) {
		t.Fatalf("single-option act features = %v", f)
	}
}

func TestLoadCheckpointFailsClosed(t *testing.T) {
	if _, err := LoadCheckpoint(t.TempDir(), LoadOptions{}); err == nil {
		t.Fatal("an empty directory must not load")
	}
	agent, err := LoadCheckpoint(parityDir+"checkpoint", LoadOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agent.Predict(context.Background(), "x", nil, model.Options{}); err == nil {
		t.Fatal("no questions must be rejected")
	}
	if _, err := agent.Predict(context.Background(), "x", []Question{{ID: "empty", Type: Choice, Instructions: "?"}}, model.Options{}); err == nil {
		t.Fatal("a choice without options must be rejected")
	}
	many := Question{ID: "many", Type: Choice, Instructions: "?"}
	for i := 0; i < 20; i++ {
		many.Options = append(many.Options, Option{Key: "option"})
	}
	if _, err := agent.Predict(context.Background(), "x", []Question{many}, model.Options{}); err == nil {
		t.Fatal("options beyond head_max_len must be rejected, as Laya does")
	}
}
