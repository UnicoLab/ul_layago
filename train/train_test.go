package train

import (
	"math"
	"slices"
	"testing"

	"github.com/UnicoLab/ul_layago/laya"
)

func TestTaskSelectiveDistillationConfig(t *testing.T) {
	qaTasks := DefaultQADistillTasks()
	for _, want := range []string{
		"q_intent", "q_operation", "q_metric",
		"a_entailment", "a_scope_preserved", "a_limits_preserved", "a_no_new_claims",
	} {
		if !slices.Contains(qaTasks, want) {
			t.Fatalf("DefaultQADistillTasks missing %q", want)
		}
	}
	if slices.Contains(qaTasks, "q_scope") {
		t.Fatalf("DefaultQADistillTasks must exclude q_scope")
	}

	cfg := DefaultConfig()
	cfg.DistillWeight = 0.5
	cfg.DistillTemp = 2.0
	cfg.DistillTasks = DefaultQADistillTasks()

	if got := cfg.TaskDistillWeight("q_intent"); got != 0.5 {
		t.Fatalf("TaskDistillWeight(q_intent) = %v, want 0.5", got)
	}
	if got := cfg.TaskDistillWeight("q_scope"); got != 0 {
		t.Fatalf("TaskDistillWeight(q_scope) = %v, want 0", got)
	}

	cfgByTask := DefaultConfig()
	cfgByTask.DistillWeight = 0.5
	cfgByTask.DistillTemp = 2.0
	cfgByTask.DistillWeightByTask = map[string]float64{"q_metric": 0.25, "q_scope": 0}
	cfgByTask.DisableDistillTasks = []string{"q_operation"}
	if got := cfgByTask.TaskDistillWeight("q_metric"); got != 0.25 {
		t.Fatalf("TaskDistillWeight(q_metric) = %v, want 0.25", got)
	}
	if got := cfgByTask.TaskDistillWeight("q_scope"); got != 0 {
		t.Fatalf("TaskDistillWeight(q_scope) = %v, want 0", got)
	}
	if got := cfgByTask.TaskDistillWeight("q_operation"); got != 0 {
		t.Fatalf("TaskDistillWeight(q_operation) = %v, want 0", got)
	}
	if got := cfgByTask.TaskDistillWeight("q_intent"); got != 0.5 {
		t.Fatalf("TaskDistillWeight(q_intent) = %v, want 0.5", got)
	}
}

func TestForwardAndBackwardChoiceSpan(t *testing.T) {
	d := 4
	h, err := NewHead(HeadConfig{D: d, Hidden: d, ActHidden: 8, NAct: 2}, 7)
	if err != nil {
		t.Fatalf("NewHead: %v", err)
	}
	for i := range h.TypeEmb {
		h.TypeEmb[i] = float32(i+1) * 0.03
	}

	hidden := []float32{
		0.2, -0.4, 0.1, 0.5, // token 0
		0.6, 0.2, -0.3, 0.1, // token 1
		-0.2, 0.8, 0.5, -0.3, // token 2
		0.4, -0.1, 0.2, 0.7, // token 3
		-0.5, 0.3, -0.4, 0.2, // token 4
		0.1, 0.5, 0.6, -0.2, // token 5
	}

	singleSpans := [][2]int{{1, 2}, {4, 5}}
	markers := append(append([]float32(nil), hidden[1*d:2*d]...), hidden[4*d:5*d]...)
	wantSingle := h.Logits(markers, 2, laya.Choice)
	gotSingle := h.ForwardChoiceSpan(hidden, singleSpans, laya.Choice)
	for i := range wantSingle {
		if math.Float32bits(gotSingle[i]) != math.Float32bits(wantSingle[i]) {
			t.Fatalf("single-token span %d: got %v, want %v", i, gotSingle[i], wantSingle[i])
		}
	}

	multiSpans := [][2]int{{1, 3}, {3, 6}}
	gotMulti := h.ForwardChoiceSpan(hidden, multiSpans)
	if len(gotMulti) != 2 {
		t.Fatalf("len(gotMulti) = %d, want 2", len(gotMulti))
	}

	dLogits := []float32{0.75, -0.35}
	dHidden := make([]float32, len(hidden))
	h.BackwardChoiceSpan(hidden, multiSpans, dLogits, dHidden, laya.Choice)

	for i := 0; i < d; i++ {
		if dHidden[i] != 0 {
			t.Fatalf("token 0 dim %d outside spans has non-zero grad %v", i, dHidden[i])
		}
	}
	for i := 0; i < d; i++ {
		if math.Float32bits(dHidden[1*d+i]) != math.Float32bits(dHidden[2*d+i]) {
			t.Fatalf("span [1,3) tokens 1 and 2 dim %d grad mismatch: %v vs %v", i, dHidden[1*d+i], dHidden[2*d+i])
		}
	}
	for i := 0; i < d; i++ {
		if math.Float32bits(dHidden[3*d+i]) != math.Float32bits(dHidden[4*d+i]) ||
			math.Float32bits(dHidden[3*d+i]) != math.Float32bits(dHidden[5*d+i]) {
			t.Fatalf("span [3,6) tokens 3,4,5 dim %d grad mismatch", i)
		}
	}

	const eps = 1e-3
	for idx := range hidden {
		orig := hidden[idx]
		hidden[idx] = orig + eps
		sp := h.ScoresChoiceSpan(hidden, multiSpans, laya.Choice)
		hidden[idx] = orig - eps
		sm := h.ScoresChoiceSpan(hidden, multiSpans, laya.Choice)
		hidden[idx] = orig

		var numGrad float64
		for k := range multiSpans {
			numGrad += float64(dLogits[k]) * (sp[k] - sm[k]) / (2 * eps)
		}
		if math.Abs(float64(dHidden[idx])-numGrad) > 1e-4 {
			t.Fatalf("dHidden[%d] = %v, finite-diff = %v", idx, dHidden[idx], numGrad)
		}
	}
}
