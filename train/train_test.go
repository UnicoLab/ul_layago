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

func TestHeadSwishActivationGradientsAndConvergence(t *testing.T) {
	// 1. Verify exact float64 finite-difference agreement of silu64 / siluGrad64 (< 1e-9).
	for _, x := range []float64{-3.5, -1.2784645, -0.75, -0.1, 0.0, 0.35, 1.2, 2.8} {
		const h64 = 1e-7
		fd := (silu64(x+h64) - silu64(x-h64)) / (2 * h64)
		ana := siluGrad64(x)
		if math.Abs(ana-fd) > 1e-9 {
			t.Fatalf("siluGrad64(%g) = %g, finite-diff = %g (diff=%g)", x, ana, fd, math.Abs(ana-fd))
		}
	}

	// 2. Verify "swish" and "silu" produce bit-for-bit identical scores and differ from default "gelu".
	d := 4
	hSwish, err := NewHead(HeadConfig{D: d, Hidden: 6, ActHidden: 8, NAct: 2, Activation: "swish"}, 19)
	if err != nil {
		t.Fatalf("NewHead(swish): %v", err)
	}
	hSiLU, err := NewHead(HeadConfig{D: d, Hidden: 6, ActHidden: 8, NAct: 2, Activation: "silu"}, 19)
	if err != nil {
		t.Fatalf("NewHead(silu): %v", err)
	}
	hGELU, err := NewHead(HeadConfig{D: d, Hidden: 6, ActHidden: 8, NAct: 2}, 19)
	if err != nil {
		t.Fatalf("NewHead(gelu): %v", err)
	}
	for i := range hSwish.TypeEmb {
		v := float32(i+1) * 0.04
		hSwish.TypeEmb[i] = v
		hSiLU.TypeEmb[i] = v
		hGELU.TypeEmb[i] = v
	}

	markers := []float32{
		0.4, -0.7, 0.2, 0.9,
		-0.5, 0.3, -0.8, 0.1,
		0.2, 0.6, -0.3, -0.4,
	}
	pooled := []float32{0.3, -0.2, 0.5, -0.6}

	sSwish := hSwish.Scores(markers, 3, laya.Choice)
	sSiLU := hSiLU.Scores(markers, 3, laya.Choice)
	sGELU := hGELU.Scores(markers, 3, laya.Choice)
	for i := range sSwish {
		if sSwish[i] != sSiLU[i] {
			t.Fatalf("Swish vs SiLU score[%d] mismatch: %g vs %g", i, sSwish[i], sSiLU[i])
		}
	}
	if sSwish[0] == sGELU[0] && sSwish[1] == sGELU[1] {
		t.Fatalf("expected Swish scores %v to differ from GELU scores %v", sSwish, sGELU)
	}

	// 3. Finite-difference gradient check (< 1e-6) for all scorer parameters under Activation: "swish".
	sc := newScorerScratch(hSwish.Config)
	gScorer := newGrads(hSwish.Config, false)
	du := make([]float64, d)
	scorerLoss := func(h *Head) float64 {
		scores := h.Scores(markers, 3, laya.Choice)
		probs := softmax64(scores, 1)
		return -math.Log(probs[0])
	}
	baseScores := hSwish.Scores(markers, 3, laya.Choice)
	baseProbs := softmax64(baseScores, 1)
	te := hSwish.TypeEmb[int(laya.Choice)*d : (int(laya.Choice)+1)*d]
	for k := 0; k < 3; k++ {
		_ = hSwish.score(markers[k*d:(k+1)*d], te, sc)
		ds := baseProbs[k]
		if k == 0 {
			ds -= 1.0
		}
		hSwish.backwardScore(ds, laya.Choice, sc, gScorer, du)
	}

	params := hSwish.Params()
	gradSlices := gScorer.slices()
	const step32 = float32(1.0 / 2048.0) // exact power-of-two step in float32
	for pIdx := 0; pIdx < scorerParams; pIdx++ {
		pData := params[pIdx].Data
		gData := gradSlices[pIdx]
		for i := range pData {
			orig := pData[i]
			pPlus := orig + step32
			pMinus := orig - step32
			actualStep := float64(pPlus) - float64(pMinus)
			pData[i] = pPlus
			lp := scorerLoss(hSwish)
			pData[i] = pMinus
			lm := scorerLoss(hSwish)
			pData[i] = orig

			fd := (lp - lm) / actualStep
			ana := gData[i]
			if math.Abs(ana-fd) > 1e-6 {
				t.Fatalf("scorer param %s[%d]: analytical=%g, finite-diff=%g (diff=%g)",
					params[pIdx].Name, i, ana, fd, math.Abs(ana-fd))
			}
		}
	}

	// 4. Finite-difference gradient check (< 1e-6) for all act head parameters under Activation: "swish".
	servedLogits := hSwish.Logits(markers, 3, laya.Choice)
	actSc := newActScratch(hSwish.Config)
	gAct := newGrads(hSwish.Config, true)
	_ = hSwish.actExampleGrad(pooled, laya.Choice, servedLogits, 0, 1.25, actSc, gAct)

	actLoss := func(h *Head) float64 {
		o := h.actForward(pooled, laya.Choice, servedLogits, newActScratch(h.Config))
		p := softmax64(o, 1)
		return -1.25 * math.Log(p[0])
	}
	actGradSlices := gAct.slices()
	for pIdx := scorerParams; pIdx < len(params); pIdx++ {
		pData := params[pIdx].Data
		gData := actGradSlices[pIdx]
		for i := range pData {
			orig := pData[i]
			pPlus := orig + step32
			pMinus := orig - step32
			actualStep := float64(pPlus) - float64(pMinus)
			pData[i] = pPlus
			lp := actLoss(hSwish)
			pData[i] = pMinus
			lm := actLoss(hSwish)
			pData[i] = orig

			fd := (lp - lm) / actualStep
			ana := gData[i]
			if math.Abs(ana-fd) > 1e-6 {
				t.Fatalf("act param %s[%d]: analytical=%g, finite-diff=%g (diff=%g)",
					params[pIdx].Name, i, ana, fd, math.Abs(ana-fd))
			}
		}
	}

	// 5. Training convergence of Head with Activation: "swish" on both scorer and act stages.
	initLoss := scorerLoss(hSwish)
	const lr = 0.08
	for step := 0; step < 60; step++ {
		gScorer.zero()
		scores := hSwish.Scores(markers, 3, laya.Choice)
		probs := softmax64(scores, 1)
		teSlice := hSwish.TypeEmb[int(laya.Choice)*d : (int(laya.Choice)+1)*d]
		for k := 0; k < 3; k++ {
			_ = hSwish.score(markers[k*d:(k+1)*d], teSlice, sc)
			ds := probs[k]
			if k == 0 {
				ds -= 1.0
			}
			hSwish.backwardScore(ds, laya.Choice, sc, gScorer, du)
		}
		for pIdx := 0; pIdx < scorerParams; pIdx++ {
			pData := params[pIdx].Data
			gData := gradSlices[pIdx]
			for i := range pData {
				pData[i] -= float32(lr * gData[i])
			}
		}
	}
	finalLoss := scorerLoss(hSwish)
	if !(finalLoss < initLoss*0.05 && finalLoss < 0.02) {
		t.Fatalf("expected Swish scorer training loss to converge: init=%g, final=%g", initLoss, finalLoss)
	}

	// Act stage convergence.
	updatedLogits := hSwish.Logits(markers, 3, laya.Choice)
	initActLoss := actLoss(hSwish)
	for step := 0; step < 40; step++ {
		gAct.zero()
		_ = hSwish.actExampleGrad(pooled, laya.Choice, updatedLogits, 0, 1.25, actSc, gAct)
		for pIdx := scorerParams; pIdx < len(params); pIdx++ {
			pData := params[pIdx].Data
			gData := actGradSlices[pIdx]
			for i := range pData {
				pData[i] -= float32(lr * gData[i])
			}
		}
	}
	finalActLoss := actLoss(hSwish)
	if !(finalActLoss < initActLoss*0.1 && ActProbability(hSwish.ActLogits(pooled, laya.Choice, updatedLogits)) > 0.95) {
		t.Fatalf("expected Swish act head to converge: init=%g, final=%g", initActLoss, finalActLoss)
	}
}
