package train

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/UnicoLab/ul_layago/safetensors"
)

// TestTemperatureRecoversTheGeneratingTemperature draws labels from
// softmax(z/T*) and checks the NLL fit recovers T*, the per-cell tables, the
// fallback of a cell below the minimum and the minimum on "*" itself.
func TestTemperatureRecoversTheGeneratingTemperature(t *testing.T) {
	r := rand.New(rand.NewPCG(4, 4)) //nolint:gosec // G404: test data
	var items []tempRecord
	gen := func(task string, k, n int, tStar float64) {
		for i := 0; i < n; i++ {
			s := make([]float64, k)
			for j := range s {
				s[j] = r.NormFloat64() * 3
			}
			p := softmax64(s, tStar)
			u, y := r.Float64(), 0
			for acc := p[0]; u > acc && y < k-1; {
				y++
				acc += p[y]
			}
			items = append(items, tempRecord{task: task, k: k, scores: s, label: y})
		}
	}
	gen("sharp", 4, 6000, 0.7)
	gen("soft", 3, 6000, 2.2)
	gen("soft", 7, 30, 2.2) // below the minimum: falls back to the task
	tt := fitTemperatures(items, 50)
	for task, want := range map[string]float64{"sharp": 0.7, "soft": 2.2} {
		if got := tt[task].T; math.Abs(got-want) > 0.08*want {
			t.Errorf("task %s: T = %.3f, generated with %.3f", task, got, want)
		}
	}
	if got, key := tt.Lookup("sharp", 4); key != "sharp:3-5" || math.Abs(got-0.7) > 0.06 {
		t.Errorf("sharp:3-5 = %.3f from %s", got, key)
	}
	if _, key := tt.Lookup("soft", 7); key != "soft" {
		t.Errorf("soft:6-10 has 30 items (< 50) and must fall back to the task, got %s", key)
	}
	if _, key := tt.Lookup("unknown", 2); key != "*" {
		t.Errorf("unknown task must use the overall temperature, got %s", key)
	}
	for key, e := range tt {
		if e.NLL > e.NLLAtOne+1e-12 || e.T != math.Round(e.T*1e6)/1e6 {
			t.Errorf("%s: fitted NLL %.5f above T=1 NLL %.5f, or T %v not on the 1e-6 grid", key, e.NLL, e.NLLAtOne, e.T)
		}
	}
	// The overall cell needs the minimum too: a handful of items fits no
	// temperature at all, and the lookup returns 1.0 with an empty key.
	small := fitTemperatures(items[:10], 50)
	if len(small) != 0 {
		t.Fatalf("10 items fitted %v", small)
	}
	if v, key := small.Lookup("sharp", 4); v != 1 || key != "" {
		t.Fatalf("empty table: %v %q", v, key)
	}
}

// TestQuantizeProbabilityRoundsHalfUp: the contract's expression, floor(p ·
// 10000 + 0.5) / 10000, not a floor.
func TestQuantizeProbabilityRoundsHalfUp(t *testing.T) {
	for p, want := range map[float64]float64{0.12345: 0.1235, 0.12344: 0.1234, 0.99995: 1, 0.5: 0.5, 0: 0, 0.00004: 0} {
		if got := QuantizeProbability(p); got != want {
			t.Errorf("QuantizeProbability(%v) = %v, want %v", p, got, want)
		}
	}
}

// TestBinomialTailAndClopperPearson checks the exact tail against direct
// summation and the bound against known values.
func TestBinomialTailAndClopperPearson(t *testing.T) {
	direct := func(k, n int, p float64) float64 {
		var s float64
		for i := 0; i <= k; i++ {
			c := 1.0
			for j := 0; j < i; j++ {
				c = c * float64(n-j) / float64(j+1)
			}
			s += c * math.Pow(p, float64(i)) * math.Pow(1-p, float64(n-i))
		}
		return s
	}
	for _, c := range []struct {
		k, n int
		p    float64
	}{{0, 10, 0.05}, {3, 40, 0.1}, {7, 50, 0.2}, {12, 30, 0.5}} {
		if got, want := binomCDF(c.k, c.n, c.p), direct(c.k, c.n, c.p); math.Abs(got-want) > 1e-12 {
			t.Errorf("binomCDF(%d,%d,%g) = %.15f, want %.15f", c.k, c.n, c.p, got, want)
		}
	}
	for _, n := range []int{20, 150, 1000} {
		want := 1 - math.Pow(0.05, 1/float64(n))
		if got := clopperPearsonUpper(0, n, 0.05); math.Abs(got-want) > 1e-9 {
			t.Errorf("CP upper(0/%d) = %.9f, want %.9f", n, got, want)
		}
	}
}

// lttData is deterministic calibration data the Python trainer's
// ltt_threshold was run on (values below).
func lttData(n int) ([]float64, []bool) {
	s := make([]float64, n)
	c := make([]bool, n)
	for i := range s {
		s[i] = float64((i*37)%1000)/1000 + 0.00005*float64(i%3)
		c[i] = float64((i*7919)%97) < 97*(0.55+0.45*s[i])
	}
	return s, c
}

// TestLearnThenTestMatchesPythonTrainer: the fixed-sequence procedure
// certifies exactly what calibrate.py ltt_threshold certifies on the same
// data (lambda, acted, errors, thresholds tested, start).
func TestLearnThenTestMatchesPythonTrainer(t *testing.T) {
	for _, c := range []struct {
		n                               int
		alpha, delta                    float64
		nMin                            int
		certified                       bool
		lambda                          float64
		acted, errors, tested, nStarted int
	}{
		{400, 0.1, 0.1, 20, true, 0.6921, 125, 7, 1502, 66},
		{400, 0.05, 0.05, 20, false, 0, 0, 0, 1, 176},
		{150, 0.2, 0.2, 10, true, 0.2582, 108, 17, 5921, 22},
		{60, 0.1, 0.1, 20, false, 0, 0, 0, 0, 66},
	} {
		s, ok := lttData(c.n)
		th := LearnThenTest(s, ok, c.alpha, c.delta, c.nMin)
		lam := 0.0
		if th.Lambda != nil {
			lam = *th.Lambda
		}
		if th.Certified != c.certified || lam != c.lambda || th.Acted != c.acted || th.Errors != c.errors || th.Tested != c.tested ||
			th.NStart != c.nStarted {
			t.Errorf("n=%d alpha=%g: got certified=%v lambda=%v acted=%d errors=%d tested=%d start=%d, Python gives %+v",
				c.n, c.alpha, th.Certified, lam, th.Acted, th.Errors, th.Tested, th.NStart, c)
		}
	}
}

// TestLearnThenTestControlsSelectiveRisk: over many calibration draws from
// a known population, the certified threshold's true selective risk exceeds
// alpha in at most about delta of the draws.
func TestLearnThenTestControlsSelectiveRisk(t *testing.T) {
	// Confidence c ~ U(0,1); the item is wrong with probability 0.3·(1-c)²,
	// so the true risk of {c >= λ} is ∫λ^1 0.3(1-c)² dc / (1-λ) = 0.1(1-λ)².
	trueRisk := func(lam float64) float64 { return 0.1 * (1 - lam) * (1 - lam) }
	const alpha, delta = 0.05, 0.1
	r := rand.New(rand.NewPCG(11, 11)) //nolint:gosec // G404: test data
	violations, certified := 0, 0
	const trials = 200
	var coverage float64
	for trial := 0; trial < trials; trial++ {
		n := 2000
		scores := make([]float64, n)
		ok := make([]bool, n)
		for i := range scores {
			c := r.Float64()
			scores[i] = c
			ok[i] = r.Float64() >= 0.3*(1-c)*(1-c)
		}
		th := LearnThenTest(scores, ok, alpha, delta, 20)
		if !th.Certified {
			continue
		}
		certified++
		coverage += th.Coverage
		if trueRisk(*th.Lambda) > alpha {
			violations++
		}
		if math.Abs(*th.Lambda*1e4-math.Round(*th.Lambda*1e4)) > 1e-6 {
			t.Fatalf("lambda %v is off the 1e-4 grid", *th.Lambda)
		}
	}
	if certified < trials/2 {
		t.Fatalf("only %d of %d draws certified a threshold", certified, trials)
	}
	if rate := float64(violations) / float64(certified); rate > delta {
		t.Fatalf("true risk exceeded alpha in %.3f of certified draws, want <= %g", rate, delta)
	}
	t.Logf("%d/%d certified, violation rate %.3f, mean coverage %.3f", certified, trials, float64(violations)/float64(certified),
		coverage/float64(certified))
	if th := LearnThenTest([]float64{0.9, 0.8}, []bool{false, false}, 0.05, 0.1, 1); th.Certified || th.Lambda != nil {
		t.Fatalf("all-wrong calibration must never act, got %+v", th)
	}
}

// TestChooseThresholdsPaysForTheScoreChoice: delta is split between both
// score families, and the act head's score is read only above AUROC 0.80
// and above max_probability's.
func TestChooseThresholdsPaysForTheScoreChoice(t *testing.T) {
	s, ok := lttData(400)
	var rows []thresholdRow
	for i := range s {
		// max_probability tracks correctness weakly, act_probability well.
		act := 0.2
		if ok[i] {
			act = 0.9
		}
		rows = append(rows, thresholdRow{task: "q_operation", correct: ok[i], maxProb: s[i], actProb: act + 0.0001*float64(i%7)})
		rows = append(rows, thresholdRow{task: "q_time_supported", correct: ok[i], maxProb: s[i], actProb: 0.5})
	}
	ths := chooseThresholds(rows, 0.1, 0.1, 20, 0)
	op, ts := ths["q_operation"], ths["q_time_supported"]
	if op.Score != ScoreActProbability || ts.Score != ScoreMaxProbability {
		t.Fatalf("scores %s and %s (AUROC %v, %v)", op.Score, ts.Score, op.AUROC, ts.AUROC)
	}
	for _, th := range []Threshold{op, ts} {
		if th.Delta != 0.05 || th.DeltaFamily != 0.1 || th.Alpha != 0.1 || th.AUROC[ScoreMaxProbability] == nil {
			t.Fatalf("threshold %+v", th)
		}
	}
	// At delta/2 the max-probability threshold is at least as conservative
	// as the single-family one.
	single := LearnThenTest(s, ok, 0.1, 0.1, 20)
	if ts.Certified && single.Certified && *ts.Lambda < *single.Lambda {
		t.Fatalf("Bonferroni threshold %v below the single-family %v", *ts.Lambda, *single.Lambda)
	}
	if _, ok := ths["q_unknown"]; ok {
		t.Fatal("a task without dev_conf rows must get no threshold")
	}
}

func TestAUROCHandlesTies(t *testing.T) {
	if got := AUROC([]float64{0.9, 0.8, 0.8, 0.1}, []bool{true, true, false, false}); math.Abs(got-0.875) > 1e-12 {
		t.Fatalf("AUROC = %v, want 0.875", got)
	}
	if got := AUROC([]float64{0.5, 0.5}, []bool{true, true}); !math.IsNaN(got) {
		t.Fatalf("AUROC without negatives = %v, want NaN", got)
	}
}

func TestAdamWDecouplesDecayAndConverges(t *testing.T) {
	x := []float32{3, -2}
	params := []Param{{Name: "x", Shape: []int{2}, Data: x, Decay: false}}
	opt := NewAdamW(params, 0.5)
	for i := 0; i < 2000; i++ {
		g := []float64{2 * (float64(x[0]) - 1), 2 * (float64(x[1]) + 4)}
		opt.Step([][]float64{g}, 0.05)
	}
	if math.Abs(float64(x[0])-1) > 1e-3 || math.Abs(float64(x[1])+4) > 1e-3 {
		t.Fatalf("AdamW converged to %v, want [1 -4] (no decay on this parameter)", x)
	}
	y := []float32{1}
	decayed := NewAdamW([]Param{{Name: "y", Data: y, Decay: true}}, 0.1)
	decayed.Step([][]float64{{0}}, 0.5)
	if want := float32(1 - 0.5*0.1); y[0] != want {
		t.Fatalf("zero-gradient step with decay: %v, want %v (lr·wd·θ, decoupled)", y[0], want)
	}
	s := LRSchedule{Peak: 1, Floor: 0.1, Warmup: 10, Total: 110}
	if s.At(5) != 0.5 || s.At(10) != 1 || math.Abs(s.At(110)-0.1) > 1e-12 || math.Abs(s.At(60)-0.55) > 1e-12 {
		t.Fatalf("schedule: %v %v %v %v", s.At(5), s.At(10), s.At(60), s.At(110))
	}
}

// TestFloat16RoundTrip checks the cache's float16 encoder against the
// safetensors decoder: exact for representable values, nearest-even for
// ties, overflow reported.
func TestFloat16RoundTrip(t *testing.T) {
	for h := 0; h < 0x10000; h++ {
		f := safetensors.Float16ToFloat32(uint16(h))
		if math.IsNaN(float64(f)) {
			continue
		}
		back, ok := Float32ToFloat16(f)
		if !ok && !math.IsInf(float64(f), 0) {
			t.Fatalf("%04x: overflow on a representable value", h)
		}
		if back != uint16(h) {
			t.Fatalf("%04x -> %v -> %04x", h, f, back)
		}
	}
	r := rand.New(rand.NewPCG(8, 8)) //nolint:gosec // G404: test data
	for i := 0; i < 200000; i++ {
		f := float32(r.NormFloat64() * math.Pow(10, float64(r.IntN(10)-6)))
		h, ok := Float32ToFloat16(f)
		if !ok {
			continue
		}
		got := float64(safetensors.Float16ToFloat32(h))
		for _, nb := range []uint16{h - 1, h + 1} {
			alt := float64(safetensors.Float16ToFloat32(nb))
			if math.Abs(alt-float64(f)) < math.Abs(got-float64(f)) && (nb&0x7c00) != 0x7c00 {
				t.Fatalf("%v rounded to %v but %v is closer", f, got, alt)
			}
		}
	}
	if _, ok := Float32ToFloat16(70000); ok {
		t.Fatal("70000 must overflow float16")
	}
	if h, _ := Float32ToFloat16(2049); safetensors.Float16ToFloat32(h) != 2048 {
		t.Fatal("2049 must round to even (2048)")
	}
}

// TestMetricsOnHandCases checks summarize against hand-computed values,
// including an unscorable row (escalated, counted, not scored).
func TestMetricsOnHandCases(t *testing.T) {
	keys := []string{"a", "b", "c"}
	row := func(y int, probs []float64, acted bool) Row {
		r := Row{Label: y, GoldKey: keys[y], Scorable: true, Probs: probs, Acted: acted}
		r.Pred = argmax(probs)
		r.PredKey, r.TopProb, r.ProbLabel, r.Correct = keys[r.Pred], probs[r.Pred], probs[y], r.Pred == y
		return r
	}
	rows := []Row{
		row(0, []float64{0.8, 0.1, 0.1}, true),  // correct a
		row(1, []float64{0.7, 0.2, 0.1}, false), // gold b, pred a
		row(2, []float64{0.1, 0.1, 0.8}, true),  // correct c
		row(1, []float64{0.1, 0.6, 0.3}, false), // correct b
		{Label: 0, GoldKey: "a"},                // unscorable
	}
	g := summarize(rows)
	// F1: a = 2/(2+1+0) = 2/3, b = 2/(2+0+1) = 2/3, c = 1.
	if math.Abs(*g.MacroF1-(2.0/3+2.0/3+1)/3) > 1e-12 || *g.Accuracy != 0.75 || g.N != 5 || g.UnscorableRate != 0.2 {
		t.Fatalf("macro-F1 %v accuracy %v n %d unscorable %v", *g.MacroF1, *g.Accuracy, g.N, g.UnscorableRate)
	}
	if g.Coverage != 0.4 || *g.SelectiveAccuracy != 1 || math.Abs(g.AbstentionRate-0.6) > 1e-12 {
		t.Fatalf("coverage %v selective %v abstention %v", g.Coverage, *g.SelectiveAccuracy, g.AbstentionRate)
	}
	wantNLL := -(math.Log(0.8) + math.Log(0.2) + math.Log(0.8) + math.Log(0.6)) / 4
	if math.Abs(*g.NLL-wantNLL) > 1e-12 {
		t.Fatalf("NLL %v want %v", *g.NLL, wantNLL)
	}
	// ECE over 10 bins: (0.7, 0.8] holds 0.8 twice and 0.7 falls in (0.6,
	// 0.7] with 0.6 in (0.5, 0.6].
	wantECE := (2*0.2 + 0.7 + 0.4) / 4
	if math.Abs(*g.ECE-wantECE) > 1e-12 {
		t.Fatalf("ECE %v want %v", *g.ECE, wantECE)
	}
	if e := summarize(rows[4:]); e.Accuracy != nil || e.SelectiveAccuracy != nil || e.AbstentionRate != 1 {
		t.Fatalf("only unscorable: %+v", e)
	}
}
