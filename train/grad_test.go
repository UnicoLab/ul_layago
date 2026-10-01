package train

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// gradHead is a small head with every parameter perturbed away from its
// initial value, so no gradient is trivially zero.
func gradHead(t *testing.T, seed uint64) *Head {
	t.Helper()
	h, err := NewHead(HeadConfig{D: 8, Hidden: 6, ActHidden: 5, NAct: 2}, seed)
	if err != nil {
		t.Fatal(err)
	}
	r := rand.New(rand.NewPCG(seed, 99)) //nolint:gosec // G404: test data
	for _, p := range h.Params() {
		for i := range p.Data {
			p.Data[i] += float32(r.NormFloat64() * 0.3)
		}
	}
	return h
}

func randFloats(r *rand.Rand, n int, std float64) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(r.NormFloat64() * std)
	}
	return out
}

type gradCase struct {
	name string
	ex   Example
	reg  float64
}

func gradCases() []gradCase {
	opts := func(k int) []Option {
		o := make([]Option, k)
		for i := range o {
			o[i] = Option{Key: fmt.Sprintf("o%d", i), Text: fmt.Sprintf("t%d", i)}
		}
		return o
	}
	return []gradCase{
		{"choice", Example{Type: TaskChoice, Task: "q_c", Options: opts(4), Label: 2}, 0},
		{"choice_listwise_label_dist", Example{Type: TaskChoice, Task: "r_rank", Options: opts(3), Label: 0,
			LabelDist: []float64{2.0 / 3, 1.0 / 3, 0}}, 0},
		{"yes_no_true", Example{Type: TaskYesNo, Task: "q_y", Label: 1}, 0},
		{"yes_no_false_with_texts", Example{Type: TaskYesNo, Task: "q_y", Options: opts(2), Label: 0}, 0},
		{"yes_no_soft", Example{Type: TaskYesNo, Task: "q_y", Label: 1, LabelDist: []float64{0.3, 0.7}}, 0},
		{"score_level_with_regression", Example{Type: TaskScore, Task: "r_s", Options: opts(3), Label: 2}, 0.5},
		{"score_soft_with_regression", Example{Type: TaskScore, Task: "r_s", Options: opts(4), Label: 1,
			LabelDist: []float64{0.2, 0.4, 0.3, 0.1}}, 0.5},
	}
}

// TestObjectiveGradientsMatchFiniteDifferences checks dL/ds of every loss
// (one-hot and label_dist targets) at several temperatures against central
// differences in float64.
func TestObjectiveGradientsMatchFiniteDifferences(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2)) //nolint:gosec // G404: test data
	for _, c := range gradCases() {
		for _, temp := range []float64{1, 0.7, 2.5} {
			obj := Objective{RegWeight: c.reg}
			target := c.ex.Target()
			k := c.ex.NumOptions()
			s := make([]float64, k)
			for i := range s {
				s[i] = r.NormFloat64() * 2
			}
			ds := make([]float64, k)
			obj.Loss(c.ex.Type, target, s, temp, ds)
			for i := range s {
				const eps = 1e-6
				orig := s[i]
				s[i] = orig + eps
				up := obj.Loss(c.ex.Type, target, s, temp, nil)
				s[i] = orig - eps
				dn := obj.Loss(c.ex.Type, target, s, temp, nil)
				s[i] = orig
				fd := (up - dn) / (2 * eps)
				if math.Abs(fd-ds[i]) > 1e-7*math.Max(1, math.Abs(fd)) {
					t.Errorf("%s T=%g: dL/ds[%d] = %.10f, finite difference %.10f", c.name, temp, i, ds[i], fd)
				}
			}
		}
	}
}

// TestLogisticYesNoEqualsTwoWaySoftmax: the yes/no logistic loss on the
// margin is the two-marker softmax cross-entropy, for hard and soft targets.
func TestLogisticYesNoEqualsTwoWaySoftmax(t *testing.T) {
	for _, target := range [][]float64{{1, 0}, {0, 1}, {0.25, 0.75}} {
		for _, s := range [][]float64{{0.3, -1.2}, {5, 4}, {-30, 40}} {
			a, b := make([]float64, 2), make([]float64, 2)
			la := Objective{}.Loss(TaskYesNo, target, s, 1.3, a)
			lb := Objective{}.Loss(TaskChoice, target, s, 1.3, b)
			if math.Abs(la-lb) > 1e-12 || math.Abs(a[0]-b[0]) > 1e-12 || math.Abs(a[1]-b[1]) > 1e-12 {
				t.Fatalf("target %v s=%v: logistic %v %v, softmax %v %v", target, s, la, a, lb, b)
			}
		}
	}
}

// TestListwiseLossIsSoftCrossEntropy: with a label_dist the choice loss is
// the listwise softmax cross-entropy -Σ t log p.
func TestListwiseLossIsSoftCrossEntropy(t *testing.T) {
	s := []float64{0.4, -0.3, 1.1}
	target := []float64{2.0 / 3, 1.0 / 3, 0}
	p := softmax64(s, 1)
	want := -(target[0]*math.Log(p[0]) + target[1]*math.Log(p[1]))
	if got := (Objective{}).Loss(TaskChoice, target, s, 1, nil); math.Abs(got-want) > 1e-12 {
		t.Fatalf("listwise loss %v, want %v", got, want)
	}
}

// sampleOf builds a training sample for a gradient case.
func sampleOf(c gradCase, rec *Record) *sample {
	ex := c.ex
	return &sample{it: &Item{Ex: &ex, Scorable: true}, rec: rec, target: ex.Target()}
}

// TestHeadGradientsMatchFiniteDifferences checks every scorer parameter
// (type embedding, LayerNorm, both linears) through the full chain
// features → head → loss against central differences, for every task type
// and target kind.
func TestHeadGradientsMatchFiniteDifferences(t *testing.T) {
	for ci, c := range gradCases() {
		h := gradHead(t, uint64(ci)+10)
		r := rand.New(rand.NewPCG(uint64(ci), 5)) //nolint:gosec // G404: test data
		d := h.Config.D
		k := c.ex.NumOptions()
		rec := &Record{Features: Features{Markers: randFloats(r, k*d, 1.5), Pooled: randFloats(r, d, 1), K: k}}
		s := sampleOf(c, rec)
		tr := newTrainer(h, Objective{RegWeight: c.reg}, 1)
		g := newGrads(h.Config, false)
		w := tr.work[0]
		tr.exampleGrad(s, g, w)
		lossOf := func() float64 {
			return tr.obj.Loss(c.ex.Type, s.target, h.Scores(rec.Markers, k, c.ex.Type.QType()), 1, nil)
		}
		checkParams(t, c.name, h.Params()[:scorerParams], g.slices()[:scorerParams], lossOf)
	}
}

// TestActGradientsMatchFiniteDifferences checks the act head's weighted
// cross-entropy gradient for both targets.
func TestActGradientsMatchFiniteDifferences(t *testing.T) {
	for target := 0; target < 2; target++ {
		h := gradHead(t, 77+uint64(target))
		r := rand.New(rand.NewPCG(3, uint64(target))) //nolint:gosec // G404: test data
		pooled := randFloats(r, h.Config.D, 1)
		logits := randFloats(r, 3, 2)
		g := newGrads(h.Config, true)
		sc := newActScratch(h.Config)
		const weight = 1.7
		h.actExampleGrad(pooled, TaskChoice.QType(), logits, target, weight, sc, g)
		lossOf := func() float64 {
			p := softmax64(h.actForward(pooled, TaskChoice.QType(), logits, newActScratch(h.Config)), 1)
			return -weight * math.Log(p[target])
		}
		checkParams(t, fmt.Sprintf("act target %d", target), h.Params()[scorerParams:], g.slices()[scorerParams:], lossOf)
	}
}

// checkParams perturbs every element of every parameter by ±eps (the
// actual float32 step is used as the denominator) and compares the central
// difference with the analytic gradient.
func checkParams(t *testing.T, name string, params []Param, gs [][]float64, lossOf func() float64) {
	t.Helper()
	const eps = 1e-3
	checked, worst := 0, 0.0
	for pi, p := range params {
		for i := range p.Data {
			orig := p.Data[i]
			up, dn := float32(float64(orig)+eps), float32(float64(orig)-eps)
			p.Data[i] = up
			lu := lossOf()
			p.Data[i] = dn
			ld := lossOf()
			p.Data[i] = orig
			fd := (lu - ld) / (float64(up) - float64(dn))
			an := gs[pi][i]
			err := math.Abs(fd-an) / math.Max(1e-2, math.Max(math.Abs(fd), math.Abs(an)))
			worst = math.Max(worst, err)
			if err > 2e-3 {
				t.Errorf("%s: %s[%d] analytic %.8f, finite difference %.8f", name, p.Name, i, an, fd)
			}
			checked++
		}
	}
	t.Logf("%s: %d parameters checked, worst relative error %.2e", name, checked, worst)
}

// TestGradientReductionIsIndependentOfWorkers: one batch step gives the
// same gradient bits with 1, 3 and 8 workers.
func TestGradientReductionIsIndependentOfWorkers(t *testing.T) {
	var ref [][]float64
	for _, workers := range []int{1, 3, 8} {
		h := gradHead(t, 5)
		r := rand.New(rand.NewPCG(9, 9)) //nolint:gosec // G404: test data
		var batch []*sample
		for i := 0; i < 29; i++ {
			c := gradCases()[i%len(gradCases())]
			k := c.ex.NumOptions()
			batch = append(batch, sampleOf(c, &Record{Features: Features{Markers: randFloats(r, k*h.Config.D, 1), K: k}}))
		}
		tr := newTrainer(h, Objective{RegWeight: 0.5}, workers)
		tr.scorerStep(batch)
		got := tr.total.slices()[:scorerParams]
		if ref == nil {
			for _, s := range got {
				ref = append(ref, append([]float64(nil), s...))
			}
			continue
		}
		for i := range got {
			for j := range got[i] {
				if got[i][j] != ref[i][j] {
					t.Fatalf("workers=%d: gradient %d[%d] = %v, workers=1 gave %v", workers, i, j, got[i][j], ref[i][j])
				}
			}
		}
	}
}
