package train

import "math"

// softmax64 returns softmax(s/t) with the maximum subtracted.
func softmax64(s []float64, t float64) []float64 {
	out := make([]float64, len(s))
	if len(s) == 0 {
		return out
	}
	m := math.Inf(-1)
	for _, v := range s {
		m = math.Max(m, v/t)
	}
	var sum float64
	for i, v := range s {
		out[i] = math.Exp(v/t - m)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// logSumExp returns log Σ exp(s_i/t).
func logSumExp(s []float64, t float64) float64 {
	m := math.Inf(-1)
	for _, v := range s {
		m = math.Max(m, v/t)
	}
	var sum float64
	for _, v := range s {
		sum += math.Exp(v/t - m)
	}
	return m + math.Log(sum)
}

// softplus is log(1 + e^x) without overflow.
func softplus(x float64) float64 {
	if x > 0 {
		return x + math.Log1p(math.Exp(-x))
	}
	return math.Log1p(math.Exp(x))
}

func sigmoid(x float64) float64 {
	if x >= 0 {
		return 1 / (1 + math.Exp(-x))
	}
	e := math.Exp(x)
	return e / (1 + e)
}

// Objective is the per-example loss. The target t over the options is the
// example's one-hot label, or its label_dist soft label when the dataset
// gives one (train split only):
//
//   - choice and score: softmax cross-entropy against t,
//     L = -Σ t_i log softmax(s/T)_i, dL/ds = (p - t)/T. With a label_dist
//     over items to rank (r_answer_rank's evidence items, graded by
//     relevance), this is the listwise softmax ranking loss (ListNet, top
//     one probability);
//   - score levels add RegWeight times the ordinal regression term
//     ((E_p[level] - E_t[level])/(k-1))², the squared error of the expected
//     level, dL/ds_j = 2w(E_p-E_t)/(k-1)²·p_j(j-E_p)/T;
//   - yes/no: logistic loss on the margin Δ = (s_true - s_false)/T with
//     y = t_true, L = log(1+e^Δ) - yΔ, dL/dΔ = σ(Δ) - y (the two-marker
//     softmax cross-entropy written in one variable).
type Objective struct {
	// RegWeight weights the expected-level regression term on score tasks
	// (0 disables it).
	RegWeight float64
}

// Loss returns the loss of scores s against target at temperature t and,
// when ds is non-nil, writes dL/ds into it.
func (o Objective) Loss(tt TaskType, target, s []float64, t float64, ds []float64) float64 {
	if ds != nil {
		clear(ds)
	}
	if tt == TaskYesNo && len(s) == 2 {
		delta := (s[1] - s[0]) / t
		y := target[1]
		if ds != nil {
			g := (sigmoid(delta) - y) / t
			ds[1], ds[0] = g, -g
		}
		return softplus(delta) - y*delta
	}
	lse := logSumExp(s, t)
	loss := lse
	for i, ti := range target {
		loss -= ti * s[i] / t
	}
	if ds != nil {
		p := softmax64(s, t)
		for i := range s {
			ds[i] = (p[i] - target[i]) / t
		}
	}
	if tt == TaskScore && o.RegWeight > 0 {
		var y float64
		for i, ti := range target {
			y += float64(i) * ti
		}
		loss += o.expectedLevel(s, t, y, o.RegWeight, ds)
	}
	return loss
}

// expectedLevel adds w·((E-y)/(k-1))² and its gradient into ds.
func (o Objective) expectedLevel(s []float64, t, y, w float64, ds []float64) float64 {
	k := len(s)
	p := softmax64(s, t)
	var e float64
	for i, v := range p {
		e += float64(i) * v
	}
	norm := float64((k - 1) * (k - 1))
	diff := e - y
	if ds != nil {
		for j := range s {
			ds[j] += w * 2 * diff / norm * p[j] * (float64(j) - e) / t
		}
	}
	return w * diff * diff / norm
}

// nll is -log softmax(s/t)[label].
func nll(s []float64, t float64, label int) float64 {
	return logSumExp(s, t) - s[label]/t
}
