package train

import (
	"math"
	"slices"
	"sort"

	"github.com/UnicoLab/ul_layago/laya"
)

// OptionBucket is Laya's option-count bucket (common.py temp_bucket): 2,
// 3-5, 6-10 or 11+.
func OptionBucket(k int) string {
	switch {
	case k <= 2:
		return "2"
	case k <= 5:
		return "3-5"
	case k <= 10:
		return "6-10"
	default:
		return "11+"
	}
}

// TempEntry is one fitted temperature cell (manifest
// calibration.temperatures).
type TempEntry struct {
	T float64 `json:"t"`
	N int     `json:"n"`
	// NLLAtOne and NLL are the cell's mean NLL at T=1 and at T.
	NLLAtOne float64 `json:"nll_t1"`
	NLL      float64 `json:"nll"`
}

// Temperatures are keyed "<task>:<bucket>", "<task>" and "*"; a cell with
// fewer than the minimum items is absent, and Lookup falls back in that
// order, then to 1.0.
type Temperatures map[string]TempEntry

// Lookup returns the temperature for a task with k options and the key it
// came from ("" for the 1.0 fallback), clamped like every Laya temperature,
// as laya.Artifact.Temperature does.
func (tt Temperatures) Lookup(task string, k int) (float64, string) {
	for _, key := range []string{task + ":" + OptionBucket(k), task, "*"} {
		if e, ok := tt[key]; ok {
			return laya.ClampTemperature(e.T), key
		}
	}
	return 1, ""
}

// tempRecord is one scorable dev_cal item for the temperature fit.
type tempRecord struct {
	task   string
	k      int
	scores []float64
	label  int
}

// fitTemperatures fits one temperature per (task, option-count bucket), per
// task and overall ("*") on dev_cal by minimising the mean NLL of the label,
// as calibrate.py fit_temperatures does. Every cell, the overall one
// included, needs at least minCell items; smaller cells are not fitted and
// the lookup falls back. Only the label's NLL is fitted: a soft label is
// never on dev_cal, and every example has one class label.
func fitTemperatures(records []tempRecord, minCell int) Temperatures {
	cells := map[string][]tempRecord{}
	for _, r := range records {
		cells[r.task+":"+OptionBucket(r.k)] = append(cells[r.task+":"+OptionBucket(r.k)], r)
		cells[r.task] = append(cells[r.task], r)
		cells["*"] = append(cells["*"], r)
	}
	out := Temperatures{}
	for _, key := range sortedKeys(cells) {
		rs := cells[key]
		if len(rs) < minCell || len(rs) == 0 {
			continue
		}
		mean := func(t float64) float64 {
			var s float64
			for _, r := range rs {
				s += nll(r.scores, t, r.label)
			}
			return s / float64(len(rs))
		}
		t := fitTemperature(mean)
		out[key] = TempEntry{T: t, N: len(rs), NLLAtOne: mean(1), NLL: mean(t)}
	}
	return out
}

// fitTemperature minimises f over T in [0.5, 5]: a 61-point log grid, then
// golden-section search around the best grid point, then the clamp bounds
// (golden section only approaches an optimum at a bound). The result is
// rounded to 6 decimals, the precision the manifest records and every
// decision then uses.
func fitTemperature(f func(float64) float64) float64 {
	lo, hi := math.Log(laya.TempMin), math.Log(laya.TempMax)
	const n = 61
	grid := make([]float64, n)
	best, bestV := 0, math.Inf(1)
	for i := range grid {
		grid[i] = lo + (hi-lo)*float64(i)/float64(n-1)
		if v := f(math.Exp(grid[i])); v < bestV {
			best, bestV = i, v
		}
	}
	a, b := grid[max(0, best-1)], grid[min(n-1, best+1)]
	phi := (math.Sqrt(5) - 1) / 2
	c, d := b-phi*(b-a), a+phi*(b-a)
	fc, fd := f(math.Exp(c)), f(math.Exp(d))
	for i := 0; i < 40; i++ {
		if fc < fd {
			b, d, fd = d, c, fc
			c = b - phi*(b-a)
			fc = f(math.Exp(c))
		} else {
			a, c, fc = c, d, fd
			d = a + phi*(b-a)
			fd = f(math.Exp(d))
		}
	}
	t := laya.ClampTemperature(math.Exp((a + b) / 2))
	cands := []float64{t, laya.TempMin, laya.TempMax}
	sort.SliceStable(cands, func(i, j int) bool { return f(cands[i]) < f(cands[j]) })
	return math.Round(cands[0]*1e6) / 1e6
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	return keys
}

// QuantizeProbability rounds p to the 1e-4 grid as floor(p·10000 + 0.5) /
// 10000, the expression laya.QuantizeProbability and the Python trainer
// use, so every side compares the same number with a threshold.
func QuantizeProbability(p float64) float64 { return math.Floor(p*1e4+0.5) / 1e4 }

// Threshold is one task's act threshold (manifest act.thresholds.<task>): act
// when the quantised score of kind Score reaches Lambda. An uncertified
// threshold has a null lambda, and the task always escalates.
type Threshold struct {
	Score     string   `json:"score"`
	Lambda    *float64 `json:"lambda"`
	Certified bool     `json:"certified"`
	// Alpha is the selective-risk bound; Delta this score family's share of
	// the failure probability, DeltaFamily the whole family's.
	Alpha       float64 `json:"alpha"`
	Delta       float64 `json:"delta"`
	DeltaFamily float64 `json:"delta_family"`
	NMin        int     `json:"n_min"`
	// NStart is where the fixed sequence starts: the fewest acted items
	// tested.
	NStart int `json:"n_start"`
	// N items in dev_conf, Tested thresholds tested, Acted and Errors at
	// the certified lambda.
	N        int     `json:"n"`
	Tested   int     `json:"tested"`
	Acted    int     `json:"acted"`
	Errors   int     `json:"errors"`
	Coverage float64 `json:"coverage"`
	PValue   float64 `json:"p_value,omitempty"`
	// RiskUpperBound is the one-sided (1-Delta) Clopper-Pearson bound on the
	// error rate among acted items at lambda (reported, not a decision).
	RiskUpperBound float64             `json:"risk_upper_bound,omitempty"`
	AUROC          map[string]*float64 `json:"auroc"`
	SelectionRule  string              `json:"selection_rule"`
}

// Acts reports whether a score acts under the threshold (nil: escalate).
func (t *Threshold) Acts(score float64) bool {
	if t == nil || !t.Certified || t.Lambda == nil {
		return false
	}
	return QuantizeProbability(score) >= *t.Lambda-1e-12
}

const probabilityGrid = 10000

// LearnThenTest certifies an act threshold on scores (dev_conf) whose items
// are correct or not: fixed-sequence Learn-then-Test (Angelopoulos et al.,
// 2021) over the 1e-4 grid, as calibrate.py ltt_threshold does. For lambda =
// 1.0000, 0.9999, ..., 0 the null "the error rate among items acting at
// lambda exceeds alpha" is tested with the exact binomial tail P(Bin(n_act,
// alpha) <= errors), in fixed sequence from the most conservative lambda
// down, stopping at the first p > delta; the lowest rejected lambda is the
// threshold (no rejection: uncertified). The sequence starts where at least
// n_start = max(nMin, ceil(3 ln delta / ln(1-alpha))) items would act: below
// that the test has too little power, and a non-rejection there would end
// the sequence before any threshold with enough data. The start depends only
// on the unlabelled scores.
func LearnThenTest(scores []float64, correct []bool, alpha, delta float64, nMin int) Threshold {
	return LearnThenTestFrom(scores, correct, alpha, delta, nMin, LTTStart(alpha, delta, nMin, 0))
}

// LTTStart is the fewest acted items the fixed sequence first tests at
// risk bound alpha and failure probability delta. With startErrors 0 it is
// max(nMin, ceil(3 ln delta / ln(1 - alpha))), where zero errors reject
// at delta cubed; with startErrors E > 0 it is max(nMin, the smallest n
// with P(Bin(n, alpha) <= E) <= delta), the fewest acted items at which E
// errors can still reject. Both depend only on constants.
func LTTStart(alpha, delta float64, nMin, startErrors int) int {
	if startErrors <= 0 {
		return max(nMin, int(math.Ceil(3*math.Log(delta)/math.Log1p(-alpha))))
	}
	n := startErrors + 1
	for binomCDF(startErrors, n, alpha) > delta {
		n++
	}
	return max(nMin, n)
}

// LearnThenTestFrom is LearnThenTest with the sequence starting at nStart
// acted items.
func LearnThenTestFrom(scores []float64, correct []bool, alpha, delta float64, nMin, nStart int) Threshold {
	n := len(scores)
	s := make([]float64, n)
	for i, v := range scores {
		s[i] = QuantizeProbability(v)
	}
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return s[order[a]] > s[order[b]] })
	cumErr := make([]int, n)
	for j, i := range order {
		if j > 0 {
			cumErr[j] = cumErr[j-1]
		}
		if !correct[i] {
			cumErr[j]++
		}
	}
	th := Threshold{Alpha: alpha, Delta: delta, NMin: nMin, N: n, NStart: max(nMin, nStart)}
	memo := map[[2]int]float64{}
	acted := 0
	for j := probabilityGrid; j >= 0; j-- {
		lam := float64(j) / probabilityGrid
		for acted < n && s[order[acted]] >= lam-1e-12 {
			acted++
		}
		if acted < th.NStart {
			continue
		}
		errs := cumErr[acted-1]
		th.Tested++
		key := [2]int{errs, acted}
		p, ok := memo[key]
		if !ok {
			p = binomCDF(errs, acted, alpha)
			memo[key] = p
		}
		if p > delta {
			break
		}
		l := lam
		th.Lambda, th.Certified, th.Acted, th.Errors, th.PValue = &l, true, acted, errs, p
	}
	if th.Certified {
		th.Coverage = float64(th.Acted) / float64(max(1, n))
		th.RiskUpperBound = clopperPearsonUpper(th.Errors, th.Acted, delta)
	}
	return th
}

// thresholdRow is one scorable dev_conf item for the threshold fit.
type thresholdRow struct {
	task             string
	correct          bool
	maxProb, actProb float64
}

// actHeadMinAUROC is the design's floor for reading the act head instead of
// the calibrated top probability (revise_laya-go §3, "Escalation").
const actHeadMinAUROC = 0.80

// chooseThresholds certifies one threshold per task on dev_conf. Both
// scores (the calibrated max probability and the act head's P(act)) are
// candidates; act_probability is read only when its AUROC for correctness is
// at least 0.80 and above max_probability's. Because the choice is made on
// the same labels the test uses, delta is split between the two score
// families (Bonferroni over both, delta/2 each), so the certificate holds at
// the family's delta whichever score was chosen. A task without dev_conf
// items gets no threshold and always escalates.
func chooseThresholds(rows []thresholdRow, alpha, delta float64, nMin, startErrors int) map[string]Threshold {
	byTask := map[string][]thresholdRow{}
	for _, r := range rows {
		byTask[r.task] = append(byTask[r.task], r)
	}
	out := map[string]Threshold{}
	for _, task := range sortedKeys(byTask) {
		rs := byTask[task]
		correct := make([]bool, len(rs))
		maxp, actp := make([]float64, len(rs)), make([]float64, len(rs))
		for i, r := range rs {
			correct[i], maxp[i], actp[i] = r.correct, r.maxProb, r.actProb
		}
		am, aa := AUROC(maxp, correct), AUROC(actp, correct)
		kind, scores := ScoreMaxProbability, maxp
		if !math.IsNaN(aa) && aa >= actHeadMinAUROC && (math.IsNaN(am) || aa > am) {
			kind, scores = ScoreActProbability, actp
		}
		th := LearnThenTestFrom(scores, correct, alpha, delta/2, nMin, LTTStart(alpha, delta/2, nMin, startErrors))
		th.Score, th.DeltaFamily = kind, delta
		th.AUROC = map[string]*float64{ScoreMaxProbability: nanNil(am), ScoreActProbability: nanNil(aa)}
		th.SelectionRule = "act_probability if its dev_conf AUROC for correctness is >= 0.80 and above max_probability's, " +
			"else max_probability; delta split between both score families (Bonferroni)"
		out[task] = th
	}
	return out
}

func nanNil(v float64) *float64 {
	if math.IsNaN(v) {
		return nil
	}
	return &v
}

// Scores a threshold can be certified on.
const (
	ScoreMaxProbability = "max_probability"
	ScoreActProbability = "act_probability"
)

// binomCDF is P(X <= k) for X ~ Binomial(n, p), summed in log space.
func binomCDF(k, n int, p float64) float64 {
	switch {
	case n == 0 || k >= n:
		return 1
	case k < 0:
		return 0
	case p <= 0:
		return 1
	case p >= 1:
		return 0
	}
	lgN, _ := math.Lgamma(float64(n + 1))
	lp, lq := math.Log(p), math.Log1p(-p)
	terms := make([]float64, k+1)
	m := math.Inf(-1)
	for i := 0; i <= k; i++ {
		a, _ := math.Lgamma(float64(i + 1))
		b, _ := math.Lgamma(float64(n - i + 1))
		terms[i] = lgN - a - b + float64(i)*lp + float64(n-i)*lq
		m = math.Max(m, terms[i])
	}
	var s float64
	for _, t := range terms {
		s += math.Exp(t - m)
	}
	return math.Min(1, math.Exp(m)*s)
}

// clopperPearsonUpper is the one-sided (1-delta) upper confidence bound on a
// binomial proportion after k errors in n trials.
func clopperPearsonUpper(k, n int, delta float64) float64 {
	if n == 0 || k >= n {
		return 1
	}
	lo, hi := float64(k)/float64(n), 1.0
	for i := 0; i < 100; i++ {
		mid := (lo + hi) / 2
		if binomCDF(k, n, mid) > delta {
			lo = mid
		} else {
			hi = mid
		}
	}
	return hi
}

// AUROC is the probability that a random correct item scores above a random
// wrong one (ties count half), or NaN without both classes.
func AUROC(scores []float64, positive []bool) float64 {
	type pair struct {
		s   float64
		pos bool
	}
	ps := make([]pair, len(scores))
	npos := 0
	for i := range scores {
		ps[i] = pair{scores[i], positive[i]}
		if positive[i] {
			npos++
		}
	}
	nneg := len(ps) - npos
	if npos == 0 || nneg == 0 {
		return math.NaN()
	}
	sort.SliceStable(ps, func(i, j int) bool { return ps[i].s < ps[j].s })
	var rankSum float64
	for i := 0; i < len(ps); {
		j := i
		for j < len(ps) && ps[j].s == ps[i].s {
			j++
		}
		avg := float64(i+j+1) / 2 // mean of 1-based ranks i+1..j
		for t := i; t < j; t++ {
			if ps[t].pos {
				rankSum += avg
			}
		}
		i = j
	}
	return (rankSum - float64(npos)*float64(npos+1)/2) / (float64(npos) * float64(nneg))
}
