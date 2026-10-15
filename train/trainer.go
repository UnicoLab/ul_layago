package train

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"time"

	"github.com/UnicoLab/ul_layago/tensor"
)

// Config controls one training run. Train uses every value as given, zero
// included (a zero weight decay is no decay); start from DefaultConfig and
// change what differs.
type Config struct {
	// FitScorer is an optional accelerated scorer stage. Calibration/export remain Go.
	FitScorer   ScorerFitFunc                `json:"-"`
	Backend     string                       `json:"scorer_backend,omitempty"`
	BackendInfo map[string]any               `json:"backend_info,omitempty"`
	Progress    func(stage string, step int) `json:"-"`
	// InitHead optionally warm-starts a compatible zero-transformer-layer head.
	// The caller must verify base identity; Train clones it and refits calibration.
	InitHead          *Head  `json:"-"`
	InitWeightsSHA256 string `json:"init_weights_sha256,omitempty"`
	// Seed drives initialisation and shuffling; a run is deterministic for a
	// fixed seed, data and cache, whatever Workers is.
	Seed uint64 `json:"seed"`
	// Epochs is the maximum number of passes; Patience stops after that many
	// epochs without a better dev_cal NLL.
	Epochs int `json:"max_epochs"`
	// ScheduleEpochs decouples cosine decay from the maximum/early-stop cap (0 uses Epochs).
	ScheduleEpochs int `json:"schedule_epochs,omitempty"`
	Patience       int `json:"patience"`
	// BatchSize, LR (peak), WeightDecay (matrices only) and WarmupFrac shape
	// AdamW with warmup and cosine decay to 10 %.
	BatchSize   int     `json:"batch_size"`
	LR          float64 `json:"learning_rate"`
	WeightDecay float64 `json:"weight_decay"`
	WarmupFrac  float64 `json:"warmup_fraction"`
	// ScoreRegWeight weights the expected-level squared error added to the
	// cross-entropy of score tasks (0 disables it).
	ScoreRegWeight float64 `json:"score_regression_weight"`
	// DistillWeight and DistillTemp configure teacher KL distillation.
	// DistillTasks, DisableDistillTasks, and DistillWeightByTask control
	// task-selective distillation.
	DistillWeight       float64            `json:"distill_weight,omitempty"`
	DistillTemp         float64            `json:"distill_temperature,omitempty"`
	DistillTasks        []string           `json:"distill_tasks,omitempty"`
	DisableDistillTasks []string           `json:"disable_distill_tasks,omitempty"`
	DistillWeightByTask map[string]float64 `json:"distill_weight_by_task,omitempty"`
	// ActEpochs and ActLR train the act head on dev_cal.
	ActEpochs int     `json:"act_epochs"`
	ActLR     float64 `json:"act_learning_rate"`
	// MinCell is the fewest dev_cal items a temperature cell (task:bucket,
	// task or "*") needs to be fitted.
	MinCell int `json:"min_cell"`
	// Alpha and Delta set the act-threshold guarantee on dev_conf: error
	// rate among acted items at most Alpha with probability at least
	// 1-Delta. NMin is the fewest acted items the fixed sequence tests.
	Alpha float64 `json:"alpha"`
	Delta float64 `json:"delta"`
	NMin  int     `json:"n_min"`
	// Int8 also exports an int8 head, recalibrated and recertified on its
	// own outputs.
	Int8 bool `json:"export_int8"`
	// FitLangs, when set, restricts every fit on dev to items of these
	// languages (primary subtags): early stopping, the act head and the
	// temperatures on dev_cal, the act thresholds on dev_conf. The scorer
	// still trains on train in every language, and every item is
	// evaluated.
	FitLangs []string `json:"fit_langs,omitempty"`
	// YesNoBalance weights each yes_no train example's loss by n / (2 n_c),
	// n_c the count of its answer among its task's train examples, so both
	// answers of a task weigh the same in the scorer's loss.
	YesNoBalance bool `json:"yes_no_class_balanced"`
	// LTTStartErrors sets where the Learn-then-Test sequence starts: 0 is
	// max(n_min, ceil(3 ln delta / ln(1 - alpha))) acted items; E > 0 is
	// max(n_min, the fewest acted items at which E errors can still
	// reject). Either depends only on constants, never on labels.
	LTTStartErrors int `json:"ltt_start_errors"`
	// Workers bounds parallelism (GOMAXPROCS when 0); it never changes
	// results.
	Workers int `json:"-"`
	// Log, when set, receives progress lines.
	Log func(format string, args ...any) `json:"-"`
}

// DefaultQADistillTasks returns the Q and A tasks that benefit from teacher
// KL distillation while excluding q_scope so high-cardinality dimension-value
// logits are not smoothed by teacher soft distributions.
func DefaultQADistillTasks() []string {
	return []string{
		"q_intent",
		"q_operation",
		"q_metric",
		"a_entailment",
		"a_scope_preserved",
		"a_limits_preserved",
		"a_no_new_claims",
	}
}

// TaskDistillWeight returns the effective KL-distillation weight for task.
// DisableDistillTasks always forces 0. DistillWeightByTask overrides the
// scalar DistillWeight for tasks present in the map. When DistillTasks is
// non-empty, only tasks listed in DistillTasks receive DistillWeight.
func (c Config) TaskDistillWeight(task string) float64 {
	if slices.Contains(c.DisableDistillTasks, task) {
		return 0
	}
	if c.DistillWeightByTask != nil {
		if w, ok := c.DistillWeightByTask[task]; ok {
			if w <= 0 {
				return 0
			}
			return w
		}
	}
	if c.DistillWeight <= 0 {
		return 0
	}
	if len(c.DistillTasks) > 0 && !slices.Contains(c.DistillTasks, task) {
		return 0
	}
	return c.DistillWeight
}

// DefaultConfig is the configuration laya-train's flags default to.
func DefaultConfig() Config {
	return Config{Seed: 1, Epochs: 30, Patience: 4, BatchSize: 32, LR: 1e-3, WeightDecay: 0.01, WarmupFrac: 0.06,
		ScoreRegWeight: 0.5, ActEpochs: 30, ActLR: 1e-3, MinCell: 30, Alpha: 0.05, Delta: 0.1, NMin: 20, Int8: true}
}

func (c Config) validate() error {
	for _, v := range []float64{c.LR, c.ActLR, c.WeightDecay, c.WarmupFrac, c.ScoreRegWeight, c.Alpha, c.Delta, c.DistillWeight, c.DistillTemp} {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("train: configuration values must be finite")
		}
	}
	switch {
	case c.Epochs < 1, c.Patience < 1, c.BatchSize < 1, c.ActEpochs < 0:
		return fmt.Errorf("train: epochs %d, patience %d and batch size %d must be positive, act epochs %d not negative",
			c.Epochs, c.Patience, c.BatchSize, c.ActEpochs)
	case c.ScheduleEpochs < 0 || c.ScheduleEpochs > c.Epochs:
		return fmt.Errorf("train: schedule epochs must be 0 or at most max_epochs")
	case !(c.LR > 0) || !(c.ActLR > 0):
		return fmt.Errorf("train: learning rates %v and %v must be positive", c.LR, c.ActLR)
	case c.WeightDecay < 0 || c.WarmupFrac < 0 || c.WarmupFrac >= 1 || c.ScoreRegWeight < 0:
		return fmt.Errorf("train: weight decay %v, warmup %v and score regression weight %v must be in range",
			c.WeightDecay, c.WarmupFrac, c.ScoreRegWeight)
	case c.MinCell < 1 || c.NMin < 1:
		return fmt.Errorf("train: min cell %d and n_min %d must be positive", c.MinCell, c.NMin)
	case !(c.Alpha > 0 && c.Alpha < 1) || !(c.Delta > 0 && c.Delta < 1):
		return fmt.Errorf("train: alpha %v and delta %v must be in (0, 1)", c.Alpha, c.Delta)
	case c.LTTStartErrors < 0:
		return fmt.Errorf("train: ltt start errors %d must not be negative", c.LTTStartErrors)
	case c.DistillWeight < 0:
		return fmt.Errorf("train: distill weight %v must not be negative", c.DistillWeight)
	}
	return nil
}

// fits reports whether an item's language is one the dev fits use.
func (c Config) fits(it *Item) bool {
	return len(c.FitLangs) == 0 || slices.Contains(c.FitLangs, primaryLanguage(it.Ex.Lang))
}

// EpochStats is one epoch of the scorer stage.
type EpochStats struct {
	Epoch             int     `json:"epoch"`
	TrainLoss         float64 `json:"train_loss"`
	DevCalNLL         float64 `json:"dev_cal_nll"`
	DevCalAccuracy    float64 `json:"dev_cal_accuracy"`
	LR                float64 `json:"learning_rate_at_end"`
	Seconds           float64 `json:"seconds"`
	ExamplesPerSecond float64 `json:"examples_per_second"`
}

// ActReport describes the act head's fit.
type ActReport struct {
	Trained   bool    `json:"trained"`
	Reason    string  `json:"reason,omitempty"`
	N         int     `json:"n"`
	Positives int     `json:"positives"`
	FinalLoss float64 `json:"final_loss"`
	Epochs    int     `json:"epochs"`
	Objective string  `json:"objective"`
}

// Int8Delta compares the int8 head with the float32 head, each under its own
// temperatures and thresholds (manifest evaluation.delta_vs_float32).
type Int8Delta struct {
	MaxAbsProb    float64 `json:"max_abs_prob"`
	MaxAbsLogit   float64 `json:"max_abs_logit"`
	ArgmaxFlips   int     `json:"argmax_flips"`
	DecisionFlips int     `json:"decision_flips"`
	N             int     `json:"n"`
	Note          string  `json:"note"`
}

// Variant is one numeric path of a trained head with its own calibration and
// evaluation: the float32 head, or the int8 head (dequantised, recalibrated
// and recertified on its own outputs).
type Variant struct {
	Format       string
	Head         *Head
	Temperatures Temperatures
	Thresholds   map[string]Threshold
	Rows         []Row
	Evaluation   map[string]SplitEval
	Generalise   map[string]map[string]Summary
	// Logits and ActLogits are the served logits of every scorable item, by
	// id.
	Logits, ActLogits map[string][]float32
	Delta             *Int8Delta
}

// DevHalfStats describes one dev half.
type DevHalfStats struct {
	Stories  int            `json:"stories"`
	Examples int            `json:"examples"`
	Scorable int            `json:"scorable"`
	ByTask   map[string]int `json:"by_task"`
}

// DevHalfReport is the manifest's training.dev_halves (train.py
// dev_half_report): stories and scorable examples per half and task, and what
// is too small to calibrate on.
type DevHalfReport struct {
	DevCal             DevHalfStats `json:"dev_cal"`
	DevConf            DevHalfStats `json:"dev_conf"`
	Stories            int          `json:"stories"`
	CertificationStart int          `json:"certification_start"`
	// FitLangs are the languages the counts cover (Config.FitLangs), nil
	// for every language.
	FitLangs   []string `json:"fit_langs,omitempty"`
	Shortfalls []string `json:"shortfalls"`
	Errors     []string `json:"errors"`
}

// devHalfReport counts each dev half. dev is divided by a hash of each story,
// balanced only over many stories: a half without a scorable example is an
// error (dev_cal fits early stopping, the temperatures and the act head;
// dev_conf certifies the thresholds); a task with fewer dev_cal examples than
// MinCell (its temperature falls back) or fewer dev_conf examples than the
// Learn-then-Test start count at delta/2 (it cannot be certified and always
// escalates), and halves more than 4:1 apart in stories, are shortfalls.
func devHalfReport(p *Prepared, cfg Config) DevHalfReport {
	r := DevHalfReport{DevCal: DevHalfStats{ByTask: map[string]int{}}, DevConf: DevHalfStats{ByTask: map[string]int{}},
		Shortfalls: []string{}, Errors: []string{}, FitLangs: cfg.FitLangs}
	stories := map[string]map[string]bool{RoleDevCal: {}, RoleDevConf: {}}
	trainTasks := map[string]bool{}
	for _, it := range p.Items {
		var h *DevHalfStats
		switch it.Role {
		case RoleDevCal:
			h = &r.DevCal
		case RoleDevConf:
			h = &r.DevConf
		case SplitTrain:
			trainTasks[it.Ex.Task] = true
			continue
		default:
			continue
		}
		if !cfg.fits(it) {
			continue
		}
		stories[it.Role][it.Ex.StoryID] = true
		h.Examples++
		if it.Scorable {
			h.Scorable++
			h.ByTask[it.Ex.Task]++
		}
	}
	r.DevCal.Stories, r.DevConf.Stories = len(stories[RoleDevCal]), len(stories[RoleDevConf])
	r.Stories = r.DevCal.Stories + r.DevConf.Stories
	r.CertificationStart = LTTStart(cfg.Alpha, cfg.Delta/2, cfg.NMin, cfg.LTTStartErrors)
	for _, h := range []struct {
		name string
		s    DevHalfStats
	}{{RoleDevCal, r.DevCal}, {RoleDevConf, r.DevConf}} {
		if h.s.Scorable == 0 {
			r.Errors = append(r.Errors, fmt.Sprintf("%s has no scorable example (dev has %d stories: %d in dev_cal, %d in dev_conf); "+
				"dev_cal fits early stopping, temperatures and the act head and dev_conf certifies the act thresholds, so dev "+
				"needs more stories", h.name, r.Stories, r.DevCal.Stories, r.DevConf.Stories))
		}
	}
	if r.Stories > 0 && len(r.Errors) == 0 && min(r.DevCal.Stories, r.DevConf.Stories)*4 < r.Stories {
		r.Shortfalls = append(r.Shortfalls, fmt.Sprintf("dev halves are unbalanced: %d stories in dev_cal, %d in dev_conf",
			r.DevCal.Stories, r.DevConf.Stories))
	}
	tasks := map[string]bool{}
	for _, m := range []map[string]int{r.DevCal.ByTask, r.DevConf.ByTask} {
		for t := range m {
			tasks[t] = true
		}
	}
	for t := range trainTasks {
		tasks[t] = true
	}
	for _, task := range sortedKeys(tasks) {
		if n := r.DevCal.ByTask[task]; n < cfg.MinCell {
			r.Shortfalls = append(r.Shortfalls, fmt.Sprintf("%s: %d dev_cal examples, under min_cell %d; its temperature falls back "+
				"to the instance-wide one or 1.0", task, n, cfg.MinCell))
		}
		if n := r.DevConf.ByTask[task]; n < r.CertificationStart {
			r.Shortfalls = append(r.Shortfalls, fmt.Sprintf("%s: %d dev_conf examples, under the %d a threshold needs (n_min %d, "+
				"alpha %v, delta %v); it cannot be certified and always escalates", task, n, r.CertificationStart, cfg.NMin, cfg.Alpha, cfg.Delta))
		}
	}
	return r
}

// Result is a trained head with its calibration and evaluation.
type Result struct {
	Config    Config
	Prepared  *Prepared
	Float32   *Variant
	Int8      *Variant
	History   []EpochStats
	BestEpoch int
	Act       ActReport
	DevHalves DevHalfReport
	// Roles counts items per role; Unscorable counts those whose markers
	// did not fit (never trained on, escalated in evaluation).
	Roles      map[string]int
	Unscorable map[string]int
	Seconds    float64
}

type sample struct {
	it     *Item
	rec    *Record
	target []float64
	// weight scales the example's scorer loss (class balance); 0 and 1
	// leave it as it is.
	weight float64
}

// yesNoWeights returns, per yes_no task, the weight of each answer:
// n / (2 n_c) over the task's train samples, 1 for an answer that never
// occurs.
func yesNoWeights(train []*sample) map[string][2]float64 {
	counts := map[string][2]int{}
	for _, s := range train {
		if s.it.Ex.Type != TaskYesNo || !s.it.Ex.HasGold() {
			continue
		}
		c := counts[s.it.Ex.Task]
		c[s.it.Ex.Label]++
		counts[s.it.Ex.Task] = c
	}
	out := map[string][2]float64{}
	for task, c := range counts {
		n := float64(c[0] + c[1])
		w := [2]float64{1, 1}
		for i := range c {
			if c[i] > 0 {
				w[i] = n / (2 * float64(c[i]))
			}
		}
		out[task] = w
	}
	return out
}

// Train fits a head on cached features and calibrates it. recs must hold
// the features of every scorable item of p.
func Train(ctx context.Context, p *Prepared, recs map[string]*Record, d int, cfg Config) (*Result, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	cfg.BackendInfo = nil // Provenance is produced by the fitter, never accepted from input config.
	if cfg.FitScorer != nil && (cfg.Backend == "" || cfg.Backend == "go") {
		cfg.Backend = "external"
	}
	logf := cfg.Log
	if logf == nil {
		logf = func(string, ...any) {}
	}
	started := time.Now()
	res := &Result{Config: cfg, Prepared: p, Roles: map[string]int{}, Unscorable: map[string]int{}, DevHalves: devHalfReport(p, cfg)}
	if len(res.DevHalves.Errors) > 0 {
		return nil, fmt.Errorf("train: %s", strings.Join(res.DevHalves.Errors, "; "))
	}
	for _, s := range res.DevHalves.Shortfalls {
		logf("warning: %s", s)
	}
	var all []*sample
	byRole := map[string][]*sample{}
	for _, it := range p.Items {
		res.Roles[it.Role]++
		if !it.Scorable {
			res.Unscorable[it.Role]++
			continue
		}
		rec := recs[it.Ex.ID]
		if rec == nil {
			return nil, fmt.Errorf("train: no cached features for %s", it.Ex.ID)
		}
		if rec.K != it.Ex.NumOptions() || len(rec.Pooled) != d || len(rec.Markers) != rec.K*d {
			return nil, fmt.Errorf("train: features of %s do not match its %d options at width %d", it.Ex.ID, it.Ex.NumOptions(), d)
		}
		s := &sample{it: it, rec: rec, target: it.Ex.Target(), weight: 1}
		all = append(all, s)
		byRole[it.Role] = append(byRole[it.Role], s)
	}
	// Train on every train item whose gold is offered; fit on dev_cal
	// items of the fit languages (early stopping and temperatures only on
	// those whose gold is offered, the act head on all of them: an offered
	// set without the gold is an error the act head should see).
	var trainSet, devCal, devCalNLL []*sample
	for _, s := range byRole[SplitTrain] {
		if s.it.Ex.HasGold() {
			trainSet = append(trainSet, s)
		}
	}
	for _, s := range byRole[RoleDevCal] {
		if cfg.fits(s.it) {
			devCal = append(devCal, s)
			if s.it.Ex.HasGold() {
				devCalNLL = append(devCalNLL, s)
			}
		}
	}
	if len(trainSet) == 0 {
		return nil, fmt.Errorf("train: no scorable train item")
	}
	if len(devCalNLL) == 0 {
		return nil, fmt.Errorf("train: no scorable dev_cal item of the fit languages %v", cfg.FitLangs)
	}
	if cfg.YesNoBalance {
		weights := yesNoWeights(trainSet)
		for _, s := range trainSet {
			if w, ok := weights[s.it.Ex.Task]; ok {
				s.weight = w[s.it.Ex.Label]
			}
		}
		for _, task := range sortedKeys(weights) {
			logf("yes_no class balance %s: false x%.3f, true x%.3f", task, weights[task][0], weights[task][1])
		}
	}

	head, err := NewHead(HeadConfig{D: d, Hidden: d, ActHidden: layaActHidden, NAct: len(ActOutputs)}, cfg.Seed)
	if err != nil {
		return nil, err
	}
	if cfg.InitHead != nil {
		if cfg.InitHead.Config != head.Config {
			return nil, fmt.Errorf("train: initial head shape %+v differs from %+v", cfg.InitHead.Config, head.Config)
		}
		head = cfg.InitHead.Clone()
	}
	var tr *trainer
	if cfg.Backend != "" && cfg.Backend != "go" && cfg.FitScorer == nil {
		return nil, fmt.Errorf("train: scorer backend %q requires its fitter", cfg.Backend)
	}
	if cfg.FitScorer != nil {
		fit, err := cfg.FitScorer(ctx, head.Clone(), fitSamples(trainSet), fitSamples(devCalNLL), cfg)
		if err != nil {
			return nil, err
		}
		if err := validateScorerFit(head, fit); err != nil {
			return nil, err
		}
		for i, param := range fit.Head.Params()[:scorerParams] {
			copy(head.Params()[i].Data, param.Data)
		}
		if len(fit.History) > cfg.Epochs {
			return nil, fmt.Errorf("train: accelerated backend exceeded epoch cap")
		}
		res.History, res.BestEpoch = fit.History, fit.BestEpoch
		res.Config.BackendInfo = fit.Info
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		tr = newTrainer(head, Objective{RegWeight: cfg.ScoreRegWeight}, cfg.Workers)
	} else {
		tr = newTrainer(head, Objective{RegWeight: cfg.ScoreRegWeight}, cfg.Workers)

		// Stage 1: the scorer and type embedding on train, early-stopped on the
		// dev_cal NLL.
		params := head.Params()
		opt := NewAdamW(params[:scorerParams], cfg.WeightDecay)
		steps := (len(trainSet) + cfg.BatchSize - 1) / cfg.BatchSize
		sched := LRSchedule{Peak: cfg.LR, Floor: 0.1, Warmup: int(cfg.WarmupFrac * float64(steps*cfg.scorerScheduleEpochs())), Total: steps * cfg.scorerScheduleEpochs()}
		rng := rand.New(rand.NewPCG(cfg.Seed, cfg.Seed^0xda942042e4dd58b5)) //nolint:gosec // G404: reproducible shuffling, not a security use
		best, bad, step := math.Inf(1), 0, 0
		bestHead := head.Clone()
		for epoch := 1; epoch <= cfg.Epochs; epoch++ {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			t0 := time.Now()
			perm := rng.Perm(len(trainSet))
			var trainLoss float64
			batch := make([]*sample, 0, cfg.BatchSize)
			for b := 0; b < len(perm); b += cfg.BatchSize {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
				batch = batch[:0]
				for _, i := range perm[b:min(len(perm), b+cfg.BatchSize)] {
					batch = append(batch, trainSet[i])
				}
				trainLoss += tr.scorerStep(batch)
				step++
				opt.Step(tr.total.slices()[:scorerParams], sched.At(step))
				advance(cfg, "scorer", step)
			}
			devNLL, devAcc := tr.devNLL(devCalNLL)
			secs := time.Since(t0).Seconds()
			st := EpochStats{Epoch: epoch, TrainLoss: trainLoss / float64(len(trainSet)), DevCalNLL: devNLL, DevCalAccuracy: devAcc,
				LR: sched.At(step), Seconds: secs, ExamplesPerSecond: float64(len(trainSet)) / math.Max(secs, 1e-9)}
			res.History = append(res.History, st)
			logf("epoch %d: train loss %.4f, dev_cal NLL %.4f, dev_cal accuracy %.4f (%.0f examples/s)", epoch, st.TrainLoss, devNLL,
				devAcc, st.ExamplesPerSecond)
			if devNLL < best-1e-9 {
				best, bad, res.BestEpoch = devNLL, 0, epoch
				bestHead = head.Clone()
				continue
			}
			if bad++; bad >= cfg.Patience {
				break
			}
		}
		for i, p := range bestHead.Params()[:scorerParams] {
			copy(params[i].Data, p.Data)
		}

	}

	advance(cfg, "act", 0)
	// Stage 2: the act head on dev_cal, as a predictor of whether the
	// scorer's answer is right.
	logits := tr.servedLogits(all)
	logitsOf := map[*sample][]float32{}
	for i, s := range all {
		logitsOf[s] = logits[i]
	}
	res.Act, err = tr.trainAct(ctx, devCal, logitsOf, cfg, logf)
	if err != nil {
		return nil, err
	}

	advance(cfg, "calibration", 0)
	res.Float32 = calibrate(p, head, "float32", all, cfg)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	advance(cfg, "calibration", 1)
	logThresholds(logf, res.Float32)
	if cfg.Int8 {
		res.Int8 = calibrate(p, QuantizedHead(head), "int8", all, cfg)
		res.Int8.Delta = compareVariants(res.Float32, res.Int8)
		logThresholds(logf, res.Int8)
	}
	res.Seconds = time.Since(started).Seconds()
	return res, nil
}

func logThresholds(logf func(string, ...any), v *Variant) {
	for _, task := range sortedKeys(v.Thresholds) {
		th := v.Thresholds[task]
		lam := "null"
		if th.Lambda != nil {
			lam = fmt.Sprintf("%.4f", *th.Lambda)
		}
		logf("%s act threshold %s: score %s, lambda %s, certified %v, coverage %.3f on %d dev_conf items", v.Format, task, th.Score,
			lam, th.Certified, th.Coverage, th.N)
	}
}

// calibrate computes a head's served logits for every scorable item, fits
// its temperatures on dev_cal and certifies its act thresholds on dev_conf,
// then decodes and evaluates every item (unscorable ones escalate).
func calibrate(p *Prepared, h *Head, format string, all []*sample, cfg Config) *Variant {
	v := &Variant{Format: format, Head: h, Logits: map[string][]float32{}, ActLogits: map[string][]float32{}}
	par := tensor.Parallel{Workers: cfg.Workers}
	logits := make([][]float32, len(all))
	acts := make([][]float32, len(all))
	par.Run(len(all), func(i int) {
		s := all[i]
		qt := s.it.Ex.Type.QType()
		logits[i] = h.Logits(s.rec.Markers, s.rec.K, qt)
		acts[i] = h.ActLogits(s.rec.Pooled, qt, logits[i])
	})
	var temps []tempRecord
	for i, s := range all {
		v.Logits[s.it.Ex.ID], v.ActLogits[s.it.Ex.ID] = logits[i], acts[i]
		if s.it.Role == RoleDevCal && cfg.fits(s.it) && s.it.Ex.HasGold() {
			sc := make([]float64, len(logits[i]))
			for j, x := range logits[i] {
				sc[j] = float64(x)
			}
			temps = append(temps, tempRecord{task: s.it.Ex.Task, k: s.it.Ex.NumOptions(), scores: sc, label: s.it.Ex.Label})
		}
	}
	v.Temperatures = fitTemperatures(temps, cfg.MinCell)
	var conf []thresholdRow
	for _, s := range all {
		if s.it.Role != RoleDevConf || !cfg.fits(s.it) {
			continue
		}
		r := decodeRow(s.it, v.Logits[s.it.Ex.ID], v.ActLogits[s.it.Ex.ID], v.Temperatures, nil)
		conf = append(conf, thresholdRow{task: r.Task, correct: r.Correct, maxProb: r.TopProb, actProb: r.ActProbability})
	}
	v.Thresholds = chooseThresholds(conf, cfg.Alpha, cfg.Delta, cfg.NMin, cfg.LTTStartErrors)
	for _, it := range p.Items {
		v.Rows = append(v.Rows, decodeRow(it, v.Logits[it.Ex.ID], v.ActLogits[it.Ex.ID], v.Temperatures, v.Thresholds))
	}
	v.Evaluation = evaluate(v.Rows)
	v.Generalise = generalisation(v.Rows)
	return v
}

// compareVariants measures how far the int8 head moves the served outputs,
// each head under its own calibration.
func compareVariants(a, b *Variant) *Int8Delta {
	d := &Int8Delta{Note: "each head with its own temperatures and thresholds"}
	for i, ra := range a.Rows {
		rb := b.Rows[i]
		if !ra.Scorable {
			continue
		}
		d.N++
		for j := range ra.Probs {
			d.MaxAbsProb = math.Max(d.MaxAbsProb, math.Abs(ra.Probs[j]-rb.Probs[j]))
		}
		la, lb := a.Logits[ra.ID], b.Logits[ra.ID]
		for j := range la {
			d.MaxAbsLogit = math.Max(d.MaxAbsLogit, math.Abs(float64(la[j])-float64(lb[j])))
		}
		if ra.PredKey != rb.PredKey {
			d.ArgmaxFlips++
		}
		if ra.Acted != rb.Acted {
			d.DecisionFlips++
		}
	}
	return d
}

// trainer holds the per-chunk buffers of the data-parallel gradient. A
// batch is cut into at most `chunks` contiguous chunks; each chunk's
// gradient is summed sequentially into its own buffer, and the buffers are
// reduced in chunk order, so the result does not depend on how many
// goroutines ran the chunks.
type trainer struct {
	head   *Head
	obj    Objective
	par    tensor.Parallel
	chunks int
	bufs   []*grads
	total  *grads
	work   []*work
	actBuf []*grads
	actTot *grads
}

type work struct {
	sc   []*scorerScratch
	act  *actScratch
	du   []float64
	s    []float64
	ds   []float64
	loss float64
}

const gradChunks = 8

func newTrainer(h *Head, obj Objective, workers int) *trainer {
	tr := &trainer{head: h, obj: obj, par: tensor.Parallel{Workers: workers}, chunks: gradChunks,
		total: newGrads(h.Config, false), actTot: newGrads(h.Config, true)}
	for c := 0; c < gradChunks; c++ {
		tr.bufs = append(tr.bufs, newGrads(h.Config, false))
		tr.actBuf = append(tr.actBuf, newGrads(h.Config, true))
		tr.work = append(tr.work, &work{act: newActScratch(h.Config), du: make([]float64, h.Config.D)})
	}
	return tr
}

func (w *work) ensure(c HeadConfig, k int) {
	for len(w.sc) < k {
		w.sc = append(w.sc, newScorerScratch(c))
	}
	if cap(w.s) < k {
		w.s, w.ds = make([]float64, k), make([]float64, k)
	}
	w.s, w.ds = w.s[:k], w.ds[:k]
}

// exampleGrad adds one example's scorer gradient into g and returns its
// loss.
func (tr *trainer) exampleGrad(s *sample, g *grads, w *work) float64 {
	h := tr.head
	d := h.Config.D
	k := s.rec.K
	qt := s.it.Ex.Type.QType()
	te := h.TypeEmb[int(qt)*d : (int(qt)+1)*d]
	w.ensure(h.Config, k)
	for i := 0; i < k; i++ {
		w.s[i] = h.score(s.rec.Markers[i*d:(i+1)*d], te, w.sc[i])
	}
	loss := tr.obj.Loss(s.it.Ex.Type, s.target, w.s, 1, w.ds)
	if s.weight > 0 && s.weight != 1 { // 0 is an unset weight
		loss *= s.weight
		for i := 0; i < k; i++ {
			w.ds[i] *= s.weight
		}
	}
	for i := 0; i < k; i++ {
		if w.ds[i] != 0 {
			h.backwardScore(w.ds[i], qt, w.sc[i], g, w.du)
		}
	}
	return loss
}

// scorerStep computes the mean gradient of a batch into tr.total and
// returns the batch's summed loss.
func (tr *trainer) scorerStep(batch []*sample) float64 {
	c := min(len(batch), tr.chunks)
	tr.par.Run(c, func(ci int) {
		lo, hi := ci*len(batch)/c, (ci+1)*len(batch)/c
		g, w := tr.bufs[ci], tr.work[ci]
		g.zero()
		w.loss = 0
		for _, s := range batch[lo:hi] {
			w.loss += tr.exampleGrad(s, g, w)
		}
	})
	tr.total.zero()
	var loss float64
	inv := 1 / float64(len(batch))
	for ci := 0; ci < c; ci++ {
		tr.total.addScaled(tr.bufs[ci], inv)
		loss += tr.work[ci].loss
	}
	return loss
}

// devNLL is the mean label NLL at T=1 and the accuracy over group.
func (tr *trainer) devNLL(group []*sample) (float64, float64) {
	losses := make([]float64, len(group))
	correct := make([]bool, len(group))
	tr.par.Run(len(group), func(i int) {
		s := group[i]
		sc := tr.head.Scores(s.rec.Markers, s.rec.K, s.it.Ex.Type.QType())
		losses[i] = nll(sc, 1, s.it.Ex.Label)
		correct[i] = argmax(sc) == s.it.Ex.Label
	})
	var sum float64
	n := 0
	for i, l := range losses {
		sum += l
		if correct[i] {
			n++
		}
	}
	return sum / float64(len(group)), float64(n) / float64(len(group))
}

func (tr *trainer) servedLogits(group []*sample) [][]float32 {
	out := make([][]float32, len(group))
	tr.par.Run(len(group), func(i int) {
		s := group[i]
		out[i] = tr.head.Logits(s.rec.Markers, s.rec.K, s.it.Ex.Type.QType())
	})
	return out
}

// trainAct fits the act head as a correctness predictor on dev_cal: target
// output 0 (act) when the scorer's argmax is the label, 1 (escalate)
// otherwise, with class-balanced weights so rare errors are not ignored.
// The scorer, type embedding and features are fixed. With dev_cal holding
// only one outcome the act head keeps its initialisation, and the
// thresholds then read max_probability unless the act head's AUROC wins.
func (tr *trainer) trainAct(ctx context.Context, group []*sample, logitsOf map[*sample][]float32, cfg Config,
	logf func(string, ...any)) (ActReport, error) {
	n := len(group)
	rep := ActReport{N: n, Epochs: cfg.ActEpochs, Objective: "class-balanced cross-entropy: act when the scorer's argmax is the label"}
	targets := make([]int, n)
	for i, s := range group {
		l := logitsOf[s]
		sc := make([]float64, len(l))
		for j, v := range l {
			sc[j] = float64(v)
		}
		if argmax(sc) == s.it.Ex.Label {
			rep.Positives++
		} else {
			targets[i] = 1
		}
	}
	if rep.Positives == 0 || rep.Positives == n || cfg.ActEpochs == 0 {
		rep.Reason = "dev_cal has no examples of both outcomes"
		if cfg.ActEpochs == 0 {
			rep.Reason = "act epochs is 0"
		}
		return rep, nil
	}
	weights := [2]float64{float64(n) / (2 * float64(rep.Positives)), float64(n) / (2 * float64(n-rep.Positives))}
	params := tr.head.Params()[scorerParams:]
	opt := NewAdamW(params, cfg.WeightDecay)
	const batchSize = 64
	steps := (n + batchSize - 1) / batchSize
	sched := LRSchedule{Peak: cfg.ActLR, Floor: 0.1, Warmup: steps, Total: steps * cfg.ActEpochs}
	rng := rand.New(rand.NewPCG(cfg.Seed^0x9e3779b97f4a7c15, cfg.Seed)) //nolint:gosec // G404: reproducible shuffling, not a security use
	step := 0
	var last float64
	for epoch := 0; epoch < cfg.ActEpochs; epoch++ {
		if err := ctx.Err(); err != nil {
			return rep, err
		}
		perm := rng.Perm(n)
		last = 0
		for b := 0; b < n; b += batchSize {
			if err := ctx.Err(); err != nil {
				return rep, err
			}
			idx := perm[b:min(n, b+batchSize)]
			c := min(len(idx), tr.chunks)
			tr.par.Run(c, func(ci int) {
				g, w := tr.actBuf[ci], tr.work[ci]
				g.zero()
				w.loss = 0
				for _, i := range idx[ci*len(idx)/c : (ci+1)*len(idx)/c] {
					s := group[i]
					w.loss += tr.head.actExampleGrad(s.rec.Pooled, s.it.Ex.Type.QType(), logitsOf[s], targets[i], weights[targets[i]], w.act, g)
				}
			})
			tr.actTot.zero()
			inv := 1 / float64(len(idx))
			for ci := 0; ci < c; ci++ {
				tr.actTot.addScaled(tr.actBuf[ci], inv)
				last += tr.work[ci].loss
			}
			step++
			opt.Step(tr.actTot.slices()[scorerParams:], sched.At(step))
			advance(cfg, "act", step)
		}
	}
	rep.Trained, rep.FinalLoss = true, last/float64(n)
	logf("act head: %d dev_cal items (%d correct), final weighted loss %.4f", n, rep.Positives, rep.FinalLoss)
	return rep, nil
}
