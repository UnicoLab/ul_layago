package train

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"

	"github.com/UnicoLab/ul_layago/laya"
	"github.com/UnicoLab/ul_layago/safetensors"
	"github.com/UnicoLab/ul_layago/tensor"
)

// Artifact is a hive.laya-artifact/v1 directory loaded by the trainer's own
// runtime, which computes the head exactly as training did (float64 over
// float32 states) and so reproduces the trainer's logits bit for bit on the
// feature kernel. Hive serves the same directory with laya.LoadArtifact
// (float32 GEMM), within the golden's tolerance.
type Artifact struct {
	Dir      string
	Manifest Manifest
	Head     *Head
}

// LoadArtifact reads an artifact directory, verifies the head weights'
// digest and loads the head (int8 linear weights dequantised exactly as
// QuantizedHead does).
func LoadArtifact(dir string) (*Artifact, error) {
	raw, err := os.ReadFile(filepath.Join(dir, ArtifactManifest)) //nolint:gosec // G304: artifact directory chosen by the operator
	if err != nil {
		return nil, err
	}
	var m Manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("train: manifest: %w", err)
	}
	switch {
	case m.Contract != ArtifactContract:
		return nil, fmt.Errorf("train: manifest contract %q is not %s", m.Contract, ArtifactContract)
	case m.Sequence.Format != SequenceFormat || m.Sequence.PromptFormat != PromptFormat || m.Sequence.TruncateLeft:
		return nil, fmt.Errorf("train: sequence %+v is not %s/%s", m.Sequence, SequenceFormat, PromptFormat)
	case m.Sequence.Shortlist != nil && (m.Sequence.Shortlist.K <= 0 || m.Sequence.Shortlist.Method != ShortlistMethod):
		return nil, fmt.Errorf("train: catalog shortlist %+v is not %s with k > 0", *m.Sequence.Shortlist, ShortlistMethod)
	case m.Head.Layers != 0:
		return nil, fmt.Errorf("train: this runtime serves heads with zero head layers, the artifact has %d (use laya.LoadArtifact)", m.Head.Layers)
	case len(bytes.TrimSpace(m.Adapter)) > 0 && string(bytes.TrimSpace(m.Adapter)) != "null":
		return nil, fmt.Errorf("train: an adapter is declared; hive.laya-artifact/v1 heads run over the frozen base only")
	case m.Weights.Format != "float32" && m.Weights.Format != "int8":
		return nil, fmt.Errorf("train: weights format %q is not float32 or int8", m.Weights.Format)
	}
	for name, t := range m.Tasks {
		if t == nil {
			return nil, fmt.Errorf("train: task %q has a null specification", name)
		}
		t.Name = name
	}
	if !filepath.IsLocal(m.Weights.File) || !filepath.IsLocal(m.ParityGolden.File) {
		return nil, fmt.Errorf("train: manifest file names %q and %q must be local paths", m.Weights.File, m.ParityGolden.File)
	}
	wpath := filepath.Join(dir, m.Weights.File)
	wraw, err := os.ReadFile(wpath) //nolint:gosec // G304: inside the artifact directory
	if err != nil {
		return nil, err
	}
	sum := sha256.Sum256(wraw)
	if got := hex.EncodeToString(sum[:]); got != m.Weights.SHA256 {
		return nil, fmt.Errorf("train: %s sha256 %s, manifest pins %s", m.Weights.File, got, m.Weights.SHA256)
	}
	f, err := safetensors.Parse(wraw)
	if err != nil {
		return nil, err
	}
	cfg := HeadConfig{D: m.Base.HiddenSize, Hidden: m.Head.ScorerHidden, ActHidden: m.Head.ActHidden, NAct: len(m.Head.ActOutputs)}
	h, err := NewHead(cfg, 0)
	if err != nil {
		return nil, err
	}
	used := map[string]bool{}
	for _, p := range h.Params() {
		t, err := f.Tensor(p.Name)
		if err != nil || !slices.Equal(t.Shape, p.Shape) {
			return nil, fmt.Errorf("train: artifact tensor %s missing or not shaped %v", p.Name, p.Shape)
		}
		used[p.Name] = true
		var v []float32
		if t.DType == safetensors.I8 {
			if !isLinearWeight(p) {
				return nil, fmt.Errorf("train: %s is int8 but not a linear weight", p.Name)
			}
			q, err := t.Int8()
			if err != nil {
				return nil, err
			}
			st, err := f.Tensor(p.Name + "_scale")
			if err != nil || !slices.Equal(st.Shape, p.Shape[:1]) {
				return nil, fmt.Errorf("train: int8 tensor %s has no per-row scale", p.Name)
			}
			used[p.Name+"_scale"] = true
			scale, err := st.Float32()
			if err != nil {
				return nil, err
			}
			cols := p.Shape[1]
			v = make([]float32, len(q))
			for r := 0; r < p.Shape[0]; r++ {
				for c := 0; c < cols; c++ {
					v[r*cols+c] = float32(q[r*cols+c]) * scale[r]
				}
			}
		} else if v, err = t.Float32(); err != nil {
			return nil, err
		}
		copy(p.Data, v)
	}
	for _, name := range f.Header.Names() {
		if !used[name] {
			return nil, fmt.Errorf("train: unexpected artifact tensor %s", name)
		}
	}
	return &Artifact{Dir: dir, Manifest: m, Head: h}, nil
}

// Predictor serves an artifact end to end: hive.laya-prompt/v1 state,
// Laya sequence, frozen trunk, head, temperature, act decision.
type Predictor struct {
	Trunk    *Trunk
	Artifact *Artifact
	Parallel tensor.Parallel
	// KernelMatches is false when this process computes trunk states with
	// another kernel than the artifact was trained on (the assembly kernels
	// match the portable one bit for bit on arm64, not on amd64), so the
	// logits then agree within tolerance only.
	KernelMatches bool
}

// NewPredictor binds an artifact to a trunk, refusing a trunk whose weights,
// config, tokenizer, special tokens, depth or sequence limits differ from
// the ones the head was trained on.
func NewPredictor(t *Trunk, a *Artifact, par tensor.Parallel) (*Predictor, error) {
	want, got := a.Manifest.Base, t.Identity
	type pair struct {
		name     string
		got, exp any
	}
	for _, p := range []pair{
		{"weights_sha256", got.WeightsSHA256, want.WeightsSHA256}, {"config_sha256", got.ConfigSHA256, want.ConfigSHA256},
		{"tokenizer_sha256", got.TokenizerSHA256, a.Manifest.Tokenizer.SHA256}, {"layers_used", got.LayersUsed, want.LayersUsed},
		{"hidden_size", got.HiddenSize, want.HiddenSize}, {"numeric_path", got.NumericPath, "fp32"},
		{"max_len", t.Spec.MaxLen, a.Manifest.Sequence.MaxLen}, {"head_max_len", t.Spec.HeadMaxLen, a.Manifest.Sequence.HeadMaxLen},
	} {
		if p.got != p.exp {
			return nil, fmt.Errorf("train: trunk %s is %v, the artifact was trained with %v", p.name, p.got, p.exp)
		}
	}
	for name, st := range t.SpecialTokenManifest() {
		if w, ok := a.Manifest.Tokenizer.SpecialTokens[name]; !ok || w != st {
			return nil, fmt.Errorf("train: special token %s is %+v, the artifact was trained with %+v", name, st, w)
		}
	}
	return &Predictor{Trunk: t, Artifact: a, Parallel: par, KernelMatches: kernelName(par) == a.Manifest.Head.FeatureKernel}, nil
}

// Prediction is one served answer. It is a nomination: probabilities from a
// head of the synthetic_engineering tier are not evidence.
type Prediction struct {
	ID             string    `json:"id"`
	Task           string    `json:"task"`
	Keys           []string  `json:"keys"`
	Unscorable     bool      `json:"unscorable,omitempty"`
	Logits         []float32 `json:"logits"`
	ActLogits      []float32 `json:"act_logits"`
	Temperature    float64   `json:"temperature"`
	TemperatureKey string    `json:"temperature_key"`
	Probabilities  []float64 `json:"probabilities"`
	Prediction     string    `json:"prediction"`
	MaxProbability float64   `json:"max_probability"`
	ActProbability float64   `json:"act_probability"`
	ScoreKind      string    `json:"score_kind"`
	DecisionScore  float64   `json:"decision_score"`
	Threshold      *float64  `json:"threshold"`
	Decision       string    `json:"decision"`
	Tokens         int       `json:"tokens"`
	EvidenceTier   string    `json:"evidence_tier"`
	IDs            []int32   `json:"-"`
	Markers        []int     `json:"-"`
}

// Predict answers examples of the artifact's instance (labels ignored) in
// one stacked trunk pass. A task the artifact was trained with fixed options
// must be asked with exactly those options. An unscorable example (option
// markers cut by the sequence limits) escalates on its own.
func (p *Predictor) Predict(ctx context.Context, exs []Example) ([]Prediction, error) {
	m := &p.Artifact.Manifest
	items := make([]*Item, len(exs))
	var seqs [][]int32
	var markers [][]int
	var scored []int
	for i := range exs {
		ex := &exs[i]
		spec, ok := m.Tasks[ex.Task]
		if !ok || ex.Instance != m.Instance {
			return nil, fmt.Errorf("train: %s: the artifact has no %s task %q", ex.ID, ex.Instance, ex.Task)
		}
		if err := spec.checkOptions(ex); err != nil {
			return nil, fmt.Errorf("train: %s: %w", ex.ID, err)
		}
		it, err := layOut(ex, spec, p.Trunk, m.Sequence.Shortlist)
		if err != nil {
			return nil, err
		}
		items[i] = it
		if it.Scorable {
			scored = append(scored, i)
			seqs, markers = append(seqs, it.IDs), append(markers, it.Markers)
		}
	}
	var feats []Features
	if len(seqs) > 0 {
		var err error
		if feats, err = p.Trunk.Features(ctx, seqs, markers, p.Parallel); err != nil {
			return nil, err
		}
	}
	out := make([]Prediction, len(exs))
	for i, it := range items {
		out[i] = Prediction{ID: it.Ex.ID, Task: it.Ex.Task, Keys: it.Ex.ClassKeys(), Unscorable: !it.Scorable, Decision: "escalate",
			Tokens: len(it.IDs), EvidenceTier: m.EvidenceTier, IDs: it.IDs, Markers: it.Markers}
	}
	for j, i := range scored {
		it := items[i]
		qt := it.Ex.Type.QType()
		logits := p.Artifact.Head.Logits(feats[j].Markers, feats[j].K, qt)
		act := p.Artifact.Head.ActLogits(feats[j].Pooled, qt, logits)
		r := decodeRow(it, logits, act, m.Calibration.Temperatures, m.Act.Thresholds)
		pr := &out[i]
		pr.Logits, pr.ActLogits, pr.Temperature, pr.TemperatureKey = logits, act, r.Temperature, r.TemperatureKey
		pr.Probabilities, pr.Prediction, pr.MaxProbability, pr.ActProbability = r.Probs, r.PredKey, r.TopProb, r.ActProbability
		pr.DecisionScore, pr.ScoreKind = r.DecisionScore, ScoreMaxProbability
		if th, ok := m.Act.Thresholds[it.Ex.Task]; ok {
			pr.ScoreKind = th.Score
			if th.Certified {
				pr.Threshold = th.Lambda
			}
		}
		if r.Acted {
			pr.Decision = "act"
		}
	}
	return out, nil
}

// checkOptions refuses an example of a fixed-option task whose options are
// not the trained ones.
func (s *TaskSpec) checkOptions(ex *Example) error {
	if s.Options != "fixed" {
		return nil
	}
	var want any
	switch ex.Type {
	case TaskChoice:
		want = ex.Options
	default:
		texts := make([]string, ex.NumOptions())
		for i, o := range ex.Options {
			texts[i] = o.Text
		}
		want = texts
	}
	got, _ := json.Marshal(want)
	var a, b any
	if json.Unmarshal(got, &a) != nil || json.Unmarshal(s.FixedOptions, &b) != nil || !jsonEqual(a, b) {
		return fmt.Errorf("task %s was trained on fixed options %s, asked with %s", s.Name, s.FixedOptions, got)
	}
	return nil
}

func jsonEqual(a, b any) bool {
	x, _ := json.Marshal(a)
	y, _ := json.Marshal(b)
	return bytes.Equal(x, y)
}

// GoldenReport is the outcome of VerifyGolden.
type GoldenReport struct {
	Cases            int     `json:"cases"`
	Exact            int     `json:"bit_identical"`
	MaxAbsLogitDelta float64 `json:"max_abs_logit_delta"`
	MaxAbsProbDelta  float64 `json:"max_abs_prob_delta"`
	DecisionFlips    int     `json:"decision_flips"`
	PredictionFlips  int     `json:"prediction_flips"`
	KernelMatches    bool    `json:"kernel_matches_training"`
}

// ReadGolden reads and pins an artifact's parity_golden.json.
func ReadGolden(a *Artifact) (*Golden, error) {
	info := a.Manifest.ParityGolden
	raw, err := os.ReadFile(filepath.Join(a.Dir, info.File)) //nolint:gosec // G304: inside the artifact directory
	if err != nil {
		return nil, err
	}
	sum := sha256.Sum256(raw)
	if got := hex.EncodeToString(sum[:]); got != info.SHA256 {
		return nil, fmt.Errorf("train: %s sha256 %s, manifest pins %s", info.File, got, info.SHA256)
	}
	var g Golden
	if err := json.Unmarshal(raw, &g); err != nil {
		return nil, err
	}
	if g.Contract != ArtifactContract || len(g.Cases) != info.Cases {
		return nil, fmt.Errorf("train: %s is malformed", info.File)
	}
	return &g, nil
}

// question rebuilds the Laya question a golden case was computed on.
func (q GoldenQuestion) question(id string) (laya.Question, []string, error) {
	out := laya.Question{ID: id, Instructions: q.Instructions}
	var keys []string
	switch q.Type {
	case "choice":
		out.Type = laya.Choice
		for _, o := range q.Options {
			out.Options = append(out.Options, laya.Option{Key: o.Key, Description: o.Description})
			keys = append(keys, o.Key)
		}
	case "score":
		out.Type, out.Levels = laya.Score, q.Levels
		for i := range q.Levels {
			keys = append(keys, strconv.Itoa(i))
		}
	case "noul":
		out.Type = laya.Noul
		if q.False != nil {
			out.False = *q.False
		}
		if q.True != nil {
			out.True = *q.True
		}
		keys = []string{"false", "true"}
	default:
		return out, nil, fmt.Errorf("train: golden %s has question type %q", id, q.Type)
	}
	return out, keys, nil
}

// VerifyGolden re-serves parity_golden.json from its recorded states and
// questions: token ids and markers must match exactly, logits within the
// golden's tolerance (bit for bit on the feature kernel), and the
// temperature cell, the decision (except within one quantum of a threshold)
// and the prediction (except at a near tie) must agree.
func (p *Predictor) VerifyGolden(ctx context.Context) (*GoldenReport, error) {
	g, err := ReadGolden(p.Artifact)
	if err != nil {
		return nil, err
	}
	m := &p.Artifact.Manifest
	a := p.Trunk
	seqs := make([][]int32, len(g.Cases))
	markers := make([][]int, len(g.Cases))
	keys := make([][]string, len(g.Cases))
	for i, c := range g.Cases {
		q, k, err := c.Question.question(c.ID)
		if err != nil {
			return nil, err
		}
		ids, mk, err := laya.BuildSequence(a.Tokenizer, a.Special, c.State, q, a.Spec.MaxLen, a.Spec.HeadMaxLen, nil, false)
		if err != nil {
			return nil, err
		}
		if !slices.Equal(ids, c.IDs) || !slices.Equal(mk, c.Markers) {
			return nil, fmt.Errorf("train: golden %s: token ids or markers differ from training", c.ID)
		}
		seqs[i], markers[i], keys[i] = ids, mk, k
	}
	feats, err := a.Features(ctx, seqs, markers, p.Parallel)
	if err != nil {
		return nil, err
	}
	rep := &GoldenReport{Cases: len(g.Cases), KernelMatches: p.KernelMatches}
	for i, c := range g.Cases {
		spec, ok := m.Tasks[c.Task]
		if !ok {
			return nil, fmt.Errorf("train: golden %s: unknown task %s", c.ID, c.Task)
		}
		qt := spec.Type.QType()
		logits := p.Artifact.Head.Logits(feats[i].Markers, feats[i].K, qt)
		act := p.Artifact.Head.ActLogits(feats[i].Pooled, qt, logits)
		exact := len(logits) == len(c.Logits) && len(act) == len(c.ActLogits)
		for j := range logits {
			exact = exact && logits[j] == float32(c.Logits[j])
			rep.MaxAbsLogitDelta = math.Max(rep.MaxAbsLogitDelta, math.Abs(float64(logits[j])-c.Logits[j]))
		}
		for j := range act {
			exact = exact && act[j] == float32(c.ActLogits[j])
			rep.MaxAbsLogitDelta = math.Max(rep.MaxAbsLogitDelta, math.Abs(float64(act[j])-c.ActLogits[j]))
		}
		if exact {
			rep.Exact++
		}
		t, key := m.Calibration.Temperatures.Lookup(c.Task, len(logits))
		if key != c.TemperatureKey || t != c.Temperature {
			return nil, fmt.Errorf("train: golden %s: temperature %v (%q), trained %v (%q)", c.ID, t, key, c.Temperature, c.TemperatureKey)
		}
		probs := ServedProbabilities(logits, t)
		for j := range probs {
			rep.MaxAbsProbDelta = math.Max(rep.MaxAbsProbDelta, math.Abs(probs[j]-c.Probs[j]))
		}
		actP := ActProbability(act)
		rep.MaxAbsProbDelta = math.Max(rep.MaxAbsProbDelta, math.Abs(actP-c.ActProbability))
		th := m.Act.Thresholds[c.Task]
		score := slices.Max(probs)
		if th.Score == ScoreActProbability {
			score = actP
		}
		decision := "escalate"
		if th.Acts(score) {
			decision = "act"
		}
		nearThreshold := th.Lambda != nil && math.Abs(QuantizeProbability(score)-*th.Lambda) <= ProbabilityQuantum
		if decision != c.Decision && !nearThreshold {
			rep.DecisionFlips++
		}
		sorted := slices.Clone(c.Probs)
		slices.Sort(sorted)
		if keys[i][argmax(probs)] != c.Prediction && sorted[len(sorted)-1]-sorted[len(sorted)-2] > 1e-4 {
			rep.PredictionFlips++
		}
	}
	if rep.MaxAbsLogitDelta > g.Tolerance.LogitAbs || rep.MaxAbsProbDelta > g.Tolerance.ProbAbs || rep.DecisionFlips > 0 || rep.PredictionFlips > 0 {
		return rep, fmt.Errorf("train: golden logits within %.2e (tolerance %.0e), probabilities within %.2e (tolerance %.0e), "+
			"%d decision and %d prediction flips", rep.MaxAbsLogitDelta, g.Tolerance.LogitAbs, rep.MaxAbsProbDelta, g.Tolerance.ProbAbs,
			rep.DecisionFlips, rep.PredictionFlips)
	}
	return rep, nil
}
