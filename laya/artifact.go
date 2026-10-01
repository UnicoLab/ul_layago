package laya

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/UnicoLab/ul_layago/model"
	"github.com/UnicoLab/ul_layago/safetensors"
	"github.com/UnicoLab/ul_layago/tensor"
	"github.com/UnicoLab/ul_layago/tokenizer"
)

// Contract identifiers a trained head directory must declare. They are
// defined in docs/architecture/intelligence-contracts.md ("Model engine and
// data") and written by tools/laya-train.
const (
	ArtifactContract     = "hive.laya-artifact/v1"
	ArtifactPromptFormat = "hive.laya-prompt/v1"
	ArtifactSequence     = "laya-build-sequence/0.3.5"
)

// Act decisions of an artifact answer.
const (
	DecisionAct      = "act"
	DecisionEscalate = "escalate"
)

// Decision score kinds a threshold can be certified on.
const (
	ScoreMaxProbability = "max_probability"
	ScoreActProbability = "act_probability"
)

// probabilityGrid is 1 / the probability quantum: scores are compared with
// thresholds as floor(p*10000+0.5)/10000, exactly as the trainer does.
const probabilityGrid = 10000

// ArtifactManifest is the part of a hive.laya-artifact/v1 manifest.json the
// runtime depends on. Training records, evaluation and limitations are kept
// in the file for review and ignored here.
type ArtifactManifest struct {
	Contract    string                  `json:"contract"`
	Instance    string                  `json:"instance"`
	Base        ArtifactBase            `json:"base"`
	Tokenizer   ArtifactTokenizer       `json:"tokenizer"`
	Sequence    ArtifactSequenceSpec    `json:"sequence"`
	Head        ArtifactHeadSpec        `json:"head"`
	Weights     ArtifactWeights         `json:"weights"`
	Tasks       map[string]ArtifactTask `json:"tasks"`
	Calibration ArtifactCalibration     `json:"calibration"`
	Act         ArtifactAct             `json:"act"`
}

// ArtifactBase pins the frozen trunk by digest; the files live in a base
// directory given at load, never in the artifact.
type ArtifactBase struct {
	ModelID       string `json:"model_id"`
	ConfigFile    string `json:"config_file"`
	ConfigSHA256  string `json:"config_sha256"`
	WeightsFile   string `json:"weights_file"`
	WeightsSHA256 string `json:"weights_sha256"`
	TensorPrefix  string `json:"tensor_prefix"`
	LayersTotal   int    `json:"layers_total"`
	LayersUsed    int    `json:"layers_used"`
	HiddenSize    int    `json:"hidden_size"`
}

// ArtifactSpecialToken is one special token with the id it had at training.
type ArtifactSpecialToken struct {
	Token string `json:"token"`
	ID    int32  `json:"id"`
}

// ArtifactTokenizer pins tokenizer.json by digest.
type ArtifactTokenizer struct {
	File          string                          `json:"file"`
	SHA256        string                          `json:"sha256"`
	SpecialTokens map[string]ArtifactSpecialToken `json:"special_tokens"`
}

// ArtifactSequenceSpec is the build_sequence layout the head was trained on.
type ArtifactSequenceSpec struct {
	Format       string `json:"format"`
	PromptFormat string `json:"prompt_format"`
	MaxLen       int    `json:"max_len"`
	HeadMaxLen   int    `json:"head_max_len"`
	TruncateLeft bool   `json:"truncate_left"`
}

// ArtifactHeadSpec is the decision head's shape.
type ArtifactHeadSpec struct {
	Layers     int      `json:"layers"`
	ActOutputs []string `json:"act_outputs"`
}

// ArtifactWeights pins the head tensors file.
type ArtifactWeights struct {
	File   string `json:"file"`
	SHA256 string `json:"sha256"`
	Format string `json:"format"`
}

// ArtifactOption is one fixed choice option.
type ArtifactOption struct {
	Key  string `json:"key"`
	Text string `json:"text"`
}

// ArtifactTask is one trained task. FixedOptions holds, when the task's
// options never varied in training, the choice options (objects), the score
// level texts or the yes_no [false, true] texts (strings).
type ArtifactTask struct {
	TaskType     string          `json:"task_type"`
	QType        string          `json:"qtype"`
	Instructions string          `json:"instructions"`
	Options      string          `json:"options"`
	LabelSet     []string        `json:"label_set"`
	FixedOptions json.RawMessage `json:"fixed_options"`
}

// ArtifactTemperature is one fitted temperature cell.
type ArtifactTemperature struct {
	T float64 `json:"t"`
	N int     `json:"n"`
}

// ArtifactCalibration holds temperatures keyed "<task>:<bucket>", "<task>"
// and "*".
type ArtifactCalibration struct {
	Temperatures map[string]ArtifactTemperature `json:"temperatures"`
}

// ArtifactThreshold is one task's act/escalate threshold. An uncertified
// threshold (or a task without one) always escalates.
type ArtifactThreshold struct {
	Score     string   `json:"score"`
	Lambda    *float64 `json:"lambda"`
	Certified bool     `json:"certified"`
}

// ArtifactAct holds the per-task thresholds.
type ArtifactAct struct {
	Thresholds map[string]ArtifactThreshold `json:"thresholds"`
}

// ArtifactOptions selects the numeric path of the trunk. The head's format
// (float32 or int8) is fixed by the artifact itself.
type ArtifactOptions struct {
	// Int8Encoder quantizes the base encoder to per-row int8 at load.
	Int8Encoder bool
}

// Artifact is a loaded, digest-verified trained head over its frozen trunk.
type Artifact struct {
	Manifest       ArtifactManifest
	ManifestSHA256 string
	Agent          *Agent
	tasks          map[string]artifactTask
}

type artifactTask struct {
	ArtifactTask
	qtype        QType
	fixedChoices []Option
	fixedTexts   []string
}

// Size bounds of the pinned files read whole.
const (
	maxArtifactConfigBytes    = 1 << 20
	maxArtifactTokenizerBytes = 256 << 20
	maxArtifactHeadBytes      = 4 << 30
)

// LoadArtifact loads a hive.laya-artifact/v1 directory over the base model
// files in baseDir. Every pinned digest (head tensors, base config, base
// weights, tokenizer) is recomputed and must match, and the special tokens
// must have the ids they had at training; any mismatch fails the load.
//
// Each pinned file is read or opened once and parsed from what was hashed:
// the head tensors, config and tokenizer from the hashed bytes, the base
// weights through the same open file they were hashed through. A file
// replaced between the check and the load is therefore never loaded; only an
// in-place rewrite of the open base weights file could be, so the base
// directory should be read-only.
func LoadArtifact(dir, baseDir string, opts ArtifactOptions) (*Artifact, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "manifest.json")) //nolint:gosec // G304: artifact directory chosen by the caller
	if err != nil {
		return nil, err
	}
	var m ArtifactManifest
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("laya: parse manifest.json: %w", err)
	}
	if err := m.validate(); err != nil {
		return nil, err
	}
	sum := sha256.Sum256(raw)
	art := &Artifact{Manifest: m, ManifestSHA256: hex.EncodeToString(sum[:]), tasks: map[string]artifactTask{}}
	for name, t := range m.Tasks {
		at, err := compileTask(name, t)
		if err != nil {
			return nil, err
		}
		art.tasks[name] = at
	}

	headRaw, err := readPinned(dir, m.Weights.File, m.Weights.SHA256, maxArtifactHeadBytes)
	if err != nil {
		return nil, err
	}
	cfgRaw, err := readPinned(baseDir, m.Base.ConfigFile, m.Base.ConfigSHA256, maxArtifactConfigBytes)
	if err != nil {
		return nil, err
	}
	tokRaw, err := readPinned(baseDir, m.Tokenizer.File, m.Tokenizer.SHA256, maxArtifactTokenizerBytes)
	if err != nil {
		return nil, err
	}
	baseFile, baseSize, err := openPinned(baseDir, m.Base.WeightsFile, m.Base.WeightsSHA256)
	if err != nil {
		return nil, err
	}
	defer func() { _ = baseFile.Close() }()

	full, err := model.ParseHFConfig(cfgRaw)
	if err != nil {
		return nil, err
	}
	if full.NumLayers != m.Base.LayersTotal || full.HiddenSize != m.Base.HiddenSize {
		return nil, fmt.Errorf("laya: base config has %d layers of %d, manifest says %d of %d",
			full.NumLayers, full.HiddenSize, m.Base.LayersTotal, m.Base.HiddenSize)
	}
	cfg, err := full.Truncate(m.Base.LayersUsed)
	if err != nil {
		return nil, err
	}
	enc, err := loadArtifactEncoder(baseFile, baseSize, cfg, m.Base.TensorPrefix, opts.Int8Encoder)
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.Load(tokRaw)
	if err != nil {
		return nil, err
	}
	sp, err := m.Tokenizer.special(tok)
	if err != nil {
		return nil, err
	}
	hf, err := safetensors.Parse(headRaw)
	if err != nil {
		return nil, err
	}
	head, err := loadHead(hf, cfg.HiddenSize, m.Head.Layers, len(m.Head.ActOutputs), "")
	if err != nil {
		return nil, err
	}
	layers := m.Head.Layers
	agent, err := NewAgent(AgentConfig{Encoder: m.Base.ModelID, HeadLayers: &layers, MaxLen: m.Sequence.MaxLen,
		HeadMaxLen: m.Sequence.HeadMaxLen}, tok, sp, enc, head)
	if err != nil {
		return nil, err
	}
	art.Agent = agent
	return art, nil
}

func loadArtifactEncoder(r io.ReaderAt, size int64, cfg model.Config, prefix string, quantize bool) (*model.Encoder, error) {
	f, err := safetensors.OpenReaderAt(r, size)
	if err != nil {
		return nil, err
	}
	w, err := model.LoadWeights(f, cfg, model.LoadOptions{Prefix: prefix, Int8: quantize, SkipDeeperLayers: true})
	if err != nil {
		return nil, err
	}
	return model.NewEncoder(cfg, w)
}

func (m *ArtifactManifest) validate() error {
	switch {
	case m.Contract != ArtifactContract:
		return fmt.Errorf("laya: manifest contract %q is not %s", m.Contract, ArtifactContract)
	case m.Instance != "Q" && m.Instance != "R" && m.Instance != "A":
		return fmt.Errorf("laya: instance %q is not Q, R or A", m.Instance)
	case m.Sequence.Format != ArtifactSequence || m.Sequence.PromptFormat != ArtifactPromptFormat:
		return fmt.Errorf("laya: sequence format %q/%q is not %s/%s", m.Sequence.Format, m.Sequence.PromptFormat, ArtifactSequence, ArtifactPromptFormat)
	case m.Sequence.TruncateLeft:
		return errors.New("laya: truncate_left artifacts are not supported")
	case m.Sequence.MaxLen <= 0 || m.Sequence.HeadMaxLen <= 0:
		return errors.New("laya: sequence limits must be positive")
	case m.Head.Layers < 0:
		return errors.New("laya: head layers must not be negative")
	case !slices.Equal(m.Head.ActOutputs, []string{DecisionAct, DecisionEscalate}):
		return fmt.Errorf("laya: act outputs %v are not [act escalate]", m.Head.ActOutputs)
	case m.Weights.Format != "float32" && m.Weights.Format != "int8":
		return fmt.Errorf("laya: weights format %q is not float32 or int8", m.Weights.Format)
	case m.Base.LayersUsed < 1 || m.Base.LayersUsed > m.Base.LayersTotal:
		return fmt.Errorf("laya: layers_used %d is outside 1..%d", m.Base.LayersUsed, m.Base.LayersTotal)
	case len(m.Tasks) == 0:
		return errors.New("laya: manifest declares no tasks")
	}
	for key, t := range m.Calibration.Temperatures {
		if math.IsNaN(t.T) || t.T < TempMin || t.T > TempMax {
			return fmt.Errorf("laya: temperature %q = %v is outside [%v, %v]", key, t.T, TempMin, TempMax)
		}
	}
	for task, th := range m.Act.Thresholds {
		if _, ok := m.Tasks[task]; !ok {
			return fmt.Errorf("laya: threshold for unknown task %q", task)
		}
		if th.Score != ScoreMaxProbability && th.Score != ScoreActProbability {
			return fmt.Errorf("laya: task %q threshold score %q is unknown", task, th.Score)
		}
		if th.Certified {
			if th.Lambda == nil || *th.Lambda < 0 || *th.Lambda > 1 ||
				math.Abs(*th.Lambda*probabilityGrid-math.Round(*th.Lambda*probabilityGrid)) > 1e-6 {
				return fmt.Errorf("laya: task %q certified threshold must be on the 1e-4 grid in [0, 1]", task)
			}
		}
	}
	return nil
}

func compileTask(name string, t ArtifactTask) (artifactTask, error) {
	at := artifactTask{ArtifactTask: t}
	switch {
	case t.TaskType == "choice" && t.QType == "choice":
		at.qtype = Choice
	case t.TaskType == "score" && t.QType == "score":
		at.qtype = Score
	case t.TaskType == "yes_no" && t.QType == "noul":
		at.qtype = Noul
	default:
		return at, fmt.Errorf("laya: task %q has task_type %q and qtype %q", name, t.TaskType, t.QType)
	}
	if len(t.FixedOptions) == 0 || string(t.FixedOptions) == "null" {
		return at, nil
	}
	var err error
	if at.qtype == Choice {
		var opts []ArtifactOption
		if err = json.Unmarshal(t.FixedOptions, &opts); err == nil {
			for _, o := range opts {
				at.fixedChoices = append(at.fixedChoices, Option{Key: o.Key, Description: o.Text})
			}
		}
	} else {
		err = json.Unmarshal(t.FixedOptions, &at.fixedTexts)
		if err == nil && at.qtype == Noul && len(at.fixedTexts) != 2 {
			err = errors.New("yes_no fixed options must be [false, true]")
		}
	}
	if err != nil {
		return at, fmt.Errorf("laya: task %q fixed_options: %w", name, err)
	}
	return at, nil
}

func (t ArtifactTokenizer) special(tok *tokenizer.Tokenizer) (SpecialTokens, error) {
	var sp SpecialTokens
	for _, item := range []struct {
		name string
		dst  *int32
	}{{"cls", &sp.CLS}, {"sep", &sp.SEP}, {"mask", &sp.Mask}, {"pad", &sp.Pad}} {
		st, ok := t.SpecialTokens[item.name]
		if !ok || st.Token == "" {
			return SpecialTokens{}, fmt.Errorf("laya: manifest names no %s token", item.name)
		}
		id, ok := tok.TokenID(st.Token)
		if !ok || id != st.ID {
			return SpecialTokens{}, fmt.Errorf("laya: %s token %q has id %d in the tokenizer, %d at training", item.name, st.Token, id, st.ID)
		}
		*item.dst = id
	}
	sp.MaskToken = t.SpecialTokens["mask"].Token
	return sp, nil
}

// pinnedPath checks that a manifest file name is local (a manifest cannot
// point outside its directory) and joins it to root.
func pinnedPath(root, name string) (string, error) {
	if name == "" || !filepath.IsLocal(name) {
		return "", fmt.Errorf("laya: manifest file name %q is not a local path", name)
	}
	return filepath.Join(root, name), nil
}

func checkDigest(name string, sum []byte, want string) error {
	if got := hex.EncodeToString(sum); !strings.EqualFold(got, want) {
		return fmt.Errorf("laya: %s has sha256 %s, manifest pins %s", name, got, want)
	}
	return nil
}

// readPinned reads root/name whole (at most limit bytes) and checks its
// sha256; the caller parses the returned bytes, the ones that were hashed.
func readPinned(root, name, want string, limit int64) ([]byte, error) {
	path, err := pinnedPath(root, name)
	if err != nil {
		return nil, err
	}
	f, err := os.Open(path) //nolint:gosec // G304: checked local name under the caller's directory
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	raw, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(raw)) > limit {
		return nil, fmt.Errorf("laya: %s is larger than %d bytes", name, limit)
	}
	sum := sha256.Sum256(raw)
	if err := checkDigest(name, sum[:], want); err != nil {
		return nil, err
	}
	return raw, nil
}

// openPinned opens root/name, streams it through sha256 and returns the open
// file, so the caller reads the contents that were verified.
func openPinned(root, name, want string) (*os.File, int64, error) {
	path, err := pinnedPath(root, name)
	if err != nil {
		return nil, 0, err
	}
	f, err := os.Open(path) //nolint:gosec // G304: checked local name under the caller's directory
	if err != nil {
		return nil, 0, err
	}
	st, err := f.Stat()
	if err == nil {
		h := sha256.New()
		if _, err = io.Copy(h, io.NewSectionReader(f, 0, st.Size())); err == nil {
			err = checkDigest(name, h.Sum(nil), want)
		}
	}
	if err != nil {
		_ = f.Close()
		return nil, 0, err
	}
	return f, st.Size(), nil
}

// ArtifactQuestion asks one trained task. For a task with fixed options
// leave Options, Levels, False and True empty (or pass exactly the fixed
// ones); a per-example choice task needs its Options.
type ArtifactQuestion struct {
	ID          string
	Task        string
	Options     []Option
	Levels      []string
	False, True string
}

// ArtifactAnswer is one calibrated answer with its act decision. It is a
// nomination: no probability here is evidence or numeric authority.
type ArtifactAnswer struct {
	Answer
	Task string
	// TemperatureKey is the calibration cell used ("" when none applied).
	TemperatureKey string
	// Decision is act or escalate; ScoreKind names the score compared with
	// Threshold (nil when the task always escalates).
	Decision      string
	DecisionScore float64
	ScoreKind     string
	Threshold     *float64
	// Unscorable marks a question whose option markers did not fit the
	// sequence limits: it is escalated on its own, with no probabilities,
	// exactly as the trainer escalates it, and never fails the batch.
	Unscorable bool
}

// Question builds the typed question for a task with the instructions and
// type it was trained with.
func (a *Artifact) Question(q ArtifactQuestion) (Question, error) {
	t, ok := a.tasks[q.Task]
	if !ok {
		return Question{}, fmt.Errorf("laya: the artifact has no task %q", q.Task)
	}
	out := Question{ID: q.ID, Type: t.qtype, Instructions: t.Instructions}
	switch t.qtype {
	case Choice:
		out.Options = q.Options
		if out.Options == nil {
			out.Options = t.fixedChoices
		} else if t.fixedChoices != nil && !slices.Equal(out.Options, t.fixedChoices) {
			return Question{}, fmt.Errorf("laya: task %q was trained on fixed options", q.Task)
		}
		if len(out.Options) == 0 {
			return Question{}, fmt.Errorf("laya: task %q needs options", q.Task)
		}
	case Score:
		out.Levels = q.Levels
		if out.Levels == nil {
			out.Levels = t.fixedTexts
		} else if t.fixedTexts != nil && !slices.Equal(out.Levels, t.fixedTexts) {
			return Question{}, fmt.Errorf("laya: task %q was trained on fixed levels", q.Task)
		}
		if len(out.Levels) == 0 {
			return Question{}, fmt.Errorf("laya: task %q needs levels", q.Task)
		}
	default:
		out.False, out.True = q.False, q.True
		if t.fixedTexts != nil {
			if out.False == "" && out.True == "" {
				out.False, out.True = t.fixedTexts[0], t.fixedTexts[1]
			} else if out.False != t.fixedTexts[0] || out.True != t.fixedTexts[1] {
				return Question{}, fmt.Errorf("laya: task %q was trained on fixed false/true texts", q.Task)
			}
		}
	}
	return out, nil
}

// Temperature returns the calibrated temperature for a task with k options
// and the cell it came from: "<task>:<bucket>", then "<task>", then "*",
// else 1.0 with an empty key. It is clamped like every Laya temperature.
func (a *Artifact) Temperature(task string, k int) (float64, string) {
	for _, key := range []string{task + ":" + optionBucket(k), task, "*"} {
		if t, ok := a.Manifest.Calibration.Temperatures[key]; ok {
			return ClampTemperature(t.T), key
		}
	}
	return 1, ""
}

// optionBucket is the size half of TempBucket: 2, 3-5, 6-10 or 11+.
func optionBucket(k int) string {
	_, size, _ := strings.Cut(TempBucket(Choice, k), ":")
	return size
}

// Predict answers questions over one serialised state in one batched pass,
// applying the artifact's temperatures and act thresholds. The returned
// Prediction holds every question's sequence and markers in question order.
// An unscorable question (its option markers cut by the sequence limits) is
// escalated on its own with a zero Answer, as the trainer escalates it; the
// other questions still run.
func (a *Artifact) Predict(ctx context.Context, state string, qs []ArtifactQuestion, opt model.Options) ([]ArtifactAnswer, *Prediction, error) {
	type asked struct {
		q    Question
		task string
	}
	items := make([]asked, 0, len(qs))
	for _, aq := range qs {
		built, err := a.Question(aq)
		if err != nil {
			return nil, nil, err
		}
		items = append(items, asked{q: built, task: aq.Task})
	}
	pred := &Prediction{Answers: make([]Answer, len(items)), Sequences: make([][]int32, len(items)), Markers: make([][]int, len(items))}
	out := make([]ArtifactAnswer, len(items))
	var scored []int
	for i, it := range items {
		ids, markers, err := BuildSequence(a.Agent.Tokenizer, a.Agent.Special, state, it.q, a.Agent.Config.MaxLen, a.Agent.Config.HeadMaxLen, nil, false)
		if err != nil {
			return nil, nil, err
		}
		pred.Sequences[i], pred.Markers[i] = ids, markers
		if len(markers) == len(RenderOptions(it.q)) {
			scored = append(scored, i)
			continue
		}
		pred.Answers[i] = Answer{ID: it.q.ID, Type: it.q.Type}
		out[i] = ArtifactAnswer{Answer: pred.Answers[i], Task: it.task, Decision: DecisionEscalate, Unscorable: true}
	}
	if len(scored) == 0 {
		return out, pred, nil
	}
	sub := make([]Question, len(scored))
	for j, i := range scored {
		sub[j] = items[i].q
	}
	res, err := a.Agent.Predict(ctx, state, sub, opt)
	if err != nil {
		return nil, nil, err
	}
	if len(res.Answers) != len(scored) {
		return nil, nil, fmt.Errorf("laya: %d answers for %d questions", len(res.Answers), len(scored))
	}
	pred.InputTokens = res.InputTokens
	for j, i := range scored {
		raw, it := res.Answers[j], items[i]
		pred.Answers[i] = raw
		t, key := a.Temperature(it.task, len(raw.Logits))
		ans := ArtifactAnswer{Answer: decodeLogits(it.q, raw.Logits, raw.ActLogits, t), Task: it.task, TemperatureKey: key}
		ans.Decision, ans.DecisionScore, ans.ScoreKind, ans.Threshold = a.decide(it.task, ans.Answer)
		out[i] = ans
	}
	return out, pred, nil
}

// decide applies the task's certified threshold to the quantised score.
func (a *Artifact) decide(task string, ans Answer) (decision string, score float64, kind string, threshold *float64) {
	th, ok := a.Manifest.Act.Thresholds[task]
	kind = ScoreMaxProbability
	if ok && th.Score == ScoreActProbability {
		kind = ScoreActProbability
	}
	score = ans.ActProbability
	if kind == ScoreMaxProbability {
		score = slices.Max(ans.Probs)
	}
	if !ok || !th.Certified || th.Lambda == nil {
		return DecisionEscalate, score, kind, nil
	}
	if QuantizeProbability(score) >= *th.Lambda-1e-12 {
		return DecisionAct, score, kind, th.Lambda
	}
	return DecisionEscalate, score, kind, th.Lambda
}

// QuantizeProbability rounds p to the 1e-4 grid as floor(p*10000+0.5)/10000,
// the trainer's expression, so both sides compare the same number.
func QuantizeProbability(p float64) float64 {
	return math.Floor(p*probabilityGrid+0.5) / probabilityGrid
}

// Prepack builds the packed GEMM panels for kernel k (see Agent.Prepack).
func (a *Artifact) Prepack(k tensor.Kernel) { a.Agent.Prepack(k) }
