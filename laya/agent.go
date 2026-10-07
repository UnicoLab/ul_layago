package laya

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/UnicoLab/ul_layago/model"
	"github.com/UnicoLab/ul_layago/safetensors"
	"github.com/UnicoLab/ul_layago/tensor"
	"github.com/UnicoLab/ul_layago/tokenizer"
)

// Temperature clamp bounds from laya/common.py: a fitted temperature below
// 0.5 would sharpen a coin flip into a certainty, so it is never applied.
const (
	TempMin = 0.5
	TempMax = 5.0
)

// ClampTemperature confines t to [TempMin, TempMax]; NaN and infinities
// become 1.0 (clamp_temperature).
func ClampTemperature(t float64) float64 {
	if math.IsNaN(t) || math.IsInf(t, 0) {
		return 1
	}
	return math.Min(TempMax, math.Max(TempMin, t))
}

// TempBucket is laya/common.py temp_bucket: "<type>:<size>" with size one of
// 2, 3-5, 6-10, 11+.
func TempBucket(q QType, k int) string {
	size := "11+"
	switch {
	case k <= 2:
		size = "2"
	case k <= 5:
		size = "3-5"
	case k <= 10:
		size = "6-10"
	}
	return q.String() + ":" + size
}

// AgentConfig is rl_agent_config.json.
type AgentConfig struct {
	Encoder              string             `json:"encoder"`
	HeadLayers           *int               `json:"head_layers"`
	MaxLen               int                `json:"max_len"`
	HeadMaxLen           int                `json:"head_max_len"`
	ActCosts             map[string]float64 `json:"act_costs"`
	Temperature          []float64          `json:"temperature"`
	TemperatureByOptions map[string]float64 `json:"temperature_by_options"`
}

// LoadOptions selects the numeric path.
type LoadOptions struct {
	// Int8 quantizes the encoder's embeddings and linear layers to per-row
	// int8. The decision head stays float32.
	Int8 bool
}

// Agent is a loaded Laya checkpoint.
type Agent struct {
	Config    AgentConfig
	Tokenizer *tokenizer.Tokenizer
	Special   SpecialTokens
	Encoder   *model.Encoder
	Head      *Head

	artifactBase *VerifiedBase
	temps        [3]float64
	tempsByOpt   map[string]float64
}

// LoadCheckpoint reads a Laya checkpoint directory in the published layout:
// rl_agent_config.json, model.safetensors, tokenizer/{tokenizer.json,
// tokenizer_config.json} and encoder/config.json.
func LoadCheckpoint(dir string, opts LoadOptions) (*Agent, error) {
	var cfg AgentConfig
	if err := readJSON(filepath.Join(dir, "rl_agent_config.json"), &cfg); err != nil {
		return nil, err
	}
	if cfg.Encoder == "" || cfg.HeadLayers == nil {
		return nil, errors.New("laya: rl_agent_config.json must declare encoder and head_layers")
	}
	if cfg.MaxLen == 0 {
		cfg.MaxLen = 512
	}
	if cfg.HeadMaxLen == 0 {
		cfg.HeadMaxLen = 192
	}
	encRaw, err := os.ReadFile(filepath.Join(dir, "encoder", "config.json")) //nolint:gosec // G304: checkpoint directory chosen by the caller
	if err != nil {
		return nil, err
	}
	encCfg, err := model.ParseHFConfig(encRaw)
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.LoadFile(filepath.Join(dir, "tokenizer", "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	sp, err := LoadSpecialTokens(filepath.Join(dir, "tokenizer", "tokenizer_config.json"), tok)
	if err != nil {
		return nil, err
	}
	f, err := safetensors.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	w, err := model.LoadWeights(f, encCfg, model.LoadOptions{Prefix: "encoder.", Int8: opts.Int8})
	if err != nil {
		return nil, err
	}
	enc, err := model.NewEncoder(encCfg, w)
	if err != nil {
		return nil, err
	}
	head, err := loadHead(f, encCfg.HiddenSize, *cfg.HeadLayers, len(cfg.ActCosts)+1, "encoder.")
	if err != nil {
		return nil, err
	}
	return NewAgent(cfg, tok, sp, enc, head)
}

// NewAgent assembles an agent and clamps its temperatures once.
func NewAgent(cfg AgentConfig, tok *tokenizer.Tokenizer, sp SpecialTokens, enc *model.Encoder, head *Head) (*Agent, error) {
	a := &Agent{Config: cfg, Tokenizer: tok, Special: sp, Encoder: enc, Head: head, temps: [3]float64{1, 1, 1}}
	if len(cfg.Temperature) != 0 {
		if len(cfg.Temperature) != 3 {
			return nil, fmt.Errorf("laya: temperature must have 3 entries, got %d", len(cfg.Temperature))
		}
		for i, t := range cfg.Temperature {
			a.temps[i] = ClampTemperature(t)
		}
	}
	a.tempsByOpt = make(map[string]float64, len(cfg.TemperatureByOptions))
	for k, t := range cfg.TemperatureByOptions {
		a.tempsByOpt[k] = ClampTemperature(t)
	}
	return a, nil
}

func readJSON(path string, v any) error {
	data, err := os.ReadFile(path) //nolint:gosec // G304: checkpoint directory chosen by the caller
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, v); err != nil {
		return fmt.Errorf("laya: parse %s: %w", filepath.Base(path), err)
	}
	return nil
}

// LoadSpecialTokens resolves the cls, sep, mask and pad tokens a Hugging Face
// tokenizer_config.json names against tok.
func LoadSpecialTokens(path string, tok *tokenizer.Tokenizer) (SpecialTokens, error) {
	var tc struct {
		CLS  string `json:"cls_token"`
		SEP  string `json:"sep_token"`
		Mask string `json:"mask_token"`
		Pad  string `json:"pad_token"`
	}
	if err := readJSON(path, &tc); err != nil {
		return SpecialTokens{}, err
	}
	var sp SpecialTokens
	for _, item := range []struct {
		name string
		dst  *int32
	}{{tc.CLS, &sp.CLS}, {tc.SEP, &sp.SEP}, {tc.Mask, &sp.Mask}, {tc.Pad, &sp.Pad}} {
		id, ok := tok.TokenID(item.name)
		if item.name == "" || !ok {
			return SpecialTokens{}, fmt.Errorf("laya: special token %q is not in the tokenizer", item.name)
		}
		*item.dst = id
	}
	sp.MaskToken = tc.Mask
	return sp, nil
}

// Answer is one decoded question. Probabilities follow option order.
type Answer struct {
	ID    string
	Type  QType
	Keys  []string // choice keys, score levels "0".."k-1", or ["false", "true"]
	Probs []float64
	// Choice is the argmax key (choice), Score the expected ordinal (score)
	// and Noul P(true) (noul).
	Choice string
	Score  float64
	Noul   float64
	// Confidence is 1 - H(p)/log(k) for choice and score, max(p, 1-p) for noul.
	Confidence     float64
	ActProbability float64
	// Logits are the raw marker scores before temperature scaling.
	Logits []float32
	// ActLogits are the act head's raw outputs (index 0 is act).
	ActLogits   []float32
	Temperature float64
}

// Prediction is the result of one Predict call.
type Prediction struct {
	Answers     []Answer
	InputTokens int
	Sequences   [][]int32
	Markers     [][]int
}

// Prepack builds the packed GEMM panels of the encoder and the head for
// kernel k before the first Predict (see tensor.Linear.Prepack).
func (a *Agent) Prepack(k tensor.Kernel) {
	a.Encoder.Weights.Prepack(k)
	a.Head.Prepack(k)
}

// Predict evaluates every question over state in one batched pass
// (Agent.system_one). The state must already be serialised text: Laya
// passes dicts through json.dumps(state, ensure_ascii=False).
func (a *Agent) Predict(ctx context.Context, state string, qs []Question, opt model.Options) (*Prediction, error) {
	if len(qs) == 0 {
		return nil, errors.New("laya: no questions")
	}
	pred := &Prediction{}
	qtypes := make([]QType, len(qs))
	for i, q := range qs {
		if q.Type != Choice && q.Type != Score && q.Type != Noul {
			return nil, fmt.Errorf("laya: question %q has unknown type %d", q.ID, q.Type)
		}
		if n := len(RenderOptions(q)); n == 0 {
			return nil, fmt.Errorf("laya: question %q has no options", q.ID)
		}
		ids, markers, err := BuildSequence(a.Tokenizer, a.Special, state, q, a.Config.MaxLen, a.Config.HeadMaxLen, nil, false)
		if err != nil {
			return nil, err
		}
		if len(markers) != len(RenderOptions(q)) {
			return nil, fmt.Errorf("laya: question %q options exceed head_max_len=%d", q.ID, a.Config.HeadMaxLen)
		}
		pred.Sequences = append(pred.Sequences, ids)
		pred.Markers = append(pred.Markers, markers)
		pred.InputTokens += len(ids)
		qtypes[i] = q.Type
	}
	var res *model.Result
	var err error
	if a.artifactBase != nil {
		res, err = a.artifactBase.forward(ctx, pred.Sequences, opt)
	} else {
		res, err = a.Encoder.Forward(ctx, pred.Sequences, opt)
	}
	if err != nil {
		return nil, err
	}
	items := make([]headInput, len(qs))
	for i := range items {
		items[i] = headInput{hidden: res.Hidden[i], qtype: qtypes[i], markers: pred.Markers[i]}
	}
	outs := a.Head.forward(items, opt.Parallel)
	for i, q := range qs {
		pred.Answers = append(pred.Answers, a.decode(q, outs[i]))
	}
	return pred, nil
}

func (a *Agent) decode(q Question, out headOutput) Answer {
	k := len(out.logits)
	t, ok := a.tempsByOpt[TempBucket(q.Type, k)]
	if !ok {
		t = a.temps[q.Type]
	}
	return decodeLogits(q, out.logits, out.actLogits, t)
}

// decodeLogits applies temperature t to raw marker logits and decodes them
// by question type.
func decodeLogits(q Question, logits, actLogits []float32, t float64) Answer {
	z := make([]float64, len(logits))
	for i, v := range logits {
		z[i] = float64(v) / t
	}
	p := tensor.Softmax64(z)
	act := make([]float64, len(actLogits))
	for i, v := range actLogits {
		act[i] = float64(v)
	}
	ans := Answer{ID: q.ID, Type: q.Type, Probs: p, Logits: logits, ActLogits: actLogits, Temperature: t,
		ActProbability: tensor.Softmax64(act)[0]}
	switch q.Type {
	case Choice:
		best := 0
		for i := range p {
			if p[i] > p[best] {
				best = i
			}
		}
		for _, o := range q.Options {
			ans.Keys = append(ans.Keys, o.Key)
		}
		ans.Choice = ans.Keys[best]
		ans.Confidence = entropyConfidence(p)
	case Score:
		for i, v := range p {
			ans.Score += float64(i) * v
			ans.Keys = append(ans.Keys, fmt.Sprint(i))
		}
		ans.Confidence = entropyConfidence(p)
	default:
		ans.Keys = []string{"false", "true"}
		ans.Noul = p[1]
		ans.Confidence = math.Max(p[1], 1-p[1])
	}
	return ans
}

// entropyConfidence is confidence_from_probs: 1 - H(p)/log(k), clipped to
// [0, 1], and 1 for a single option.
func entropyConfidence(p []float64) float64 {
	if len(p) < 2 {
		return 1
	}
	var h float64
	for _, v := range p {
		h -= v * math.Log(math.Min(1, math.Max(v, 1e-12)))
	}
	return math.Min(1, math.Max(0, 1-h/math.Log(float64(len(p)))))
}
