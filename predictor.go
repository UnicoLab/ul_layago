package layago

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/UnicoLab/ul_layago/laya"
	"github.com/UnicoLab/ul_layago/model"
	"github.com/UnicoLab/ul_layago/tensor"
)

const Version = "0.2.0"
const PredictionContract = "ul.layago-predict/v1"

var ErrTruncation = errors.New("layago: input exceeds model budgets; shorten or segment it, or explicitly allow truncation")

// OpenOptions selects local weights. Exactly one of Checkpoint or Artifact is required.
// Artifact additionally requires Base, the exact encoder used in training.
type OpenOptions struct {
	Checkpoint string
	Artifact   string
	Base       string
	Workers    int
	Kernel     tensor.Kernel
}

type Option struct {
	Key         string `json:"key"`
	Description string `json:"description,omitempty"`
}

// Question uses a type and instructions with checkpoints, or a pinned task with artifacts.
type Question struct {
	ID           string   `json:"id"`
	Type         string   `json:"type,omitempty"`
	Task         string   `json:"task,omitempty"`
	Instructions string   `json:"instructions,omitempty"`
	Options      []Option `json:"options,omitempty"`
	Levels       []string `json:"levels,omitempty"`
	False        string   `json:"false,omitempty"`
	True         string   `json:"true,omitempty"`
}

type Request struct {
	ID              string     `json:"id,omitempty"`
	State           string     `json:"state"`
	Questions       []Question `json:"questions"`
	AllowTruncation bool       `json:"allow_truncation,omitempty"`
}

// Decision is a model nomination. Artifact act thresholds certify only their calibration distribution.
type Decision struct {
	ID             string    `json:"id"`
	Type           string    `json:"type"`
	Task           string    `json:"task,omitempty"`
	Value          any       `json:"value"`
	Keys           []string  `json:"keys"`
	Probabilities  []float64 `json:"probabilities"`
	ActProbability float64   `json:"act_probability"`
	Decision       string    `json:"decision"`
	Truncated      bool      `json:"truncated"`
}

type Metadata struct {
	AdapterID string `json:"adapter_id,omitempty"`
	Kind      string `json:"kind"`
	SHA256    string `json:"sha256"`
	Kernel    string `json:"kernel"`
	Workers   int    `json:"workers"`
	MaxLen    int    `json:"max_len"`
}

type Response struct {
	Contract                 string     `json:"contract"`
	Version                  string     `json:"version"`
	ID                       string     `json:"id,omitempty"`
	Decisions                []Decision `json:"decisions"`
	InputTokens              int        `json:"input_tokens"`
	ElapsedMS                float64    `json:"elapsed_ms"`
	Model                    Metadata   `json:"model"`
	RequiresDomainValidation bool       `json:"requires_domain_validation"`
}

// Predictor owns loaded weights and serializes requests, while using intra-operation parallelism.
// Calls are safe from multiple goroutines; waiting for a slot respects context cancellation.
type Predictor struct {
	agent    *laya.Agent
	artifact *laya.Artifact
	parallel tensor.Parallel
	gate     chan struct{}
	metadata Metadata
}

func Open(opts OpenOptions) (*Predictor, error) {
	if (opts.Checkpoint == "") == (opts.Artifact == "") {
		return nil, errors.New("layago: choose exactly one checkpoint or artifact")
	}
	if opts.Workers < 0 {
		return nil, errors.New("layago: workers must be nonnegative")
	}
	if _, err := tensor.ParseKernel(opts.Kernel.String()); err != nil {
		return nil, err
	}
	var agent *laya.Agent
	var artifact *laya.Artifact
	var err error
	kind, digest := "checkpoint", ""
	if opts.Artifact != "" {
		if opts.Base == "" {
			return nil, errors.New("layago: artifact requires its base encoder")
		}
		artifact, err = laya.LoadArtifact(opts.Artifact, opts.Base, laya.ArtifactOptions{})
		if err != nil {
			return nil, err
		}
		agent, kind, digest = artifact.Agent, "artifact", artifact.ManifestSHA256
	} else {
		if opts.Base != "" {
			return nil, errors.New("layago: base applies only to artifacts")
		}
		agent, err = laya.LoadCheckpoint(opts.Checkpoint, laya.LoadOptions{})
		if err != nil {
			return nil, err
		}
		f, err := os.Open(filepath.Join(opts.Checkpoint, "model.safetensors"))
		if err != nil {
			return nil, err
		}
		h := sha256.New()
		_, err = io.Copy(h, f)
		closeErr := f.Close()
		if err != nil {
			return nil, err
		}
		if closeErr != nil {
			return nil, closeErr
		}
		digest = hex.EncodeToString(h.Sum(nil))
	}
	parallel := tensor.Parallel{Workers: opts.Workers, Kernel: opts.Kernel}
	agent.Prepack(parallel.Kernel)
	return &Predictor{agent: agent, artifact: artifact, parallel: parallel, gate: make(chan struct{}, 1), metadata: Metadata{Kind: kind, SHA256: digest, Kernel: parallel.EffectiveKernel().String(), Workers: parallel.N(), MaxLen: agent.Config.MaxLen}}, nil
}

func (p *Predictor) Metadata() Metadata { return p.metadata }

func (p *Predictor) question(q Question) (laya.Question, error) {
	var options []laya.Option
	if q.Options != nil {
		options = make([]laya.Option, len(q.Options))
	}
	keys := map[string]bool{}
	for i, option := range q.Options {
		if strings.TrimSpace(option.Key) == "" || keys[option.Key] {
			return laya.Question{}, fmt.Errorf("layago: question %q has empty or duplicate option keys", q.ID)
		}
		keys[option.Key] = true
		options[i] = laya.Option{Key: option.Key, Description: option.Description}
	}
	if p.artifact != nil {
		if q.Task == "" || q.Type != "" || q.Instructions != "" {
			return laya.Question{}, errors.New("layago: artifact questions require task and use their trained type/instructions")
		}
		built, err := p.artifact.Question(laya.ArtifactQuestion{ID: q.ID, Task: q.Task, Options: options, Levels: q.Levels, False: q.False, True: q.True})
		if err != nil {
			return built, err
		}
		if (built.Type != laya.Choice && q.Options != nil) || (built.Type != laya.Score && q.Levels != nil) || (built.Type != laya.Noul && (q.False != "" || q.True != "")) {
			return built, errors.New("layago: criteria do not match the trained question type")
		}
		return built, nil
	}
	if q.Task != "" {
		return laya.Question{}, errors.New("layago: task applies only to artifacts")
	}
	if strings.TrimSpace(q.Instructions) == "" {
		return laya.Question{}, errors.New("layago: instructions are required")
	}
	built := laya.Question{ID: q.ID, Instructions: q.Instructions}
	switch q.Type {
	case "choice":
		if len(options) < 2 || len(q.Levels) != 0 || q.False != "" || q.True != "" {
			return built, errors.New("layago: choice requires at least two options and no levels or yes/no criteria")
		}
		built.Type, built.Options = laya.Choice, options
	case "score":
		if len(q.Levels) < 2 || len(options) != 0 || q.False != "" || q.True != "" {
			return built, errors.New("layago: score requires at least two levels and no choice/yes-no criteria")
		}
		built.Type, built.Levels = laya.Score, q.Levels
	case "yes_no":
		if len(options) != 0 || len(q.Levels) != 0 {
			return built, errors.New("layago: yes_no does not accept choice options or score levels")
		}
		built.Type, built.False, built.True = laya.Noul, q.False, q.True
	default:
		return built, fmt.Errorf("layago: unknown type %q", q.Type)
	}
	return built, nil
}

// truncated checks state, instruction and option clipping before calling the parity-preserving engine.
func (p *Predictor) truncated(state string, q laya.Question) (bool, error) {
	encode := func(text string) (int, error) {
		encoded, err := p.agent.Tokenizer.Encode(strings.ReplaceAll(text, p.agent.Special.MaskToken, " "), false)
		return len(encoded.IDs), err
	}
	stateTokens, err := encode(state)
	if err != nil {
		return false, err
	}
	headTokens, err := encode(q.Type.String() + " question: " + q.Instructions)
	if err != nil {
		return false, err
	}
	fullEmpty := headTokens + 4
	for _, option := range laya.RenderOptions(q) {
		n, err := encode(" " + option)
		if err != nil {
			return false, err
		}
		fullEmpty += n + 1
	}
	empty, _, err := laya.BuildSequence(p.agent.Tokenizer, p.agent.Special, "", q, p.agent.Config.MaxLen, p.agent.Config.HeadMaxLen, nil, false)
	if err != nil {
		return false, err
	}
	return fullEmpty > len(empty) || len(empty)+stateTokens > p.agent.Config.MaxLen, nil
}

func (p *Predictor) Predict(ctx context.Context, request Request) (*Response, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(request.State) > 1<<20 || len(request.Questions) == 0 || len(request.Questions) > 64 {
		return nil, errors.New("layago: state limit is 1 MiB; questions must number 1–64")
	}
	select {
	case p.gate <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	defer func() { <-p.gate }()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	started := time.Now()
	questions := make([]laya.Question, len(request.Questions))
	truncated := make([]bool, len(questions))
	ids := map[string]bool{}
	for i, q := range request.Questions {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if strings.TrimSpace(q.ID) == "" || ids[q.ID] {
			return nil, errors.New("layago: question IDs must be nonempty and unique")
		}
		ids[q.ID] = true
		if len(q.Options) > 128 || len(q.Levels) > 128 {
			return nil, errors.New("layago: at most 128 options/levels per question")
		}
		built, err := p.question(q)
		if err != nil {
			return nil, err
		}
		questions[i] = built
		truncated[i], err = p.truncated(request.State, built)
		if err != nil {
			return nil, err
		}
		if truncated[i] && !request.AllowTruncation {
			return nil, fmt.Errorf("%w (question %q)", ErrTruncation, q.ID)
		}
	}
	var answers []laya.Answer
	var decisions []string
	var prediction *laya.Prediction
	var err error
	if p.artifact == nil {
		prediction, err = p.agent.Predict(ctx, request.State, questions, model.Options{Parallel: p.parallel})
		if err != nil {
			return nil, err
		}
		answers = prediction.Answers
		decisions = make([]string, len(answers))
		for i := range decisions {
			decisions[i] = "escalate"
		}
	} else {
		asked := make([]laya.ArtifactQuestion, len(questions))
		for i, q := range questions {
			asked[i] = laya.ArtifactQuestion{ID: q.ID, Task: request.Questions[i].Task, Options: q.Options, Levels: q.Levels, False: q.False, True: q.True}
		}
		var calibrated []laya.ArtifactAnswer
		calibrated, prediction, err = p.artifact.Predict(ctx, request.State, asked, model.Options{Parallel: p.parallel})
		if err != nil {
			return nil, err
		}
		answers, decisions = make([]laya.Answer, len(calibrated)), make([]string, len(calibrated))
		for i, a := range calibrated {
			answers[i], decisions[i] = a.Answer, a.Decision
		}
	}
	response := &Response{Contract: PredictionContract, Version: Version, ID: request.ID, InputTokens: prediction.InputTokens, Model: p.metadata, RequiresDomainValidation: true, Decisions: make([]Decision, len(answers))}
	for i, a := range answers {
		kind, value := "choice", any(a.Choice)
		if questions[i].Type == laya.Score {
			kind, value = "score", a.Score
		}
		if questions[i].Type == laya.Noul {
			kind, value = "yes_no", a.Noul >= 0.5
		}
		if len(a.Probs) == 0 {
			value = nil
		}
		decision := decisions[i]
		if truncated[i] {
			decision = "escalate"
		}
		response.Decisions[i] = Decision{ID: questions[i].ID, Type: kind, Task: request.Questions[i].Task, Value: value, Keys: a.Keys, Probabilities: a.Probs, ActProbability: a.ActProbability, Decision: decision, Truncated: truncated[i]}
	}
	response.ElapsedMS = float64(time.Since(started).Microseconds()) / 1000
	return response, nil
}
