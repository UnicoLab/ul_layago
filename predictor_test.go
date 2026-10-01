package layago

import (
	"context"
	"errors"
	"github.com/UnicoLab/ul_layago/laya"
	"github.com/UnicoLab/ul_layago/model"
	"github.com/UnicoLab/ul_layago/tensor"
	"math"
	"reflect"
	"strings"
	"testing"
)

func TestCheckpointFacadePreservesNumericalParity(t *testing.T) {
	p, err := Open(OpenOptions{Checkpoint: "testdata/parity/checkpoint", Workers: 1, Kernel: tensor.KernelGeneric})
	if err != nil {
		t.Fatal(err)
	}
	qs := []Question{{ID: "choice", Type: "choice", Instructions: "Pick the topic.", Options: []Option{{Key: "a"}, {Key: "b"}}}, {ID: "score", Type: "score", Instructions: "Score urgency.", Levels: []string{"low", "high"}}, {ID: "boolean", Type: "yes_no", Instructions: "Is this urgent?"}}
	request := Request{State: "Hello", Questions: qs}
	got, err := p.Predict(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := laya.LoadCheckpoint("testdata/parity/checkpoint", laya.LoadOptions{})
	if err != nil {
		t.Fatal(err)
	}
	built := make([]laya.Question, len(qs))
	for i, q := range qs {
		built[i], err = p.question(q)
		if err != nil {
			t.Fatal(err)
		}
	}
	expected, err := raw.Predict(context.Background(), request.State, built, model.Options{Parallel: tensor.Parallel{Workers: 1, Kernel: tensor.KernelGeneric}})
	if err != nil {
		t.Fatal(err)
	}
	for i, d := range got.Decisions {
		if !reflect.DeepEqual(d.Probabilities, expected.Answers[i].Probs) {
			t.Fatalf("numerical drift: %v != %v", d, expected.Answers[i])
		}
		if d.Decision != "escalate" || d.Truncated {
			t.Fatalf("uncalibrated checkpoint decision: %v", d)
		}
		sum := 0.0
		for _, v := range d.Probabilities {
			sum += v
		}
		if math.Abs(sum-1) > 1e-9 {
			t.Fatal(sum)
		}
	}
	if got.Decisions[1].Type != "score" || got.Decisions[2].Type != "yes_no" || !got.RequiresDomainValidation {
		t.Fatal(got)
	}
}

func TestInputBudgetsAndCancellation(t *testing.T) {
	p, err := Open(OpenOptions{Checkpoint: "testdata/parity/checkpoint", Workers: 1})
	if err != nil {
		t.Fatal(err)
	}
	q := Question{ID: "q", Type: "choice", Instructions: "Pick.", Options: []Option{{Key: "a"}, {Key: "b"}}}
	r := Request{State: strings.Repeat("hello world ", 200), Questions: []Question{q}}
	if _, err := p.Predict(context.Background(), r); !errors.Is(err, ErrTruncation) {
		t.Fatalf("silent truncation: %v", err)
	}
	r.AllowTruncation = true
	got, err := p.Predict(context.Background(), r)
	if err != nil {
		t.Fatal(err)
	}
	if !got.Decisions[0].Truncated || got.Decisions[0].Decision != "escalate" {
		t.Fatal(got)
	}
	r.State = "Hi"
	r.AllowTruncation = false
	r.Questions[0].Options[0].Description = strings.Repeat("word ", 200)
	if _, err := p.Predict(context.Background(), r); !errors.Is(err, ErrTruncation) {
		t.Fatalf("option clipping: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := p.Predict(ctx, r); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	p.gate <- struct{}{}
	if _, err := p.Predict(ctx, r); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	<-p.gate
}

func TestArtifactUsesPinnedOptions(t *testing.T) {
	p, err := Open(OpenOptions{Artifact: "testdata/artifact/tiny-q/float32", Base: "testdata/parity/checkpoint", Workers: 1})
	if err != nil {
		t.Fatal(err)
	}
	q, err := p.question(Question{ID: "q", Task: "q_operation"})
	if err != nil {
		t.Fatal(err)
	}
	if len(q.Options) < 2 {
		t.Fatal("trained options missing")
	}
	if _, err := p.question(Question{ID: "q", Task: "q_operation", Options: []Option{{Key: "wrong"}}}); err == nil {
		t.Fatal("allowed incompatible options")
	}
	if _, err := p.question(Question{ID: "q", Task: "q_operation", Levels: []string{"wrong"}}); err == nil {
		t.Fatal("allowed irrelevant levels")
	}
}
