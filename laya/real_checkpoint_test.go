package laya

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"testing"
	"time"

	"github.com/UnicoLab/ul_layago/model"
	"github.com/UnicoLab/ul_layago/tensor"
)

type realGolden struct {
	Producer      string `json:"producer"`
	WeightsSHA256 string `json:"weights_sha256"`
	Questions     []struct {
		ID           string `json:"id"`
		Type         string `json:"type"`
		Instructions string `json:"instructions"`
		Options      []struct {
			Key         string `json:"key"`
			Description string `json:"description"`
		} `json:"options"`
	} `json:"questions"`
	Cases []struct {
		ID      string `json:"id"`
		State   string `json:"state"`
		Answers []struct {
			ID             string    `json:"id"`
			IDs            []int32   `json:"ids"`
			Markers        []int     `json:"markers"`
			Logits         []float64 `json:"logits"`
			ActProbability float64   `json:"act_probability"`
			Choice         string    `json:"choice"`
			Expected       string    `json:"expected"`
		} `json:"answers"`
	} `json:"cases"`
}

// TestRealCheckpointParity runs the Go port on the Apache-2.0 Laya
// multilingual checkpoint (mmBERT-base, 22 layers) over Hive's frozen
// judgment fixture and compares it with PyTorch outputs recorded by
// testdata/parity/laya_checkpoint_golden.py. The checkpoint is not in the
// repository and stays rejected for Hive judgment; set LAYAGO_LAYA_CHECKPOINT
// to its directory to run this offline replication check, optionally
// LAYAGO_LAYA_KERNEL (auto, generic, arm64_neon, amd64_avx2) to pick the GEMM
// kernel, and LAYAGO_LAYA_PATH=int8 to quantize the encoder (embeddings and
// linears, per-row int8) and hold it to the design's quantized-path bound
// against the same float32 PyTorch reference. That int8 run is the parity
// half an int8 M0-LAYA gate pass needs (gateParityPrerequisite).
func TestRealCheckpointParity(t *testing.T) {
	dir := os.Getenv("LAYAGO_LAYA_CHECKPOINT")
	if dir == "" {
		t.Skip("set LAYAGO_LAYA_CHECKPOINT to the Laya multilingual checkpoint directory")
	}
	path := os.Getenv("LAYAGO_LAYA_PATH")
	if path == "" {
		path = "fp32"
	}
	bound, err := realCheckpointBound(path)
	if err != nil {
		t.Fatal(err)
	}
	kernelName := os.Getenv("LAYAGO_LAYA_KERNEL")
	if kernelName == "" {
		kernelName = "auto"
	}
	kern, err := tensor.ParseKernel(kernelName)
	if err != nil {
		t.Fatal(err)
	}
	par := tensor.Parallel{Kernel: kern}
	if kern != tensor.KernelAuto && par.EffectiveKernel() != kern {
		t.Fatalf("LAYAGO_LAYA_KERNEL=%s is not available on this CPU/build (it would run %s)", kern, par.EffectiveKernel())
	}
	t.Logf("GEMM kernel: %s, path: %s", par.EffectiveKernel(), path)
	raw, err := os.ReadFile("testdata/laya_multilingual_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g realGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	if got := fileSHA256(t, filepath.Join(dir, "model.safetensors")); got != g.WeightsSHA256 {
		t.Fatalf("model.safetensors sha256 %s, golden recorded %s", got, g.WeightsSHA256)
	}
	started := time.Now()
	agent, err := LoadCheckpoint(dir, LoadOptions{Int8: path == "int8"})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("loaded checkpoint in %s", time.Since(started))
	var qs []Question
	for _, gq := range g.Questions {
		if gq.Type != "choice" {
			t.Fatalf("golden question %s has type %s; only choice is recorded", gq.ID, gq.Type)
		}
		q := Question{ID: gq.ID, Type: Choice, Instructions: gq.Instructions}
		for _, o := range gq.Options {
			q.Options = append(q.Options, Option{Key: o.Key, Description: o.Description})
		}
		qs = append(qs, q)
	}

	type row struct {
		question string
		correct  bool
		top      float64
	}
	var rows []row
	var worstLogit, worstProb, worstAct float64
	choiceFlips, anyChange := 0, 0
	for _, c := range g.Cases {
		pred, err := agent.Predict(context.Background(), c.State, qs, model.Options{Parallel: par})
		if err != nil {
			t.Fatal(err)
		}
		for i, want := range c.Answers {
			got := pred.Answers[i]
			if !slices.Equal(pred.Sequences[i], want.IDs) || !slices.Equal(pred.Markers[i], want.Markers) {
				t.Fatalf("%s/%s: token sequence differs from Laya's build_sequence", c.ID, want.ID)
			}
			wantP := tensor.Softmax64(want.Logits)
			for j := range want.Logits {
				worstLogit = math.Max(worstLogit, math.Abs(float64(got.Logits[j])-want.Logits[j]))
				worstProb = math.Max(worstProb, math.Abs(got.Probs[j]-wantP[j]))
			}
			worstAct = math.Max(worstAct, math.Abs(got.ActProbability-want.ActProbability))
			if got.Choice != want.Choice {
				anyChange++
				if topMargin(wantP) > 1e-3 {
					choiceFlips++
					t.Errorf("%s/%s: choice %q, PyTorch %q (PyTorch top-2 margin %.3g)", c.ID, want.ID, got.Choice, want.Choice, topMargin(wantP))
				}
			}
			top := 0.0
			for _, p := range got.Probs {
				top = math.Max(top, math.Round(p*1e4)/1e4)
			}
			rows = append(rows, row{question: want.ID, correct: got.Choice == want.Expected, top: top})
		}
	}
	t.Logf("%s vs Go (%s, %s): %d cases x %d questions; max |Δ logit| %.3g, max |Δ p| %.3g, max |Δ act p| %.3g, choice flips %d, choice changes at any margin %d",
		g.Producer, path, par.EffectiveKernel(), len(g.Cases), len(qs), worstLogit, worstProb, worstAct, choiceFlips, anyChange)
	saturated := true
	for _, c := range g.Cases {
		for _, a := range c.Answers {
			saturated = saturated && a.ActProbability == 1
		}
	}
	if saturated {
		// The checkpoint's act head outputs exactly 1.0 in float32 for every
		// fixture input, so this comparison cannot detect an error here; the
		// synthetic goldens cover the act head against the laya package.
		t.Log("act head saturated at 1.0 on every input: act-probability parity is vacuous for this checkpoint")
	}
	if worstProb > bound || choiceFlips > 0 {
		t.Fatalf("real-checkpoint %s parity outside tolerance: |Δp| %.3g (bound %g), flips %d", path, worstProb, bound, choiceFlips)
	}

	// Informational: the same metrics evaluate_laya.py recorded for this
	// checkpoint on 2026-09-22 (transformers 5.17 on MPS).
	byQuestion := map[string][]row{}
	for _, r := range rows {
		byQuestion[r.question] = append(byQuestion[r.question], r)
	}
	names := make([]string, 0, len(byQuestion))
	for name := range byQuestion {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		rs := byQuestion[name]
		correct, sum := 0, 0.0
		for _, r := range rs {
			if r.correct {
				correct++
			}
			sum += r.top
		}
		t.Logf("%-24s correct %2d/%d  mean top probability %.6f", name, correct, len(rs), sum/float64(len(rs)))
	}
}

// realCheckpointBound is the design's final-probability tolerance per path
// (revise_laya-go.md, "Encoder parity vs PyTorch"): fp32 |Δp| <= 1e-4 and
// the quantized path |Δp| <= 0.005, both with argmax agreement wherever
// PyTorch's top-2 margin exceeds 1e-3.
func realCheckpointBound(path string) (float64, error) {
	switch path {
	case "fp32":
		return 1e-4, nil
	case "int8":
		return 0.005, nil
	}
	return 0, fmt.Errorf("LAYAGO_LAYA_PATH must be fp32 or int8, got %q", path)
}

func topMargin(p []float64) float64 {
	s := append([]float64(nil), p...)
	sort.Sort(sort.Reverse(sort.Float64Slice(s)))
	if len(s) < 2 {
		return 1
	}
	return s[0] - s[1]
}

func fileSHA256(t *testing.T, path string) string {
	t.Helper()
	f, err := os.Open(path) //nolint:gosec // test reads the checkpoint the caller pointed at
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		t.Fatal(err)
	}
	return hex.EncodeToString(h.Sum(nil))
}
