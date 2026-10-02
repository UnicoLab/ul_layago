package train

import (
	"context"
	"encoding/json"
	"math"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// Optional actual-GPU parity. Normal CPU-only CI explicitly skips this test;
// macOS validation runs with LAYAGO_MLX_PYTHON=python3, never a mock backend.
func TestMLXForwardAndGradientParity(t *testing.T) {
	python := os.Getenv("LAYAGO_MLX_PYTHON")
	if python == "" {
		t.Skip("set LAYAGO_MLX_PYTHON for actual Apple GPU parity")
	}
	script, err := filepath.Abs("../scripts/mlx_head.py")
	if err != nil {
		t.Fatal(err)
	}
	for ci, c := range gradCases() {
		t.Run(c.name, func(t *testing.T) {
			h := gradHead(t, uint64(ci)+10)
			r := rand.New(rand.NewPCG(uint64(ci), 5))
			k := c.ex.NumOptions()
			rec := &Record{Features: Features{Markers: randFloats(r, k*h.Config.D, 1.5), Pooled: randFloats(r, h.Config.D, 1), K: k}}
			s := sampleOf(c, rec)
			s.weight = 1
			tr := newTrainer(h, Objective{RegWeight: c.reg}, 1)
			g := newGrads(h.Config, false)
			loss := tr.exampleGrad(s, g, tr.work[0])
			logits := h.Scores(rec.Markers, k, c.ex.Type.QType())
			params := map[string]any{}
			for _, p := range h.Params()[:scorerParams] {
				params[p.Name] = map[string]any{"shape": p.Shape, "data": p.Data}
			}
			data, err := json.Marshal(map[string]any{"params": params, "sample": fitSamples([]*sample{s})[0], "reg_weight": c.reg})
			if err != nil {
				t.Fatal(err)
			}
			dir := t.TempDir()
			input, output := filepath.Join(dir, "in.json"), filepath.Join(dir, "out.json")
			if err = os.WriteFile(input, data, 0600); err != nil {
				t.Fatal(err)
			}
			cmd := exec.CommandContext(context.Background(), python, script, "--probe", input, "--output", output)
			if log, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("%v: %s", err, log)
			}
			var result struct {
				Logits    []float64            `json:"logits"`
				Loss      float64              `json:"loss"`
				Gradients map[string][]float64 `json:"gradients"`
			}
			raw, err := os.ReadFile(output)
			if err != nil {
				t.Fatal(err)
			}
			if err = json.Unmarshal(raw, &result); err != nil {
				t.Fatal(err)
			}
			close := func(name string, a, b float64) {
				t.Helper()
				if math.Abs(a-b) > 2e-5*math.Max(1, math.Abs(a)) {
					t.Fatalf("%s Go=%g MLX=%g", name, a, b)
				}
			}
			close("loss", loss, result.Loss)
			if len(result.Logits) != len(logits) {
				t.Fatal("logit shape")
			}
			for i, x := range logits {
				close("logit", x, result.Logits[i])
			}
			for pi, p := range h.Params()[:scorerParams] {
				got := result.Gradients[p.Name]
				want := g.slices()[pi]
				if len(got) != len(want) {
					t.Fatal("gradient shape")
				}
				for i, x := range want {
					close(p.Name, x, got[i])
				}
			}
		})
	}
}

func TestScorerBackendRejectsNonFinite(t *testing.T) {
	// Exercise the split boundary through the normal trainer fixture.
	// Shape/non-finite guards protect plug-in output regardless of runtime.
	h, err := NewHead(HeadConfig{D: 8, Hidden: 8, ActHidden: 256, NAct: 2}, 1)
	if err != nil {
		t.Fatal(err)
	}
	fit := &ScorerFit{Head: h.Clone(), History: []EpochStats{{Epoch: 1, DevCalAccuracy: .5, DevCalNLL: 1}}, BestEpoch: 1}
	if err = validateScorerFit(h, fit); err != nil {
		t.Fatal(err)
	}
	fit.Head.W1[0] = float32(math.Inf(1))
	if validateScorerFit(h, fit) == nil {
		t.Fatal("accepted non-finite weights")
	}
}

func TestScorerBackendReceivesOnlyFittingSplits(t *testing.T) {
	l := qLoop(t)
	role := map[string]string{}
	for _, it := range l.p.Items {
		role[it.Ex.ID] = it.Role
	}
	cfg := loopConfig(1)
	cfg.Epochs = 1
	cfg.ActEpochs = 0
	cfg.Int8 = false
	cfg.Backend = "test"
	calls := 0
	cfg.FitScorer = func(ctx context.Context, h *Head, training, dev []FitSample, c Config) (*ScorerFit, error) {
		calls++
		for _, s := range training {
			if role[s.ID] != SplitTrain {
				t.Fatalf("training leak %s %s", s.ID, role[s.ID])
			}
		}
		for _, s := range dev {
			if role[s.ID] != RoleDevCal {
				t.Fatalf("dev leak %s %s", s.ID, role[s.ID])
			}
		}
		if len(training) == 0 || len(dev) == 0 {
			t.Fatal("empty fitting splits")
		}
		return &ScorerFit{Head: h, History: []EpochStats{{Epoch: 1, DevCalNLL: 1}}, BestEpoch: 1}, nil
	}
	if _, err := Train(context.Background(), l.p, l.recs, l.trunk.Identity.HiddenSize, cfg); err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatal("backend not invoked")
	}
}

func TestMLXFitCalibrateExportAndResume(t *testing.T) {
	python := os.Getenv("LAYAGO_MLX_PYTHON")
	if python == "" {
		t.Skip("set LAYAGO_MLX_PYTHON for actual Apple GPU training")
	}
	l := qLoop(t)
	script, err := filepath.Abs("../scripts/mlx_head.py")
	if err != nil {
		t.Fatal(err)
	}
	cfg := loopConfig(1)
	cfg.Epochs = 2
	cfg.ActEpochs = 2
	cfg.Int8 = false
	cfg.Backend = "mlx"
	work := filepath.Join(t.TempDir(), "mlx")
	cfg.FitScorer = MLXScorer(MLXOptions{Python: python, Script: script, WorkDir: work, MemoryLimit: 256 << 20})
	result, err := Train(context.Background(), l.p, l.recs, l.trunk.Identity.HiddenSize, cfg)
	if err != nil {
		t.Fatal(err)
	}
	dirs, err := Export(filepath.Join(t.TempDir(), "export"), result, ExportMeta{Code: CodeVersion{GitCommit: "gpu-test"}, Trunk: l.trunk, Cache: l.cache})
	if err != nil {
		t.Fatal(err)
	}
	artifact, err := LoadArtifact(dirs[0])
	if err != nil {
		t.Fatal(err)
	}
	scored, err := artifact.ScoreCached(l.p, l.recs, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(scored) != len(result.Float32.Rows) {
		t.Fatal("export lost rows")
	}
	for i, s := range scored {
		if s.Row.PredKey != result.Float32.Rows[i].PredKey {
			t.Fatalf("export prediction parity %s", s.Item.Ex.ID)
		}
	}
	resumed, err := Train(context.Background(), l.p, l.recs, l.trunk.Identity.HiddenSize, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if resumed.Config.BackendInfo["resumed_from_epoch"] != float64(2) {
		t.Fatalf("did not resume: %+v", resumed.Config.BackendInfo)
	}
	for i, p := range result.Float32.Head.Params() {
		for j, x := range p.Data {
			if x != resumed.Float32.Head.Params()[i].Data[j] {
				t.Fatal("resume changed complete-run weights")
			}
		}
	}
	cfg.LR *= 2
	if _, err = Train(context.Background(), l.p, l.recs, l.trunk.Identity.HiddenSize, cfg); err == nil {
		t.Fatal("resumed incompatible optimizer config")
	}
}

func TestScorerDecayHorizonSurvivesEarlyStopCap(t *testing.T) {
	l := qLoop(t)
	cfg := loopConfig(1)
	cfg.Epochs = 4
	cfg.ScheduleEpochs = 2
	cfg.Patience = 4
	cfg.ActEpochs = 0
	cfg.Int8 = false
	result, err := Train(context.Background(), l.p, l.recs, l.trunk.Identity.HiddenSize, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.History) != 4 {
		t.Fatalf("epochs=%d", len(result.History))
	}
	for _, st := range result.History[1:] {
		if math.Abs(st.LR-.1*cfg.LR) > 1e-12 {
			t.Fatalf("epoch %d still decays against maximum cap: %g", st.Epoch, st.LR)
		}
	}
	cfg.ScheduleEpochs = 5
	if _, err = Train(context.Background(), l.p, l.recs, l.trunk.Identity.HiddenSize, cfg); err == nil {
		t.Fatal("accepted horizon above epoch cap")
	}
}

func TestMLXWorkspaceDoesNotOverwriteUnrelatedFiles(t *testing.T) {
	work := t.TempDir()
	path := filepath.Join(work, "input.safetensors")
	if err := os.WriteFile(path, []byte("keep-me"), 0600); err != nil {
		t.Fatal(err)
	}
	head, err := NewHead(HeadConfig{D: 8, Hidden: 8, ActHidden: 256, NAct: 2}, 1)
	if err != nil {
		t.Fatal(err)
	}
	s := FitSample{K: 2, Markers: make([]float32, 16), Target: []float64{1, 0}, Weight: 1}
	fit := MLXScorer(MLXOptions{Python: "python3", Script: "unused", WorkDir: work, MemoryLimit: 1 << 20})
	if _, err = fit(context.Background(), head, []FitSample{s}, []FitSample{s}, DefaultConfig()); err == nil {
		t.Fatal("accepted unrelated workspace")
	}
	raw, err := os.ReadFile(path)
	if err != nil || string(raw) != "keep-me" {
		t.Fatal("overwrote existing content")
	}
}

func TestTrainingRejectsNonFiniteConfig(t *testing.T) {
	cfg := DefaultConfig()
	cfg.LR = math.Inf(1)
	if err := cfg.validate(); err == nil {
		t.Fatal("accepted infinite learning rate")
	}
	cfg = DefaultConfig()
	cfg.ScoreRegWeight = math.NaN()
	if err := cfg.validate(); err == nil {
		t.Fatal("accepted NaN loss weight")
	}
}
