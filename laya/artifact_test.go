package laya

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
	"strings"
	"testing"

	"github.com/UnicoLab/ul_layago/model"
	"github.com/UnicoLab/ul_layago/tensor"
)

// The tiny-q artifacts were exported by tools/laya-train (python -m
// laya_train parity-fixture): a head trained on the generated fixture
// dataset over the synthetic parity trunk truncated to 4 of its 5 layers.
// parity_golden.json holds PyTorch's CPU float32 numbers for the exported
// tensors (dequantised for int8).
const (
	artifactDir     = "../testdata/artifact/tiny-q/"
	artifactBaseDir = parityDir + "checkpoint"
)

type artifactCase struct {
	ID             string         `json:"id"`
	Task           string         `json:"task"`
	Lang           string         `json:"lang"`
	State          string         `json:"state"`
	Question       goldenQuestion `json:"question"`
	IDs            []int32        `json:"ids"`
	Markers        []int          `json:"markers"`
	Logits         []float64      `json:"logits"`
	ActLogits      []float64      `json:"act_logits"`
	Temperature    float64        `json:"temperature"`
	TemperatureKey string         `json:"temperature_key"`
	Probs          []float64      `json:"probs"`
	ActProbability float64        `json:"act_probability"`
	DecisionScore  float64        `json:"decision_score"`
	Decision       string         `json:"decision"`
	Prediction     string         `json:"prediction"`
}

type artifactGolden struct {
	Contract  string `json:"contract"`
	Tolerance struct {
		LogitAbs float64 `json:"logit_abs"`
		ProbAbs  float64 `json:"prob_abs"`
	} `json:"tolerance"`
	Cases []artifactCase `json:"cases"`
}

func readArtifactGolden(t *testing.T, dir string) artifactGolden {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(dir, "parity_golden.json")) //nolint:gosec // test fixture path
	if err != nil {
		t.Fatal(err)
	}
	var g artifactGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	if g.Contract != ArtifactContract || len(g.Cases) == 0 || g.Tolerance.LogitAbs <= 0 || g.Tolerance.ProbAbs <= 0 {
		t.Fatalf("%s: malformed parity golden", dir)
	}
	return g
}

func artifactQuestion(c artifactCase) ArtifactQuestion {
	q := ArtifactQuestion{ID: c.ID, Task: c.Task, Levels: c.Question.Levels, False: c.Question.False, True: c.Question.True}
	for _, o := range c.Question.Options {
		q.Options = append(q.Options, Option{Key: o.Key, Description: o.Description})
	}
	return q
}

// checkArtifactParity runs every golden case through the Go runtime and
// holds it to the golden's tolerances; decisions must agree except where the
// score sits within one quantum of the threshold.
func checkArtifactParity(t *testing.T, art *Artifact, g artifactGolden, par tensor.Parallel) (worstLogit, worstProb float64) {
	t.Helper()
	for _, c := range g.Cases {
		q := artifactQuestion(c)
		built, err := art.Question(q)
		if err != nil {
			t.Fatal(err)
		}
		if built.Instructions != c.Question.Instructions || built.Type.String() != c.Question.Type {
			t.Fatalf("%s: question built as %s %q, trained as %s %q", c.ID, built.Type, built.Instructions, c.Question.Type, c.Question.Instructions)
		}
		answers, pred, err := art.Predict(context.Background(), c.State, []ArtifactQuestion{q}, model.Options{Parallel: par})
		if err != nil {
			t.Fatal(err)
		}
		a := answers[0]
		if !slices.Equal(pred.Sequences[0], c.IDs) || !slices.Equal(pred.Markers[0], c.Markers) {
			t.Fatalf("%s: token ids or markers differ from training\n got %v %v\nwant %v %v", c.ID, pred.Sequences[0], pred.Markers[0], c.IDs, c.Markers)
		}
		if math.Abs(a.Temperature-c.Temperature) > 1e-12 || a.TemperatureKey != c.TemperatureKey {
			t.Fatalf("%s: temperature %v (%q), want %v (%q)", c.ID, a.Temperature, a.TemperatureKey, c.Temperature, c.TemperatureKey)
		}
		for j := range c.Logits {
			worstLogit = math.Max(worstLogit, math.Abs(float64(a.Logits[j])-c.Logits[j]))
			worstProb = math.Max(worstProb, math.Abs(a.Probs[j]-c.Probs[j]))
		}
		for j := range c.ActLogits {
			worstLogit = math.Max(worstLogit, math.Abs(float64(a.ActLogits[j])-c.ActLogits[j]))
		}
		worstProb = math.Max(worstProb, math.Abs(a.ActProbability-c.ActProbability))
		worstProb = math.Max(worstProb, math.Abs(a.DecisionScore-c.DecisionScore))
		if a.Decision != c.Decision && !(a.Threshold != nil && math.Abs(QuantizeProbability(a.DecisionScore)-*a.Threshold) <= 1e-4) {
			t.Errorf("%s: decision %s, want %s (score %v)", c.ID, a.Decision, c.Decision, a.DecisionScore)
		}
		pick := a.Choice
		switch a.Type {
		case Noul:
			pick = map[bool]string{false: "false", true: "true"}[a.Noul >= 0.5]
		case Score:
			pick = a.Keys[slices.Index(a.Probs, slices.Max(a.Probs))]
		}
		if pick != c.Prediction && math.Abs(sortedMargin(c.Probs)) > 1e-4 {
			t.Errorf("%s: prediction %q, want %q", c.ID, pick, c.Prediction)
		}
	}
	if worstLogit > g.Tolerance.LogitAbs || worstProb > g.Tolerance.ProbAbs {
		t.Fatalf("max |Δ logit| %.3g (tol %g), max |Δ p| %.3g (tol %g)", worstLogit, g.Tolerance.LogitAbs, worstProb, g.Tolerance.ProbAbs)
	}
	return worstLogit, worstProb
}

func sortedMargin(p []float64) float64 {
	s := slices.Clone(p)
	slices.Sort(s)
	if len(s) < 2 {
		return 1
	}
	return s[len(s)-1] - s[len(s)-2]
}

func TestArtifactMatchesPyTorchGolden(t *testing.T) {
	for _, variant := range []string{"float32", "int8"} {
		dir := artifactDir + variant
		art, err := LoadArtifact(dir, artifactBaseDir, ArtifactOptions{})
		if err != nil {
			t.Fatalf("%s: %v", variant, err)
		}
		if art.Manifest.Weights.Format != variant || art.Manifest.Base.LayersUsed != 4 || art.Manifest.Base.LayersTotal != 5 {
			t.Fatalf("%s: unexpected manifest %+v %+v", variant, art.Manifest.Weights, art.Manifest.Base)
		}
		if got := len(art.Agent.Encoder.Weights.Layers); got != 4 {
			t.Fatalf("%s: encoder has %d layers, want the first 4", variant, got)
		}
		g := readArtifactGolden(t, dir)
		for _, kern := range tensor.Kernels() {
			for _, workers := range []int{1, 3} {
				l, p := checkArtifactParity(t, art, g, tensor.Parallel{Workers: workers, Kernel: kern})
				t.Logf("%s %s workers=%d: %d cases, max |Δ logit| %.3g, max |Δ p| %.3g", variant, kern, workers, len(g.Cases), l, p)
			}
		}
	}
}

// The committed artifact certified no threshold (a random tiny trunk is
// near chance), so its goldens only escalate. This drives the act branch
// with thresholds set in memory against the golden decision scores.
func TestArtifactDecisionsFollowCertifiedThresholds(t *testing.T) {
	art, err := LoadArtifact(artifactDir+"float32", artifactBaseDir, ArtifactOptions{})
	if err != nil {
		t.Fatal(err)
	}
	g := readArtifactGolden(t, artifactDir+"float32")
	c := g.Cases[0]
	for _, kind := range []string{ScoreMaxProbability, ScoreActProbability} {
		want := slices.Max(c.Probs)
		if kind == ScoreActProbability {
			want = c.ActProbability
		}
		q := QuantizeProbability(want)
		for _, tc := range []struct {
			lambda float64
			act    bool
		}{{q, true}, {q - 1e-4, true}, {q + 1e-4, false}, {0, true}, {1.0001, false}} {
			lambda := tc.lambda
			art.Manifest.Act.Thresholds = map[string]ArtifactThreshold{c.Task: {Score: kind, Lambda: &lambda, Certified: true}}
			answers, _, err := art.Predict(context.Background(), c.State, []ArtifactQuestion{artifactQuestion(c)}, model.Options{})
			if err != nil {
				t.Fatal(err)
			}
			a := answers[0]
			if a.ScoreKind != kind || (a.Decision == DecisionAct) != tc.act || a.Threshold == nil || *a.Threshold != lambda {
				t.Fatalf("%s at lambda %v: %s on score %v (%s), want act=%v", kind, lambda, a.Decision, a.DecisionScore, a.ScoreKind, tc.act)
			}
		}
		lambda := 0.0
		art.Manifest.Act.Thresholds = map[string]ArtifactThreshold{c.Task: {Score: kind, Lambda: &lambda, Certified: false}}
		answers, _, err := art.Predict(context.Background(), c.State, []ArtifactQuestion{artifactQuestion(c)}, model.Options{})
		if err != nil {
			t.Fatal(err)
		}
		if answers[0].Decision != DecisionEscalate || answers[0].Threshold != nil {
			t.Fatalf("an uncertified threshold must escalate, got %s", answers[0].Decision)
		}
	}
}

func TestQuantizeProbabilityMatchesTrainer(t *testing.T) {
	for _, tc := range []struct{ p, want float64 }{
		{0, 0}, {1, 1}, {0.12344999, 0.1234}, {0.12345, 0.1235}, {0.99995, 1}, {0.00004, 0},
	} {
		if got := QuantizeProbability(tc.p); math.Abs(got-tc.want) > 1e-15 {
			t.Errorf("QuantizeProbability(%v) = %v, want %v", tc.p, got, tc.want)
		}
	}
}

func TestArtifactTemperatureLookupOrder(t *testing.T) {
	art := &Artifact{Manifest: ArtifactManifest{Calibration: ArtifactCalibration{Temperatures: map[string]ArtifactTemperature{
		"a:3-5": {T: 2}, "a": {T: 3}, "*": {T: 0.7}, "b": {T: 9},
	}}}}
	for _, tc := range []struct {
		task string
		k    int
		t    float64
		key  string
	}{
		{"a", 4, 2, "a:3-5"}, {"a", 2, 3, "a"}, {"c", 2, 0.7, "*"}, {"b", 7, TempMax, "b"},
	} {
		got, key := art.Temperature(tc.task, tc.k)
		if got != tc.t || key != tc.key {
			t.Errorf("Temperature(%s, %d) = %v %q, want %v %q", tc.task, tc.k, got, key, tc.t, tc.key)
		}
	}
	empty := &Artifact{}
	if got, key := empty.Temperature("a", 3); got != 1 || key != "" {
		t.Fatalf("no calibration must give 1.0, got %v %q", got, key)
	}
}

func TestArtifactQuestionUsesTrainedOptions(t *testing.T) {
	art, err := LoadArtifact(artifactDir+"float32", artifactBaseDir, ArtifactOptions{})
	if err != nil {
		t.Fatal(err)
	}
	fixed, err := art.Question(ArtifactQuestion{ID: "x", Task: "q_operation"})
	if err != nil {
		t.Fatal(err)
	}
	if fixed.Type != Choice || len(fixed.Options) != 4 || fixed.Options[0].Key != "lookup" || fixed.Options[0].Description == "" {
		t.Fatalf("fixed options not taken from the manifest: %+v", fixed)
	}
	if _, err := art.Question(ArtifactQuestion{Task: "q_operation", Options: []Option{{Key: "lookup"}, {Key: "rank"}}}); err == nil {
		t.Fatal("options other than the trained fixed set must be refused")
	}
	if _, err := art.Question(ArtifactQuestion{Task: "q_metric_choice"}); err == nil {
		t.Fatal("a per-example task without options must be refused")
	}
	if _, err := art.Question(ArtifactQuestion{Task: "q_unknown"}); err == nil {
		t.Fatal("an unknown task must be refused")
	}
	yn, err := art.Question(ArtifactQuestion{Task: "q_time_supported"})
	if err != nil || yn.Type != Noul {
		t.Fatalf("yes_no task: %+v %v", yn, err)
	}
	// q_time_supported was trained with the default false/true criteria only
	for _, texts := range [][2]string{{"no", ""}, {"", "yes"}, {"not supported", "supported"}} {
		if _, err := art.Question(ArtifactQuestion{Task: "q_time_supported", False: texts[0], True: texts[1]}); err == nil {
			t.Fatalf("yes_no texts %q other than the trained ones must be refused", texts)
		}
	}
}

// A question whose option markers do not fit the sequence limits is escalated
// on its own, as the trainer escalates it; it never fails the other questions
// of the batch.
func TestArtifactEscalatesAnUnscorableQuestionAlone(t *testing.T) {
	art, err := LoadArtifact(artifactDir+"float32", artifactBaseDir, ArtifactOptions{})
	if err != nil {
		t.Fatal(err)
	}
	g := readArtifactGolden(t, artifactDir+"float32")
	var c artifactCase
	for _, gc := range g.Cases {
		if gc.Task == "q_operation" {
			c = gc
			break
		}
	}
	if c.ID == "" {
		t.Fatal("the golden has no q_operation case")
	}
	// 45 options of at least 4 tokens each push markers past max_len
	var many []Option
	for i := range 45 {
		many = append(many, Option{Key: fmt.Sprintf("metric.m%02d", i), Description: "a long and distinct metric label"})
	}
	unscorable := ArtifactQuestion{ID: "too-many", Task: "q_metric_choice", Options: many}
	alone, _, err := art.Predict(context.Background(), c.State, []ArtifactQuestion{artifactQuestion(c)}, model.Options{})
	if err != nil {
		t.Fatal(err)
	}
	both, pred, err := art.Predict(context.Background(), c.State, []ArtifactQuestion{unscorable, artifactQuestion(c)}, model.Options{})
	if err != nil {
		t.Fatalf("an unscorable question must not fail the batch: %v", err)
	}
	u := both[0]
	if !u.Unscorable || u.Decision != DecisionEscalate || u.Probs != nil || u.Task != "q_metric_choice" || u.ID != "too-many" {
		t.Fatalf("unscorable answer = %+v", u)
	}
	if len(pred.Markers[0]) >= len(many) || len(pred.Sequences) != 2 {
		t.Fatalf("expected the markers of the unscorable question to be cut: %d of %d", len(pred.Markers[0]), len(many))
	}
	if both[1].Unscorable || !slices.Equal(both[1].Logits, alone[0].Logits) || both[1].Decision != alone[0].Decision {
		t.Fatalf("the scorable question changed when batched with an unscorable one")
	}
	only, _, err := art.Predict(context.Background(), c.State, []ArtifactQuestion{unscorable}, model.Options{})
	if err != nil || len(only) != 1 || !only[0].Unscorable {
		t.Fatalf("a batch of unscorable questions must escalate them: %+v %v", only, err)
	}
}

func TestPinnedFilesAreReadOnce(t *testing.T) {
	dir := t.TempDir()
	raw := []byte("pinned contents")
	if err := os.WriteFile(filepath.Join(dir, "f.bin"), raw, 0o600); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	want := hex.EncodeToString(sum[:])
	got, err := readPinned(dir, "f.bin", want, 1<<10)
	if err != nil || !bytes.Equal(got, raw) {
		t.Fatalf("readPinned = %q, %v", got, err)
	}
	if _, err := readPinned(dir, "f.bin", strings.Repeat("0", 64), 1<<10); err == nil || !strings.Contains(err.Error(), "sha256") {
		t.Fatalf("a digest mismatch must fail: %v", err)
	}
	if _, err := readPinned(dir, "f.bin", want, 4); err == nil || !strings.Contains(err.Error(), "larger") {
		t.Fatalf("an oversized file must fail: %v", err)
	}
	if _, err := readPinned(dir, "../f.bin", want, 1<<10); err == nil {
		t.Fatal("a non-local name must fail")
	}
	f, size, err := openPinned(dir, "f.bin", want)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	// the open file keeps the verified contents even if the path is replaced
	if err := os.WriteFile(filepath.Join(dir, "g.bin"), []byte("swapped in!!!!!"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(filepath.Join(dir, "g.bin"), filepath.Join(dir, "f.bin")); err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, size)
	if _, err := f.ReadAt(buf, 0); err != nil || !bytes.Equal(buf, raw) {
		t.Fatalf("the pinned file read %q after the swap, want %q", buf, raw)
	}
	if _, _, err := openPinned(dir, "f.bin", want); err == nil {
		t.Fatal("the swapped file must no longer verify")
	}
}

// copyArtifact copies an artifact directory so a test can tamper with it.
func copyArtifact(t *testing.T, src string) string {
	t.Helper()
	dst := t.TempDir()
	for _, name := range []string{"manifest.json", "model.safetensors", "parity_golden.json"} {
		raw, err := os.ReadFile(filepath.Join(src, name)) //nolint:gosec // test fixture path
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dst, name), raw, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return dst
}

func editManifest(t *testing.T, dir string, edit func(m map[string]any)) {
	t.Helper()
	path := filepath.Join(dir, "manifest.json")
	raw, err := os.ReadFile(path) //nolint:gosec // test temp path
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	edit(m)
	out, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, out, 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestLoadArtifactFailsClosed(t *testing.T) {
	src := artifactDir + "int8"
	section := func(m map[string]any, key string) map[string]any { return m[key].(map[string]any) }
	for _, tc := range []struct {
		name string
		edit func(t *testing.T, dir string)
		want string
	}{
		{"head tensors changed", func(t *testing.T, dir string) {
			path := filepath.Join(dir, "model.safetensors")
			raw, err := os.ReadFile(path) //nolint:gosec // test temp path
			if err != nil {
				t.Fatal(err)
			}
			raw[len(raw)-1] ^= 1
			if err := os.WriteFile(path, raw, 0o600); err != nil {
				t.Fatal(err)
			}
		}, "sha256"},
		{"another base", func(t *testing.T, dir string) {
			editManifest(t, dir, func(m map[string]any) { section(m, "base")["weights_sha256"] = strings.Repeat("0", 64) })
		}, "sha256"},
		{"another tokenizer", func(t *testing.T, dir string) {
			editManifest(t, dir, func(m map[string]any) { section(m, "tokenizer")["sha256"] = strings.Repeat("1", 64) })
		}, "sha256"},
		{"wrong contract", func(t *testing.T, dir string) {
			editManifest(t, dir, func(m map[string]any) { m["contract"] = "hive.laya-artifact/v0" })
		}, "contract"},
		{"file outside the base", func(t *testing.T, dir string) {
			editManifest(t, dir, func(m map[string]any) { section(m, "base")["weights_file"] = "../model.safetensors" })
		}, "local path"},
		{"too many layers", func(t *testing.T, dir string) {
			editManifest(t, dir, func(m map[string]any) { section(m, "base")["layers_used"] = 6 })
		}, "layers_used"},
		{"special token moved", func(t *testing.T, dir string) {
			editManifest(t, dir, func(m map[string]any) {
				section(section(section(m, "tokenizer"), "special_tokens"), "mask")["id"] = 3
			})
		}, "mask token"},
		{"temperature out of range", func(t *testing.T, dir string) {
			editManifest(t, dir, func(m map[string]any) {
				section(section(m, "calibration"), "temperatures")["*"] = map[string]any{"t": 0.1, "n": 5}
			})
		}, "temperature"},
		{"threshold off the grid", func(t *testing.T, dir string) {
			editManifest(t, dir, func(m map[string]any) {
				section(section(m, "act"), "thresholds")["q_operation"] = map[string]any{"score": "max_probability", "lambda": 0.12345, "certified": true}
			})
		}, "grid"},
		{"unknown score", func(t *testing.T, dir string) {
			editManifest(t, dir, func(m map[string]any) {
				section(section(m, "act"), "thresholds")["q_operation"] = map[string]any{"score": "vibes", "lambda": nil, "certified": false}
			})
		}, "score"},
		{"task type mismatch", func(t *testing.T, dir string) {
			editManifest(t, dir, func(m map[string]any) { section(section(m, "tasks"), "q_operation")["qtype"] = "noul" })
		}, "qtype"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := copyArtifact(t, src)
			tc.edit(t, dir)
			_, err := LoadArtifact(dir, artifactBaseDir, ArtifactOptions{})
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want an error mentioning %q", err, tc.want)
			}
		})
	}
	if _, err := LoadArtifact(copyArtifact(t, src), t.TempDir(), ArtifactOptions{}); err == nil {
		t.Fatal("a missing base directory must not load")
	}
}

// TestExternalArtifactParity checks a real exported artifact against its
// own goldens, e.g. one trained on mmBERT-base:
//
//	LAYAGO_ARTIFACT_DIR=/path/artifact/float32 LAYAGO_ARTIFACT_BASE_DIR=/path/mmBERT-base \
//	    go test -timeout 60m -run TestExternalArtifactParity ./internal/layago/laya
func TestExternalArtifactParity(t *testing.T) {
	dir, base := os.Getenv("LAYAGO_ARTIFACT_DIR"), os.Getenv("LAYAGO_ARTIFACT_BASE_DIR")
	if dir == "" || base == "" {
		t.Skip("set LAYAGO_ARTIFACT_DIR and LAYAGO_ARTIFACT_BASE_DIR to check a real artifact")
	}
	art, err := LoadArtifact(dir, base, ArtifactOptions{})
	if err != nil {
		t.Fatal(err)
	}
	g := readArtifactGolden(t, dir)
	l, p := checkArtifactParity(t, art, g, tensor.Parallel{})
	t.Logf("%s (%s, %d of %d layers): %d cases, max |Δ logit| %.3g, max |Δ p| %.3g", dir, art.Manifest.Weights.Format,
		art.Manifest.Base.LayersUsed, art.Manifest.Base.LayersTotal, len(g.Cases), l, p)
}
