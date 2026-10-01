package train

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/UnicoLab/ul_layago/laya"
	"github.com/UnicoLab/ul_layago/model"
	"github.com/UnicoLab/ul_layago/safetensors"
	"github.com/UnicoLab/ul_layago/tokenizer"
)

// convergedManifestErrors applies, to an exported artifact directory, every
// check the converged loaders make before they serve it:
//
//   - laya.LoadArtifact (w10/laya-train 5d048e53, internal/layago/laya/
//     artifact.go): ArtifactManifest.validate, compileTask, the pinned
//     sha256 of the head, base config, base weights and tokenizer, the base
//     config's depth and width, the special-token ids, and loadHead's
//     tensor names, shapes and dtypes with an empty encoder prefix;
//   - routermodel.Load (w10/laya-integrate a509f540, internal/routermodel/
//     artifact.go): parseReview (adapter, dataset contract, the training data
//     digest recomputed from training.files, option counts) and, for Q,
//     checkQTasks.
//
// Those loaders are in this tree now; TestConvergedLoadersServeGoArtifacts
// runs the real ones when LAYAGO_CONVERGED_TREES names a module root. This
// mirror keeps the contract checked in every test run.
func convergedManifestErrors(dir, baseDir string) []string {
	var errs []string
	fail := func(format string, args ...any) { errs = append(errs, fmt.Sprintf(format, args...)) }
	raw, err := os.ReadFile(filepath.Join(dir, ArtifactManifest)) //nolint:gosec // G304: test artifact
	if err != nil {
		return []string{err.Error()}
	}
	var m struct {
		Contract string `json:"contract"`
		Instance string `json:"instance"`
		Base     struct {
			ModelID, ConfigFile, ConfigSHA256, WeightsFile, WeightsSHA256, TensorPrefix string
			LayersTotal, LayersUsed, HiddenSize                                         int
		} `json:"-"`
		BaseRaw   map[string]any `json:"base"`
		Tokenizer struct {
			File          string `json:"file"`
			SHA256        string `json:"sha256"`
			SpecialTokens map[string]struct {
				Token string `json:"token"`
				ID    int32  `json:"id"`
			} `json:"special_tokens"`
		} `json:"tokenizer"`
		Sequence struct {
			Format       string `json:"format"`
			PromptFormat string `json:"prompt_format"`
			MaxLen       int    `json:"max_len"`
			HeadMaxLen   int    `json:"head_max_len"`
			TruncateLeft bool   `json:"truncate_left"`
		} `json:"sequence"`
		Head struct {
			Layers     int      `json:"layers"`
			ActOutputs []string `json:"act_outputs"`
		} `json:"head"`
		Weights struct {
			File   string `json:"file"`
			SHA256 string `json:"sha256"`
			Format string `json:"format"`
		} `json:"weights"`
		Adapter json.RawMessage `json:"adapter"`
		Tasks   map[string]struct {
			TaskType     string          `json:"task_type"`
			QType        string          `json:"qtype"`
			Instructions string          `json:"instructions"`
			Options      string          `json:"options"`
			FixedOptions json.RawMessage `json:"fixed_options"`
			OptionCounts []int           `json:"option_counts"`
		} `json:"tasks"`
		Calibration struct {
			Temperatures map[string]struct {
				T float64 `json:"t"`
				N int     `json:"n"`
			} `json:"temperatures"`
		} `json:"calibration"`
		Act struct {
			Thresholds map[string]struct {
				Score     string   `json:"score"`
				Lambda    *float64 `json:"lambda"`
				Certified bool     `json:"certified"`
			} `json:"thresholds"`
		} `json:"act"`
		Training struct {
			DatasetContract string        `json:"dataset_contract"`
			DataDigest      string        `json:"data_digest"`
			Files           []DatasetFile `json:"files"`
		} `json:"training"`
	}
	if err := json.Unmarshal(raw, &m); err != nil {
		return []string{err.Error()}
	}
	str := func(k string) string { s, _ := m.BaseRaw[k].(string); return s }
	num := func(k string) int { f, _ := m.BaseRaw[k].(float64); return int(f) }
	b := &m.Base
	b.ModelID, b.ConfigFile, b.ConfigSHA256, b.WeightsFile = str("model_id"), str("config_file"), str("config_sha256"), str("weights_file")
	b.WeightsSHA256, b.TensorPrefix = str("weights_sha256"), str("tensor_prefix")
	b.LayersTotal, b.LayersUsed, b.HiddenSize = num("layers_total"), num("layers_used"), num("hidden_size")

	// laya ArtifactManifest.validate.
	switch {
	case m.Contract != "hive.laya-artifact/v1":
		fail("contract %q", m.Contract)
	case m.Instance != "Q" && m.Instance != "R" && m.Instance != "A":
		fail("instance %q", m.Instance)
	case m.Sequence.Format != "laya-build-sequence/0.3.5" || m.Sequence.PromptFormat != "hive.laya-prompt/v1":
		fail("sequence format %q/%q", m.Sequence.Format, m.Sequence.PromptFormat)
	case m.Sequence.TruncateLeft:
		fail("truncate_left")
	case m.Sequence.MaxLen <= 0 || m.Sequence.HeadMaxLen <= 0:
		fail("sequence limits")
	case m.Head.Layers < 0:
		fail("head layers")
	case !slices.Equal(m.Head.ActOutputs, []string{"act", "escalate"}):
		fail("act outputs %v", m.Head.ActOutputs)
	case m.Weights.Format != "float32" && m.Weights.Format != "int8":
		fail("weights format %q", m.Weights.Format)
	case b.LayersUsed < 1 || b.LayersUsed > b.LayersTotal:
		fail("layers_used %d of %d", b.LayersUsed, b.LayersTotal)
	case len(m.Tasks) == 0:
		fail("no tasks")
	}
	for key, t := range m.Calibration.Temperatures {
		if math.IsNaN(t.T) || t.T < laya.TempMin || t.T > laya.TempMax {
			fail("temperature %s = %v", key, t.T)
		}
	}
	for task, th := range m.Act.Thresholds {
		if _, ok := m.Tasks[task]; !ok {
			fail("threshold for unknown task %s", task)
		}
		if th.Score != "max_probability" && th.Score != "act_probability" {
			fail("task %s score %q", task, th.Score)
		}
		if th.Certified && (th.Lambda == nil || *th.Lambda < 0 || *th.Lambda > 1 ||
			math.Abs(*th.Lambda*1e4-math.Round(*th.Lambda*1e4)) > 1e-6) {
			fail("task %s certified threshold off the grid", task)
		}
	}
	// laya compileTask.
	for name, t := range m.Tasks {
		ok := (t.TaskType == "choice" && t.QType == "choice") || (t.TaskType == "score" && t.QType == "score") ||
			(t.TaskType == "yes_no" && t.QType == "noul")
		if !ok {
			fail("task %s has task_type %q and qtype %q", name, t.TaskType, t.QType)
		}
		if len(t.FixedOptions) == 0 || string(t.FixedOptions) == "null" {
			continue
		}
		if t.TaskType == "choice" {
			var opts []struct{ Key, Text string }
			if json.Unmarshal(t.FixedOptions, &opts) != nil || len(opts) < 2 {
				fail("task %s fixed choice options %s", name, t.FixedOptions)
			}
		} else {
			var texts []string
			if json.Unmarshal(t.FixedOptions, &texts) != nil || (t.TaskType == "yes_no" && len(texts) != 2) {
				fail("task %s fixed options %s", name, t.FixedOptions)
			}
		}
	}
	// Pinned files: local names, digests.
	pinned := func(root, name, want string) []byte {
		if name == "" || !filepath.IsLocal(name) {
			fail("file name %q is not local", name)
			return nil
		}
		data, err := os.ReadFile(filepath.Join(root, name)) //nolint:gosec // G304: test artifact
		if err != nil {
			fail("%v", err)
			return nil
		}
		sum := sha256.Sum256(data)
		if got := hex.EncodeToString(sum[:]); !strings.EqualFold(got, want) {
			fail("%s has sha256 %s, manifest pins %s", name, got, want)
		}
		return data
	}
	head := pinned(dir, m.Weights.File, m.Weights.SHA256)
	cfgRaw := pinned(baseDir, b.ConfigFile, b.ConfigSHA256)
	tokRaw := pinned(baseDir, m.Tokenizer.File, m.Tokenizer.SHA256)
	pinned(baseDir, b.WeightsFile, b.WeightsSHA256)
	if cfgRaw != nil {
		full, err := model.ParseHFConfig(cfgRaw)
		if err != nil || full.NumLayers != b.LayersTotal || full.HiddenSize != b.HiddenSize {
			fail("base config %v: %d layers of %d, manifest %d of %d", err, full.NumLayers, full.HiddenSize, b.LayersTotal, b.HiddenSize)
		}
	}
	if tokRaw != nil {
		tok, err := tokenizer.Load(tokRaw)
		if err != nil {
			fail("%v", err)
		} else {
			for _, name := range []string{"cls", "sep", "mask", "pad"} {
				st, ok := m.Tokenizer.SpecialTokens[name]
				if id, found := tok.TokenID(st.Token); !ok || st.Token == "" || !found || id != st.ID {
					fail("special token %s %+v", name, st)
				}
			}
		}
	}
	// laya loadHead with an empty prefix: every tensor used, exact shapes,
	// I8 only on linear weights with a per-row F32 scale.
	if head != nil {
		f, err := safetensors.Parse(head)
		if err != nil {
			fail("%v", err)
		} else {
			d := b.HiddenSize
			want := map[string][]int{"type_emb.weight": {3, d}, "scorer.0.weight": {d}, "scorer.0.bias": {d},
				"scorer.1.weight": {d, d}, "scorer.1.bias": {d}, "scorer.3.weight": {1, d}, "scorer.3.bias": {1},
				"act_head.0.weight": {256, d + 4}, "act_head.0.bias": {256}, "act_head.2.weight": {len(m.Head.ActOutputs), 256},
				"act_head.2.bias": {len(m.Head.ActOutputs)}}
			for _, name := range f.Header.Names() {
				t, _ := f.Tensor(name)
				base, isScale := strings.CutSuffix(name, "_scale")
				switch shape, ok := want[name]; {
				case ok && !slices.Equal(t.Shape, shape):
					fail("%s has shape %v, want %v", name, t.Shape, shape)
				case ok && t.DType == safetensors.I8 && len(shape) != 2:
					fail("%s is I8 but not a linear weight", name)
				case ok && t.DType == safetensors.I8:
					if st, err := f.Tensor(name + "_scale"); err != nil || !slices.Equal(st.Shape, shape[:1]) || st.DType != safetensors.F32 {
						fail("%s has no per-row scale", name)
					}
				case isScale && len(want[base]) == 2:
					if w, _ := f.Tensor(base); w.DType != safetensors.I8 {
						fail("%s scales a weight that is not I8", name)
					}
				case !ok:
					fail("unexpected weight %s", name)
				}
			}
			for name := range want {
				if _, ok := f.Header.Lookup(name); !ok {
					fail("missing weight %s", name)
				}
			}
			if (m.Weights.Format == "int8") != slices.ContainsFunc(f.Header.Names(), func(n string) bool { return strings.HasSuffix(n, "_scale") }) {
				fail("weights format %s does not match the tensors", m.Weights.Format)
			}
		}
	}
	// routermodel parseReview.
	hex64 := regexp.MustCompile(`^[0-9a-f]{64}$`)
	if a := strings.TrimSpace(string(m.Adapter)); a != "null" {
		fail("adapter %q (routermodel requires null)", a)
	}
	if m.Training.DatasetContract != "hive.laya-dataset/v1" || !hex64.MatchString(m.Training.DataDigest) || len(m.Training.Files) == 0 {
		fail("training %+v", m.Training)
	}
	for _, f := range m.Training.Files {
		if f.Name == "" || !hex64.MatchString(f.SHA256) {
			fail("training file %+v", f)
		}
	}
	if got := DataDigest(m.Training.Files); got != m.Training.DataDigest {
		fail("training.data_digest %s does not follow from training.files (%s)", m.Training.DataDigest, got)
	}
	for task, t := range m.Tasks {
		for _, k := range t.OptionCounts {
			if k < 2 {
				fail("task %s option count %d", task, k)
			}
		}
	}
	// routermodel checkQTasks.
	if m.Instance == "Q" {
		profiles := map[string]struct {
			types   []string
			options string
		}{
			"q_operation": {[]string{"choice"}, "fixed"}, "q_metric_choice": {[]string{"choice"}, "per_example"},
			"q_scope_choice": {[]string{"choice"}, "per_example"}, "q_time_supported": {[]string{"yes_no"}, "fixed"},
			"q_single_series": {[]string{"yes_no"}, "fixed"}, "q_act_or_escalate": {[]string{"yes_no", "choice"}, "fixed"},
		}
		for name, t := range m.Tasks {
			if p, ok := profiles[name]; ok && (!slices.Contains(p.types, t.TaskType) || t.Options != p.options) {
				fail("Q task %s is %s with %s options; Hive serves it as %v with %s", name, t.TaskType, t.Options, p.types, p.options)
			}
		}
	}
	return errs
}

func checkConvergedManifest(t *testing.T, dir, baseDir string) {
	t.Helper()
	if errs := convergedManifestErrors(dir, baseDir); len(errs) > 0 {
		t.Fatalf("%s is not a converged hive.laya-artifact/v1 artifact:\n%s", dir, strings.Join(errs, "\n"))
	}
}

// TestConvergedManifestCheckCatchesTheOldLayout: the mirror is live. Each
// of the review's contract differences (no prompt_format, no base file
// fields, a raw-file data digest, a non-null adapter, int8 weights beside
// float32 ones) makes it fail.
func TestConvergedManifestCheckCatchesTheOldLayout(t *testing.T) {
	l := qLoop(t)
	for name, c := range map[string]struct {
		edit func(m map[string]any)
		want string
	}{
		"no prompt format":  {func(m map[string]any) { delete(m["sequence"].(map[string]any), "prompt_format") }, "sequence format"},
		"no weights file":   {func(m map[string]any) { delete(m["base"].(map[string]any), "weights_file") }, "is not local"},
		"wrong config file": {func(m map[string]any) { m["base"].(map[string]any)["config_file"] = "config.json" }, "config.json"},
		"file digest":       {func(m map[string]any) { m["training"].(map[string]any)["data_digest"] = strings.Repeat("0", 64) }, "does not follow"},
		"adapter":           {func(m map[string]any) { m["adapter"] = map[string]any{"kind": "lora"} }, "adapter"},
		"no training files": {func(m map[string]any) { delete(m["training"].(map[string]any), "files") }, "training"},
		"int8 in float32":   {func(m map[string]any) { m["weights"].(map[string]any)["format"] = "int8" }, "does not match the tensors"},
		"q task option": {func(m map[string]any) {
			m["tasks"].(map[string]any)["q_operation"].(map[string]any)["options"] = "per_example"
		}, "Hive serves it"},
	} {
		dir := filepath.Join(t.TempDir(), "art")
		copyDir(t, l.dirs[0], dir)
		raw, _ := os.ReadFile(filepath.Join(dir, ArtifactManifest))
		var m map[string]any
		if err := json.Unmarshal(raw, &m); err != nil {
			t.Fatal(err)
		}
		c.edit(m)
		out, _ := json.Marshal(m)
		if err := os.WriteFile(filepath.Join(dir, ArtifactManifest), out, 0o600); err != nil {
			t.Fatal(err)
		}
		errs := strings.Join(convergedManifestErrors(dir, tinyTrunkDir), "\n")
		if !strings.Contains(errs, c.want) {
			t.Errorf("%s: errors %q, want %q", name, errs, c.want)
		}
	}
}
