package train

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"

	"github.com/UnicoLab/ul_layago/safetensors"
	"github.com/UnicoLab/ul_layago/tensor"
)

// Artifact file names (one directory per numeric path).
const (
	ArtifactWeights  = "model.safetensors"
	ArtifactManifest = "manifest.json"
	ArtifactGolden   = "parity_golden.json"
)

// Golden tolerances: the converged contract's, which laya.LoadArtifact's
// float32 GEMM path must meet against this trainer's float64 head.
const (
	GoldenLogitAbs = 1e-4
	GoldenProbAbs  = 2e-5
)

// statesNote says which trunk states a head was calibrated and evaluated
// on: the Go encoder's float32 states, the numbers laya.LoadArtifact serves.
const statesNote = "float32 trunk states of the Go encoder (internal/layago/model), the numbers laya.LoadArtifact serves"

// Manifest is manifest.json of a hive.laya-artifact/v1 directory, in the
// converged layout (docs/architecture/intelligence-contracts.md; the Python
// exporter tools/laya-train/laya_train/export.py writes the same sections).
// laya.LoadArtifact reads contract, instance, base, tokenizer, sequence,
// head, weights, tasks, calibration and act; routermodel.Load also checks
// adapter, training (dataset contract, files and data digest) and reports
// evaluation and limitations.
type Manifest struct {
	Contract     string               `json:"contract"`
	Instance     string               `json:"instance"`
	EvidenceTier string               `json:"evidence_tier"`
	Authority    string               `json:"authority"`
	CodeVersion  CodeVersion          `json:"code_version"`
	Base         TrunkIdentity        `json:"base"`
	Tokenizer    TokenizerInfo        `json:"tokenizer"`
	Sequence     SequenceInfo         `json:"sequence"`
	Head         HeadInfo             `json:"head"`
	Weights      WeightsInfo          `json:"weights"`
	Adapter      json.RawMessage      `json:"adapter"`
	Tasks        map[string]*TaskSpec `json:"tasks"`
	Calibration  CalibrationInfo      `json:"calibration"`
	Act          ActInfo              `json:"act"`
	Training     TrainingInfo         `json:"training"`
	Evaluation   EvaluationInfo       `json:"evaluation"`
	ParityGolden GoldenInfo           `json:"parity_golden"`
	Limitations  []string             `json:"limitations"`
}

// CodeVersion identifies the code that trained the head.
type CodeVersion struct {
	Trainer   string `json:"trainer"`
	GitCommit string `json:"git_commit"`
	GitDirty  bool   `json:"git_dirty"`
	GoVersion string `json:"go_version"`
}

// TokenizerInfo pins the tokenizer.
type TokenizerInfo struct {
	File          string                  `json:"file"`
	SHA256        string                  `json:"sha256"`
	SpecialTokens map[string]SpecialToken `json:"special_tokens"`
}

// SequenceInfo is the layout the head was trained on.
type SequenceInfo struct {
	Format       string `json:"format"`
	PromptFormat string `json:"prompt_format"`
	MaxLen       int    `json:"max_len"`
	HeadMaxLen   int    `json:"head_max_len"`
	TruncateLeft bool   `json:"truncate_left"`
	// Shortlist, when present, is the catalog shortlist every input is
	// laid out over before rendering (CatalogShortlist); a runtime that
	// does not apply it must refuse the artifact.
	Shortlist *ShortlistSpec `json:"catalog_shortlist,omitempty"`
}

// HeadInfo describes the head: Laya's DecisionModel head with zero head
// transformer layers (see Head).
type HeadInfo struct {
	Layers         int      `json:"layers"`
	AttentionHeads int      `json:"attention_heads"`
	Scoring        string   `json:"scoring"`
	Pooling        string   `json:"pooling"`
	ActOutputs     []string `json:"act_outputs"`
	ActFeatures    []string `json:"act_features"`
	ScorerHidden   int      `json:"scorer_hidden"`
	ActHidden      int      `json:"act_hidden"`
	// FeatureKernel is the GEMM kernel the trunk states were computed with:
	// serving reproduces the logits bit for bit on this kernel.
	FeatureKernel string `json:"feature_kernel"`
}

// TensorShape is one tensor's dtype and shape.
type TensorShape struct {
	DType string `json:"dtype"`
	Shape []int  `json:"shape"`
}

// WeightsInfo pins model.safetensors.
type WeightsInfo struct {
	File    string                 `json:"file"`
	SHA256  string                 `json:"sha256"`
	Format  string                 `json:"format"`
	Tensors map[string]TensorShape `json:"tensors"`
}

// CalibrationInfo is the manifest's calibration section.
type CalibrationInfo struct {
	Method       string       `json:"method"`
	FitSplit     string       `json:"fit_split"`
	FittedOn     string       `json:"fitted_on"`
	States       string       `json:"states"`
	Clamp        [2]float64   `json:"clamp"`
	MinCell      int          `json:"min_cell"`
	LookupOrder  []string     `json:"lookup_order"`
	Buckets      []string     `json:"buckets"`
	Temperatures Temperatures `json:"temperatures"`
	// FitLangs are the languages of the dev_cal items the temperatures were
	// fitted on (absent: every language).
	FitLangs []string `json:"fit_langs,omitempty"`
}

// ActInfo is the manifest's act section.
type ActInfo struct {
	Method             string               `json:"method"`
	FitSplit           string               `json:"fit_split"`
	FittedOn           string               `json:"fitted_on"`
	States             string               `json:"states"`
	Risk               string               `json:"risk"`
	ProbabilityQuantum float64              `json:"probability_quantum"`
	Quantize           string               `json:"quantize"`
	Alpha              float64              `json:"alpha"`
	Delta              float64              `json:"delta"`
	NMin               int                  `json:"n_min"`
	Thresholds         map[string]Threshold `json:"thresholds"`
	Uncertified        string               `json:"uncertified"`
	// FitLangs are the languages of the dev_conf items the thresholds were
	// certified on (absent: every language); StartRule says where each
	// fixed sequence starts.
	FitLangs  []string `json:"fit_langs,omitempty"`
	StartRule string   `json:"start_rule,omitempty"`
}

// SplitRecord describes one split of the training data.
type SplitRecord struct {
	Examples     int            `json:"examples"`
	IDsSHA256    string         `json:"ids_sha256"`
	Languages    map[string]int `json:"languages"`
	Domains      map[string]int `json:"domains"`
	Tasks        map[string]int `json:"tasks"`
	LabelSources map[string]int `json:"label_sources"`
}

// TrainingInfo is the lineage of a trained head.
type TrainingInfo struct {
	DatasetContract   string         `json:"dataset_contract"`
	DataDigest        string         `json:"data_digest"`
	Files             []DatasetFile  `json:"files"`
	Splits            map[string]any `json:"splits"`
	DevHalves         DevHalfReport  `json:"dev_halves"`
	SoftLabelled      int            `json:"soft_labelled_examples"`
	Unscorable        map[string]int `json:"unscorable"`
	GeneratorVersions []string       `json:"generator_versions"`
	LLMProvenance     map[string]any `json:"llm_provenance"`
	Duplicates        map[string]int `json:"rendered_duplicates_train_vs_eval"`
	SplitInvariants   map[string]any `json:"split_invariants"`
	Drift             map[string]any `json:"calibration_drift_dev_conf"`
	LockedTest        bool           `json:"locked_test_evaluated"`
	Config            Config         `json:"config"`
	History           []EpochStats   `json:"history"`
	BestEpoch         int            `json:"best_epoch"`
	ActHead           ActReport      `json:"act_head"`
	Timing            map[string]any `json:"timing"`
	FeatureCache      CacheSummary   `json:"feature_cache"`
	Loss              string         `json:"loss"`
	// TrainSubsample is present when only a subsample of the train split
	// was encoded and fitted (PrepareOptions.TrainFraction); Files, the data
	// digest and Splits still describe the whole dataset.
	TrainSubsample *Subsample `json:"train_subsample,omitempty"`
	// EvalLangs is present when the evaluated splits were restricted to
	// these languages (PrepareOptions.EvalLangs).
	EvalLangs []string `json:"eval_langs,omitempty"`
}

// CacheSummary is the feature store and view a head was trained from.
type CacheSummary struct {
	Key            string     `json:"cache_id"`
	Contract       string     `json:"contract,omitempty"`
	SequenceDigest string     `json:"sequence_digest,omitempty"`
	DType          string     `json:"dtype"`
	Examples       int        `json:"examples"`
	Sequences      int        `json:"distinct_sequences,omitempty"`
	Skipped        int        `json:"skipped"`
	SeqLenP50      int        `json:"sequence_tokens_p50"`
	SeqLenP95      int        `json:"sequence_tokens_p95"`
	SeqLenMean     float64    `json:"sequence_tokens_mean,omitempty"`
	SeqLenMax      int        `json:"sequence_tokens_max,omitempty"`
	Throughput     Throughput `json:"last_build_throughput"`
}

// EvaluationInfo is the manifest's evaluation section.
type EvaluationInfo struct {
	ECEBins        int                           `json:"ece_bins"`
	States         string                        `json:"states"`
	Splits         map[string]SplitEval          `json:"splits"`
	Generalisation map[string]map[string]Summary `json:"generalisation"`
	DeltaVsFloat32 *Int8Delta                    `json:"delta_vs_float32,omitempty"`
}

// GoldenInfo pins parity_golden.json.
type GoldenInfo struct {
	File      string    `json:"file"`
	SHA256    string    `json:"sha256"`
	Cases     int       `json:"cases"`
	Tolerance Tolerance `json:"tolerance"`
}

// Tolerance is a golden's tolerance.
type Tolerance struct {
	LogitAbs float64 `json:"logit_abs"`
	ProbAbs  float64 `json:"prob_abs"`
}

// ExportMeta is what Export needs beyond the training result.
type ExportMeta struct {
	Code  CodeVersion
	Trunk *Trunk
	Cache *Cache
	// GoldenPerTask is the number of golden cases per task (default 3).
	GoldenPerTask int
}

// Export writes one hive.laya-artifact/v1 directory per numeric path under
// dir: dir/float32 and, when the head was trained with Int8, dir/int8. Each
// holds model.safetensors (the head under Laya's tensor names; in the int8
// directory every linear weight is I8 with a per-row F32 "<name>_scale"),
// manifest.json and parity_golden.json. It returns the directories.
func Export(dir string, res *Result, meta ExportMeta) ([]string, error) {
	variants := []*Variant{res.Float32}
	if res.Int8 != nil {
		variants = append(variants, res.Int8)
	}
	var out []string
	for _, v := range variants {
		d := filepath.Join(dir, v.Format)
		if err := exportVariant(d, res, v, meta); err != nil {
			return nil, err
		}
		out = append(out, d)
	}
	return out, nil
}

// headTensors lays a head out as safetensors: every parameter F32, or, for
// the int8 format, each linear weight quantised with its per-row scale.
func headTensors(h *Head, format string) []safetensors.NamedTensor {
	var out []safetensors.NamedTensor
	for _, p := range h.Params() {
		if format == "int8" && isLinearWeight(p) {
			q, scale := tensor.QuantizeRows(p.Data, p.Shape[0], p.Shape[1])
			out = append(out, safetensors.Int8Tensor(p.Name, p.Shape, q), safetensors.Float32Tensor(p.Name+"_scale", []int{p.Shape[0]}, scale))
			continue
		}
		out = append(out, safetensors.Float32Tensor(p.Name, p.Shape, p.Data))
	}
	return out
}

func exportVariant(dir string, res *Result, v *Variant, meta ExportMeta) error {
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return err
	}
	p := res.Prepared
	hc := v.Head.Config
	if hc.Hidden != hc.D || hc.ActHidden != layaActHidden || hc.NAct != len(ActOutputs) {
		return fmt.Errorf("train: head %+v is not Laya's shape (scorer width d, act width %d)", hc, layaActHidden)
	}
	tensors := headTensors(v.Head, v.Format)
	var buf bytes.Buffer
	if err := safetensors.Write(&buf, tensors, map[string]string{"laya_artifact": ArtifactContract + " " + v.Format}); err != nil {
		return err
	}
	if err := writeAtomic(filepath.Join(dir, ArtifactWeights), buf.Bytes()); err != nil {
		return err
	}
	wsum := sha256.Sum256(buf.Bytes())
	shapes := map[string]TensorShape{}
	for _, t := range tensors {
		dt := "float32"
		if t.DType == safetensors.I8 {
			dt = "int8"
		}
		shapes[t.Name] = TensorShape{DType: dt, Shape: t.Shape}
	}
	golden := buildGolden(res, v, meta)
	graw, err := json.MarshalIndent(golden, "", " ")
	if err != nil {
		return err
	}
	if err := writeAtomic(filepath.Join(dir, ArtifactGolden), append(graw, '\n')); err != nil {
		return err
	}
	gsum := sha256.Sum256(append(graw, '\n'))

	trunk, cache := meta.Trunk, meta.Cache
	fittedOn := "the float32 head"
	if v.Format == "int8" {
		fittedOn = "the dequantised int8 head"
	}
	code := meta.Code
	code.Trainer, code.GoVersion = TrainerVersion, runtime.Version()
	m := Manifest{
		Contract: ArtifactContract, Instance: p.Instance, EvidenceTier: EvidenceTier, Authority: "nomination_only", CodeVersion: code,
		Base: trunk.Identity,
		Tokenizer: TokenizerInfo{File: trunk.Identity.TokenizerFile, SHA256: trunk.Identity.TokenizerSHA256,
			SpecialTokens: trunk.SpecialTokenManifest()},
		Sequence: SequenceInfo{Format: SequenceFormat, PromptFormat: PromptFormat, MaxLen: trunk.Spec.MaxLen, HeadMaxLen: trunk.Spec.HeadMaxLen,
			Shortlist: p.Shortlist},
		Head: HeadInfo{Layers: 0, AttentionHeads: max(1, hc.D/64), Scoring: "laya_mask_marker_per_option_softmax",
			Pooling: "position_0_after_head_layers", ActOutputs: ActOutputs,
			ActFeatures: []string{"top1", "top1_minus_top2", "normalised_entropy", "k_over_255"}, ScorerHidden: hc.Hidden,
			ActHidden: hc.ActHidden, FeatureKernel: cache.Info.Kernel},
		Weights: WeightsInfo{File: ArtifactWeights, SHA256: hex.EncodeToString(wsum[:]), Format: v.Format, Tensors: shapes},
		Adapter: json.RawMessage("null"),
		Tasks:   p.Tasks,
		Calibration: CalibrationInfo{Method: "temperature_per_task_option_bucket", FitSplit: RoleDevCal, FittedOn: fittedOn,
			States: statesNote, Clamp: [2]float64{0.5, 5}, MinCell: res.Config.MinCell,
			LookupOrder: []string{"<task>:<bucket>", "<task>", "*", "1.0"}, Buckets: []string{"2", "3-5", "6-10", "11+"},
			Temperatures: v.Temperatures, FitLangs: res.Config.FitLangs},
		Act: ActInfo{Method: "learn_then_test_fixed_sequence_binomial", FitSplit: RoleDevConf, FittedOn: fittedOn, States: statesNote,
			Risk: "error rate among acted examples", ProbabilityQuantum: ProbabilityQuantum, Quantize: "floor(p * 10000 + 0.5) / 10000",
			Alpha: res.Config.Alpha, Delta: res.Config.Delta, NMin: res.Config.NMin, Thresholds: v.Thresholds, Uncertified: "escalate",
			FitLangs: res.Config.FitLangs, StartRule: ltStartRule(res.Config)},
		Training: trainingRecord(res, cache),
		Evaluation: EvaluationInfo{ECEBins: ECEBins, States: statesNote + ", every split including train", Splits: v.Evaluation,
			Generalisation: v.Generalise, DeltaVsFloat32: v.Delta},
		ParityGolden: GoldenInfo{File: ArtifactGolden, SHA256: hex.EncodeToString(gsum[:]), Cases: len(golden.Cases),
			Tolerance: Tolerance{LogitAbs: GoldenLogitAbs, ProbAbs: GoldenProbAbs}},
		Limitations: limitations(p, res.DevHalves),
	}
	raw, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	return writeAtomic(filepath.Join(dir, ArtifactManifest), append(raw, '\n'))
}

// ltStartRule describes where each task's Learn-then-Test sequence starts.
func ltStartRule(c Config) string {
	n := LTTStart(c.Alpha, c.Delta/2, c.NMin, c.LTTStartErrors)
	if c.LTTStartErrors <= 0 {
		return fmt.Sprintf("max(n_min, ceil(3 ln(delta/2) / ln(1 - alpha))) = %d acted items", n)
	}
	return fmt.Sprintf("max(n_min, fewest acted items at which %d error(s) can reject at delta/2) = %d acted items", c.LTTStartErrors, n)
}

func limitations(p *Prepared, halves DevHalfReport) []string {
	out := []string{
		EvidenceTier + " evidence: trained and evaluated on generated data only",
		"outputs are nominations with no numeric or evidence authority",
		"temperatures and thresholds are synthetic calibration until verified labels exist",
	}
	if n := len(p.Dataset.SplitViolations); n > 0 {
		out = append(out, fmt.Sprintf("split invariants overridden (%d violations in training.split_invariants): held-out metrics "+
			"and the act-threshold certificate assume independent splits and are not valid", n))
	}
	if n := len(halves.Shortfalls); n > 0 {
		out = append(out, fmt.Sprintf("dev is too small for part of the calibration (%d shortfalls in training.dev_halves): the tasks "+
			"named there fall back to a wider temperature or cannot certify an act threshold and always escalate", n))
	}
	if n := sumValues(p.Duplicates); n > 0 {
		out = append(out, fmt.Sprintf("%d evaluated examples render identically to train examples (training.rendered_duplicates_train_vs_eval)", n))
	}
	if s := p.TrainSubsample; s != nil {
		out = append(out, fmt.Sprintf("trained on a stratified subsample of train: %d of %d instance-%s train examples (fraction %v, "+
			"seed %d, training.train_subsample); train metrics cover only those, the evaluated splits are whole", s.KeptExamples,
			s.SourceExamples, p.Instance, s.Fraction, s.Seed))
	}
	if len(p.EvalLangs) > 0 {
		out = append(out, fmt.Sprintf("the evaluated splits were restricted to %v (training.eval_langs): no other language is evaluated", p.EvalLangs))
	}
	if len(halves.FitLangs) > 0 {
		out = append(out, fmt.Sprintf("early stopping, the act head, the temperatures and the act thresholds were fitted on dev items in %v "+
			"only: other languages are served under calibration and thresholds fitted on those", halves.FitLangs))
	}
	if s := p.Shortlist; s != nil {
		out = append(out, fmt.Sprintf("inputs are laid out over a lexical catalog shortlist (sequence.catalog_shortlist, k=%d, %s): a runtime "+
			"that does not apply the same shortlist must refuse this artifact, and an item whose gold option the shortlist dropped is "+
			"always wrong", s.K, s.Method))
	}
	return out
}

func sumValues(m map[string]int) int {
	n := 0
	for _, v := range m {
		n += v
	}
	return n
}

func idDigest(ids []string) string {
	s := slices.Clone(ids)
	slices.Sort(s)
	sum := sha256.Sum256([]byte(strings.Join(s, "\n")))
	return hex.EncodeToString(sum[:])
}

// trainingRecord describes the data the head was trained and evaluated on.
// A sealed locked_test is published as its size and the files that hold it,
// nothing about its content (no language, domain, task or id digest), and is
// left out of every other count.
func trainingRecord(res *Result, cache *Cache) TrainingInfo {
	p := res.Prepared
	ds := p.Dataset
	splits := map[string]any{}
	gens := map[string]bool{}
	llm := map[string]bool{}
	llmExamples, soft := 0, 0
	for _, split := range Splits {
		var ids []string
		files := map[string]bool{}
		rec := SplitRecord{Languages: map[string]int{}, Domains: map[string]int{}, Tasks: map[string]int{}, LabelSources: map[string]int{}}
		for i := range ds.Examples {
			ex := &ds.Examples[i]
			if ex.Split != split {
				continue
			}
			ids = append(ids, ex.ID)
			files[filepath.Base(ex.File)] = true
			rec.Languages[ex.Lang]++
			rec.Domains[ex.DomainID]++
			rec.Tasks[ex.Task]++
			rec.LabelSources[ex.LabelSource]++
		}
		switch {
		case len(ids) == 0:
		case split == SplitLockedTest && p.Sealed:
			var held []map[string]string
			for _, f := range ds.Files {
				if files[f.Name] {
					held = append(held, map[string]string{"name": f.Name, "sha256": f.SHA256})
				}
			}
			splits[split] = map[string]any{"sealed": true, "examples": len(ids), "files": held}
		default:
			rec.Examples, rec.IDsSHA256 = len(ids), idDigest(ids)
			splits[split] = rec
		}
	}
	for i := range ds.Examples {
		ex := &ds.Examples[i]
		if ex.Split == SplitLockedTest && p.Sealed {
			continue
		}
		gens[ex.GeneratorVersion] = true
		if len(ex.Provenance) > 0 {
			llmExamples++
		}
		if ex.LabelDist != nil {
			soft++
		}
		for _, id := range ex.Provenance {
			llm[id] = true
		}
	}
	provenance := map[string]any{"examples": llmExamples, "interaction_ids": len(llm), "interaction_ids_sha256": nil}
	if len(llm) > 0 {
		provenance["interaction_ids_sha256"] = idDigest(sortedKeys(llm))
	}
	var headSecs float64
	for _, h := range res.History {
		headSecs += h.Seconds
	}
	th := cache.Info.Throughput
	device := "cpu (" + runtime.GOARCH + ")"
	scorerBackend := res.Config.Backend
	if scorerBackend == "" {
		scorerBackend = "go"
	}
	scorerDevice := device
	drift := map[string]any{"same_path": true, "note": "features, training, calibration and served inference share the Go float32 trunk path; with a float32 cache the served logits equal the trained ones bit for bit on the feature kernel"}
	if scorerBackend != "go" {
		scorerDevice = "external backend"
		if actual, ok := res.Config.BackendInfo["device"].(string); ok {
			scorerDevice = actual
		}
		drift = map[string]any{"same_path": false, "note": "encoder features are cached in Go; scorer fitting uses " + scorerBackend + " arithmetic. Final calibration/evaluation/serving use the same Go forward path; the MLX fitter additionally verifies exported development NLL in Go. Training is not bit-identical across backends."}
		device = "hybrid: scorer " + scorerDevice + "; encoder/act/calibration " + device
	}
	return TrainingInfo{
		DatasetContract: DatasetContract, DataDigest: ds.Digest(), Files: ds.Files, Splits: splits, DevHalves: res.DevHalves,
		SoftLabelled: soft, Unscorable: res.Unscorable, GeneratorVersions: sortedKeys(gens), LLMProvenance: provenance, Duplicates: p.Duplicates,
		SplitInvariants: map[string]any{
			"checked":    []string{"one split per story", "heldout_domain domains not in train/dev", "heldout_language languages not in train/dev"},
			"violations": nonNil(ds.SplitViolations), "overridden": len(ds.SplitViolations) > 0,
		},
		Drift:      drift,
		LockedTest: p.LockedTestEvaluated, Config: res.Config, History: res.History, BestEpoch: res.BestEpoch, ActHead: res.Act,
		Timing: map[string]any{
			"trunk_cache_seconds": th.Seconds, "trunk_sequences_computed": th.Examples, "trunk_sequences_per_second": th.ExamplesPerSecond,
			"trunk_tokens_per_cpu_second": th.TokensPerCPUSecond, "head_epochs": len(res.History), "head_seconds": headSecs,
			"train_and_calibrate_seconds": res.Seconds, "device": device,
			"scorer_backend": scorerBackend, "scorer_device": scorerDevice, "encoder_act_calibration_device": "cpu (" + runtime.GOARCH + ")",
		},
		FeatureCache: CacheSummary{Key: cache.Info.Key, Contract: CacheContract, SequenceDigest: cache.Info.SequenceDigest,
			DType: cache.Info.DType, Examples: cache.Info.Examples, Sequences: cache.Info.Sequences, Skipped: len(cache.Info.Skipped),
			SeqLenP50: cache.Info.SeqLenP50, SeqLenP95: cache.Info.SeqLenP95, SeqLenMean: cache.Info.SeqLenMean,
			SeqLenMax: cache.Info.SeqLenMax, Throughput: th},
		TrainSubsample: p.TrainSubsample,
		EvalLangs:      p.EvalLangs,
		Loss: fmt.Sprintf("choice and score: softmax cross-entropy against the label or its label_dist (listwise softmax "+
			"ranking loss over graded items); score adds %.2f x normalised expected-level squared error; yes_no: logistic "+
			"loss on the margin; act head: class-balanced cross-entropy", res.Config.ScoreRegWeight),
	}
}

func nonNil(s []string) []string {
	if s == nil {
		return []string{}
	}
	return s
}

// GoldenQuestion is the question as the goldens record it (prompt.py
// question_json): options for choice, levels for score, false and true for
// noul.
type GoldenQuestion struct {
	Type         string         `json:"type"`
	Instructions string         `json:"instructions"`
	Options      []GoldenOption `json:"options,omitempty"`
	Levels       []string       `json:"levels,omitempty"`
	False        *string        `json:"false,omitempty"`
	True         *string        `json:"true,omitempty"`
}

// GoldenOption is one choice option of a golden question.
type GoldenOption struct {
	Key         string `json:"key"`
	Description string `json:"description"`
}

// GoldenCase is one example the artifact must reproduce.
type GoldenCase struct {
	ID             string         `json:"id"`
	Task           string         `json:"task"`
	Lang           string         `json:"lang"`
	Split          string         `json:"split"`
	State          string         `json:"state"`
	Question       GoldenQuestion `json:"question"`
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

// Golden is parity_golden.json.
type Golden struct {
	Contract     string       `json:"contract"`
	Kind         string       `json:"kind"`
	ComputedWith string       `json:"computed_with"`
	Tolerance    Tolerance    `json:"tolerance"`
	Cases        []GoldenCase `json:"cases"`
}

func goldenQuestion(it *Item) GoldenQuestion {
	q := it.Question
	g := GoldenQuestion{Type: q.Type.String(), Instructions: q.Instructions}
	switch it.Ex.Type {
	case TaskChoice:
		for _, o := range q.Options {
			g.Options = append(g.Options, GoldenOption{Key: o.Key, Description: o.Description})
		}
	case TaskScore:
		g.Levels = q.Levels
	default:
		f, t := q.False, q.True
		g.False, g.True = &f, &t
	}
	return g
}

func float64s(v []float32) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x)
	}
	return out
}

// buildGolden picks up to GoldenPerTask scorable items per task, from
// dev_conf, heldout_language, heldout_domain and then dev_cal, each in id
// order (as export.py does), with the variant's served numbers.
func buildGolden(res *Result, v *Variant, meta ExportMeta) Golden {
	per := meta.GoldenPerTask
	if per <= 0 {
		per = 3
	}
	g := Golden{Contract: ArtifactContract, Kind: "parity_golden",
		ComputedWith: fmt.Sprintf("Go %s %s, float64 head over float32 trunk states (%s), %s head", TrainerVersion, runtime.Version(),
			meta.Cache.Info.Kernel, v.Format),
		Tolerance: Tolerance{LogitAbs: GoldenLogitAbs, ProbAbs: GoldenProbAbs}}
	rows := map[string]Row{}
	for _, r := range v.Rows {
		rows[r.ID] = r
	}
	count := map[string]int{}
	byRole := res.Prepared.ItemsByRole()
	for _, role := range []string{RoleDevConf, SplitHeldoutLanguage, SplitHeldoutDomain, RoleDevCal} {
		items := slices.Clone(byRole[role])
		slices.SortFunc(items, func(a, b *Item) int { return strings.Compare(a.Ex.ID, b.Ex.ID) })
		for _, it := range items {
			if !it.Scorable || count[it.Ex.Task] >= per {
				continue
			}
			r := rows[it.Ex.ID]
			dec := "escalate"
			if r.Acted {
				dec = "act"
			}
			g.Cases = append(g.Cases, GoldenCase{ID: it.Ex.ID, Task: it.Ex.Task, Lang: it.Ex.Lang, Split: it.Role, State: it.State,
				Question: goldenQuestion(it), IDs: it.IDs, Markers: it.Markers, Logits: float64s(v.Logits[it.Ex.ID]),
				ActLogits: float64s(v.ActLogits[it.Ex.ID]), Temperature: r.Temperature, TemperatureKey: r.TemperatureKey, Probs: r.Probs,
				ActProbability: r.ActProbability, DecisionScore: r.DecisionScore, Decision: dec, Prediction: r.PredKey})
			count[it.Ex.Task]++
		}
	}
	return g
}
