package train

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/UnicoLab/ul_layago/laya"
)

// TaskSpec is one trained task as the artifact manifest records it
// (prompt.py TaskSpec.manifest).
type TaskSpec struct {
	Name         string   `json:"-"`
	Type         TaskType `json:"task_type"`
	QType        string   `json:"qtype"`
	Instructions string   `json:"instructions"`
	// Options is "fixed" when every example of the task offers the same
	// options in the same order, else "per_example".
	Options string `json:"options"`
	// LabelSet is the class keys of a fixed task (always ["false", "true"]
	// for yes_no), else null.
	LabelSet []string `json:"label_set"`
	// FixedOptions are the exact option texts of a fixed task: [{"key",
	// "text"}] for choice, level texts for score, [false_text, true_text]
	// for yes_no (empty strings mean Laya's default criteria). The runtime
	// renders these when a caller passes no options.
	FixedOptions json.RawMessage `json:"fixed_options"`
	OptionCounts []int           `json:"option_counts"`
}

// BuildTaskSpecs defines one spec per task from the examples, as prompt.py
// build_task_specs does, and refuses a task whose examples disagree on its
// type. instructions overrides DefaultInstruction per task.
func BuildTaskSpecs(examples []*Example, instructions map[string]string) (map[string]*TaskSpec, error) {
	specs := map[string]*TaskSpec{}
	optsets := map[string]map[string]bool{}
	firstSet := map[string][]Option{}
	counts := map[string]map[int]bool{}
	for _, ex := range examples {
		spec := specs[ex.Task]
		if spec == nil {
			ins := instructions[ex.Task]
			if ins == "" {
				ins = DefaultInstruction(ex.Task)
			}
			spec = &TaskSpec{Name: ex.Task, Type: ex.Type, QType: ex.Type.QType().String(), Instructions: ins}
			specs[ex.Task] = spec
			optsets[ex.Task], counts[ex.Task] = map[string]bool{}, map[int]bool{}
		} else if spec.Type != ex.Type {
			return nil, fmt.Errorf("train: task %s has examples of types %s and %s", ex.Task, spec.Type, ex.Type)
		}
		opts := ex.Options
		if ex.Type == TaskYesNo && len(opts) == 0 {
			opts = []Option{{Key: "0"}, {Key: "1"}}
		}
		key, _ := json.Marshal(opts)
		if len(optsets[ex.Task]) == 0 {
			firstSet[ex.Task] = opts
		}
		optsets[ex.Task][string(key)] = true
		counts[ex.Task][ex.NumOptions()] = true
	}
	for name, spec := range specs {
		fixed := len(optsets[name]) == 1
		opts := firstSet[name]
		var fixedJSON any
		switch {
		case spec.Type == TaskYesNo:
			spec.LabelSet = []string{"false", "true"}
			if fixed {
				fixedJSON = []string{opts[0].Text, opts[1].Text}
			}
		case fixed && spec.Type == TaskChoice:
			spec.LabelSet = make([]string, len(opts))
			objs := make([]Option, len(opts))
			for i, o := range opts {
				spec.LabelSet[i] = o.Key
				objs[i] = o
			}
			fixedJSON = objs
		case fixed:
			spec.LabelSet = make([]string, len(opts))
			texts := make([]string, len(opts))
			for i, o := range opts {
				spec.LabelSet[i] = o.Key
				texts[i] = o.Text
			}
			fixedJSON = texts
		}
		spec.Options = "per_example"
		spec.FixedOptions = json.RawMessage("null")
		if fixedJSON != nil {
			spec.Options = "fixed"
			raw, err := json.Marshal(fixedJSON)
			if err != nil {
				return nil, err
			}
			spec.FixedOptions = raw
		}
		for k := range counts[name] {
			spec.OptionCounts = append(spec.OptionCounts, k)
		}
		slices.Sort(spec.OptionCounts)
	}
	return specs, nil
}

// Question is the Laya question an example is asked as (prompt.py
// render_question): the task's instructions and the example's options,
// levels or false/true criteria.
func (s *TaskSpec) Question(ex *Example) laya.Question {
	q := laya.Question{ID: ex.ID, Type: ex.Type.QType(), Instructions: s.Instructions}
	switch ex.Type {
	case TaskChoice:
		for _, o := range ex.Options {
			q.Options = append(q.Options, laya.Option{Key: o.Key, Description: o.Text})
		}
	case TaskScore:
		for _, o := range ex.Options {
			q.Levels = append(q.Levels, o.Text)
		}
	default:
		if len(ex.Options) == 2 {
			q.False, q.True = ex.Options[0].Text, ex.Options[1].Text
		}
	}
	return q
}

// Item is one example laid out for the trunk: its role, rendered state and
// question, and token sequence. It is Scorable when every option's [MASK]
// marker fits the sequence limits; an unscorable item is never trained on
// and is escalated (and counted) in evaluation, as the runtime escalates it.
type Item struct {
	Ex *Example
	// Source is the dataset example Ex was laid out from: Ex itself, or
	// its copy over a catalog shortlist. Serving starts from the source.
	Source   *Example
	Role     string
	State    string
	Question laya.Question
	IDs      []int32
	Markers  []int
	Scorable bool
}

// source is the example an item was laid out from (Ex for an item built
// without Source).
func (it *Item) source() *Example {
	if it.Source != nil {
		return it.Source
	}
	return it.Ex
}

// PrepareOptions control how a dataset becomes training items.
type PrepareOptions struct {
	// Instructions override the default instructions per task.
	Instructions map[string]string
	// UnsealLockedTest evaluates locked_test. It needs AccessLog, a JSONL
	// file every unsealing is appended to. A sealed locked_test shapes
	// nothing, not even the task definitions.
	UnsealLockedTest bool
	AccessLog        string
	// AccessNote is recorded with an unsealing (for example the output
	// directory).
	AccessNote string
	// AllowSplitLeakage trains despite split-invariant violations, which the
	// manifest then records with a limitation.
	AllowSplitLeakage bool
	// AllowDuplicates trains although rendered train examples reappear in
	// evaluated splits.
	AllowDuplicates bool
	// TrainFraction, when strictly between 0 and 1, keeps a stratified
	// subsample of train (SubsampleRule) when the trunk cannot encode
	// all of it within a compute budget. The other train examples are
	// neither encoded, trained on nor evaluated; dev and the held-out splits
	// are never subsampled, and the task definitions still come from every
	// visible example. 0 and 1 keep all of train.
	TrainFraction float64
	// SubsampleSeed salts the subsample's ranking.
	SubsampleSeed uint64
	// EvalLangs, when set, keeps only examples of these languages (primary
	// subtags) in the evaluated splits (dev and the held-out splits); train
	// is untouched. A design experiment on a subset of languages then
	// encodes nothing else. The task definitions still come from every
	// visible example.
	EvalLangs []string
	// Shortlist, when K > 0, lays every example out over a lexical
	// shortlist of its catalog (CatalogShortlist) instead of the whole
	// catalog.
	Shortlist ShortlistSpec
}

// SubsampleRule is how PrepareOptions.TrainFraction picks train
// examples: deterministic, stratified and independent of file order.
const SubsampleRule = "stratified by (task, lang, domain_id): in each stratum of n train examples keep the " +
	"ceil(fraction * n) with the smallest sha256(\"laya-train-subsample\\x00\" + decimal seed + \"\\x00\" + id), ties by id"

// Subsample records the subsample of train a head was fitted on
// (manifest training.train_subsample). The dataset's files and data digest
// stay those of the whole dataset.
type Subsample struct {
	Rule            string         `json:"rule"`
	Fraction        float64        `json:"fraction"`
	Seed            uint64         `json:"seed"`
	Strata          int            `json:"strata"`
	SourceExamples  int            `json:"source_examples"`
	KeptExamples    int            `json:"kept_examples"`
	SourceIDsSHA256 string         `json:"source_ids_sha256"`
	KeptIDsSHA256   string         `json:"kept_ids_sha256"`
	KeptByTask      map[string]int `json:"kept_by_task"`
	KeptByLang      map[string]int `json:"kept_by_lang"`
	KeptByDomain    map[string]int `json:"kept_by_domain"`
}

// subsampleRank orders train examples within a stratum.
func subsampleRank(seed uint64, id string) string {
	sum := sha256.Sum256([]byte("laya-train-subsample\x00" + strconv.FormatUint(seed, 10) + "\x00" + id))
	return string(sum[:])
}

// subsampleTrain applies SubsampleRule to the train examples among
// examples and keeps every other example, in the original order. It returns
// examples unchanged, and a nil record, when fraction is 0 or 1.
func subsampleTrain(examples []*Example, fraction float64, seed uint64) ([]*Example, *Subsample, error) {
	if math.IsNaN(fraction) || fraction < 0 || fraction > 1 {
		return nil, nil, fmt.Errorf("train: train fraction %v is not in [0, 1]", fraction)
	}
	if fraction == 0 || fraction == 1 {
		return examples, nil, nil
	}
	strata := map[string][]*Example{}
	rank := map[*Example]string{}
	var source []string
	for _, ex := range examples {
		if ex.Split != SplitTrain {
			continue
		}
		key := ex.Task + "\x00" + ex.Lang + "\x00" + ex.DomainID
		strata[key] = append(strata[key], ex)
		rank[ex] = subsampleRank(seed, ex.ID)
		source = append(source, ex.ID)
	}
	if len(source) == 0 {
		return nil, nil, errors.New("train: a train fraction was given but the dataset has no train example")
	}
	keep := map[*Example]bool{}
	for _, s := range strata {
		slices.SortFunc(s, func(a, b *Example) int {
			if c := strings.Compare(rank[a], rank[b]); c != 0 {
				return c
			}
			return strings.Compare(a.ID, b.ID)
		})
		// The epsilon keeps a product such as 0.3 * 10 from rounding up
		// past the intended count.
		n := int(math.Ceil(fraction*float64(len(s)) - 1e-9))
		for _, ex := range s[:max(1, min(len(s), n))] {
			keep[ex] = true
		}
	}
	sub := &Subsample{Rule: SubsampleRule, Fraction: fraction, Seed: seed, Strata: len(strata),
		SourceExamples: len(source), SourceIDsSHA256: idDigest(source),
		KeptByTask: map[string]int{}, KeptByLang: map[string]int{}, KeptByDomain: map[string]int{}}
	out := make([]*Example, 0, len(examples)-len(source)+len(keep))
	var kept []string
	for _, ex := range examples {
		if ex.Split == SplitTrain {
			if !keep[ex] {
				continue
			}
			kept = append(kept, ex.ID)
			sub.KeptByTask[ex.Task]++
			sub.KeptByLang[ex.Lang]++
			sub.KeptByDomain[ex.DomainID]++
		}
		out = append(out, ex)
	}
	sub.KeptExamples, sub.KeptIDsSHA256 = len(kept), idDigest(kept)
	return out, sub, nil
}

// Prepared is a dataset rendered and laid out for one instance.
type Prepared struct {
	Dataset  *Dataset
	Instance string
	Tasks    map[string]*TaskSpec
	Items    []*Item
	// Duplicates counts, per evaluated split, the items whose rendered
	// (task, state, question) also occurs in train.
	Duplicates map[string]int
	// LockedTestEvaluated is true when locked_test was unsealed and has
	// items; Sealed when it was not unsealed (the manifest then records only
	// its size and files).
	LockedTestEvaluated bool
	Sealed              bool
	// SequenceDigest is sha256 over every item's id, token ids and markers:
	// the feature cache key's view of the rendering.
	SequenceDigest string
	// TrainSubsample is set when only a subsample of train was kept
	// (PrepareOptions.TrainFraction).
	TrainSubsample *Subsample
	// EvalLangs are the languages the evaluated splits were restricted to
	// (PrepareOptions.EvalLangs), nil when every language was kept.
	EvalLangs []string
	// Shortlist is the catalog shortlist the items were laid out with
	// (nil: the whole catalog).
	Shortlist *ShortlistSpec
}

// Prepare checks the split invariants, defines the tasks from the examples
// that are not sealed, renders every visible example with
// hive.laya-prompt/v1 and lays it out with laya.BuildSequence. It fails on
// a split-invariant violation or a rendered train duplicate unless the
// options allow it, and on locked_test unsealed without an access log.
func Prepare(ds *Dataset, t *Trunk, opts PrepareOptions) (*Prepared, error) {
	if ds.Instance == "" {
		return nil, errors.New("train: a head is trained for one instance; load the dataset with Q, R or A")
	}
	if opts.UnsealLockedTest && opts.AccessLog == "" {
		return nil, errors.New("train: unsealing locked_test needs an access log")
	}
	if n := len(ds.SplitViolations); n > 0 && !opts.AllowSplitLeakage {
		shown := ds.SplitViolations[:min(n, 10)]
		return nil, fmt.Errorf("train: the dataset breaks %d split invariant(s) of %s: %s", n, DatasetContract, strings.Join(shown, "; "))
	}
	var visible []*Example
	for i := range ds.Examples {
		ex := &ds.Examples[i]
		if ex.Split == SplitLockedTest && !opts.UnsealLockedTest {
			continue
		}
		visible = append(visible, ex)
	}
	specs, err := BuildTaskSpecs(visible, opts.Instructions)
	if err != nil {
		return nil, err
	}
	visible, sub, err := subsampleTrain(visible, opts.TrainFraction, opts.SubsampleSeed)
	if err != nil {
		return nil, err
	}
	p := &Prepared{Dataset: ds, Instance: ds.Instance, Tasks: specs, Sealed: !opts.UnsealLockedTest, TrainSubsample: sub}
	if len(opts.EvalLangs) > 0 {
		p.EvalLangs = slices.Clone(opts.EvalLangs)
		slices.Sort(p.EvalLangs)
		kept := visible[:0:0]
		for _, ex := range visible {
			if ex.Split == SplitTrain || slices.Contains(p.EvalLangs, primaryLanguage(ex.Lang)) {
				kept = append(kept, ex)
			}
		}
		visible = kept
	}
	if opts.Shortlist.K > 0 {
		if ds.Instance != InstanceQ {
			return nil, fmt.Errorf("train: a catalog shortlist applies to instance Q, not %s", ds.Instance)
		}
		sl := opts.Shortlist
		sl.Method = ShortlistMethod
		p.Shortlist = &sl
	}
	h := sha256.New()
	locked := 0
	for _, ex := range visible {
		it, err := layOut(ex, specs[ex.Task], t, p.Shortlist)
		if err != nil {
			return nil, err
		}
		p.Items = append(p.Items, it)
		if it.Role == SplitLockedTest {
			locked++
		}
		ids, _ := json.Marshal(it.IDs)
		mk, _ := json.Marshal(it.Markers)
		h.Write([]byte(ex.ID + "\x00"))
		h.Write(ids)
		h.Write(mk)
		h.Write([]byte{0})
	}
	p.SequenceDigest = hex.EncodeToString(h.Sum(nil))
	p.Duplicates = duplicateReport(p.Items)
	if len(p.Duplicates) > 0 && !opts.AllowDuplicates {
		return nil, fmt.Errorf("train: rendered train examples reappear in evaluated splits: %v", p.Duplicates)
	}
	if locked > 0 {
		if err := logLockedAccess(opts, ds, locked); err != nil {
			return nil, err
		}
		p.LockedTestEvaluated = true
	}
	return p, nil
}

func layOut(ex *Example, spec *TaskSpec, t *Trunk, sl *ShortlistSpec) (*Item, error) {
	source := ex
	if sl != nil && sl.K > 0 {
		short, res := CatalogShortlist(ex, sl.K)
		if !res.GoldKept {
			short.MissingGold = ex.GoldKey()
		}
		ex = short
	}
	state, err := laya.RenderPromptState(ex.Instance, ex.Input)
	if err != nil {
		return nil, fmt.Errorf("%s:%d: %w", ex.File, ex.Line, err)
	}
	q := spec.Question(ex)
	ids, markers, err := laya.BuildSequence(t.Tokenizer, t.Special, state, q, t.Spec.MaxLen, t.Spec.HeadMaxLen, nil, false)
	if err != nil {
		return nil, fmt.Errorf("%s:%d: %w", ex.File, ex.Line, err)
	}
	return &Item{Ex: ex, Source: source, Role: ex.Role(), State: state, Question: q, IDs: ids, Markers: markers,
		Scorable: len(markers) == ex.NumOptions()}, nil
}

// duplicateReport counts evaluated items whose rendered (task, state,
// question) also occurs in train; the question's id is not part of what the
// head reads.
func duplicateReport(items []*Item) map[string]int {
	digest := func(it *Item) string {
		q := it.Question
		q.ID = ""
		raw, _ := json.Marshal([]any{it.Ex.Task, it.State, q})
		sum := sha256.Sum256(raw)
		return string(sum[:])
	}
	train := map[string]bool{}
	for _, it := range items {
		if it.Role == SplitTrain {
			train[digest(it)] = true
		}
	}
	out := map[string]int{}
	for _, it := range items {
		if it.Role != SplitTrain && train[digest(it)] {
			out[it.Role]++
		}
	}
	return out
}

// logLockedAccess appends one unsealing record to the access log.
func logLockedAccess(opts PrepareOptions, ds *Dataset, n int) error {
	rec, err := json.Marshal(map[string]any{
		"ts": time.Now().UTC().Format(time.RFC3339Nano), "event": "locked_test_evaluated", "dataset_digest": ds.Digest(),
		"examples": n, "note": opts.AccessNote, "trainer": TrainerVersion,
	})
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(opts.AccessLog), 0o750); err != nil {
		return err
	}
	f, err := os.OpenFile(opts.AccessLog, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600) //nolint:gosec // G304: access log chosen by the operator
	if err != nil {
		return err
	}
	if _, err := f.Write(append(rec, '\n')); err != nil {
		_ = f.Close()
		return err
	}
	return f.Close()
}

// ItemsByRole groups items by role.
func (p *Prepared) ItemsByRole() map[string][]*Item {
	out := map[string][]*Item{}
	for _, it := range p.Items {
		out[it.Role] = append(out[it.Role], it)
	}
	return out
}
