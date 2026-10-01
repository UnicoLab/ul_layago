package train

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/UnicoLab/ul_layago/laya"
)

// TaskType is a dataset task type.
type TaskType string

// Dataset task types.
const (
	TaskChoice TaskType = "choice"
	TaskScore  TaskType = "score"
	TaskYesNo  TaskType = "yes_no"
)

// QType is the Laya question type a task type is asked as.
func (t TaskType) QType() laya.QType {
	switch t {
	case TaskScore:
		return laya.Score
	case TaskYesNo:
		return laya.Noul
	default:
		return laya.Choice
	}
}

// Option is one option as the contract defines it: a choice option (key and
// optional text), a score level (key "0", "1", ... and its text) or a yes_no
// criterion (key "0" for false, "1" for true, and its text).
type Option struct {
	Key  string `json:"key"`
	Text string `json:"text"`
}

// Example is one validated hive.laya-dataset/v1 line.
type Example struct {
	ID       string
	Split    string
	Lang     string
	Instance string
	Task     string
	Type     TaskType
	// Input is the decoded input object (numbers as json.Number), rendered
	// into the state by laya.RenderPromptState.
	Input map[string]any
	// Options are empty for a yes_no example that uses Laya's default
	// criteria.
	Options []Option
	// Label is the option index (choice), the level index (score) or 0/1
	// for false/true (yes_no), fixed from the structured spec. It is -1
	// only on an example a catalog shortlist laid out without its gold
	// option (MissingGold then holds the gold key): such an example is
	// never trained on and is always wrong.
	Label int
	// MissingGold is the gold class key when Label is -1.
	MissingGold string
	// LabelDist is the optional soft label over the options (train only):
	// the loss trains on it, every metric scores Label.
	LabelDist        []float64
	LabelSource      string
	DomainID         string
	StoryID          string
	GeneratorVersion string
	Seed             json.Number
	// Provenance holds the hive.llm-interaction/v1 ids of the calls that
	// wrote the text.
	Provenance []string
	// Slices are the optional meta.slices evaluation slice names (for
	// example unseen_template), reported and never trained on.
	Slices []string
	File   string
	Line   int
}

// NumOptions is the number of markers the example is scored over.
func (e *Example) NumOptions() int {
	if e.Type == TaskYesNo {
		return 2
	}
	return len(e.Options)
}

// ClassKeys are the class keys metrics count: option keys (choice), level
// indices (score) or false/true (yes_no), as prompt.py option_keys.
func (e *Example) ClassKeys() []string {
	switch e.Type {
	case TaskChoice:
		out := make([]string, len(e.Options))
		for i, o := range e.Options {
			out[i] = o.Key
		}
		return out
	case TaskScore:
		out := make([]string, len(e.Options))
		for i := range out {
			out[i] = strconv.Itoa(i)
		}
		return out
	default:
		return []string{"false", "true"}
	}
}

// Target is the training target over the options: LabelDist when given,
// else one-hot on Label (all zero when the gold is not among the options).
func (e *Example) Target() []float64 {
	t := make([]float64, e.NumOptions())
	if e.LabelDist != nil {
		copy(t, e.LabelDist)
		return t
	}
	if e.Label >= 0 {
		t[e.Label] = 1
	}
	return t
}

// HasGold reports whether the gold option is among the example's options.
func (e *Example) HasGold() bool { return e.Label >= 0 }

// GoldKey is the gold class key, whether or not it is among the options.
func (e *Example) GoldKey() string {
	if e.Label < 0 {
		return e.MissingGold
	}
	return e.ClassKeys()[e.Label]
}

// Role is the example's evaluation split: dev_cal or dev_conf for dev (by
// story, DevHalf), else its split.
func (e *Example) Role() string {
	if e.Split == SplitDev {
		return DevHalf(e.StoryID)
	}
	return e.Split
}

// DevHalf divides dev by story, as dataset.py dev_half does: the first byte
// of sha256("laya-dev-half\x00" + story_id), even for dev_cal and odd for
// dev_conf. One story never feeds both halves, so the thresholds certified
// on dev_conf are not certified on paraphrases or translations of what the
// temperatures and the act head were fitted on.
func DevHalf(storyID string) string {
	h := sha256.Sum256([]byte("laya-dev-half\x00" + storyID))
	if h[0]&1 == 0 {
		return RoleDevCal
	}
	return RoleDevConf
}

// DatasetFile is one file of a dataset, as the manifest records it.
type DatasetFile struct {
	Name     string `json:"name"`
	SHA256   string `json:"sha256"`
	Examples int    `json:"examples"`
}

// Dataset is a loaded, validated hive.laya-dataset/v1 dataset.
type Dataset struct {
	Files []DatasetFile
	// Instance is the instance kept ("" keeps every instance).
	Instance string
	// Examples are the examples of Instance, in file then line order.
	Examples []Example
	// Total counts every valid example of every instance.
	Total int
	// SplitViolations lists the split invariants the data breaks, checked
	// over every example of every file whatever instance was kept.
	SplitViolations []string
}

// Digest is the training data digest: sha256 over the contract name and,
// for each file in (name, sha256) order, a NUL, the name, a NUL and its
// sha256 (dataset.py Dataset.digest, routermodel.DatasetDigest).
func (d *Dataset) Digest() string { return DataDigest(d.Files) }

// DataDigest computes the training data digest of files.
func DataDigest(files []DatasetFile) string {
	sorted := slices.Clone(files)
	slices.SortFunc(sorted, func(a, b DatasetFile) int {
		if c := strings.Compare(a.Name, b.Name); c != 0 {
			return c
		}
		return strings.Compare(a.SHA256, b.SHA256)
	})
	h := sha256.New()
	h.Write([]byte(DatasetContract))
	for _, f := range sorted {
		h.Write([]byte("\x00" + f.Name + "\x00" + f.SHA256))
	}
	return hex.EncodeToString(h.Sum(nil))
}

// DataFile is one dataset file's name and bytes.
type DataFile struct {
	Name string
	Data []byte
}

// LoadDataset reads, validates and digests dataset files (JSONL). Every
// example of every file is validated, whatever instance is kept; ids must
// be unique across files. instance "" keeps every instance.
func LoadDataset(paths []string, instance string) (*Dataset, error) {
	sorted := slices.Clone(paths)
	slices.Sort(sorted)
	files := make([]DataFile, 0, len(sorted))
	for _, p := range sorted {
		raw, err := os.ReadFile(p) //nolint:gosec // G304: dataset paths chosen by the operator
		if err != nil {
			return nil, err
		}
		files = append(files, DataFile{Name: p, Data: raw})
	}
	return ParseDataset(files, instance)
}

// ParseDataset validates dataset files given in memory. File names are
// reported in errors; the digest uses their base names.
func ParseDataset(files []DataFile, instance string) (*Dataset, error) {
	if instance != "" && !isOneOf(instance, Instances) {
		return nil, fmt.Errorf("train: instance %q is not one of %v", instance, Instances)
	}
	if len(files) == 0 {
		return nil, errors.New("train: no dataset file")
	}
	ds := &Dataset{Instance: instance}
	seen := map[string]string{}
	var everything []*Example
	for _, f := range files {
		count := 0
		for line, text := range jsonlLines(f.Data) {
			if strings.TrimFunc(string(text), isPySpace) == "" {
				continue
			}
			ex, err := parseExample(text, f.Name, line)
			if err != nil {
				return nil, err
			}
			if first, dup := seen[ex.ID]; dup {
				return nil, fmt.Errorf("%s:%d: duplicate id %q (first at %s)", f.Name, line, ex.ID, first)
			}
			seen[ex.ID] = fmt.Sprintf("%s:%d", f.Name, line)
			count++
			everything = append(everything, ex)
			if instance == "" || ex.Instance == instance {
				ds.Examples = append(ds.Examples, *ex)
			}
		}
		sum := sha256.Sum256(f.Data)
		ds.Files = append(ds.Files, DatasetFile{Name: filepath.Base(f.Name), SHA256: hex.EncodeToString(sum[:]), Examples: count})
	}
	ds.Total = len(everything)
	if len(ds.Examples) == 0 {
		if instance != "" {
			return nil, fmt.Errorf("train: the dataset has no examples for instance %s", instance)
		}
		return nil, errors.New("train: the dataset has no examples")
	}
	ds.SplitViolations = splitViolations(everything)
	return ds, nil
}

// jsonlLines yields the 1-based lines of data without their line endings.
func jsonlLines(data []byte) func(func(int, []byte) bool) {
	return func(yield func(int, []byte) bool) {
		for i := 1; len(data) > 0; i++ {
			line, rest, _ := bytes.Cut(data, []byte{'\n'})
			data = rest
			if !yield(i, bytes.TrimSuffix(line, []byte{'\r'})) {
				return
			}
		}
	}
}

var (
	langRE   = regexp.MustCompile(`^[a-z]{2,3}(-[A-Za-z0-9]{1,8})*$`)
	taskRE   = regexp.MustCompile(`^[a-z][a-z0-9_]{0,63}$`)
	sha256RE = regexp.MustCompile(`^[0-9a-f]{64}$`)
)

type lineError struct {
	file string
	line int
}

func (l lineError) errorf(format string, args ...any) error {
	return fmt.Errorf("%s:%d: %s", l.file, l.line, fmt.Sprintf(format, args...))
}

// parseExample validates one line exactly as dataset.py parse_example does,
// in the same order, so both loaders refuse the same lines for the same
// reason.
func parseExample(text []byte, file string, line int) (*Example, error) {
	at := lineError{file, line}
	if !utf8.Valid(text) {
		return nil, at.errorf("invalid UTF-8")
	}
	dec := json.NewDecoder(bytes.NewReader(text))
	dec.UseNumber()
	var raw any
	if err := dec.Decode(&raw); err != nil {
		return nil, at.errorf("invalid JSON: %v", err)
	}
	if _, err := dec.Token(); !errors.Is(err, io.EOF) {
		return nil, at.errorf("invalid JSON: trailing data")
	}
	obj, ok := raw.(map[string]any)
	if !ok {
		return nil, at.errorf("an example must be a JSON object")
	}
	str := func(key string) (string, error) {
		v, ok := obj[key].(string)
		if !ok || strings.TrimFunc(v, isPySpace) == "" {
			return "", at.errorf("field %q must be a non-empty string", key)
		}
		return v, nil
	}
	ex := &Example{File: file, Line: line}
	var err error
	if ex.ID, err = str("id"); err != nil {
		return nil, err
	}
	if ex.Split, err = str("split"); err != nil {
		return nil, err
	}
	if !isOneOf(ex.Split, Splits) {
		return nil, at.errorf("split %q is not one of %v", ex.Split, Splits)
	}
	if ex.Lang, err = str("lang"); err != nil {
		return nil, err
	}
	if !langRE.MatchString(ex.Lang) {
		return nil, at.errorf("lang %q is not a BCP-47 tag", ex.Lang)
	}
	if ex.Instance, err = str("instance"); err != nil {
		return nil, err
	}
	if !isOneOf(ex.Instance, Instances) {
		return nil, at.errorf("instance %q is not one of %v", ex.Instance, Instances)
	}
	if ex.Task, err = str("task"); err != nil {
		return nil, err
	}
	if !taskRE.MatchString(ex.Task) {
		return nil, at.errorf("task %q must be lower snake case", ex.Task)
	}
	if !strings.HasPrefix(ex.Task, strings.ToLower(ex.Instance)+"_") {
		return nil, at.errorf("task %q does not belong to instance %s", ex.Task, ex.Instance)
	}
	tt, err := str("task_type")
	if err != nil {
		return nil, err
	}
	ex.Type = TaskType(tt)
	if ex.Type != TaskChoice && ex.Type != TaskScore && ex.Type != TaskYesNo {
		return nil, at.errorf("task_type %q is not one of [choice score yes_no]", tt)
	}
	if ex.Options, err = parseOptions(obj["options"], ex.Type, at); err != nil {
		return nil, err
	}
	n := ex.NumOptions()
	if ex.Label, err = parseLabel(obj["label"], ex.Type, n, at); err != nil {
		return nil, err
	}
	if ex.LabelDist, err = parseLabelDist(obj, n, ex.Split, ex.Label, at); err != nil {
		return nil, err
	}
	if ex.LabelSource, err = str("label_source"); err != nil {
		return nil, err
	}
	if !isOneOf(ex.LabelSource, LabelSources) {
		return nil, at.errorf("label_source %q is not one of %v", ex.LabelSource, LabelSources)
	}
	if ex.Provenance, err = parseProvenance(obj["provenance"], at); err != nil {
		return nil, err
	}
	if isOneOf(ex.LabelSource, llmLabelSources) && len(ex.Provenance) == 0 {
		return nil, at.errorf("label_source %q needs the %s id of the call that wrote the text", ex.LabelSource, LLMInteractionContract)
	}
	seed, ok := obj["seed"].(json.Number)
	if !ok || !isIntegerLiteral(string(seed)) {
		return nil, at.errorf("seed must be an integer")
	}
	ex.Seed = seed
	if ex.Input, err = parseInput(obj["input"], ex.Instance, at); err != nil {
		return nil, err
	}
	if ex.DomainID, err = str("domain_id"); err != nil {
		return nil, err
	}
	if ex.StoryID, err = str("story_id"); err != nil {
		return nil, err
	}
	if ex.GeneratorVersion, err = str("generator_version"); err != nil {
		return nil, err
	}
	ex.Slices = metaSlices(obj["meta"])
	return ex, nil
}

// metaSlices reads the optional meta.slices list of evaluation slice names
// (for example unseen_template). meta is free-form: anything else is
// ignored, and slices are only reported, never trained on.
func metaSlices(meta any) []string {
	m, ok := meta.(map[string]any)
	if !ok {
		return nil
	}
	list, ok := m["slices"].([]any)
	if !ok {
		return nil
	}
	var out []string
	for _, v := range list {
		if s, ok := v.(string); ok && s != "" {
			out = append(out, s)
		}
	}
	return out
}

func isIntegerLiteral(s string) bool {
	return s != "" && !strings.ContainsAny(s, ".eE")
}

func parseOptions(raw any, tt TaskType, at lineError) ([]Option, error) {
	if raw == nil {
		raw = []any{}
	}
	list, ok := raw.([]any)
	if !ok {
		return nil, at.errorf("options must be a list")
	}
	out := make([]Option, 0, len(list))
	for i, o := range list {
		switch v := o.(type) {
		case string:
			if strings.TrimFunc(v, isPySpace) == "" {
				return nil, at.errorf("option %d is empty", i)
			}
			if tt == TaskChoice {
				out = append(out, Option{Key: v})
			} else {
				out = append(out, Option{Key: strconv.Itoa(i), Text: v})
			}
		case map[string]any:
			text := ""
			if t, present := v["text"]; present {
				s, ok := t.(string)
				if !ok {
					return nil, at.errorf("option %d text must be a string", i)
				}
				text = s
			}
			if tt == TaskChoice {
				key, ok := v["key"].(string)
				if !ok || strings.TrimFunc(key, isPySpace) == "" {
					return nil, at.errorf("option %d needs a non-empty key", i)
				}
				out = append(out, Option{Key: key, Text: text})
			} else {
				if strings.TrimFunc(text, isPySpace) == "" {
					return nil, at.errorf("option %d needs a non-empty text", i)
				}
				out = append(out, Option{Key: strconv.Itoa(i), Text: text})
			}
		default:
			return nil, at.errorf("option %d must be a string or an object", i)
		}
	}
	switch tt {
	case TaskChoice:
		if len(out) < 2 {
			return nil, at.errorf("a choice task needs at least 2 options")
		}
		keys := map[string]bool{}
		for _, o := range out {
			if keys[o.Key] {
				return nil, at.errorf("choice option keys must be unique")
			}
			keys[o.Key] = true
		}
	case TaskScore:
		if len(out) < 2 {
			return nil, at.errorf("a score task needs at least 2 levels")
		}
	default:
		if len(out) != 0 && len(out) != 2 {
			return nil, at.errorf("yes_no options, when given, are exactly [false_text, true_text]")
		}
	}
	return out, nil
}

// parseLabel reads the label: a JSON boolean for yes_no (0 and 1 are
// refused, never coerced), else an option index.
func parseLabel(v any, tt TaskType, n int, at lineError) (int, error) {
	if tt == TaskYesNo {
		b, ok := v.(bool)
		if !ok {
			return 0, at.errorf("a yes_no label must be a boolean (true or false), not %s", jsonText(v))
		}
		if b {
			return 1, nil
		}
		return 0, nil
	}
	num, ok := v.(json.Number)
	if !ok || !isIntegerLiteral(string(num)) {
		return 0, at.errorf("a %s label must be an option index", tt)
	}
	i, err := strconv.Atoi(string(num))
	if err != nil || i < 0 || i >= n {
		return 0, at.errorf("label %s is outside the %d options", num, n)
	}
	return i, nil
}

// parseLabelDist reads the optional soft label: train only, n non-negative
// numbers summing to 1, with label among its most probable options (the loss
// trains on label_dist and every metric scores label, so both must name the
// same answer).
func parseLabelDist(obj map[string]any, n int, split string, label int, at lineError) ([]float64, error) {
	v, present := obj["label_dist"]
	if !present || v == nil {
		return nil, nil
	}
	if split != SplitTrain {
		return nil, at.errorf("label_dist (soft labels) is allowed only on the train split")
	}
	list, ok := v.([]any)
	if !ok || len(list) != n {
		return nil, at.errorf("label_dist must be %d non-negative numbers", n)
	}
	out := make([]float64, n)
	var sum, best float64
	for i, x := range list {
		num, ok := x.(json.Number)
		if !ok {
			return nil, at.errorf("label_dist must be %d non-negative numbers", n)
		}
		f, err := num.Float64()
		if err != nil || f < 0 || math.IsInf(f, 0) || math.IsNaN(f) {
			return nil, at.errorf("label_dist must be %d non-negative numbers", n)
		}
		out[i] = f
		sum += f
		best = math.Max(best, f)
	}
	if math.Abs(sum-1) > 1e-6 {
		return nil, at.errorf("label_dist must sum to 1 (got %v)", sum)
	}
	if out[label] < best {
		return nil, at.errorf("label %d is not a most probable option of label_dist %v", label, out)
	}
	return out, nil
}

// parseProvenance accepts an id, a list of ids, or {"llm_interaction_ids":
// [...]} / {"llm_interaction_id": id}; every id is a sha256.
func parseProvenance(raw any, at lineError) ([]string, error) {
	if m, ok := raw.(map[string]any); ok {
		if v, ok := m["llm_interaction_ids"]; ok {
			raw = v
		} else {
			raw = m["llm_interaction_id"]
		}
	}
	var ids []any
	switch v := raw.(type) {
	case nil:
		return nil, nil
	case string:
		ids = []any{v}
	case []any:
		ids = v
	default:
		return nil, at.errorf("provenance must be an id, a list of ids or an object")
	}
	out := make([]string, 0, len(ids))
	for _, x := range ids {
		s, ok := x.(string)
		if !ok || !sha256RE.MatchString(s) {
			return nil, at.errorf("provenance id %s is not a %s sha256", jsonText(x), LLMInteractionContract)
		}
		out = append(out, s)
	}
	return out, nil
}

func parseInput(raw any, instance string, at lineError) (map[string]any, error) {
	in, ok := raw.(map[string]any)
	if !ok {
		return nil, at.errorf("input must be an object")
	}
	if q, ok := in["question"].(string); !ok || strings.TrimFunc(q, isPySpace) == "" {
		return nil, at.errorf("field %q must be a non-empty string", "question")
	}
	list := func(key string) bool {
		v, present := in[key]
		if !present {
			return true
		}
		_, ok := v.([]any)
		return ok
	}
	switch instance {
	case InstanceQ:
		if !list("catalog") {
			return nil, at.errorf("input.catalog must be a list of metrics")
		}
	case InstanceR:
		if !list("evidence") {
			return nil, at.errorf("input.evidence must be a list of evidence items")
		}
	case InstanceA:
		for _, key := range []string{"draft", "sentence"} {
			if v, present := in[key]; present {
				if _, ok := v.(string); !ok {
					return nil, at.errorf("input.%s must be a string", key)
				}
			}
		}
	}
	return in, nil
}

func jsonText(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprint(v)
	}
	return string(b)
}

// primaryLanguage is a tag's primary subtag, lower-cased.
func primaryLanguage(tag string) string {
	p, _, _ := strings.Cut(tag, "-")
	return strings.ToLower(p)
}

// splitViolations checks the split invariants of hive.laya-dataset/v1, as
// dataset.py split_violations does:
//
//   - a story belongs to exactly one split (so locked_test stories appear
//     nowhere else, and no dev story is a train story);
//   - no heldout_domain domain_id appears in train or dev;
//   - no heldout_language language appears in train or dev, compared by
//     primary subtag.
//
// Each violation names the file (by name only: the list is copied into the
// manifest) and line of the first example on each side, in the Python
// trainer's wording. The dev_conf certificate assumes dev_conf is
// independent of everything fitted, and the held-out splits are reported as
// generalisation to unseen domains and languages; a violation breaks both.
func splitViolations(examples []*Example) []string {
	storyFirst := map[string]map[string]*Example{}
	fitDomainFirst := map[string]*Example{}
	fitLangFirst := map[string]map[string]*Example{}
	heldDomainFirst, heldLangFirst := map[string]*Example{}, map[string]*Example{}
	first := func(m map[string]*Example, key string, ex *Example) {
		if _, ok := m[key]; !ok {
			m[key] = ex
		}
	}
	for _, ex := range examples {
		if storyFirst[ex.StoryID] == nil {
			storyFirst[ex.StoryID] = map[string]*Example{}
		}
		first(storyFirst[ex.StoryID], ex.Split, ex)
		switch {
		case isOneOf(ex.Split, fitSplits):
			first(fitDomainFirst, ex.DomainID, ex)
			p := primaryLanguage(ex.Lang)
			if fitLangFirst[p] == nil {
				fitLangFirst[p] = map[string]*Example{}
			}
			first(fitLangFirst[p], ex.Lang, ex)
		case ex.Split == SplitHeldoutDomain:
			first(heldDomainFirst, ex.DomainID, ex)
		case ex.Split == SplitHeldoutLanguage:
			first(heldLangFirst, ex.Lang, ex)
		}
	}
	where := func(ex *Example) string { return fmt.Sprintf("%s:%d", filepath.Base(ex.File), ex.Line) }
	var out []string
	for _, story := range sortedKeys(storyFirst) {
		firsts := storyFirst[story]
		if len(firsts) < 2 {
			continue
		}
		splits := sortedKeys(firsts)
		at := make([]string, len(splits))
		for i, s := range splits {
			at[i] = s + " at " + where(firsts[s])
		}
		out = append(out, fmt.Sprintf("story %s appears in more than one split: %s (%s)", pyRepr(story), pyList(splits), strings.Join(at, ", ")))
	}
	for _, d := range sortedKeys(heldDomainFirst) {
		if fit, ok := fitDomainFirst[d]; ok {
			out = append(out, fmt.Sprintf("heldout_domain domain %s also appears in train or dev (heldout_domain at %s, %s at %s)",
				pyRepr(d), where(heldDomainFirst[d]), fit.Split, where(fit)))
		}
	}
	for _, l := range sortedKeys(heldLangFirst) {
		seen := fitLangFirst[primaryLanguage(l)]
		if len(seen) == 0 {
			continue
		}
		langs := sortedKeys(seen)
		fit := seen[langs[0]]
		out = append(out, fmt.Sprintf("heldout_language %s shares its language with train or dev examples %s (heldout_language at %s, %s %s at %s)",
			pyRepr(l), pyList(langs), where(heldLangFirst[l]), fit.Split, pyRepr(fit.Lang), where(fit)))
	}
	return out
}

// pyRepr quotes a plain string as Python's repr does.
func pyRepr(s string) string {
	if strings.Contains(s, "'") && !strings.Contains(s, `"`) {
		return `"` + s + `"`
	}
	return "'" + strings.ReplaceAll(strings.ReplaceAll(s, `\`, `\\`), "'", `\'`) + "'"
}

// pyList formats strings as Python prints a list of them.
func pyList(items []string) string {
	q := make([]string, len(items))
	for i, s := range items {
		q[i] = pyRepr(s)
	}
	return "[" + strings.Join(q, ", ") + "]"
}
