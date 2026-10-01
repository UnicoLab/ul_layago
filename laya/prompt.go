package laya

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/big"
	"sort"
	"strconv"
	"strings"
)

// ErrPromptInput means an input cannot be rendered as the trainer would: a
// required field is missing, or a value is a JSON array or object where the
// trainer would print Python's repr of it (not ported).
var ErrPromptInput = errors.New("laya: input cannot be rendered with " + ArtifactPromptFormat)

// RenderPromptState renders the hive.laya-prompt/v1 state of one
// hive.laya-dataset/v1 input, as tools/laya-train/laya_train/prompt.py
// render_state does, byte for byte (TestPromptMatchesTrainer):
//
//	question: <question>
//	catalog:                                   (Q, when input.catalog is non-empty)
//	- <key>: <label> | <label>; unit: <unit>; dimensions: <d>, <d>; scope: <dim>=<v>/<v>, <dim>=<v>
//	evidence:                                  (R, when input.evidence is non-empty)
//	- <id>: <text>
//	draft: <draft>                             (A, when present)
//	sentence: <sentence>                       (A, when present)
//
// input is the decoded JSON object. Numbers must be json.Number (decode with
// UseNumber) or Go integers, so an integer and a float render as Python's
// str() renders them ("2024", "2024.0"); a float64 is taken to be a Python
// float. Strings are trimmed with Python's str.strip() whitespace; interior
// newlines and tabs are kept, as the trainer keeps them.
func RenderPromptState(instance string, input map[string]any) (string, error) {
	question, ok := input["question"].(string)
	if !ok {
		return "", fmt.Errorf("%w: question must be a string", ErrPromptInput)
	}
	lines := []string{"question: " + pyStrip(question)}
	switch instance {
	case "Q":
		catalog, err := promptList(input["catalog"], "catalog")
		if err != nil {
			return "", err
		}
		if len(catalog) > 0 {
			lines = append(lines, "catalog:")
			for _, m := range catalog {
				line, err := promptMetricLine(m)
				if err != nil {
					return "", err
				}
				lines = append(lines, line)
			}
		}
	case "R":
		evidence, err := promptList(input["evidence"], "evidence")
		if err != nil {
			return "", err
		}
		if len(evidence) > 0 {
			lines = append(lines, "evidence:")
		}
		for i, item := range evidence {
			if obj, ok := item.(map[string]any); ok {
				id, err := pyStr(getOr(obj, "id", i))
				if err != nil {
					return "", err
				}
				text, err := pyStr(getOr(obj, "text", ""))
				if err != nil {
					return "", err
				}
				lines = append(lines, "- "+pyStrip(id)+": "+pyStrip(text))
				continue
			}
			text, err := pyStr(item)
			if err != nil {
				return "", err
			}
			lines = append(lines, "- "+strconv.Itoa(i)+": "+pyStrip(text))
		}
	case "A":
		for _, key := range []string{"draft", "sentence"} {
			if v, ok := input[key].(string); ok && pyStrip(v) != "" {
				lines = append(lines, key+": "+pyStrip(v))
			}
		}
	default:
		return "", fmt.Errorf("%w: instance %q is not Q, R or A", ErrPromptInput, instance)
	}
	return strings.Join(lines, "\n"), nil
}

// promptList is Python's `input.get(name) or []` for a list field.
func promptList(v any, name string) ([]any, error) {
	switch l := v.(type) {
	case nil:
		return nil, nil
	case []any:
		return l, nil
	case []map[string]any:
		out := make([]any, len(l))
		for i := range l {
			out[i] = l[i]
		}
		return out, nil
	default:
		return nil, fmt.Errorf("%w: %s must be a list", ErrPromptInput, name)
	}
}

// getOr is Python's dict.get(key, def): a present key returns its value,
// null included.
func getOr(obj map[string]any, key string, def any) any {
	if v, ok := obj[key]; ok {
		return v
	}
	return def
}

// promptMetricLine is prompt.py _metric_line.
func promptMetricLine(m any) (string, error) {
	if s, ok := m.(string); ok {
		return "- " + s, nil
	}
	obj, ok := m.(map[string]any)
	if !ok {
		s, err := pyStr(m)
		return "- " + s, err
	}
	key, err := pyStr(getOr(obj, "key", ""))
	if err != nil {
		return "", err
	}
	key = pyStrip(key)
	var rawLabels any
	if v, ok := obj["labels"]; ok {
		rawLabels = v
	} else {
		rawLabels = obj["label"]
	}
	labels := promptLabels(rawLabels)
	var parts []string
	head := key
	if len(labels) > 0 {
		head = strings.Join(labels, " | ")
		if key != "" {
			head = key + ": " + head
		}
	}
	if head != "" {
		parts = append(parts, head)
	}
	if unit, ok := obj["unit"].(string); ok && pyStrip(unit) != "" {
		parts = append(parts, "unit: "+pyStrip(unit))
	}
	if dims, ok := asList(obj["dimensions"]); ok {
		var names []string
		for _, d := range dims {
			s, err := pyStr(d)
			if err != nil {
				return "", err
			}
			if s = pyStrip(s); s != "" {
				names = append(names, s)
			}
		}
		if len(names) > 0 {
			parts = append(parts, "dimensions: "+strings.Join(names, ", "))
		}
	}
	if scope, ok := asMap(obj["scope_values"]); ok {
		dims := make([]string, 0, len(scope))
		for dim := range scope {
			dims = append(dims, dim)
		}
		sort.Strings(dims) // byte order of UTF-8 is code point order, as Python sorts
		var items []string
		for _, dim := range dims {
			vals, ok := asList(scope[dim])
			if !ok {
				continue
			}
			var vs []string
			for _, v := range vals {
				s, err := pyStr(v)
				if err != nil {
					return "", err
				}
				if s = pyStrip(s); s != "" {
					vs = append(vs, s)
				}
			}
			if len(vs) > 0 {
				items = append(items, dim+"="+strings.Join(vs, "/"))
			}
		}
		if len(items) > 0 {
			parts = append(parts, "scope: "+strings.Join(items, ", "))
		}
	}
	return "- " + strings.Join(parts, "; "), nil
}

// promptLabels is prompt.py _labels: a string, the string values of an
// object in sorted key order, or the strings of a list; trimmed, empty ones
// dropped, first occurrence kept.
func promptLabels(raw any) []string {
	var vals []string
	switch v := raw.(type) {
	case string:
		vals = []string{v}
	case []string:
		vals = v
	case []any:
		for _, x := range v {
			if s, ok := x.(string); ok {
				vals = append(vals, s)
			}
		}
	case map[string]string:
		keys := make([]string, 0, len(v))
		for k := range v {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			vals = append(vals, v[k])
		}
	case map[string]any:
		keys := make([]string, 0, len(v))
		for k := range v {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			if s, ok := v[k].(string); ok {
				vals = append(vals, s)
			}
		}
	}
	var out []string
	for _, s := range vals {
		if s = pyStrip(s); s != "" && !containsString(out, s) {
			out = append(out, s)
		}
	}
	return out
}

func containsString(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

func asList(v any) ([]any, bool) {
	switch l := v.(type) {
	case []any:
		return l, true
	case []string:
		out := make([]any, len(l))
		for i, s := range l {
			out[i] = s
		}
		return out, true
	}
	return nil, false
}

func asMap(v any) (map[string]any, bool) {
	switch m := v.(type) {
	case map[string]any:
		return m, true
	case map[string][]string:
		out := make(map[string]any, len(m))
		for k, l := range m {
			out[k] = l
		}
		return out, true
	}
	return nil, false
}

// pyStrip is Python's str.strip(): it trims the characters str.isspace()
// accepts, which are Go's unicode.IsSpace plus U+001C..U+001F.
func pyStrip(s string) string {
	return strings.TrimFunc(s, isPySpace)
}

func isPySpace(r rune) bool {
	switch r {
	case '\t', '\n', '\v', '\f', '\r', 0x1c, 0x1d, 0x1e, 0x1f, ' ', 0x85, 0xa0, 0x1680,
		0x2028, 0x2029, 0x202f, 0x205f, 0x3000:
		return true
	}
	return r >= 0x2000 && r <= 0x200a
}

// pyStr is Python's str() of a JSON scalar: strings as they are, integers in
// decimal, floats as repr, True/False and None. Arrays and objects are
// refused (Python would print their repr).
func pyStr(v any) (string, error) {
	switch x := v.(type) {
	case string:
		return x, nil
	case nil:
		return "None", nil
	case bool:
		if x {
			return "True", nil
		}
		return "False", nil
	case json.Number:
		lit := string(x)
		if !strings.ContainsAny(lit, ".eE") {
			n, ok := new(big.Int).SetString(lit, 10)
			if !ok {
				return "", fmt.Errorf("%w: number %q", ErrPromptInput, lit)
			}
			return n.String(), nil
		}
		f, err := strconv.ParseFloat(lit, 64)
		if err != nil {
			return "", fmt.Errorf("%w: number %q", ErrPromptInput, lit)
		}
		return pyFloatRepr(f), nil
	case float64:
		return pyFloatRepr(x), nil
	case float32:
		return pyFloatRepr(float64(x)), nil
	case int:
		return strconv.Itoa(x), nil
	case int64:
		return strconv.FormatInt(x, 10), nil
	case int32:
		return strconv.FormatInt(int64(x), 10), nil
	}
	return "", fmt.Errorf("%w: a %T value has no ported Python str()", ErrPromptInput, v)
}

// pyFloatRepr is Python's repr(float): the shortest round-tripping digits,
// in fixed notation with at least one fractional digit when the decimal
// exponent is in [-4, 16), else d.ddde±XX.
func pyFloatRepr(f float64) string {
	switch {
	case math.IsNaN(f):
		return "nan"
	case math.IsInf(f, 1):
		return "inf"
	case math.IsInf(f, -1):
		return "-inf"
	}
	sci := strconv.FormatFloat(f, 'e', -1, 64) // e.g. -1.2345e+06
	sign := ""
	if strings.HasPrefix(sci, "-") {
		sign, sci = "-", sci[1:]
	}
	mant, expText, _ := strings.Cut(sci, "e")
	exp, _ := strconv.Atoi(expText)
	digits := strings.Replace(mant, ".", "", 1)
	if exp < -4 || exp >= 16 {
		out := digits[:1]
		if len(digits) > 1 {
			out += "." + digits[1:]
		}
		expSign := "+"
		if exp < 0 {
			expSign, exp = "-", -exp
		}
		return fmt.Sprintf("%s%se%s%02d", sign, out, expSign, exp)
	}
	if exp < 0 {
		return sign + "0." + strings.Repeat("0", -exp-1) + digits
	}
	if len(digits) <= exp+1 {
		return sign + digits + strings.Repeat("0", exp+1-len(digits)) + ".0"
	}
	return sign + digits[:exp+1] + "." + digits[exp+1:]
}
