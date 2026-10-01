package train

import (
	"encoding/json"
	"fmt"
	"math"
	"math/big"
	"sort"
	"strconv"
	"strings"

	"github.com/UnicoLab/ul_layago/laya"
)

// The trainer renders every state with laya.RenderPromptState, the renderer
// Hive's routermodel serves with, so a head trained here reads at serving
// time exactly the state it was trained on (TestPromptMatchesTrainer holds
// that path to the 81 trainer-rendered golden states).
//
// The helpers below read an input's catalog as that renderer does, for the
// trainer's own uses of the same values: the catalog shortlist (shortlist.go),
// the hashed-feature floor (floor.go) and dataset validation (dataset.go).
// They restate laya's unexported helpers of the same names rule for rule;
// exporting those from laya would remove this last copy and is a tracked
// follow-up (internal/layago/README.md, "Not done yet"). Their refusals wrap
// laya.ErrPromptInput, the renderer's.

// catalogList is Python's `input.get("catalog") or []`, as laya's
// promptList reads the catalog.
func catalogList(v any) ([]any, error) {
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
		return nil, fmt.Errorf("%w: catalog must be a list", laya.ErrPromptInput)
	}
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
				return "", fmt.Errorf("%w: number %q", laya.ErrPromptInput, lit)
			}
			return n.String(), nil
		}
		f, err := strconv.ParseFloat(lit, 64)
		if err != nil {
			return "", fmt.Errorf("%w: number %q", laya.ErrPromptInput, lit)
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
	return "", fmt.Errorf("%w: a %T value has no ported Python str()", laya.ErrPromptInput, v)
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
