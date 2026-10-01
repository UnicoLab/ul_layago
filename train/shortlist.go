package train

import (
	"fmt"
	"slices"
	"sort"
	"strings"
	"unicode"
)

// ShortlistMethod names the catalog shortlist CatalogShortlist computes.
const ShortlistMethod = "lexical-trigram-word-containment/1"

// ShortlistSpec selects a catalog shortlist: the K catalog metrics and the
// K distinct scope values most lexically similar to the question. K 0 keeps
// the whole catalog.
type ShortlistSpec struct {
	K      int    `json:"k"`
	Method string `json:"method"`
}

// ShortlistResult describes one example's shortlist.
type ShortlistResult struct {
	// Metrics and ScopeValues count the catalog's metrics and distinct
	// (dimension, value) pairs before and after.
	MetricsBefore, MetricsAfter int
	ScopeBefore, ScopeAfter     int
	// OptionsBefore and OptionsAfter count the example's options.
	OptionsBefore, OptionsAfter int
	// GoldKept is false when the example's options come from the catalog
	// (q_metric_choice, q_scope_choice) and the shortlist dropped the gold
	// option: the gold is never put back, so such an example cannot be
	// answered correctly.
	GoldKept bool
	// OptionsFromCatalog is true when the shortlist filtered the options.
	OptionsFromCatalog bool
}

// CatalogShortlist returns a copy of a Q example whose catalog keeps the k
// metrics and the k distinct scope values most similar to the question,
// and whose catalog-derived options (metric keys for q_metric_choice,
// "dimension=value" for q_scope_choice) keep only what the shortlist kept.
// It never looks at the label: when the gold option is dropped the copy's
// label is -1 and GoldKept is false. Similarity is deterministic and
// lexical: accent- and case-folded word containment plus character-trigram
// containment of each candidate (a metric's key and every label; a scope
// value) in the question; ties keep catalog order. Kept metrics are
// rendered most relevant first (so a max_len cut drops the least relevant
// lines), keep their dimensions and unit, and list only the kept scope
// values, most relevant first within each dimension. Options keep their
// order, so an option's position says nothing about its relevance.
func CatalogShortlist(ex *Example, k int) (*Example, ShortlistResult) {
	out := *ex
	res := ShortlistResult{GoldKept: true}
	catalog, _ := catalogList(ex.Input["catalog"])
	question, _ := ex.Input["question"].(string)
	q := newLexicon(question)

	type metric struct {
		idx   int
		obj   map[string]any
		key   string
		score float64
	}
	var metrics []metric
	for i, m := range catalog {
		obj, ok := m.(map[string]any)
		if !ok {
			continue
		}
		key, _ := obj["key"].(string)
		texts := []string{strings.ReplaceAll(key, "_", " ")}
		var labels any
		if v, ok := obj["labels"]; ok {
			labels = v
		} else {
			labels = obj["label"]
		}
		texts = append(texts, promptLabels(labels)...)
		best := 0.0
		for _, t := range texts {
			best = max(best, q.similarity(t))
		}
		metrics = append(metrics, metric{idx: i, obj: obj, key: key, score: best})
	}
	res.MetricsBefore = len(catalog)
	ranked := slices.Clone(metrics)
	sort.SliceStable(ranked, func(a, b int) bool { return ranked[a].score > ranked[b].score })
	keep := map[int]bool{}
	for _, m := range ranked[:min(k, len(ranked))] {
		keep[m.idx] = true
	}
	// Kept metrics are rendered most relevant first, so a max_len cut
	// drops the least relevant lines.
	keptKeys := map[string]bool{}
	var kept []metric
	for _, m := range ranked {
		if keep[m.idx] {
			kept = append(kept, m)
			keptKeys[m.key] = true
		}
	}

	// Distinct (dimension, value) pairs of the kept metrics, dimensions in
	// the renderer's sorted order.
	type pair struct{ dim, val string }
	var pairs []pair
	seen := map[pair]bool{}
	allPairs := map[pair]bool{}
	for _, m := range metrics {
		scope, _ := asMap(m.obj["scope_values"])
		dims := sortedKeys(scope)
		for _, d := range dims {
			vals, _ := asList(scope[d])
			for _, v := range vals {
				s, err := pyStr(v)
				if err != nil {
					continue
				}
				p := pair{d, pyStrip(s)}
				allPairs[p] = true
				if keep[m.idx] && !seen[p] {
					seen[p] = true
					pairs = append(pairs, p)
				}
			}
		}
	}
	res.ScopeBefore = len(allPairs)
	scores := make([]float64, len(pairs))
	order := make([]int, len(pairs))
	for i, p := range pairs {
		scores[i], order[i] = q.similarity(p.val), i
	}
	sort.SliceStable(order, func(a, b int) bool { return scores[order[a]] > scores[order[b]] })
	keptPairs := map[pair]bool{}
	rank := map[pair]int{}
	for r, i := range order[:min(k, len(order))] {
		keptPairs[pairs[i]] = true
		rank[pairs[i]] = r
	}
	res.ScopeAfter = len(keptPairs)
	res.MetricsAfter = len(kept)

	newCatalog := make([]any, 0, len(kept))
	for _, m := range kept {
		obj := make(map[string]any, len(m.obj))
		for key, v := range m.obj {
			obj[key] = v
		}
		if scope, ok := asMap(m.obj["scope_values"]); ok {
			ns := map[string]any{}
			for _, d := range sortedKeys(scope) {
				vals, _ := asList(scope[d])
				var keptVals []any
				var ranks []int
				for _, v := range vals {
					s, err := pyStr(v)
					if p := (pair{d, pyStrip(s)}); err == nil && keptPairs[p] {
						keptVals = append(keptVals, v)
						ranks = append(ranks, rank[p])
					}
				}
				// Most relevant value first within its dimension.
				idx := make([]int, len(keptVals))
				for i := range idx {
					idx[i] = i
				}
				sort.SliceStable(idx, func(a, b int) bool { return ranks[idx[a]] < ranks[idx[b]] })
				sorted := make([]any, len(idx))
				for i, j := range idx {
					sorted[i] = keptVals[j]
				}
				if len(sorted) > 0 {
					ns[d] = sorted
				}
			}
			obj["scope_values"] = ns
		}
		newCatalog = append(newCatalog, obj)
	}
	in := make(map[string]any, len(ex.Input))
	for key, v := range ex.Input {
		in[key] = v
	}
	if _, ok := ex.Input["catalog"]; ok {
		in["catalog"] = newCatalog
	}
	out.Input = in

	res.OptionsBefore = len(ex.Options)
	var keepOption func(Option) bool
	switch ex.Task {
	case "q_metric_choice":
		keepOption = func(o Option) bool { return keptKeys[o.Key] }
	case "q_scope_choice":
		keepOption = func(o Option) bool {
			d, v, ok := strings.Cut(o.Key, "=")
			return ok && keptPairs[pair{d, v}]
		}
	}
	if keepOption != nil && ex.Type == TaskChoice {
		res.OptionsFromCatalog = true
		out.Options, out.LabelDist, out.Label = nil, nil, -1
		for i, o := range ex.Options {
			if !keepOption(o) {
				continue
			}
			if i == ex.Label {
				out.Label = len(out.Options)
			}
			out.Options = append(out.Options, o)
		}
		res.GoldKept = out.Label >= 0
	}
	res.OptionsAfter = len(out.Options)
	return &out, res
}

// lexicon is a question's folded words and character trigrams.
type lexicon struct {
	words    map[string]bool
	trigrams map[string]bool
}

func newLexicon(s string) lexicon {
	l := lexicon{words: map[string]bool{}, trigrams: map[string]bool{}}
	for _, w := range foldWords(s) {
		l.words[w] = true
		for _, t := range trigrams(w) {
			l.trigrams[t] = true
		}
	}
	return l
}

// similarity is the containment of a candidate's words plus the
// containment of its character trigrams in the question, in [0, 2].
func (l lexicon) similarity(candidate string) float64 {
	words := foldWords(candidate)
	if len(words) == 0 {
		return 0
	}
	var hit, tri, triHit int
	for _, w := range words {
		if l.words[w] {
			hit++
		}
		for _, t := range trigrams(w) {
			tri++
			if l.trigrams[t] {
				triHit++
			}
		}
	}
	s := float64(hit) / float64(len(words))
	if tri > 0 {
		s += float64(triHit) / float64(tri)
	}
	return s
}

// foldWords lower-cases, strips common Latin diacritics and splits on
// anything that is not a letter or digit.
func foldWords(s string) []string {
	var b strings.Builder
	for _, r := range strings.ToLower(s) {
		if f, ok := latinFold[r]; ok {
			b.WriteString(f)
			continue
		}
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
			continue
		}
		b.WriteByte(' ')
	}
	return strings.Fields(b.String())
}

// trigrams are a word's character trigrams with boundary marks; a word of
// one or two characters is its own single gram.
func trigrams(w string) []string {
	r := []rune("_" + w + "_")
	if len(r) <= 4 {
		return []string{string(r)}
	}
	out := make([]string, 0, len(r)-2)
	for i := 0; i+3 <= len(r); i++ {
		out = append(out, string(r[i:i+3]))
	}
	return out
}

// latinFold maps lower-case Latin letters with diacritics to their base
// letters (standard library only, so no Unicode normalisation tables).
var latinFold = func() map[rune]string {
	m := map[rune]string{'ß': "ss", 'æ': "ae", 'œ': "oe", 'ø': "o", 'ł': "l", 'đ': "d", 'ı': "i"}
	groups := map[string]string{
		"a": "àáâãäåāăą", "c": "çćĉċč", "d": "ď", "e": "èéêëēĕėęě", "g": "ĝğġģ", "h": "ĥ", "i": "ìíîïĩīĭįİ",
		"j": "ĵ", "k": "ķ", "l": "ĺļľ", "n": "ñńņňŉ", "o": "òóôõöōŏő", "r": "ŕŗř", "s": "śŝşšș", "t": "ţťț",
		"u": "ùúûüũūŭůűų", "w": "ŵ", "y": "ýÿŷ", "z": "źżž",
	}
	for base, letters := range groups {
		for _, r := range letters {
			m[r] = base
		}
	}
	return m
}()

// String describes the spec for logs and manifests.
func (s ShortlistSpec) String() string {
	if s.K <= 0 {
		return "whole catalog"
	}
	return fmt.Sprintf("catalog shortlist k=%d (%s)", s.K, s.Method)
}
