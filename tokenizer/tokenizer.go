// Package tokenizer loads a Hugging Face tokenizer.json and encodes text with
// the same token ids and offsets the Rust tokenizers library produces, for
// the BPE family used by mmBERT and Gemma-2 (Metaspace/Replace normalization,
// byte fallback, added tokens, TemplateProcessing).
//
// It is fail-closed: a tokenizer.json that declares any component outside the
// supported set is rejected at load time instead of being approximated,
// because a silently different tokenization changes every downstream
// probability. Supported components:
//
//   - normalizer: Replace (literal pattern), Prepend, Sequence, or none;
//   - pre_tokenizer: Metaspace (prepend_scheme always|first|never, split), Sequence, or none;
//   - model: BPE with byte_fallback, fuse_unk, ignore_merges, no dropout and no subword prefix/suffix;
//   - post_processor: TemplateProcessing, or none;
//   - added tokens with special, normalized, lstrip, rstrip and single_word flags.
//
// The decoder is not needed for inference and is ignored, as are the file's
// truncation and padding settings: callers truncate explicitly, which is what
// transformers' tokenizer __call__ does as well.
package tokenizer

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"
)

// ErrUnsupported reports a tokenizer.json component this package does not
// implement.
var ErrUnsupported = errors.New("tokenizer: unsupported component")

// Encoding is the result of encoding one text.
type Encoding struct {
	IDs    []int32
	Tokens []string
	// Offsets are [start, end) byte offsets into the input text. Tokens added
	// by the post-processor have [0, 0).
	Offsets [][2]int
	// Special marks tokens inserted by the post-processor template.
	Special []bool
}

// AddedToken mirrors one tokenizer.json added_tokens entry.
type AddedToken struct {
	ID         int32  `json:"id"`
	Content    string `json:"content"`
	SingleWord bool   `json:"single_word"`
	LStrip     bool   `json:"lstrip"`
	RStrip     bool   `json:"rstrip"`
	Normalized bool   `json:"normalized"`
	Special    bool   `json:"special"`
}

// Tokenizer is an immutable, concurrency-safe encoder.
type Tokenizer struct {
	normalizers []normalizerStep
	pre         []metaspace
	bpe         *bpeModel
	added       []AddedToken
	rawTrie     matcher // added tokens matched on the original text
	normTrie    matcher // added tokens matched after normalization
	template    []templatePiece
	ids         map[string]int32 // model vocabulary plus added tokens
	byID        []string
}

type normalizerStep struct {
	kind            string // "replace" or "prepend"
	pattern, output string
}

type metaspace struct {
	replacement string
	scheme      string // always | first | never
	split       bool
}

type templatePiece struct {
	special bool
	ids     []int32
	tokens  []string
}

type rawComponent struct {
	Type string `json:"type"`
}

type fileJSON struct {
	AddedTokens   []AddedToken    `json:"added_tokens"`
	Normalizer    json.RawMessage `json:"normalizer"`
	PreTokenizer  json.RawMessage `json:"pre_tokenizer"`
	PostProcessor json.RawMessage `json:"post_processor"`
	Model         json.RawMessage `json:"model"`
}

// LoadFile reads and parses a tokenizer.json file.
func LoadFile(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path) //nolint:gosec // G304: the caller chooses which local artifact to load
	if err != nil {
		return nil, err
	}
	return Load(data)
}

// Load parses tokenizer.json bytes.
func Load(data []byte) (*Tokenizer, error) {
	var f fileJSON
	if err := json.Unmarshal(data, &f); err != nil {
		return nil, fmt.Errorf("tokenizer: parse tokenizer.json: %w", err)
	}
	t := &Tokenizer{}
	var err error
	if t.normalizers, err = parseNormalizer(f.Normalizer); err != nil {
		return nil, err
	}
	if t.pre, err = parsePreTokenizer(f.PreTokenizer); err != nil {
		return nil, err
	}
	if t.bpe, err = parseBPE(f.Model); err != nil {
		return nil, err
	}
	if t.template, err = parseTemplate(f.PostProcessor); err != nil {
		return nil, err
	}
	t.added = f.AddedTokens
	t.ids = make(map[string]int32, len(t.bpe.vocab)+len(f.AddedTokens))
	for tok, id := range t.bpe.vocab {
		t.ids[tok] = id
	}
	t.byID = append([]string(nil), t.bpe.byID...)
	var raw, norm []matchEntry
	for i, a := range f.AddedTokens {
		if a.Content == "" {
			return nil, fmt.Errorf("tokenizer: added token %d has empty content", a.ID)
		}
		if a.ID < 0 || a.ID > 1<<28 {
			return nil, fmt.Errorf("tokenizer: added token %q has id %d out of range", a.Content, a.ID)
		}
		// Added tokens are looked up by id and content, but only the model
		// vocabulary takes part in BPE, as in Hugging Face.
		t.ids[a.Content] = a.ID
		for int(a.ID) >= len(t.byID) {
			t.byID = append(t.byID, "")
		}
		t.byID[a.ID] = a.Content
		if a.Normalized {
			// Hugging Face normalizes a normalized added token's content
			// before matching it against normalized text.
			content := t.normalize(newNormalized(a.Content, 0)).s
			if content != "" {
				norm = append(norm, matchEntry{content: content, index: i})
			}
		} else {
			raw = append(raw, matchEntry{content: a.Content, index: i})
		}
	}
	t.rawTrie = newMatcher(raw)
	t.normTrie = newMatcher(norm)
	return t, nil
}

func parseNormalizer(raw json.RawMessage) ([]normalizerStep, error) {
	if isNull(raw) {
		return nil, nil
	}
	var head rawComponent
	if err := json.Unmarshal(raw, &head); err != nil {
		return nil, fmt.Errorf("tokenizer: normalizer: %w", err)
	}
	switch head.Type {
	case "Sequence":
		var seq struct {
			Normalizers []json.RawMessage `json:"normalizers"`
		}
		if err := json.Unmarshal(raw, &seq); err != nil {
			return nil, err
		}
		var out []normalizerStep
		for _, n := range seq.Normalizers {
			steps, err := parseNormalizer(n)
			if err != nil {
				return nil, err
			}
			out = append(out, steps...)
		}
		return out, nil
	case "Replace":
		var rep struct {
			Pattern map[string]string `json:"pattern"`
			Content string            `json:"content"`
		}
		if err := json.Unmarshal(raw, &rep); err != nil {
			return nil, err
		}
		pattern, ok := rep.Pattern["String"]
		if !ok || len(rep.Pattern) != 1 || pattern == "" {
			return nil, fmt.Errorf("%w: Replace normalizer needs a literal String pattern", ErrUnsupported)
		}
		return []normalizerStep{{kind: "replace", pattern: pattern, output: rep.Content}}, nil
	case "Prepend":
		var p struct {
			Prepend string `json:"prepend"`
		}
		if err := json.Unmarshal(raw, &p); err != nil {
			return nil, err
		}
		return []normalizerStep{{kind: "prepend", output: p.Prepend}}, nil
	default:
		return nil, fmt.Errorf("%w: normalizer %q", ErrUnsupported, head.Type)
	}
}

func parsePreTokenizer(raw json.RawMessage) ([]metaspace, error) {
	if isNull(raw) {
		return nil, nil
	}
	var head rawComponent
	if err := json.Unmarshal(raw, &head); err != nil {
		return nil, fmt.Errorf("tokenizer: pre_tokenizer: %w", err)
	}
	switch head.Type {
	case "Sequence":
		var seq struct {
			PreTokenizers []json.RawMessage `json:"pretokenizers"`
		}
		if err := json.Unmarshal(raw, &seq); err != nil {
			return nil, err
		}
		var out []metaspace
		for _, p := range seq.PreTokenizers {
			steps, err := parsePreTokenizer(p)
			if err != nil {
				return nil, err
			}
			out = append(out, steps...)
		}
		return out, nil
	case "Metaspace":
		var m struct {
			Replacement     string `json:"replacement"`
			PrependScheme   string `json:"prepend_scheme"`
			AddPrefixSpace  *bool  `json:"add_prefix_space"`
			Split           *bool  `json:"split"`
			LegacyReplaceme string `json:"str_rep"`
		}
		if err := json.Unmarshal(raw, &m); err != nil {
			return nil, err
		}
		if utf8.RuneCountInString(m.Replacement) != 1 {
			return nil, fmt.Errorf("%w: Metaspace replacement must be one character", ErrUnsupported)
		}
		scheme := m.PrependScheme
		if scheme == "" {
			scheme = "always"
			if m.AddPrefixSpace != nil && !*m.AddPrefixSpace {
				scheme = "never"
			}
		}
		if scheme != "always" && scheme != "first" && scheme != "never" {
			return nil, fmt.Errorf("%w: Metaspace prepend_scheme %q", ErrUnsupported, scheme)
		}
		split := true
		if m.Split != nil {
			split = *m.Split
		}
		return []metaspace{{replacement: m.Replacement, scheme: scheme, split: split}}, nil
	default:
		return nil, fmt.Errorf("%w: pre_tokenizer %q", ErrUnsupported, head.Type)
	}
}

func parseTemplate(raw json.RawMessage) ([]templatePiece, error) {
	if isNull(raw) {
		return nil, nil
	}
	var tp struct {
		Type   string `json:"type"`
		Single []struct {
			SpecialToken *struct {
				ID string `json:"id"`
			} `json:"SpecialToken"`
			Sequence *struct {
				ID string `json:"id"`
			} `json:"Sequence"`
		} `json:"single"`
		SpecialTokens map[string]struct {
			IDs    []int32  `json:"ids"`
			Tokens []string `json:"tokens"`
		} `json:"special_tokens"`
	}
	if err := json.Unmarshal(raw, &tp); err != nil {
		return nil, fmt.Errorf("tokenizer: post_processor: %w", err)
	}
	if tp.Type != "TemplateProcessing" {
		return nil, fmt.Errorf("%w: post_processor %q", ErrUnsupported, tp.Type)
	}
	var out []templatePiece
	sequences := 0
	for _, item := range tp.Single {
		switch {
		case item.SpecialToken != nil:
			st, ok := tp.SpecialTokens[item.SpecialToken.ID]
			if !ok || len(st.IDs) == 0 || len(st.IDs) != len(st.Tokens) {
				return nil, fmt.Errorf("tokenizer: template special token %q is not declared", item.SpecialToken.ID)
			}
			out = append(out, templatePiece{special: true, ids: st.IDs, tokens: st.Tokens})
		case item.Sequence != nil:
			if item.Sequence.ID != "A" {
				return nil, fmt.Errorf("%w: single template references sequence %q", ErrUnsupported, item.Sequence.ID)
			}
			sequences++
			out = append(out, templatePiece{})
		default:
			return nil, fmt.Errorf("%w: template piece", ErrUnsupported)
		}
	}
	if sequences != 1 {
		return nil, fmt.Errorf("%w: single template must contain sequence A exactly once", ErrUnsupported)
	}
	return out, nil
}

func isNull(raw json.RawMessage) bool {
	s := strings.TrimSpace(string(raw))
	return s == "" || s == "null"
}

// TokenID returns the id of a vocabulary or added token.
func (t *Tokenizer) TokenID(token string) (int32, bool) {
	id, ok := t.ids[token]
	return id, ok
}

// Token returns the string of id, or "" when id is out of range.
func (t *Tokenizer) Token(id int32) string {
	if id < 0 || int(id) >= len(t.byID) {
		return ""
	}
	return t.byID[id]
}

// VocabSize returns one more than the largest token id.
func (t *Tokenizer) VocabSize() int { return len(t.byID) }

// AddedTokens returns a copy of the added-token table.
func (t *Tokenizer) AddedTokens() []AddedToken { return append([]AddedToken(nil), t.added...) }

func (t *Tokenizer) normalize(n normalized) normalized {
	for _, step := range t.normalizers {
		switch step.kind {
		case "replace":
			n = n.replace(step.pattern, step.output)
		case "prepend":
			n = n.prepend(step.output)
		}
	}
	return n
}

// segment is a piece of input either bound to an added token or awaiting the
// model.
type segment struct {
	text  normalized
	added int // index into t.added, or -1
}

// Encode tokenizes text. addSpecial applies the post-processor template.
func (t *Tokenizer) Encode(text string, addSpecial bool) (Encoding, error) {
	if !utf8.ValidString(text) {
		return Encoding{}, errors.New("tokenizer: input is not valid UTF-8")
	}
	var enc Encoding
	if addSpecial && t.template != nil {
		for _, piece := range t.template {
			if !piece.special {
				t.encodeText(&enc, text)
				continue
			}
			for i, id := range piece.ids {
				enc.IDs = append(enc.IDs, id)
				enc.Tokens = append(enc.Tokens, piece.tokens[i])
				enc.Offsets = append(enc.Offsets, [2]int{0, 0})
				enc.Special = append(enc.Special, true)
			}
		}
		return enc, nil
	}
	t.encodeText(&enc, text)
	return enc, nil
}

func (t *Tokenizer) encodeText(enc *Encoding, text string) {
	// 1. Split on non-normalized added tokens in the original text, 2.
	// normalize the rest, 3. split on normalized added tokens.
	var segments []segment
	for _, m := range t.rawTrie.split(text, t.added) {
		if m.added >= 0 {
			segments = append(segments, segment{text: newNormalized(text[m.start:m.end], m.start), added: m.added})
			continue
		}
		n := t.normalize(newNormalized(text[m.start:m.end], m.start))
		for _, nm := range t.normTrie.split(n.s, t.added) {
			segments = append(segments, segment{text: n.slice(nm.start, nm.end), added: nm.added})
		}
	}
	for _, seg := range segments {
		if seg.added >= 0 {
			a := t.added[seg.added]
			enc.IDs = append(enc.IDs, a.ID)
			enc.Tokens = append(enc.Tokens, a.Content)
			enc.Offsets = append(enc.Offsets, seg.text.originalRange(0, len(seg.text.s)))
			enc.Special = append(enc.Special, false)
			continue
		}
		for _, word := range t.preTokenize(seg.text) {
			t.bpe.encodeWord(enc, word)
		}
	}
}

// preTokenize applies the Metaspace pre-tokenizers to one normalized segment.
func (t *Tokenizer) preTokenize(n normalized) []normalized {
	pieces := []normalized{n}
	for _, m := range t.pre {
		var next []normalized
		for _, p := range pieces {
			p = p.replace(" ", m.replacement)
			switch m.scheme {
			case "always":
				if !strings.HasPrefix(p.s, m.replacement) {
					p = p.prepend(m.replacement)
				}
			case "first":
				if !strings.HasPrefix(p.s, m.replacement) && len(p.align) > 0 && p.align[0][0] == 0 {
					p = p.prepend(m.replacement)
				}
			}
			if m.split {
				next = append(next, splitMergedWithNext(p, m.replacement)...)
			} else {
				next = append(next, p)
			}
		}
		pieces = next
	}
	return pieces
}

// splitMergedWithNext splits on every occurrence of delim and attaches each
// delimiter to the text that follows it, so "▁a▁▁b" becomes "▁a", "▁", "▁b".
func splitMergedWithNext(n normalized, delim string) []normalized {
	if n.s == "" {
		return []normalized{n}
	}
	var out []normalized
	start := 0
	for i := 0; i < len(n.s); {
		if strings.HasPrefix(n.s[i:], delim) {
			if i > start {
				out = append(out, n.slice(start, i))
			}
			start = i
			i += len(delim)
			continue
		}
		_, size := utf8.DecodeRuneInString(n.s[i:])
		i += size
	}
	if start < len(n.s) {
		out = append(out, n.slice(start, len(n.s)))
	}
	return out
}

// matcher finds added tokens with leftmost-longest semantics.
type matcher struct {
	byFirst map[byte][]matchEntry // candidates sorted longest first
}

type matchEntry struct {
	content string
	index   int
}

type match struct {
	start, end int
	added      int
}

func newMatcher(entries []matchEntry) matcher {
	m := matcher{byFirst: map[byte][]matchEntry{}}
	for _, e := range entries {
		m.byFirst[e.content[0]] = append(m.byFirst[e.content[0]], e)
	}
	for k := range m.byFirst {
		list := m.byFirst[k]
		sort.SliceStable(list, func(i, j int) bool { return len(list[i].content) > len(list[j].content) })
	}
	return m
}

// split partitions s into added-token matches and the text between them,
// applying single_word, lstrip and rstrip the way AddedVocabulary does.
func (m matcher) split(s string, added []AddedToken) []match {
	if s == "" {
		return []match{{0, 0, -1}}
	}
	var out []match
	startOffset := 0
	for i := 0; i < len(s); {
		found, length := -1, 0
		for _, c := range m.byFirst[s[i]] {
			if strings.HasPrefix(s[i:], c.content) {
				// c.content is the normalized form for normalized tokens, so
				// its length, not the raw content's, is the match length.
				found, length = c.index, len(c.content)
				break
			}
		}
		if found < 0 {
			i++
			continue
		}
		tok := added[found]
		start, stop := i, i+length
		// The scan continues after the literal match whatever happens to it,
		// like the non-overlapping aho-corasick iterator Hugging Face uses.
		i = stop
		if tok.SingleWord {
			startOK := start == 0 || !endsWithWordChar(s[:start])
			stopOK := stop == len(s) || !startsWithWordChar(s[stop:])
			if !startOK || !stopOK {
				continue // not a whole word
			}
		}
		if tok.LStrip {
			start = max(len(strings.TrimRightFunc(s[:start], unicode.IsSpace)), startOffset)
		}
		if tok.RStrip {
			stop += len(s[stop:]) - len(strings.TrimLeftFunc(s[stop:], unicode.IsSpace))
		}
		if startOffset < start {
			out = append(out, match{startOffset, start, -1})
		}
		out = append(out, match{start, stop, found})
		startOffset = stop
	}
	if startOffset < len(s) {
		out = append(out, match{startOffset, len(s), -1})
	}
	return out
}

// isWordChar matches the Rust regex crate's Unicode \w class: Alphabetic, marks,
// decimal digits, connector punctuation and the joiners.
func isWordChar(r rune) bool {
	return unicode.IsLetter(r) || unicode.Is(unicode.Nl, r) || unicode.Is(unicode.Other_Alphabetic, r) ||
		unicode.IsMark(r) || unicode.Is(unicode.Nd, r) || unicode.Is(unicode.Pc, r) ||
		r == 0x200C || r == 0x200D
}

func endsWithWordChar(s string) bool {
	r, _ := utf8.DecodeLastRuneInString(s)
	return r != utf8.RuneError && isWordChar(r)
}

func startsWithWordChar(s string) bool {
	r, _ := utf8.DecodeRuneInString(s)
	return r != utf8.RuneError && isWordChar(r)
}
