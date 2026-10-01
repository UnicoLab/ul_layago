package tokenizer

import (
	"strings"
	"unicode/utf8"
)

// normalized is a string under transformation that remembers, for every
// byte, the original byte range it came from. It follows Hugging Face
// tokenizers' NormalizedString alignment rules so byte offsets can be mapped
// back to the caller's input exactly.
type normalized struct {
	s     string
	align [][2]int // one entry per byte of s: [start, end) in the original input
}

// newNormalized wraps orig, which starts at byte base of the original input.
// Every byte of a character aligns to the whole character's range.
func newNormalized(orig string, base int) normalized {
	n := normalized{s: orig, align: make([][2]int, len(orig))}
	for i := 0; i < len(orig); {
		_, size := utf8.DecodeRuneInString(orig[i:])
		for j := 0; j < size; j++ {
			n.align[i+j] = [2]int{base + i, base + i + size}
		}
		i += size
	}
	return n
}

// slice returns the sub-string [a, b) of the normalized text.
func (n normalized) slice(a, b int) normalized {
	return normalized{s: n.s[a:b], align: n.align[a:b]}
}

// originalRange maps normalized bytes [a, b) to the original input.
func (n normalized) originalRange(a, b int) [2]int {
	if a >= b || b > len(n.align) {
		if len(n.align) == 0 {
			return [2]int{0, 0}
		}
		if a >= len(n.align) {
			end := n.align[len(n.align)-1][1]
			return [2]int{end, end}
		}
		return [2]int{n.align[a][0], n.align[a][0]}
	}
	return [2]int{n.align[a][0], n.align[b-1][1]}
}

// replace substitutes every non-overlapping occurrence of pattern with
// content. Inserted bytes align with the last byte of the replaced match,
// which is what NormalizedString::replace does.
func (n normalized) replace(pattern, content string) normalized {
	if pattern == "" || !strings.Contains(n.s, pattern) {
		return n
	}
	var b strings.Builder
	align := make([][2]int, 0, len(n.align))
	last := 0
	for {
		i := strings.Index(n.s[last:], pattern)
		if i < 0 {
			break
		}
		start, end := last+i, last+i+len(pattern)
		b.WriteString(n.s[last:start])
		align = append(align, n.align[last:start]...)
		b.WriteString(content)
		a := n.align[end-1]
		for j := 0; j < len(content); j++ {
			align = append(align, a)
		}
		last = end
	}
	b.WriteString(n.s[last:])
	align = append(align, n.align[last:]...)
	return normalized{s: b.String(), align: align}
}

// prepend inserts s before the first character, aligned with that character.
// An empty string is left unchanged.
func (n normalized) prepend(s string) normalized {
	if n.s == "" || s == "" {
		return n
	}
	a := n.align[0]
	align := make([][2]int, 0, len(s)+len(n.align))
	for j := 0; j < len(s); j++ {
		align = append(align, a)
	}
	return normalized{s: s + n.s, align: append(align, n.align...)}
}
