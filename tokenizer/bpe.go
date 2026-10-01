package tokenizer

import (
	"container/heap"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"unicode/utf8"
)

type mergeValue struct {
	rank  int32
	newID int32
}

type bpeModel struct {
	vocab        map[string]int32
	byID         []string
	merges       map[uint64]mergeValue
	unk          int32 // -1 without an unk token
	fuseUnk      bool
	byteFallback bool
	ignoreMerges bool
	byteIDs      [256]int32 // -1 where <0xXX> is absent
}

// pairKey packs two non-negative token ids into one map key.
func pairKey(a, b int32) uint64 {
	return uint64(uint32(a))<<32 | uint64(uint32(b)) //nolint:gosec // G115: ids are validated non-negative at load
}

func parseBPE(raw json.RawMessage) (*bpeModel, error) {
	var m struct {
		Type                    string          `json:"type"`
		Dropout                 *float64        `json:"dropout"`
		UnkToken                *string         `json:"unk_token"`
		ContinuingSubwordPrefix *string         `json:"continuing_subword_prefix"`
		EndOfWordSuffix         *string         `json:"end_of_word_suffix"`
		FuseUnk                 bool            `json:"fuse_unk"`
		ByteFallback            bool            `json:"byte_fallback"`
		IgnoreMerges            bool            `json:"ignore_merges"`
		Vocab                   map[string]int  `json:"vocab"`
		Merges                  json.RawMessage `json:"merges"`
	}
	if isNull(raw) {
		return nil, fmt.Errorf("%w: tokenizer.json has no model", ErrUnsupported)
	}
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("tokenizer: model: %w", err)
	}
	if m.Type != "BPE" {
		return nil, fmt.Errorf("%w: model %q", ErrUnsupported, m.Type)
	}
	if m.Dropout != nil && *m.Dropout != 0 {
		return nil, fmt.Errorf("%w: BPE dropout makes encoding non-deterministic", ErrUnsupported)
	}
	if (m.ContinuingSubwordPrefix != nil && *m.ContinuingSubwordPrefix != "") ||
		(m.EndOfWordSuffix != nil && *m.EndOfWordSuffix != "") {
		return nil, fmt.Errorf("%w: BPE continuing_subword_prefix/end_of_word_suffix", ErrUnsupported)
	}
	b := &bpeModel{
		vocab:        make(map[string]int32, len(m.Vocab)),
		merges:       map[uint64]mergeValue{},
		unk:          -1,
		fuseUnk:      m.FuseUnk,
		byteFallback: m.ByteFallback,
		ignoreMerges: m.IgnoreMerges,
	}
	maxID := -1
	for tok, id := range m.Vocab {
		if id < 0 || id > math.MaxInt32-1 {
			return nil, fmt.Errorf("tokenizer: vocab id %d out of range", id)
		}
		b.vocab[tok] = int32(id)
		maxID = max(maxID, id)
	}
	b.byID = make([]string, maxID+1)
	for tok, id := range b.vocab {
		b.byID[id] = tok
	}
	if m.UnkToken != nil {
		id, ok := b.vocab[*m.UnkToken]
		if !ok {
			return nil, fmt.Errorf("tokenizer: unk_token %q is not in the vocabulary", *m.UnkToken)
		}
		b.unk = id
	}
	for i := range b.byteIDs {
		b.byteIDs[i] = -1
		if id, ok := b.vocab[fmt.Sprintf("<0x%02X>", i)]; ok {
			b.byteIDs[i] = id
		}
	}
	pairs, err := parseMerges(m.Merges)
	if err != nil {
		return nil, err
	}
	for rank, p := range pairs {
		a, okA := b.vocab[p[0]]
		c, okC := b.vocab[p[1]]
		n, okN := b.vocab[p[0]+p[1]]
		if !okA || !okC || !okN {
			return nil, fmt.Errorf("tokenizer: merge %d (%q %q) references tokens outside the vocabulary", rank, p[0], p[1])
		}
		// A repeated pair keeps its last rank: Hugging Face collects the
		// merges into a HashMap, so later entries overwrite earlier ones.
		b.merges[pairKey(a, c)] = mergeValue{rank: int32(rank), newID: n} //nolint:gosec // G115: rank < len(merges) <= MaxInt32 in practice
	}
	return b, nil
}

// parseMerges accepts both the ["a", "b"] pair form and the legacy "a b"
// string form.
func parseMerges(raw json.RawMessage) ([][2]string, error) {
	if isNull(raw) {
		return nil, nil
	}
	var pairs [][2]string
	if err := json.Unmarshal(raw, &pairs); err == nil {
		return pairs, nil
	}
	var legacy []string
	if err := json.Unmarshal(raw, &legacy); err != nil {
		return nil, fmt.Errorf("tokenizer: merges: %w", err)
	}
	pairs = make([][2]string, len(legacy))
	for i, s := range legacy {
		a, b, ok := strings.Cut(s, " ")
		if !ok || a == "" || b == "" || strings.Contains(b, " ") {
			return nil, fmt.Errorf("tokenizer: legacy merge %d %q is not \"a b\"", i, s)
		}
		pairs[i] = [2]string{a, b}
	}
	return pairs, nil
}

type symbol struct {
	id         int32
	prev, next int
	size       int // bytes of normalized text covered
}

// encodeWord appends the BPE tokens of one pre-tokenized word. It follows
// the Hugging Face model: characters found in the vocabulary become symbols,
// others fall back to <0xXX> byte tokens or to a (possibly fused) unk, and
// merges are applied lowest rank first, leftmost first on ties.
func (b *bpeModel) encodeWord(enc *Encoding, word normalized) {
	w := word.s
	if w == "" {
		return
	}
	if b.ignoreMerges {
		if id, ok := b.vocab[w]; ok {
			b.emit(enc, word, []symbol{{id: id, size: len(w)}})
			return
		}
	}
	syms := make([]symbol, 0, len(w))
	pendingUnk, pendingSize := false, 0
	for i := 0; i < len(w); {
		_, size := utf8.DecodeRuneInString(w[i:])
		ch := w[i : i+size]
		i += size
		if id, ok := b.vocab[ch]; ok {
			if pendingUnk {
				syms = append(syms, symbol{id: b.unk, size: pendingSize})
				pendingUnk = false
			}
			syms = append(syms, symbol{id: id, size: size})
			continue
		}
		if b.byteFallback {
			ok := true
			for j := 0; j < len(ch); j++ {
				if b.byteIDs[ch[j]] < 0 {
					ok = false
					break
				}
			}
			if ok {
				// Hugging Face does not flush a pending unk here, so a
				// preceding unk lands after these byte tokens; mirrored for
				// exact parity.
				for j := 0; j < len(ch); j++ {
					syms = append(syms, symbol{id: b.byteIDs[ch[j]], size: 1})
				}
				continue
			}
		}
		if b.unk < 0 {
			continue // no unk token: the character is dropped
		}
		switch {
		case pendingUnk && b.fuseUnk:
			pendingSize += size
		case pendingUnk:
			syms = append(syms, symbol{id: b.unk, size: pendingSize})
			pendingSize = size
		default:
			pendingUnk, pendingSize = true, size
		}
	}
	if pendingUnk {
		syms = append(syms, symbol{id: b.unk, size: pendingSize})
	}
	for i := range syms {
		syms[i].prev, syms[i].next = i-1, i+1
	}
	if len(syms) > 0 {
		syms[len(syms)-1].next = -1
	}
	b.mergeAll(syms)
	b.emit(enc, word, syms)
}

func (b *bpeModel) emit(enc *Encoding, word normalized, syms []symbol) {
	pos := 0
	for _, s := range syms {
		if s.size == 0 {
			continue // merged away
		}
		enc.IDs = append(enc.IDs, s.id)
		enc.Tokens = append(enc.Tokens, b.byID[s.id])
		enc.Offsets = append(enc.Offsets, word.originalRange(pos, pos+s.size))
		enc.Special = append(enc.Special, false)
		pos += s.size
	}
}

type pendingMerge struct {
	pos   int
	rank  int32
	newID int32
}

type mergeQueue []pendingMerge

func (q mergeQueue) Len() int { return len(q) }
func (q mergeQueue) Less(i, j int) bool {
	if q[i].rank != q[j].rank {
		return q[i].rank < q[j].rank
	}
	return q[i].pos < q[j].pos
}
func (q mergeQueue) Swap(i, j int) { q[i], q[j] = q[j], q[i] }
func (q *mergeQueue) Push(x any)   { *q = append(*q, x.(pendingMerge)) }
func (q *mergeQueue) Pop() any {
	old := *q
	x := old[len(old)-1]
	*q = old[:len(old)-1]
	return x
}

// mergeAll is Word::merge_all without dropout.
func (b *bpeModel) mergeAll(syms []symbol) {
	q := mergeQueue{}
	for i := 0; i+1 < len(syms); i++ {
		if m, ok := b.merges[pairKey(syms[i].id, syms[i+1].id)]; ok {
			q = append(q, pendingMerge{pos: i, rank: m.rank, newID: m.newID})
		}
	}
	heap.Init(&q)
	for q.Len() > 0 {
		top := heap.Pop(&q).(pendingMerge)
		cur := &syms[top.pos]
		if cur.size == 0 || cur.next == -1 {
			continue
		}
		nextPos := cur.next
		right := syms[nextPos]
		m, ok := b.merges[pairKey(cur.id, right.id)]
		if !ok || m.newID != top.newID {
			continue // stale entry
		}
		cur.id = top.newID
		cur.size += right.size
		cur.next = right.next
		syms[nextPos].size = 0
		if right.next >= 0 && right.next < len(syms) {
			syms[right.next].prev = top.pos
		}
		if cur.prev >= 0 {
			if m, ok := b.merges[pairKey(syms[cur.prev].id, cur.id)]; ok {
				heap.Push(&q, pendingMerge{pos: cur.prev, rank: m.rank, newID: m.newID})
			}
		}
		if cur.next >= 0 && cur.next < len(syms) {
			if m, ok := b.merges[pairKey(cur.id, syms[cur.next].id)]; ok {
				heap.Push(&q, pendingMerge{pos: top.pos, rank: m.rank, newID: m.newID})
			}
		}
	}
}
