// Package laya is a Go port of the inference contract of Laya
// (github.com/NandhaKishorM/laya, Apache-2.0, source commit 573e5b62,
// version 0.3.5): typed questions (choice, score, noul) over a state, one
// token sequence per question laid out as
//
//	[CLS] <type> question: instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
//
// a ModernBERT backbone, a two-layer transformer decision head, a per-option
// marker scorer with a softmax per question, and an act/escalate head.
//
// It reproduces laya/common.py and laya/agent.py for inference, including
// their truncation rules and the Python slicing quirks noted inline. Outputs
// are model nominations: probabilities from an uncalibrated or synthetically
// calibrated head are not evidence and carry no numeric authority.
package laya

import (
	"fmt"
	"strings"

	"github.com/UnicoLab/ul_layago/tokenizer"
)

// QType is a Laya question type.
type QType int

// Question types, numbered as laya/common.py QTYPES.
const (
	Choice QType = 0
	Score  QType = 1
	Noul   QType = 2
)

// String returns the Laya type name.
func (q QType) String() string {
	switch q {
	case Choice:
		return "choice"
	case Score:
		return "score"
	case Noul:
		return "noul"
	default:
		return fmt.Sprintf("qtype(%d)", int(q))
	}
}

// Option is one choice criterion. An empty Description renders the key alone,
// as a None or "" criterion does in Laya.
type Option struct {
	Key         string
	Description string
}

// Question is one typed question. Choice uses Options (in order), Score uses
// Levels, and Noul may override the False and True criterion texts.
type Question struct {
	ID           string
	Type         QType
	Instructions string
	Options      []Option
	Levels       []string
	False, True  string
}

// RenderOptions returns option texts in label order (render_options).
func RenderOptions(q Question) []string {
	switch q.Type {
	case Choice:
		out := make([]string, len(q.Options))
		for i, o := range q.Options {
			if o.Description == "" {
				out[i] = o.Key
			} else {
				out[i] = o.Key + ": " + o.Description
			}
		}
		return out
	case Score:
		out := make([]string, len(q.Levels))
		for i, l := range q.Levels {
			out[i] = fmt.Sprintf("level %d: %s", i, l)
		}
		return out
	default:
		f, t := q.False, q.True
		if f == "" {
			f = "no, the statement does not hold"
		}
		if t == "" {
			t = "yes, the statement holds"
		}
		return []string{"false: " + f, "true: " + t}
	}
}

// SpecialTokens are the ids build_sequence needs, resolved from
// tokenizer_config.json.
type SpecialTokens struct {
	CLS, SEP, Mask, Pad int32
	MaskToken           string
}

// maxOptionTokens is Laya's per-option cap after the marker.
const maxOptionTokens = 48

// BuildSequence ports laya/common.py build_sequence. optionOrder may be nil
// for label order. It returns the token ids and the marker positions.
func BuildSequence(tok *tokenizer.Tokenizer, sp SpecialTokens, state string, q Question,
	maxLen, headMaxLen int, optionOrder []int, truncateLeft bool) ([]int32, []int, error) {
	encode := func(s string) ([]int32, error) {
		enc, err := tok.Encode(s, false)
		return enc.IDs, err
	}
	opts := RenderOptions(q)
	order := optionOrder
	if order == nil {
		order = make([]int, len(opts))
		for i := range order {
			order[i] = i
		}
	}
	ins := strings.ReplaceAll(q.Instructions, sp.MaskToken, " ")
	headIDs, err := encode(q.Type.String() + " question: " + ins)
	if err != nil {
		return nil, nil, err
	}
	optIDs := make([][]int32, 0, len(order))
	for _, i := range order {
		if i < 0 || i >= len(opts) {
			return nil, nil, fmt.Errorf("laya: option order index %d out of range", i)
		}
		ids, err := encode(" " + strings.ReplaceAll(opts[i], sp.MaskToken, " "))
		if err != nil {
			return nil, nil, err
		}
		o := append([]int32{sp.Mask}, ids[:min(len(ids), maxOptionTokens)]...)
		optIDs = append(optIDs, o)
	}
	sumLen := func() int {
		n := 0
		for _, o := range optIDs {
			n += len(o)
		}
		return n
	}
	optBudget := headMaxLen - sumLen()
	if optBudget < 16 {
		per := max(4, floorDiv(headMaxLen-16, max(1, len(optIDs))))
		for i, o := range optIDs {
			optIDs[i] = o[:min(len(o), per)]
		}
		optBudget = headMaxLen - sumLen()
	}
	headIDs = headIDs[:min(len(headIDs), max(8, optBudget))]
	ids := make([]int32, 0, maxLen)
	ids = append(ids, sp.CLS)
	ids = append(ids, headIDs...)
	ids = append(ids, sp.SEP)
	markers := make([]int, 0, len(optIDs))
	for _, o := range optIDs {
		markers = append(markers, len(ids))
		ids = append(ids, o...)
	}
	ids = append(ids, sp.SEP)
	room := max(0, maxLen-len(ids)-1)
	st, err := encode(strings.ReplaceAll(state, sp.MaskToken, " "))
	if err != nil {
		return nil, nil, err
	}
	switch {
	case truncateLeft && room == 0:
		// Python's st[-0:] is the whole list, not an empty one; the final
		// ids[:max_len] cut then applies. Kept for exact parity.
	case truncateLeft:
		st = st[max(0, len(st)-room):]
	default:
		st = st[:min(len(st), room)]
	}
	ids = append(ids, st...)
	ids = append(ids, sp.SEP)
	if len(ids) > maxLen {
		ids = ids[:maxLen]
	}
	kept := markers[:0]
	for _, m := range markers {
		if m < maxLen {
			kept = append(kept, m)
		}
	}
	return ids, kept, nil
}

// floorDiv is Python's // for ints.
func floorDiv(a, b int) int {
	q := a / b
	if (a%b != 0) && ((a < 0) != (b < 0)) {
		q--
	}
	return q
}
