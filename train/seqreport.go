package train

import (
	"math"
	"slices"

	"github.com/UnicoLab/ul_layago/laya"
)

// SeqCell summarises the sequences of one group of items: how long each
// would be uncut, how many are cut at max_len, and how much of the
// rendered state the trunk still sees.
type SeqCell struct {
	N int `json:"n"`
	// FullTokens* describe the uncut sequence (the same layout without the
	// max_len cut); Tokens* the sequence the trunk reads.
	FullTokensP50  int     `json:"full_tokens_p50"`
	FullTokensP95  int     `json:"full_tokens_p95"`
	FullTokensMean float64 `json:"full_tokens_mean"`
	TokensMean     float64 `json:"tokens_mean"`
	// OverMaxLen is the share of items whose uncut sequence exceeds max_len.
	OverMaxLen float64 `json:"over_max_len"`
	// StateKeptP50 and StateKeptMean are the share of the rendered state's
	// tokens that fit (1 when nothing is cut).
	StateKeptP50  float64 `json:"state_kept_p50"`
	StateKeptMean float64 `json:"state_kept_mean"`
	// Unscorable counts items whose option markers do not fit.
	Unscorable int `json:"unscorable"`
	// Shortlist is present when the items were laid out over a catalog
	// shortlist.
	Shortlist *ShortlistCell `json:"shortlist,omitempty"`
}

// ShortlistCell summarises a catalog shortlist over a group of items.
type ShortlistCell struct {
	MetricsBefore float64 `json:"metrics_before_mean"`
	MetricsAfter  float64 `json:"metrics_after_mean"`
	ScopeBefore   float64 `json:"scope_values_before_mean"`
	ScopeAfter    float64 `json:"scope_values_after_mean"`
	// CatalogOptionItems counts the items whose options come from the
	// catalog (q_metric_choice, q_scope_choice); Recall is the share of
	// them whose gold option the shortlist kept.
	CatalogOptionItems int      `json:"catalog_option_items"`
	Recall             *float64 `json:"gold_recall"`
	OptionsBefore      float64  `json:"options_before_mean"`
	OptionsAfter       float64  `json:"options_after_mean"`
}

// SeqReport is the sequence-length diagnostic of a prepared dataset, by
// role ("<role>"), role and language ("<role>/<lang>"), role and task
// ("<role>/<task>") and role, language and task ("<role>/<lang>/<task>").
type SeqReport struct {
	MaxLen     int                `json:"max_len"`
	HeadMaxLen int                `json:"head_max_len"`
	Shortlist  *ShortlistSpec     `json:"shortlist"`
	Cells      map[string]SeqCell `json:"cells"`
}

type seqObs struct {
	full, tokens int
	stateKept    float64
	scorable     bool
	sl           *ShortlistResult
}

// SequenceReport lays every prepared item out again without the max_len cut
// and reports, per group, the uncut and cut lengths and the share of the
// rendered state the trunk reads; with a shortlist, also its sizes and
// gold recall. It reads only the prepared items (their state, question and
// sequence) and the tokenizer.
func SequenceReport(p *Prepared, t *Trunk) (*SeqReport, error) {
	groups := map[string][]seqObs{}
	source := map[string]*Example{}
	if p.Shortlist != nil {
		for i := range p.Dataset.Examples {
			source[p.Dataset.Examples[i].ID] = &p.Dataset.Examples[i]
		}
	}
	for _, it := range p.Items {
		full, _, err := laya.BuildSequence(t.Tokenizer, t.Special, it.State, it.Question, 1<<20, t.Spec.HeadMaxLen, nil, false)
		if err != nil {
			return nil, err
		}
		st, err := t.Tokenizer.Encode(it.State, false)
		if err != nil {
			return nil, err
		}
		prefix := len(full) - len(st.IDs) - 1
		kept := 1.0
		if n := len(st.IDs); n > 0 {
			kept = math.Max(0, float64(len(it.IDs)-prefix-1)) / float64(n)
		}
		o := seqObs{full: len(full), tokens: len(it.IDs), stateKept: math.Min(1, kept), scorable: it.Scorable}
		if src := source[it.Ex.ID]; src != nil {
			_, res := CatalogShortlist(src, p.Shortlist.K)
			o.sl = &res
		}
		lang := primaryLanguage(it.Ex.Lang)
		for _, key := range []string{it.Role, it.Role + "/" + lang, it.Role + "/" + it.Ex.Task, it.Role + "/" + lang + "/" + it.Ex.Task} {
			groups[key] = append(groups[key], o)
		}
	}
	r := &SeqReport{MaxLen: t.Spec.MaxLen, HeadMaxLen: t.Spec.HeadMaxLen, Shortlist: p.Shortlist, Cells: map[string]SeqCell{}}
	for key, obs := range groups {
		r.Cells[key] = seqCell(obs, t.Spec.MaxLen)
	}
	return r, nil
}

func seqCell(obs []seqObs, maxLen int) SeqCell {
	c := SeqCell{N: len(obs)}
	fulls := make([]int, len(obs))
	kept := make([]float64, len(obs))
	var sumFull, sumTok, sumKept float64
	over := 0
	var sl ShortlistCell
	slN, hit := 0, 0
	for i, o := range obs {
		fulls[i], kept[i] = o.full, o.stateKept
		sumFull += float64(o.full)
		sumTok += float64(o.tokens)
		sumKept += o.stateKept
		if o.full > maxLen {
			over++
		}
		if !o.scorable {
			c.Unscorable++
		}
		if o.sl != nil {
			slN++
			sl.MetricsBefore += float64(o.sl.MetricsBefore)
			sl.MetricsAfter += float64(o.sl.MetricsAfter)
			sl.ScopeBefore += float64(o.sl.ScopeBefore)
			sl.ScopeAfter += float64(o.sl.ScopeAfter)
			if o.sl.OptionsFromCatalog {
				sl.CatalogOptionItems++
				sl.OptionsBefore += float64(o.sl.OptionsBefore)
				sl.OptionsAfter += float64(o.sl.OptionsAfter)
				if o.sl.GoldKept {
					hit++
				}
			}
		}
	}
	if len(obs) == 0 {
		return c
	}
	slices.Sort(fulls)
	slices.Sort(kept)
	n := float64(len(obs))
	c.FullTokensP50, c.FullTokensP95 = fulls[percentileIndex(len(fulls), 0.5)], fulls[percentileIndex(len(fulls), 0.95)]
	c.FullTokensMean, c.TokensMean = round4(sumFull/n), round4(sumTok/n)
	c.OverMaxLen = round4(float64(over) / n)
	c.StateKeptP50, c.StateKeptMean = round4(kept[percentileIndex(len(kept), 0.5)]), round4(sumKept/n)
	if slN > 0 {
		m := float64(slN)
		sl.MetricsBefore, sl.MetricsAfter = round4(sl.MetricsBefore/m), round4(sl.MetricsAfter/m)
		sl.ScopeBefore, sl.ScopeAfter = round4(sl.ScopeBefore/m), round4(sl.ScopeAfter/m)
		if k := float64(sl.CatalogOptionItems); k > 0 {
			sl.OptionsBefore, sl.OptionsAfter = round4(sl.OptionsBefore/k), round4(sl.OptionsAfter/k)
			rec := round4(float64(hit) / k)
			sl.Recall = &rec
		}
		c.Shortlist = &sl
	}
	return c
}

func round4(v float64) float64 { return math.Round(v*1e4) / 1e4 }
