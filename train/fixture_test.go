package train

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/UnicoLab/ul_layago/laya"
)

// The fixture is a small template generator for hive.laya-dataset/v1, in the
// shape of the Python trainer's (tools/laya-train/laya_train/fixture.py):
// labels are drawn from the structured spec first and the text is rendered
// for them. It covers instance Q (every task Hive's routermodel serves but
// q_scope_choice), instance R (graded evidence relevance, and answer ranking
// with label_dist soft labels on train) and a few A lines, in five
// languages: EN/FR/DE are fitted on, Japanese and Arabic are held out as
// languages, energy is held out as a domain, and locked_test uses stories of
// its own. Only fixtureSample is committed.
const (
	fixturePath      = "testdata/fixture_sample.jsonl"
	fixtureGenerator = "layago-train-fixture/2"
)

var (
	fixtureFitLangs  = []string{"en", "fr", "de"}
	fixtureHeldLangs = []string{"ja", "ar"}
	fixtureAllLangs  = []string{"en", "fr", "de", "ja", "ar"}
)

type fixtureMetric struct {
	key, unit string
	labels    []string // en, fr, de, ja, ar
}

type fixtureDomain struct {
	metrics []fixtureMetric
	dim     string
	values  []string
}

func fm(key, unit string, labels ...string) fixtureMetric { return fixtureMetric{key, unit, labels} }

var fixtureDomains = map[string]fixtureDomain{
	"retail": {metrics: []fixtureMetric{
		fm("retail.revenue", "EUR", "Revenue", "Chiffre d'affaires", "Umsatz", "売上", "الإيرادات"),
		fm("retail.orders", "count", "Orders", "Commandes", "Bestellungen", "注文数", "الطلبات"), //nolint:misspell // French label
		fm("retail.basket", "EUR", "Average basket", "Panier moyen", "Warenkorb", "平均客単価", "متوسط السلة"),
		fm("retail.returns", "ratio", "Return rate", "Taux de retour", "Retourenquote", "返品率", "معدل الإرجاع"),
	}, dim: "store", values: []string{"Paris", "Lyon", "Berlin", "Madrid"}},
	"saas": {metrics: []fixtureMetric{
		fm("saas.mrr", "EUR", "MRR", "Revenu récurrent", "Wiederkehrender Umsatz", "月次経常収益", "الإيرادات المتكررة"),
		fm("saas.churn", "ratio", "Churn rate", "Taux d'attrition", "Abwanderungsrate", "解約率", "معدل التسرب"),
		fm("saas.signups", "count", "Signups", "Inscriptions", "Anmeldungen", "新規登録", "التسجيلات"),
		fm("saas.active", "count", "Active users", "Utilisateurs actifs", "Aktive Nutzer", "アクティブユーザー", "المستخدمون النشطون"),
	}, dim: "region", values: []string{"EMEA", "NA", "APAC", "LATAM"}},
	"logistics": {metrics: []fixtureMetric{
		fm("logistics.shipments", "count", "Shipments", "Expéditions", "Sendungen", "出荷数", "الشحنات"),
		fm("logistics.ontime", "ratio", "On-time rate", "Taux de ponctualité", "Pünktlichkeit", "定時率", "معدل الالتزام بالموعد"),
		fm("logistics.cost", "EUR", "Cost per parcel", "Coût par colis", "Kosten pro Paket", "小包あたりの費用", "تكلفة الطرد"),
		fm("logistics.delay", "hours", "Average delay", "Retard moyen", "Verspätung", "平均遅延", "متوسط التأخير"),
	}, dim: "hub", values: []string{"Lille", "Hamburg", "Valencia", "Torino"}},
	"energy": {metrics: []fixtureMetric{
		fm("energy.consumption", "kWh", "Consumption", "Consommation", "Verbrauch", "消費量", "الاستهلاك"),
		fm("energy.peak", "MW", "Peak load", "Pointe de charge", "Spitzenlast", "ピーク負荷", "ذروة الحمل"),
		fm("energy.outages", "count", "Outages", "Pannes", "Ausfälle", "停電", "الانقطاعات"),
		fm("energy.tariff", "EUR/kWh", "Average tariff", "Tarif moyen", "Durchschnittstarif", "平均料金", "متوسط التعرفة"),
	}, dim: "site", values: []string{"North", "South", "East"}},
}

var fixtureOperations = []Option{
	{Key: "lookup", Text: "read one recorded value"},
	{Key: "change", Text: "how a value moved between two periods"},
	{Key: "compare", Text: "compare two segments"},
	{Key: "rank", Text: "rank the segments"},
}

var fixtureTemplates = map[string]map[string]string{
	"en": {"lookup": "What was {m} in {s} {t}?", "change": "How did {m} change in {s} {t}?",
		"compare": "Compare {m} between {s} and {s2} {t}.", "rank": "Which {d} had the highest {m} {t}?"},
	"fr": {"lookup": "Quel était {m} à {s} {t} ?", "change": "Comment a évolué {m} à {s} {t} ?",
		"compare": "Compare {m} entre {s} et {s2} {t}.", "rank": "Quel {d} a eu le plus haut {m} {t} ?"},
	"de": {"lookup": "Wie hoch war {m} in {s} {t}?", "change": "Wie hat sich {m} in {s} {t} verändert?",
		"compare": "Vergleiche {m} zwischen {s} und {s2} {t}.", "rank": "Welcher {d} hatte {t} den höchsten Wert für {m}?"},
	"ja": {"lookup": "{t}の{s}の{m}はいくらでしたか？", "change": "{t}、{s}の{m}はどう変化しましたか？",
		"compare": "{t}の{m}を{s}と{s2}で比較してください。", "rank": "{t}に{m}が最も高かった{d}はどこですか？"},
	"ar": {"lookup": "كم كان {m} في {s} {t}؟", "change": "كيف تغير {m} في {s} {t}؟",
		"compare": "قارن {m} بين {s} و{s2} {t}.", "rank": "أي {d} كان لديه أعلى {m} {t}؟"},
}

var fixtureEvidence = map[string]string{
	"en": "{m} in {s} moved by {n}% {t}.",
	"fr": "{m} à {s} a varié de {n} % {t}.",
	"de": "{m} in {s} hat sich {t} um {n} % verändert.",
	"ja": "{t}、{s}の{m}は{n}%変動しました。",
	"ar": "تغير {m} في {s} بنسبة {n}% {t}.",
}

var fixtureTimes = map[string][2][]string{
	"en": {{"last week", "last month", "yesterday", "this quarter"}, {"since the last full moon", "during my next holiday", "on a lucky day"}},
	"fr": {{"la semaine dernière", "le mois dernier", "hier", "ce trimestre"}, {"depuis la dernière pleine lune", "pendant mes prochaines vacances", "un jour de chance"}},
	"de": {{"letzte Woche", "letzten Monat", "gestern", "dieses Quartal"}, {"seit dem letzten Vollmond", "in meinem nächsten Urlaub", "an einem Glückstag"}},
	"ja": {{"先週", "先月", "昨日", "今四半期"}, {"前回の満月以来", "次の休暇中", "幸運な日に"}},
	"ar": {{"الأسبوع الماضي", "الشهر الماضي", "أمس", "هذا الربع"}, {"منذ آخر بدر", "خلال عطلتي القادمة", "في يوم محظوظ"}},
}

var fixtureDimWords = map[string]map[string]string{
	"store":  {"en": "store", "fr": "magasin", "de": "Laden", "ja": "店舗", "ar": "متجر"},
	"region": {"en": "region", "fr": "région", "de": "Region", "ja": "地域", "ar": "منطقة"},
	"hub":    {"en": "hub", "fr": "hub", "de": "Hub", "ja": "拠点", "ar": "مركز"},
	"site":   {"en": "site", "fr": "site", "de": "Standort", "ja": "拠点", "ar": "موقع"},
}

var fixtureSingleSeries = []string{"the question asks about several metric series", "the question asks about exactly one metric series"}

func (m fixtureMetric) label(lang string) string {
	return m.labels[slices.Index(fixtureAllLangs, lang)]
}

// catalogEntry is one catalog metric: its labels in English and the
// question's language, unit, dimension and the first three scope values.
func (m fixtureMetric) catalogEntry(lang string, d fixtureDomain) map[string]any {
	labels := map[string]any{"en": m.label("en"), lang: m.label(lang)}
	vals := make([]any, 0, 3)
	for _, v := range d.values[:3] {
		vals = append(vals, v)
	}
	return map[string]any{"key": m.key, "labels": labels, "unit": m.unit, "dimensions": []any{d.dim},
		"scope_values": map[string]any{d.dim: vals}}
}

func fill(tmpl string, kv ...string) string {
	return strings.NewReplacer(kv...).Replace(tmpl)
}

func fixtureSeed(parts ...string) int64 {
	sum := sha256.Sum256([]byte(strings.Join(parts, "\x00")))
	return int64(sum[0])<<24 | int64(sum[1])<<16 | int64(sum[2])<<8 | int64(sum[3])
}

// storyExamples writes one story's examples of the given instances.
func storyExamples(domain, lang, story, split string, r *rand.Rand) []map[string]any {
	d := fixtureDomains[domain]
	var out []map[string]any
	base := func(instance, task, taskType string, input map[string]any) map[string]any {
		return map[string]any{
			"id": fmt.Sprintf("fx-%s-%s", story, task), "split": split, "lang": lang, "instance": instance, "task": task,
			"task_type": taskType, "input": input, "label_source": "template", "domain_id": domain, "story_id": story,
			"generator_version": fixtureGenerator, "seed": fixtureSeed(story, task),
		}
	}
	pick := func(list []string) string { return list[r.IntN(len(list))] }
	metricIdx := r.IntN(len(d.metrics))
	metric := d.metrics[metricIdx]
	others := slices.Delete(slices.Clone(d.metrics), metricIdx, metricIdx+1)
	vals := d.values[:3]
	si := r.Perm(len(vals))
	s, s2 := vals[si[0]], vals[si[1]]
	supported := fixtureTimes[lang][0]
	question := func(op string, m fixtureMetric, t string) string {
		return fill(fixtureTemplates[lang][op], "{m}", m.label(lang), "{s}", s, "{s2}", s2, "{t}", t, "{d}", fixtureDimWords[d.dim][lang])
	}
	catalog := func(ms ...fixtureMetric) []any {
		out := make([]any, len(ms))
		for i, m := range ms {
			out[i] = m.catalogEntry(lang, d)
		}
		return out
	}
	other := others[r.IntN(len(others))]

	// q_operation: the operation first, then the text for it.
	op := r.IntN(len(fixtureOperations))
	ex := base(InstanceQ, "q_operation", "choice", map[string]any{"question": question(fixtureOperations[op].Key, metric, pick(supported)),
		"catalog": catalog(metric, other)})
	ops := make([]any, len(fixtureOperations))
	for i, o := range fixtureOperations {
		ops[i] = map[string]any{"key": o.Key, "text": o.Text}
	}
	ex["options"], ex["label"] = ops, op
	out = append(out, ex)

	// q_metric_choice: the gold metric and 2 or 3 distractors, shuffled;
	// the options are the catalog, as Hive offers them.
	cands := append([]fixtureMetric{metric}, others[:2+r.IntN(2)]...)
	r.Shuffle(len(cands), func(i, j int) { cands[i], cands[j] = cands[j], cands[i] })
	ex = base(InstanceQ, "q_metric_choice", "choice", map[string]any{"question": question("lookup", metric, pick(supported)),
		"catalog": catalog(cands...)})
	var mopts []any
	for i, c := range cands {
		mopts = append(mopts, map[string]any{"key": c.key, "text": c.label(lang)})
		if c.key == metric.key {
			ex["label"] = i
		}
	}
	ex["options"] = mopts
	out = append(out, ex)

	// q_time_supported: supported or not decided first (Laya's default
	// yes/no criteria).
	ok := r.IntN(2) == 0
	t := pick(fixtureTimes[lang][1])
	if ok {
		t = pick(supported)
	}
	ex = base(InstanceQ, "q_time_supported", "yes_no", map[string]any{"question": question("lookup", metric, t), "catalog": catalog(metric)})
	ex["label"] = ok
	out = append(out, ex)

	// q_single_series: lookup and change read one series, compare and rank
	// several (yes_no with [false_text, true_text] options).
	op = r.IntN(len(fixtureOperations))
	ex = base(InstanceQ, "q_single_series", "yes_no", map[string]any{"question": question(fixtureOperations[op].Key, metric, pick(supported)),
		"catalog": catalog(metric, other)})
	ex["options"] = []any{fixtureSingleSeries[0], fixtureSingleSeries[1]}
	ex["label"] = op <= 1
	out = append(out, ex)

	// q_act_or_escalate: act when the window is supported and the metric is
	// in the catalog.
	mode := r.IntN(3)
	t, cat := pick(supported), catalog(metric, other)
	switch mode {
	case 1:
		t = pick(fixtureTimes[lang][1])
	case 2:
		cat = catalog(others...)
	}
	ex = base(InstanceQ, "q_act_or_escalate", "yes_no", map[string]any{"question": question("lookup", metric, t), "catalog": cat})
	ex["label"] = mode == 0
	out = append(out, ex)

	// R: evidence about the question's metric is directly relevant (2),
	// about another metric of the domain related (1), about another domain
	// irrelevant (0).
	far := fixtureDomains[fixtureOtherDomain(domain)].metrics[r.IntN(4)]
	evidence := func(m fixtureMetric) string {
		return fill(fixtureEvidence[lang], "{m}", m.label(lang), "{s}", s, "{t}", pick(supported), "{n}", fmt.Sprint(2+r.IntN(30)))
	}
	grades := []int{2, 1, 0}
	items := []string{evidence(metric), evidence(other), evidence(far)}
	g := r.IntN(3)
	q := question("change", metric, pick(supported))
	ex = base(InstanceR, "r_evidence_relevance", "score", map[string]any{"question": q,
		"evidence": []any{map[string]any{"id": "ev1", "text": items[g]}}})
	ex["options"], ex["label"] = []any{"irrelevant", "related", "directly relevant"}, grades[g]
	out = append(out, ex)

	// r_answer_rank: the same three items to rank; the label is the directly
	// relevant one, and on train label_dist spreads the grades (2, 1, 0) as
	// a listwise target.
	order := r.Perm(3)
	var evs, eopts []any
	dist := make([]any, 3)
	best := 0
	for i, j := range order {
		id := fmt.Sprintf("ev%d", i+1)
		evs = append(evs, map[string]any{"id": id, "text": items[j]})
		eopts = append(eopts, map[string]any{"key": id, "text": items[j]})
		dist[i] = json.Number([]string{"0.6666666666666666", "0.3333333333333333", "0.0"}[j])
		if j == 0 {
			best = i
		}
	}
	ex = base(InstanceR, "r_answer_rank", "choice", map[string]any{"question": q, "evidence": evs})
	ex["options"], ex["label"] = eopts, best
	if split == SplitTrain {
		ex["label_dist"] = dist
	}
	return append(out, ex)
}

func fixtureOtherDomain(domain string) string {
	switch domain {
	case "retail":
		return "logistics"
	case "saas":
		return "retail"
	default:
		return "saas"
	}
}

// fixtureAExamples are A lines for the loader (one with LLM provenance).
func fixtureAExamples() []map[string]any {
	sum := sha256.Sum256([]byte("fixture-call"))
	common := func(id, split, lang, story string, label bool) map[string]any {
		return map[string]any{"id": id, "split": split, "lang": lang, "instance": InstanceA, "task": "a_causal_licensed",
			"task_type": "yes_no", "label": label, "label_source": "template", "domain_id": "retail", "story_id": story,
			"generator_version": fixtureGenerator, "seed": 1}
	}
	a := common("fx-a-1", "train", "en", "a-story-1", false)
	a["input"] = map[string]any{"question": "Why did revenue fall?", "draft": "Revenue fell 4% last week.",
		"sentence": "The store closure caused the fall."}
	b := common("fx-a-2", "dev", "fr", "a-story-2", true)
	b["input"] = map[string]any{"question": "Pourquoi le chiffre d'affaires a-t-il baissé ?",
		"draft":    "La fermeture du magasin explique 3 points de la baisse, selon l'analyse d'attribution.",
		"sentence": "La fermeture du magasin a causé une partie de la baisse."}
	b["label_source"] = "llm_paraphrase"
	b["provenance"] = map[string]any{"llm_interaction_ids": []any{hex.EncodeToString(sum[:])}}
	return []map[string]any{a, b}
}

// fixtureLines generates the full fixture: per fitted domain and language,
// storiesPerCell stories (two thirds train, the rest dev) and one
// locked_test story; per fitted domain one held-out-language story in each
// of Japanese and Arabic; per fitted language one energy story as the
// held-out domain. Evaluated examples whose content also occurs in train are
// dropped (the template space is small), so no split tests what train
// memorised.
func fixtureLines(storiesPerCell int) []map[string]any {
	r := rand.New(rand.NewPCG(20260928, 2)) //nolint:gosec // G404: deterministic fixture, not a security use
	var out []map[string]any
	for _, domain := range []string{"retail", "saas", "logistics"} {
		for _, lang := range fixtureFitLangs {
			for i := 0; i < storiesPerCell; i++ {
				split := SplitTrain
				if i >= storiesPerCell*2/3 {
					split = SplitDev
				}
				out = append(out, storyExamples(domain, lang, fmt.Sprintf("%s-%s-%d", domain, lang, i), split, r)...)
			}
			out = append(out, storyExamples(domain, lang, fmt.Sprintf("%s-%s-lt", domain, lang), SplitLockedTest, r)...)
		}
		for _, lang := range fixtureHeldLangs {
			out = append(out, storyExamples(domain, lang, fmt.Sprintf("%s-%s-h", domain, lang), SplitHeldoutLanguage, r)...)
		}
	}
	for _, lang := range fixtureFitLangs {
		out = append(out, storyExamples("energy", lang, "energy-"+lang+"-h", SplitHeldoutDomain, r)...)
	}
	out = append(out, fixtureAExamples()...)
	content := func(ex map[string]any) string {
		raw, _ := json.Marshal([]any{ex["task"], ex["input"], ex["options"]})
		return string(raw)
	}
	train := map[string]bool{}
	for _, ex := range out {
		if ex["split"] == SplitTrain {
			train[content(ex)] = true
		}
	}
	kept := out[:0]
	for _, ex := range out {
		if ex["split"] == SplitTrain || !train[content(ex)] {
			kept = append(kept, ex)
		}
	}
	return kept
}

func encodeLines(t testing.TB, lines []map[string]any) []byte {
	t.Helper()
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	for _, line := range lines {
		if err := enc.Encode(line); err != nil {
			t.Fatal(err)
		}
	}
	return buf.Bytes()
}

// fixtureStories is the stories per (domain, language) cell of the tests'
// fixture.
const fixtureStories = 9

// fixtureBytes is the full generated fixture. Tests build it in memory; only
// fixtureSample is committed.
func fixtureBytes(t testing.TB) []byte {
	t.Helper()
	return encodeLines(t, fixtureLines(fixtureStories))
}

// fixtureSample is the committed contract example: the first line of every
// (instance, task, split) triple.
func fixtureSample() []map[string]any {
	seen := map[string]bool{}
	var out []map[string]any
	for _, l := range fixtureLines(fixtureStories) {
		key := fmt.Sprint(l["instance"], l["task"], l["split"])
		if !seen[key] {
			seen[key] = true
			out = append(out, l)
		}
	}
	return out
}

// TestFixtureSampleIsCurrent pins the committed sample to its generator;
// LAYAGO_TRAIN_WRITE_FIXTURE=1 rewrites it, and LAYAGO_TRAIN_FULL_FIXTURE_OUT
// names a file (outside the repository) to receive the full fixture, for
// running laya-train or the Python trainer on it.
func TestFixtureSampleIsCurrent(t *testing.T) {
	if out := os.Getenv("LAYAGO_TRAIN_FULL_FIXTURE_OUT"); out != "" {
		if err := os.WriteFile(out, fixtureBytes(t), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	want := encodeLines(t, fixtureSample())
	if os.Getenv("LAYAGO_TRAIN_WRITE_FIXTURE") == "1" {
		if err := os.WriteFile(fixturePath, want, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	got, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("%s is stale; run LAYAGO_TRAIN_WRITE_FIXTURE=1 go test -run TestFixtureSampleIsCurrent ./internal/layago/train", fixturePath)
	}
	sample, err := ParseDataset([]DataFile{{Name: fixturePath, Data: got}}, "")
	if err != nil {
		t.Fatal(err)
	}
	full, err := ParseDataset([]DataFile{{Name: "fixture.jsonl", Data: fixtureBytes(t)}}, "")
	if err != nil {
		t.Fatal(err)
	}
	if len(full.SplitViolations) != 0 {
		t.Fatalf("the fixture breaks split invariants: %v", full.SplitViolations)
	}
	// LAYAGO_TRAIN_STATES_OUT receives {"id", "state"} per example, for
	// comparing laya.RenderPromptState with the Python trainer's render_state
	// (testdata/prompt_oracle.py).
	if out := os.Getenv("LAYAGO_TRAIN_STATES_OUT"); out != "" {
		var lines []map[string]any
		for _, ex := range full.Examples {
			s, err := laya.RenderPromptState(ex.Instance, ex.Input)
			if err != nil {
				t.Fatal(err)
			}
			lines = append(lines, map[string]any{"id": ex.ID, "state": s, "digest": full.Digest()})
		}
		if err := os.WriteFile(out, encodeLines(t, lines), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	counts := map[string]int{}
	for _, ex := range full.Examples {
		counts[ex.Instance+"/"+ex.Split]++
	}
	t.Logf("fixture: %d examples %v; committed sample %d lines", full.Total, counts, sample.Total)
	for _, want := range []string{"Q/train", "Q/dev", "Q/locked_test", "Q/heldout_language", "Q/heldout_domain", "R/train", "R/dev", "A/train"} {
		if counts[want] == 0 {
			t.Fatalf("the fixture has no %s examples: %v", want, counts)
		}
	}
}
