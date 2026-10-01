package train

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"github.com/UnicoLab/ul_layago/laya"
)

// shortlistExample parses one Q example line.
func shortlistExample(t *testing.T, task string, question string, options []any, label int) *Example {
	t.Helper()
	catalog := []any{
		map[string]any{"key": "ticket_backlog", "labels": map[string]any{"en": "Ticket backlog"}, "unit": "count",
			"dimensions": []any{"squad"}, "scope_values": map[string]any{"squad": []any{"Team Grupei", "Team Pebres"}}},
		map[string]any{"key": "occ_agents", "labels": map[string]any{"en": "Agent occupancy", "fr": "Occupation des agents"},
			"unit": "%", "dimensions": []any{"area"}, "scope_values": map[string]any{"area": []any{"Île-de-France", "Bretagne", "Est"}}},
		map[string]any{"key": "new_tickets", "labels": map[string]any{"en": "Tickets created"}, "unit": "count",
			"dimensions": []any{"area", "file"}, "scope_values": map[string]any{"area": []any{"Île-de-France", "Est"}, "file": []any{"Billing"}}},
		map[string]any{"key": "csat_score", "labels": map[string]any{"en": "Customer satisfaction score"}, "unit": "score",
			"dimensions": []any{"channel"}, "scope_values": map[string]any{"channel": []any{"Web", "Store"}}},
	}
	line := map[string]any{"id": "s1-q01-" + task, "split": "dev", "lang": "fr", "instance": "Q", "task": task,
		"input": map[string]any{"question": question, "catalog": catalog}, "label_source": "template", "domain_id": "support",
		"story_id": "s1", "generator_version": "test", "seed": 1}
	if options == nil {
		line["task_type"], line["label"] = "yes_no", label == 1
	} else {
		line["task_type"], line["options"], line["label"] = "choice", options, label
	}
	raw, err := json.Marshal(line)
	if err != nil {
		t.Fatal(err)
	}
	ds, err := ParseDataset([]DataFile{{Name: "x.jsonl", Data: raw}}, InstanceQ)
	if err != nil {
		t.Fatal(err)
	}
	return &ds.Examples[0]
}

func metricOptions() []any {
	return []any{
		map[string]any{"key": "ticket_backlog", "text": "Ticket backlog"}, map[string]any{"key": "occ_agents", "text": "Agent occupancy"},
		map[string]any{"key": "new_tickets", "text": "Tickets created"}, map[string]any{"key": "csat_score", "text": "Customer satisfaction score"},
	}
}

// TestCatalogShortlistKeepsTheMostSimilar: the k metrics and scope values
// most similar to the question stay, most relevant first; accents and case
// fold; options keep their order and the label follows its option.
func TestCatalogShortlistKeepsTheMostSimilar(t *testing.T) {
	q := "Tickets created et customer satisfaction score en ILE-DE-FRANCE sur le canal web ?"
	ex := shortlistExample(t, "q_metric_choice", q, metricOptions(), 2)
	short, res := CatalogShortlist(ex, 2)
	if !res.GoldKept || !res.OptionsFromCatalog || res.MetricsBefore != 4 || res.MetricsAfter != 2 || res.ScopeAfter != 2 {
		t.Fatalf("result %+v", res)
	}
	keys := make([]string, len(short.Options))
	for i, o := range short.Options {
		keys[i] = o.Key
	}
	if !slices.Equal(keys, []string{"new_tickets", "csat_score"}) || short.Label != 0 || short.Options[short.Label].Key != "new_tickets" {
		t.Fatalf("options %v, label %d", keys, short.Label)
	}
	state, err := laya.RenderPromptState(InstanceQ, short.Input)
	if err != nil {
		t.Fatal(err)
	}
	want := "question: " + q + "\ncatalog:\n" +
		"- new_tickets: Tickets created; unit: count; dimensions: area, file; scope: area=Île-de-France\n" +
		"- csat_score: Customer satisfaction score; unit: score; dimensions: channel; scope: channel=Web"
	if state != want {
		t.Fatalf("state\n%s\nwant\n%s", state, want)
	}
	if orig, _ := laya.RenderPromptState(InstanceQ, ex.Input); !strings.Contains(orig, "ticket_backlog") {
		t.Fatalf("the source example was modified: %s", orig)
	}
	// A k at least the catalog's size keeps every metric, value and option.
	whole, res := CatalogShortlist(ex, 16)
	if res.MetricsAfter != 4 || res.ScopeAfter != res.ScopeBefore || len(whole.Options) != 4 || whole.Label != 2 {
		t.Fatalf("k 16: %+v, label %d", res, whole.Label)
	}
}

// TestCatalogShortlistNeverInjectsGold: when the shortlist misses the
// gold option it stays out; the label is -1 and layOut records the gold
// key, so the item is always wrong and never trained on.
func TestCatalogShortlistNeverInjectsGold(t *testing.T) {
	ex := shortlistExample(t, "q_metric_choice", "Combien de tickets created en Bretagne ?", metricOptions(), 3)
	short, res := CatalogShortlist(ex, 1)
	if res.GoldKept || short.Label != -1 || len(short.Options) != 1 || short.Options[0].Key != "new_tickets" || short.HasGold() {
		t.Fatalf("result %+v, options %+v, label %d", res, short.Options, short.Label)
	}
	trunk := tinyTrunk(t, tinySpec)
	spec := &TaskSpec{Name: ex.Task, Type: ex.Type, Instructions: DefaultInstruction(ex.Task)}
	it, err := layOut(ex, spec, trunk, &ShortlistSpec{K: 1, Method: ShortlistMethod})
	if err != nil {
		t.Fatal(err)
	}
	if it.Ex.GoldKey() != "csat_score" || it.Ex.HasGold() || slices.ContainsFunc(it.Question.Options, func(o laya.Option) bool { return o.Key == "csat_score" }) {
		t.Fatalf("item gold %q, options %+v", it.Ex.GoldKey(), it.Question.Options)
	}
	if tgt := it.Ex.Target(); slices.Max(tgt) != 0 {
		t.Fatalf("target %v", tgt)
	}
	// Serving and latency start again from the source example; laying the
	// shortlisted copy out again keeps its missing gold (it once panicked).
	if it.source() != ex {
		t.Fatal("the item lost its source example")
	}
	again, err := layOut(it.Ex, spec, trunk, &ShortlistSpec{K: 1, Method: ShortlistMethod})
	if err != nil || again.Ex.GoldKey() != "csat_score" || !slices.Equal(again.IDs, it.IDs) {
		t.Fatalf("re-laid out %v: gold %q", err, again.Ex.GoldKey())
	}
	r := decodeRow(it, []float32{1}, []float32{0, 0}, Temperatures{}, nil)
	if !r.Scorable || r.Correct || r.GoldKey != "csat_score" || r.PredKey != "new_tickets" || r.ProbLabel != 0 {
		t.Fatalf("row %+v", r)
	}
	// Brier: probability 1 on a wrong option and 0 on the unoffered gold.
	if s := summarize([]Row{r}); *s.Accuracy != 0 || *s.Brier != 2 {
		t.Fatalf("summary %+v", s)
	}
}

// TestCatalogShortlistFiltersScopeOptions: q_scope_choice options are
// "dimension=value" and keep only the shortlisted pairs.
func TestCatalogShortlistFiltersScopeOptions(t *testing.T) {
	opts := []any{"area=Bretagne", "area=Est", "area=Île-de-France", "channel=Store", "channel=Web", "file=Billing",
		"squad=Team Grupei", "squad=Team Pebres"}
	ex := shortlistExample(t, "q_scope_choice", "Backlog de la Team-Pebres ?", opts, 7)
	short, res := CatalogShortlist(ex, 3)
	if !res.GoldKept || res.OptionsBefore != 8 || res.OptionsAfter != 3 || short.Options[short.Label].Key != "squad=Team Pebres" {
		t.Fatalf("result %+v, options %+v, label %d", res, short.Options, short.Label)
	}
	yes := shortlistExample(t, "q_act_or_escalate", "Backlog de la Team-Pebres ?", nil, 1)
	sy, ry := CatalogShortlist(yes, 3)
	if ry.OptionsFromCatalog || !ry.GoldKept || sy.Label != 1 || len(sy.Options) != 0 {
		t.Fatalf("yes_no %+v, label %d", ry, sy.Label)
	}
}

// TestPrepareEvalLangsAndShortlist: evaluated splits keep only the named
// languages while train stays whole, and the shortlist is recorded.
func TestPrepareEvalLangsAndShortlist(t *testing.T) {
	trunk := tinyTrunk(t, tinySpec)
	ds := fixtureDataset(t, InstanceQ)
	full, err := Prepare(ds, trunk, PrepareOptions{})
	if err != nil {
		t.Fatal(err)
	}
	// The fixture's template space is small: a two-entry catalog can render
	// a dev state exactly as a train one, which Prepare refuses unless
	// allowed.
	p, err := Prepare(ds, trunk, PrepareOptions{EvalLangs: []string{"fr", "en"}, Shortlist: ShortlistSpec{K: 2}, AllowDuplicates: true})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(p.EvalLangs, []string{"en", "fr"}) || p.Shortlist == nil || p.Shortlist.Method != ShortlistMethod {
		t.Fatalf("recorded %v %+v", p.EvalLangs, p.Shortlist)
	}
	count := func(q *Prepared) (train, eval int) {
		for _, it := range q.Items {
			if it.Role == SplitTrain {
				train++
				continue
			}
			if l := primaryLanguage(it.Ex.Lang); q.EvalLangs != nil && l != "en" && l != "fr" {
				t.Fatalf("%s kept in %s", it.Ex.ID, it.Role)
			}
			eval++
		}
		return
	}
	ft, fe := count(full)
	pt, pe := count(p)
	if ft != pt || pe >= fe || pe == 0 {
		t.Fatalf("train %d -> %d, evaluated %d -> %d", ft, pt, fe, pe)
	}
	if _, err := Prepare(fixtureDataset(t, InstanceR), trunk, PrepareOptions{Shortlist: ShortlistSpec{K: 2}}); err == nil {
		t.Fatal("a catalog shortlist on instance R was accepted")
	}
}

// TestLTTStartRules: the contract start and the E-error start.
func TestLTTStartRules(t *testing.T) {
	for _, c := range []struct {
		errors, want int
	}{{0, 176}, {1, 93}, {2, 124}} {
		if got := LTTStart(0.05, 0.05, 20, c.errors); got != c.want {
			t.Errorf("start with %d errors: %d, want %d", c.errors, got, c.want)
		}
	}
	if got := LTTStart(0.05, 0.05, 150, 1); got != 150 {
		t.Errorf("n_min 150 floors the start: %d", got)
	}
	// Scores 1..0 over 120 items with one error at the top: the contract
	// start (176) tests nothing; the one-error start (93) certifies.
	scores := make([]float64, 120)
	correct := make([]bool, 120)
	for i := range scores {
		scores[i], correct[i] = 1-float64(i)/200, i != 0
	}
	if th := LearnThenTestFrom(scores, correct, 0.05, 0.05, 20, LTTStart(0.05, 0.05, 20, 0)); th.Certified || th.Tested != 0 {
		t.Fatalf("contract start %+v", th)
	}
	th := LearnThenTestFrom(scores, correct, 0.05, 0.05, 20, LTTStart(0.05, 0.05, 20, 1))
	if !th.Certified || th.NStart != 93 || th.Acted < 93 || th.Errors != 1 {
		t.Fatalf("one-error start %+v", th)
	}
}

// TestMetaSlicesAreReadNotRequired: meta.slices is read for reporting;
// a missing or malformed meta is ignored.
func TestMetaSlicesAreReadNotRequired(t *testing.T) {
	base := `{"id":"x","split":"dev","lang":"en","instance":"Q","task":"q_time_supported","task_type":"yes_no","label":true,` +
		`"label_source":"template","input":{"question":"q?"},"domain_id":"d","story_id":"s","generator_version":"g","seed":1`
	for _, c := range []struct {
		meta string
		want []string
	}{
		{`,"meta":{"slices":["unseen_template","spelling_variant"],"train_similarity":0.3}}`, []string{"unseen_template", "spelling_variant"}},
		{`,"meta":{"slices":"unseen_template"}}`, nil},
		{`,"meta":null}`, nil},
		{`}`, nil},
	} {
		ds, err := ParseDataset([]DataFile{{Name: "m.jsonl", Data: []byte(base + c.meta)}}, "")
		if err != nil {
			t.Fatal(err)
		}
		if got := ds.Examples[0].Slices; !slices.Equal(got, c.want) {
			t.Errorf("meta %s: slices %v, want %v", c.meta, got, c.want)
		}
	}
}

// TestYesNoWeights: each answer of a yes_no task weighs n / (2 n_c).
func TestYesNoWeights(t *testing.T) {
	mk := func(task string, tt TaskType, label int) *sample {
		return &sample{it: &Item{Ex: &Example{Task: task, Type: tt, Label: label}}}
	}
	train := []*sample{mk("q_x", TaskYesNo, 1), mk("q_x", TaskYesNo, 1), mk("q_x", TaskYesNo, 1), mk("q_x", TaskYesNo, 0),
		mk("q_c", TaskChoice, 2)}
	w := yesNoWeights(train)
	if len(w) != 1 || w["q_x"] != [2]float64{2, 4.0 / 6} {
		t.Fatalf("weights %v", w)
	}
}

// TestFitLangsRestrictEveryDevFit: with FitLangs, early stopping, the act
// head, the temperatures and the thresholds read only dev items of those
// languages, while every item is still evaluated.
func TestFitLangsRestrictEveryDevFit(t *testing.T) {
	ctx := context.Background()
	trunk := tinyTrunk(t, tinySpec)
	p, err := Prepare(fixtureDataset(t, InstanceQ), trunk, PrepareOptions{})
	if err != nil {
		t.Fatal(err)
	}
	cache, err := BuildCache(ctx, trunk, p, CacheSpec{Root: t.TempDir(), ShardSize: 100})
	if err != nil {
		t.Fatal(err)
	}
	recs, err := cache.Records()
	if err != nil {
		t.Fatal(err)
	}
	cfg := loopConfig(0)
	cfg.Epochs, cfg.Int8, cfg.MinCell, cfg.FitLangs, cfg.YesNoBalance = 3, false, 1, []string{"en"}, true
	res, err := Train(ctx, p, recs, trunk.Identity.HiddenSize, cfg)
	if err != nil {
		t.Fatal(err)
	}
	cal, conf := 0, map[string]int{}
	evaluated := 0
	for _, it := range p.Items {
		if it.Role == RoleDevConf || it.Role == RoleDevCal {
			evaluated++
		}
		if primaryLanguage(it.Ex.Lang) != "en" || !it.Scorable {
			continue
		}
		switch it.Role {
		case RoleDevCal:
			cal++
		case RoleDevConf:
			conf[it.Ex.Task]++
		}
	}
	if cal == 0 || res.Act.N != cal || res.Float32.Temperatures["*"].N != cal || res.DevHalves.DevCal.Scorable != cal {
		t.Fatalf("dev_cal en %d: act head %d, temperature %d, dev halves %d", cal, res.Act.N, res.Float32.Temperatures["*"].N,
			res.DevHalves.DevCal.Scorable)
	}
	for task, th := range res.Float32.Thresholds {
		if th.N != conf[task] {
			t.Fatalf("%s threshold on %d items, dev_conf en has %d", task, th.N, conf[task])
		}
	}
	rows := 0
	for _, r := range res.Float32.Rows {
		if r.Split == RoleDevConf || r.Split == RoleDevCal {
			rows++
		}
	}
	if rows != evaluated || !slices.Equal(res.DevHalves.FitLangs, []string{"en"}) {
		t.Fatalf("%d dev rows of %d, fit langs %v", rows, evaluated, res.DevHalves.FitLangs)
	}
}
