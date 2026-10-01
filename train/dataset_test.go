package train

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"
)

const goodLine = `{"id":"a","split":"train","lang":"en","instance":"Q","task":"q_operation","task_type":"choice",` +
	`"input":{"question":"What was revenue?","catalog":[]},"options":["lookup",{"key":"rank","text":"rank the segments"}],"label":1,` +
	`"label_source":"template","domain_id":"retail","story_id":"s1","generator_version":"g/1","seed":7}`

func parseLines(t *testing.T, lines ...string) (*Dataset, error) {
	t.Helper()
	return ParseDataset([]DataFile{{Name: "d.jsonl", Data: []byte(strings.Join(lines, "\n") + "\n")}}, "")
}

// editGood returns goodLine changed by fn.
func editGood(t *testing.T, fn func(m map[string]any)) string {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal([]byte(goodLine), &m); err != nil {
		t.Fatal(err)
	}
	fn(m)
	raw, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

// TestParseDatasetAcceptsContractForms: every form hive.laya-dataset/v1
// allows, including the ones the previous parser refused (yes_no
// [false_text, true_text] options) or ignored (label_dist).
func TestParseDatasetAcceptsContractForms(t *testing.T) {
	yesNoText := editGood(t, func(m map[string]any) {
		m["id"], m["task"], m["task_type"] = "b", "q_time_supported", "yes_no"
		m["options"] = []any{"the period is not supported", "the period is supported"}
		m["label"] = true
	})
	yesNoDefault := editGood(t, func(m map[string]any) {
		m["id"], m["task"], m["task_type"], m["split"] = "c", "q_time_supported", "yes_no", "dev"
		delete(m, "options")
		m["label"] = false
	})
	score := editGood(t, func(m map[string]any) {
		m["id"], m["instance"], m["task"], m["task_type"], m["split"] = "d", "R", "r_evidence_relevance", "score", "heldout_domain"
		m["input"] = map[string]any{"question": "q", "evidence": []any{map[string]any{"id": "e1", "text": "t"}}}
		m["options"] = []any{"irrelevant", map[string]any{"text": "related"}, "directly relevant"}
		m["label"] = 2
	})
	soft := editGood(t, func(m map[string]any) {
		m["id"], m["instance"], m["task"] = "e", "R", "r_answer_rank"
		m["input"] = map[string]any{"question": "q", "evidence": []any{}}
		m["options"] = []any{"ev1", "ev2", "ev3"}
		m["label"], m["label_dist"] = 0, []any{0.5, 0.5, 0}
	})
	llm := editGood(t, func(m map[string]any) {
		m["id"], m["label_source"] = "f", "llm_translation"
		m["provenance"] = strings.Repeat("ab", 32)
	})
	ds, err := parseLines(t, goodLine, yesNoText, yesNoDefault, score, soft, llm)
	if err != nil {
		t.Fatal(err)
	}
	ex := func(id string) *Example {
		for i := range ds.Examples {
			if ds.Examples[i].ID == id {
				return &ds.Examples[i]
			}
		}
		t.Fatalf("no %s", id)
		return nil
	}
	if e := ex("a"); e.Label != 1 || !slices.Equal(e.Options, []Option{{Key: "lookup"}, {Key: "rank", Text: "rank the segments"}}) ||
		e.Seed != "7" || e.Role() != SplitTrain {
		t.Errorf("a: %+v", e)
	}
	if e := ex("b"); e.Label != 1 || e.NumOptions() != 2 || e.Options[1] != (Option{Key: "1", Text: "the period is supported"}) {
		t.Errorf("b: yes_no text options %+v label %d", e.Options, e.Label)
	}
	if e := ex("c"); e.Label != 0 || len(e.Options) != 0 || e.NumOptions() != 2 || !slices.Equal(e.ClassKeys(), []string{"false", "true"}) ||
		e.Role() != DevHalf("s1") {
		t.Errorf("c: %+v role %s", e, e.Role())
	}
	if e := ex("d"); e.Label != 2 || e.Options[1] != (Option{Key: "1", Text: "related"}) || !slices.Equal(e.ClassKeys(), []string{"0", "1", "2"}) {
		t.Errorf("d: %+v", e)
	}
	if e := ex("e"); !slices.Equal(e.LabelDist, []float64{0.5, 0.5, 0}) || !slices.Equal(e.Target(), []float64{0.5, 0.5, 0}) {
		t.Errorf("e: label_dist %v target %v", e.LabelDist, e.Target())
	}
	if e := ex("f"); len(e.Provenance) != 1 {
		t.Errorf("f: provenance %v", e.Provenance)
	}
	if !slices.Equal(ex("a").Target(), []float64{0, 1}) {
		t.Errorf("one-hot target %v", ex("a").Target())
	}
	q, err := ParseDataset([]DataFile{{Name: "d.jsonl", Data: []byte(strings.Join([]string{goodLine, score}, "\n"))}}, "R")
	if err != nil || len(q.Examples) != 1 || q.Total != 2 || q.Digest() != DataDigest(q.Files) {
		t.Fatalf("instance R: %v, %d of %d", err, len(q.Examples), q.Total)
	}
}

// TestParseDatasetRejectsContractViolations: the loader refuses what the
// contract refuses, with the file and line, and coerces nothing.
func TestParseDatasetRejectsContractViolations(t *testing.T) {
	for _, c := range []struct {
		name string
		fn   func(m map[string]any)
		want string
	}{
		{"missing id", func(m map[string]any) { delete(m, "id") }, `"id" must be a non-empty string`},
		{"blank id", func(m map[string]any) { m["id"] = " 　" }, `"id" must be a non-empty string`},
		{"unknown split", func(m map[string]any) { m["split"] = "test" }, "split \"test\" is not one of"},
		{"bad lang", func(m map[string]any) { m["lang"] = "English" }, "is not a BCP-47 tag"},
		{"unknown instance", func(m map[string]any) { m["instance"] = "X" }, "instance \"X\""},
		{"task case", func(m map[string]any) { m["task"] = "Q_op" }, "lower snake case"},
		{"task of another instance", func(m map[string]any) { m["task"] = "r_op" }, "does not belong to instance Q"},
		{"task type", func(m map[string]any) { m["task_type"] = "multi" }, "task_type \"multi\""},
		{"one option", func(m map[string]any) { m["options"] = []any{"x"} }, "at least 2 options"},
		{"repeated keys", func(m map[string]any) { m["options"] = []any{"x", "x"} }, "keys must be unique"},
		{"empty option", func(m map[string]any) { m["options"] = []any{"x", " "} }, "option 1 is empty"},
		{"choice label is a key", func(m map[string]any) { m["label"] = "rank" }, "must be an option index"},
		{"choice label float", func(m map[string]any) { m["label"] = json.Number("1.0") }, "must be an option index"},
		{"label out of range", func(m map[string]any) { m["label"] = 2 }, "outside the 2 options"},
		{"yes_no label 1", func(m map[string]any) {
			m["task"], m["task_type"], m["label"] = "q_time_supported", "yes_no", 1
			delete(m, "options")
		}, "must be a boolean"},
		{"yes_no three options", func(m map[string]any) {
			m["task"], m["task_type"], m["label"] = "q_time_supported", "yes_no", true
			m["options"] = []any{"a", "b", "c"}
		}, "exactly [false_text, true_text]"},
		{"label_dist on dev", func(m map[string]any) { m["split"], m["label_dist"] = "dev", []any{0, 1} }, "only on the train split"},
		{"label_dist sum", func(m map[string]any) { m["label_dist"] = []any{0.2, 0.7} }, "must sum to 1"},
		{"label_dist length", func(m map[string]any) { m["label_dist"] = []any{1} }, "must be 2 non-negative numbers"},
		{"label_dist negative", func(m map[string]any) { m["label_dist"] = []any{-0.5, 1.5} }, "must be 2 non-negative numbers"},
		{"label not most probable", func(m map[string]any) { m["label_dist"] = []any{0.6, 0.4} }, "not a most probable option"},
		{"label source", func(m map[string]any) { m["label_source"] = "fixture" }, "label_source \"fixture\""},
		{"llm source without provenance", func(m map[string]any) { m["label_source"] = "claude_author" }, "needs the hive.llm-interaction/v1 id"},
		{"bad provenance id", func(m map[string]any) {
			m["label_source"], m["provenance"] = "llm_paraphrase", map[string]any{"llm_interaction_ids": []any{"abc"}}
		}, "is not a hive.llm-interaction/v1 sha256"},
		{"seed string", func(m map[string]any) { m["seed"] = "7" }, "seed must be an integer"},
		{"seed float", func(m map[string]any) { m["seed"] = 7.5 }, "seed must be an integer"},
		{"input string", func(m map[string]any) { m["input"] = "What was revenue?" }, "input must be an object"},
		{"no question", func(m map[string]any) { m["input"] = map[string]any{"catalog": []any{}} }, `"question" must be a non-empty string`},
		{"catalog not a list", func(m map[string]any) { m["input"] = map[string]any{"question": "q", "catalog": "x"} }, "input.catalog must be a list"},
		{"missing story", func(m map[string]any) { delete(m, "story_id") }, `"story_id" must be a non-empty string`},
		{"missing domain", func(m map[string]any) { m["domain_id"] = "" }, `"domain_id" must be a non-empty string`},
	} {
		_, err := parseLines(t, goodLine, editGood(t, func(m map[string]any) {
			m["id"] = "z"
			c.fn(m)
		}))
		if err == nil || !strings.Contains(err.Error(), c.want) || !strings.Contains(err.Error(), "d.jsonl:2:") {
			t.Errorf("%s: error %v, want d.jsonl:2 and %q", c.name, err, c.want)
		}
	}
	if _, err := parseLines(t, goodLine, goodLine); err == nil || !strings.Contains(err.Error(), "duplicate id") {
		t.Errorf("duplicate id: %v", err)
	}
	if _, err := parseLines(t, goodLine, `{"id":`); err == nil || !strings.Contains(err.Error(), "invalid JSON") {
		t.Errorf("truncated JSON: %v", err)
	}
	if _, err := ParseDataset([]DataFile{{Name: "d.jsonl", Data: []byte(goodLine)}}, "R"); err == nil ||
		!strings.Contains(err.Error(), "no examples for instance R") {
		t.Errorf("empty instance: %v", err)
	}
}

// TestDataDigestFollowsTheContract: the digest of the converged trainer's
// tiny-q fixture, as its manifest records it (w10/laya-train
// internal/layago/testdata/artifact/tiny-q/float32/manifest.json, which
// routermodel.Load recomputes the same way).
func TestDataDigestFollowsTheContract(t *testing.T) {
	files := []DatasetFile{{Name: "fixture.jsonl", SHA256: "e9f5369e83371f5a5407d26bab8e2994f13fb58b76a712e28f655c389e2b915e", Examples: 542}}
	if got, want := DataDigest(files), "7e46d78a52b70b1b1bfe15711f289ec26d57d1bf71834d3b458cf92b4413b5ab"; got != want {
		t.Fatalf("data digest %s, the contract gives %s", got, want)
	}
	two := []DatasetFile{{Name: "b.jsonl", SHA256: "02"}, {Name: "a.jsonl", SHA256: "01"}}
	if DataDigest(two) != DataDigest([]DatasetFile{two[1], two[0]}) {
		t.Fatal("the digest depends on file order")
	}
}

// TestDevHalfFollowsThePythonTrainer: the story hash of dataset.py
// dev_half, checked against values computed with it.
func TestDevHalfFollowsThePythonTrainer(t *testing.T) {
	for story, want := range map[string]string{
		"retail-en-4": "dev_conf", "retail-en-5": "dev_conf", "saas-de-3": "dev_cal", "logistics-fr-8": "dev_cal", "s1": "dev_conf",
		"retail-fr-6": "dev_cal", "retail-fr-7": "dev_conf",
	} {
		if got := DevHalf(story); got != want {
			t.Errorf("DevHalf(%q) = %s, dataset.py dev_half gives %s", story, got, want)
		}
	}
}

// TestSplitViolations: one split per story, held-out domains and languages
// absent from train and dev (by primary subtag).
func TestSplitViolations(t *testing.T) {
	line := func(id, split, lang, domain, story string) string {
		return editGood(t, func(m map[string]any) {
			m["id"], m["split"], m["lang"], m["domain_id"], m["story_id"] = id, split, lang, domain, story
		})
	}
	ds, err := parseLines(t,
		line("1", "train", "en", "retail", "s1"),
		line("2", "dev", "en", "retail", "s1"),
		line("3", "heldout_domain", "en", "retail", "s2"),
		line("4", "heldout_language", "pt-BR", "saas", "s3"),
		line("5", "train", "pt", "saas", "s4"),
		line("6", "locked_test", "en", "saas", "s5"),
		line("7", "heldout_language", "ja", "saas", "s6"),
	)
	if err != nil {
		t.Fatal(err)
	}
	// The Python trainer's wording, file by name and line.
	want := []string{
		`story 's1' appears in more than one split: ['dev', 'train'] (dev at d.jsonl:2, train at d.jsonl:1)`,
		`heldout_domain domain 'retail' also appears in train or dev (heldout_domain at d.jsonl:3, train at d.jsonl:1)`,
		`heldout_language 'pt-BR' shares its language with train or dev examples ['pt'] (heldout_language at d.jsonl:4, train 'pt' at d.jsonl:5)`,
	}
	if !slices.Equal(ds.SplitViolations, want) {
		t.Fatalf("violations\n got %q\nwant %q", ds.SplitViolations, want)
	}
}
