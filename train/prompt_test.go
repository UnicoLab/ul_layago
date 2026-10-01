package train

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/UnicoLab/ul_layago/laya"
)

// promptGolden is ../testdata/prompt/prompt_golden.jsonl, written by the
// Python trainer's own renderer (make_prompt_golden.py). laya's own
// TestPromptMatchesTrainer holds laya.RenderPromptState to the same file.
type promptGolden struct {
	Example json.RawMessage `json:"example"`
	State   string          `json:"state"`
}

func readPromptGolden(t *testing.T) []promptGolden {
	t.Helper()
	raw, err := os.ReadFile("../testdata/prompt/prompt_golden.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	var out []promptGolden
	sc := bufio.NewScanner(bytes.NewReader(raw))
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		var g promptGolden
		if err := json.Unmarshal(sc.Bytes(), &g); err != nil {
			t.Fatal(err)
		}
		out = append(out, g)
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return out
}

// TestPromptMatchesTrainer holds the trainer's rendering path, which is
// laya.RenderPromptState (the renderer routermodel serves with), to the
// states the Python trainer rendered for 81 examples (the converged
// fixture's, its R and A examples and edge cases), byte for byte. The golden
// examples are also valid hive.laya-dataset/v1 lines for this loader, and
// render the same through it.
func TestPromptMatchesTrainer(t *testing.T) {
	golden := readPromptGolden(t)
	if len(golden) < 60 {
		t.Fatalf("only %d golden examples", len(golden))
	}
	instances := map[string]int{}
	parsed := 0
	for i, g := range golden {
		dec := json.NewDecoder(bytes.NewReader(g.Example))
		dec.UseNumber()
		var ex struct {
			ID       string         `json:"id"`
			Instance string         `json:"instance"`
			Input    map[string]any `json:"input"`
		}
		if err := dec.Decode(&ex); err != nil {
			t.Fatalf("golden %d: %v", i, err)
		}
		got, err := laya.RenderPromptState(ex.Instance, ex.Input)
		if err != nil {
			t.Fatalf("%s: %v", ex.ID, err)
		}
		if got != g.State {
			t.Errorf("%s:\n got %q\nwant %q", ex.ID, got, g.State)
		}
		instances[ex.Instance]++
		if ds, err := ParseDataset([]DataFile{{Name: "prompt_golden.jsonl", Data: g.Example}}, ""); err == nil {
			parsed++
			if s, err := laya.RenderPromptState(ds.Examples[0].Instance, ds.Examples[0].Input); err != nil || s != g.State {
				t.Errorf("%s: the loaded example renders %q (%v)", ex.ID, s, err)
			}
		}
	}
	if instances["Q"] == 0 || instances["R"] == 0 || instances["A"] == 0 {
		t.Fatalf("golden covers instances %v, want Q, R and A", instances)
	}
	t.Logf("%d states byte-identical to the trainer's; %d of the examples are valid dataset lines", len(golden), parsed)
}

// The trainer's catalog readers (pyvalue.go), which the shortlist and the
// floor use, read a catalog entry exactly as the served renderer prints it:
// the key and labels trimmed with Python's whitespace, labels in sorted key
// order without repeats, and scope values as Python's str() prints them.
func TestCatalogReadersAgreeWithTheRenderer(t *testing.T) {
	dec := json.NewDecoder(strings.NewReader(`{"question": "q", "catalog": [{"key": " rev ",
		"labels": {"fr": " Chiffre ", "en": "Revenue\u001c", "de": "Revenue"},
		"scope_values": {"year": [2024, 2024.0, 1e16, 0.0001, 1e-5, true, null, " x\u00a0", ""]}}]}`))
	dec.UseNumber()
	var input map[string]any
	if err := dec.Decode(&input); err != nil {
		t.Fatal(err)
	}
	catalog, err := catalogList(input["catalog"])
	if err != nil || len(catalog) != 1 {
		t.Fatalf("catalog %v, %v", catalog, err)
	}
	entry := catalog[0].(map[string]any)
	scope, ok := asMap(entry["scope_values"])
	if !ok {
		t.Fatal("scope_values is not read as a map")
	}
	values, ok := asList(scope["year"])
	if !ok {
		t.Fatal("a scope's values are not read as a list")
	}
	var shown []string
	for _, v := range values {
		s, err := pyStr(v)
		if err != nil {
			t.Fatal(err)
		}
		if s = pyStrip(s); s != "" {
			shown = append(shown, s)
		}
	}
	key := pyStrip(entry["key"].(string))
	want := "question: q\ncatalog:\n- " + key + ": " + strings.Join(promptLabels(entry["labels"]), " | ") +
		"; scope: year=" + strings.Join(shown, "/")
	if pinned := "question: q\ncatalog:\n- rev: Revenue | Chiffre; scope: year=2024/2024.0/1e+16/0.0001/1e-05/True/None/x"; want != pinned {
		t.Fatalf("the trainer's readers give\n%q\nwant\n%q", want, pinned)
	}
	got, err := laya.RenderPromptState("Q", input)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("laya.RenderPromptState renders\n%q\nthe trainer's readers\n%q", got, want)
	}
}

// The readers refuse what the renderer refuses, with the renderer's error.
func TestCatalogReadersRefuseWhatTheRendererRefuses(t *testing.T) {
	if _, err := catalogList(map[string]any{"a": 1}); !errors.Is(err, laya.ErrPromptInput) {
		t.Errorf("a catalog that is not a list: err %v, want laya.ErrPromptInput", err)
	}
	if _, err := pyStr([]any{"x"}); !errors.Is(err, laya.ErrPromptInput) {
		t.Errorf("a nested value: err %v, want laya.ErrPromptInput", err)
	}
	for name, input := range map[string]map[string]any{
		"catalog not a list": {"question": "q", "catalog": map[string]any{"a": 1}},
		"nested scope value": {"question": "q", "catalog": []any{map[string]any{"scope_values": map[string]any{"d": []any{[]any{"x"}}}}}},
	} {
		if _, err := laya.RenderPromptState("Q", input); !errors.Is(err, laya.ErrPromptInput) {
			t.Errorf("%s: the renderer returns %v, want laya.ErrPromptInput", name, err)
		}
	}
}

func TestPyFloatRepr(t *testing.T) {
	for f, want := range map[float64]string{
		0: "0.0", 1: "1.0", 100: "100.0", 0.1: "0.1", 1.5: "1.5", 1e15: "1000000000000000.0", 1e16: "1e+16",
		1.2345e20: "1.2345e+20", 1e-4: "0.0001", 1e-5: "1e-05", 1.5e-7: "1.5e-07", -2.5: "-2.5",
		123456789.125: "123456789.125", 1.0 / 3: "0.3333333333333333", 2.5e-300: "2.5e-300",
		math.MaxFloat64: "1.7976931348623157e+308",
	} {
		if got := pyFloatRepr(f); got != want {
			t.Errorf("repr(%v) = %q, want %q", f, got, want)
		}
	}
}
