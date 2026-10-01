package laya

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"math"
	"os"
	"testing"
)

// promptGolden is testdata/prompt/prompt_golden.jsonl, written by the
// trainer's own renderer (make_prompt_golden.py).
type promptGolden struct {
	Example struct {
		ID       string         `json:"id"`
		Instance string         `json:"instance"`
		Input    map[string]any `json:"input"`
	} `json:"example"`
	State string `json:"state"`
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
		dec := json.NewDecoder(bytes.NewReader(sc.Bytes()))
		dec.UseNumber()
		var g promptGolden
		if err := dec.Decode(&g); err != nil {
			t.Fatal(err)
		}
		out = append(out, g)
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return out
}

// TestPromptMatchesTrainer holds the Go renderer to the trainer's states for
// the tiny-q golden examples, a sample of the fixture and every edge case.
func TestPromptMatchesTrainer(t *testing.T) {
	golden := readPromptGolden(t)
	if len(golden) < 60 {
		t.Fatalf("only %d golden examples", len(golden))
	}
	instances := map[string]int{}
	for _, g := range golden {
		got, err := RenderPromptState(g.Example.Instance, g.Example.Input)
		if err != nil {
			t.Fatalf("%s: %v", g.Example.ID, err)
		}
		if got != g.State {
			t.Errorf("%s:\n got %q\nwant %q", g.Example.ID, got, g.State)
		}
		instances[g.Example.Instance]++
	}
	if instances["Q"] == 0 || instances["R"] == 0 || instances["A"] == 0 {
		t.Fatalf("golden covers instances %v, want Q, R and A", instances)
	}
}

// The tiny-q parity goldens were computed on states this renderer must
// reproduce from the example inputs.
func TestPromptReproducesArtifactGoldenStates(t *testing.T) {
	inputs := map[string]promptGolden{}
	for _, g := range readPromptGolden(t) {
		inputs[g.Example.ID] = g
	}
	for _, variant := range []string{"float32", "int8"} {
		g := readArtifactGolden(t, artifactDir+variant)
		for _, c := range g.Cases {
			ex, ok := inputs[c.ID]
			if !ok {
				t.Fatalf("%s: golden case missing from prompt_golden.jsonl", c.ID)
			}
			got, err := RenderPromptState(ex.Example.Instance, ex.Example.Input)
			if err != nil || got != c.State {
				t.Fatalf("%s %s: rendered %q (%v), trained on %q", variant, c.ID, got, err, c.State)
			}
		}
	}
}

func TestPromptGoTypedInput(t *testing.T) {
	got, err := RenderPromptState("Q", map[string]any{
		"question": " Revenue? ",
		"catalog": []map[string]any{{
			"key": "revenue", "labels": []string{"revenue", "Umsatz", "revenue"}, "unit": "EUR",
			"dimensions":   []string{"store"},
			"scope_values": map[string][]string{"store": {"Paris", " "}},
		}},
	})
	want := "question: Revenue?\ncatalog:\n- revenue: revenue | Umsatz; unit: EUR; dimensions: store; scope: store=Paris"
	if err != nil || got != want {
		t.Fatalf("got %q (%v), want %q", got, err, want)
	}
}

func TestPromptRefusesWhatIsNotPorted(t *testing.T) {
	for name, input := range map[string]map[string]any{
		"no question":         {"catalog": []any{}},
		"catalog not a list":  {"question": "q", "catalog": map[string]any{"a": 1}},
		"nested scope value":  {"question": "q", "catalog": []any{map[string]any{"scope_values": map[string]any{"d": []any{[]any{"x"}}}}}},
		"nested metric value": {"question": "q", "catalog": []any{[]any{"x"}}},
	} {
		if _, err := RenderPromptState("Q", input); !errors.Is(err, ErrPromptInput) {
			t.Errorf("%s: err %v, want ErrPromptInput", name, err)
		}
	}
	if _, err := RenderPromptState("X", map[string]any{"question": "q"}); !errors.Is(err, ErrPromptInput) {
		t.Errorf("unknown instance: err %v", err)
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
	if got := pyFloatRepr(math.Copysign(0, -1)); got != "-0.0" {
		t.Errorf("repr(-0.0) = %q", got)
	}
}
