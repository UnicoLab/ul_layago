package tokenizer

import (
	"encoding/json"
	"errors"
	"os"
	"testing"
)

const parityDir = "../testdata/parity/"

type goldenCase struct {
	Text    string   `json:"text"`
	IDs     []int32  `json:"ids"`
	Offsets [][2]int `json:"offsets"`
}

func loadGolden(t *testing.T) map[string][]goldenCase {
	t.Helper()
	raw, err := os.ReadFile(parityDir + "tokenizer_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g map[string][]goldenCase
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

// The goldens come from the numpy-free Python reference in
// testdata/parity/layago_reference.py, which crosscheck_torch.py verified
// against Hugging Face tokenizers 0.22.2 (26/26 exact per layout).
func TestSyntheticTokenizerParity(t *testing.T) {
	golden := loadGolden(t)
	for name, path := range map[string]string{
		"synthetic":   parityDir + "checkpoint/tokenizer/tokenizer.json",
		"gemma_style": parityDir + "tokenizer_gemma_style.json",
	} {
		tok, err := LoadFile(path)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if len(golden[name]) == 0 {
			t.Fatalf("no golden cases for %s", name)
		}
		for _, c := range golden[name] {
			enc, err := tok.Encode(c.Text, false)
			if err != nil {
				t.Fatal(err)
			}
			if !sameEncoding(enc, c.IDs, c.Offsets) {
				t.Errorf("%s %q:\n got ids %v offsets %v\nwant ids %v offsets %v", name, c.Text, enc.IDs, enc.Offsets, c.IDs, c.Offsets)
			}
		}
	}
}

func TestTemplateAddsSpecialTokens(t *testing.T) {
	tok, err := LoadFile(parityDir + "checkpoint/tokenizer/tokenizer.json")
	if err != nil {
		t.Fatal(err)
	}
	plain, _ := tok.Encode("hello", false)
	withSpecial, err := tok.Encode("hello", true)
	if err != nil {
		t.Fatal(err)
	}
	bos, _ := tok.TokenID("<bos>")
	eos, _ := tok.TokenID("<eos>")
	if len(withSpecial.IDs) != len(plain.IDs)+2 || withSpecial.IDs[0] != bos || withSpecial.IDs[len(withSpecial.IDs)-1] != eos {
		t.Fatalf("template ids = %v", withSpecial.IDs)
	}
	if !withSpecial.Special[0] || withSpecial.Special[1] || withSpecial.Offsets[0] != [2]int{0, 0} {
		t.Fatalf("special flags or offsets wrong: %v %v", withSpecial.Special, withSpecial.Offsets)
	}
	if got := tok.Token(bos); got != "<bos>" {
		t.Fatalf("Token(%d) = %q", bos, got)
	}
	if id, ok := tok.TokenID("zz"); !ok || tok.Token(id) != "zz" {
		t.Fatal("an added token outside the model vocabulary must resolve both ways")
	}
}

func TestInvalidUTF8Rejected(t *testing.T) {
	tok, err := LoadFile(parityDir + "checkpoint/tokenizer/tokenizer.json")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := tok.Encode("bad \xff byte", false); err == nil {
		t.Fatal("invalid UTF-8 must be rejected")
	}
}

func TestUnsupportedComponentsFailClosed(t *testing.T) {
	raw, err := os.ReadFile(parityDir + "checkpoint/tokenizer/tokenizer.json")
	if err != nil {
		t.Fatal(err)
	}
	edited := func(t *testing.T, edit func(doc map[string]any)) []byte {
		t.Helper()
		var doc map[string]any
		if err := json.Unmarshal(raw, &doc); err != nil {
			t.Fatal(err)
		}
		edit(doc)
		out, err := json.Marshal(doc)
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	obj := func(doc map[string]any, key string) map[string]any { return doc[key].(map[string]any) }
	cases := map[string]func(doc map[string]any){
		"nfkc normalizer": func(doc map[string]any) { doc["normalizer"] = map[string]any{"type": "NFKC"} },
		"regex replace": func(doc map[string]any) {
			obj(doc, "normalizer")["pattern"] = map[string]any{"Regex": `\s`}
		},
		"bytelevel pre":  func(doc map[string]any) { doc["pre_tokenizer"] = map[string]any{"type": "ByteLevel"} },
		"unigram model":  func(doc map[string]any) { obj(doc, "model")["type"] = "Unigram" },
		"bpe dropout":    func(doc map[string]any) { obj(doc, "model")["dropout"] = 0.1 },
		"roberta post":   func(doc map[string]any) { obj(doc, "post_processor")["type"] = "RobertaProcessing" },
		"subword prefix": func(doc map[string]any) { obj(doc, "model")["continuing_subword_prefix"] = "##" },
		"pair only template": func(doc map[string]any) {
			obj(doc, "post_processor")["single"] = []any{map[string]any{"Sequence": map[string]any{"id": "B"}}}
		},
	}
	for name, edit := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := Load(edited(t, edit))
			if !errors.Is(err, ErrUnsupported) {
				t.Fatalf("Load error = %v, want ErrUnsupported", err)
			}
		})
	}
	if _, err := Load(edited(t, func(doc map[string]any) { obj(doc, "model")["unk_token"] = "<nope>" })); err == nil {
		t.Fatal("an unk_token outside the vocabulary must be rejected")
	}
	if _, err := Load(edited(t, func(doc map[string]any) {
		obj(doc, "model")["merges"] = []any{[]any{"q", "qq"}}
	})); err == nil {
		t.Fatal("a merge outside the vocabulary must be rejected")
	}
}

func FuzzEncode(f *testing.F) {
	tok, err := LoadFile(parityDir + "checkpoint/tokenizer/tokenizer.json")
	if err != nil {
		f.Fatal(err)
	}
	for _, s := range []string{"hello <mask> world", "zz new york <r>  x", "😀é\n\n\t", ""} {
		f.Add(s)
	}
	// Offsets may overlap: Hugging Face re-emits an added token that an rstrip
	// token already swallowed ("<r>\n\n" gives "<r>\n\n" then "\n\n"), and the
	// golden corpus pins that quirk. The invariants here are range, id bounds
	// and determinism.
	f.Fuzz(func(t *testing.T, s string) {
		enc, err := tok.Encode(s, true)
		if err != nil {
			return
		}
		again, _ := tok.Encode(s, true)
		if !sameEncoding(again, enc.IDs, enc.Offsets) {
			t.Fatalf("non-deterministic encoding for %q", s)
		}
		for i, off := range enc.Offsets {
			if off[0] < 0 || off[1] > len(s) || off[0] > off[1] {
				t.Fatalf("offset %v out of range for %q", off, s)
			}
			if enc.IDs[i] < 0 || int(enc.IDs[i]) >= tok.VocabSize() {
				t.Fatalf("id %d out of range", enc.IDs[i])
			}
		}
	})
}
