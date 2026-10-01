package tokenizer

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"reflect"
	"testing"
	"time"
)

type externalGolden struct {
	// SourceFileSHA256 pins the tokenizer.json the ids were produced from.
	SourceFileSHA256 string `json:"source_file_sha256"`
	Producer         string `json:"producer"`
	Cases            []struct {
		Text    string   `json:"text"`
		IDs     []int32  `json:"ids"`
		Offsets [][2]int `json:"offsets"`
	} `json:"cases"`
}

// TestExternalTokenizerParity compares against ids and byte offsets that the
// Hugging Face tokenizers library produced for a real tokenizer.json. The
// tokenizer file is not in the repository, so the test runs only when
// LAYAGO_TOKENIZER_JSON points at it; testdata/mmbert_tokenizer_golden.json
// pins the file digest and the expected encodings.
func TestExternalTokenizerParity(t *testing.T) {
	path := os.Getenv("LAYAGO_TOKENIZER_JSON")
	if path == "" {
		t.Skip("set LAYAGO_TOKENIZER_JSON to a tokenizer.json matching testdata/mmbert_tokenizer_golden.json")
	}
	goldenPath := os.Getenv("LAYAGO_TOKENIZER_GOLDEN")
	if goldenPath == "" {
		goldenPath = "testdata/mmbert_tokenizer_golden.json"
	}
	raw, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatal(err)
	}
	var golden externalGolden
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(data)
	if got := hex.EncodeToString(sum[:]); golden.SourceFileSHA256 != "" && got != golden.SourceFileSHA256 {
		t.Fatalf("tokenizer.json sha256 = %s, golden was produced from %s", got, golden.SourceFileSHA256)
	}
	started := time.Now()
	tok, err := Load(data)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("loaded %d-token vocabulary in %s", tok.VocabSize(), time.Since(started))
	mismatches := 0
	for _, c := range golden.Cases {
		enc, err := tok.Encode(c.Text, false)
		if err != nil {
			t.Fatal(err)
		}
		if !sameEncoding(enc, c.IDs, c.Offsets) {
			mismatches++
			t.Errorf("%q:\n got ids %v offsets %v\nwant ids %v offsets %v", c.Text, enc.IDs, enc.Offsets, c.IDs, c.Offsets)
		}
	}
	t.Logf("%d/%d cases exact (ids and byte offsets) against %s", len(golden.Cases)-mismatches, len(golden.Cases), golden.Producer)
}

// BenchmarkEncodeQuestion times one Laya-style prompt with the tokenizer
// named by LAYAGO_TOKENIZER_JSON, or the synthetic parity tokenizer.
func BenchmarkEncodeQuestion(b *testing.B) {
	path := os.Getenv("LAYAGO_TOKENIZER_JSON")
	if path == "" {
		path = parityDir + "checkpoint/tokenizer/tokenizer.json"
	}
	tok, err := LoadFile(path)
	if err != nil {
		b.Fatal(err)
	}
	text := "choice question: Does the current admitted measurement have exactly the same metric key, " +
		"unit, grain, and scope as the objective? Pourquoi le chiffre d'affaires a-t-il baissé ? 今月のGMVは?"
	var n int
	for b.Loop() {
		enc, err := tok.Encode(text, true)
		if err != nil {
			b.Fatal(err)
		}
		n = len(enc.IDs)
	}
	b.ReportMetric(float64(n), "tokens")
}

// sameEncoding compares ids and offsets, treating nil and empty alike.
func sameEncoding(enc Encoding, ids []int32, offsets [][2]int) bool {
	if len(enc.IDs) != len(ids) || len(enc.Offsets) != len(offsets) {
		return false
	}
	return len(ids) == 0 || (reflect.DeepEqual(enc.IDs, ids) && reflect.DeepEqual(enc.Offsets, offsets))
}
