package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFixedOptionsCompareSemantically(t *testing.T) {
	if !equalJSON(json.RawMessage(`[{"key":"a","text":"Alpha"}]`), json.RawMessage("[ { \"text\": \"Alpha\", \"key\": \"a\" } ]")) {
		t.Fatal("formatting changed task identity")
	}
	if equalJSON(json.RawMessage(`[{"key":"a","text":"Alpha"}]`), json.RawMessage(`[{"key":"a","text":"Beta"}]`)) {
		t.Fatal("different options accepted")
	}
}
func TestConfigAndUnsealingValidation(t *testing.T) {
	f := filepath.Join(t.TempDir(), "config.json")
	for _, text := range []string{"null", `{} {}`, `{"unknown":1}`} {
		if err := os.WriteFile(f, []byte(text), 0600); err != nil {
			t.Fatal(err)
		}
		var v struct{}
		if readJSON(f, &v) == nil {
			t.Fatalf("accepted %s", text)
		}
	}
	var out strings.Builder
	err := run(context.Background(), []string{"train", "--dataset", "unused", "--unseal-locked-test", "--access-log", "audit.jsonl"}, &out)
	if err == nil || !strings.Contains(err.Error(), "only be unsealed") {
		t.Fatal(err)
	}
}
