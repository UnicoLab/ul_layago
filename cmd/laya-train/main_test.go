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

func TestBadTrainingConfigFailsBeforeDatasetOrModelIO(t *testing.T) {
	f := filepath.Join(t.TempDir(), "invalid.json")
	if err := os.WriteFile(f, []byte(`{"max_epochs":0}`), 0600); err != nil {
		t.Fatal(err)
	}
	var out strings.Builder
	err := run(context.Background(), []string{"train", "--dataset", "missing-dataset", "--base", "missing-base", "--config", f}, &out)
	if err == nil || !strings.Contains(err.Error(), "epochs") {
		t.Fatalf("loaded resources before config validation: %v", err)
	}
}
func TestReadinessRequiresNoBaseAndUsesActualCertificationPolicy(t *testing.T) {
	var out strings.Builder
	err := run(context.Background(), []string{"readiness", "--dataset", "../../train/testdata/fixture_sample.jsonl", "--instance", "R"}, &out)
	if err != nil {
		t.Fatal(err)
	}
	var report struct {
		Contract       string `json:"contract"`
		Start          int    `json:"certification_start"`
		TokenRetention bool   `json:"token_retention_checked"`
	}
	if err = json.Unmarshal([]byte(out.String()), &report); err != nil {
		t.Fatal(err)
	}
	if report.Contract != "ul.layago-calibration-plan/v1" || report.Start != 176 || report.TokenRetention {
		t.Fatal(report)
	}
}
