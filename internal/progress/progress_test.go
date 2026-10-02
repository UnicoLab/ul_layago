package progress

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func TestProgressDurableAndHeartbeatCannotAdvance(t *testing.T) {
	path := filepath.Join(t.TempDir(), "progress.json")
	w := New(path)
	for i := 0; i < 3; i++ {
		if err := w.Advance("cache", 1); err != nil {
			t.Fatal(err)
		}
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var state struct {
		Sequence int    `json:"sequence"`
		Stage    string `json:"stage"`
	}
	if err = json.Unmarshal(raw, &state); err != nil {
		t.Fatal(err)
	}
	if state.Sequence != 1 || state.Stage != "cache" {
		t.Fatalf("%+v", state)
	}
	var wg sync.WaitGroup
	for i := 2; i < 12; i++ {
		wg.Add(1)
		go func(step int) {
			defer wg.Done()
			if err := w.Advance("scorer", step); err != nil {
				t.Error(err)
			}
		}(i)
	}
	wg.Wait()
	raw, err = os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(raw, &state); err != nil {
		t.Fatal(err)
	}
	if state.Sequence != 11 {
		t.Fatalf("got counter %d", state.Sequence)
	}
}
