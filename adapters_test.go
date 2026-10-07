package layago

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/UnicoLab/ul_layago/tensor"
)

func adapterFixture(t *testing.T) *AdapterSet {
	t.Helper()
	s, err := OpenAdapters(AdapterSetOptions{Base: "testdata/parity/checkpoint", Artifacts: map[string]string{"float": "testdata/artifact/tiny-q/float32", "quant": "testdata/artifact/tiny-q/int8"}, Workers: 1, Kernel: tensor.KernelGeneric, CacheBytes: 1 << 20})
	if err != nil {
		t.Fatal(err)
	}
	return s
}
func adapterRequest() Request {
	return Request{ID: "request", State: "Hello", Questions: []Question{{ID: "question", Task: "q_metric_choice", Options: []Option{{Key: "a", Description: "red"}, {Key: "b", Description: "blue"}}}}}
}
func TestAdapterSetExplicitSelectionParityAndCancellation(t *testing.T) {
	set := adapterFixture(t)
	r := adapterRequest()
	if set.predictors["float"].agent.Encoder != set.predictors["quant"].agent.Encoder {
		t.Fatal("duplicated encoder")
	}
	for id, dir := range map[string]string{"float": "float32", "quant": "int8"} {
		p, err := Open(OpenOptions{Artifact: "testdata/artifact/tiny-q/" + dir, Base: "testdata/parity/checkpoint", Workers: 1, Kernel: tensor.KernelGeneric})
		if err != nil {
			t.Fatal(err)
		}
		a, err := set.Predict(context.Background(), id, r)
		if err != nil {
			t.Fatal(err)
		}
		b, err := p.Predict(context.Background(), r)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(a.Decisions, b.Decisions) || a.Model.AdapterID != id || a.Model.SHA256 != b.Model.SHA256 {
			t.Fatal("adapter output drift")
		}
	}
	for _, id := range []string{"", "unknown"} {
		if _, err := set.Predict(context.Background(), id, r); !errors.Is(err, ErrUnknownAdapter) {
			t.Fatal(err)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := set.Predict(ctx, "float", r); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	set.predictors["float"].gate <- struct{}{}
	if _, err := set.Predict(ctx, "quant", r); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	<-set.predictors["float"].gate
	r.State = strings.Repeat("word ", 300)
	if _, err := set.Predict(context.Background(), "float", r); !errors.Is(err, ErrTruncation) {
		t.Fatal(err)
	}
	r.AllowTruncation = true
	a, err := set.Predict(context.Background(), "float", r)
	if err != nil || !a.Decisions[0].Truncated || a.Decisions[0].Decision != "escalate" {
		t.Fatalf("truncation lost %v", err)
	}
}
func TestAdapterSetCalibrationAndThresholdsCannotLeakAcrossHeads(t *testing.T) {
	dir := t.TempDir()
	for _, file := range []string{"manifest.json", "model.safetensors"} {
		data, err := os.ReadFile("testdata/artifact/tiny-q/float32/" + file)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, file), data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	path := filepath.Join(dir, "manifest.json")
	data, _ := os.ReadFile(path)
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatal(err)
	}
	m["calibration"].(map[string]any)["temperatures"] = map[string]any{"*": map[string]any{"t": 0.5, "n": 100}}
	m["act"].(map[string]any)["thresholds"] = map[string]any{}
	data, _ = json.Marshal(m)
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	set, err := OpenAdapters(AdapterSetOptions{Base: "testdata/parity/checkpoint", Artifacts: map[string]string{"original": "testdata/artifact/tiny-q/float32", "changed": dir}, Workers: 1, Kernel: tensor.KernelGeneric, CacheBytes: 1 << 20})
	if err != nil {
		t.Fatal(err)
	}
	r := adapterRequest()
	original, err := set.Predict(context.Background(), "original", r)
	if err != nil {
		t.Fatal(err)
	}
	changed, err := set.Predict(context.Background(), "changed", r)
	if err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(original.Decisions[0].Probabilities, changed.Decisions[0].Probabilities) || changed.Decisions[0].Decision != "escalate" {
		t.Fatal("calibration/threshold leakage")
	}
	restored, err := set.Predict(context.Background(), "original", r)
	if err != nil || !reflect.DeepEqual(restored.Decisions, original.Decisions) {
		t.Fatal("global active head changed")
	}
	metadata := set.Metadata()
	metadata["original"] = Metadata{}
	if set.Metadata()["original"].SHA256 == "" {
		t.Fatal("mutable registry metadata")
	}
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			id := []string{"original", "changed"}[i%2]
			expected := original
			if id == "changed" {
				expected = changed
			}
			a, err := set.Predict(context.Background(), id, r)
			if err != nil || !reflect.DeepEqual(a.Decisions, expected.Decisions) {
				t.Errorf("concurrent adapter leak: %v", err)
			}
		}(i)
	}
	wg.Wait()
	if set.CacheStats().Hits < 10 {
		t.Fatal("identical input not reused")
	}
}
func TestAdapterSetRejectsInvalidConfiguration(t *testing.T) {
	for _, opts := range []AdapterSetOptions{{}, {Base: "x"}, {Base: "x", Artifacts: map[string]string{"": "x"}}, {Base: "x", Artifacts: map[string]string{"id": "x"}, CacheBytes: -1}, {Base: "x", Artifacts: map[string]string{"id": "x"}, Workers: -1}, {Base: "x", Artifacts: map[string]string{"id": "x"}, Kernel: tensor.Kernel(255)}} {
		if _, err := OpenAdapters(opts); err == nil {
			t.Fatal("invalid config accepted")
		}
	}
}

type queuedContext struct {
	context.Context
	waiting chan struct{}
	once    sync.Once
}

func (c *queuedContext) Done() <-chan struct{} {
	c.once.Do(func() { close(c.waiting) })
	return c.Context.Done()
}
func TestAdapterQueueCancellationDoesNotRunOrCache(t *testing.T) {
	set := adapterFixture(t)
	gate := set.predictors["float"].gate
	gate <- struct{}{}
	defer func() { <-gate }()
	before := set.CacheStats()
	parent, cancel := context.WithCancel(context.Background())
	defer cancel()
	ctx := &queuedContext{Context: parent, waiting: make(chan struct{})}
	done := make(chan error, 1)
	go func() { _, err := set.Predict(ctx, "quant", adapterRequest()); done <- err }()
	// Done is evaluated by the occupied-gate select, after both early Err checks.
	select {
	case <-ctx.waiting:
	case <-time.After(time.Second):
		t.Fatal("request never reached shared compute queue")
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("queued cancellation blocked")
	}
	if set.CacheStats() != before {
		t.Fatal("cancelled queued pass touched cache")
	}
}
