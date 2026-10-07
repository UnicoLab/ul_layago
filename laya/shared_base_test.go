package laya

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"github.com/UnicoLab/ul_layago/safetensors"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/UnicoLab/ul_layago/model"
	"github.com/UnicoLab/ul_layago/tensor"
)

func TestSharedBasePreservesIndependentArtifactParity(t *testing.T) {
	base, err := LoadArtifactBase(artifactDir+"float32", artifactBaseDir, ArtifactOptions{FeatureCacheBytes: 1 << 20})
	if err != nil {
		t.Fatal(err)
	}
	for _, format := range []string{"float32", "int8"} {
		shared, err := LoadArtifactWithBase(artifactDir+format, base)
		if err != nil {
			t.Fatal(err)
		}
		standalone, err := LoadArtifact(artifactDir+format, artifactBaseDir, ArtifactOptions{})
		if err != nil {
			t.Fatal(err)
		}
		if shared.Agent.Encoder != base.encoder || shared.Agent.Tokenizer != base.tokenizer {
			t.Fatal("base was duplicated")
		}
		golden := readArtifactGolden(t, artifactDir+format)
		checkArtifactParity(t, shared, golden, tensor.Parallel{Workers: 1, Kernel: tensor.KernelGeneric})
		for _, c := range golden.Cases {
			q := []ArtifactQuestion{artifactQuestion(c)}
			opt := model.Options{Parallel: tensor.Parallel{Workers: 1, Kernel: tensor.KernelGeneric}}
			a, ap, err := shared.Predict(context.Background(), c.State, q, opt)
			if err != nil {
				t.Fatal(err)
			}
			b, bp, err := standalone.Predict(context.Background(), c.State, q, opt)
			if err != nil {
				t.Fatal(err)
			}
			// Threshold pointers are independently owned but values/outputs are identical.
			if !reflect.DeepEqual(a, b) || !reflect.DeepEqual(ap, bp) {
				t.Fatalf("%s %s shared drift", format, c.ID)
			}
		}
	}
	if s := base.CacheStats(); s.Hits == 0 || s.Bytes > s.BudgetBytes {
		t.Fatal(s)
	}
}

func TestSharedBaseRejectsEveryChangedPinAndEncoderAdapter(t *testing.T) {
	base, err := LoadArtifactBase(artifactDir+"float32", artifactBaseDir, ArtifactOptions{})
	if err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*ArtifactManifest){
		"weights": func(m *ArtifactManifest) { m.Base.WeightsSHA256 = "wrong" }, "config": func(m *ArtifactManifest) { m.Base.ConfigSHA256 = "wrong" },
		"tokenizer": func(m *ArtifactManifest) { m.Tokenizer.SHA256 = "wrong" }, "special": func(m *ArtifactManifest) {
			s := m.Tokenizer.SpecialTokens["mask"]
			s.ID++
			m.Tokenizer.SpecialTokens["mask"] = s
		},
		"depth": func(m *ArtifactManifest) { m.Base.LayersUsed-- }, "sequence": func(m *ArtifactManifest) { m.Sequence.Format = "unsupported" },
		"unsupported_adapter": func(m *ArtifactManifest) { m.Adapter = json.RawMessage(`{"kind":"lora"}`) },
	} {
		t.Run(name, func(t *testing.T) {
			dir := copyArtifact(t, artifactDir+"float32")
			raw, _ := os.ReadFile(filepath.Join(dir, "manifest.json"))
			var m ArtifactManifest
			if err := json.Unmarshal(raw, &m); err != nil {
				t.Fatal(err)
			}
			mutate(&m)
			raw, _ = json.Marshal(m)
			if err := os.WriteFile(filepath.Join(dir, "manifest.json"), raw, 0600); err != nil {
				t.Fatal(err)
			}
			if _, err := LoadArtifactWithBase(dir, base); err == nil {
				t.Fatal("accepted changed pin")
			}
			if name == "unsupported_adapter" {
				if _, err := LoadArtifact(dir, artifactBaseDir, ArtifactOptions{}); err == nil {
					t.Fatal("legacy loader ignored adapter")
				}
			}
		})
	}
	if _, err := LoadArtifactWithBase(artifactDir+"float32", nil); err == nil {
		t.Fatal("nil base")
	}
	if _, err := LoadArtifactBase(artifactDir+"float32", artifactBaseDir, ArtifactOptions{FeatureCacheBytes: -1}); err == nil {
		t.Fatal("negative budget")
	}
	// Mutating the first attached manifest cannot change the private base pins.
	art, err := LoadArtifactWithBase(artifactDir+"float32", base)
	if err != nil {
		t.Fatal(err)
	}
	s := art.Manifest.Tokenizer.SpecialTokens["mask"]
	s.ID++
	art.Manifest.Tokenizer.SpecialTokens["mask"] = s
	if _, err := LoadArtifactWithBase(artifactDir+"float32", base); err != nil {
		t.Fatal(err)
	}
}

func TestFeatureCacheBoundsOwnershipAndExactSequenceKeys(t *testing.T) {
	cache := newFeatureCache(144)
	par := tensor.Parallel{Kernel: tensor.KernelGeneric}
	key := sequenceKey([]int32{1, 2}, par)
	values := []float32{1, 2, 3, 4}
	cache.put(key, values)
	values[0] = 99
	got, ok := cache.get(key)
	if !ok || got[0] != 1 {
		t.Fatal("borrowed caller data")
	}
	got[0] = 88
	got, _ = cache.get(key)
	if got[0] != 1 {
		t.Fatal("returned mutable cache data")
	}
	other := sequenceKey([]int32{2, 1}, par)
	if other == key {
		t.Fatal("order ignored")
	}
	if sequenceKey([]int32{1}, par) == key {
		t.Fatal("prefix reused")
	}
	cache.put(other, []float32{5, 6, 7, 8})
	if _, ok := cache.get(key); ok {
		t.Fatal("budget did not evict")
	}
	cache.put(key, make([]float32, 5))
	s := cache.stats()
	if s.Bytes > 144 || s.Entries != 1 || s.Evictions != 1 {
		t.Fatal(s)
	}
}

func TestSharedCacheInstrumentationCancellationAndConcurrentCalls(t *testing.T) {
	base, err := LoadArtifactBase(artifactDir+"float32", artifactBaseDir, ArtifactOptions{FeatureCacheBytes: 1 << 20})
	if err != nil {
		t.Fatal(err)
	}
	art, err := LoadArtifactWithBase(artifactDir+"float32", base)
	if err != nil {
		t.Fatal(err)
	}
	c := readArtifactGolden(t, artifactDir+"float32").Cases[0]
	qs := []ArtifactQuestion{artifactQuestion(c)}
	opt := model.Options{Parallel: tensor.Parallel{Workers: 1, Kernel: tensor.KernelGeneric}}
	expected, _, err := art.Predict(context.Background(), c.State, qs, opt)
	if err != nil {
		t.Fatal(err)
	}
	before := base.CacheStats()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := art.Predict(ctx, c.State, qs, opt); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if base.CacheStats() != before {
		t.Fatal("cancelled cache used")
	}
	var tm model.Timings
	opt.Timings = &tm
	if _, _, err := art.Predict(context.Background(), c.State, qs, opt); err != nil {
		t.Fatal(err)
	}
	if base.CacheStats() != before || tm.Total() == 0 {
		t.Fatal("instrumented forward hidden by cache")
	}
	opt.Timings = nil
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			a, _, err := art.Predict(context.Background(), c.State, qs, opt)
			if err != nil || !reflect.DeepEqual(a, expected) {
				t.Errorf("concurrent drift %v", err)
			}
		}()
	}
	wg.Wait()
	if !slices.Equal(base.encoder.Config.Layers, art.Agent.Encoder.Config.Layers) {
		t.Fatal("shared encoder changed")
	}
}

func TestSharedBasePreservesCompleteTransformerHead(t *testing.T) {
	// Keep both synthetic pretrained head layers; never flatten them into the
	// zero-layer training architecture. The same full states serve either head.
	dir := copyArtifact(t, artifactDir+"float32")
	file, err := safetensors.Open(filepath.Join(artifactBaseDir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	var tensors []safetensors.NamedTensor
	for _, name := range file.Header.Names() {
		if strings.HasPrefix(name, "encoder.") || name == "temperature" {
			continue
		}
		tensor, err := file.Tensor(name)
		if err != nil {
			t.Fatal(err)
		}
		v, err := tensor.Float32()
		if err != nil {
			t.Fatal(err)
		}
		tensors = append(tensors, safetensors.Float32Tensor(name, tensor.Shape, v))
	}
	var bytes bytes.Buffer
	if err := safetensors.Write(&bytes, tensors, nil); err != nil {
		t.Fatal(err)
	}
	raw := bytes.Bytes()
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), raw, 0600); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	editManifest(t, dir, func(m map[string]any) {
		m["head"].(map[string]any)["layers"] = 2
		m["weights"].(map[string]any)["sha256"] = hex.EncodeToString(sum[:])
	})
	base, err := LoadArtifactBase(artifactDir+"float32", artifactBaseDir, ArtifactOptions{FeatureCacheBytes: 1 << 20})
	if err != nil {
		t.Fatal(err)
	}
	cached, err := LoadArtifactWithBase(dir, base)
	if err != nil {
		t.Fatal(err)
	}
	plain, err := LoadArtifact(dir, artifactBaseDir, ArtifactOptions{})
	if err != nil {
		t.Fatal(err)
	}
	c := readArtifactGolden(t, artifactDir+"float32").Cases[0]
	q := []ArtifactQuestion{artifactQuestion(c)}
	opt := model.Options{Parallel: tensor.Parallel{Kernel: tensor.KernelGeneric, Workers: 1}}
	expected, _, err := plain.Predict(context.Background(), c.State, q, opt)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		got, _, err := cached.Predict(context.Background(), c.State, q, opt)
		if err != nil || !reflect.DeepEqual(got, expected) {
			t.Fatalf("complete head drift %v", err)
		}
	}
	if len(cached.Agent.Head.layers) != 2 || base.CacheStats().Hits != 1 {
		t.Fatal("layers/cache lost")
	}
}

func TestSharedBaseKeepsIndependentSequenceBudgets(t *testing.T) {
	dir := copyArtifact(t, artifactDir+"float32")
	editManifest(t, dir, func(m map[string]any) {
		m["sequence"].(map[string]any)["max_len"] = 256
		m["sequence"].(map[string]any)["head_max_len"] = 112
	})
	base, err := LoadArtifactBase(artifactDir+"float32", artifactBaseDir, ArtifactOptions{FeatureCacheBytes: 1 << 20})
	if err != nil {
		t.Fatal(err)
	}
	original, err := LoadArtifactWithBase(artifactDir+"float32", base)
	if err != nil {
		t.Fatal(err)
	}
	larger, err := LoadArtifactWithBase(dir, base)
	if err != nil {
		t.Fatal(err)
	}
	if original.Agent.Config.MaxLen != 160 || original.Agent.Config.HeadMaxLen != 64 || larger.Agent.Config.MaxLen != 256 || larger.Agent.Config.HeadMaxLen != 112 {
		t.Fatal("budget inheritance")
	}
	c := readArtifactGolden(t, artifactDir+"float32").Cases[0]
	qs := []ArtifactQuestion{artifactQuestion(c)}
	opt := model.Options{Parallel: tensor.Parallel{Workers: 1, Kernel: tensor.KernelGeneric}}
	for _, test := range []struct {
		dir string
		art *Artifact
	}{{artifactDir + "float32", original}, {dir, larger}} {
		standalone, err := LoadArtifact(test.dir, artifactBaseDir, ArtifactOptions{})
		if err != nil {
			t.Fatal(err)
		}
		a, ap, err := test.art.Predict(context.Background(), c.State, qs, opt)
		if err != nil {
			t.Fatal(err)
		}
		b, bp, err := standalone.Predict(context.Background(), c.State, qs, opt)
		if err != nil || !reflect.DeepEqual(a, b) || !reflect.DeepEqual(ap, bp) {
			t.Fatal("adapter-specific budget drift")
		}
	}
	if base.CacheStats().Entries != 2 {
		t.Fatal("distinct budgeted token sequences aliased")
	}
}
