package model

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/UnicoLab/ul_layago/safetensors"
	"github.com/UnicoLab/ul_layago/tensor"
)

const parityDir = "../testdata/parity/"

func loadParityEncoder(t testing.TB, quantized bool) *Encoder {
	t.Helper()
	raw, err := os.ReadFile(parityDir + "checkpoint/encoder/config.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := ParseHFConfig(raw)
	if err != nil {
		t.Fatal(err)
	}
	f, err := safetensors.Open(parityDir + "checkpoint/model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	w, err := LoadWeights(f, cfg, LoadOptions{Prefix: "encoder.", Int8: quantized})
	if err != nil {
		t.Fatal(err)
	}
	enc, err := NewEncoder(cfg, w)
	if err != nil {
		t.Fatal(err)
	}
	return enc
}

type encoderGolden struct {
	Cases []struct {
		IDs []int32 `json:"ids"`
	} `json:"cases"`
}

func maxAbsDiff(t *testing.T, got []float32, want safetensors.Tensor) float64 {
	t.Helper()
	w, err := want.Float32()
	if err != nil {
		t.Fatal(err)
	}
	if len(w) != len(got) {
		t.Fatalf("%s: %d values, want %d", want.Name, len(got), len(w))
	}
	var worst float64
	for i := range w {
		worst = math.Max(worst, math.Abs(float64(got[i])-float64(w[i])))
	}
	return worst
}

// The goldens are the float64 numpy reference (testdata/parity/
// layago_reference.py), which crosscheck_torch.py matched against
// transformers 4.57.6 ModernBertModel to 1.6e-5 on every hidden state.
func TestEncoderMatchesNumpyReference(t *testing.T) {
	raw, err := os.ReadFile(parityDir + "encoder_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var meta encoderGolden
	if err := json.Unmarshal(raw, &meta); err != nil {
		t.Fatal(err)
	}
	golden, err := safetensors.Open(parityDir + "encoder_golden.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = golden.Close() }()
	seqs := make([][]int32, len(meta.Cases))
	for i, c := range meta.Cases {
		seqs[i] = c.IDs
	}
	for _, run := range kernelRuns([]int{1, 4}) {
		workers, kern := run.Workers, run.Kernel
		enc := loadParityEncoder(t, false)
		res, err := enc.Forward(context.Background(), seqs, Options{Parallel: run, CaptureLayers: true})
		if err != nil {
			t.Fatal(err)
		}
		if len(res.Layers) != enc.Config.NumLayers+1 {
			t.Fatalf("captured %d hidden states, want %d", len(res.Layers), enc.Config.NumLayers+1)
		}
		var worstLayer, worstFinal float64
		for c := range seqs {
			for k := range res.Layers {
				want, err := golden.Tensor(fmt.Sprintf("case%d.hidden%d", c, k))
				if err != nil {
					t.Fatal(err)
				}
				worstLayer = math.Max(worstLayer, maxAbsDiff(t, res.Layers[k][c], want))
			}
			want, err := golden.Tensor(fmt.Sprintf("case%d.final", c))
			if err != nil {
				t.Fatal(err)
			}
			worstFinal = math.Max(worstFinal, maxAbsDiff(t, res.Hidden[c], want))
		}
		t.Logf("%s workers=%d: max |Δ| any hidden state %.3g, last hidden state %.3g", kern, workers, worstLayer, worstFinal)
		// M0 tolerance: per-layer max abs diff 1e-3 (design V5); the tiny
		// model is held to 1e-4 on every kernel.
		if worstLayer > 1e-4 || worstFinal > 1e-4 {
			t.Fatalf("%s: encoder parity exceeded 1e-4: layers %.3g, final %.3g", kern, worstLayer, worstFinal)
		}
	}
}

// kernelRuns pairs every worker count with every GEMM kernel this process can
// run (the portable one always, plus NEON or AVX2 where available).
func kernelRuns(workers []int) []tensor.Parallel {
	var out []tensor.Parallel
	for _, k := range tensor.Kernels() {
		for _, w := range workers {
			out = append(out, tensor.Parallel{Workers: w, Kernel: k})
		}
	}
	return out
}

func TestInt8EncoderMatchesDequantizedReference(t *testing.T) {
	raw, err := os.ReadFile(parityDir + "encoder_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var meta encoderGolden
	if err := json.Unmarshal(raw, &meta); err != nil {
		t.Fatal(err)
	}
	golden, err := safetensors.Open(parityDir + "encoder_golden.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = golden.Close() }()
	enc := loadParityEncoder(t, true)
	for _, kern := range tensor.Kernels() {
		opt := Options{Parallel: tensor.Parallel{Kernel: kern}}
		for c, cs := range meta.Cases {
			res, err := enc.Forward(context.Background(), [][]int32{cs.IDs}, opt)
			if err != nil {
				t.Fatal(err)
			}
			want, err := golden.Tensor(fmt.Sprintf("case%d.final_int8", c))
			if err != nil {
				t.Fatal(err)
			}
			if d := maxAbsDiff(t, res.Hidden[0], want); d > 1e-4 {
				t.Fatalf("%s case %d int8 path differs from the dequantized reference by %.3g", kern, c, d)
			}
			fp, err := golden.Tensor(fmt.Sprintf("case%d.final", c))
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("%s case %d: int8 vs fp32 reference max |Δ| %.3g (quantization error, reported not gated)", kern, c, maxAbsDiff(t, res.Hidden[0], fp))
		}
	}
}

// Batching sequences together must not change any sequence's output, and
// repeated runs must be bit-identical, on every kernel.
func TestForwardIsDeterministicAndBatchInvariant(t *testing.T) {
	enc := loadParityEncoder(t, false)
	seqs := [][]int32{{2, 40, 41, 42}, {2, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 1}}
	for _, kern := range tensor.Kernels() {
		batch, err := enc.Forward(context.Background(), seqs, Options{Parallel: tensor.Parallel{Workers: 3, Kernel: kern}})
		if err != nil {
			t.Fatal(err)
		}
		for i, s := range seqs {
			single, err := enc.Forward(context.Background(), [][]int32{s}, Options{Parallel: tensor.Parallel{Workers: 1, Kernel: kern}})
			if err != nil {
				t.Fatal(err)
			}
			for j := range single.Hidden[0] {
				if math.Float32bits(single.Hidden[0][j]) != math.Float32bits(batch.Hidden[i][j]) {
					t.Fatalf("%s sequence %d element %d differs between batched and single forward", kern, i, j)
				}
			}
		}
	}
	var tm Timings
	if _, err := enc.Forward(context.Background(), seqs, Options{Timings: &tm}); err != nil {
		t.Fatal(err)
	}
	if tm.Total() <= 0 || len(tm.Map()) != int(numOps) {
		t.Fatalf("timings not recorded: %v", tm.Map())
	}
}

func TestForwardValidatesInputsAndContext(t *testing.T) {
	enc := loadParityEncoder(t, false)
	if _, err := enc.Forward(context.Background(), [][]int32{{}}, Options{}); err == nil {
		t.Fatal("empty sequence must be rejected")
	}
	if _, err := enc.Forward(context.Background(), [][]int32{{int32(enc.Config.VocabSize)}}, Options{}); err == nil {
		t.Fatal("out-of-vocabulary id must be rejected")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := enc.Forward(ctx, [][]int32{{2, 3}}, Options{}); err == nil {
		t.Fatal("a cancelled context must stop the forward pass")
	}
	// Longer than the initial rotary table: tables grow on demand.
	long := make([]int32, 700)
	for i := range long {
		long[i] = int32(5 + i%200)
	}
	if _, err := enc.Forward(context.Background(), [][]int32{long}, Options{}); err != nil {
		t.Fatal(err)
	}
}

func TestParseHFConfigLayouts(t *testing.T) {
	v5 := []byte(`{"model_type":"modernbert","vocab_size":10,"hidden_size":8,"intermediate_size":4,
		"num_hidden_layers":3,"num_attention_heads":2,"norm_eps":1e-5,"hidden_activation":"gelu",
		"local_attention":128,"global_attn_every_n_layers":3,
		"layer_types":["full_attention","sliding_attention","full_attention"],
		"rope_parameters":{"full_attention":{"rope_theta":160000,"rope_type":"default"},
		"sliding_attention":{"rope_theta":160000,"rope_type":"default"}}}`)
	cfg, err := ParseHFConfig(v5)
	if err != nil {
		t.Fatal(err)
	}
	// Explicit layer_types win over the every-n rule (layer 2 is global).
	want := []LayerSpec{{true, -1, 160000}, {false, 64, 160000}, {true, -1, 160000}}
	for i, l := range cfg.Layers {
		if l != want[i] {
			t.Fatalf("layer %d = %+v, want %+v", i, l, want[i])
		}
	}
	for name, doc := range map[string]string{
		"not modernbert": `{"model_type":"bert"}`,
		"tanh gelu": `{"model_type":"modernbert","vocab_size":10,"hidden_size":8,"intermediate_size":4,"num_hidden_layers":1,
			"num_attention_heads":2,"norm_eps":1e-5,"hidden_activation":"gelu_pytorch_tanh","global_rope_theta":1,"global_attn_every_n_layers":1}`,
		"no theta": `{"model_type":"modernbert","vocab_size":10,"hidden_size":8,"intermediate_size":4,"num_hidden_layers":1,
			"num_attention_heads":2,"norm_eps":1e-5,"hidden_activation":"gelu","global_attn_every_n_layers":1}`,
		"odd head dim": `{"model_type":"modernbert","vocab_size":10,"hidden_size":6,"intermediate_size":4,"num_hidden_layers":1,
			"num_attention_heads":2,"norm_eps":1e-5,"hidden_activation":"gelu","global_rope_theta":1,"global_attn_every_n_layers":1}`,
		"yarn rope": `{"model_type":"modernbert","vocab_size":10,"hidden_size":8,"intermediate_size":4,"num_hidden_layers":1,
			"num_attention_heads":2,"norm_eps":1e-5,"hidden_activation":"gelu","global_attn_every_n_layers":1,
			"rope_parameters":{"full_attention":{"rope_theta":1,"rope_type":"yarn"}}}`,
	} {
		if _, err := ParseHFConfig([]byte(doc)); err == nil {
			t.Fatalf("%s: config accepted", name)
		}
	}
}

func TestLoadWeightsRejectsMismatchedCheckpoints(t *testing.T) {
	raw, err := os.ReadFile(parityDir + "checkpoint/encoder/config.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := ParseHFConfig(raw)
	if err != nil {
		t.Fatal(err)
	}
	f, err := safetensors.Open(parityDir + "checkpoint/model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	wrong := cfg
	wrong.IntermediateSize++
	if _, err := LoadWeights(f, wrong, LoadOptions{Prefix: "encoder."}); err == nil {
		t.Fatal("a shape mismatch must be rejected")
	}
	fewer := cfg
	fewer.NumLayers--
	fewer.Layers = fewer.Layers[:fewer.NumLayers]
	if _, err := LoadWeights(f, fewer, LoadOptions{Prefix: "encoder."}); err == nil {
		t.Fatal("unexpected extra layer weights must be rejected")
	}
	if _, err := LoadWeights(f, cfg, LoadOptions{Prefix: "missing."}); err == nil {
		t.Fatal("a missing prefix must be rejected")
	}
}
