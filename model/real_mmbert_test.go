package model

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/UnicoLab/ul_layago/safetensors"
	"github.com/UnicoLab/ul_layago/tensor"
	"github.com/UnicoLab/ul_layago/tokenizer"
)

// mmbertGolden is testdata/mmbert_base_golden.json, written by
// testdata/parity/mmbert_base_golden.py from transformers' ModernBertModel
// on the same files.
type mmbertGolden struct {
	WeightsSHA256   string `json:"weights_sha256"`
	ConfigSHA256    string `json:"config_sha256"`
	TokenizerSHA256 string `json:"tokenizer_sha256"`
	Producer        string `json:"producer"`
	SampleDims      []int  `json:"sample_dims"`
	Cases           []struct {
		Lang       string      `json:"lang"`
		Text       string      `json:"text"`
		IDs        []int32     `json:"ids"`
		LayerNorms [][]float64 `json:"layer_norms"`
		FinalNorms []float64   `json:"final_norms"`
		FinalSums  []float64   `json:"final_sums"`
		FinalDims  [][]float64 `json:"final_dims"`
	} `json:"cases"`
}

func fileSHA256(t testing.TB, path string) string {
	t.Helper()
	f, err := os.Open(path) //nolint:gosec // G304: test fixture path from the environment
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		t.Fatal(err)
	}
	return hex.EncodeToString(h.Sum(nil))
}

// LoadMMBERTDir loads a converted Hugging Face ModernBERT directory
// (config.json, model.safetensors with "model." tensors, tokenizer.json).
func loadMMBERTDir(t testing.TB, dir string, quantized bool) (*Encoder, *tokenizer.Tokenizer) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(dir, "config.json")) //nolint:gosec // G304: test fixture path from the environment
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := ParseHFConfig(raw)
	if err != nil {
		t.Fatal(err)
	}
	f, err := safetensors.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	w, err := LoadWeights(f, cfg, LoadOptions{Prefix: "model.", Int8: quantized})
	if err != nil {
		t.Fatal(err)
	}
	enc, err := NewEncoder(cfg, w)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.LoadFile(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	return enc, tok
}

func rowNorm(row []float32) float64 {
	var ss float64
	for _, v := range row {
		ss += float64(v) * float64(v)
	}
	return math.Sqrt(ss)
}

// TestRealMMBERTBaseParity runs the full 22-layer encoder on the real
// jhu-clsp/mmBERT-base weights and compares it with PyTorch's
// ModernBertModel on EN/FR/DE/JA/AR inputs and one 173-token mixed-language
// Laya prompt, which exercises the sliding-window layers. The weights are not
// in the repository: set LAYAGO_MMBERT_DIR to the converted directory. The
// committed golden holds token ids, per-token L2 norms of every hidden state
// and 16 sampled dimensions of the final state; LAYAGO_MMBERT_FULL_GOLDEN may
// point at the full safetensors dump the same script writes (--full), which
// is then compared element by element.
func TestRealMMBERTBaseParity(t *testing.T) {
	dir := os.Getenv("LAYAGO_MMBERT_DIR")
	if dir == "" {
		t.Skip("set LAYAGO_MMBERT_DIR to a converted mmBERT-base directory matching testdata/mmbert_base_golden.json")
	}
	raw, err := os.ReadFile("testdata/mmbert_base_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var golden mmbertGolden
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string]string{
		"model.safetensors": golden.WeightsSHA256, "config.json": golden.ConfigSHA256, "tokenizer.json": golden.TokenizerSHA256,
	} {
		if got := fileSHA256(t, filepath.Join(dir, name)); got != want {
			t.Fatalf("%s sha256 = %s, golden was produced from %s", name, got, want)
		}
	}
	enc, tok := loadMMBERTDir(t, dir, false)
	var full *safetensors.File
	if p := os.Getenv("LAYAGO_MMBERT_FULL_GOLDEN"); p != "" {
		if full, err = safetensors.Open(p); err != nil {
			t.Fatal(err)
		}
		defer func() { _ = full.Close() }()
	}
	kernel := tensor.KernelAuto
	if k := os.Getenv("LAYAGO_MMBERT_KERNEL"); k != "" {
		if kernel, err = tensor.ParseKernel(k); err != nil {
			t.Fatal(err)
		}
	}
	par := tensor.Parallel{Kernel: kernel}
	t.Logf("kernel %s, reference %s", par.EffectiveKernel(), golden.Producer)

	var worstNormRel, worstDim, worstSumRel, worstFull, worstFullRel float64
	for ci, c := range golden.Cases {
		ids, err := tok.Encode(c.Text, true)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(ids.IDs, c.IDs) {
			t.Fatalf("%s: token ids differ from Hugging Face tokenizers:\n got %v\nwant %v", c.Lang, ids.IDs, c.IDs)
		}
		res, err := enc.Forward(context.Background(), [][]int32{ids.IDs}, Options{Parallel: par, CaptureLayers: true})
		if err != nil {
			t.Fatal(err)
		}
		d := enc.Config.HiddenSize
		if len(res.Layers) != len(c.LayerNorms) {
			t.Fatalf("%s: %d hidden states, golden has %d", c.Lang, len(res.Layers), len(c.LayerNorms))
		}
		var caseWorst float64
		for k, states := range res.Layers {
			x := states[0]
			for tok := range c.IDs {
				got := rowNorm(x[tok*d : (tok+1)*d])
				rel := math.Abs(got-c.LayerNorms[k][tok]) / math.Max(1, c.LayerNorms[k][tok])
				worstNormRel = math.Max(worstNormRel, rel)
				caseWorst = math.Max(caseWorst, rel)
			}
			if full != nil {
				ref, err := full.Tensor(fmt.Sprintf("case%d.layer%02d", ci, k))
				if err != nil {
					t.Fatal(err)
				}
				abs, rel := compareFull(t, x, ref)
				worstFull, worstFullRel = math.Max(worstFull, abs), math.Max(worstFullRel, rel)
			}
		}
		h := res.Hidden[0]
		for tok := range c.IDs {
			row := h[tok*d : (tok+1)*d]
			worstNormRel = math.Max(worstNormRel, math.Abs(rowNorm(row)-c.FinalNorms[tok])/math.Max(1, c.FinalNorms[tok]))
			var sum float64
			for _, v := range row {
				sum += float64(v)
			}
			worstSumRel = math.Max(worstSumRel, math.Abs(sum-c.FinalSums[tok])/math.Max(1, c.FinalNorms[tok]))
			for j, dim := range golden.SampleDims {
				worstDim = math.Max(worstDim, math.Abs(float64(row[dim])-c.FinalDims[tok][j]))
			}
		}
		if full != nil {
			ref, err := full.Tensor(fmt.Sprintf("case%d.final", ci))
			if err != nil {
				t.Fatal(err)
			}
			abs, rel := compareFull(t, h, ref)
			worstFull, worstFullRel = math.Max(worstFull, abs), math.Max(worstFullRel, rel)
		}
		t.Logf("%-10s %3d tokens: worst relative hidden-state norm error %.2e", c.Lang, len(c.IDs), caseWorst)
	}
	t.Logf("all %d cases: norm rel %.2e, final sampled dims abs %.2e, final row sum rel %.2e", len(golden.Cases), worstNormRel, worstDim, worstSumRel)
	if full != nil {
		t.Logf("full dump: max abs %.2e, max abs relative to the row's max |x| %.2e", worstFull, worstFullRel)
	}
	// Float32 reassociation across 22 layers (PyTorch's BLAS against the Go
	// kernels) stays well inside these bounds; a wrong mask, rope, norm or
	// activation moves every row by orders of magnitude more.
	const normTol, dimTol = 1e-4, 1e-3
	if worstNormRel > normTol || worstSumRel > normTol*10 || worstDim > dimTol {
		t.Fatalf("parity outside tolerance: norm rel %.2e (<= %g), row sum rel %.2e (<= %g), sampled dims %.2e (<= %g)",
			worstNormRel, normTol, worstSumRel, normTol*10, worstDim, dimTol)
	}
	if full != nil && worstFullRel > 1e-4 {
		t.Fatalf("full dump: max relative error %.2e > 1e-4", worstFullRel)
	}
}

// compareFull returns the max absolute difference and the max difference
// relative to the largest magnitude in the same row.
func compareFull(t *testing.T, got []float32, ref safetensors.Tensor) (float64, float64) {
	t.Helper()
	want, err := ref.Float32()
	if err != nil {
		t.Fatal(err)
	}
	if len(want) != len(got) || len(ref.Shape) != 2 {
		t.Fatalf("%s: %d values, want %d", ref.Name, len(got), len(want))
	}
	d := ref.Shape[1]
	var worst, worstRel float64
	for r := 0; r < ref.Shape[0]; r++ {
		var scale float64
		for _, v := range want[r*d : (r+1)*d] {
			scale = math.Max(scale, math.Abs(float64(v)))
		}
		for i := r * d; i < (r+1)*d; i++ {
			diff := math.Abs(float64(got[i]) - float64(want[i]))
			worst = math.Max(worst, diff)
			worstRel = math.Max(worstRel, diff/math.Max(1, scale))
		}
	}
	return worst, worstRel
}
