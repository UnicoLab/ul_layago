package model

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/UnicoLab/ul_layago/tensor"
)

var (
	fullOnce       sync.Once
	fullFP, fullQ8 *Encoder
)

// fullSizeEncoders builds the mmBERT-base shape with random weights once per
// process: a float32 encoder and an int8 one (about 1.2 GB and 0.3 GB).
func fullSizeEncoders(tb testing.TB) (*Encoder, *Encoder) {
	fullOnce.Do(func() {
		cfg := MMBERTBase()
		var err error
		if fullFP, err = NewEncoder(cfg, RandomWeights(cfg, 1)); err != nil {
			tb.Fatal(err)
		}
		q := RandomWeights(cfg, 1)
		q.Quantize()
		if fullQ8, err = NewEncoder(cfg, q); err != nil {
			tb.Fatal(err)
		}
	})
	return fullFP, fullQ8
}

// BenchmarkEncoderMMBERTBase times one full-size forward pass (22 layers,
// d=768, 12 heads, GeGLU 1152, random weights) per iteration, for every GEMM
// kernel this process can run (portable Go, and NEON or AVX2 where
// available), float32 and int8, on 1, 8 and all workers.
//
//	go test -run XXX -bench EncoderMMBERTBase -benchtime 10x ./internal/layago/model/
func BenchmarkEncoderMMBERTBase(b *testing.B) {
	fp, q8 := fullSizeEncoders(b)
	for _, kern := range tensor.Kernels() {
		fp.Weights.Prepack(kern)
		q8.Weights.Prepack(kern)
	}
	for _, n := range []int{32, 64, 128, 256} {
		ids := make([]int32, n)
		for i := range ids {
			ids[i] = int32(5 + (i*7919)%250000)
		}
		for _, enc := range []*Encoder{fp, q8} {
			path := "fp32"
			if enc == q8 {
				path = "int8"
			}
			for _, kern := range tensor.Kernels() {
				for _, workers := range []int{1, 8, 0} {
					par := tensor.Parallel{Workers: workers, Kernel: kern}
					b.Run(fmt.Sprintf("tokens=%d/%s/%s/workers=%d", n, path, kern, par.N()), func(b *testing.B) {
						for b.Loop() {
							if _, err := enc.Forward(context.Background(), [][]int32{ids}, Options{Parallel: par}); err != nil {
								b.Fatal(err)
							}
						}
						b.ReportMetric(enc.Config.FLOPs(n)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
					})
				}
			}
		}
	}
}

func TestFLOPsCountsWindowedAttention(t *testing.T) {
	cfg := MMBERTBase()
	perToken := cfg.FLOPs(1)
	// Dense work is 2 * 22 * 5.01M multiply-adds per token; one token
	// attends only to itself in every layer.
	if perToken < 220e6 || perToken > 221e6 {
		t.Fatalf("FLOPs(1) = %.4g, want about 220.5M", perToken)
	}
	if cfg.FLOPs(256) >= 256*cfg.FLOPs(1)+2*2*22*256*256*768 {
		t.Fatal("sliding layers must count fewer keys than global ones")
	}
}
