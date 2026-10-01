package tensor

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// asmKernels lists the assembly kernels this process can run (none under
// purego, on other architectures, or on an amd64 CPU without AVX2+FMA).
func asmKernels(t testing.TB) []Kernel {
	var out []Kernel
	for _, k := range Kernels() {
		if k != KernelGeneric {
			out = append(out, k)
		}
	}
	if len(out) == 0 {
		t.Logf("no assembly kernel on this CPU/build; best kernel is %s", BestKernel())
	}
	return out
}

type gemmCase struct {
	m, out, in int
	int8, bias bool
}

func (c gemmCase) String() string {
	return fmt.Sprintf("m%d_n%d_k%d_int8=%v_bias=%v", c.m, c.out, c.in, c.int8, c.bias)
}

// kernelCases covers every residue of M mod 8 and 6, N mod 8 and 16, and K
// mod 4 (the unrolled main loop and its one-step tail), plus seeded random
// shapes with M, N and K in 1..700.
func kernelCases() []gemmCase {
	var cases []gemmCase
	i := 0
	for _, m := range []int{1, 2, 5, 6, 7, 8, 9, 12, 13, 47, 48, 49, 64, 65} {
		for _, out := range []int{1, 7, 8, 9, 16, 17, 33} {
			for _, in := range []int{1, 2, 3, 4, 5, 8, 67} {
				cases = append(cases, gemmCase{m, out, in, i%2 == 1, i%3 == 0})
				i++
			}
		}
	}
	r := rand.New(rand.NewPCG(11, 12))
	for len(cases) < 686+80 {
		c := gemmCase{1 + r.IntN(700), 1 + r.IntN(700), 1 + r.IntN(700), r.IntN(2) == 1, r.IntN(2) == 1}
		if c.m*c.out*c.in > 12_000_000 {
			continue
		}
		cases = append(cases, c)
	}
	return cases
}

// referenceLinear returns the float64 value of each output and the sum of
// |x·w| it came from, which bounds the rounding error of any summation order.
func referenceLinear(x []float32, l *Linear, m int) (want, mag []float64) {
	want = make([]float64, m*l.Out)
	mag = make([]float64, m*l.Out)
	for r := 0; r < m; r++ {
		for n := 0; n < l.Out; n++ {
			var s, a float64
			for k := 0; k < l.In; k++ {
				wv := 0.0
				if l.Q != nil {
					wv = float64(l.Q[n*l.In+k])
				} else {
					wv = float64(l.W[n*l.In+k])
				}
				p := float64(x[r*l.In+k]) * wv
				s += p
				a += math.Abs(p)
			}
			if l.Q != nil {
				s *= float64(l.Scale[n])
				a *= math.Abs(float64(l.Scale[n]))
			}
			if l.Bias != nil {
				s += float64(l.Bias[n])
				a += math.Abs(float64(l.Bias[n]))
			}
			want[r*l.Out+n], mag[r*l.Out+n] = s, a
		}
	}
	return want, mag
}

func newCaseLinear(t testing.TB, r *rand.Rand, c gemmCase) (*Linear, []float32) {
	x := randomSlice(r, c.m*c.in)
	var bias []float32
	if c.bias {
		bias = randomSlice(r, c.out)
	}
	l, err := NewLinear(c.out, c.in, randomSlice(r, c.out*c.in), bias)
	if err != nil {
		t.Fatal(err)
	}
	if c.int8 {
		l.Quantize()
	}
	return l, x
}

// genericFused reports whether the generic kernel computes column n of an
// out-column Linear with fused multiply-adds (see the fma_*_test.go files).
func genericFused(n, out int) bool {
	return genericFusesFMA && (genericFusesEdgeColumns || n < out-out%4)
}

// Every assembly kernel is compared with the generic kernel and with a
// float64 reference. Wherever the Go compiler fused the generic kernel's
// multiply-adds (all of arm64; amd64 v3 except the last Out mod 4 columns)
// the two must agree bit for bit; elsewhere (amd64 v1 and v2) within the
// rounding bound. Against float64 the error is held to 1e-5 of sum |x·w|
// (the design's relative-error gate, normalised so that cancellation cannot
// make it meaningless).
func TestKernelsMatchGenericAndFloat64(t *testing.T) {
	kernels := asmKernels(t)
	if len(kernels) == 0 {
		t.Skip("no assembly kernel to compare")
	}
	r := rand.New(rand.NewPCG(13, 14))
	cases := kernelCases()
	for _, kern := range kernels {
		var worstRel float64
		var identical, total, mustMatch int
		for _, c := range cases {
			l, x := newCaseLinear(t, r, c)
			gen := make([]float32, c.m*c.out)
			l.Forward(gen, x, c.m, Parallel{Workers: 3, Kernel: KernelGeneric})
			got := make([]float32, c.m*c.out)
			l.Forward(got, x, c.m, Parallel{Workers: 3, Kernel: kern})
			want, mag := referenceLinear(x, l, c.m)
			for i := range got {
				total++
				fused := genericFused(i%c.out, c.out)
				if fused {
					mustMatch++
				}
				if math.Float32bits(got[i]) == math.Float32bits(gen[i]) {
					identical++
				} else if fused {
					t.Fatalf("%s %v element %d = %x, generic %x: must be bit-identical where the compiler fused", kern, c, i, math.Float32bits(got[i]), math.Float32bits(gen[i]))
				}
				rel := math.Abs(float64(got[i])-want[i]) / (mag[i] + 1e-30)
				worstRel = math.Max(worstRel, rel)
				if rel > 1e-5 {
					t.Fatalf("%s %v element %d = %g, float64 %g (|Δ|/Σ|xw| = %.3g)", kern, c, i, got[i], want[i], rel)
				}
			}
		}
		t.Logf("%s: %d shapes, %d/%d elements bit-identical to generic (%d required: the fused ones), worst |Δ|/Σ|xw| vs float64 %.3g",
			kern, len(cases), identical, total, mustMatch, worstRel)
	}
}

// Chunking K into several assembly calls stores and reloads the float32
// accumulators, which is exact, so any chunk size gives the same bits.
func TestKChunkingIsBitIdentical(t *testing.T) {
	kernels := asmKernels(t)
	if len(kernels) == 0 {
		t.Skip("no assembly kernel")
	}
	r := rand.New(rand.NewPCG(15, 16))
	for _, kern := range kernels {
		for _, c := range []gemmCase{{13, 35, 129, false, true}, {9, 20, 700, true, true}, {50, 17, 33, true, false}} {
			l, x := newCaseLinear(t, r, c)
			base := make([]float32, c.m*c.out)
			l.Forward(base, x, c.m, Parallel{Workers: 2, Kernel: kern})
			saved := kChunk
			for _, chunk := range []int{1, 3, 4, 5, 64} {
				kChunk = chunk
				got := make([]float32, c.m*c.out)
				l.Forward(got, x, c.m, Parallel{Workers: 2, Kernel: kern})
				for i := range got {
					if math.Float32bits(got[i]) != math.Float32bits(base[i]) {
						kChunk = saved
						t.Fatalf("%s %v kChunk=%d element %d differs", kern, c, chunk, i)
					}
				}
			}
			kChunk = saved
		}
	}
}

func TestKernelSelection(t *testing.T) {
	best := BestKernel()
	if !best.Available() || best == KernelAuto {
		t.Fatalf("BestKernel() = %s", best)
	}
	if got := (Parallel{}).EffectiveKernel(); got != best {
		t.Fatalf("auto resolves to %s, want %s", got, best)
	}
	if got := (Parallel{Kernel: KernelGeneric}).EffectiveKernel(); got != KernelGeneric {
		t.Fatalf("generic resolves to %s", got)
	}
	for _, k := range []Kernel{KernelNEON, KernelAVX2, Kernel(99)} {
		want := k
		if !k.Available() {
			want = KernelGeneric
		}
		if got := (Parallel{Kernel: k}).EffectiveKernel(); got != want {
			t.Fatalf("%s resolves to %s, want %s", k, got, want)
		}
	}
	if (Parallel{Kernel: KernelGeneric}).microkernel() != nil {
		t.Fatal("the generic kernel must not dispatch to assembly")
	}
	for _, k := range []Kernel{KernelAuto, KernelGeneric, KernelNEON, KernelAVX2} {
		back, err := ParseKernel(k.String())
		if err != nil || back != k {
			t.Fatalf("ParseKernel(%q) = %v, %v", k.String(), back, err)
		}
	}
	if _, err := ParseKernel("sse2"); err == nil {
		t.Fatal("unknown kernel names must be rejected")
	}
	if ks := Kernels(); ks[0] != KernelGeneric || ks[len(ks)-1] != best {
		t.Fatalf("Kernels() = %v", ks)
	}
	if mk := archMicrokernel(); mk != nil {
		if mr, nr := mk.id.Tile(); mr != mk.mr || nr != mk.nr || mr*nr > maxTile {
			t.Fatalf("%s: Tile() = %dx%d, descriptor %dx%d (max %d)", mk.id, mr, nr, mk.mr, mk.nr, maxTile)
		}
		if mk.blockM%mk.mr != 0 || mk.blockN%mk.nr != 0 {
			t.Fatalf("%s: task block %dx%d is not a multiple of the tile", mk.id, mk.blockM, mk.blockN)
		}
	}
	t.Logf("best kernel on this CPU/build: %s (available: %v)", best, Kernels())
}

// Quantize must drop panels packed from the float32 weights, and the int8
// panels built afterwards must reproduce the generic int8 path.
func TestQuantizeDropsPackedPanels(t *testing.T) {
	kernels := asmKernels(t)
	if len(kernels) == 0 {
		t.Skip("no assembly kernel")
	}
	r := rand.New(rand.NewPCG(17, 18))
	m, out, in := 10, 21, 30
	x := randomSlice(r, m*in)
	for _, kern := range kernels {
		l, _ := NewLinear(out, in, randomSlice(r, out*in), randomSlice(r, out))
		if l.PackedBytes() != 0 {
			t.Fatal("packing must be lazy")
		}
		l.Prepack(KernelGeneric)
		if l.PackedBytes() != 0 {
			t.Fatal("the generic kernel needs no packed copy")
		}
		l.Prepack(kern)
		fpBytes := l.PackedBytes()
		if fpBytes < 4*out*in {
			t.Fatalf("packed %d bytes for %dx%d float32 weights", fpBytes, out, in)
		}
		dst := make([]float32, m*out)
		l.Forward(dst, x, m, Parallel{Kernel: kern})
		l.Quantize()
		if l.PackedBytes() != 0 {
			t.Fatal("Quantize kept the float32 panels")
		}
		got := make([]float32, m*out)
		l.Forward(got, x, m, Parallel{Kernel: kern})
		if b := l.PackedBytes(); b == 0 || b >= fpBytes {
			t.Fatalf("int8 panels hold %d bytes (float32 held %d)", b, fpBytes)
		}
		want := make([]float32, m*out)
		l.Forward(want, x, m, Parallel{Kernel: KernelGeneric})
		for i := range got {
			if d := math.Abs(float64(got[i] - want[i])); d > 1e-5*(1+math.Abs(float64(want[i]))) || (genericFused(i%out, out) && d != 0) {
				t.Fatalf("%s int8 element %d = %g, generic %g", kern, i, got[i], want[i])
			}
		}
	}
}

// Concurrent first calls race to pack; every caller must still see a
// complete panel set and the same result.
func TestConcurrentFirstForwardPacksOnce(t *testing.T) {
	kernels := asmKernels(t)
	if len(kernels) == 0 {
		t.Skip("no assembly kernel")
	}
	r := rand.New(rand.NewPCG(19, 20))
	m, out, in := 17, 45, 64
	x := randomSlice(r, m*in)
	w := randomSlice(r, out*in)
	for _, kern := range kernels {
		l, _ := NewLinear(out, in, w, nil)
		want := make([]float32, m*out)
		l2, _ := NewLinear(out, in, w, nil)
		l2.Forward(want, x, m, Parallel{Workers: 1, Kernel: kern})
		results := make([][]float32, 8)
		Parallel{Workers: 8}.Run(8, func(i int) {
			results[i] = make([]float32, m*out)
			l.Forward(results[i], x, m, Parallel{Workers: 2, Kernel: kern})
		})
		for i, res := range results {
			for j := range res {
				if math.Float32bits(res[j]) != math.Float32bits(want[j]) {
					t.Fatalf("%s caller %d element %d differs", kern, i, j)
				}
			}
		}
	}
}
