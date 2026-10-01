package tensor

import (
	"fmt"
	"sync"
)

//go:generate go run ./asmgen -dir .

// Kernel names the GEMM microkernel behind Linear.Forward.
//
// KernelGeneric is the portable Go path in gemm.go. It is always available,
// it is the fallback on every other CPU, and it is the oracle the assembly
// kernels are tested against. The assembly kernels keep the portable path's
// arithmetic: every output element is one fused multiply-add chain over K in
// increasing order, so wherever the Go compiler fused the portable kernel's
// a*b+c they are bit-identical to it. That is every element on arm64. On
// amd64 with GOAMD64=v3 it is every element except the last Out mod 4
// columns, whose one-column loop the compiler turns into MULSS-from-memory
// plus ADDSS instead (none of the mmBERT projections has such columns).
// Below v3 (the release image builds at v1) nothing is fused, and the AVX2
// kernel, which rounds once per step instead of twice, agrees within
// floating-point tolerance.
type Kernel uint8

const (
	// KernelAuto selects the best kernel this CPU supports.
	KernelAuto Kernel = iota
	// KernelGeneric is the portable Go kernel.
	KernelGeneric
	// KernelNEON is the arm64 ASIMD kernel (8x8 tiles).
	KernelNEON
	// KernelAVX2 is the amd64 AVX2+FMA kernel (6x16 tiles).
	KernelAVX2
)

var kernelNames = [...]string{"auto", "generic", "arm64_neon", "amd64_avx2"}

// String returns the kernel's receipt name.
func (k Kernel) String() string {
	if int(k) < len(kernelNames) {
		return kernelNames[k]
	}
	return fmt.Sprintf("kernel(%d)", k)
}

// ParseKernel maps a receipt or configuration name back to a Kernel.
func ParseKernel(name string) (Kernel, error) {
	for i, n := range kernelNames {
		if n == name {
			return Kernel(i), nil
		}
	}
	return 0, fmt.Errorf("tensor: unknown kernel %q (want auto, generic, arm64_neon or amd64_avx2)", name)
}

// Tile returns the output tile one microkernel step computes, rows x columns
// (the generic kernel's dot4x4 is 4x4), and 0, 0 for KernelAuto.
func (k Kernel) Tile() (mr, nr int) {
	switch k {
	case KernelGeneric:
		return 4, 4
	case KernelNEON:
		return 8, 8
	case KernelAVX2:
		return 6, 16
	}
	return 0, 0
}

// MaxCallKSteps is the most K steps one assembly call runs; with Tile it
// bounds the work between two preemption points.
func MaxCallKSteps() int { return kChunk }

// Available reports whether k can run in this process: the generic kernel
// always can; an assembly kernel only on its architecture, with the CPU
// features it needs, and in a build without the purego tag.
func (k Kernel) Available() bool {
	switch k {
	case KernelAuto, KernelGeneric:
		return true
	}
	return microkernelFor(k) != nil
}

// BestKernel returns the kernel KernelAuto resolves to on this CPU.
func BestKernel() Kernel {
	if mk := archMicrokernel(); mk != nil {
		return mk.id
	}
	return KernelGeneric
}

// Kernels lists the kernels available in this process, generic first.
func Kernels() []Kernel {
	out := []Kernel{KernelGeneric}
	if best := BestKernel(); best != KernelGeneric {
		out = append(out, best)
	}
	return out
}

// EffectiveKernel resolves p.Kernel: KernelAuto becomes BestKernel(), and a
// kernel this process cannot run becomes KernelGeneric. Receipts record this
// value, never the request.
func (p Parallel) EffectiveKernel() Kernel {
	if p.Kernel == KernelAuto {
		return BestKernel()
	}
	if !p.Kernel.Available() {
		return KernelGeneric
	}
	return p.Kernel
}

func (p Parallel) microkernel() *microkernel {
	return microkernelFor(p.EffectiveKernel())
}

func microkernelFor(k Kernel) *microkernel {
	if mk := archMicrokernel(); mk != nil && mk.id == k {
		return mk
	}
	return nil
}

// microkernel describes one assembly tile kernel. f32 and q8 compute an
// mr x nr tile of C over kc steps of K:
//
//	c[i*nr+j] = acc ? c[i*nr+j] : 0
//	for k := 0; k < kc; k++ { c[i*nr+j] = fma(a[i*lda+k], b[k*nr+j], c[i*nr+j]) }
//
// where lda is in bytes, b is a packed panel (float32, or int8 widened to
// float32 in registers: the int8 path's dequantization happens inside the
// kernel) and c is a contiguous mr x nr scratch tile. Each call handles one
// tile and at most kChunk steps, so it returns within microseconds: Go
// assembly is not asynchronously preemptible, and the blocking loops that
// call it stay in Go.
type microkernel struct {
	id     Kernel
	mr, nr int
	// blockM and blockN size one parallel task (multiples of mr and nr).
	blockM, blockN int
	f32            func(kc int, a *float32, lda int, b *float32, c *float32, acc int)
	q8             func(kc int, a *float32, lda int, b *int8, c *float32, acc int)
}

// kChunk bounds the K steps of one assembly call. At the largest tile (6x16)
// a full chunk is 393k multiply-adds, a few microseconds on any AVX2 core.
// It is a variable so tests can force chunking and prove it bit-identical.
var kChunk = 4096

// maxTile is the largest mr*nr of any microkernel.
const maxTile = 128

// packedWeights is a Linear's weight matrix re-laid out for one microkernel:
// ceil(Out/nr) panels, panel p holding K rows of nr consecutive output
// columns (column j is weight row p*nr+j), zero-padded past Out. Exactly one
// of f32 and q8 is set, matching the Linear.
type packedWeights struct {
	kernel Kernel
	f32    []float32
	q8     []int8
}

// Prepack builds the packed weight panels for kernel k now instead of on the
// first Forward that uses it, so a loader pays the one-time re-layout (and
// the extra copy of the weights it costs) before serving. It is a no-op for
// the generic kernel or a kernel this process cannot run.
func (l *Linear) Prepack(k Kernel) {
	if mk := microkernelFor(Parallel{Kernel: k}.EffectiveKernel()); mk != nil {
		l.packedFor(mk)
	}
}

// PackedBytes reports the memory held by packed panels (0 when none).
func (l *Linear) PackedBytes() int {
	pw := l.packed.Load()
	if pw == nil {
		return 0
	}
	return 4*len(pw.f32) + len(pw.q8)
}

func (l *Linear) packedFor(mk *microkernel) *packedWeights {
	if pw := l.packed.Load(); pw != nil && pw.kernel == mk.id {
		return pw
	}
	l.packMu.Lock()
	defer l.packMu.Unlock()
	if pw := l.packed.Load(); pw != nil && pw.kernel == mk.id {
		return pw
	}
	pw := &packedWeights{kernel: mk.id}
	nr, k := mk.nr, l.In
	panels := (l.Out + nr - 1) / nr
	if l.Q != nil {
		pw.q8 = make([]int8, panels*k*nr)
		for n := 0; n < l.Out; n++ {
			p, j := n/nr, n%nr
			dst := pw.q8[p*k*nr:]
			for kk, v := range l.Q[n*k : (n+1)*k] {
				dst[kk*nr+j] = v
			}
		}
	} else {
		pw.f32 = make([]float32, panels*k*nr)
		for n := 0; n < l.Out; n++ {
			p, j := n/nr, n%nr
			dst := pw.f32[p*k*nr:]
			for kk, v := range l.W[n*k : (n+1)*k] {
				dst[kk*nr+j] = v
			}
		}
	}
	l.packed.Store(pw)
	return pw
}

// taskScratch is one task's working memory, pooled so the packed path does
// not allocate per task: the output tile the kernel accumulates into, and
// the rows of a partial row tile padded to mr, so rows at the edge of a
// block run through the same kernel as every other row (an element's value
// never depends on its row's position in a stacked batch).
type taskScratch struct {
	tile [maxTile]float32
	pad  []float32
}

var scratchPool = sync.Pool{New: func() any { return new(taskScratch) }}

// forwardPacked is Forward on an assembly microkernel. The loops are in Go;
// each kernel call computes one mr x nr tile over at most kChunk steps.
func (l *Linear) forwardPacked(dst, x []float32, m int, par Parallel, mk *microkernel) {
	pw := l.packedFor(mk)
	k, out, mr, nr := l.In, l.Out, mk.mr, mk.nr
	panelLen := k * nr
	mBlocks := (m + mk.blockM - 1) / mk.blockM
	nBlocks := (out + mk.blockN - 1) / mk.blockN
	par.Run(mBlocks*nBlocks, func(task int) {
		mb, nb := task/nBlocks, task%nBlocks
		m0, n0 := mb*mk.blockM, nb*mk.blockN
		m1, n1 := min(m0+mk.blockM, m), min(n0+mk.blockN, out)
		full := m0 + (m1-m0)/mr*mr
		s := scratchPool.Get().(*taskScratch)
		var pad []float32
		if full < m1 {
			if cap(s.pad) < mr*k {
				s.pad = make([]float32, mr*k)
			}
			pad = s.pad[:mr*k]
			clear(pad)
			copy(pad, x[full*k:m1*k])
		}
		tile := &s.tile
		for n := n0; n < n1; n += nr {
			cols := min(nr, n1-n)
			p := n / nr
			for r := m0; r < m1; r += mr {
				rows := mr
				a := pad
				if r < full {
					a = x[r*k : (r+mr)*k]
				} else {
					rows = m1 - r
				}
				if pw.q8 != nil {
					panel := pw.q8[p*panelLen : (p+1)*panelLen]
					for k0 := 0; k0 < k; k0 += kChunk {
						kc := min(kChunk, k-k0)
						callQ8(mk.q8, kc, &a[k0], 4*k, &panel[k0*nr], &tile[0], min(k0, 1))
					}
				} else {
					panel := pw.f32[p*panelLen : (p+1)*panelLen]
					for k0 := 0; k0 < k; k0 += kChunk {
						kc := min(kChunk, k-k0)
						callF32(mk.f32, kc, &a[k0], 4*k, &panel[k0*nr], &tile[0], min(k0, 1))
					}
				}
				l.storeTile(dst, tile[:mr*nr], r, n, rows, cols, nr)
			}
		}
		scratchPool.Put(s)
	})
}

// callF32 and callQ8 put a preemption point in front of every assembly call.
// The runtime asks a goroutine to stop (for the scheduler or a
// stop-the-world) by signalling its thread and by poisoning its stack guard.
// The signal cannot interrupt assembly, and the assembly kernels, being
// NOSPLIT leaves, never check the guard; the prologue of this non-inlined Go
// function does, so a goroutine that was asked to stop does so before its
// next tile, a few microseconds later, instead of whenever a signal happens
// to land between two calls.
//
//go:noinline
func callF32(f func(int, *float32, int, *float32, *float32, int), kc int, a *float32, lda int, b, c *float32, acc int) {
	f(kc, a, lda, b, c, acc)
}

//go:noinline
func callQ8(f func(int, *float32, int, *int8, *float32, int), kc int, a *float32, lda int, b *int8, c *float32, acc int) {
	f(kc, a, lda, b, c, acc)
}

// storeTile writes a computed tile into dst, applying the int8 row scale and
// the bias as separately rounded steps, exactly as the generic path does
// (the explicit float32 conversion forbids fusing them).
func (l *Linear) storeTile(dst, tile []float32, r, n, rows, cols, nr int) {
	ldd := l.Out
	for i := 0; i < rows; i++ {
		row := dst[(r+i)*ldd+n : (r+i)*ldd+n+cols]
		t := tile[i*nr : i*nr+cols]
		switch {
		case l.Q != nil && l.Bias != nil:
			s, b := l.Scale[n:n+cols], l.Bias[n:n+cols]
			for j := range row {
				row[j] = float32(t[j]*s[j]) + b[j]
			}
		case l.Q != nil:
			s := l.Scale[n : n+cols]
			for j := range row {
				row[j] = t[j] * s[j]
			}
		case l.Bias != nil:
			b := l.Bias[n : n+cols]
			for j := range row {
				row[j] = t[j] + b[j]
			}
		default:
			copy(row, t)
		}
	}
}
