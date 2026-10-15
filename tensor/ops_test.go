package tensor

import (
	"math"
	"testing"
)

func TestSwiGLUMatchesReference(t *testing.T) {
	rows, inter := 5, 4
	src := make([]float32, rows*2*inter)
	for i := range src {
		src[i] = float32(i)*0.25 - 2.0
	}
	dst := make([]float32, rows*inter)
	SwiGLU(dst, src, rows, inter, Parallel{Workers: 2})
	for r := 0; r < rows; r++ {
		for j := 0; j < inter; j++ {
			u := float64(src[r*2*inter+j])
			v := float64(src[r*2*inter+inter+j])
			silu := u / (1 + math.Exp(-u))
			want := float32(silu) * float32(v)
			got := dst[r*inter+j]
			if math.Abs(float64(got-want)) > 1e-5 {
				t.Fatalf("row %d col %d: got %g, want %g", r, j, got, want)
			}
		}
	}
}

func TestSiLUInPlaceMatchesSiLU(t *testing.T) {
	in := []float32{-3.5, -1.0, 0, 0.5, 2.25}
	got := append([]float32(nil), in...)
	SiLUInPlace(got, Parallel{Workers: 2})
	for i, x := range in {
		want := SiLU(x)
		if math.Float32bits(got[i]) != math.Float32bits(want) {
			t.Fatalf("index %d: got %g, want %g", i, got[i], want)
		}
	}
}

func TestSwishAndAnalyticalGradients(t *testing.T) {
	xs := []float32{-4.0, -2.5, -1.2784645, -1.0, -0.5, -0.1, 0.0, 0.25, 1.0, 3.0}

	// 1. Beta = 1 coincides bit-for-bit with SiLU and SiLUGrad.
	for _, x := range xs {
		if math.Float32bits(Swish(x, 1)) != math.Float32bits(SiLU(x)) {
			t.Fatalf("Swish(%g, 1) = %g, want SiLU %g", x, Swish(x, 1), SiLU(x))
		}
		if math.Float32bits(SwishGrad(x, 1)) != math.Float32bits(SiLUGrad(x)) {
			t.Fatalf("SwishGrad(%g, 1) = %g, want SiLUGrad %g", x, SwishGrad(x, 1), SiLUGrad(x))
		}
	}

	// 2. SwishInPlace matches Swish bit-for-bit.
	got := append([]float32(nil), xs...)
	SwishInPlace(got, 1.5, Parallel{Workers: 2})
	for i, x := range xs {
		want := Swish(x, 1.5)
		if math.Float32bits(got[i]) != math.Float32bits(want) {
			t.Fatalf("SwishInPlace[%d]: got %g, want %g", i, got[i], want)
		}
	}

	// 3. Non-monotonic negative dip: for x in (-2, 0), Swish(x, 1) < 0,
	// with minimum near x* ≈ -1.2784645 (where Swish(x*, 1) ≈ -0.2784645 and derivative ≈ 0).
	for _, x := range []float32{-1.8, -1.2784645, -0.7, -0.2} {
		if y := Swish(x, 1); !(y < 0) {
			t.Fatalf("expected negative dip Swish(%g, 1) < 0, got %g", x, y)
		}
	}
	minX := float32(-1.2784645)
	if minVal := Swish(minX, 1); math.Abs(float64(minVal)-(-0.2784645)) > 1e-5 {
		t.Fatalf("Swish(%g, 1) = %g, want ≈ -0.2784645", minX, minVal)
	}
	if gradAtMin := SiLUGrad(minX); math.Abs(float64(gradAtMin)) > 1e-5 {
		t.Fatalf("SiLUGrad(%g) = %g, want ≈ 0 at stationary point", minX, gradAtMin)
	}
	// Smooth non-zero gradient on negative inputs away from the single stationary point
	// (avoids ReLU's dead zero-gradient regime for x < 0).
	if gNeg := SiLUGrad(-0.5); !(gNeg > 0.05) {
		t.Fatalf("expected positive gradient at x=-0.5, got %g", gNeg)
	}
	if gNeg2 := SiLUGrad(-2.0); !(gNeg2 < -0.05) {
		t.Fatalf("expected negative gradient in dip recovery at x=-2.0, got %g", gNeg2)
	}

	// 4. Finite-difference verification of SwishGrad across multiple β values.
	const h = 1e-4
	for _, beta := range []float32{0.5, 1.0, 1.702, 2.5} {
		for _, x := range []float32{-3.2, -1.8, -0.9, -0.3, 0.0, 0.4, 1.5, 2.8} {
			xp := float64(x) + h
			xm := float64(x) - h
			b := float64(beta)
			fp := xp / (1 + math.Exp(-b*xp))
			fm := xm / (1 + math.Exp(-b*xm))
			fd := (fp - fm) / (2 * h)
			ana := float64(SwishGrad(x, beta))
			relErr := math.Abs(ana-fd) / math.Max(1e-4, math.Abs(fd))
			if relErr > 1e-4 {
				t.Fatalf("SwishGrad(x=%g, beta=%g) = %g, finite-diff = %g (relErr=%g)", x, beta, ana, fd, relErr)
			}
		}
	}
}

func TestSwiGLUBackwardFiniteDiff(t *testing.T) {
	rows, inter := 4, 5
	src := make([]float32, rows*2*inter)
	for i := range src {
		src[i] = float32((i%11)-5) * 0.35
	}
	dDst := make([]float32, rows*inter)
	for i := range dDst {
		dDst[i] = float32((i%7)-3) * 0.25
	}
	dSrc := make([]float32, rows*2*inter)
	SwiGLUBackward(dSrc, dDst, src, rows, inter, Parallel{Workers: 2})

	const eps = float32(1.0 / 256.0)
	dstPlus := make([]float32, rows*inter)
	dstMinus := make([]float32, rows*inter)
	for idx := range src {
		orig := src[idx]
		pPlus := orig + eps
		pMinus := orig - eps
		step := float64(pPlus) - float64(pMinus)
		src[idx] = pPlus
		SwiGLU(dstPlus, src, rows, inter, Parallel{Workers: 1})
		src[idx] = pMinus
		SwiGLU(dstMinus, src, rows, inter, Parallel{Workers: 1})
		src[idx] = orig

		var numGrad float64
		for j := range dDst {
			numGrad += float64(dDst[j]) * (float64(dstPlus[j]) - float64(dstMinus[j])) / step
		}
		ana := float64(dSrc[idx])
		denom := math.Max(1e-3, math.Max(math.Abs(ana), math.Abs(numGrad)))
		if math.Abs(ana-numGrad)/denom > 1e-4 {
			t.Fatalf("dSrc[%d] = %g, finite-diff = %g", idx, ana, numGrad)
		}
	}
}

func TestEffectiveRankCollapsedAndOrthogonal(t *testing.T) {
	// Collapsed collinear / rank-1 vectors -> EffectiveRank == 1.0
	collapsed := [][]float32{
		{1, 2, 3, 4},
		{2, 4, 6, 8},
		{3, 6, 9, 12},
		{4, 8, 12, 16},
	}
	if er := EffectiveRank(collapsed); math.Abs(er-1.0) > 1e-5 {
		t.Fatalf("collapsed collinear EffectiveRank = %g, want 1.0", er)
	}

	// Collapsed identical vectors -> EffectiveRank == 1.0
	identical := [][]float32{
		{0.5, -1.5, 2.0},
		{0.5, -1.5, 2.0},
		{0.5, -1.5, 2.0},
	}
	if er := EffectiveRank(identical); math.Abs(er-1.0) > 1e-5 {
		t.Fatalf("identical EffectiveRank = %g, want 1.0", er)
	}

	// Orthogonal basis vectors (N <= D) -> EffectiveRank == K
	ortho := [][]float32{
		{1, 0, 0, 0, 0},
		{0, 1, 0, 0, 0},
		{0, 0, 1, 0, 0},
		{0, 0, 0, 1, 0},
	}
	if er := EffectiveRank(ortho); math.Abs(er-4.0) > 1e-5 {
		t.Fatalf("orthogonal 4-axis EffectiveRank = %g, want 4.0", er)
	}

	// Symmetric isotropic orthogonal vectors (D < N) -> EffectiveRank == D
	isotropic := [][]float32{
		{1, 0, 0},
		{-1, 0, 0},
		{0, 1, 0},
		{0, -1, 0},
		{0, 0, 1},
		{0, 0, -1},
	}
	if er := EffectiveRank(isotropic); math.Abs(er-3.0) > 1e-5 {
		t.Fatalf("isotropic 3-axis (N=6, D=3) EffectiveRank = %g, want 3.0", er)
	}
}
