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
