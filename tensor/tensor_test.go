package tensor

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

func randomSlice(r *rand.Rand, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(r.NormFloat64())
	}
	return out
}

func naiveLinear(x, w, bias []float32, m, out, in int) []float64 {
	y := make([]float64, m*out)
	for r := 0; r < m; r++ {
		for n := 0; n < out; n++ {
			var s float64
			for k := 0; k < in; k++ {
				s += float64(x[r*in+k]) * float64(w[n*in+k])
			}
			if bias != nil {
				s += float64(bias[n])
			}
			y[r*out+n] = s
		}
	}
	return y
}

func TestLinearMatchesFloat64Reference(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, shape := range [][3]int{{1, 1, 1}, {3, 5, 7}, {4, 4, 4}, {5, 33, 17}, {70, 65, 129}, {130, 97, 64}} {
		m, out, in := shape[0], shape[1], shape[2]
		x := randomSlice(r, m*in)
		w := randomSlice(r, out*in)
		b := randomSlice(r, out)
		l, err := NewLinear(out, in, w, b)
		if err != nil {
			t.Fatal(err)
		}
		dst := make([]float32, m*out)
		l.Forward(dst, x, m, Parallel{Workers: 3})
		want := naiveLinear(x, w, b, m, out, in)
		for i := range dst {
			tol := 1e-5 * math.Sqrt(float64(in)) * (1 + math.Abs(want[i]))
			if math.Abs(float64(dst[i])-want[i]) > tol {
				t.Fatalf("shape %v element %d = %g, want %g", shape, i, dst[i], want[i])
			}
		}
	}
}

// Every element must be bit-identical whatever the worker count or row
// position, for every kernel, because routing decisions have to replay
// exactly and a question's rows move when other sequences share its batch.
func TestLinearIsDeterministicAcrossWorkersAndTiles(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	m, out, in := 131, 71, 100
	x := randomSlice(r, m*in)
	w := randomSlice(r, out*in)
	lf, _ := NewLinear(out, in, append([]float32(nil), w...), nil)
	lq, _ := NewLinear(out, in, append([]float32(nil), w...), nil)
	lq.Quantize()
	for _, kern := range Kernels() {
		for _, l := range []*Linear{lf, lq} {
			base := make([]float32, m*out)
			l.Forward(base, x, m, Parallel{Workers: 1, Kernel: kern})
			for _, workers := range []int{2, 5, 16} {
				got := make([]float32, m*out)
				l.Forward(got, x, m, Parallel{Workers: workers, Kernel: kern})
				for i := range got {
					if math.Float32bits(got[i]) != math.Float32bits(base[i]) {
						t.Fatalf("%s quantized=%v workers=%d element %d differs", kern, l.Quantized(), workers, i)
					}
				}
			}
			// A row computed alone (edge or padded tile) equals the same
			// row inside a full tile.
			for _, row := range []int{0, 5, 127, 130} {
				single := make([]float32, out)
				l.Forward(single, x[row*in:(row+1)*in], 1, Parallel{Workers: 1, Kernel: kern})
				for n := 0; n < out; n++ {
					if math.Float32bits(single[n]) != math.Float32bits(base[row*out+n]) {
						t.Fatalf("%s quantized=%v row %d col %d differs between full and edge tiles", kern, l.Quantized(), row, n)
					}
				}
			}
		}
	}
}

func TestQuantizedLinearMatchesDequantizedReference(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	m, out, in := 9, 37, 66
	x := randomSlice(r, m*in)
	w := randomSlice(r, out*in)
	w[3] = 0 // exercise an exact zero
	for i := 0; i < in; i++ {
		w[(out-1)*in+i] = 0 // an all-zero row keeps scale 1 and q 0
	}
	q, scale := QuantizeRows(w, out, in)
	deq := make([]float32, len(w))
	for n := 0; n < out; n++ {
		for k := 0; k < in; k++ {
			v := q[n*in+k]
			if v < -127 {
				t.Fatalf("q out of range: %d", v)
			}
			deq[n*in+k] = float32(float64(v) * float64(scale[n]))
			if err := math.Abs(float64(deq[n*in+k] - w[n*in+k])); err > float64(scale[n])/2+1e-7 {
				t.Fatalf("quantization error %g exceeds half a step %g", err, scale[n]/2)
			}
		}
	}
	l, err := NewQuantizedLinear(out, in, q, scale, nil)
	if err != nil {
		t.Fatal(err)
	}
	dst := make([]float32, m*out)
	l.Forward(dst, x, m, Parallel{})
	want := naiveLinear(x, deq, nil, m, out, in)
	for i := range dst {
		if math.Abs(float64(dst[i])-want[i]) > 1e-4*(1+math.Abs(want[i])) {
			t.Fatalf("element %d = %g, want %g", i, dst[i], want[i])
		}
	}
}

func TestLayerNormGELUGeGLUSoftmax(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	rows, cols := 5, 11
	x := randomSlice(r, rows*cols)
	w := randomSlice(r, cols)
	b := randomSlice(r, cols)
	dst := make([]float32, rows*cols)
	LayerNorm(dst, x, rows, cols, w, b, 1e-5, Parallel{Workers: 2})
	for row := 0; row < rows; row++ {
		var mean, variance float64
		for i := 0; i < cols; i++ {
			mean += float64(x[row*cols+i])
		}
		mean /= float64(cols)
		for i := 0; i < cols; i++ {
			d := float64(x[row*cols+i]) - mean
			variance += d * d
		}
		variance /= float64(cols)
		for i := 0; i < cols; i++ {
			want := (float64(x[row*cols+i])-mean)/math.Sqrt(variance+1e-5)*float64(w[i]) + float64(b[i])
			if math.Abs(float64(dst[row*cols+i])-want) > 1e-5 {
				t.Fatalf("layer norm [%d,%d] = %g, want %g", row, i, dst[row*cols+i], want)
			}
		}
	}

	for _, v := range []float64{-6, -1, -0.5, 0, 0.3, 1, 4} {
		want := 0.5 * v * (1 + math.Erf(v/math.Sqrt2))
		if got := GELU(float32(v)); math.Abs(float64(got)-want) > 1e-6 {
			t.Fatalf("GELU(%g) = %g, want %g", v, got, want)
		}
	}

	src := []float32{1, -2, 3, 0.5, 2, -1}
	out := make([]float32, 3)
	GeGLU(out, src, 1, 3, Parallel{})
	for i, want := range []float32{GELU(1) * 0.5, GELU(-2) * 2, GELU(3) * -1} {
		if out[i] != want {
			t.Fatalf("GeGLU[%d] = %g, want %g (act on the first half, gate is the second)", i, out[i], want)
		}
	}

	sm := []float32{1, 2, 3, -1e4}
	SoftmaxInPlace(sm)
	e := []float64{math.Exp(-2), math.Exp(-1), 1, 0}
	total := e[0] + e[1] + e[2]
	for i := range sm {
		if math.Abs(float64(sm[i])-e[i]/total) > 1e-7 {
			t.Fatalf("softmax[%d] = %g", i, sm[i])
		}
	}
}

func TestRopeMatchesRotateHalfDefinition(t *testing.T) {
	dim, theta := 8, 10000.0
	rope := NewRope(dim, theta, 4)
	x := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	got := append([]float32(nil), x...)
	rope.Apply(got, 3)
	half := dim / 2
	for i := 0; i < dim; i++ {
		f := i % half
		inv := 1 / math.Pow(theta, float64(2*f)/float64(dim))
		angle := 3 * inv
		// rotate_half(x) = [-x[half:], x[:half]]
		var rot float64
		if i < half {
			rot = -float64(x[i+half])
		} else {
			rot = float64(x[i-half])
		}
		want := float64(x[i])*math.Cos(angle) + rot*math.Sin(angle)
		if math.Abs(float64(got[i])-want) > 1e-5 {
			t.Fatalf("rope[%d] = %g, want %g", i, got[i], want)
		}
	}
	rope.Ensure(10)
	if rope.Positions() != 10 {
		t.Fatalf("positions = %d", rope.Positions())
	}
}

// BenchmarkLinear times the mmBERT-base projections (Wqkv/Wi 2304x768, WoM
// 768x1152) per kernel, float32 and int8, on one worker and on all of them.
//
//	go test -run XXX -bench Linear ./internal/layago/tensor/
func BenchmarkLinear(b *testing.B) {
	r := rand.New(rand.NewPCG(9, 10))
	for _, shape := range [][3]int{{32, 2304, 768}, {64, 2304, 768}, {256, 2304, 768}, {256, 768, 1152}} {
		for _, quant := range []bool{false, true} {
			m, out, in := shape[0], shape[1], shape[2]
			x := randomSlice(r, m*in)
			l, _ := NewLinear(out, in, randomSlice(r, out*in), nil)
			if quant {
				l.Quantize()
			}
			dst := make([]float32, m*out)
			for _, kern := range Kernels() {
				l.Prepack(kern)
				for _, workers := range []int{1, 0} {
					par := Parallel{Workers: workers, Kernel: kern}
					b.Run(fmt.Sprintf("m%d_n%d_k%d/int8=%v/%s/workers=%d", m, out, in, quant, kern, par.N()), func(b *testing.B) {
						b.SetBytes(int64(2 * m * out * in)) // reported MB/s reads as MFLOP/s
						for b.Loop() {
							l.Forward(dst, x, m, par)
						}
						b.ReportMetric(float64(2*m*out*in)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
					})
				}
			}
		}
	}
}
