package tensor

import (
	"fmt"
	"math"
)

// LayerNorm normalises each of rows rows of width cols from x into dst:
// (x - mean) / sqrt(var + eps) * weight (+ bias). Mean and biased variance are
// accumulated in float64. bias may be nil (ModernBERT norms have none).
func LayerNorm(dst, x []float32, rows, cols int, weight, bias []float32, eps float64, par Parallel) {
	if len(weight) != cols || (bias != nil && len(bias) != cols) {
		panic(fmt.Sprintf("tensor: layer norm width %d with weight %d bias %d", cols, len(weight), len(bias)))
	}
	par.Run(rowTasks(rows), func(task int) {
		r0, r1 := rowRange(task, rows)
		for r := r0; r < r1; r++ {
			in := x[r*cols : (r+1)*cols]
			out := dst[r*cols : (r+1)*cols]
			var sum float64
			for _, v := range in {
				sum += float64(v)
			}
			mean := sum / float64(cols)
			var ss float64
			for _, v := range in {
				d := float64(v) - mean
				ss += d * d
			}
			inv := 1 / math.Sqrt(ss/float64(cols)+eps)
			for i, v := range in {
				y := (float64(v) - mean) * inv
				if bias != nil {
					out[i] = float32(y*float64(weight[i]) + float64(bias[i]))
				} else {
					out[i] = float32(y * float64(weight[i]))
				}
			}
		}
	})
}

const rowsPerTask = 16

func rowTasks(rows int) int { return (rows + rowsPerTask - 1) / rowsPerTask }

func rowRange(task, rows int) (int, int) {
	r0 := task * rowsPerTask
	return r0, min(r0+rowsPerTask, rows)
}

// GELU is the exact erf form used by PyTorch's nn.GELU() and Hugging Face's
// "gelu" activation: 0.5·x·(1 + erf(x/√2)), evaluated in float64.
func GELU(x float32) float32 {
	v := float64(x)
	return float32(0.5 * v * (1 + math.Erf(v/math.Sqrt2)))
}

// GELUInPlace applies GELU to every element.
func GELUInPlace(x []float32, par Parallel) {
	chunks := (len(x) + 4095) / 4096
	par.Run(chunks, func(task int) {
		part := x[task*4096 : min((task+1)*4096, len(x))]
		for i, v := range part {
			part[i] = GELU(v)
		}
	})
}

// ReLUInPlace clamps negative values to zero.
func ReLUInPlace(x []float32) {
	for i, v := range x {
		if v < 0 {
			x[i] = 0
		}
	}
}

// GeGLU implements ModernBertMLP's gating: the Wi projection (rows x 2·inter)
// is split into input and gate halves, and dst (rows x inter) receives
// GELU(input) · gate.
func GeGLU(dst, src []float32, rows, inter int, par Parallel) {
	par.Run(rowTasks(rows), func(task int) {
		r0, r1 := rowRange(task, rows)
		for r := r0; r < r1; r++ {
			in := src[r*2*inter : r*2*inter+inter]
			gate := src[r*2*inter+inter : (r+1)*2*inter]
			out := dst[r*inter : (r+1)*inter]
			for i := range out {
				out[i] = GELU(in[i]) * gate[i]
			}
		}
	})
}

// AddInPlace adds src to dst element-wise.
func AddInPlace(dst, src []float32) {
	src = src[:len(dst)]
	for i := range dst {
		dst[i] += src[i]
	}
}

// SoftmaxInPlace replaces v with softmax(v), subtracting the maximum and
// summing in float64.
func SoftmaxInPlace(v []float32) {
	if len(v) == 0 {
		return
	}
	maxV := v[0]
	for _, x := range v[1:] {
		if x > maxV {
			maxV = x
		}
	}
	var sum float64
	for i, x := range v {
		e := math.Exp(float64(x - maxV))
		v[i] = float32(e)
		sum += e
	}
	inv := 1 / sum
	for i := range v {
		v[i] = float32(float64(v[i]) * inv)
	}
}

// Softmax64 returns softmax(v) in float64.
func Softmax64(v []float64) []float64 {
	out := make([]float64, len(v))
	if len(v) == 0 {
		return out
	}
	maxV := v[0]
	for _, x := range v[1:] {
		maxV = math.Max(maxV, x)
	}
	var sum float64
	for i, x := range v {
		out[i] = math.Exp(x - maxV)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// Rope holds rotary position tables for one (head dim, theta) pair, computed
// the way Hugging Face's default rope does: inv_freq, positions and angles in
// float32, then cos/sin rounded to float32.
type Rope struct {
	Dim   int
	Theta float64
	inv   []float32
	cos   []float32 // pos*Dim + i, with the half-dim pattern repeated
	sin   []float32
}

// NewRope builds tables for positions [0, maxPos).
func NewRope(dim int, theta float64, maxPos int) *Rope {
	half := dim / 2
	r := &Rope{Dim: dim, Theta: theta, inv: make([]float32, half)}
	for i := 0; i < half; i++ {
		exponent := float32(2*i) / float32(dim)
		r.inv[i] = float32(1 / float64(float32(math.Pow(theta, float64(exponent)))))
	}
	r.grow(maxPos)
	return r
}

func (r *Rope) grow(maxPos int) {
	have := len(r.cos) / r.Dim
	if maxPos <= have {
		return
	}
	half := r.Dim / 2
	cos := make([]float32, maxPos*r.Dim)
	sin := make([]float32, maxPos*r.Dim)
	copy(cos, r.cos)
	copy(sin, r.sin)
	for p := have; p < maxPos; p++ {
		for i := 0; i < half; i++ {
			angle := float32(p) * r.inv[i]
			c, s := float32(math.Cos(float64(angle))), float32(math.Sin(float64(angle)))
			cos[p*r.Dim+i], cos[p*r.Dim+half+i] = c, c
			sin[p*r.Dim+i], sin[p*r.Dim+half+i] = s, s
		}
	}
	r.cos, r.sin = cos, sin
}

// Positions returns how many positions the tables cover.
func (r *Rope) Positions() int { return len(r.cos) / r.Dim }

// Apply rotates one head vector at position pos in place:
// x·cos + rotate_half(x)·sin, with rotate_half(x) = [-x2, x1]. The tables
// must already cover pos (see Positions).
func (r *Rope) Apply(x []float32, pos int) {
	half := r.Dim / 2
	cos := r.cos[pos*r.Dim : (pos+1)*r.Dim]
	sin := r.sin[pos*r.Dim : (pos+1)*r.Dim]
	x = x[:r.Dim]
	for i := 0; i < half; i++ {
		a, b := x[i], x[half+i]
		x[i] = a*cos[i] - b*sin[i]
		x[half+i] = b*cos[half+i] + a*sin[half+i]
	}
}

// Ensure extends the tables to cover maxPos positions. It is not safe to call
// concurrently with Apply.
func (r *Rope) Ensure(maxPos int) { r.grow(maxPos) }
