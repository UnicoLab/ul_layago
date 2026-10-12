package tensor

import (
	"errors"
	"fmt"
	"math"
	"slices"
)

// Tensor represents a dense multidimensional array of float32 values.
type Tensor struct {
	Data  []float32
	Shape []int
}

// NewTensor creates a new Tensor with the given shape and data.
// If data is nil, a zero-initialized buffer is allocated for product(shape) elements.
func NewTensor(shape []int, data []float32) (*Tensor, error) {
	if len(shape) == 0 {
		return nil, errors.New("tensor: shape cannot be empty")
	}
	total := 1
	for _, dim := range shape {
		if dim <= 0 {
			return nil, fmt.Errorf("tensor: invalid non-positive dimension %d", dim)
		}
		total *= dim
	}
	if data != nil {
		if len(data) != total {
			return nil, fmt.Errorf("tensor: data length %d mismatch shape %v (%d)", len(data), shape, total)
		}
	} else {
		data = make([]float32, total)
	}
	return &Tensor{
		Data:  data,
		Shape: slices.Clone(shape),
	}, nil
}

// NumElements returns the total count of elements in the tensor.
func (t *Tensor) NumElements() int {
	if t == nil || len(t.Shape) == 0 {
		return 0
	}
	total := 1
	for _, d := range t.Shape {
		total *= d
	}
	return total
}

// Reshape returns a shallow-copy Tensor with a new shape if element count matches.
func (t *Tensor) Reshape(newShape ...int) (*Tensor, error) {
	if t == nil {
		return nil, errors.New("tensor: cannot reshape nil tensor")
	}
	total := 1
	for _, dim := range newShape {
		if dim <= 0 {
			return nil, fmt.Errorf("tensor: invalid dimension %d in reshape", dim)
		}
		total *= dim
	}
	if total != len(t.Data) {
		return nil, fmt.Errorf("tensor: cannot reshape %d elements into shape %v (%d)", len(t.Data), newShape, total)
	}
	return &Tensor{
		Data:  t.Data,
		Shape: slices.Clone(newShape),
	}, nil
}

// QuantizedTensor holds INT8 quantized weights/activations with scaling parameters.
type QuantizedTensor struct {
	Data        []int8
	Shape       []int
	Scale       float32
	ZeroPoint   int8
	BlockScales []float32
	BlockSize   int
}

// QuantizeINT8 quantizes a float32 tensor to symmetric INT8 with a single tensor-wide scale.
func QuantizeINT8(src *Tensor) (*QuantizedTensor, error) {
	if src == nil || len(src.Data) == 0 {
		return nil, errors.New("tensor: cannot quantize nil or empty tensor")
	}

	var maxAbs float32
	for _, v := range src.Data {
		if a := float32(math.Abs(float64(v))); a > maxAbs {
			maxAbs = a
		}
	}

	qData := make([]int8, len(src.Data))
	if maxAbs == 0 {
		return &QuantizedTensor{
			Data:      qData,
			Shape:     slices.Clone(src.Shape),
			Scale:     1.0,
			ZeroPoint: 0,
		}, nil
	}

	scale := float32(float64(maxAbs) / 127.0)
	invScale := 1.0 / float64(scale)

	for i, v := range src.Data {
		r := math.RoundToEven(float64(v) * invScale)
		if r > 127 {
			r = 127
		} else if r < -127 {
			r = -127
		}
		qData[i] = int8(r)
	}

	return &QuantizedTensor{
		Data:      qData,
		Shape:     slices.Clone(src.Shape),
		Scale:     scale,
		ZeroPoint: 0,
	}, nil
}

// QuantizeINT8Block quantizes a float32 tensor into INT8 with block scaling (e.g. 32 elements per block).
func QuantizeINT8Block(src *Tensor, blockSize int) (*QuantizedTensor, error) {
	if src == nil || len(src.Data) == 0 {
		return nil, errors.New("tensor: cannot quantize nil or empty tensor")
	}
	if blockSize <= 0 {
		blockSize = 32
	}

	n := len(src.Data)
	numBlocks := (n + blockSize - 1) / blockSize
	qData := make([]int8, n)
	blockScales := make([]float32, numBlocks)

	for b := 0; b < numBlocks; b++ {
		start := b * blockSize
		end := start + blockSize
		if end > n {
			end = n
		}

		var maxAbs float32
		for i := start; i < end; i++ {
			if a := float32(math.Abs(float64(src.Data[i]))); a > maxAbs {
				maxAbs = a
			}
		}

		if maxAbs == 0 {
			blockScales[b] = 1.0
			continue
		}

		scale := float32(float64(maxAbs) / 127.0)
		blockScales[b] = scale
		invScale := 1.0 / float64(scale)

		for i := start; i < end; i++ {
			r := math.RoundToEven(float64(src.Data[i]) * invScale)
			if r > 127 {
				r = 127
			} else if r < -127 {
				r = -127
			}
			qData[i] = int8(r)
		}
	}

	return &QuantizedTensor{
		Data:        qData,
		Shape:       slices.Clone(src.Shape),
		Scale:       blockScales[0],
		ZeroPoint:   0,
		BlockScales: blockScales,
		BlockSize:   blockSize,
	}, nil
}

// DequantizeINT8 converts an INT8 QuantizedTensor back to float32 Tensor.
func DequantizeINT8(q *QuantizedTensor) (*Tensor, error) {
	if q == nil || len(q.Data) == 0 {
		return nil, errors.New("tensor: cannot dequantize nil or empty quantized tensor")
	}

	n := len(q.Data)
	out := make([]float32, n)

	if len(q.BlockScales) > 0 && q.BlockSize > 0 {
		bs := q.BlockSize
		numBlocks := len(q.BlockScales)
		for b := 0; b < numBlocks; b++ {
			start := b * bs
			end := start + bs
			if end > n {
				end = n
			}
			scale := q.BlockScales[b]
			for i := start; i < end; i++ {
				out[i] = float32(q.Data[i]) * scale
			}
		}
	} else {
		scale := q.Scale
		zp := float32(q.ZeroPoint)
		for i, v := range q.Data {
			out[i] = (float32(v) - zp) * scale
		}
	}

	return &Tensor{
		Data:  out,
		Shape: slices.Clone(q.Shape),
	}, nil
}

// QuantizedDotProduct computes the dot product of two INT8 slices with 8-way unrolled loops
// for compiler SIMD auto-vectorization, scaled by (scaleA * scaleB).
func QuantizedDotProduct(a, b []int8, scaleA, scaleB float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}

	var acc0, acc1, acc2, acc3 int32
	var acc4, acc5, acc6, acc7 int32

	i := 0
	limit := n - 7
	for ; i < limit; i += 8 {
		acc0 += int32(a[i]) * int32(b[i])
		acc1 += int32(a[i+1]) * int32(b[i+1])
		acc2 += int32(a[i+2]) * int32(b[i+2])
		acc3 += int32(a[i+3]) * int32(b[i+3])
		acc4 += int32(a[i+4]) * int32(b[i+4])
		acc5 += int32(a[i+5]) * int32(b[i+5])
		acc6 += int32(a[i+6]) * int32(b[i+6])
		acc7 += int32(a[i+7]) * int32(b[i+7])
	}

	acc := (acc0 + acc1 + acc2 + acc3) + (acc4 + acc5 + acc6 + acc7)
	for ; i < n; i++ {
		acc += int32(a[i]) * int32(b[i])
	}

	return float32(acc) * (scaleA * scaleB)
}

// QuantizedMatMul multiplies float32 input tensor a by INT8 weights tensor b into out,
// supporting block scaling and 8-way unrolled inner kernels.
func QuantizedMatMul(a *Tensor, b *QuantizedTensor, out *Tensor) error {
	if a == nil || b == nil || out == nil {
		return errors.New("tensor: matmul arguments cannot be nil")
	}
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		return fmt.Errorf("tensor: matmul requires 2D tensors, got a: %v, b: %v", a.Shape, b.Shape)
	}

	M := a.Shape[0]
	K := a.Shape[1]

	var N int
	var weightsRowMajor bool

	if b.Shape[0] == K {
		N = b.Shape[1]
		weightsRowMajor = false
	} else if b.Shape[1] == K {
		N = b.Shape[0]
		weightsRowMajor = true
	} else {
		return fmt.Errorf("tensor: matmul dimension mismatch: a has shape %v (K=%d), b has shape %v", a.Shape, K, b.Shape)
	}

	reqSize := M * N
	if len(out.Data) < reqSize {
		return fmt.Errorf("tensor: out data buffer size %d is less than required %d (%dx%d)", len(out.Data), reqSize, M, N)
	}
	out.Shape = []int{M, N}

	hasBlockScaling := len(b.BlockScales) > 0 && b.BlockSize > 0

	if hasBlockScaling {
		bs := b.BlockSize
		blocksK := (K + bs - 1) / bs

		if weightsRowMajor {
			// B is [N, K], row-major weights
			for m := 0; m < M; m++ {
				aRow := a.Data[m*K : (m+1)*K]
				for n := 0; n < N; n++ {
					bRow := b.Data[n*K : (n+1)*K]
					var sum float32
					for blk := 0; blk < blocksK; blk++ {
						k0 := blk * bs
						k1 := k0 + bs
						if k1 > K {
							k1 = K
						}
						scale := b.BlockScales[n*blocksK+blk]

						var acc0, acc1, acc2, acc3 float32
						var acc4, acc5, acc6, acc7 float32

						k := k0
						limit := k1 - 7
						for ; k < limit; k += 8 {
							acc0 += aRow[k] * float32(bRow[k])
							acc1 += aRow[k+1] * float32(bRow[k+1])
							acc2 += aRow[k+2] * float32(bRow[k+2])
							acc3 += aRow[k+3] * float32(bRow[k+3])
							acc4 += aRow[k+4] * float32(bRow[k+4])
							acc5 += aRow[k+5] * float32(bRow[k+5])
							acc6 += aRow[k+6] * float32(bRow[k+6])
							acc7 += aRow[k+7] * float32(bRow[k+7])
						}
						bAcc := (acc0 + acc1 + acc2 + acc3) + (acc4 + acc5 + acc6 + acc7)
						for ; k < k1; k++ {
							bAcc += aRow[k] * float32(bRow[k])
						}
						sum += bAcc * scale
					}
					out.Data[m*N+n] = sum
				}
			}
		} else {
			// B is [K, N], column-major weights
			for m := 0; m < M; m++ {
				aRow := a.Data[m*K : (m+1)*K]
				for n := 0; n < N; n++ {
					var sum float32
					for blk := 0; blk < blocksK; blk++ {
						k0 := blk * bs
						k1 := k0 + bs
						if k1 > K {
							k1 = K
						}
						scale := b.BlockScales[blk*N+n]

						var bAcc float32
						for k := k0; k < k1; k++ {
							bAcc += aRow[k] * float32(b.Data[k*N+n])
						}
						sum += bAcc * scale
					}
					out.Data[m*N+n] = sum
				}
			}
		}
	} else {
		// Single scale
		scale := b.Scale
		zp := b.ZeroPoint

		if weightsRowMajor {
			for m := 0; m < M; m++ {
				aRow := a.Data[m*K : (m+1)*K]
				for n := 0; n < N; n++ {
					bRow := b.Data[n*K : (n+1)*K]
					var acc0, acc1, acc2, acc3 float32
					var acc4, acc5, acc6, acc7 float32

					k := 0
					limit := K - 7
					for ; k < limit; k += 8 {
						acc0 += aRow[k] * float32(bRow[k]-zp)
						acc1 += aRow[k+1] * float32(bRow[k+1]-zp)
						acc2 += aRow[k+2] * float32(bRow[k+2]-zp)
						acc3 += aRow[k+3] * float32(bRow[k+3]-zp)
						acc4 += aRow[k+4] * float32(bRow[k+4]-zp)
						acc5 += aRow[k+5] * float32(bRow[k+5]-zp)
						acc6 += aRow[k+6] * float32(bRow[k+6]-zp)
						acc7 += aRow[k+7] * float32(bRow[k+7]-zp)
					}
					acc := (acc0 + acc1 + acc2 + acc3) + (acc4 + acc5 + acc6 + acc7)
					for ; k < K; k++ {
						acc += aRow[k] * float32(bRow[k]-zp)
					}
					out.Data[m*N+n] = acc * scale
				}
			}
		} else {
			for m := 0; m < M; m++ {
				aRow := a.Data[m*K : (m+1)*K]
				for n := 0; n < N; n++ {
					var acc float32
					for k := 0; k < K; k++ {
						acc += aRow[k] * float32(b.Data[k*N+n]-zp)
					}
					out.Data[m*N+n] = acc * scale
				}
			}
		}
	}

	return nil
}

// QuantFormat represents the 4-bit quantization encoding format.
type QuantFormat int

const (
	// FormatINT4 represents 4-bit signed integer quantization.
	FormatINT4 QuantFormat = iota
	// FormatFP4 represents 4-bit floating point (E2M1) quantization.
	FormatFP4
)

// BlockQuantized4Bit stores 4-bit block-quantized weights (per 32 elements).
// Two 4-bit values are packed into each byte (low 4 bits first).
type BlockQuantized4Bit struct {
	Data      []byte
	Scales    []float32
	Shape     []int
	BlockSize int
	Format    QuantFormat
}

// QuantizeBlockINT4 quantizes a float32 tensor to 4-bit signed integers (-7 to 7)
// with block scaling (default blockSize = 32 elements).
func QuantizeBlockINT4(src *Tensor, blockSize int) (*BlockQuantized4Bit, error) {
	if src == nil || len(src.Data) == 0 {
		return nil, errors.New("tensor: cannot quantize nil or empty tensor")
	}
	if blockSize <= 0 {
		blockSize = 32
	}

	n := len(src.Data)
	numBlocks := (n + blockSize - 1) / blockSize
	packedBytes := (n + 1) / 2
	data := make([]byte, packedBytes)
	scales := make([]float32, numBlocks)

	for b := 0; b < numBlocks; b++ {
		start := b * blockSize
		end := start + blockSize
		if end > n {
			end = n
		}

		var maxAbs float32
		for i := start; i < end; i++ {
			if a := float32(math.Abs(float64(src.Data[i]))); a > maxAbs {
				maxAbs = a
			}
		}

		if maxAbs == 0 {
			scales[b] = 1.0
			continue
		}

		scale := float32(float64(maxAbs) / 7.0)
		scales[b] = scale
		invScale := 1.0 / float64(scale)

		for i := start; i < end; i++ {
			r := math.RoundToEven(float64(src.Data[i]) * invScale)
			if r > 7 {
				r = 7
			} else if r < -7 {
				r = -7
			}
			q := int8(r)
			byteIdx := i / 2
			nibble := byte(int(q) & 0x0F)
			if i%2 == 0 {
				data[byteIdx] |= nibble
			} else {
				data[byteIdx] |= (nibble << 4)
			}
		}
	}

	return &BlockQuantized4Bit{
		Data:      data,
		Scales:    scales,
		Shape:     slices.Clone(src.Shape),
		BlockSize: blockSize,
		Format:    FormatINT4,
	}, nil
}

// fp4LUT contains the 16 exact values of FP4 (E2M1):
// Positive magnitudes: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
// Negative magnitudes: 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
var fp4LUT = [16]float32{
	0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
	0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
}

// QuantizeBlockFP4 quantizes a float32 tensor to FP4 (E2M1) with block scaling (default blockSize = 32).
func QuantizeBlockFP4(src *Tensor, blockSize int) (*BlockQuantized4Bit, error) {
	if src == nil || len(src.Data) == 0 {
		return nil, errors.New("tensor: cannot quantize nil or empty tensor")
	}
	if blockSize <= 0 {
		blockSize = 32
	}

	n := len(src.Data)
	numBlocks := (n + blockSize - 1) / blockSize
	packedBytes := (n + 1) / 2
	data := make([]byte, packedBytes)
	scales := make([]float32, numBlocks)

	for b := 0; b < numBlocks; b++ {
		start := b * blockSize
		end := start + blockSize
		if end > n {
			end = n
		}

		var maxAbs float32
		for i := start; i < end; i++ {
			if a := float32(math.Abs(float64(src.Data[i]))); a > maxAbs {
				maxAbs = a
			}
		}

		if maxAbs == 0 {
			scales[b] = 1.0
			continue
		}

		// Maximum representable value in FP4 (E2M1) is 6.0
		scale := float32(float64(maxAbs) / 6.0)
		scales[b] = scale
		invScale := 1.0 / float64(scale)

		for i := start; i < end; i++ {
			v := float32(float64(src.Data[i]) * invScale)
			// Find nearest FP4 index
			bestIdx := 0
			bestDiff := float32(math.Abs(float64(v - fp4LUT[0])))
			for code := 1; code < 16; code++ {
				diff := float32(math.Abs(float64(v - fp4LUT[code])))
				if diff < bestDiff {
					bestDiff = diff
					bestIdx = code
				}
			}

			byteIdx := i / 2
			nibble := uint8(bestIdx) & 0x0F
			if i%2 == 0 {
				data[byteIdx] |= nibble
			} else {
				data[byteIdx] |= (nibble << 4)
			}
		}
	}

	return &BlockQuantized4Bit{
		Data:      data,
		Scales:    scales,
		Shape:     slices.Clone(src.Shape),
		BlockSize: blockSize,
		Format:    FormatFP4,
	}, nil
}

func decodeINT4Nibble(byteVal byte, low bool) int8 {
	var v int
	if low {
		v = int(byteVal & 0x0F)
	} else {
		v = int(byteVal >> 4)
	}
	if v >= 8 {
		v -= 16
	}
	return int8(v)
}

// Dequantize converts a BlockQuantized4Bit tensor back to float32 Tensor.
func (b *BlockQuantized4Bit) Dequantize() (*Tensor, error) {
	if b == nil || len(b.Data) == 0 {
		return nil, errors.New("tensor: cannot dequantize nil or empty 4-bit tensor")
	}

	total := 1
	for _, dim := range b.Shape {
		total *= dim
	}

	out := make([]float32, total)
	bs := b.BlockSize
	if bs <= 0 {
		bs = 32
	}

	if b.Format == FormatINT4 {
		for i := 0; i < total; i++ {
			byteVal := b.Data[i/2]
			qVal := decodeINT4Nibble(byteVal, i%2 == 0)
			blk := i / bs
			out[i] = float32(qVal) * b.Scales[blk]
		}
	} else {
		// FP4 format
		for i := 0; i < total; i++ {
			byteVal := b.Data[i/2]
			var code uint8
			if i%2 == 0 {
				code = byteVal & 0x0F
			} else {
				code = (byteVal >> 4) & 0x0F
			}
			blk := i / bs
			out[i] = fp4LUT[code] * b.Scales[blk]
		}
	}

	return &Tensor{
		Data:  out,
		Shape: slices.Clone(b.Shape),
	}, nil
}

// QuantizedMatMul4Bit multiplies float32 input a by BlockQuantized4Bit weights b into out.
func QuantizedMatMul4Bit(a *Tensor, b *BlockQuantized4Bit, out *Tensor) error {
	if a == nil || b == nil || out == nil {
		return errors.New("tensor: matmul arguments cannot be nil")
	}
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		return fmt.Errorf("tensor: matmul requires 2D tensors, got a: %v, b: %v", a.Shape, b.Shape)
	}

	M := a.Shape[0]
	K := a.Shape[1]

	var N int
	var weightsRowMajor bool

	if b.Shape[0] == K {
		N = b.Shape[1]
		weightsRowMajor = false
	} else if b.Shape[1] == K {
		N = b.Shape[0]
		weightsRowMajor = true
	} else {
		return fmt.Errorf("tensor: matmul dimension mismatch: a has shape %v (K=%d), b has shape %v", a.Shape, K, b.Shape)
	}

	reqSize := M * N
	if len(out.Data) < reqSize {
		return fmt.Errorf("tensor: out data buffer size %d is less than required %d", len(out.Data), reqSize)
	}
	out.Shape = []int{M, N}

	bs := b.BlockSize
	if bs <= 0 {
		bs = 32
	}
	blocksK := (K + bs - 1) / bs

	if b.Format == FormatINT4 {
		if weightsRowMajor {
			for m := 0; m < M; m++ {
				aRow := a.Data[m*K : (m+1)*K]
				for n := 0; n < N; n++ {
					var sum float32
					rowOffset := n * K
					for blk := 0; blk < blocksK; blk++ {
						k0 := blk * bs
						k1 := k0 + bs
						if k1 > K {
							k1 = K
						}
						scale := b.Scales[n*blocksK+blk]

						var bAcc float32
						for k := k0; k < k1; k++ {
							elemIdx := rowOffset + k
							byteVal := b.Data[elemIdx/2]
							qVal := decodeINT4Nibble(byteVal, elemIdx%2 == 0)
							bAcc += aRow[k] * float32(qVal)
						}
						sum += bAcc * scale
					}
					out.Data[m*N+n] = sum
				}
			}
		} else {
			for m := 0; m < M; m++ {
				aRow := a.Data[m*K : (m+1)*K]
				for n := 0; n < N; n++ {
					var sum float32
					for blk := 0; blk < blocksK; blk++ {
						k0 := blk * bs
						k1 := k0 + bs
						if k1 > K {
							k1 = K
						}
						scale := b.Scales[blk*N+n]

						var bAcc float32
						for k := k0; k < k1; k++ {
							elemIdx := k*N + n
							byteVal := b.Data[elemIdx/2]
							qVal := decodeINT4Nibble(byteVal, elemIdx%2 == 0)
							bAcc += aRow[k] * float32(qVal)
						}
						sum += bAcc * scale
					}
					out.Data[m*N+n] = sum
				}
			}
		}
	} else {
		// FP4 format
		if weightsRowMajor {
			for m := 0; m < M; m++ {
				aRow := a.Data[m*K : (m+1)*K]
				for n := 0; n < N; n++ {
					var sum float32
					rowOffset := n * K
					for blk := 0; blk < blocksK; blk++ {
						k0 := blk * bs
						k1 := k0 + bs
						if k1 > K {
							k1 = K
						}
						scale := b.Scales[n*blocksK+blk]

						var bAcc float32
						for k := k0; k < k1; k++ {
							elemIdx := rowOffset + k
							byteVal := b.Data[elemIdx/2]
							var code uint8
							if elemIdx%2 == 0 {
								code = byteVal & 0x0F
							} else {
								code = (byteVal >> 4) & 0x0F
							}
							bAcc += aRow[k] * fp4LUT[code]
						}
						sum += bAcc * scale
					}
					out.Data[m*N+n] = sum
				}
			}
		} else {
			for m := 0; m < M; m++ {
				aRow := a.Data[m*K : (m+1)*K]
				for n := 0; n < N; n++ {
					var sum float32
					for blk := 0; blk < blocksK; blk++ {
						k0 := blk * bs
						k1 := k0 + bs
						if k1 > K {
							k1 = K
						}
						scale := b.Scales[blk*N+n]

						var bAcc float32
						for k := k0; k < k1; k++ {
							elemIdx := k*N + n
							byteVal := b.Data[elemIdx/2]
							var code uint8
							if elemIdx%2 == 0 {
								code = byteVal & 0x0F
							} else {
								code = (byteVal >> 4) & 0x0F
							}
							bAcc += aRow[k] * fp4LUT[code]
						}
						sum += bAcc * scale
					}
					out.Data[m*N+n] = sum
				}
			}
		}
	}

	return nil
}

// MatMul is a convenience method on BlockQuantized4Bit.
func (b *BlockQuantized4Bit) MatMul(a *Tensor, out *Tensor) error {
	return QuantizedMatMul4Bit(a, b, out)
}
