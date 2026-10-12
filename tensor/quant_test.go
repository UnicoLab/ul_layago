package tensor

import (
	"math"
	"math/rand/v2"
	"testing"
)

func relativeError(orig, approx []float32) float64 {
	if len(orig) != len(approx) || len(orig) == 0 {
		return 1.0
	}
	var diffSq, origSq float64
	for i := range orig {
		d := float64(orig[i] - approx[i])
		diffSq += d * d
		origSq += float64(orig[i]) * float64(orig[i])
	}
	if origSq == 0 {
		if diffSq == 0 {
			return 0
		}
		return 1.0
	}
	return math.Sqrt(diffSq) / math.Sqrt(origSq)
}

func TestQuantizeINT8RoundtripAndPrecision(t *testing.T) {
	rng := rand.New(rand.NewPCG(100, 200))
	shapes := [][]int{
		{64},
		{32, 32},
		{128, 64},
		{15, 37},
	}

	for _, shape := range shapes {
		total := 1
		for _, d := range shape {
			total *= d
		}
		data := make([]float32, total)
		for i := range data {
			data[i] = float32(rng.NormFloat64() * 2.5)
		}

		tSrc, err := NewTensor(shape, data)
		if err != nil {
			t.Fatalf("NewTensor failed: %v", err)
		}

		q, err := QuantizeINT8(tSrc)
		if err != nil {
			t.Fatalf("QuantizeINT8 failed: %v", err)
		}

		deq, err := DequantizeINT8(q)
		if err != nil {
			t.Fatalf("DequantizeINT8 failed: %v", err)
		}

		relErr := relativeError(tSrc.Data, deq.Data)
		t.Logf("Shape %v relative error: %.4f%%", shape, relErr*100)

		// Must satisfy < 1% error
		if relErr >= 0.01 {
			t.Fatalf("Shape %v: precision loss %.4f%% exceeds 1%% threshold", shape, relErr*100)
		}
	}
}

func TestQuantizeINT8BlockRoundtripAndPrecision(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 99))
	sizes := []int{32, 64, 128, 512, 1024, 77}

	for _, size := range sizes {
		data := make([]float32, size)
		for i := range data {
			data[i] = float32(rng.NormFloat64() * 3.0)
		}

		tSrc, err := NewTensor([]int{size}, data)
		if err != nil {
			t.Fatal(err)
		}

		q, err := QuantizeINT8Block(tSrc, 32)
		if err != nil {
			t.Fatal(err)
		}

		deq, err := DequantizeINT8(q)
		if err != nil {
			t.Fatal(err)
		}

		relErr := relativeError(tSrc.Data, deq.Data)
		t.Logf("Block INT8 (size %d) relative error: %.4f%%", size, relErr*100)

		if relErr >= 0.01 {
			t.Fatalf("Block INT8 (size %d): error %.4f%% exceeds 1%% threshold", size, relErr*100)
		}
	}
}

func vectorNorm(v []float32) float64 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return math.Sqrt(sum)
}

func TestQuantizedDotProduct(t *testing.T) {
	rng := rand.New(rand.NewPCG(55, 66))
	lengths := []int{7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 256, 1000}

	for _, n := range lengths {
		aFloat := make([]float32, n)
		bFloat := make([]float32, n)
		for i := 0; i < n; i++ {
			aFloat[i] = float32(rng.NormFloat64())
			bFloat[i] = float32(rng.NormFloat64())
		}

		// Reference FP32 dot product
		var refDot float32
		for i := 0; i < n; i++ {
			refDot += aFloat[i] * bFloat[i]
		}

		tA, _ := NewTensor([]int{n}, aFloat)
		tB, _ := NewTensor([]int{n}, bFloat)

		qA, _ := QuantizeINT8(tA)
		qB, _ := QuantizeINT8(tB)

		qDot := QuantizedDotProduct(qA.Data, qB.Data, qA.Scale, qB.Scale)

		diff := math.Abs(float64(refDot - qDot))
		normA := vectorNorm(aFloat)
		normB := vectorNorm(bFloat)
		// Cauchy-Schwarz normalized relative error: |a.b - qA.qB| / (||a|| * ||b||)
		relDiff := diff / (normA * normB)

		t.Logf("Length %d: ref=%.4f, qDot=%.4f, normalized error=%.4f%%", n, refDot, qDot, relDiff*100)

		if relDiff >= 0.015 { // normalized precision loss must be < 1.5%
			t.Errorf("Length %d dot product normalized error %.4f%% is too high", n, relDiff*100)
		}
	}
}

func TestQuantizedMatMul(t *testing.T) {
	rng := rand.New(rand.NewPCG(777, 888))

	testCases := []struct {
		M, K, N    int
		blockScale bool
	}{
		{8, 128, 16, false},
		{8, 128, 16, true},
		{16, 256, 32, false},
		{16, 256, 32, true},
		{4, 64, 8, true},
	}

	for _, tc := range testCases {
		aData := make([]float32, tc.M*tc.K)
		bData := make([]float32, tc.N*tc.K) // PyTorch row-major layout [N, K]
		for i := range aData {
			aData[i] = float32(rng.NormFloat64() * 0.5)
		}
		for i := range bData {
			bData[i] = float32(rng.NormFloat64() * 0.5)
		}

		tA, err := NewTensor([]int{tc.M, tc.K}, aData)
		if err != nil {
			t.Fatal(err)
		}
		tB, err := NewTensor([]int{tc.N, tc.K}, bData)
		if err != nil {
			t.Fatal(err)
		}

		// Reference FP32 matmul
		refOut := make([]float32, tc.M*tc.N)
		for m := 0; m < tc.M; m++ {
			for n := 0; n < tc.N; n++ {
				var sum float32
				for k := 0; k < tc.K; k++ {
					sum += aData[m*tc.K+k] * bData[n*tc.K+k]
				}
				refOut[m*tc.N+n] = sum
			}
		}

		var qB *QuantizedTensor
		if tc.blockScale {
			qB, err = QuantizeINT8Block(tB, 32)
		} else {
			qB, err = QuantizeINT8(tB)
		}
		if err != nil {
			t.Fatal(err)
		}

		outTensor, err := NewTensor([]int{tc.M, tc.N}, nil)
		if err != nil {
			t.Fatal(err)
		}

		err = QuantizedMatMul(tA, qB, outTensor)
		if err != nil {
			t.Fatalf("QuantizedMatMul failed: %v", err)
		}

		relErr := relativeError(refOut, outTensor.Data)
		t.Logf("MatMul [%dx%d] x [%dx%d] (blockScale=%v) error: %.4f%%",
			tc.M, tc.K, tc.N, tc.K, tc.blockScale, relErr*100)

		if relErr >= 0.01 {
			t.Fatalf("QuantizedMatMul error %.4f%% exceeds 1%% threshold", relErr*100)
		}
	}
}

func TestBlockQuantized4BitINT4(t *testing.T) {
	rng := rand.New(rand.NewPCG(12, 34))
	n := 128
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(rng.NormFloat64() * 2.0)
	}

	tSrc, err := NewTensor([]int{n}, data)
	if err != nil {
		t.Fatal(err)
	}

	q4, err := QuantizeBlockINT4(tSrc, 32)
	if err != nil {
		t.Fatalf("QuantizeBlockINT4 failed: %v", err)
	}

	if q4.BlockSize != 32 {
		t.Errorf("expected BlockSize 32, got %d", q4.BlockSize)
	}
	if len(q4.Scales) != 4 {
		t.Errorf("expected 4 scales for 128 elements, got %d", len(q4.Scales))
	}
	if len(q4.Data) != 64 {
		t.Errorf("expected 64 packed bytes, got %d", len(q4.Data))
	}

	deq, err := q4.Dequantize()
	if err != nil {
		t.Fatalf("Dequantize failed: %v", err)
	}

	relErr := relativeError(tSrc.Data, deq.Data)
	t.Logf("INT4 block roundtrip relative error: %.2f%%", relErr*100)

	// 4-bit has 15 levels, so expected error is ~8-12%
	if relErr > 0.15 {
		t.Fatalf("INT4 error %.2f%% exceeds 15%% bound", relErr*100)
	}
}

func TestBlockQuantized4BitFP4(t *testing.T) {
	rng := rand.New(rand.NewPCG(56, 78))
	n := 256
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(rng.NormFloat64() * 2.0)
	}

	tSrc, err := NewTensor([]int{n}, data)
	if err != nil {
		t.Fatal(err)
	}

	qFP4, err := QuantizeBlockFP4(tSrc, 32)
	if err != nil {
		t.Fatalf("QuantizeBlockFP4 failed: %v", err)
	}

	if qFP4.Format != FormatFP4 {
		t.Errorf("expected FormatFP4, got %v", qFP4.Format)
	}
	if len(qFP4.Scales) != 8 {
		t.Errorf("expected 8 scales, got %d", len(qFP4.Scales))
	}

	deq, err := qFP4.Dequantize()
	if err != nil {
		t.Fatalf("Dequantize failed: %v", err)
	}

	relErr := relativeError(tSrc.Data, deq.Data)
	t.Logf("FP4 block roundtrip relative error: %.2f%%", relErr*100)

	if relErr > 0.12 {
		t.Fatalf("FP4 error %.2f%% exceeds 12%% bound", relErr*100)
	}
}

func TestBlockQuantizedMatMul4Bit(t *testing.T) {
	rng := rand.New(rand.NewPCG(999, 111))
	M, K, N := 8, 64, 16

	aData := make([]float32, M*K)
	bData := make([]float32, N*K)
	for i := range aData {
		aData[i] = float32(rng.NormFloat64())
	}
	for i := range bData {
		bData[i] = float32(rng.NormFloat64())
	}

	tA, _ := NewTensor([]int{M, K}, aData)
	tB, _ := NewTensor([]int{N, K}, bData)

	// Reference FP32 matmul
	refOut := make([]float32, M*N)
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			var sum float32
			for k := 0; k < K; k++ {
				sum += aData[m*K+k] * bData[n*K+k]
			}
			refOut[m*N+n] = sum
		}
	}

	// INT4 MatMul
	q4INT, err := QuantizeBlockINT4(tB, 32)
	if err != nil {
		t.Fatal(err)
	}
	outINT4, _ := NewTensor([]int{M, N}, nil)
	err = q4INT.MatMul(tA, outINT4)
	if err != nil {
		t.Fatalf("q4INT.MatMul failed: %v", err)
	}
	errINT4 := relativeError(refOut, outINT4.Data)
	t.Logf("INT4 MatMul relative error: %.2f%%", errINT4*100)

	// FP4 MatMul
	q4FP, err := QuantizeBlockFP4(tB, 32)
	if err != nil {
		t.Fatal(err)
	}
	outFP4, _ := NewTensor([]int{M, N}, nil)
	err = q4FP.MatMul(tA, outFP4)
	if err != nil {
		t.Fatalf("q4FP.MatMul failed: %v", err)
	}
	errFP4 := relativeError(refOut, outFP4.Data)
	t.Logf("FP4 MatMul relative error: %.2f%%", errFP4*100)

	if errINT4 > 0.15 || errFP4 > 0.15 {
		t.Fatalf("4-bit matmul relative error too high: int4=%.2f%%, fp4=%.2f%%", errINT4*100, errFP4*100)
	}
}

func TestQuantizationEdgeCases(t *testing.T) {
	// Zero tensor
	z, _ := NewTensor([]int{32}, make([]float32, 32))
	qz, err := QuantizeINT8(z)
	if err != nil {
		t.Fatal(err)
	}
	if qz.Scale != 1.0 {
		t.Errorf("expected scale 1.0 for zero tensor, got %f", qz.Scale)
	}
	deqZ, err := DequantizeINT8(qz)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range deqZ.Data {
		if v != 0 {
			t.Errorf("index %d expected 0, got %f", i, v)
		}
	}

	// Nil / Empty error handling
	_, err = QuantizeINT8(nil)
	if err == nil {
		t.Error("expected error for nil tensor")
	}
	_, err = DequantizeINT8(nil)
	if err == nil {
		t.Error("expected error for nil quantized tensor")
	}

	// Reshape
	t1, _ := NewTensor([]int{4, 4}, make([]float32, 16))
	tReshaped, err := t1.Reshape(2, 8)
	if err != nil {
		t.Fatal(err)
	}
	if tReshaped.Shape[0] != 2 || tReshaped.Shape[1] != 8 {
		t.Errorf("unexpected shape %v", tReshaped.Shape)
	}
	_, err = t1.Reshape(3, 5)
	if err == nil {
		t.Error("expected error for incompatible reshape")
	}
}

func BenchmarkQuantizedDotProduct(b *testing.B) {
	n := 1024
	a := make([]int8, n)
	c := make([]int8, n)
	for i := 0; i < n; i++ {
		a[i] = int8(i % 127)
		c[i] = int8((i * 3) % 127)
	}
	scaleA := float32(0.01)
	scaleB := float32(0.02)

	b.ResetTimer()
	b.SetBytes(int64(n * 2))
	for i := 0; i < b.N; i++ {
		_ = QuantizedDotProduct(a, c, scaleA, scaleB)
	}
}

func BenchmarkQuantizedMatMul(b *testing.B) {
	M, K, N := 64, 256, 128
	aData := make([]float32, M*K)
	bData := make([]float32, N*K)
	tA, _ := NewTensor([]int{M, K}, aData)
	tB, _ := NewTensor([]int{N, K}, bData)

	qB, _ := QuantizeINT8Block(tB, 32)
	out, _ := NewTensor([]int{M, N}, nil)

	b.ResetTimer()
	b.SetBytes(int64(M * K * N * 2))
	for i := 0; i < b.N; i++ {
		_ = QuantizedMatMul(tA, qB, out)
	}
}
