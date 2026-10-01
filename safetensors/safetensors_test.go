package safetensors

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func sampleFile(t *testing.T) []byte {
	t.Helper()
	var buf bytes.Buffer
	half := make([]byte, 4)
	binary.LittleEndian.PutUint16(half, 0x3c00)     // 1.0
	binary.LittleEndian.PutUint16(half[2:], 0xc000) // -2.0
	bf := make([]byte, 4)
	binary.LittleEndian.PutUint16(bf, 0x3f80)     // 1.0
	binary.LittleEndian.PutUint16(bf[2:], 0x4049) // 3.140625
	err := Write(&buf, []NamedTensor{
		Float32Tensor("a.weight", []int{2, 3}, []float32{1, 2, 3, 4, 5, 6}),
		{Name: "b.half", DType: F16, Shape: []int{2}, Data: half},
		{Name: "c.bf16", DType: BF16, Shape: []int{2}, Data: bf},
		Int8Tensor("d.q", []int{3}, []int8{-127, 0, 127}),
		Float32Tensor("e.scalar", []int{}, []float32{0.5}),
	}, map[string]string{"format": "pt"})
	if err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestRoundTripAndConversions(t *testing.T) {
	f, err := Parse(sampleFile(t))
	if err != nil {
		t.Fatal(err)
	}
	if got := f.Header.Metadata["format"]; got != "pt" {
		t.Fatalf("metadata = %q", got)
	}
	want := map[string][]float32{
		"a.weight": {1, 2, 3, 4, 5, 6},
		"b.half":   {1, -2},
		"c.bf16":   {1, 3.140625},
		"e.scalar": {0.5},
	}
	for name, values := range want {
		tensor, err := f.Tensor(name)
		if err != nil {
			t.Fatal(err)
		}
		got, err := tensor.Float32()
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, values) {
			t.Fatalf("%s = %v, want %v", name, got, values)
		}
	}
	q, err := f.Tensor("d.q")
	if err != nil {
		t.Fatal(err)
	}
	ints, err := q.Int8()
	if err != nil || !reflect.DeepEqual(ints, []int8{-127, 0, 127}) {
		t.Fatalf("int8 = %v, %v", ints, err)
	}
	if _, err := q.Float32(); err == nil {
		t.Fatal("I8 must not convert through Float32")
	}
	if _, err := f.Tensor("missing"); err == nil {
		t.Fatal("missing tensor must error")
	}
}

func TestFloat16Conversion(t *testing.T) {
	cases := map[uint16]float32{
		0x0000: 0,
		0x8000: float32(math.Copysign(0, -1)),
		0x3c00: 1,
		0x3555: 0.333251953125,
		0x7bff: 65504,
		0x0001: 5.960464477539063e-08, // smallest subnormal
		0x03ff: 6.097555160522461e-05, // largest subnormal
		0x0400: 6.103515625e-05,       // smallest normal
		0xfc00: float32(math.Inf(-1)),
	}
	for bits, want := range cases {
		got := Float16ToFloat32(bits)
		if math.Float32bits(got) != math.Float32bits(want) {
			t.Fatalf("Float16ToFloat32(%#04x) = %g, want %g", bits, got, want)
		}
	}
	if v := Float16ToFloat32(0x7e00); !math.IsNaN(float64(v)) {
		t.Fatalf("NaN not preserved: %v", v)
	}
	if v := BFloat16ToFloat32(0xbf80); v != -1 {
		t.Fatalf("BFloat16ToFloat32 = %v", v)
	}
}

func TestMappedAndReaderAtPathsAgree(t *testing.T) {
	data := sampleFile(t)
	path := filepath.Join(t.TempDir(), "model.safetensors")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	mapped, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = mapped.Close() }()
	lazy, err := OpenReaderAt(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(mapped.Header.Names(), lazy.Header.Names()) {
		t.Fatal("tensor names differ between load paths")
	}
	for _, name := range mapped.Header.Names() {
		a, err := mapped.Tensor(name)
		if err != nil {
			t.Fatal(err)
		}
		b, err := lazy.Tensor(name)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(a.Data, b.Data) || !reflect.DeepEqual(a.Shape, b.Shape) || a.DType != b.DType {
			t.Fatalf("tensor %s differs between mmap and ReaderAt paths", name)
		}
	}
}

func rawFile(header string, buffer int) []byte {
	var out bytes.Buffer
	var prefix [8]byte
	binary.LittleEndian.PutUint64(prefix[:], uint64(len(header)))
	out.Write(prefix[:])
	out.WriteString(header)
	out.Write(make([]byte, buffer))
	return out.Bytes()
}

func TestMalformedHeadersFailClosed(t *testing.T) {
	cases := map[string][]byte{
		"short":            {1, 2, 3},
		"zero header":      rawFile("", 0),
		"not object":       rawFile(`[1,2]`, 0),
		"invalid json":     rawFile(`{"a":`, 0),
		"unknown dtype":    rawFile(`{"a":{"dtype":"F12","shape":[1],"data_offsets":[0,4]}}`, 4),
		"missing dtype":    rawFile(`{"a":{"shape":[1],"data_offsets":[0,4]}}`, 4),
		"missing shape":    rawFile(`{"a":{"dtype":"F32","data_offsets":[0,4]}}`, 4),
		"missing offsets":  rawFile(`{"a":{"dtype":"F32","shape":[1]}}`, 4),
		"size mismatch":    rawFile(`{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}`, 4),
		"reversed offsets": rawFile(`{"a":{"dtype":"F32","shape":[0],"data_offsets":[4,0]}}`, 4),
		"out of bounds":    rawFile(`{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}`, 4),
		"hole":             rawFile(`{"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}`, 8),
		"trailing buffer":  rawFile(`{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`, 8),
		"overlap": rawFile(`{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},`+
			`"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}`, 8),
		"duplicate key": rawFile(`{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},`+
			`"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`, 4),
		"unknown field":    rawFile(`{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"extra":1}}`, 4),
		"negative dim":     rawFile(`{"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}`, 4),
		"float dim":        rawFile(`{"a":{"dtype":"F32","shape":[1.5],"data_offsets":[0,4]}}`, 4),
		"overflow shape":   rawFile(`{"a":{"dtype":"F32","shape":[2147483647,2147483647,2147483647],"data_offsets":[0,4]}}`, 4),
		"bad metadata":     rawFile(`{"__metadata__":{"k":1}}`, 0),
		"trailing garbage": rawFile(`{} {}`, 0),
		"invalid utf8":     rawFile("{\"\xff\":{}}", 0),
	}
	for name, data := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := Parse(data)
			if !errors.Is(err, ErrInvalidArtifact) {
				t.Fatalf("Parse error = %v, want ErrInvalidArtifact", err)
			}
		})
	}
	var huge [8]byte
	binary.LittleEndian.PutUint64(huge[:], MaxHeaderBytes+1)
	if _, err := Parse(append(huge[:], '{', '}')); !errors.Is(err, ErrInvalidArtifact) {
		t.Fatalf("oversized header accepted: %v", err)
	}
}

func TestEmptyHeaderAndPaddingAccepted(t *testing.T) {
	f, err := Parse(rawFile(`{}      `, 0))
	if err != nil {
		t.Fatal(err)
	}
	if len(f.Header.Tensors) != 0 {
		t.Fatal("expected no tensors")
	}
}

func FuzzParse(f *testing.F) {
	f.Add(rawFile(`{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`, 4))
	f.Add(rawFile(`{"__metadata__":{"k":"v"},"b":{"dtype":"BF16","shape":[2,1],"data_offsets":[0,4]}}`, 4))
	f.Add([]byte{0, 0, 0, 0, 0, 0, 0, 0})
	f.Fuzz(func(t *testing.T, data []byte) {
		file, err := Parse(data)
		if err != nil {
			if !errors.Is(err, ErrInvalidArtifact) {
				t.Fatalf("unexpected error class: %v", err)
			}
			return
		}
		for _, name := range file.Header.Names() {
			tensor, err := file.Tensor(name)
			if err != nil {
				t.Fatal(err)
			}
			switch tensor.DType {
			case F32, F16, BF16, F64:
				if _, err := tensor.Float32(); err != nil {
					t.Fatal(err)
				}
			case I8:
				if _, err := tensor.Int8(); err != nil {
					t.Fatal(err)
				}
			}
		}
	})
}
