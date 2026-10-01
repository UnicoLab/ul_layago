package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
)

// NamedTensor is one tensor to serialise. Data must already hold the
// little-endian bytes for DType and Shape.
type NamedTensor struct {
	Name  string
	DType DType
	Shape []int
	Data  []byte
}

// Float32Tensor encodes values as an F32 tensor.
func Float32Tensor(name string, shape []int, values []float32) NamedTensor {
	b := make([]byte, 4*len(values))
	for i, v := range values {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
	}
	return NamedTensor{Name: name, DType: F32, Shape: shape, Data: b}
}

// Int8Tensor encodes values as an I8 tensor.
func Int8Tensor(name string, shape []int, values []int8) NamedTensor {
	b := make([]byte, len(values))
	for i, v := range values {
		b[i] = byte(v) //nolint:gosec // G115: two's-complement reinterpretation into the stored byte
	}
	return NamedTensor{Name: name, DType: I8, Shape: shape, Data: b}
}

// Write serialises tensors in name order with an 8-byte aligned, space-padded
// header, which is the layout the reference Python writer produces.
func Write(w io.Writer, tensors []NamedTensor, metadata map[string]string) error {
	sorted := append([]NamedTensor(nil), tensors...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Name < sorted[j].Name })
	header := map[string]any{}
	if len(metadata) > 0 {
		header[metaKey] = metadata
	}
	var offset int64
	for _, t := range sorted {
		if _, dup := header[t.Name]; dup || t.Name == metaKey {
			return fmt.Errorf("safetensors: duplicate or reserved tensor name %q", t.Name)
		}
		n := 1
		for _, d := range t.Shape {
			n *= d
		}
		if t.DType.Size() == 0 || n*t.DType.Size() != len(t.Data) {
			return fmt.Errorf("safetensors: tensor %q data does not match %s%v", t.Name, t.DType, t.Shape)
		}
		shape := t.Shape
		if shape == nil {
			shape = []int{}
		}
		end := offset + int64(len(t.Data))
		header[t.Name] = map[string]any{"dtype": t.DType, "shape": shape, "data_offsets": []int64{offset, end}}
		offset = end
	}
	hdr, err := json.Marshal(header)
	if err != nil {
		return err
	}
	for len(hdr)%8 != 0 {
		hdr = append(hdr, ' ')
	}
	var prefix [8]byte
	binary.LittleEndian.PutUint64(prefix[:], uint64(len(hdr)))
	if _, err := w.Write(prefix[:]); err != nil {
		return err
	}
	if _, err := w.Write(hdr); err != nil {
		return err
	}
	for _, t := range sorted {
		if _, err := w.Write(t.Data); err != nil {
			return err
		}
	}
	return nil
}
