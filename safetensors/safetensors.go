// Package safetensors reads and writes the safetensors tensor container used
// by Hugging Face checkpoints: an 8-byte little-endian header length, a JSON
// header mapping each tensor name to {dtype, shape, data_offsets}, and the raw
// little-endian byte buffer.
//
// Parsing is strict and fail-closed. The header is bounded, every tensor
// declares a known dtype and a shape whose byte size equals its offset range,
// and the declared ranges must tile the buffer exactly: sorted, contiguous,
// non-overlapping and covering the whole buffer. That is the format's own
// "no holes" rule, which prevents polyglot files. Duplicate header keys are
// rejected rather than resolved by whichever copy a JSON decoder keeps.
//
// The package imports only the standard library so internal/layago stays
// extractable as a standalone module.
package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"unicode/utf8"
)

// MaxHeaderBytes bounds the JSON header. Real checkpoints use tens of
// kilobytes; the format specification itself caps the header at 100 MB, and a
// much lower bound keeps a hostile file from forcing a large allocation.
const MaxHeaderBytes = 16 << 20

// ErrInvalidArtifact reports a malformed or inconsistent safetensors file.
var ErrInvalidArtifact = errors.New("safetensors: invalid artifact")

// DType names a safetensors element type exactly as it appears in the header.
type DType string

// Element types defined by the safetensors format.
const (
	F64     DType = "F64"
	F32     DType = "F32"
	F16     DType = "F16"
	BF16    DType = "BF16"
	I64     DType = "I64"
	I32     DType = "I32"
	I16     DType = "I16"
	I8      DType = "I8"
	U64     DType = "U64"
	U32     DType = "U32"
	U16     DType = "U16"
	U8      DType = "U8"
	Bool    DType = "BOOL"
	F8E4M3  DType = "F8_E4M3"
	F8E5M2  DType = "F8_E5M2"
	metaKey       = "__metadata__"
)

var dtypeSizes = map[DType]int{
	F64: 8, F32: 4, F16: 2, BF16: 2,
	I64: 8, I32: 4, I16: 2, I8: 1,
	U64: 8, U32: 4, U16: 2, U8: 1,
	Bool: 1, F8E4M3: 1, F8E5M2: 1,
}

// Size returns the element size in bytes, or 0 for an unknown dtype.
func (d DType) Size() int { return dtypeSizes[d] }

// TensorInfo is one validated header entry.
type TensorInfo struct {
	Name  string
	DType DType
	Shape []int
	// Begin and End are byte offsets relative to the start of the data buffer.
	Begin, End int64
}

// Elements returns the product of the shape (1 for a scalar).
func (t TensorInfo) Elements() int {
	n := 1
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

// Header is a parsed and validated safetensors header.
type Header struct {
	Metadata map[string]string
	Tensors  []TensorInfo // sorted by Begin
	// DataOffset is the absolute file offset of the data buffer (8 + header length).
	DataOffset int64
	byName     map[string]int
}

// Lookup returns the header entry for name.
func (h *Header) Lookup(name string) (TensorInfo, bool) {
	i, ok := h.byName[name]
	if !ok {
		return TensorInfo{}, false
	}
	return h.Tensors[i], true
}

// Names returns the tensor names in buffer order.
func (h *Header) Names() []string {
	out := make([]string, len(h.Tensors))
	for i, t := range h.Tensors {
		out[i] = t.Name
	}
	return out
}

type rawEntry struct {
	DType       DType   `json:"dtype"`
	Shape       []int64 `json:"shape"`
	DataOffsets []int64 `json:"data_offsets"`
}

func invalid(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrInvalidArtifact, fmt.Sprintf(format, args...))
}

// ReadHeaderLength decodes the 8-byte little-endian prefix and bounds it.
func ReadHeaderLength(prefix []byte, fileSize int64) (int64, error) {
	if len(prefix) < 8 {
		return 0, invalid("file shorter than the 8-byte header length")
	}
	n := binary.LittleEndian.Uint64(prefix[:8])
	if n < 2 || n > MaxHeaderBytes {
		return 0, invalid("header length %d outside [2, %d]", n, MaxHeaderBytes)
	}
	length := int64(n)
	if fileSize >= 0 && 8+length > fileSize {
		return 0, invalid("header length %d exceeds file size %d", length, fileSize)
	}
	return length, nil
}

// ParseHeader validates the JSON header bytes against a data buffer of
// bufferSize bytes. It never allocates in proportion to tensor sizes.
func ParseHeader(headerJSON []byte, bufferSize int64) (*Header, error) {
	if bufferSize < 0 {
		return nil, invalid("negative buffer size")
	}
	if !utf8.Valid(headerJSON) {
		return nil, invalid("header is not valid UTF-8")
	}
	trimmed := bytes.TrimRight(headerJSON, " ")
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return nil, invalid("header is not a JSON object")
	}
	dec := json.NewDecoder(bytes.NewReader(trimmed))
	dec.DisallowUnknownFields()
	tok, err := dec.Token()
	if err != nil || tok != json.Delim('{') {
		return nil, invalid("header is not a JSON object")
	}
	h := &Header{DataOffset: 8 + int64(len(headerJSON)), byName: map[string]int{}}
	seen := map[string]bool{}
	for dec.More() {
		keyTok, err := dec.Token()
		if err != nil {
			return nil, invalid("header key: %v", err)
		}
		key, ok := keyTok.(string)
		if !ok {
			return nil, invalid("header key is not a string")
		}
		if seen[key] {
			return nil, invalid("duplicate header key %q", key)
		}
		seen[key] = true
		if key == metaKey {
			var meta map[string]string
			if err := dec.Decode(&meta); err != nil {
				return nil, invalid("__metadata__ must map strings to strings: %v", err)
			}
			h.Metadata = meta
			continue
		}
		var raw rawEntry
		if err := dec.Decode(&raw); err != nil {
			return nil, invalid("tensor %q: %v", key, err)
		}
		info, err := validateEntry(key, raw)
		if err != nil {
			return nil, err
		}
		h.Tensors = append(h.Tensors, info)
	}
	if tok, err := dec.Token(); err != nil || tok != json.Delim('}') {
		return nil, invalid("header object is not terminated")
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, invalid("trailing data after the header object")
	}
	sort.SliceStable(h.Tensors, func(i, j int) bool {
		if h.Tensors[i].Begin != h.Tensors[j].Begin {
			return h.Tensors[i].Begin < h.Tensors[j].Begin
		}
		return h.Tensors[i].End < h.Tensors[j].End
	})
	var cursor int64
	for i, t := range h.Tensors {
		if t.Begin != cursor {
			if t.Begin < cursor {
				return nil, invalid("tensor %q overlaps the previous tensor", t.Name)
			}
			return nil, invalid("hole before tensor %q at buffer offset %d", t.Name, cursor)
		}
		cursor = t.End
		h.byName[t.Name] = i
	}
	if cursor != bufferSize {
		return nil, invalid("tensors cover %d bytes but the buffer has %d", cursor, bufferSize)
	}
	return h, nil
}

func validateEntry(name string, raw rawEntry) (TensorInfo, error) {
	size := raw.DType.Size()
	if size == 0 {
		return TensorInfo{}, invalid("tensor %q has unknown dtype %q", name, raw.DType)
	}
	if raw.Shape == nil {
		return TensorInfo{}, invalid("tensor %q has no shape", name)
	}
	if len(raw.DataOffsets) != 2 {
		return TensorInfo{}, invalid("tensor %q data_offsets must have two entries", name)
	}
	begin, end := raw.DataOffsets[0], raw.DataOffsets[1]
	if begin < 0 || end < begin {
		return TensorInfo{}, invalid("tensor %q has invalid data_offsets [%d, %d]", name, begin, end)
	}
	shape := make([]int, len(raw.Shape))
	elements := int64(1)
	for i, d := range raw.Shape {
		if d < 0 || d > math.MaxInt32 {
			return TensorInfo{}, invalid("tensor %q has invalid dimension %d", name, d)
		}
		shape[i] = int(d)
		if d != 0 && elements > math.MaxInt64/d {
			return TensorInfo{}, invalid("tensor %q shape overflows", name)
		}
		elements *= d
	}
	if elements > math.MaxInt64/int64(size) {
		return TensorInfo{}, invalid("tensor %q byte size overflows", name)
	}
	if elements*int64(size) != end-begin {
		return TensorInfo{}, invalid("tensor %q shape %v x %s needs %d bytes but data_offsets span %d",
			name, raw.Shape, raw.DType, elements*int64(size), end-begin)
	}
	if elements > math.MaxInt32*int64(8) {
		return TensorInfo{}, invalid("tensor %q is too large", name)
	}
	return TensorInfo{Name: name, DType: raw.DType, Shape: shape, Begin: begin, End: end}, nil
}

// Tensor is a view of one tensor's bytes. For a memory-mapped File the bytes
// alias the mapping and become invalid after File.Close.
type Tensor struct {
	TensorInfo
	Data []byte
}

// Float32 converts a floating-point tensor to a new float32 slice.
func (t Tensor) Float32() ([]float32, error) {
	n := t.Elements()
	out := make([]float32, n)
	b := t.Data
	switch t.DType {
	case F32:
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		}
	case F16:
		for i := range out {
			out[i] = Float16ToFloat32(binary.LittleEndian.Uint16(b[2*i:]))
		}
	case BF16:
		for i := range out {
			out[i] = BFloat16ToFloat32(binary.LittleEndian.Uint16(b[2*i:]))
		}
	case F64:
		for i := range out {
			out[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(b[8*i:])))
		}
	default:
		return nil, fmt.Errorf("safetensors: tensor %q has dtype %s, not a float type", t.Name, t.DType)
	}
	return out, nil
}

// Int8 returns a copy of an I8 tensor.
func (t Tensor) Int8() ([]int8, error) {
	if t.DType != I8 {
		return nil, fmt.Errorf("safetensors: tensor %q has dtype %s, not I8", t.Name, t.DType)
	}
	out := make([]int8, len(t.Data))
	for i, v := range t.Data {
		out[i] = int8(v) //nolint:gosec // G115: two's-complement reinterpretation of the stored byte
	}
	return out, nil
}

// BFloat16ToFloat32 widens a bfloat16 bit pattern exactly.
func BFloat16ToFloat32(h uint16) float32 {
	return math.Float32frombits(uint32(h) << 16)
}

// Float16ToFloat32 widens an IEEE 754 binary16 bit pattern exactly, including
// subnormals, infinities and NaN payloads.
func Float16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff
	switch {
	case exp == 0 && mant == 0:
		return math.Float32frombits(sign)
	case exp == 0:
		// Subnormal: normalise the mantissa.
		e := uint32(127 - 15 + 1)
		for mant&0x400 == 0 {
			mant <<= 1
			e--
		}
		mant &= 0x3ff
		return math.Float32frombits(sign | e<<23 | mant<<13)
	case exp == 0x1f:
		return math.Float32frombits(sign | 0xff<<23 | mant<<13)
	default:
		return math.Float32frombits(sign | (exp+127-15)<<23 | mant<<13)
	}
}

// File is an opened safetensors container.
type File struct {
	Header *Header
	data   []byte // whole data buffer when mapped or read eagerly
	ra     io.ReaderAt
	unmap  func() error
	mapped bool
}

// Mapped reports whether the data buffer is a read-only memory mapping.
func (f *File) Mapped() bool { return f.mapped }

// Tensor returns the tensor called name. With a ReaderAt-backed File the
// bytes are read into a fresh buffer.
func (f *File) Tensor(name string) (Tensor, error) {
	info, ok := f.Header.Lookup(name)
	if !ok {
		return Tensor{}, fmt.Errorf("safetensors: tensor %q not found", name)
	}
	if f.data != nil {
		return Tensor{TensorInfo: info, Data: f.data[info.Begin:info.End:info.End]}, nil
	}
	buf := make([]byte, info.End-info.Begin)
	if _, err := f.ra.ReadAt(buf, f.Header.DataOffset+info.Begin); err != nil {
		return Tensor{}, fmt.Errorf("safetensors: read %q: %w", name, err)
	}
	return Tensor{TensorInfo: info, Data: buf}, nil
}

// Close releases the mapping. Tensor views from a mapped File must not be
// used afterwards.
func (f *File) Close() error {
	if f.unmap != nil {
		err := f.unmap()
		f.unmap = nil
		f.data = nil
		return err
	}
	return nil
}

// Parse validates an in-memory safetensors file. The returned File aliases b.
func Parse(b []byte) (*File, error) {
	length, err := ReadHeaderLength(b, int64(len(b)))
	if err != nil {
		return nil, err
	}
	h, err := ParseHeader(b[8:8+length], int64(len(b))-8-length)
	if err != nil {
		return nil, err
	}
	return &File{Header: h, data: b[8+length:]}, nil
}

// OpenReaderAt validates the header of a size-byte container read through r
// and reads tensors lazily.
func OpenReaderAt(r io.ReaderAt, size int64) (*File, error) {
	var prefix [8]byte
	if _, err := r.ReadAt(prefix[:], 0); err != nil {
		return nil, invalid("read header length: %v", err)
	}
	length, err := ReadHeaderLength(prefix[:], size)
	if err != nil {
		return nil, err
	}
	hdr := make([]byte, length)
	if _, err := r.ReadAt(hdr, 8); err != nil {
		return nil, invalid("read header: %v", err)
	}
	h, err := ParseHeader(hdr, size-8-length)
	if err != nil {
		return nil, err
	}
	return &File{Header: h, ra: r}, nil
}

// Open opens path, memory-mapping it read-only where the platform allows and
// falling back to reading it into memory otherwise.
func Open(path string) (*File, error) {
	fh, err := os.Open(path) //nolint:gosec // G304: the caller chooses which local artifact to load
	if err != nil {
		return nil, err
	}
	defer func() { _ = fh.Close() }()
	st, err := fh.Stat()
	if err != nil {
		return nil, err
	}
	size := st.Size()
	if data, unmap, err := mmapFile(fh, size); err == nil {
		f, perr := Parse(data)
		if perr != nil {
			_ = unmap()
			return nil, perr
		}
		f.unmap = unmap
		f.mapped = true
		return f, nil
	}
	data, err := io.ReadAll(io.LimitReader(fh, size))
	if err != nil {
		return nil, err
	}
	return Parse(data)
}
