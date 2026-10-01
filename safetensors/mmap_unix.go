//go:build unix

package safetensors

import (
	"errors"
	"math"
	"os"
	"syscall"
)

// mmapFile maps size bytes of f read-only. The mapping outlives f's
// descriptor, so the caller may close the file immediately.
func mmapFile(f *os.File, size int64) ([]byte, func() error, error) {
	if size <= 0 || size > math.MaxInt {
		return nil, nil, errors.New("safetensors: file size cannot be mapped")
	}
	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED) //nolint:gosec // G115: Fd fits in int on unix
	if err != nil {
		return nil, nil, err
	}
	return data, func() error { return syscall.Munmap(data) }, nil
}
