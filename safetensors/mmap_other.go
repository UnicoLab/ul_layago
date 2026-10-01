//go:build !unix

package safetensors

import (
	"errors"
	"os"
)

func mmapFile(_ *os.File, _ int64) ([]byte, func() error, error) {
	return nil, nil, errors.New("safetensors: memory mapping is not supported on this platform")
}
