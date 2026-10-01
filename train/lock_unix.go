//go:build unix

package train

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"syscall"
)

// lockStore takes an exclusive advisory lock on a feature store, so two
// builds never append to it at once. The kernel releases the lock when the
// process exits, so an interrupted build leaves no stale lock behind.
func lockStore(dir string) (func(), error) {
	f, err := os.OpenFile(filepath.Join(dir, ".lock"), os.O_CREATE|os.O_RDWR, 0o600) //nolint:gosec // G304: inside the store
	if err != nil {
		return nil, err
	}
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil { //nolint:gosec // G115: a file descriptor fits an int
		_ = f.Close()
		if errors.Is(err, syscall.EWOULDBLOCK) {
			return nil, fmt.Errorf("train: another build is writing the feature store %s", dir)
		}
		return nil, err
	}
	return func() {
		_ = syscall.Flock(int(f.Fd()), syscall.LOCK_UN) //nolint:gosec // G115: a file descriptor fits an int
		_ = f.Close()
	}, nil
}
