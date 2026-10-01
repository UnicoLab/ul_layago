//go:build !unix

package train

import "time"

// processCPU is not measured on this platform.
func processCPU() time.Duration { return 0 }
