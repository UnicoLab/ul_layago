//go:build purego || !(arm64 || amd64)

package tensor

// archMicrokernel reports no assembly kernel: this architecture has none, or
// the purego build tag removed it. Linear.Forward runs the generic kernel.
func archMicrokernel() *microkernel { return nil }
