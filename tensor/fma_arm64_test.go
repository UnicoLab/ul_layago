//go:build arm64

package tensor

// The arm64 compiler turns every c += a*b of the generic kernel into FMADDS
// (arm64 has no load-and-multiply form to get in the way), so the assembly
// kernels must match it bit for bit on every column.
const (
	genericFusesFMA         = true
	genericFusesEdgeColumns = true
)
