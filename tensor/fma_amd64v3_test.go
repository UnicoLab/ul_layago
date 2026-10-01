//go:build amd64.v3

package tensor

// With GOAMD64=v3 the compiler fuses the generic kernel's 4-column dot
// products (dot4x4, dot1x4) into VFMADD231SS, but in the one-column loops
// for the last Out mod 4 columns (dot1x1, dot1x1q) it folds the weight load
// into MULSS first and then emits ADDSS, so those columns round twice
// (checked with go tool objdump, Go 1.27). The AVX2 kernel must match the
// fused columns bit for bit and the edge columns within the rounding bound.
const (
	genericFusesFMA         = true
	genericFusesEdgeColumns = false
)
