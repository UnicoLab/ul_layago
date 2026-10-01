//go:build !(arm64 || amd64.v3)

package tensor

// Below GOAMD64=v3 (the release image builds at v1) the generic kernel rounds
// the product and the sum separately, so an FMA kernel agrees with it only
// within the rounding bound.
const (
	genericFusesFMA         = false
	genericFusesEdgeColumns = false
)
