//go:build !unix

package train

// lockStore does not lock on this platform: run one build per feature
// store at a time.
func lockStore(string) (func(), error) { return func() {}, nil }
