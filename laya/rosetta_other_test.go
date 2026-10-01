//go:build !darwin

package laya

// translatedByRosetta is false off macOS: Rosetta 2 exists only there.
func translatedByRosetta() bool { return false }
