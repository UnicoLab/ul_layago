package laya

import "syscall"

// translatedByRosetta reports whether this process runs under Rosetta 2.
// sysctl.proc_translated describes the calling process, so it must be read
// in-process: a sysctl(8) child is native and would always say 0.
func translatedByRosetta() bool {
	v, err := syscall.Sysctl("sysctl.proc_translated")
	return err == nil && len(v) > 0 && v[0] == 1
}
