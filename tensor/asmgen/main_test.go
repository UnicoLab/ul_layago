package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// The committed assembly must be exactly what this generator prints, so a
// review of the generator is a review of the kernels.
func TestGeneratedAssemblyIsCurrent(t *testing.T) {
	for name, want := range Files() {
		got, err := os.ReadFile(filepath.Join("..", name))
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Errorf("%s is stale: run go generate ./tensor", name)
		}
	}
}

// The by-element FMLA words were checked against the system assembler
// (as -arch arm64): fmla v0.4s, v24.4s, v16.s[0] and friends.
func TestFMLAByElementEncoding(t *testing.T) {
	for _, c := range []struct {
		d, n, m, lane int
		want          uint32
	}{
		{0, 24, 16, 0, 0x4f901300},
		{1, 25, 16, 1, 0x4fb01321},
		{14, 30, 23, 2, 0x4f971bce},
		{15, 31, 23, 3, 0x4fb71bef},
	} {
		if got := fmlaElem(c.d, c.n, c.m, c.lane); got != c.want {
			t.Errorf("fmla v%d.4s, v%d.4s, v%d.s[%d] = %#08x, want %#08x", c.d, c.n, c.m, c.lane, got, c.want)
		}
	}
	for _, bad := range [][4]int{{32, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, 40, 0}, {0, 0, 0, 4}} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("fmlaElem%v encoded an impossible instruction", bad)
				}
			}()
			fmlaElem(bad[0], bad[1], bad[2], bad[3])
		}()
	}
}

// The int8 path's widening and conversion words, checked against the system
// assembler (as -arch arm64, Apple clang 17): sxtl v25.8h, v24.8b and the
// rest of the kernel's sequence, plus the extreme registers.
func TestWidenAndConvertEncoding(t *testing.T) {
	for _, c := range []struct {
		d, n, bits int
		upper      bool
		want       uint32
	}{
		{25, 24, 8, false, 0x0f08a719},
		{26, 24, 8, true, 0x4f08a71a},
		{27, 25, 16, false, 0x0f10a73b},
		{28, 25, 16, true, 0x4f10a73c},
		{29, 26, 16, false, 0x0f10a75d},
		{30, 26, 16, true, 0x4f10a75e},
		{0, 31, 8, false, 0x0f08a7e0},
		{31, 0, 16, true, 0x4f10a41f},
	} {
		if got := sxtl(c.d, c.n, c.bits, c.upper); got != c.want {
			t.Errorf("sxtl(v%d, v%d, %d bits, upper=%v) = %#08x, want %#08x", c.d, c.n, c.bits, c.upper, got, c.want)
		}
	}
	for _, c := range []struct {
		d, n int
		want uint32
	}{
		{27, 27, 0x4e21db7b},
		{30, 30, 0x4e21dbde},
		{0, 31, 0x4e21dbe0},
		{31, 0, 0x4e21d81f},
	} {
		if got := scvtf4s(c.d, c.n); got != c.want {
			t.Errorf("scvtf v%d.4s, v%d.4s = %#08x, want %#08x", c.d, c.n, got, c.want)
		}
	}
	for name, bad := range map[string]func(){
		"register 32":   func() { sxtl(32, 0, 8, false) },
		"register -1":   func() { scvtf4s(0, -1) },
		"32-bit source": func() { sxtl(0, 0, 32, false) },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: encoded an impossible instruction", name)
				}
			}()
			bad()
		}()
	}
}

// Every generated file must assemble with the toolchain that runs the test,
// for its own architecture, on any host: CI runs amd64 with the pinned Go
// 1.26.7, so this is where an arm64 mnemonic that only a newer assembler
// knows (VSXTL, VSXTL2 and VSCVTF were such mnemonics) fails, instead of on
// the first Apple silicon build or arm64 release.
func TestGeneratedAssemblyAssemblesWithThisToolchain(t *testing.T) {
	goBin, err := exec.LookPath("go")
	if err != nil {
		t.Skipf("no go command on PATH (a copied test binary): %v", err)
	}
	env := func(key string) string {
		out, err := exec.Command(goBin, "env", key).Output() //nolint:gosec // G204: the go command go test put on PATH
		if err != nil {
			t.Fatalf("go env %s: %v", key, err)
		}
		return strings.TrimSpace(string(out))
	}
	goroot, version := env("GOROOT"), env("GOVERSION")
	dir := t.TempDir()
	for name, src := range Files() {
		arch := strings.TrimSuffix(strings.TrimPrefix(name, "gemm_"), ".s")
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, src, 0o600); err != nil {
			t.Fatal(err)
		}
		cmd := exec.Command(goBin, "tool", "asm", //nolint:gosec // G204: fixed arguments and a file this test wrote
			"-p", "github.com/UnicoLab/ul_layago/tensor",
			"-I", filepath.Join(goroot, "pkg", "include"),
			"-o", filepath.Join(dir, arch+".o"), path)
		cmd.Env = append(os.Environ(), "GOOS=linux", "GOARCH="+arch)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Errorf("%s does not assemble with %s for linux/%s: %v\n%s", name, version, arch, err, out)
			continue
		}
		t.Logf("%s assembles with %s for linux/%s", name, version, arch)
	}
}
