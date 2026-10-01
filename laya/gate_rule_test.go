//go:build unix

package laya

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// gateCandidate is one serving configuration the M0-LAYA gate may pass on.
type gateCandidate struct {
	ID       string `json:"id"`
	Path     string `json:"path"`
	Workers  int    `json:"intra_op_workers"`
	Inflight int    `json:"max_inflight_passes"`
}

// gatePreregistration is the M0-LAYA decision rule, fixed before any gate
// run and restated in internal/layago/README.md ("M0-LAYA gate procedure").
// LAYAGO_GATE_* variables can take a run out of it (the receipt then says
// not_a_gate_run) but cannot change it, so no rate, budget or configuration
// can be chosen after seeing which ones pass.
//
//   - Rates: every one of 5, 10 and 20 rps per instance, the open-loop
//     Poisson load the design sets for the fast lane (revise_laya-go.md:
//     "under an open-loop Poisson load of 5, 10 and 20 rps per instance").
//   - Budget: p95 <= 120 ms at 64 tokens (the plan's M0-LAYA gate; the fast
//     lane caps input at 64 tokens).
//   - Candidates: the four serving configurations of runs 1 to 4. A pass
//     binds production to that configuration.
//   - Depths: 22 layers, else the 6- and then 4-layer student.
//   - Host: the AVX2 kernel, GOMAXPROCS 8, 120 s per cell, and the Go
//     toolchain deploy/docker/Dockerfile builds the release with.
//
// A pass also needs the parity half of the gate for the passing path (see
// gateParityPrerequisite); the harness cannot see that evidence, so the
// receipt states it and the README's decision rule enforces it.
var gatePreregistration = struct {
	Tokens     int             `json:"tokens"`
	BudgetMs   float64         `json:"budget_p95_ms"`
	RatesRPS   []float64       `json:"every_offered_rps"`
	MinSeconds float64         `json:"min_seconds_per_cell"`
	Kernel     string          `json:"kernel"`
	GOMAXPROCS int             `json:"gomaxprocs"`
	GoVersion  string          `json:"go_version"`
	Layers     []int           `json:"layers_in_order"`
	Candidates []gateCandidate `json:"candidates"`
}{
	Tokens:     64,
	BudgetMs:   120,
	RatesRPS:   []float64{5, 10, 20},
	MinSeconds: 120,
	Kernel:     "amd64_avx2",
	GOMAXPROCS: 8,
	GoVersion:  "go1.26.7",
	Layers:     []int{22, 6, 4},
	Candidates: []gateCandidate{
		{ID: "C1", Path: "fp32", Workers: 8, Inflight: 1},
		{ID: "C2", Path: "int8", Workers: 8, Inflight: 1},
		{ID: "C3", Path: "fp32", Workers: 4, Inflight: 2},
		{ID: "C4", Path: "fp32", Workers: 1, Inflight: 8},
	},
}

// gateParityPrerequisite is the parity evidence a passing run of this path
// must have on the real Laya checkpoint (TestRealCheckpointParity) with the
// gate's kernel before its pass counts.
func gateParityPrerequisite(path string) string {
	if path == "int8" {
		return "int8 counts only with a passing TestRealCheckpointParity at LAYAGO_LAYA_PATH=int8 and LAYAGO_LAYA_KERNEL=amd64_avx2 against the fp32 PyTorch reference: max |Δp| <= 0.005 and no choice flip where PyTorch's top-2 margin exceeds 1e-3 (README parity table); without it this run is informational"
	}
	return "fp32 counts only with a passing TestRealCheckpointParity at LAYAGO_LAYA_KERNEL=amd64_avx2: max |Δp| <= 1e-4 and no choice flip where PyTorch's top-2 margin exceeds 1e-3 (README parity table)"
}

// gateHost is what the rule needs to know about where the run happened.
type gateHost struct {
	Kernel     string
	GOMAXPROCS int
	GoVersion  string
}

// gateDecision is the harness's reading of one run against the
// pre-registered rule.
type gateDecision struct {
	// Verdict is pass, fail or not_a_gate_run.
	Verdict            string   `json:"verdict"`
	Candidate          string   `json:"candidate,omitempty"`
	Layers             int      `json:"layers"`
	Rule               string   `json:"rule"`
	Deviations         []string `json:"deviations,omitempty"`
	FailingCells       []string `json:"failing_cells,omitempty"`
	ParityPrerequisite string   `json:"parity_prerequisite"`
}

func judgeGate(c gateConfig, cells []gateCell, host gateHost) gateDecision {
	p := gatePreregistration
	d := gateDecision{
		Layers: c.Layers,
		Rule: fmt.Sprintf("pass iff every %d-token cell at each of %v rps reads pass (p95 <= %.0f ms from scheduled arrival, queue drained), on a pre-registered candidate and depth",
			p.Tokens, p.RatesRPS, p.BudgetMs),
		ParityPrerequisite: gateParityPrerequisite(c.Path),
	}
	for _, cand := range p.Candidates {
		if cand.Path == c.Path && cand.Workers == c.Workers && cand.Inflight == c.Inflight {
			d.Candidate = cand.ID
		}
	}
	deviate := func(format string, args ...any) { d.Deviations = append(d.Deviations, fmt.Sprintf(format, args...)) }
	if d.Candidate == "" {
		deviate("path %s with %d workers and %d in flight is not a pre-registered candidate", c.Path, c.Workers, c.Inflight)
	}
	if !slices.Contains(p.Layers, c.Layers) {
		deviate("%d layers is not a pre-registered depth %v", c.Layers, p.Layers)
	}
	if c.BudgetMs != p.BudgetMs || c.BudgetTokens != p.Tokens {
		deviate("budget p95 <= %.0f ms at %d tokens differs from the pre-registered %.0f ms at %d", c.BudgetMs, c.BudgetTokens, p.BudgetMs, p.Tokens)
	}
	if !slices.Contains(c.Tokens, p.Tokens) {
		deviate("no %d-token cells", p.Tokens)
	}
	for _, rps := range p.RatesRPS {
		if !slices.Contains(c.RPS, rps) {
			deviate("no cells at %.0f rps", rps)
		}
	}
	if c.Seconds < p.MinSeconds {
		deviate("%g s per cell is below the pre-registered %g s", c.Seconds, p.MinSeconds)
	}
	if host.Kernel != p.Kernel {
		deviate("kernel %s is not %s", host.Kernel, p.Kernel)
	}
	if host.GOMAXPROCS != p.GOMAXPROCS {
		deviate("GOMAXPROCS %d is not %d", host.GOMAXPROCS, p.GOMAXPROCS)
	}
	if host.GoVersion != p.GoVersion {
		deviate("built with %s, not the release toolchain %s", host.GoVersion, p.GoVersion)
	}
	if len(d.Deviations) > 0 {
		d.Verdict = "not_a_gate_run"
		return d
	}
	seen := map[float64]bool{}
	for _, cell := range cells {
		if cell.Tokens != p.Tokens || !slices.Contains(p.RatesRPS, cell.OfferedRPS) {
			continue
		}
		seen[cell.OfferedRPS] = true
		if !strings.HasPrefix(cell.BudgetVerdict, "pass") {
			d.FailingCells = append(d.FailingCells, fmt.Sprintf("%d tokens at %.0f rps: %s", cell.Tokens, cell.OfferedRPS, cell.BudgetVerdict))
		}
	}
	for _, rps := range p.RatesRPS {
		if !seen[rps] {
			d.FailingCells = append(d.FailingCells, fmt.Sprintf("%d tokens at %.0f rps: no cell", p.Tokens, rps))
		}
	}
	d.Verdict = "pass"
	if len(d.FailingCells) > 0 {
		d.Verdict = "fail"
	}
	return d
}

// The rule is fixed: a run passes only on a pre-registered candidate with
// every rate passing, a failing rate cannot be dropped after the fact, and
// any override of the rule's inputs makes the run not a gate run.
func TestGateRuleIsPreregistered(t *testing.T) {
	good := gateConfig{
		Tokens: []int{32, 64, 128, 256}, RPS: []float64{5, 10, 20}, Seconds: 120,
		Workers: 8, Inflight: 1, Path: "fp32", Layers: 22, BudgetMs: 120, BudgetTokens: 64,
	}
	host := gateHost{Kernel: "amd64_avx2", GOMAXPROCS: 8, GoVersion: "go1.26.7"}
	cells := func(verdicts map[float64]string) []gateCell {
		var out []gateCell
		for _, tokens := range []int{32, 64} {
			for _, rps := range []float64{5, 10, 20} {
				cell := gateCell{Tokens: tokens, OfferedRPS: rps}
				if tokens == 64 {
					cell.BudgetVerdict = verdicts[rps]
				}
				out = append(out, cell)
			}
		}
		return out
	}
	allPass := cells(map[float64]string{5: "pass: p95 80 ms", 10: "pass: p95 90 ms", 20: "pass: p95 110 ms"})
	if d := judgeGate(good, allPass, host); d.Verdict != "pass" || d.Candidate != "C1" || len(d.Deviations) > 0 {
		t.Fatalf("every rate passing on C1: %+v", d)
	}
	// 20 rps failing cannot be read as a pass at 5 or 10 rps.
	at20 := cells(map[float64]string{5: "pass: p95 80 ms", 10: "pass: p95 90 ms", 20: "fail: p95 180 ms > 120 ms"})
	if d := judgeGate(good, at20, host); d.Verdict != "fail" || len(d.FailingCells) != 1 {
		t.Fatalf("a failing 20 rps cell: %+v", d)
	}
	// Leaving the failing rate out of the run takes it out of the gate.
	low := good
	low.RPS = []float64{5, 10}
	if d := judgeGate(low, allPass, host); d.Verdict != "not_a_gate_run" {
		t.Fatalf("a run without 20 rps: %+v", d)
	}
	for name, mutate := range map[string]func(*gateConfig, *gateHost){
		"looser budget":      func(c *gateConfig, _ *gateHost) { c.BudgetMs = 150 },
		"other budget size":  func(c *gateConfig, _ *gateHost) { c.BudgetTokens = 32 },
		"short cells":        func(c *gateConfig, _ *gateHost) { c.Seconds = 60 },
		"unregistered split": func(c *gateConfig, _ *gateHost) { c.Workers, c.Inflight = 2, 4 },
		"unregistered depth": func(c *gateConfig, _ *gateHost) { c.Layers = 12 },
		"int8 at 4 workers":  func(c *gateConfig, _ *gateHost) { c.Path, c.Workers, c.Inflight = "int8", 4, 2 },
		"portable kernel":    func(_ *gateConfig, h *gateHost) { h.Kernel = "generic" },
		"not 8 vCPU":         func(_ *gateConfig, h *gateHost) { h.GOMAXPROCS = 16 },
		"other toolchain":    func(_ *gateConfig, h *gateHost) { h.GoVersion = "go1.27.0" },
	} {
		c, h := good, host
		mutate(&c, &h)
		if d := judgeGate(c, allPass, h); d.Verdict != "not_a_gate_run" || len(d.Deviations) == 0 {
			t.Errorf("%s: %+v", name, d)
		}
	}
	// int8 can be a candidate, but its pass carries the int8 parity
	// prerequisite, and the student depths are pre-registered too.
	q := good
	q.Path = "int8"
	if d := judgeGate(q, allPass, host); d.Verdict != "pass" || d.Candidate != "C2" || !strings.Contains(d.ParityPrerequisite, "|Δp| <= 0.005") {
		t.Fatalf("int8 candidate: %+v", d)
	}
	s := good
	s.Layers, s.Workers, s.Inflight = 6, 1, 8
	if d := judgeGate(s, allPass, host); d.Verdict != "pass" || d.Candidate != "C4" || d.Layers != 6 {
		t.Fatalf("6-layer student on C4: %+v", d)
	}
}

// The pre-registered toolchain is the one the release image is built with,
// so the gate times the portable operations (attention, GELU, LayerNorm)
// from the production compiler. Skipped when the tree is extracted without
// the Dockerfile.
func TestGateToolchainIsTheReleaseToolchain(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "..", "deploy", "docker", "Dockerfile"))
	if err != nil {
		t.Skipf("no release Dockerfile here: %v", err)
	}
	want := "FROM golang:" + strings.TrimPrefix(gatePreregistration.GoVersion, "go") + "-"
	if !strings.Contains(string(raw), want) {
		t.Fatalf("deploy/docker/Dockerfile does not build with %s (no %q): update gatePreregistration.GoVersion and the README procedure together with it", gatePreregistration.GoVersion, want)
	}
}
