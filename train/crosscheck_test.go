package train

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestConvergedLoadersServeGoArtifacts is the cross-check against the
// converged loaders themselves: laya.LoadArtifact, laya.RenderPromptState and
// routermodel.Load, in this tree since the W10 merge (they converged on
// w10/laya-train and w10/laya-integrate). LAYAGO_CONVERGED_TREES names one or
// more module roots that hold them, comma-separated: this checkout, or another
// tree, for example a revision extracted with `git archive <rev> go.mod go.sum
// internal/layago internal/routermodel | tar -x -C <dir>`. For each tree the
// test exports this trainer's Q artifacts (float32 and int8) and R artifact
// from the tiny-trunk loop and runs, inside that tree:
//
//   - laya's TestExternalArtifactParity on every artifact: LoadArtifact
//     must accept it (pinned digests, prompt format, sequence, tasks,
//     temperatures, thresholds, head tensors) and reproduce its
//     parity_golden.json (token ids exact, logits within 1e-4,
//     probabilities within 2e-5, same temperature cell, decision and
//     prediction);
//   - routermodel's TestRealArtifactParity on the Q artifacts, when the tree
//     has routermodel: Load must accept them (digests, adapter, data digest
//     from training.files, Hive's Q task profiles) and Hive's own rendering
//     of each golden example's catalog (laya.RenderPromptState) must give
//     the state, question, token ids and logits the trainer recorded.
//
// Run it from the module root with:
//
//	LAYAGO_CONVERGED_TREES=$PWD \
//	    go test -count=1 -run TestConvergedLoadersServeGoArtifacts -v ./train
func TestConvergedLoadersServeGoArtifacts(t *testing.T) {
	trees := os.Getenv("LAYAGO_CONVERGED_TREES")
	if trees == "" {
		t.Skip("set LAYAGO_CONVERGED_TREES to a module root holding laya.LoadArtifact and routermodel.Load (this checkout will do) to run the converged loaders")
	}
	l := qLoop(t)
	base, err := filepath.Abs(tinyTrunkDir)
	if err != nil {
		t.Fatal(err)
	}
	rDir := exportR(t)
	goBin, err := exec.LookPath("go")
	if err != nil {
		t.Fatalf("the cross-check runs go test in the converged trees: %v", err)
	}
	run := func(tree, pkg, test string, env ...string) {
		t.Helper()
		cmd := exec.CommandContext(context.Background(), goBin, "test", "-count=1", "-run", "^"+test+"$", "-v", pkg) //nolint:gosec // G204: fixed go test invocation in a tree the operator named
		cmd.Dir = tree
		cmd.Env = append(os.Environ(), append([]string{"GOWORK=off", "GOFLAGS=-mod=readonly", "GOPROXY=off", "CGO_ENABLED=0"}, env...)...)
		out, err := cmd.CombinedOutput()
		text := string(out)
		if err != nil || !strings.Contains(text, "--- PASS: "+test) {
			t.Fatalf("%s %s in %s failed (%v):\n%s", pkg, test, tree, err, text)
		}
		for _, line := range strings.Split(text, "\n") {
			if strings.Contains(line, "max |Δ logit|") || strings.Contains(line, "identity") {
				t.Logf("%s: %s", filepath.Base(tree), strings.TrimSpace(line))
			}
		}
	}
	for _, tree := range strings.Split(trees, ",") {
		tree = strings.TrimSpace(tree)
		pkg := "./internal/layago/laya"
		if _, err := os.Stat(filepath.Join(tree, "laya", "artifact.go")); err == nil {
			pkg = "./laya"
		}
		for _, dir := range append(append([]string(nil), l.dirs...), rDir) {
			run(tree, pkg, "TestExternalArtifactParity", "LAYAGO_ARTIFACT_DIR="+dir, "LAYAGO_ARTIFACT_BASE_DIR="+base)
		}
		if _, err := os.Stat(filepath.Join(tree, "internal", "routermodel", "parity_test.go")); err != nil {
			t.Logf("%s has no routermodel; laya.LoadArtifact only", tree)
			continue
		}
		for _, dir := range l.dirs {
			run(tree, "./internal/routermodel", "TestRealArtifactParity", "LAYA_ARTIFACT_DIR="+dir, "LAYA_BASE_MODEL_DIR="+base,
				"LAYA_DATASET="+l.data)
		}
	}
}

// exportR trains and exports a small instance-R artifact.
func exportR(t *testing.T) string {
	t.Helper()
	ctx := context.Background()
	trunk := tinyTrunk(t, tinySpec)
	p, err := Prepare(fixtureDataset(t, InstanceR), trunk, PrepareOptions{})
	if err != nil {
		t.Fatal(err)
	}
	cache, err := BuildCache(ctx, trunk, p, CacheSpec{Root: t.TempDir(), ShardSize: 100})
	if err != nil {
		t.Fatal(err)
	}
	recs, err := cache.Records()
	if err != nil {
		t.Fatal(err)
	}
	cfg := loopConfig(0)
	cfg.Epochs, cfg.Int8 = 6, false
	res, err := Train(ctx, p, recs, trunk.Identity.HiddenSize, cfg)
	if err != nil {
		t.Fatal(err)
	}
	dirs, err := Export(t.TempDir(), res, ExportMeta{Code: CodeVersion{GitCommit: "test"}, Trunk: trunk, Cache: cache})
	if err != nil {
		t.Fatal(err)
	}
	return dirs[0]
}
