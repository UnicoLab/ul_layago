package train

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/UnicoLab/ul_layago/laya"
	"github.com/UnicoLab/ul_layago/model"
	"github.com/UnicoLab/ul_layago/safetensors"
	"github.com/UnicoLab/ul_layago/tensor"
)

// tinyTrunkDir is the committed synthetic Laya checkpoint (d=64, 5 layers,
// global and sliding layers, random weights, 510-token BPE with byte
// fallback): the tiny random config the full loop runs on. It is also the
// base the converged trainer's tiny-q artifacts pin.
const tinyTrunkDir = "../testdata/parity/checkpoint"

// tinySpec lays the fixture out. The tiny trunk's 510-token vocabulary
// spells most words in several tokens (Japanese and Arabic byte by byte), so
// most states are cut at max_len; the question comes first and is kept.
var tinySpec = TrunkSpec{MaxLen: 192, HeadMaxLen: 112, ModelID: "layago-parity-synthetic-d64"}

func tinyTrunk(t testing.TB, spec TrunkSpec) *Trunk {
	t.Helper()
	spec.Dir = tinyTrunkDir
	tr, err := OpenTrunk(spec)
	if err != nil {
		t.Fatal(err)
	}
	return tr
}

func fixtureDataset(t testing.TB, instance string) *Dataset {
	t.Helper()
	ds, err := ParseDataset([]DataFile{{Name: "fixture.jsonl", Data: fixtureBytes(t)}}, instance)
	if err != nil {
		t.Fatal(err)
	}
	return ds
}

// loopConfig trains long enough to fit the tiny random trunk. The act
// guarantee (alpha 0.5, delta 0.5) is deliberately loose: the random trunk
// cannot certify a meaningful one on this little dev_conf, and the test needs
// both act and escalate decisions to compare served and trained decisions.
func loopConfig(workers int) Config {
	c := DefaultConfig()
	c.Seed, c.Epochs, c.Patience, c.BatchSize, c.LR, c.ActEpochs = 7, 10, 10, 16, 3e-3, 8
	c.MinCell, c.Alpha, c.Delta, c.NMin, c.Workers = 8, 0.5, 0.5, 5, workers
	return c
}

// trainedLoop is the Q pipeline on the tiny trunk, run once and shared.
type trainedLoop struct {
	trunk *Trunk
	ds    *Dataset
	p     *Prepared
	cache *Cache
	recs  map[string]*Record
	res   *Result
	dirs  []string
	data  string // the dataset file the artifacts were trained on
}

var (
	loopOnce sync.Once
	loopQ    *trainedLoop
	loopErr  error
	loopRoot string
)

func TestMain(m *testing.M) {
	code := m.Run()
	if loopRoot != "" {
		_ = os.RemoveAll(loopRoot)
	}
	os.Exit(code)
}

// qLoop prepares, caches, trains and exports instance Q once per test
// binary.
func qLoop(t *testing.T) *trainedLoop {
	t.Helper()
	loopOnce.Do(func() {
		l := &trainedLoop{}
		root, err := os.MkdirTemp("", "layago-train-loop-")
		if err != nil {
			loopErr = err
			return
		}
		loopRoot = root
		l.data = filepath.Join(root, "fixture.jsonl")
		if loopErr = os.WriteFile(l.data, fixtureBytes(t), 0o600); loopErr != nil {
			return
		}
		if l.ds, loopErr = LoadDataset([]string{l.data}, InstanceQ); loopErr != nil {
			return
		}
		spec := tinySpec
		spec.Dir = tinyTrunkDir
		if l.trunk, loopErr = OpenTrunk(spec); loopErr != nil {
			return
		}
		if l.p, loopErr = Prepare(l.ds, l.trunk, PrepareOptions{}); loopErr != nil {
			return
		}
		if l.cache, loopErr = BuildCache(context.Background(), l.trunk, l.p, CacheSpec{Root: filepath.Join(root, "cache"), ShardSize: 64, Workers: 4, Batch: 5}); loopErr != nil {
			return
		}
		if l.recs, loopErr = l.cache.Records(); loopErr != nil {
			return
		}
		if l.res, loopErr = Train(context.Background(), l.p, l.recs, l.trunk.Identity.HiddenSize, loopConfig(4)); loopErr != nil {
			return
		}
		l.dirs, loopErr = Export(filepath.Join(root, "artifact-q"), l.res, ExportMeta{Code: CodeVersion{GitCommit: "test"}, Trunk: l.trunk,
			Cache: l.cache})
		loopQ = l
	})
	if loopErr != nil {
		t.Fatal(loopErr)
	}
	return loopQ
}

// TestFullLoopOnTinyTrunk runs the whole pipeline on the tiny random trunk
// and the generated fixture: preparation (sealed locked_test, split
// invariants, duplicates), feature cache (resumable, independent of
// parallelism), training, calibration, thresholds, evaluation per split and
// language, export of both numeric paths, and served inference from the
// exported artifacts, which must reproduce the trainer's logits bit for bit.
func TestFullLoopOnTinyTrunk(t *testing.T) {
	ctx := context.Background()
	l := qLoop(t)
	p, res := l.p, l.res

	// Preparation: locked_test is sealed and shapes nothing.
	byRole := p.ItemsByRole()
	if len(byRole[SplitLockedTest]) != 0 || p.LockedTestEvaluated || len(p.Duplicates) != 0 {
		t.Fatalf("locked_test items %d, evaluated %v, duplicates %v", len(byRole[SplitLockedTest]), p.LockedTestEvaluated, p.Duplicates)
	}
	for _, role := range []string{SplitTrain, RoleDevCal, RoleDevConf, SplitHeldoutDomain, SplitHeldoutLanguage} {
		if len(byRole[role]) == 0 {
			t.Fatalf("no %s items: %v", role, res.Roles)
		}
	}
	stories := map[string]string{}
	for _, it := range p.Items {
		if it.Ex.Split == SplitDev {
			if prev, ok := stories[it.Ex.StoryID]; ok && prev != it.Role {
				t.Fatalf("story %s is in both %s and %s", it.Ex.StoryID, prev, it.Role)
			}
			stories[it.Ex.StoryID] = it.Role
		}
	}

	// Cache: features match a direct forward bit for bit, and the same key
	// built with other parallelism writes identical shards.
	for _, it := range []*Item{p.Items[0], p.Items[len(p.Items)/2], p.Items[len(p.Items)-1]} {
		f, err := l.trunk.Features(ctx, [][]int32{it.IDs}, [][]int{it.Markers}, tensor.Parallel{Workers: 1})
		if err != nil {
			t.Fatal(err)
		}
		r := l.recs[it.Ex.ID]
		if !slices.Equal(r.Markers, f[0].Markers) || !slices.Equal(r.Pooled, f[0].Pooled) || r.SeqLen != len(it.IDs) {
			t.Fatalf("%s: cached features differ from a single-sequence forward", it.Ex.ID)
		}
	}
	th := l.cache.Info.Throughput
	t.Logf("cache: %d items, sequence tokens p50 %d p95 %d max %d, %.0f items/s, %.0f tokens per CPU-second", l.cache.Info.Examples,
		l.cache.Info.SeqLenP50, l.cache.Info.SeqLenP95, l.cache.Info.SeqLenMax, th.ExamplesPerSecond, th.TokensPerCPUSecond)
	checkCacheDeterminismAndResume(t, l.trunk)

	// Training is bit-identical with one worker.
	res1, err := Train(ctx, p, l.recs, l.trunk.Identity.HiddenSize, loopConfig(1))
	if err != nil {
		t.Fatal(err)
	}
	for i, prm := range res.Float32.Head.Params() {
		if !slices.Equal(prm.Data, res1.Float32.Head.Params()[i].Data) {
			t.Fatalf("%s differs between 4 workers and 1", prm.Name)
		}
	}
	first, last := res.History[0], res.History[len(res.History)-1]
	t.Logf("epochs %d (best %d): train loss %.4f -> %.4f, dev_cal NLL %.4f -> %.4f", len(res.History), res.BestEpoch,
		first.TrainLoss, last.TrainLoss, first.DevCalNLL, res.History[res.BestEpoch-1].DevCalNLL)
	if last.TrainLoss > 0.9*first.TrainLoss {
		t.Fatalf("training loss did not fall: %.4f -> %.4f", first.TrainLoss, last.TrainLoss)
	}

	// Evaluation per split, language and task; held-out splits separate.
	ev := res.Float32.Evaluation
	for _, split := range []string{SplitTrain, RoleDevCal, RoleDevConf, SplitHeldoutDomain, SplitHeldoutLanguage} {
		if ev[split].All.N == 0 || ev[split].All.Accuracy == nil {
			t.Fatalf("evaluation has no %s split: %v", split, sortedKeys(ev))
		}
	}
	if _, ok := ev[SplitLockedTest]; ok {
		t.Fatal("sealed locked_test was evaluated")
	}
	hl := ev[SplitHeldoutLanguage]
	if hl.ByLang["ja"].N == 0 || hl.ByLang["ar"].N == 0 || hl.ByLangTask["ja/q_operation"].N == 0 {
		t.Fatalf("held-out languages: %v", sortedKeys(hl.ByLang))
	}
	if g := res.Float32.Generalise[SplitHeldoutLanguage]; g["unseen_language"].N != hl.All.N ||
		res.Float32.Generalise[SplitHeldoutDomain]["unseen_domain"].N != ev[SplitHeldoutDomain].All.N {
		t.Fatalf("generalisation groups: %v", res.Float32.Generalise)
	}
	if *ev[SplitTrain].All.Accuracy < 0.5 {
		t.Fatalf("train accuracy %.3f after training", *ev[SplitTrain].All.Accuracy)
	}
	for _, task := range sortedKeys(res.Float32.Thresholds) {
		th := res.Float32.Thresholds[task]
		if th.Delta != res.Config.Delta/2 || th.N != ev[RoleDevConf].ByTask[task].N {
			t.Fatalf("%s threshold %+v", task, th)
		}
	}
	summary, _ := json.Marshal(map[string]any{"float32": ev[RoleDevConf].All, "heldout_language": hl.All,
		"heldout_domain": ev[SplitHeldoutDomain].All, "int8_delta": res.Int8.Delta})
	t.Logf("%s", summary)

	// Export: one directory per numeric path, each a converged artifact.
	if len(l.dirs) != 2 || filepath.Base(l.dirs[0]) != "float32" || filepath.Base(l.dirs[1]) != "int8" {
		t.Fatalf("artifact directories %v", l.dirs)
	}
	for _, dir := range l.dirs {
		checkConvergedManifest(t, dir, tinyTrunkDir)
	}
	raw, err := os.ReadFile(filepath.Join(l.dirs[0], ArtifactManifest))
	if err != nil {
		t.Fatal(err)
	}
	var man struct {
		Training struct {
			Splits    map[string]map[string]any `json:"splits"`
			DevHalves DevHalfReport             `json:"dev_halves"`
			Locked    bool                      `json:"locked_test_evaluated"`
		} `json:"training"`
		Limitations []string `json:"limitations"`
	}
	if err := json.Unmarshal(raw, &man); err != nil {
		t.Fatal(err)
	}
	// A sealed locked_test is published as its size and files only.
	locked := man.Training.Splits[SplitLockedTest]
	if locked["sealed"] != true || locked["examples"] == nil || locked["files"] == nil || locked["languages"] != nil ||
		locked["ids_sha256"] != nil || man.Training.Locked {
		t.Fatalf("sealed locked_test record %v", locked)
	}
	halves := man.Training.DevHalves
	if halves.DevCal.Scorable != len(byRole[RoleDevCal]) || halves.DevConf.Scorable != len(byRole[RoleDevConf]) ||
		halves.Stories != halves.DevCal.Stories+halves.DevConf.Stories ||
		halves.CertificationStart != max(5, int(math.Ceil(3*math.Log(0.25)/math.Log1p(-0.5)))) {
		t.Fatalf("dev halves %+v", halves)
	}
	t.Logf("dev halves: %d/%d stories, %d/%d scorable; %d shortfalls", halves.DevCal.Stories, halves.DevConf.Stories,
		halves.DevCal.Scorable, halves.DevConf.Scorable, len(halves.Shortfalls))

	// Served inference from each artifact reproduces its trainer variant.
	acted := 0
	for i, dir := range l.dirs {
		v := []*Variant{res.Float32, res.Int8}[i]
		art, err := LoadArtifact(dir)
		if err != nil {
			t.Fatal(err)
		}
		for j, prm := range art.Head.Params() {
			if !slices.Equal(prm.Data, v.Head.Params()[j].Data) {
				t.Fatalf("%s: loaded %s differs from the trained head", dir, prm.Name)
			}
		}
		pred, err := NewPredictor(l.trunk, art, tensor.Parallel{Workers: 3})
		if err != nil {
			t.Fatal(err)
		}
		if !pred.KernelMatches {
			t.Fatal("same process, same kernel: KernelMatches must hold")
		}
		rows := map[string]Row{}
		for _, r := range v.Rows {
			rows[r.ID] = r
		}
		// Every item of the float32 artifact, every third of the int8 one.
		var exs []Example
		for k, it := range p.Items {
			if i == 0 || k%3 == 0 {
				exs = append(exs, *it.Ex)
			}
		}
		for b := 0; b < len(exs); b += 41 {
			batch := exs[b:min(len(exs), b+41)]
			out, err := pred.Predict(ctx, batch)
			if err != nil {
				t.Fatal(err)
			}
			for k, pr := range out {
				id := batch[k].ID
				r := rows[id]
				if !slices.Equal(pr.Logits, v.Logits[id]) || !slices.Equal(pr.ActLogits, v.ActLogits[id]) {
					t.Fatalf("%s %s: served logits %v, trainer %v", v.Format, id, pr.Logits, v.Logits[id])
				}
				if !slices.Equal(pr.Probabilities, r.Probs) || (pr.Decision == "act") != r.Acted || pr.Prediction != r.PredKey ||
					pr.TemperatureKey != r.TemperatureKey {
					t.Fatalf("%s %s: served decision differs from the trainer's", v.Format, id)
				}
				if pr.Decision == "act" && i == 0 {
					acted++
				}
			}
		}
		g, err := pred.VerifyGolden(ctx)
		if err != nil || g.Exact != g.Cases || g.Cases == 0 {
			t.Fatalf("%s parity golden: %+v, %v", v.Format, g, err)
		}
	}
	t.Logf("served %d items from each artifact: logits bit-identical to training, %d acted (float32)", len(p.Items), acted)
	if acted == 0 || acted == len(p.Items) {
		t.Fatalf("%d of %d acted: the test's loose alpha is meant to exercise both decisions", acted, len(p.Items))
	}

	// The exported head is a Laya decision head with head_layers = 0: the
	// Go Laya runtime, given the trunk's encoder tensors plus the float32
	// artifact's head tensors as a checkpoint, lays out the same sequences
	// and scores the same logits within the golden tolerance.
	agent := layaCheckpointFromArtifact(t, l.dirs[0], l.trunk)
	var worstLogit, worstAct float64
	checked := 0
	for i, it := range p.Items {
		if !it.Scorable || i%4 != 0 {
			continue
		}
		lp, err := agent.Predict(ctx, it.State, []laya.Question{it.Question}, model.Options{})
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(lp.Sequences[0], it.IDs) || !slices.Equal(lp.Markers[0], it.Markers) {
			t.Fatalf("%s: laya.Agent laid out a different sequence", it.Ex.ID)
		}
		for j, v := range lp.Answers[0].Logits {
			worstLogit = math.Max(worstLogit, math.Abs(float64(v)-float64(res.Float32.Logits[it.Ex.ID][j])))
		}
		worstAct = math.Max(worstAct, math.Abs(lp.Answers[0].ActProbability-ActProbability(res.Float32.ActLogits[it.Ex.ID])))
		checked++
	}
	t.Logf("laya.Agent (head_layers=0) on %d items: logits within %.2e, act probability within %.2e", checked, worstLogit, worstAct)
	if checked == 0 || worstLogit > GoldenLogitAbs || worstAct > GoldenProbAbs {
		t.Fatalf("laya.Agent disagrees with the trained head: logits %.2e, act %.2e over %d", worstLogit, worstAct, checked)
	}

	// Tampering and a mismatched trunk are refused.
	w := filepath.Join(t.TempDir(), "tampered")
	copyDir(t, l.dirs[0], w)
	weights, _ := os.ReadFile(filepath.Join(w, ArtifactWeights))
	weights[len(weights)-1] ^= 1
	if err := os.WriteFile(filepath.Join(w, ArtifactWeights), weights, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadArtifact(w); err == nil || !strings.Contains(err.Error(), "sha256") {
		t.Fatalf("tampered weights: %v", err)
	}
	art, err := LoadArtifact(l.dirs[0])
	if err != nil {
		t.Fatal(err)
	}
	other := tinyTrunk(t, TrunkSpec{MaxLen: 200, HeadMaxLen: 160})
	if _, err := NewPredictor(other, art, tensor.Parallel{}); err == nil || !strings.Contains(err.Error(), "max_len") {
		t.Fatalf("mismatched trunk: %v", err)
	}
	shallow := tinyTrunk(t, TrunkSpec{MaxLen: tinySpec.MaxLen, HeadMaxLen: tinySpec.HeadMaxLen, Layers: 3})
	if _, err := NewPredictor(shallow, art, tensor.Parallel{}); err == nil || !strings.Contains(err.Error(), "layers_used") {
		t.Fatalf("shallower trunk: %v", err)
	}
	// A fixed-option task asked with other options is refused.
	var op Example
	for _, it := range p.Items {
		if it.Ex.Task == "q_operation" {
			op = *it.Ex
			break
		}
	}
	op.Options = slices.Clone(op.Options)
	op.Options[0].Text = "something else"
	pred, _ := NewPredictor(l.trunk, art, tensor.Parallel{})
	if _, err := pred.Predict(ctx, []Example{op}); err == nil || !strings.Contains(err.Error(), "fixed options") {
		t.Fatalf("other fixed options: %v", err)
	}
}

// checkCacheDeterminismAndResume builds a small fixture's cache twice with
// different parallelism (identical shards, one key), then resumes it after
// deleting one shard and corrupting another.
func checkCacheDeterminismAndResume(t *testing.T, trunk *Trunk) {
	t.Helper()
	ctx := context.Background()
	ds, err := ParseDataset([]DataFile{{Name: "small.jsonl", Data: encodeLines(t, fixtureLines(2))}}, InstanceQ)
	if err != nil {
		t.Fatal(err)
	}
	p, err := Prepare(ds, trunk, PrepareOptions{})
	if err != nil {
		t.Fatal(err)
	}
	a, err := BuildCache(ctx, trunk, p, CacheSpec{Root: t.TempDir(), ShardSize: 40, Workers: 4, Batch: 5})
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	b, err := BuildCache(ctx, trunk, p, CacheSpec{Root: root, ShardSize: 40, Workers: 1, Batch: 1})
	if err != nil {
		t.Fatal(err)
	}
	if a.Info.Key != b.Info.Key || a.Info.Shards < 3 {
		t.Fatalf("cache keys %s and %s (%d shards)", a.Info.Key, b.Info.Key, a.Info.Shards)
	}
	same := func(what string) {
		for s := 0; s < a.Info.Shards; s++ {
			x, _ := os.ReadFile(shardPath(a.Dir, s))
			y, _ := os.ReadFile(shardPath(b.Dir, s))
			if !bytes.Equal(x, y) {
				t.Fatalf("shard %d differs %s", s, what)
			}
		}
	}
	same("between 4 workers x 5 sequences and 1 x 1")
	want, err := a.Records()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(shardPath(b.Dir, 1)); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(shardPath(b.Dir, 2), []byte("not a shard"), 0o600); err != nil {
		t.Fatal(err)
	}
	resumed, err := BuildCache(ctx, trunk, p, CacheSpec{Root: root, ShardSize: 40, Workers: 2})
	if err != nil {
		t.Fatal(err)
	}
	th := resumed.Info.Throughput
	if th.ResumedShards != a.Info.Shards-2 || th.Examples < 41 || th.Examples+th.ReusedSequences != a.Info.Sequences {
		t.Fatalf("resume reused %d shards and %d sequences and encoded %d (store of %d shards, %d sequences)", th.ResumedShards,
			th.ReusedSequences, th.Examples, a.Info.Shards, a.Info.Sequences)
	}
	if _, err := os.Stat(shardPath(b.Dir, 2)); !os.IsNotExist(err) {
		t.Fatalf("the corrupt shard was kept: %v", err)
	}
	got, err := resumed.Records()
	if err != nil {
		t.Fatal(err)
	}
	sameRecords(t, want, got, "after resume")
	again, err := BuildCache(ctx, trunk, p, CacheSpec{Root: root, ShardSize: 40})
	if err != nil {
		t.Fatal(err)
	}
	if again.Info.Throughput != resumed.Info.Throughput {
		t.Fatalf("a full resume replaced the measured throughput: %+v", again.Info.Throughput)
	}
	opened, err := OpenCache(again.Dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := opened.Bind(p); err != nil {
		t.Fatal(err)
	}
	if opened.Info.Throughput != resumed.Info.Throughput || opened.Info.SequenceDigest != p.SequenceDigest {
		t.Fatalf("a bound store reports %+v", opened.Info)
	}
}

func sameRecords(t *testing.T, want, got map[string]*Record, what string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%d records, want %d %s", len(got), len(want), what)
	}
	for id, w := range want {
		g := got[id]
		if g == nil || g.K != w.K || g.SeqLen != w.SeqLen || !slices.Equal(g.Pooled, w.Pooled) || !slices.Equal(g.Markers, w.Markers) {
			t.Fatalf("record %s differs %s", id, what)
		}
	}
}

// TestNestedSubsamplesReuseStoredFeatures grows a train subsample over one
// store: the larger subsample encodes only the sequences it adds, every
// shared item reads the same features, and binding a view the store cannot
// serve fails.
func TestNestedSubsamplesReuseStoredFeatures(t *testing.T) {
	ctx := context.Background()
	trunk := tinyTrunk(t, tinySpec)
	// Six stories per cell put four in train, so a 0.3 subsample keeps two
	// of each stratum's four.
	ds, err := ParseDataset([]DataFile{{Name: "small.jsonl", Data: encodeLines(t, fixtureLines(6))}}, InstanceQ)
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	half, err := Prepare(ds, trunk, PrepareOptions{TrainFraction: 0.3, SubsampleSeed: 3, AllowDuplicates: true})
	if err != nil {
		t.Fatal(err)
	}
	whole, err := Prepare(ds, trunk, PrepareOptions{AllowDuplicates: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(whole.Items) <= len(half.Items) {
		t.Fatalf("the subsample keeps %d of %d items", len(half.Items), len(whole.Items))
	}
	ch, err := BuildCache(ctx, trunk, half, CacheSpec{Root: root, ShardSize: 16})
	if err != nil {
		t.Fatal(err)
	}
	opened, err := OpenCache(ch.Dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := opened.Bind(whole); err == nil || !strings.Contains(err.Error(), "no features") {
		t.Fatalf("binding a larger view before building it: %v", err)
	}
	// While another build holds the store, a stored view still builds (it
	// only reads) and a view that needs encoding is refused.
	unlock, err := lockStore(ch.Dir)
	if err != nil {
		t.Fatal(err)
	}
	if again, err := BuildCache(ctx, trunk, half, CacheSpec{Root: root, ShardSize: 16}); err != nil || again.Encoded != 0 ||
		again.Reused != ch.Info.Sequences {
		t.Fatalf("a stored view beside a writer: %v", err)
	}
	if _, err := BuildCache(ctx, trunk, whole, CacheSpec{Root: root, ShardSize: 16}); err == nil ||
		!strings.Contains(err.Error(), "another build") {
		t.Fatalf("encoding beside a writer: %v", err)
	}
	unlock()
	cw, err := BuildCache(ctx, trunk, whole, CacheSpec{Root: root, ShardSize: 16})
	if err != nil {
		t.Fatal(err)
	}
	if ch.Dir != cw.Dir || cw.Info.Throughput.ReusedSequences != ch.Info.Sequences ||
		cw.Info.Throughput.Examples != cw.Info.Sequences-ch.Info.Sequences {
		t.Fatalf("the whole view encoded %d and reused %d of %d sequences; the half view has %d", cw.Info.Throughput.Examples,
			cw.Info.Throughput.ReusedSequences, cw.Info.Sequences, ch.Info.Sequences)
	}
	hr, err := ch.Records()
	if err != nil {
		t.Fatal(err)
	}
	wr, err := cw.Records()
	if err != nil {
		t.Fatal(err)
	}
	sub := map[string]*Record{}
	for id := range hr {
		sub[id] = wr[id]
	}
	sameRecords(t, hr, sub, "between the nested views")
	fresh, err := BuildCache(ctx, trunk, whole, CacheSpec{Root: t.TempDir(), ShardSize: 16})
	if err != nil {
		t.Fatal(err)
	}
	fr, err := fresh.Records()
	if err != nil {
		t.Fatal(err)
	}
	sameRecords(t, fr, wr, "between a grown store and a fresh one")
}

func copyDir(t *testing.T, src, dst string) {
	t.Helper()
	if err := os.MkdirAll(dst, 0o750); err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(src)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		b, err := os.ReadFile(filepath.Join(src, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dst, e.Name()), b, 0o600); err != nil {
			t.Fatal(err)
		}
	}
}

// layaCheckpointFromArtifact writes a Laya checkpoint directory holding the
// tiny trunk's encoder tensors and the artifact's head tensors, declares
// head_layers 0 and the trunk's sequence limits, and loads it with
// laya.LoadCheckpoint.
func layaCheckpointFromArtifact(t *testing.T, artDir string, trunk *Trunk) *laya.Agent {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "laya")
	for _, sub := range []string{"encoder", "tokenizer"} {
		if err := os.MkdirAll(filepath.Join(dir, sub), 0o750); err != nil {
			t.Fatal(err)
		}
	}
	for _, rel := range []string{"encoder/config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"} {
		b, err := os.ReadFile(filepath.Join(tinyTrunkDir, rel))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, rel), b, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	var tensors []safetensors.NamedTensor
	for _, src := range []struct{ path, prefix string }{
		{filepath.Join(tinyTrunkDir, "model.safetensors"), "encoder."}, {filepath.Join(artDir, ArtifactWeights), ""},
	} {
		f, err := safetensors.Open(src.path)
		if err != nil {
			t.Fatal(err)
		}
		for _, name := range f.Header.Names() {
			if !strings.HasPrefix(name, src.prefix) {
				continue
			}
			tt, err := f.Tensor(name)
			if err != nil {
				t.Fatal(err)
			}
			tensors = append(tensors, safetensors.NamedTensor{Name: name, DType: tt.DType, Shape: tt.Shape, Data: append([]byte(nil), tt.Data...)})
		}
		_ = f.Close()
	}
	var buf bytes.Buffer
	if err := safetensors.Write(&buf, tensors, nil); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), buf.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, _ := json.Marshal(map[string]any{"encoder": "tiny", "head_layers": 0, "max_len": trunk.Spec.MaxLen,
		"head_max_len": trunk.Spec.HeadMaxLen, "act_costs": map[string]float64{"escalate": 0.5}})
	if err := os.WriteFile(filepath.Join(dir, "rl_agent_config.json"), cfg, 0o600); err != nil {
		t.Fatal(err)
	}
	agent, err := laya.LoadCheckpoint(dir, laya.LoadOptions{})
	if err != nil {
		t.Fatal(err)
	}
	return agent
}

// TestInstanceRTrainsListwiseAndScoreTasks trains instance R: graded
// evidence relevance (score levels with the ordinal regression term) and
// answer ranking over evidence items with label_dist soft labels (the
// listwise loss), then serves the exported artifact bit for bit.
func TestInstanceRTrainsListwiseAndScoreTasks(t *testing.T) {
	ctx := context.Background()
	trunk := tinyTrunk(t, tinySpec)
	ds := fixtureDataset(t, InstanceR)
	soft := 0
	for _, ex := range ds.Examples {
		if ex.LabelDist != nil {
			soft++
		}
	}
	if soft == 0 {
		t.Fatal("the R fixture carries no label_dist")
	}
	p, err := Prepare(ds, trunk, PrepareOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if p.Tasks["r_evidence_relevance"].Options != "fixed" || p.Tasks["r_answer_rank"].Options != "per_example" {
		t.Fatalf("tasks %+v %+v", p.Tasks["r_evidence_relevance"], p.Tasks["r_answer_rank"])
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
	if res.Int8 != nil || res.History[len(res.History)-1].TrainLoss >= res.History[0].TrainLoss {
		t.Fatalf("history %+v", res.History)
	}
	dirs, err := Export(t.TempDir(), res, ExportMeta{Code: CodeVersion{GitCommit: "test"}, Trunk: trunk, Cache: cache})
	if err != nil || len(dirs) != 1 {
		t.Fatalf("%v %v", dirs, err)
	}
	checkConvergedManifest(t, dirs[0], tinyTrunkDir)
	art, err := LoadArtifact(dirs[0])
	if err != nil {
		t.Fatal(err)
	}
	pred, err := NewPredictor(trunk, art, tensor.Parallel{})
	if err != nil {
		t.Fatal(err)
	}
	if g, err := pred.VerifyGolden(ctx); err != nil || g.Exact != g.Cases {
		t.Fatalf("golden %+v %v", g, err)
	}
	summary, _ := json.Marshal(res.Float32.Evaluation[RoleDevConf].ByTask)
	t.Logf("R dev_conf by task: %s", summary)
}

// TestPrepareGuardsSplitsAndLockedTest: split-invariant violations and
// rendered duplicates are refused unless allowed (and then recorded), and
// locked_test is read only when unsealed with an access log.
func TestPrepareGuardsSplitsAndLockedTest(t *testing.T) {
	trunk := tinyTrunk(t, tinySpec)
	lines := fixtureLines(3)
	// Move one dev story's examples into train: its story is then in two
	// splits.
	var moved string
	for _, l := range lines {
		if l["split"] == SplitDev && l["instance"] == InstanceQ {
			if moved == "" {
				moved = l["story_id"].(string)
			}
			if l["story_id"] == moved && l["task"] == "q_operation" {
				l["split"] = SplitTrain
			}
		}
	}
	ds, err := ParseDataset([]DataFile{{Name: "leaky.jsonl", Data: encodeLines(t, lines)}}, InstanceQ)
	if err != nil {
		t.Fatal(err)
	}
	if len(ds.SplitViolations) != 1 || !strings.Contains(ds.SplitViolations[0], moved) {
		t.Fatalf("violations %v", ds.SplitViolations)
	}
	if _, err := Prepare(ds, trunk, PrepareOptions{}); err == nil || !strings.Contains(err.Error(), "split invariant") {
		t.Fatalf("leaky dataset: %v", err)
	}
	if _, err := Prepare(ds, trunk, PrepareOptions{AllowSplitLeakage: true}); err != nil {
		t.Fatal(err)
	}
	if got := limitations(&Prepared{Dataset: ds}, DevHalfReport{}); !strings.Contains(strings.Join(got, "\n"), "split invariants overridden (1 violations") {
		t.Fatalf("limitations %v", got)
	}

	clean, err := ParseDataset([]DataFile{{Name: "f.jsonl", Data: encodeLines(t, fixtureLines(3))}}, InstanceQ)
	if err != nil {
		t.Fatal(err)
	}
	// A rendered duplicate of a train example in dev is refused.
	var dup Example
	for _, ex := range clean.Examples {
		if ex.Split == SplitTrain {
			dup = ex
			break
		}
	}
	dup.ID, dup.Split, dup.StoryID = "dup-1", SplitDev, "dup-story"
	withDup := *clean
	withDup.Examples = append(slices.Clone(clean.Examples), dup)
	if _, err := Prepare(&withDup, trunk, PrepareOptions{}); err == nil || !strings.Contains(err.Error(), "reappear") {
		t.Fatalf("duplicate: %v", err)
	}

	// A dev whose stories all hash to dev_cal leaves dev_conf empty: Train
	// refuses it before reading any feature.
	oneSided := *clean
	oneSided.Examples = nil
	for _, ex := range clean.Examples {
		if ex.Split != SplitDev || DevHalf(ex.StoryID) == RoleDevCal {
			oneSided.Examples = append(oneSided.Examples, ex)
		}
	}
	p1, err := Prepare(&oneSided, trunk, PrepareOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Train(context.Background(), p1, nil, trunk.Identity.HiddenSize, loopConfig(0)); err == nil ||
		!strings.Contains(err.Error(), "dev_conf has no scorable example") {
		t.Fatalf("one-sided dev: %v", err)
	}

	if _, err := Prepare(clean, trunk, PrepareOptions{UnsealLockedTest: true}); err == nil || !strings.Contains(err.Error(), "access log") {
		t.Fatalf("unsealed without a log: %v", err)
	}
	logPath := filepath.Join(t.TempDir(), "access", "locked_test.jsonl")
	p, err := Prepare(clean, trunk, PrepareOptions{UnsealLockedTest: true, AccessLog: logPath, AccessNote: "test"})
	if err != nil {
		t.Fatal(err)
	}
	if !p.LockedTestEvaluated || len(p.ItemsByRole()[SplitLockedTest]) == 0 {
		t.Fatalf("unsealed locked_test: %v", p.LockedTestEvaluated)
	}
	f, err := os.Open(logPath) //nolint:gosec // G304: test temp file
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	sc := bufio.NewScanner(f)
	var rec map[string]any
	n := 0
	for sc.Scan() {
		n++
		if err := json.Unmarshal(sc.Bytes(), &rec); err != nil {
			t.Fatal(err)
		}
	}
	if n != 1 || rec["event"] != "locked_test_evaluated" || rec["dataset_digest"] != clean.Digest() {
		t.Fatalf("access log: %d lines, %v", n, rec)
	}
	if _, err := Prepare(&Dataset{Examples: clean.Examples}, trunk, PrepareOptions{}); err == nil || !strings.Contains(err.Error(), "one instance") {
		t.Fatalf("mixed instances: %v", err)
	}
}

// TestUnscorableItemsEscalateAndAreCounted: a budget too small for the
// options makes items unscorable; they are never encoded or trained on,
// and escalate in evaluation.
func TestUnscorableItemsEscalateAndAreCounted(t *testing.T) {
	// Options shrink to 4 tokens each after a head of at least 8, so the
	// fourth marker sits at position 22: at max_len 20 four-option questions
	// lose it, while two- and three-option questions still fit.
	trunk := tinyTrunk(t, TrunkSpec{MaxLen: 20, HeadMaxLen: 12})
	ds := fixtureDataset(t, InstanceQ)
	p, err := Prepare(ds, trunk, PrepareOptions{})
	if err != nil {
		t.Fatal(err)
	}
	cache, err := BuildCache(context.Background(), trunk, p, CacheSpec{Root: t.TempDir(), DType: "f16", ShardSize: 200})
	if err != nil {
		t.Fatal(err)
	}
	recs, err := cache.Records()
	if err != nil {
		t.Fatal(err)
	}
	if len(cache.Info.Skipped) == 0 || len(recs)+len(cache.Info.Skipped) != len(p.Items) ||
		!strings.Contains(cache.Info.Skipped[0].Reason, "markers truncated") {
		t.Fatalf("%d records, %d skipped of %d", len(recs), len(cache.Info.Skipped), len(p.Items))
	}
	cfg := loopConfig(0)
	cfg.Epochs, cfg.Int8 = 2, false
	res, err := Train(context.Background(), p, recs, trunk.Identity.HiddenSize, cfg)
	if err != nil {
		t.Fatal(err)
	}
	var unscorable, escalated int
	for _, r := range res.Float32.Rows {
		if !r.Scorable {
			unscorable++
			if !r.Acted {
				escalated++
			}
		}
	}
	if unscorable != len(cache.Info.Skipped) || escalated != unscorable || sumValues(res.Unscorable) != unscorable {
		t.Fatalf("unscorable %d, escalated %d, counted %v", unscorable, escalated, res.Unscorable)
	}
	if res.Float32.Evaluation[RoleDevConf].All.UnscorableRate == 0 {
		t.Fatalf("dev_conf %+v", res.Float32.Evaluation[RoleDevConf].All)
	}
}
