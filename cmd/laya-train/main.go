// Command laya-train validates datasets, prepares caches, trains and evaluates frozen-encoder heads.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"github.com/UnicoLab/ul_layago/tensor"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"reflect"
	"runtime"
	"runtime/debug"
	"syscall"
	"time"

	"github.com/UnicoLab/ul_layago/train"
)

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	if err := run(ctx, os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(ctx context.Context, args []string, output io.Writer) error {
	if len(args) == 0 {
		return errors.New("usage: laya-train validate|inspect|train|evaluate --dataset FILE [flags]")
	}
	action := args[0]
	if action != "validate" && action != "inspect" && action != "train" && action != "evaluate" {
		return fmt.Errorf("unknown action %q", action)
	}
	fs := flag.NewFlagSet(action, flag.ContinueOnError)
	dataset := fs.String("dataset", "", "JSONL dataset in the compatible hive.laya-dataset/v1 format")
	instance := fs.String("instance", "", "Q (question), R (evidence), A (draft); required except validate")
	base := fs.String("base", "", "Local frozen ModernBERT/mmBERT or Laya encoder")
	cacheRoot := fs.String("cache", "", "Feature-cache directory (required for train/evaluate)")
	out := fs.String("out", "", "New output directory for exported artifacts")
	artifact := fs.String("artifact", "", "Existing head artifact for evaluate")
	initArtifact := fs.String("init-artifact", "", "Warm-start a compatible trained head; recalibrate from current dev data")
	memoryLimit := fs.Int64("memory-limit", 4<<30, "Soft Go memory budget in bytes; does not bound mmap or guarantee against OOM")
	duration := fs.Duration("max-duration", 30*time.Minute, "Maximum run duration")
	cacheBatch := fs.Int("cache-batch", 2, "Sequences per cache forward pass")
	cacheWorkers := fs.Int("cache-workers", 1, "Concurrent cache batches; head workers are separate")
	shardSize := fs.Int("cache-shard-size", 32, "Sequences per durable cache shard")
	configFile := fs.String("config", "", "JSON training config overrides over DefaultConfig")
	instructionsFile := fs.String("instructions", "", "JSON mapping of task names to training instructions")
	workers := fs.Int("workers", 1, "Positive cache/training worker count")
	layers := fs.Int("layers", 0, "Keep first N encoder layers; 0 keeps all")
	maxLen := fs.Int("max-len", 512, "Model sequence budget")
	headLen := fs.Int("head-max-len", 192, "Instruction/options budget")
	unseal := fs.Bool("unseal-locked-test", false, "Evaluate sealed test split; requires access-log, only valid for evaluate")
	accessLog := fs.String("access-log", "", "Append-only log required for locked-test evaluation")
	if err := fs.Parse(args[1:]); err != nil {
		return err
	}
	if fs.NArg() != 0 || *dataset == "" || *workers <= 0 {
		return errors.New("dataset is required, workers must be positive, positional arguments are not accepted")
	}
	if *unseal && (action != "evaluate" || *accessLog == "") {
		return errors.New("locked_test may only be unsealed by evaluate with an access log")
	}
	if *memoryLimit <= 0 || *duration <= 0 || *cacheBatch <= 0 || *cacheWorkers <= 0 || *shardSize <= 0 {
		return errors.New("memory limit, duration and cache sizes must be positive")
	}
	previousLimit := debug.SetMemoryLimit(*memoryLimit)
	defer debug.SetMemoryLimit(previousLimit)
	ctx, cancel := context.WithTimeout(ctx, *duration)
	defer cancel()
	heartbeatDone := make(chan struct{})
	defer close(heartbeatDone)
	if action == "train" || action == "evaluate" {
		go func() {
			ticker := time.NewTicker(5 * time.Second)
			defer ticker.Stop()
			for {
				select {
				case <-heartbeatDone:
					return
				case <-ticker.C:
					var m runtime.MemStats
					runtime.ReadMemStats(&m)
					fmt.Fprintf(os.Stderr, "health heap_bytes=%d goroutines=%d soft_memory_limit=%d\n", m.HeapAlloc, runtime.NumGoroutine(), *memoryLimit)
				}
			}
		}()
	}
	ds, err := train.LoadDataset([]string{*dataset}, *instance)
	if err != nil {
		return err
	}
	if len(ds.SplitViolations) > 0 {
		return fmt.Errorf("split invariants violated: %v", ds.SplitViolations)
	}
	encode := func(value any) error {
		enc := json.NewEncoder(output)
		enc.SetIndent("", "  ")
		return enc.Encode(value)
	}
	if action == "validate" {
		return encode(map[string]any{"contract": train.DatasetContract, "valid": true, "examples": len(ds.Examples), "instance": ds.Instance, "files": ds.Files})
	}
	if *instance == "" || *base == "" {
		return errors.New("instance and base are required")
	}
	var instructions map[string]string
	if *instructionsFile != "" {
		if err := readJSON(*instructionsFile, &instructions); err != nil {
			return err
		}
	}
	var loaded *train.Artifact
	if action == "evaluate" {
		if *artifact == "" {
			return errors.New("artifact is required")
		}
		loaded, err = train.LoadArtifact(*artifact)
		if err != nil {
			return err
		}
		for task, instruction := range instructions {
			spec := loaded.Manifest.Tasks[task]
			if spec == nil || spec.Instructions != instruction {
				return fmt.Errorf("instructions for %q differ from the trained artifact", task)
			}
		}
		instructions = map[string]string{}
		for task, spec := range loaded.Manifest.Tasks {
			instructions[task] = spec.Instructions
		}
	}
	trunk, err := train.OpenTrunk(train.TrunkSpec{Dir: *base, Layers: *layers, MaxLen: *maxLen, HeadMaxLen: *headLen})
	if err != nil {
		return err
	}
	prepared, err := train.Prepare(ds, trunk, train.PrepareOptions{Instructions: instructions, UnsealLockedTest: *unseal, AccessLog: *accessLog, AccessNote: action + ":" + *artifact})
	if err != nil {
		return err
	}
	sequence, err := train.SequenceReport(prepared, trunk)
	if err != nil {
		return err
	}
	if action == "inspect" {
		return encode(sequence)
	}
	if loaded != nil {
		if _, err := train.NewPredictor(trunk, loaded, tensor.Parallel{Workers: *workers}); err != nil {
			return err
		}
		for task, spec := range prepared.Tasks {
			trained := loaded.Manifest.Tasks[task]
			if trained == nil || spec.Type != trained.Type || (trained.Options == "fixed" && !equalJSON(spec.FixedOptions, trained.FixedOptions)) {
				return fmt.Errorf("task %q has incompatible type or options", task)
			}
		}
	}
	if *cacheRoot == "" {
		return errors.New("cache is required")
	}
	if action == "train" {
		if *out == "" {
			return errors.New("out is required")
		}
		if _, err := os.Stat(*out); !errors.Is(err, os.ErrNotExist) {
			return errors.New("out must be a new directory; existing artifacts are never overwritten")
		}
	}
	cache, err := train.BuildCache(ctx, trunk, prepared, train.CacheSpec{Root: *cacheRoot, Workers: *cacheWorkers, IntraOp: *workers, Batch: *cacheBatch, ShardSize: *shardSize, Progress: func(p train.CacheProgress) {
		fmt.Fprintf(os.Stderr, "cache examples=%d/%d encoded=%d/%d elapsed=%s\n", p.ExamplesDone, p.Total, p.Encoded, p.Encode, p.Elapsed)
	}})
	if err != nil {
		return err
	}
	records, err := cache.Records()
	if err != nil {
		return err
	}
	if action == "evaluate" {
		if *artifact == "" {
			return errors.New("artifact is required")
		}
		a := loaded
		// Verify weights, tokenizer and sequence identity before scoring cached features.
		if _, err := train.NewPredictor(trunk, a, tensor.Parallel{Workers: *workers}); err != nil {
			return err
		}
		scored, err := a.ScoreCached(prepared, records, *workers)
		if err != nil {
			return err
		}
		rows := make([]train.Row, len(scored))
		for i, item := range scored {
			rows[i] = item.Row
		}
		return encode(map[string]any{"metrics": train.Evaluate(rows), "sequence_report": sequence, "requires_domain_validation": true})
	}
	cfg := train.DefaultConfig()
	cfg.Int8 = false
	if *configFile != "" {
		if err := readJSON(*configFile, &cfg); err != nil {
			return err
		}
	}
	if *initArtifact != "" {
		init, err := train.LoadArtifact(*initArtifact)
		if err != nil {
			return err
		}
		if _, err := train.NewPredictor(trunk, init, tensor.Parallel{Workers: *workers}); err != nil {
			return err
		}
		cfg.InitHead = init.Head
		cfg.InitWeightsSHA256 = init.Manifest.Weights.SHA256
	}
	cfg.Workers = *workers
	cfg.Log = func(format string, args ...any) { fmt.Fprintf(os.Stderr, format+"\n", args...) }
	result, err := train.Train(ctx, prepared, records, trunk.Identity.HiddenSize, cfg)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(*out), 0755); err != nil {
		return err
	}
	if err := os.Mkdir(*out, 0755); err != nil {
		return fmt.Errorf("reserve new artifact output: %w", err)
	}
	paths, err := train.Export(*out, result, train.ExportMeta{Code: train.CodeVersion{Trainer: train.TrainerVersion, GitCommit: buildRevision(), GitDirty: buildDirty(), GoVersion: runtime.Version()}, Trunk: trunk, Cache: cache})
	if err != nil {
		return err
	}
	return encode(map[string]any{"artifacts": paths, "best_epoch": result.BestEpoch, "seconds": result.Seconds, "roles": result.Roles, "sequence_report": sequence, "requires_domain_validation": true})
}

func readJSON(path string, destination any) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if len(bytes.TrimSpace(raw)) == 0 || bytes.TrimSpace(raw)[0] != '{' {
		return errors.New("configuration must be a JSON object")
	}
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	decoder := json.NewDecoder(f)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return err
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		return errors.New("expected one JSON configuration object")
	}
	return nil
}

func buildRevision() string {
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, s := range info.Settings {
			if s.Key == "vcs.revision" {
				return s.Value
			}
		}
	}
	return "unknown"
}
func buildDirty() bool {
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, s := range info.Settings {
			if s.Key == "vcs.modified" {
				return s.Value == "true"
			}
		}
	}
	return true
}

func equalJSON(a, b json.RawMessage) bool {
	var x, y any
	if json.Unmarshal(a, &x) != nil || json.Unmarshal(b, &y) != nil {
		return false
	}
	return reflect.DeepEqual(x, y)
}
