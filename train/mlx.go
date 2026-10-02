package train

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"slices"

	"github.com/UnicoLab/ul_layago/laya"
	"github.com/UnicoLab/ul_layago/safetensors"
)

type MLXOptions struct {
	Python, Script, WorkDir string
	MemoryLimit             int64
}

// MLXScorer delegates only scorer fitting to the optional local MLX worker.
// It fails explicitly when unavailable; there is no silent device/CPU fallback.
func MLXScorer(opts MLXOptions) ScorerFitFunc {
	return func(ctx context.Context, initial *Head, fitting, development []FitSample, cfg Config) (*ScorerFit, error) {
		if err := cfg.validate(); err != nil {
			return nil, err
		}
		if len(fitting) == 0 || len(development) == 0 {
			return nil, fmt.Errorf("mlx: fitting and development samples required")
		}
		if opts.Python == "" || opts.Script == "" || opts.MemoryLimit <= 0 {
			return nil, fmt.Errorf("mlx: python, script and positive memory limit are required")
		}
		dir := opts.WorkDir
		var err error
		if dir == "" {
			dir, err = os.MkdirTemp("", "layago-mlx-*")
			if err != nil {
				return nil, err
			}
			defer os.RemoveAll(dir)
		} else {
			if err = os.MkdirAll(dir, 0700); err != nil {
				return nil, err
			}
		}
		unlock, err := lockStore(dir)
		if err != nil {
			return nil, err
		}
		defer unlock()
		marker := filepath.Join(dir, "workspace.json")
		raw, readErr := os.ReadFile(marker)
		if os.IsNotExist(readErr) {
			entries, err := os.ReadDir(dir)
			if err != nil {
				return nil, err
			}
			for _, entry := range entries {
				if entry.Name() != ".lock" {
					return nil, fmt.Errorf("mlx: workspace is not empty or owned by LayaGo: %s", dir)
				}
			}
			if err = os.WriteFile(marker, []byte(`{"contract":"ul.layago-mlx-workspace/v1"}`), 0600); err != nil {
				return nil, err
			}
		} else if readErr != nil {
			return nil, readErr
		} else {
			var meta struct {
				Contract string `json:"contract"`
			}
			if json.Unmarshal(raw, &meta) != nil || meta.Contract != "ul.layago-mlx-workspace/v1" {
				return nil, fmt.Errorf("mlx: unrecognized workspace %s", dir)
			}
		}
		// Transfer binary tensors once per run, never multi-megabyte JSON gradients per step.
		tensors := []safetensors.NamedTensor{}
		for _, p := range initial.Params()[:scorerParams] {
			tensors = append(tensors, safetensors.Float32Tensor(p.Name, p.Shape, p.Data))
		}
		groups := map[string][]FitSample{"train": fitting, "dev": development}
		var estimate int64
		for _, p := range initial.Params()[:scorerParams] {
			estimate += int64(len(p.Data)) * 4 * 8
		}
		for name, group := range groups {
			maxK := 0
			for _, s := range group {
				maxK = max(maxK, s.K)
			}
			estimate += int64(len(group)) * int64(maxK) * int64(initial.Config.D) * 4 * 3
			if estimate > opts.MemoryLimit {
				return nil, fmt.Errorf("mlx: padded tensors/optimizer estimated %d bytes exceed budget %d; reduce corpus or options", estimate, opts.MemoryLimit)
			}
			markers := make([]float32, len(group)*maxK*initial.Config.D)
			target := make([]float32, len(group)*maxK)
			mask := make([]float32, len(group)*maxK)
			meta := make([]float32, len(group)*5)
			for i, s := range group {
				copy(markers[i*maxK*initial.Config.D:], s.Markers)
				for j := 0; j < s.K; j++ {
					target[i*maxK+j] = float32(s.Target[j])
					mask[i*maxK+j] = 1
				}
				meta[i*5], meta[i*5+1], meta[i*5+2], meta[i*5+3], meta[i*5+4] = float32(s.QType), float32(s.Label), float32(s.Weight), float32(s.K), 0
				if s.Type == TaskScore {
					meta[i*5+4] = 1
				}
			}
			tensors = append(tensors, safetensors.Float32Tensor(name+".markers", []int{len(group), maxK, initial.Config.D}, markers), safetensors.Float32Tensor(name+".target", []int{len(group), maxK}, target), safetensors.Float32Tensor(name+".mask", []int{len(group), maxK}, mask), safetensors.Float32Tensor(name+".meta", []int{len(group), 5}, meta))
		}
		var buf bytes.Buffer
		if err = safetensors.Write(&buf, tensors, nil); err != nil {
			return nil, err
		}
		if err = os.WriteFile(filepath.Join(dir, "input.safetensors"), buf.Bytes(), 0600); err != nil {
			return nil, err
		}
		permutations := make([][]int, cfg.Epochs)
		rng := rand.New(rand.NewPCG(cfg.Seed, cfg.Seed^0xda942042e4dd58b5))
		for i := range permutations {
			permutations[i] = rng.Perm(len(fitting))
		}
		config, err := json.Marshal(map[string]any{"contract": "ul.layago-mlx-fit/v1", "config": cfg, "permutations": permutations, "memory_limit": opts.MemoryLimit})
		if err != nil {
			return nil, err
		}
		if err = os.WriteFile(filepath.Join(dir, "request.json"), config, 0600); err != nil {
			return nil, err
		}
		cmd := exec.CommandContext(ctx, opts.Python, opts.Script, "--fit-dir", dir)
		cmd.Stdout = os.Stderr // worker stdout contains only short status; Go stdout stays JSON
		stderr, err := cmd.StderrPipe()
		if err != nil {
			return nil, err
		}
		if err = cmd.Start(); err != nil {
			return nil, fmt.Errorf("mlx start: %w", err)
		}
		scanner := bufio.NewScanner(stderr)
		scanner.Buffer(make([]byte, 4096), 1<<20)
		var last string
		for scanner.Scan() {
			line := scanner.Text()
			last = line
			var event struct {
				Kind  string  `json:"kind"`
				Step  int     `json:"step"`
				Epoch int     `json:"epoch"`
				Loss  float64 `json:"loss"`
			}
			if json.Unmarshal(scanner.Bytes(), &event) == nil && event.Kind == "progress" {
				advance(cfg, "mlx_scorer", event.Step)
			}
			if cfg.Log != nil {
				cfg.Log("mlx: %s", line)
			}
		}
		scanErr := scanner.Err()
		if scanErr != nil {
			_ = cmd.Process.Kill()
		}
		err = cmd.Wait()
		if scanErr != nil {
			return nil, fmt.Errorf("mlx progress: %w", scanErr)
		}
		if err != nil {
			return nil, fmt.Errorf("mlx fit failed: %w (%s)", err, last)
		}
		var result struct {
			History   []EpochStats   `json:"history"`
			BestEpoch int            `json:"best_epoch"`
			Info      map[string]any `json:"info"`
		}
		raw, err := os.ReadFile(filepath.Join(dir, "result.json"))
		if err != nil {
			return nil, err
		}
		if err = json.Unmarshal(raw, &result); err != nil {
			return nil, err
		}
		weights, err := safetensors.Open(filepath.Join(dir, "best.safetensors"))
		if err != nil {
			return nil, err
		}
		defer weights.Close()
		head := initial.Clone()
		for _, p := range head.Params()[:scorerParams] {
			t, err := weights.Tensor(p.Name)
			if err != nil {
				return nil, err
			}
			if !slices.Equal(t.Shape, p.Shape) || t.DType != safetensors.F32 {
				return nil, fmt.Errorf("mlx: incompatible tensor %s", p.Name)
			}
			values, err := t.Float32()
			if err != nil {
				return nil, err
			}
			copy(p.Data, values)
		}
		fit := &ScorerFit{Head: head, History: result.History, BestEpoch: result.BestEpoch, Info: result.Info}
		if err := validateScorerFit(initial, fit); err != nil {
			return nil, err
		}
		// Validate the exported best checkpoint in the same Go forward path that serves it.
		var nllSum float64
		correct := 0
		for _, s := range development {
			scores := head.Scores(s.Markers, s.K, laya.QType(s.QType))
			nllSum += nll(scores, 1, s.Label)
			if argmax(scores) == s.Label {
				correct++
			}
		}
		actual := nllSum / float64(len(development))
		reported := fit.History[fit.BestEpoch-1].DevCalNLL
		if math.Abs(actual-reported) > 1e-4*math.Max(1, math.Abs(actual)) {
			return nil, fmt.Errorf("mlx: Go export NLL parity failed: Go %.9g MLX %.9g", actual, reported)
		}
		if fit.Info == nil {
			fit.Info = map[string]any{}
		}
		fit.Info["go_export_dev_cal_nll"] = actual
		fit.Info["go_export_dev_cal_accuracy"] = float64(correct) / float64(len(development))
		return fit, nil
	}
}
