package train

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"maps"

	"github.com/UnicoLab/ul_layago/laya"
)

// CalibrationOverlap checks identical task/state/options across the two
// development roles before encoding. Labels and example/source row IDs cannot
// legitimize copied calibration input. This is an exact check, not a near-copy
// or domain-independence proof; use the data audit for those diagnostics.
func CalibrationOverlap(ds *Dataset) (map[string]int, error) {
	if ds == nil {
		return nil, fmt.Errorf("train: dataset is required")
	}
	type seenRole struct{ cal, conf int }
	seen := map[[32]byte]seenRole{}
	tasks := map[[32]byte]string{}
	for i := range ds.Examples {
		ex := &ds.Examples[i]
		role := ex.Role()
		if role != RoleDevCal && role != RoleDevConf {
			continue
		}
		input := ex.Input
		if ex.Instance == InstanceR {
			input = maps.Clone(ex.Input)
			if evidence, ok := input["evidence"].([]any); ok {
				normalized := make([]any, len(evidence))
				for j, item := range evidence {
					if obj, ok := item.(map[string]any); ok {
						c := maps.Clone(obj)
						c["id"] = j
						normalized[j] = c
					} else {
						normalized[j] = item
					}
				}
				input["evidence"] = normalized
			}
		}
		state, err := laya.RenderPromptState(ex.Instance, input)
		if err != nil {
			return nil, err
		}
		raw, err := json.Marshal(struct {
			Task, State string
			Options     []Option
		}{ex.Task, state, ex.Options})
		if err != nil {
			return nil, err
		}
		key := sha256.Sum256(raw)
		v := seen[key]
		if role == RoleDevCal {
			v.cal++
		} else {
			v.conf++
		}
		seen[key] = v
		tasks[key] = ex.Task
	}
	out := map[string]int{}
	for key, counts := range seen {
		if counts.cal > 0 && counts.conf > 0 {
			out[tasks[key]] += counts.conf
		}
	}
	return out, nil
}
