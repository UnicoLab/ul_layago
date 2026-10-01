package train

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// TestScoreCachedReproducesTraining: an exported artifact decodes the cached
// features into the rows training computed (logits, probabilities,
// predictions and act decisions, bit for bit), and Evaluate over them is
// the manifest's evaluation, for both numeric paths.
func TestScoreCachedReproducesTraining(t *testing.T) {
	l := qLoop(t)
	for i, v := range []*Variant{l.res.Float32, l.res.Int8} {
		art, err := LoadArtifact(l.dirs[i])
		if err != nil {
			t.Fatal(err)
		}
		scored, err := art.ScoreCached(l.p, l.recs, 3)
		if err != nil {
			t.Fatal(err)
		}
		if len(scored) != len(v.Rows) {
			t.Fatalf("%s: %d rows, trained %d", v.Format, len(scored), len(v.Rows))
		}
		rows := make([]Row, len(scored))
		for j, s := range scored {
			rows[j] = s.Row
			if !reflect.DeepEqual(s.Row, v.Rows[j]) {
				t.Fatalf("%s row %d: %+v, trained %+v", v.Format, j, s.Row, v.Rows[j])
			}
			if s.Row.Scorable && (!reflect.DeepEqual(s.Logits, v.Logits[s.Item.Ex.ID]) || !reflect.DeepEqual(s.ActLogits, v.ActLogits[s.Item.Ex.ID])) {
				t.Fatalf("%s %s: logits differ from training", v.Format, s.Item.Ex.ID)
			}
		}
		raw, err := os.ReadFile(filepath.Join(l.dirs[i], ArtifactManifest))
		if err != nil {
			t.Fatal(err)
		}
		var m struct {
			Evaluation struct {
				Splits json.RawMessage `json:"splits"`
			} `json:"evaluation"`
		}
		if err := json.Unmarshal(raw, &m); err != nil {
			t.Fatal(err)
		}
		got, err := json.Marshal(Evaluate(rows))
		if err != nil {
			t.Fatal(err)
		}
		var a, b any
		if json.Unmarshal(got, &a) != nil || json.Unmarshal(m.Evaluation.Splits, &b) != nil || !reflect.DeepEqual(a, b) {
			t.Fatalf("%s: Evaluate differs from the manifest's evaluation", v.Format)
		}
	}

	art, err := LoadArtifact(l.dirs[0])
	if err != nil {
		t.Fatal(err)
	}
	missing := map[string]*Record{}
	if _, err := art.ScoreCached(l.p, missing, 1); err == nil {
		t.Fatal("missing features accepted")
	}
	r := *l.p
	r.Instance = InstanceR
	if _, err := art.ScoreCached(&r, l.recs, 1); err == nil {
		t.Fatal("another instance accepted")
	}
}
