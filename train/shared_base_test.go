package train

import (
	"context"
	"github.com/UnicoLab/ul_layago/laya"
	"github.com/UnicoLab/ul_layago/model"
	"github.com/UnicoLab/ul_layago/tensor"
	"reflect"
	"testing"
)

// Included synthetic training/export fixtures qualify the reusable loader,
// not domain accuracy. Each variant keeps its own freshly fitted calibration.
func TestTrainingExportsAttachToOneSharedVerifiedBase(t *testing.T) {
	l := qLoop(t)
	base, err := laya.LoadArtifactBase(l.dirs[0], l.trunk.Spec.Dir, laya.ArtifactOptions{FeatureCacheBytes: 1 << 20})
	if err != nil {
		t.Fatal(err)
	}
	var item *Item
	for _, it := range l.p.Items {
		if it.Scorable {
			item = it
			break
		}
	}
	if item == nil {
		t.Fatal("no test item")
	}
	q := item.Question
	asked := []laya.ArtifactQuestion{{ID: q.ID, Task: item.Ex.Task, Options: q.Options, Levels: q.Levels, False: q.False, True: q.True}}
	opt := model.Options{Parallel: tensor.Parallel{Workers: 1, Kernel: tensor.KernelGeneric}}
	var encoder any
	for _, dir := range l.dirs {
		shared, err := laya.LoadArtifactWithBase(dir, base)
		if err != nil {
			t.Fatal(err)
		}
		if encoder == nil {
			encoder = shared.Agent.Encoder
		} else if encoder != shared.Agent.Encoder {
			t.Fatal("exported variants duplicated trunk")
		}
		standalone, err := laya.LoadArtifact(dir, l.trunk.Spec.Dir, laya.ArtifactOptions{})
		if err != nil {
			t.Fatal(err)
		}
		a, _, err := shared.Predict(context.Background(), item.State, asked, opt)
		if err != nil {
			t.Fatal(err)
		}
		b, _, err := standalone.Predict(context.Background(), item.State, asked, opt)
		if err != nil || !reflect.DeepEqual(a, b) {
			t.Fatalf("training export drift: %v", err)
		}
	}
	if base.CacheStats().Hits == 0 {
		t.Fatal("exported heads did not reuse states")
	}
}
