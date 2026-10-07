package train

import (
	"fmt"
	"reflect"
	"testing"
)

func planningFixture() *Dataset {
	d := &Dataset{Examples: []Example{{ID: "fit", Split: SplitTrain, Task: "r_type", StoryID: "fit"}}}
	counts := map[string]int{}
	for i := 0; counts[RoleDevCal] < 2 || counts[RoleDevConf] < 2; i++ {
		story := fmt.Sprintf("plan-%d", i)
		role := DevHalf(story)
		if counts[role] >= 2 {
			continue
		}
		counts[role]++
		for _, lang := range []string{"en", "fr"} {
			d.Examples = append(d.Examples, Example{ID: story + lang, Split: SplitDev, Task: "r_type", StoryID: story, Lang: lang})
		}
	}
	d.Examples = append(d.Examples, Example{ID: "sealed", Split: SplitLockedTest, Task: "r_type", StoryID: "sealed", Label: 999, Input: map[string]any{"unreadable": true}})
	return d
}
func TestReadinessMatchesTwoScoreAllocationAndCountsStories(t *testing.T) {
	d := planningFixture()
	cfg := DefaultConfig()
	r, e := PlanCalibration(d, cfg)
	if e != nil {
		t.Fatal(e)
	}
	if r.CertificationStart != 176 || r.ScoreFamilyDelta != .05 {
		t.Fatalf("wrong actual trainer policy %+v", r)
	}
	if r.DevConf.Examples != 4 || r.DevConf.Stories != 2 || r.DevConf.StoriesByTask["r_type"] != 2 {
		t.Fatal(r.DevConf)
	}
	if r.TokenRetentionChecked || !r.RequiresDomainValidation {
		t.Fatal("counts became a qualification")
	}
	p := &Prepared{}
	for i := range d.Examples {
		p.Items = append(p.Items, &Item{Ex: &d.Examples[i], Role: d.Examples[i].Role(), Scorable: true})
	}
	if r.CertificationStart != devHalfReport(p, cfg).CertificationStart {
		t.Fatal("plan drifted from trainer")
	}
	old := r
	d.Examples[len(d.Examples)-1].Input = map[string]any{"gold_changed": 123}
	d.Examples[len(d.Examples)-1].Label = -99
	r, e = PlanCalibration(d, cfg)
	if e != nil || !reflect.DeepEqual(old, r) {
		t.Fatal("locked content affected plan", e)
	}
}
func TestReadinessUsesSameFitLanguageFilterAndRejectsInvalidConfig(t *testing.T) {
	cfg := DefaultConfig()
	cfg.FitLangs = []string{"fr"}
	r, e := PlanCalibration(planningFixture(), cfg)
	if e != nil || r.DevCal.Examples != 2 || r.DevConf.Examples != 2 {
		t.Fatal(r, e)
	}
	for _, change := range []func(*Config){func(c *Config) { c.Alpha = 1e-300 }, func(c *Config) { c.Epochs = 0 }, func(c *Config) { c.LTTStartErrors = int(^uint(0) >> 1) }} {
		cfg = DefaultConfig()
		change(&cfg)
		if _, e = PlanCalibration(planningFixture(), cfg); e == nil {
			t.Fatal("accepted invalid or unbounded planning config")
		}
	}
	if _, e = PlanCalibration(nil, DefaultConfig()); e == nil {
		t.Fatal("accepted nil dataset")
	}
}
func TestPositiveErrorPlanningIsBoundedAndLabelsIgnored(t *testing.T) {
	cfg := DefaultConfig()
	cfg.LTTStartErrors = int(^uint(0)>>1) - 1
	huge, err := PlanCalibration(planningFixture(), cfg)
	if err != nil || !huge.CertificationStartLowerBound || huge.CertificationStart != cfg.LTTStartErrors+1 {
		t.Fatal("unbounded error budget must not allocate binomial terms", huge, err)
	}
	cfg.LTTStartErrors = 1
	r, e := PlanCalibration(planningFixture(), cfg)
	if e != nil || !r.CertificationStartLowerBound || r.CertificationStart <= r.DevConf.Examples {
		t.Fatal(r, e)
	}
	cfg.Alpha = .5
	r, e = PlanCalibration(planningFixture(), cfg)
	if e != nil || r.CertificationStartLowerBound || r.CertificationStart != LTTStart(cfg.Alpha, cfg.Delta/2, cfg.NMin, 1) {
		t.Fatal(r, e)
	}
	d := planningFixture()
	old, e := PlanCalibration(d, cfg)
	if e != nil {
		t.Fatal(e)
	}
	for i := range d.Examples {
		d.Examples[i].Label = 999
		d.Examples[i].Input = map[string]any{"changed": true}
	}
	again, e := PlanCalibration(d, cfg)
	if e != nil || !reflect.DeepEqual(old, again) {
		t.Fatal("planning consumed source/gold", e)
	}
}
