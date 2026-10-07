package train

import (
	"fmt"
	"math"
	"slices"
)

// Validate checks training configuration without loading or forwarding a model.
func (c Config) Validate() error { return c.validate() }

// CalibrationCell counts independent story IDs separately from row count.
// More translations or option-order variants do not create independent data.
type CalibrationCell struct {
	Examples      int            `json:"examples"`
	Stories       int            `json:"stories"`
	ByTask        map[string]int `json:"by_task"`
	StoriesByTask map[string]int `json:"stories_by_task"`
}

// CalibrationPlan is a metadata-only preflight, never a risk certificate.
type CalibrationPlan struct {
	Contract                     string          `json:"contract"`
	Alpha                        float64         `json:"alpha"`
	Delta                        float64         `json:"delta"`
	ScoreFamilyDelta             float64         `json:"score_family_delta"`
	CertificationStart           int             `json:"certification_start"`
	CertificationStartLowerBound bool            `json:"certification_start_lower_bound"`
	DevCal                       CalibrationCell `json:"dev_cal"`
	DevConf                      CalibrationCell `json:"dev_conf"`
	Warnings                     []string        `json:"warnings"`
	TokenRetentionChecked        bool            `json:"token_retention_checked"`
	RequiresDomainValidation     bool            `json:"requires_domain_validation"`
}

// PlanCalibration inspects only train/development metadata, not labels or source
// text. It mirrors the trainer's story partition and two-score delta allocation.
// Counts are available rows before token clipping; inspect still checks tokens.
func PlanCalibration(ds *Dataset, cfg Config) (*CalibrationPlan, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	// LTTStart's legacy positive-error search can be expensive for unreasonable
	// settings. Count only as far as the available dev_conf rows: above that,
	// certification cannot be achieved by this dataset.
	makeCell := func() CalibrationCell {
		return CalibrationCell{ByTask: map[string]int{}, StoriesByTask: map[string]int{}}
	}
	r := &CalibrationPlan{Contract: "ul.layago-calibration-plan/v1", Alpha: cfg.Alpha, Delta: cfg.Delta, ScoreFamilyDelta: cfg.Delta / 2, DevCal: makeCell(), DevConf: makeCell(), Warnings: []string{}, RequiresDomainValidation: true}
	if ds == nil {
		return nil, fmt.Errorf("train: dataset is required")
	}
	stories := map[string]map[string]bool{RoleDevCal: {}, RoleDevConf: {}}
	taskStories := map[string]map[string]map[string]bool{RoleDevCal: {}, RoleDevConf: {}}
	tasks := map[string]bool{}
	for i := range ds.Examples {
		ex := &ds.Examples[i]
		role := ex.Role()
		if role == SplitTrain {
			tasks[ex.Task] = true
			continue
		}
		if role != RoleDevCal && role != RoleDevConf {
			continue
		}
		if len(cfg.FitLangs) > 0 && !slices.Contains(cfg.FitLangs, primaryLanguage(ex.Lang)) {
			continue
		}
		h := &r.DevCal
		if role == RoleDevConf {
			h = &r.DevConf
		}
		h.Examples++
		h.ByTask[ex.Task]++
		stories[role][ex.StoryID] = true
		tasks[ex.Task] = true
		if taskStories[role][ex.Task] == nil {
			taskStories[role][ex.Task] = map[string]bool{}
		}
		taskStories[role][ex.Task][ex.StoryID] = true
	}
	for _, role := range []string{RoleDevCal, RoleDevConf} {
		h := &r.DevCal
		if role == RoleDevConf {
			h = &r.DevConf
		}
		h.Stories = len(stories[role])
		for task, ss := range taskStories[role] {
			h.StoriesByTask[task] = len(ss)
		}
	}
	if cfg.LTTStartErrors == int(^uint(0)>>1) {
		return nil, fmt.Errorf("train: positive-error starting count is not representable")
	}
	if cfg.LTTStartErrors == 0 {
		start := math.Ceil(3 * math.Log(cfg.Delta/2) / math.Log1p(-cfg.Alpha))
		if math.IsInf(start, 0) || math.IsNaN(start) || start >= float64(int(^uint(0)>>1)) {
			return nil, fmt.Errorf("train: certification starting count is not representable")
		}
		r.CertificationStart = max(cfg.NMin, int(start))
	} else if cfg.LTTStartErrors >= r.DevConf.Examples {
		// The configured error budget alone already consumes all available
		// support. Do not allocate binomial terms proportional to an arbitrary
		// configuration value in a cheap readiness command.
		r.CertificationStart = max(cfg.NMin, cfg.LTTStartErrors+1)
		r.CertificationStartLowerBound = true
		r.Warnings = append(r.Warnings, "Positive-error budget exhausts available development support; reported count is a lower bound.")
	} else {
		limit := max(r.DevConf.Examples, cfg.NMin, cfg.LTTStartErrors+1)
		// Do not perform an unbounded binomial search in a cheap metadata preflight.
		lo, hi := cfg.LTTStartErrors+1, limit
		if cfg.Delta/2 == 0 {
			return nil, fmt.Errorf("train: score-family delta underflows")
		}
		if binomCDF(cfg.LTTStartErrors, hi, cfg.Alpha) > cfg.Delta/2 {
			if limit == int(^uint(0)>>1) {
				return nil, fmt.Errorf("train: certification starting count is not representable")
			}
			r.CertificationStart = max(cfg.NMin, limit+1)
			r.CertificationStartLowerBound = true
			r.Warnings = append(r.Warnings, "Positive-error certification start exceeds available development support; reported count is a lower bound.")
		} else {
			for lo < hi {
				mid := lo + (hi-lo)/2
				if binomCDF(cfg.LTTStartErrors, mid, cfg.Alpha) <= cfg.Delta/2 {
					hi = mid
				} else {
					lo = mid + 1
				}
			}
			r.CertificationStart = max(cfg.NMin, lo)
		}
	}
	for _, task := range sortedKeys(tasks) {
		if n := r.DevCal.ByTask[task]; n < cfg.MinCell {
			r.Warnings = append(r.Warnings, fmt.Sprintf("%s: %d dev_cal rows below min_cell %d; task temperature may fall back.", task, n, cfg.MinCell))
		}
		if n := r.DevConf.ByTask[task]; n < r.CertificationStart {
			r.Warnings = append(r.Warnings, fmt.Sprintf("%s: %d dev_conf rows below certification start %d; sample size cannot certify automatic decisions.", task, n, r.CertificationStart))
		}
		if r.DevConf.StoriesByTask[task] < r.DevConf.ByTask[task] {
			r.Warnings = append(r.Warnings, fmt.Sprintf("%s: %d dev_conf rows share %d stories; correlation requires an independently justified certification design.", task, r.DevConf.ByTask[task], r.DevConf.StoriesByTask[task]))
		}
	}
	r.Warnings = append(r.Warnings, "Metadata counts do not prove label quality, token retention, independence, calibration or domain validity. Do not lower risk targets to make an undersized dataset pass.")
	return r, nil
}
