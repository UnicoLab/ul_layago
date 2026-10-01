package train

import "math"

// AdamW is Adam with decoupled weight decay (Loshchilov and Hutter): the
// decay shrinks a parameter by lr·wd directly instead of entering the
// gradient moments. Master copies and moments are float64; after every step
// the float32 parameters the forward pass reads are rounded from the master
// copies, so small updates are never lost to float32 rounding.
type AdamW struct {
	Beta1, Beta2, Eps, WeightDecay float64

	t      int
	params []Param
	master [][]float64
	m, v   [][]float64
}

// NewAdamW tracks params (the float32 slices are updated in place).
func NewAdamW(params []Param, weightDecay float64) *AdamW {
	o := &AdamW{Beta1: 0.9, Beta2: 0.999, Eps: 1e-8, WeightDecay: weightDecay, params: params}
	for _, p := range params {
		mst := make([]float64, len(p.Data))
		for i, v := range p.Data {
			mst[i] = float64(v)
		}
		o.master = append(o.master, mst)
		o.m = append(o.m, make([]float64, len(p.Data)))
		o.v = append(o.v, make([]float64, len(p.Data)))
	}
	return o
}

// Step applies one update with learning rate lr. g holds one gradient
// buffer per parameter (nil skips the parameter).
func (o *AdamW) Step(g [][]float64, lr float64) {
	o.t++
	bc1 := 1 - math.Pow(o.Beta1, float64(o.t))
	bc2 := 1 - math.Pow(o.Beta2, float64(o.t))
	if len(g) != len(o.params) {
		panic("train: AdamW.Step needs one gradient buffer per parameter")
	}
	for pi, grad := range g {
		p := o.params[pi]
		if grad == nil {
			continue
		}
		mst, m, v := o.master[pi], o.m[pi], o.v[pi]
		for i, gi := range grad {
			m[i] = o.Beta1*m[i] + (1-o.Beta1)*gi
			v[i] = o.Beta2*v[i] + (1-o.Beta2)*gi*gi
			upd := (m[i] / bc1) / (math.Sqrt(v[i]/bc2) + o.Eps)
			if p.Decay {
				mst[i] -= lr * o.WeightDecay * mst[i]
			}
			mst[i] -= lr * upd
			p.Data[i] = float32(mst[i])
		}
	}
}

// Sync reloads the master copies from the float32 parameters (after the
// caller restores a snapshot into them).
func (o *AdamW) Sync() {
	for pi, p := range o.params {
		for i, v := range p.Data {
			o.master[pi][i] = float64(v)
		}
	}
}

// LRSchedule is linear warmup to Peak over Warmup steps, then cosine decay
// to Floor·Peak at Total steps.
type LRSchedule struct {
	Peak, Floor   float64
	Warmup, Total int
}

// At returns the learning rate for 1-based step.
func (s LRSchedule) At(step int) float64 {
	if s.Warmup > 0 && step <= s.Warmup {
		return s.Peak * float64(step) / float64(s.Warmup)
	}
	span := s.Total - s.Warmup
	if span <= 0 {
		return s.Peak
	}
	frac := math.Min(1, float64(step-s.Warmup)/float64(span))
	return s.Peak * (s.Floor + (1-s.Floor)*0.5*(1+math.Cos(math.Pi*frac)))
}
