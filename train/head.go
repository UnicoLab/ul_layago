package train

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/UnicoLab/ul_layago/laya"
	"github.com/UnicoLab/ul_layago/tensor"
)

// HeadConfig sizes a Head.
type HeadConfig struct {
	D         int `json:"d"`
	Hidden    int `json:"scorer_hidden"`
	ActHidden int `json:"act_hidden"`
	NAct      int `json:"act_outputs"`
}

// Head is Laya's decision head with zero head transformer layers, over a
// frozen trunk's cached states: a question-type embedding added to each
// option's [MASK] state, Laya's marker scorer LayerNorm → Linear → GELU →
// Linear → 1 with a softmax per question, and Laya's act head Linear(d+4) →
// GELU → Linear over the pooled state (position 0 plus the type embedding)
// and the confidence vector of the marker logits (laya.ActFeatures). Laya's
// head transformer layers attend over every token of the sequence, which
// would require caching every token's state; with head_layers = 0 only the
// marker and pooled states are needed, and the frozen trunk's own layers
// already contextualise each marker against the question, the other options
// and the state. Act output 0 means act and output 1 escalate, as
// laya.Answer.ActProbability reads softmax(act)[0]. The trainer uses
// Hidden = D and ActHidden = 256, Laya's shapes, so an exported head loads in
// laya.LoadArtifact (and, beside the trunk's tensors, laya.LoadCheckpoint)
// unchanged.
//
// Parameters are float32 (the served precision). Forward passes accumulate
// in float64, and the logits a caller sees are the float32 rounding of the
// float64 score, so training, evaluation and served inference share one
// function and agree bit for bit.
type Head struct {
	Config  HeadConfig
	TypeEmb []float32 // 3 x D, indexed by laya.QType
	LNW     []float32 // D
	LNB     []float32 // D
	W1      []float32 // Hidden x D
	B1      []float32 // Hidden
	W2      []float32 // Hidden (Linear(Hidden, 1))
	B2      []float32 // 1
	A1      []float32 // ActHidden x (D+4)
	A1B     []float32 // ActHidden
	A2      []float32 // NAct x ActHidden
	A2B     []float32 // NAct
}

// Param is one named parameter tensor.
type Param struct {
	Name  string
	Shape []int
	Data  []float32
	// Decay applies AdamW weight decay (weight matrices only).
	Decay bool
	// Act marks the act head, trained in the second stage.
	Act bool
}

// Params lists the head's tensors in export order, under Laya's
// DecisionModel names (type_emb, scorer.0/1/3, act_head.0/2): with the
// scorer width equal to d and a 256-wide act head, an exported head is a
// Laya decision head with head_layers = 0, which laya.LoadCheckpoint reads.
func (h *Head) Params() []Param {
	c := h.Config
	return []Param{
		{Name: "type_emb.weight", Shape: []int{3, c.D}, Data: h.TypeEmb},
		{Name: "scorer.0.weight", Shape: []int{c.D}, Data: h.LNW},
		{Name: "scorer.0.bias", Shape: []int{c.D}, Data: h.LNB},
		{Name: "scorer.1.weight", Shape: []int{c.Hidden, c.D}, Data: h.W1, Decay: true},
		{Name: "scorer.1.bias", Shape: []int{c.Hidden}, Data: h.B1},
		{Name: "scorer.3.weight", Shape: []int{1, c.Hidden}, Data: h.W2, Decay: true},
		{Name: "scorer.3.bias", Shape: []int{1}, Data: h.B2},
		{Name: "act_head.0.weight", Shape: []int{c.ActHidden, c.D + 4}, Data: h.A1, Decay: true, Act: true},
		{Name: "act_head.0.bias", Shape: []int{c.ActHidden}, Data: h.A1B, Act: true},
		{Name: "act_head.2.weight", Shape: []int{c.NAct, c.ActHidden}, Data: h.A2, Decay: true, Act: true},
		{Name: "act_head.2.bias", Shape: []int{c.NAct}, Data: h.A2B, Act: true},
	}
}

// scorerParams counts the Params trained in the scorer stage (type_emb and
// scorer.*); the rest are the act head's.
const scorerParams = 7

// layaActHidden is the width of Laya's act head (DecisionModel act_head.0),
// which laya.LoadArtifact requires.
const layaActHidden = 256

// isLinearWeight reports whether a parameter is a 2-D linear weight, the
// tensors an int8 artifact stores as I8 with a per-row scale (every 2-D
// tensor but the type embedding, as the Python exporter quantises).
func isLinearWeight(p Param) bool { return len(p.Shape) == 2 && p.Name != "type_emb.weight" }

// QuantizedHead returns a copy of h whose linear weights went through
// symmetric per-row int8 (tensor.QuantizeRows) and back: the head an int8
// artifact serves, calibrated and evaluated on its own outputs.
func QuantizedHead(h *Head) *Head {
	q := h.Clone()
	for _, p := range q.Params() {
		if !isLinearWeight(p) {
			continue
		}
		rows, cols := p.Shape[0], p.Shape[1]
		qs, scale := tensor.QuantizeRows(p.Data, rows, cols)
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				p.Data[r*cols+c] = float32(qs[r*cols+c]) * scale[r]
			}
		}
	}
	return q
}

// NewHead initialises a head deterministically from seed: normal weights
// with standard deviation 1/sqrt(fan-in), unit norm scales, zero biases and a
// zero type embedding.
func NewHead(c HeadConfig, seed uint64) (*Head, error) {
	if c.D <= 0 || c.Hidden <= 0 || c.ActHidden <= 0 || c.NAct != 2 {
		return nil, fmt.Errorf("train: invalid head config %+v (act_outputs must be 2)", c)
	}
	r := rand.New(rand.NewPCG(seed, seed^0x2545f4914f6cdd1d)) //nolint:gosec // G404: reproducible initialisation, not a security use
	normal := func(n, fanIn int) []float32 {
		out := make([]float32, n)
		std := 1 / math.Sqrt(float64(fanIn))
		for i := range out {
			out[i] = float32(r.NormFloat64() * std)
		}
		return out
	}
	ones := make([]float32, c.D)
	for i := range ones {
		ones[i] = 1
	}
	return &Head{
		Config: c, TypeEmb: make([]float32, 3*c.D), LNW: ones, LNB: make([]float32, c.D),
		W1: normal(c.Hidden*c.D, c.D), B1: make([]float32, c.Hidden),
		W2: normal(c.Hidden, c.Hidden), B2: make([]float32, 1),
		A1: normal(c.ActHidden*(c.D+4), c.D+4), A1B: make([]float32, c.ActHidden),
		A2: normal(c.NAct*c.ActHidden, c.ActHidden), A2B: make([]float32, c.NAct),
	}, nil
}

// Clone deep-copies the head.
func (h *Head) Clone() *Head {
	cp := func(s []float32) []float32 { return append([]float32(nil), s...) }
	return &Head{Config: h.Config, TypeEmb: cp(h.TypeEmb), LNW: cp(h.LNW), LNB: cp(h.LNB), W1: cp(h.W1), B1: cp(h.B1),
		W2: cp(h.W2), B2: cp(h.B2), A1: cp(h.A1), A1B: cp(h.A1B), A2: cp(h.A2), A2B: cp(h.A2B)}
}

const headLNEps = 1e-5

// scorerScratch holds one option's intermediates for the backward pass.
type scorerScratch struct {
	xhat, u, pre, g []float64
	inv             float64
}

func newScorerScratch(c HeadConfig) *scorerScratch {
	return &scorerScratch{xhat: make([]float64, c.D), u: make([]float64, c.D), pre: make([]float64, c.Hidden), g: make([]float64, c.Hidden)}
}

func gelu64(x float64) float64 { return 0.5 * x * (1 + math.Erf(x/math.Sqrt2)) }

// geluGrad64 is d/dx of the exact GELU: Φ(x) + x·φ(x).
func geluGrad64(x float64) float64 {
	return 0.5*(1+math.Erf(x/math.Sqrt2)) + x*math.Exp(-0.5*x*x)/math.Sqrt(2*math.Pi)
}

// dot64 accumulates a float32 row times a float64 vector in float64.
func dot64(w []float32, x []float64) float64 {
	x = x[:len(w)]
	var s0, s1, s2, s3 float64
	i := 0
	for ; i+4 <= len(w); i += 4 {
		s0 += float64(w[i]) * x[i]
		s1 += float64(w[i+1]) * x[i+1]
		s2 += float64(w[i+2]) * x[i+2]
		s3 += float64(w[i+3]) * x[i+3]
	}
	for ; i < len(w); i++ {
		s0 += float64(w[i]) * x[i]
	}
	return (s0 + s1) + (s2 + s3)
}

// score runs the marker scorer on one option state m under type
// embedding te and returns the float64 score, filling sc.
func (h *Head) score(m, te []float32, sc *scorerScratch) float64 {
	d := h.Config.D
	var sum float64
	for i := 0; i < d; i++ {
		z := float64(m[i]) + float64(te[i])
		sc.xhat[i] = z
		sum += z
	}
	mean := sum / float64(d)
	var ss float64
	for i := 0; i < d; i++ {
		c := sc.xhat[i] - mean
		sc.xhat[i] = c
		ss += c * c
	}
	sc.inv = 1 / math.Sqrt(ss/float64(d)+headLNEps)
	for i := 0; i < d; i++ {
		sc.xhat[i] *= sc.inv
		sc.u[i] = sc.xhat[i]*float64(h.LNW[i]) + float64(h.LNB[i])
	}
	s := float64(h.B2[0])
	for j := 0; j < h.Config.Hidden; j++ {
		p := float64(h.B1[j]) + dot64(h.W1[j*d:(j+1)*d], sc.u)
		sc.pre[j] = p
		sc.g[j] = gelu64(p)
	}
	s += dot64(h.W2, sc.g)
	return s
}

// Scores returns one question's marker scores in float64 (k options, states
// K x D).
func (h *Head) Scores(markers []float32, k int, qt laya.QType) []float64 {
	d := h.Config.D
	te := h.TypeEmb[int(qt)*d : (int(qt)+1)*d]
	sc := newScorerScratch(h.Config)
	out := make([]float64, k)
	for i := 0; i < k; i++ {
		out[i] = h.score(markers[i*d:(i+1)*d], te, sc)
	}
	return out
}

// Logits returns the served marker logits: the float32 rounding of Scores.
func (h *Head) Logits(markers []float32, k int, qt laya.QType) []float32 {
	s := h.Scores(markers, k, qt)
	out := make([]float32, k)
	for i, v := range s {
		out[i] = float32(v)
	}
	return out
}

// actScratch holds the act head's intermediates.
type actScratch struct {
	in, pre, g []float64
}

func newActScratch(c HeadConfig) *actScratch {
	return &actScratch{in: make([]float64, c.D+4), pre: make([]float64, c.ActHidden), g: make([]float64, c.ActHidden)}
}

// actForward returns the act logits for pooled state p under qt, given the
// question's served marker logits.
func (h *Head) actForward(pooled []float32, qt laya.QType, logits []float32, sc *actScratch) []float64 {
	c := h.Config
	te := h.TypeEmb[int(qt)*c.D : (int(qt)+1)*c.D]
	for i := 0; i < c.D; i++ {
		sc.in[i] = float64(pooled[i]) + float64(te[i])
	}
	f := laya.ActFeatures(logits)
	for i, v := range f {
		sc.in[c.D+i] = float64(v)
	}
	for j := 0; j < c.ActHidden; j++ {
		p := float64(h.A1B[j]) + dot64(h.A1[j*(c.D+4):(j+1)*(c.D+4)], sc.in)
		sc.pre[j] = p
		sc.g[j] = gelu64(p)
	}
	out := make([]float64, c.NAct)
	for k := 0; k < c.NAct; k++ {
		out[k] = float64(h.A2B[k]) + dot64(h.A2[k*c.ActHidden:(k+1)*c.ActHidden], sc.g)
	}
	return out
}

// ActLogits returns the served act logits (float32 rounding).
func (h *Head) ActLogits(pooled []float32, qt laya.QType, logits []float32) []float32 {
	o := h.actForward(pooled, qt, logits, newActScratch(h.Config))
	out := make([]float32, len(o))
	for i, v := range o {
		out[i] = float32(v)
	}
	return out
}

// ActProbability is softmax(act logits)[0], computed in float64 from the
// served float32 act logits.
func ActProbability(act []float32) float64 {
	return softmax64([]float64{float64(act[0]), float64(act[1])}, 1)[0]
}

// grads holds float64 gradients for every head parameter.
type grads struct {
	typeEmb, lnw, lnb, w1, b1, w2, b2 []float64
	a1, a1b, a2, a2b                  []float64
}

func newGrads(c HeadConfig, act bool) *grads {
	g := &grads{}
	if act {
		g.a1, g.a1b = make([]float64, c.ActHidden*(c.D+4)), make([]float64, c.ActHidden)
		g.a2, g.a2b = make([]float64, c.NAct*c.ActHidden), make([]float64, c.NAct)
		return g
	}
	g.typeEmb, g.lnw, g.lnb = make([]float64, 3*c.D), make([]float64, c.D), make([]float64, c.D)
	g.w1, g.b1 = make([]float64, c.Hidden*c.D), make([]float64, c.Hidden)
	g.w2, g.b2 = make([]float64, c.Hidden), make([]float64, 1)
	return g
}

// slices returns the gradient buffers aligned with Head.Params (nil for a
// stage the buffer does not cover).
func (g *grads) slices() [][]float64 {
	return [][]float64{g.typeEmb, g.lnw, g.lnb, g.w1, g.b1, g.w2, g.b2, g.a1, g.a1b, g.a2, g.a2b}
}

func (g *grads) zero() {
	for _, s := range g.slices() {
		clear(s)
	}
}

// addScaled adds s*o into g, buffer by buffer in a fixed order.
func (g *grads) addScaled(o *grads, s float64) {
	dst, src := g.slices(), o.slices()
	for i := range dst {
		for j := range dst[i] {
			dst[i][j] += s * src[i][j]
		}
	}
}

// backwardScore accumulates the gradient of one option's score (ds) into g,
// using the scratch the forward pass filled. du is D-long scratch.
func (h *Head) backwardScore(ds float64, qt laya.QType, sc *scorerScratch, g *grads, du []float64) {
	c := h.Config
	d := c.D
	g.b2[0] += ds
	clear(du)
	for j := 0; j < c.Hidden; j++ {
		g.w2[j] += ds * sc.g[j]
		dpre := ds * float64(h.W2[j]) * geluGrad64(sc.pre[j])
		if dpre == 0 {
			continue
		}
		g.b1[j] += dpre
		row := h.W1[j*d : (j+1)*d]
		grow := g.w1[j*d : (j+1)*d]
		for i := 0; i < d; i++ {
			grow[i] += dpre * sc.u[i]
			du[i] += dpre * float64(row[i])
		}
	}
	// LayerNorm backward: dz = inv * (dx̂ - mean(dx̂) - x̂·mean(dx̂·x̂)).
	var m1, m2 float64
	for i := 0; i < d; i++ {
		g.lnw[i] += du[i] * sc.xhat[i]
		g.lnb[i] += du[i]
		dx := du[i] * float64(h.LNW[i])
		du[i] = dx
		m1 += dx
		m2 += dx * sc.xhat[i]
	}
	m1 /= float64(d)
	m2 /= float64(d)
	te := g.typeEmb[int(qt)*d : (int(qt)+1)*d]
	for i := 0; i < d; i++ {
		te[i] += sc.inv * (du[i] - m1 - sc.xhat[i]*m2)
	}
}

// actExampleGrad adds the gradient of the weighted act cross-entropy
// -w·log softmax(act)[target] into g and returns the loss (target 0 is act,
// 1 escalate).
func (h *Head) actExampleGrad(pooled []float32, qt laya.QType, logits []float32, target int, w float64, sc *actScratch, g *grads) float64 {
	o := h.actForward(pooled, qt, logits, sc)
	p := softmax64(o, 1)
	dout := []float64{w * p[0], w * p[1]}
	dout[target] -= w
	h.backwardAct(dout, sc, g)
	return -w * math.Log(math.Max(p[target], 1e-300))
}

// backwardAct accumulates the act head's gradient for d(act logits) = dout.
func (h *Head) backwardAct(dout []float64, sc *actScratch, g *grads) {
	c := h.Config
	w := c.D + 4
	for k := 0; k < c.NAct; k++ {
		g.a2b[k] += dout[k]
		row := g.a2[k*c.ActHidden : (k+1)*c.ActHidden]
		for j := range row {
			row[j] += dout[k] * sc.g[j]
		}
	}
	for j := 0; j < c.ActHidden; j++ {
		var dg float64
		for k := 0; k < c.NAct; k++ {
			dg += dout[k] * float64(h.A2[k*c.ActHidden+j])
		}
		dpre := dg * geluGrad64(sc.pre[j])
		if dpre == 0 {
			continue
		}
		g.a1b[j] += dpre
		row := g.a1[j*w : (j+1)*w]
		for i := range row {
			row[i] += dpre * sc.in[i]
		}
	}
}
