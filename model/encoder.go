package model

import (
	"context"
	"errors"
	"fmt"
	"math"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/UnicoLab/ul_layago/tensor"
)

// Encoder runs the ModernBERT forward pass. It is safe for concurrent use.
type Encoder struct {
	Config  Config
	Weights *Weights

	ropeMu sync.Mutex
	ropes  atomic.Pointer[ropeSet]
}

type ropeSet struct {
	positions int
	byTheta   map[float64]*tensor.Rope
}

// NewEncoder validates weights against cfg.
func NewEncoder(cfg Config, w *Weights) (*Encoder, error) {
	if w == nil || len(w.Layers) != cfg.NumLayers || len(cfg.Layers) != cfg.NumLayers {
		return nil, errors.New("model: weights and config disagree on the layer count")
	}
	e := &Encoder{Config: cfg, Weights: w}
	e.ensureRope(512)
	return e, nil
}

// ensureRope makes the rotary tables cover positions [0, n). Tables are
// immutable once published, so readers never race with growth.
func (e *Encoder) ensureRope(n int) *ropeSet {
	if cur := e.ropes.Load(); cur != nil && cur.positions >= n {
		return cur
	}
	e.ropeMu.Lock()
	defer e.ropeMu.Unlock()
	cur := e.ropes.Load()
	if cur != nil && cur.positions >= n {
		return cur
	}
	if cur != nil {
		n = max(n, 2*cur.positions)
	}
	next := &ropeSet{positions: n, byTheta: map[float64]*tensor.Rope{}}
	for _, l := range e.Config.Layers {
		if _, ok := next.byTheta[l.RopeTheta]; !ok {
			next.byTheta[l.RopeTheta] = tensor.NewRope(e.Config.HeadDim(), l.RopeTheta, n)
		}
	}
	e.ropes.Store(next)
	return next
}

// Op names one timed phase of the forward pass.
type Op int

// Forward-pass phases reported in Timings.
const (
	OpEmbed Op = iota
	OpNorm
	OpQKV
	OpAttention
	OpAttnOut
	OpMLPIn
	OpGeGLU
	OpMLPOut
	numOps
)

var opNames = [numOps]string{"embed", "norm", "qkv", "rope_attention", "attn_out", "mlp_in", "geglu", "mlp_out"}

// String returns the phase name used in receipts.
func (o Op) String() string { return opNames[o] }

// Timings accumulates wall time per phase over a forward pass.
type Timings [numOps]time.Duration

// Total returns the sum of all phases.
func (t *Timings) Total() time.Duration {
	var sum time.Duration
	for _, v := range t {
		sum += v
	}
	return sum
}

// Map returns the phases keyed by name.
func (t *Timings) Map() map[string]time.Duration {
	out := make(map[string]time.Duration, numOps)
	for i, v := range t {
		out[Op(i).String()] = v
	}
	return out
}

// Options control one forward pass.
type Options struct {
	Parallel tensor.Parallel
	// CaptureLayers records the hidden states Hugging Face returns with
	// output_hidden_states: the embedding output, then each layer's output
	// before the final norm.
	CaptureLayers bool
	// Timings, when non-nil, receives a per-operation breakdown.
	Timings *Timings
}

// Result holds per-sequence outputs, each a row-major len(seq) x HiddenSize
// slice.
type Result struct {
	Hidden [][]float32
	// Layers[k][s] is hidden state k (0 = embeddings) of sequence s when
	// Options.CaptureLayers is set.
	Layers [][][]float32
}

// Forward encodes each token sequence independently (no padding: rows of all
// sequences are stacked for the dense layers, and attention stays within a
// sequence). This equals Hugging Face's padded batch on every non-padded
// position.
func (e *Encoder) Forward(ctx context.Context, seqs [][]int32, opt Options) (*Result, error) {
	cfg := e.Config
	d, heads, hd, inter := cfg.HiddenSize, cfg.NumHeads, cfg.HeadDim(), cfg.IntermediateSize
	starts := make([]int, len(seqs)+1)
	maxLen := 0
	for i, s := range seqs {
		if len(s) == 0 {
			return nil, fmt.Errorf("model: sequence %d is empty", i)
		}
		for _, id := range s {
			if id < 0 || int(id) >= cfg.VocabSize {
				return nil, fmt.Errorf("model: token id %d outside vocabulary of %d", id, cfg.VocabSize)
			}
		}
		starts[i+1] = starts[i] + len(s)
		maxLen = max(maxLen, len(s))
	}
	total := starts[len(seqs)]
	ropes := e.ensureRope(maxLen)
	par := opt.Parallel
	tm := opt.Timings
	mark := time.Now()
	lap := func(op Op) {
		if tm != nil {
			now := time.Now()
			tm[op] += now.Sub(mark)
			mark = now
		}
	}

	x := make([]float32, total*d)
	for s, seq := range seqs {
		for p, id := range seq {
			row := starts[s] + p
			e.Weights.Embed.Lookup(x[row*d:(row+1)*d], int(id))
		}
	}
	w := e.Weights
	tensor.LayerNorm(x, x, total, d, w.EmbedNorm.Weight, w.EmbedNorm.Bias, cfg.NormEps, par)
	lap(OpEmbed)

	var res Result
	capture := func() {
		if opt.CaptureLayers {
			res.Layers = append(res.Layers, splitRows(x, starts, d))
		}
	}
	capture()

	h := make([]float32, total*d)
	qkv := make([]float32, total*3*d)
	attn := make([]float32, total*d)
	tmp := make([]float32, total*d)
	up := make([]float32, total*2*inter)
	act := make([]float32, total*inter)
	scale := float32(1 / math.Sqrt(float64(hd)))

	for li, layer := range w.Layers {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		runtime.Gosched()
		spec := cfg.Layers[li]
		src := x
		if layer.AttnNorm != nil {
			tensor.LayerNorm(h, x, total, d, layer.AttnNorm.Weight, layer.AttnNorm.Bias, cfg.NormEps, par)
			src = h
			lap(OpNorm)
		}
		layer.Wqkv.Forward(qkv, src, total, par)
		lap(OpQKV)
		rope := ropes.byTheta[spec.RopeTheta]
		par.Run((total+15)/16, func(task int) {
			for row := task * 16; row < min(total, task*16+16); row++ {
				pos := row - starts[seqIndex(starts, row)]
				base := row * 3 * d
				for hh := 0; hh < heads; hh++ {
					rope.Apply(qkv[base+hh*hd:base+(hh+1)*hd], pos)
					rope.Apply(qkv[base+d+hh*hd:base+d+(hh+1)*hd], pos)
				}
			}
		})
		attentionAll(attn, qkv, starts, d, heads, hd, spec.Window, scale, par)
		lap(OpAttention)
		layer.Wo.Forward(tmp, attn, total, par)
		tensor.AddInPlace(x, tmp)
		lap(OpAttnOut)
		tensor.LayerNorm(h, x, total, d, layer.MLPNorm.Weight, layer.MLPNorm.Bias, cfg.NormEps, par)
		lap(OpNorm)
		layer.Wi.Forward(up, h, total, par)
		lap(OpMLPIn)
		tensor.GeGLU(act, up, total, inter, par)
		lap(OpGeGLU)
		layer.WoM.Forward(tmp, act, total, par)
		tensor.AddInPlace(x, tmp)
		lap(OpMLPOut)
		capture()
	}
	tensor.LayerNorm(x, x, total, d, w.FinalNorm.Weight, w.FinalNorm.Bias, cfg.NormEps, par)
	lap(OpNorm)
	res.Hidden = splitRows(x, starts, d)
	return &res, nil
}

// attentionAll computes softmax(q·kᵀ·scale + mask)·v for every (sequence,
// head) pair. window < 0 is global attention; otherwise keys with
// |i-j| > window are masked, matching ModernBERT's sliding-window mask.
func attentionAll(out, qkv []float32, starts []int, d, heads, hd, window int, scale float32, par tensor.Parallel) {
	nseq := len(starts) - 1
	par.Run(nseq*heads, func(task int) {
		s, hh := task/heads, task%heads
		start, n := starts[s], starts[s+1]-starts[s]
		scores := make([]float32, n)
		acc := make([]float32, hd)
		for i := 0; i < n; i++ {
			lo, hi := 0, n-1
			if window >= 0 {
				lo, hi = max(0, i-window), min(n-1, i+window)
			}
			q := qkv[(start+i)*3*d+hh*hd : (start+i)*3*d+(hh+1)*hd]
			sc := scores[:hi-lo+1]
			for j := lo; j <= hi; j++ {
				k := qkv[(start+j)*3*d+d+hh*hd : (start+j)*3*d+d+(hh+1)*hd]
				sc[j-lo] = tensor.Dot(q, k) * scale
			}
			tensor.SoftmaxInPlace(sc)
			clear(acc)
			for j := lo; j <= hi; j++ {
				p := sc[j-lo]
				v := qkv[(start+j)*3*d+2*d+hh*hd : (start+j)*3*d+2*d+(hh+1)*hd]
				for t := range acc {
					acc[t] += p * v[t]
				}
			}
			copy(out[(start+i)*d+hh*hd:(start+i)*d+(hh+1)*hd], acc)
		}
	})
}

func seqIndex(starts []int, row int) int {
	lo, hi := 0, len(starts)-2
	for lo < hi {
		mid := (lo + hi + 1) / 2
		if starts[mid] <= row {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	return lo
}

func splitRows(x []float32, starts []int, d int) [][]float32 {
	out := make([][]float32, len(starts)-1)
	for s := range out {
		out[s] = append([]float32(nil), x[starts[s]*d:starts[s+1]*d]...)
	}
	return out
}
