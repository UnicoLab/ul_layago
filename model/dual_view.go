package model

import (
	"math"
	"math/rand/v2"

	"github.com/UnicoLab/ul_layago/tensor"
)

// Tensor2D is a contiguous row-major [Rows, Cols] float32 matrix used by
// DualViewProjector and ColBERT late-interaction scoring.
type Tensor2D struct {
	Rows int
	Cols int
	Data []float32
}

// NewTensor2D wraps or allocates a [rows, cols] row-major float32 matrix.
// When data is nil, a zeroed slice of length rows*cols is allocated.
func NewTensor2D(rows, cols int, data []float32) *Tensor2D {
	if rows <= 0 || cols <= 0 {
		return &Tensor2D{}
	}
	need := rows * cols
	if data == nil {
		data = make([]float32, need)
	} else if len(data) < need {
		buf := make([]float32, need)
		copy(buf, data)
		data = buf
	} else {
		data = data[:need]
	}
	return &Tensor2D{Rows: rows, Cols: cols, Data: data}
}

// FromRows builds a contiguous Tensor2D from a slice of row slices.
func FromRows(rows [][]float32) *Tensor2D {
	if len(rows) == 0 || len(rows[0]) == 0 {
		return &Tensor2D{}
	}
	r, c := len(rows), len(rows[0])
	data := make([]float32, r*c)
	for i, row := range rows {
		copy(data[i*c:(i+1)*c], row)
	}
	return &Tensor2D{Rows: r, Cols: c, Data: data}
}

// Row returns the r-th row slice [Cols] backed by t.Data.
func (t *Tensor2D) Row(r int) []float32 {
	if t == nil || r < 0 || r >= t.Rows || t.Cols <= 0 {
		return nil
	}
	return t.Data[r*t.Cols : (r+1)*t.Cols]
}

// ToRows returns the matrix as a slice of row slices, suitable for passing to
// tensor.EffectiveRank.
func (t *Tensor2D) ToRows() [][]float32 {
	if t == nil || t.Rows <= 0 || t.Cols <= 0 {
		return nil
	}
	out := make([][]float32, t.Rows)
	for r := 0; r < t.Rows; r++ {
		row := make([]float32, t.Cols)
		copy(row, t.Data[r*t.Cols:(r+1)*t.Cols])
		out[r] = row
	}
	return out
}

// Clone returns a deep copy of t.
func (t *Tensor2D) Clone() *Tensor2D {
	if t == nil || t.Rows <= 0 || t.Cols <= 0 {
		return &Tensor2D{}
	}
	cp := make([]float32, len(t.Data))
	copy(cp, t.Data)
	return &Tensor2D{Rows: t.Rows, Cols: t.Cols, Data: cp}
}

// L2NormalizeInPlace scales v in place to unit L2 norm. A zero or near-zero
// vector is left zeroed without introducing NaNs.
func L2NormalizeInPlace(v []float32) {
	var ss float64
	for _, x := range v {
		ss += float64(x) * float64(x)
	}
	if ss <= 1e-24 {
		clear(v)
		return
	}
	inv := float32(1.0 / math.Sqrt(ss))
	for i := range v {
		v[i] *= inv
	}
}

// L2Normalize returns a unit-L2-normalized copy of v.
func L2Normalize(v []float32) []float32 {
	out := make([]float32, len(v))
	copy(out, v)
	L2NormalizeInPlace(out)
	return out
}

// L2NormalizeRows normalizes each row of t to unit L2 norm in place.
func L2NormalizeRows(t *Tensor2D) {
	if t == nil || t.Rows <= 0 || t.Cols <= 0 {
		return
	}
	for r := 0; r < t.Rows; r++ {
		L2NormalizeInPlace(t.Data[r*t.Cols : (r+1)*t.Cols])
	}
}

// DualViewConfig configures UltraBERT's Decoupled Dual-View Representation Head.
//
// In UltraBERT, a shared encoder trunk serves both classification/NER (trained
// at 5% masking with teacher logit distillation) and dense/multi-vector
// retrieval (trained at 50% masking without logit distillation). Decoupling
// the output head into two specialized projections avoids the spectral rank
// collapse that classification logit distillation otherwise induces on
// retrieval embeddings.
type DualViewConfig struct {
	HiddenSize        int
	ClassificationDim int
	RetrievalDim      int
}

func (c DualViewConfig) withDefaults() DualViewConfig {
	if c.HiddenSize <= 0 {
		c.HiddenSize = 768
	}
	if c.ClassificationDim <= 0 {
		c.ClassificationDim = c.HiddenSize
	}
	if c.RetrievalDim <= 0 {
		c.RetrievalDim = c.HiddenSize
	}
	return c
}

// DualViewProjector holds two decoupled linear readout projections over a
// shared encoder trunk:
//   - Classification view (ClassWeight, ClassBias): preserves unnormalized
//     activation magnitudes and sharp token/span decision boundaries for 5%-view
//     classification, SpanChoiceReadout, and NER tagging.
//   - Retrieval view (RetrWeight, RetrBias): semi-orthogonal isotropic
//     projection followed by per-token L2 normalization for ColBERT late-
//     interaction multi-vector retrieval and mean-pooled dense cosine search.
type DualViewProjector struct {
	Config      DualViewConfig
	ClassWeight []float32 // [ClassificationDim, HiddenSize] row-major
	ClassBias   []float32 // [ClassificationDim]
	RetrWeight  []float32 // [RetrievalDim, HiddenSize] row-major
	RetrBias    []float32 // [RetrievalDim]
}

// NewDualViewProjector constructs a deterministic DualViewProjector using
// Gram-Schmidt semi-orthogonal initialization seeded by seed.
func NewDualViewProjector(cfg DualViewConfig, seed int64) *DualViewProjector {
	cfg = cfg.withDefaults()
	u := uint64(seed)
	rng := rand.New(rand.NewPCG(u^0x6a09e667f3bcc908, u^0xbb67ae8584caa73b)) //nolint:gosec // deterministic model init
	return &DualViewProjector{
		Config:      cfg,
		ClassWeight: semiOrthogonalMatrix(rng, cfg.ClassificationDim, cfg.HiddenSize),
		ClassBias:   make([]float32, cfg.ClassificationDim),
		RetrWeight:  semiOrthogonalMatrix(rng, cfg.RetrievalDim, cfg.HiddenSize),
		RetrBias:    make([]float32, cfg.RetrievalDim),
	}
}

// semiOrthogonalMatrix generates a deterministic [rows, cols] semi-orthogonal
// matrix in float64 via modified Gram-Schmidt and rounds to float32. When
// rows <= cols, the rows are mutually orthonormal; when rows > cols, the
// columns are mutually orthonormal and rows are unit-normalized.
func semiOrthogonalMatrix(rng *rand.Rand, rows, cols int) []float32 {
	m := make([]float64, rows*cols)
	for i := range m {
		m[i] = rng.NormFloat64()
	}
	if rows <= cols {
		for r := 0; r < rows; r++ {
			row := m[r*cols : (r+1)*cols]
			for prev := 0; prev < r; prev++ {
				pRow := m[prev*cols : (prev+1)*cols]
				var dot float64
				for k := 0; k < cols; k++ {
					dot += row[k] * pRow[k]
				}
				for k := 0; k < cols; k++ {
					row[k] -= dot * pRow[k]
				}
			}
			var normSq float64
			for _, v := range row {
				normSq += v * v
			}
			if normSq > 1e-24 {
				inv := 1.0 / math.Sqrt(normSq)
				for k := range row {
					row[k] *= inv
				}
			}
		}
	} else {
		// Orthonormalize columns, then normalize each row to unit length.
		for c := 0; c < cols; c++ {
			for prev := 0; prev < c; prev++ {
				var dot float64
				for r := 0; r < rows; r++ {
					dot += m[r*cols+c] * m[r*cols+prev]
				}
				for r := 0; r < rows; r++ {
					m[r*cols+c] -= dot * m[r*cols+prev]
				}
			}
			var normSq float64
			for r := 0; r < rows; r++ {
				v := m[r*cols+c]
				normSq += v * v
			}
			if normSq > 1e-24 {
				inv := 1.0 / math.Sqrt(normSq)
				for r := 0; r < rows; r++ {
					m[r*cols+c] *= inv
				}
			}
		}
		for r := 0; r < rows; r++ {
			row := m[r*cols : (r+1)*cols]
			var normSq float64
			for _, v := range row {
				normSq += v * v
			}
			if normSq > 1e-24 {
				inv := 1.0 / math.Sqrt(normSq)
				for k := range row {
					row[k] *= inv
				}
			}
		}
	}
	out := make([]float32, rows*cols)
	for i, v := range m {
		out[i] = float32(v)
	}
	return out
}

// ProjectClassificationTokens projects encoder token states [seqLen, HiddenSize]
// into [seqLen, ClassificationDim], preserving activation magnitude and sharp
// token-level boundaries for 5%-view classification, SpanChoiceReadout, and NER.
func (p *DualViewProjector) ProjectClassificationTokens(seq *Tensor2D) *Tensor2D {
	if p == nil || seq == nil || seq.Rows <= 0 || seq.Cols != p.Config.HiddenSize {
		return &Tensor2D{}
	}
	inDim := p.Config.HiddenSize
	outDim := p.Config.ClassificationDim
	out := NewTensor2D(seq.Rows, outDim, nil)
	for r := 0; r < seq.Rows; r++ {
		x := seq.Data[r*inDim : (r+1)*inDim]
		dst := out.Data[r*outDim : (r+1)*outDim]
		for c := 0; c < outDim; c++ {
			w := p.ClassWeight[c*inDim : (c+1)*inDim]
			var b float32
			if c < len(p.ClassBias) {
				b = p.ClassBias[c]
			}
			dst[c] = b + tensor.Dot(x, w)
		}
	}
	return out
}

// ProjectClassificationPooled pools the classification token view via
// 0.5 * ([CLS] + Mean) into a single [ClassificationDim] representation.
func (p *DualViewProjector) ProjectClassificationPooled(seq *Tensor2D) []float32 {
	toks := p.ProjectClassificationTokens(seq)
	if toks.Rows <= 0 || toks.Cols <= 0 {
		return nil
	}
	dim := toks.Cols
	out := make([]float32, dim)
	cls := toks.Row(0)
	invRows := 1.0 / float64(toks.Rows)
	for j := 0; j < dim; j++ {
		var sum float64
		for r := 0; r < toks.Rows; r++ {
			sum += float64(toks.Data[r*dim+j])
		}
		mean := sum * invRows
		out[j] = float32(0.5 * (float64(cls[j]) + mean))
	}
	return out
}

// ProjectRetrievalTokens projects encoder token states [seqLen, HiddenSize]
// into [seqLen, RetrievalDim] and applies per-token L2 normalization for
// ColBERT-style late-interaction multi-vector retrieval across up to 32K tokens.
func (p *DualViewProjector) ProjectRetrievalTokens(seq *Tensor2D) *Tensor2D {
	if p == nil || seq == nil || seq.Rows <= 0 || seq.Cols != p.Config.HiddenSize {
		return &Tensor2D{}
	}
	inDim := p.Config.HiddenSize
	outDim := p.Config.RetrievalDim
	out := NewTensor2D(seq.Rows, outDim, nil)
	for r := 0; r < seq.Rows; r++ {
		x := seq.Data[r*inDim : (r+1)*inDim]
		dst := out.Data[r*outDim : (r+1)*outDim]
		for c := 0; c < outDim; c++ {
			w := p.RetrWeight[c*inDim : (c+1)*inDim]
			var b float32
			if c < len(p.RetrBias) {
				b = p.RetrBias[c]
			}
			dst[c] = b + tensor.Dot(x, w)
		}
	}
	L2NormalizeRows(out)
	return out
}

// ProjectRetrievalDense mean-pools the L2-normalized retrieval token view and
// L2-normalizes the resulting vector for single-vector HNSW/cosine search.
func (p *DualViewProjector) ProjectRetrievalDense(seq *Tensor2D) []float32 {
	toks := p.ProjectRetrievalTokens(seq)
	if toks.Rows <= 0 || toks.Cols <= 0 {
		return nil
	}
	dim := toks.Cols
	out := make([]float32, dim)
	invRows := 1.0 / float64(toks.Rows)
	for j := 0; j < dim; j++ {
		var sum float64
		for r := 0; r < toks.Rows; r++ {
			sum += float64(toks.Data[r*dim+j])
		}
		out[j] = float32(sum * invRows)
	}
	L2NormalizeInPlace(out)
	return out
}

// ColBERTMaxSim computes the normalized ColBERT late-interaction score
// (1 / Q) * sum_{i=1..Q} max_{j=1..D} dot(q_i, d_j) over token matrices.
func ColBERTMaxSim(queryTokens, docTokens *Tensor2D) float32 {
	if queryTokens == nil || docTokens == nil ||
		queryTokens.Rows <= 0 || docTokens.Rows <= 0 ||
		queryTokens.Cols <= 0 || queryTokens.Cols != docTokens.Cols {
		return 0
	}
	dim := queryTokens.Cols
	var sumMax float64
	for qi := 0; qi < queryTokens.Rows; qi++ {
		qRow := queryTokens.Data[qi*dim : (qi+1)*dim]
		maxDot := tensor.Dot(qRow, docTokens.Data[0:dim])
		for dj := 1; dj < docTokens.Rows; dj++ {
			dRow := docTokens.Data[dj*dim : (dj+1)*dim]
			if d := tensor.Dot(qRow, dRow); d > maxDot {
				maxDot = d
			}
		}
		sumMax += float64(maxDot)
	}
	return float32(sumMax / float64(queryTokens.Rows))
}

// DenseCosine computes the cosine similarity (dot product) of two L2-normalized
// dense retrieval vectors.
func DenseCosine(qVec, dVec []float32) float32 {
	if len(qVec) == 0 || len(qVec) != len(dVec) {
		return 0
	}
	return tensor.Dot(qVec, dVec)
}
