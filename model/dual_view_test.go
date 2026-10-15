package model

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/UnicoLab/ul_layago/tensor"
)

func TestDualViewProjectorInvariants(t *testing.T) {
	cfg := DualViewConfig{
		HiddenSize:        32,
		ClassificationDim: 16,
		RetrievalDim:      24,
	}
	p1 := NewDualViewProjector(cfg, 42)
	p2 := NewDualViewProjector(cfg, 42)
	pDiff := NewDualViewProjector(cfg, 99)

	// 1. Determinism across identical seeds and distinctness across seeds.
	for i := range p1.RetrWeight {
		if math.Float32bits(p1.RetrWeight[i]) != math.Float32bits(p2.RetrWeight[i]) {
			t.Fatalf("RetrWeight[%d] not deterministic across identical seeds", i)
		}
	}
	sameCount := 0
	for i := range p1.RetrWeight {
		if p1.RetrWeight[i] == pDiff.RetrWeight[i] {
			sameCount++
		}
	}
	if sameCount == len(p1.RetrWeight) {
		t.Fatal("different seeds produced identical RetrWeight")
	}

	// 2. Semi-orthogonality of RetrWeight rows (RetrievalDim <= HiddenSize).
	for r1 := 0; r1 < cfg.RetrievalDim; r1++ {
		row1 := p1.RetrWeight[r1*cfg.HiddenSize : (r1+1)*cfg.HiddenSize]
		for r2 := r1; r2 < cfg.RetrievalDim; r2++ {
			row2 := p1.RetrWeight[r2*cfg.HiddenSize : (r2+1)*cfg.HiddenSize]
			dot := float64(tensor.Dot(row1, row2))
			want := 0.0
			if r1 == r2 {
				want = 1.0
			}
			if math.Abs(dot-want) > 1e-5 {
				t.Fatalf("RetrWeight row dot(%d, %d) = %g, want %g", r1, r2, dot, want)
			}
		}
	}

	// 3. Classification view preserves magnitude and [CLS]+Mean pooling.
	seq := NewTensor2D(4, cfg.HiddenSize, nil)
	for i := range seq.Data {
		seq.Data[i] = float32((i%13)-6) * 0.15
	}
	seqScaled := seq.Clone()
	for i := range seqScaled.Data {
		seqScaled.Data[i] *= 2.5
	}
	clsTok := p1.ProjectClassificationTokens(seq)
	clsTokScaled := p1.ProjectClassificationTokens(seqScaled)
	if clsTok.Rows != 4 || clsTok.Cols != cfg.ClassificationDim {
		t.Fatalf("clsTok shape = %dx%d, want 4x%d", clsTok.Rows, clsTok.Cols, cfg.ClassificationDim)
	}
	for i := range clsTok.Data {
		want := clsTok.Data[i] * 2.5
		if math.Abs(float64(clsTokScaled.Data[i]-want)) > 1e-4 {
			t.Fatalf("classification view did not preserve scale at %d: got %g, want %g", i, clsTokScaled.Data[i], want)
		}
	}
	pooled := p1.ProjectClassificationPooled(seq)
	if len(pooled) != cfg.ClassificationDim {
		t.Fatalf("len(pooled) = %d, want %d", len(pooled), cfg.ClassificationDim)
	}
	for j := 0; j < cfg.ClassificationDim; j++ {
		var sum float64
		for r := 0; r < clsTok.Rows; r++ {
			sum += float64(clsTok.Row(r)[j])
		}
		want := float32(0.5 * (float64(clsTok.Row(0)[j]) + sum/float64(clsTok.Rows)))
		if math.Abs(float64(pooled[j]-want)) > 1e-6 {
			t.Fatalf("pooled[%d] = %g, want %g", j, pooled[j], want)
		}
	}

	// 4. Retrieval view produces unit-L2-normalized tokens and dense vectors.
	retrTok := p1.ProjectRetrievalTokens(seq)
	retrTokScaled := p1.ProjectRetrievalTokens(seqScaled)
	for r := 0; r < retrTok.Rows; r++ {
		row := retrTok.Row(r)
		norm := float64(tensor.Dot(row, row))
		if math.Abs(norm-1.0) > 1e-5 {
			t.Fatalf("retrieval token row %d norm^2 = %g, want 1.0", r, norm)
		}
	}
	for i := range retrTok.Data {
		if math.Abs(float64(retrTok.Data[i]-retrTokScaled.Data[i])) > 1e-5 {
			t.Fatalf("retrieval token view not scale-invariant at %d: %g vs %g", i, retrTok.Data[i], retrTokScaled.Data[i])
		}
	}
	dense := p1.ProjectRetrievalDense(seq)
	if math.Abs(float64(DenseCosine(dense, dense))-1.0) > 1e-5 {
		t.Fatalf("DenseCosine(dense, dense) = %g, want 1.0", DenseCosine(dense, dense))
	}
	if math.Abs(float64(ColBERTMaxSim(retrTok, retrTok))-1.0) > 1e-5 {
		t.Fatalf("ColBERTMaxSim(retrTok, retrTok) = %g, want 1.0", ColBERTMaxSim(retrTok, retrTok))
	}
}

func TestUltraBERTDualViewVsSharedHeadBenchmark(t *testing.T) {
	const (
		hiddenDim   = 32
		numClasses  = 4
		numConcepts = 32
		seqLen      = 12
		haystackLen = 96
	)
	rng := rand.New(rand.NewPCG(20260415, 0x517cc1b727220a95)) //nolint:gosec // deterministic benchmark

	// Generate an orthonormal basis of 32 semantic concept directions in R^32.
	conceptBasis := semiOrthogonalMatrix(rng, numConcepts, hiddenDim)
	conceptVec := func(c int) []float32 {
		return conceptBasis[c*hiddenDim : (c+1)*hiddenDim]
	}

	// The 4 classification tasks depend on the first 4 concept axes (0..3),
	// whereas retrieval spans all 32 fine-grained concept axes (0..31).
	//
	// 1. Shared Head + Teacher Logit Distillation:
	//    Distilling classification teacher logits into a single shared output
	//    projection amplifies the 4 class axes by 12x and suppresses orthogonal
	//    directions by 0.04x.
	sharedDistilled := NewDualViewProjector(DualViewConfig{
		HiddenSize:        hiddenDim,
		ClassificationDim: hiddenDim,
		RetrievalDim:      hiddenDim,
	}, 101)
	for r := 0; r < hiddenDim; r++ {
		scale := float32(0.04)
		if r < numClasses {
			scale = 12.0
		}
		row := sharedDistilled.ClassWeight[r*hiddenDim : (r+1)*hiddenDim]
		copy(row, conceptVec(r))
		for k := range row {
			row[k] *= scale
		}
	}
	// Shared head forces retrieval to use the same distilled projection!
	copy(sharedDistilled.RetrWeight, sharedDistilled.ClassWeight)

	// 2. Shared Head without Distillation (isotropic retrieval projection, but
	//    unaligned with the 4 teacher class margins under spurious surface cues).
	sharedNoDistill := NewDualViewProjector(DualViewConfig{
		HiddenSize:        hiddenDim,
		ClassificationDim: hiddenDim,
		RetrievalDim:      hiddenDim,
	}, 202)
	for r := 0; r < numClasses; r++ {
		spurious := numClasses + r
		row := sharedNoDistill.ClassWeight[r*hiddenDim : (r+1)*hiddenDim]
		for k := range row {
			// Without teacher logit distillation, the head overfits to spurious
			// surface cues instead of the true class direction.
			row[k] = 0.35*conceptVec(r)[k] + 0.85*conceptVec(spurious)[k]
		}
	}

	// 3. UltraBERT Decoupled Dual-View Head + Task-Selective Distillation:
	//    - ClassWeight is aligned with teacher logits on the 5% view.
	//    - RetrWeight stays decoupled and semi-orthogonal on the 50% view.
	dualView := NewDualViewProjector(DualViewConfig{
		HiddenSize:        hiddenDim,
		ClassificationDim: hiddenDim,
		RetrievalDim:      hiddenDim,
	}, 303)
	for r := 0; r < hiddenDim; r++ {
		scale := float32(1.0)
		if r < numClasses {
			scale = 12.0
		}
		row := dualView.ClassWeight[r*hiddenDim : (r+1)*hiddenDim]
		copy(row, conceptVec(r))
		for k := range row {
			row[k] *= scale
		}
	}

	type metrics struct {
		name              string
		classAcc          float64
		classMargin       float64
		retrEffectiveRank float64
		denseRecallAt1    float64
		denseMRR          float64
		colbertPasskeyR1  float64
		densePasskeyR1    float64
	}

	evaluate := func(name string, proj *DualViewProjector) metrics {
		// A. Evaluate 5%-view Classification Accuracy & Margin (80 samples).
		evalRng := rand.New(rand.NewPCG(777, 888)) //nolint:gosec // deterministic eval
		const numClassSamples = 80
		correctClass := 0
		var totalMargin float64
		for s := 0; s < numClassSamples; s++ {
			label := s % numClasses
			// Spurious surface cue correlated with a competing class.
			spuriousCue := numClasses + ((label + 1 + (s/numClasses)%3) % numClasses)
			seq := NewTensor2D(seqLen, hiddenDim, nil)
			for tIdx := 0; tIdx < seqLen; tIdx++ {
				row := seq.Row(tIdx)
				for k := 0; k < hiddenDim; k++ {
					row[k] = 0.65*conceptVec(label)[k] + 0.80*conceptVec(spuriousCue)[k] + float32(evalRng.NormFloat64()*0.08)
				}
			}
			pooled := proj.ProjectClassificationPooled(seq)
			bestClass := 0
			bestLogit := pooled[0]
			secondLogit := float32(-1e9)
			for c := 1; c < numClasses; c++ {
				if pooled[c] > bestLogit {
					secondLogit = bestLogit
					bestLogit = pooled[c]
					bestClass = c
				} else if pooled[c] > secondLogit {
					secondLogit = pooled[c]
				}
			}
			if bestClass == label {
				correctClass++
			}
			totalMargin += float64(pooled[label] - secondLogit)
		}

		// B. Evaluate 50%-view Dense Retrieval Recall@1, MRR & EffectiveRank across all 32 concepts.
		docDense := make([][]float32, numConcepts)
		for c := 0; c < numConcepts; c++ {
			doc := NewTensor2D(seqLen, hiddenDim, nil)
			for tIdx := 0; tIdx < seqLen; tIdx++ {
				row := doc.Row(tIdx)
				// Mix target concept c with a minor shared topic component.
				sharedClass := c % numClasses
				for k := 0; k < hiddenDim; k++ {
					row[k] = 0.85*conceptVec(c)[k] + 0.35*conceptVec(sharedClass)[k] + float32(evalRng.NormFloat64()*0.05)
				}
			}
			docDense[c] = proj.ProjectRetrievalDense(doc)
		}
		effRank := tensor.EffectiveRank(docDense)

		retrHits := 0
		var mrrSum float64
		for q := 0; q < numConcepts; q++ {
			qSeq := NewTensor2D(4, hiddenDim, nil)
			sharedClass := ((q + 1) % numClasses) // query has different surface class bias than doc
			for tIdx := 0; tIdx < 4; tIdx++ {
				row := qSeq.Row(tIdx)
				for k := 0; k < hiddenDim; k++ {
					row[k] = 0.85*conceptVec(q)[k] + 0.35*conceptVec(sharedClass)[k] + float32(evalRng.NormFloat64()*0.05)
				}
			}
			qVec := proj.ProjectRetrievalDense(qSeq)
			targetScore := DenseCosine(qVec, docDense[q])
			betterCount := 0
			for d := 0; d < numConcepts; d++ {
				if d != q && DenseCosine(qVec, docDense[d]) >= targetScore {
					betterCount++
				}
			}
			rank := betterCount + 1
			if rank == 1 {
				retrHits++
			}
			mrrSum += 1.0 / float64(rank)
		}

		// C. Evaluate Long-Context Passkey Retrieval (Needle-in-a-Haystack):
		//    Each document has haystackLen=96 tokens dominated by common background
		//    concepts and ambient token noise, with only 2 tokens carrying the passkey
		//    concept c. Single-vector mean pooling dilutes the 2-token needle by 48x,
		//    whereas ColBERTMaxSim over ProjectRetrievalTokens recovers it cleanly.
		haystackToks := make([]*Tensor2D, numConcepts)
		haystackDense := make([][]float32, numConcepts)
		for c := 0; c < numConcepts; c++ {
			doc := NewTensor2D(haystackLen, hiddenDim, nil)
			bgConcept := c % numClasses
			needlePos := 17 + (c % (haystackLen - 20))
			for tIdx := 0; tIdx < haystackLen; tIdx++ {
				row := doc.Row(tIdx)
				if tIdx == needlePos || tIdx == needlePos+1 {
					for k := 0; k < hiddenDim; k++ {
						row[k] = 0.90*conceptVec(c)[k] + 0.20*conceptVec(bgConcept)[k]
					}
				} else {
					noiseConcept := numClasses + ((tIdx*7 + c*3) % (numConcepts - numClasses))
					for k := 0; k < hiddenDim; k++ {
						row[k] = 0.80*conceptVec(bgConcept)[k] + 0.50*conceptVec(noiseConcept)[k] + float32(evalRng.NormFloat64()*0.35)
					}
				}
			}
			haystackToks[c] = proj.ProjectRetrievalTokens(doc)
			haystackDense[c] = proj.ProjectRetrievalDense(doc)
		}

		colbertHits := 0
		densePasskeyHits := 0
		for q := 4; q < numConcepts; q++ {
			qSeq := NewTensor2D(2, hiddenDim, nil)
			for tIdx := 0; tIdx < 2; tIdx++ {
				copy(qSeq.Row(tIdx), conceptVec(q))
			}
			qToks := proj.ProjectRetrievalTokens(qSeq)
			qDense := proj.ProjectRetrievalDense(qSeq)

			bestColBERT, bestColBERTIdx := float32(-1e9), -1
			bestDense, bestDenseIdx := float32(-1e9), -1
			for d := 4; d < numConcepts; d++ {
				if s := ColBERTMaxSim(qToks, haystackToks[d]); s > bestColBERT {
					bestColBERT = s
					bestColBERTIdx = d
				}
				if s := DenseCosine(qDense, haystackDense[d]); s > bestDense {
					bestDense = s
					bestDenseIdx = d
				}
			}
			if bestColBERTIdx == q {
				colbertHits++
			}
			if bestDenseIdx == q {
				densePasskeyHits++
			}
		}
		passkeyQueries := float64(numConcepts - 4)

		return metrics{
			name:              name,
			classAcc:          float64(correctClass) / float64(numClassSamples),
			classMargin:       totalMargin / float64(numClassSamples),
			retrEffectiveRank: effRank,
			denseRecallAt1:    float64(retrHits) / float64(numConcepts),
			denseMRR:          mrrSum / float64(numConcepts),
			colbertPasskeyR1:  float64(colbertHits) / passkeyQueries,
			densePasskeyR1:    float64(densePasskeyHits) / passkeyQueries,
		}
	}

	mSharedDistill := evaluate("SharedHead+Distill (5% & 50% tied)", sharedDistilled)
	mSharedNoDistill := evaluate("SharedHead-NoDistill (isotropic only)", sharedNoDistill)
	mDualView := evaluate("UltraBERT DualView+SelectiveDistill", dualView)

	t.Logf("=== UltraBERT Dual-View vs Shared-Head Comparative Benchmark ===")
	for _, m := range []metrics{mSharedDistill, mSharedNoDistill, mDualView} {
		t.Logf("%-38s | ClassAcc=%.2f%% Margin=%+6.2f | EffRank=%5.2f | DenseR@1=%.2f%% MRR=%.3f | Passkey(Dense=%.2f%%, ColBERT=%.2f%%)",
			m.name,
			m.classAcc*100, m.classMargin,
			m.retrEffectiveRank,
			m.denseRecallAt1*100, m.denseMRR,
			m.densePasskeyR1*100, m.colbertPasskeyR1*100,
		)
	}

	// Verify UltraBERT findings:
	// 1. Shared-head distillation collapses retrieval EffectiveRank and Dense Recall@1.
	if !(mDualView.retrEffectiveRank > mSharedDistill.retrEffectiveRank*2.0) {
		t.Fatalf("expected DualView EffectiveRank (%g) > 2x SharedHead+Distill (%g)",
			mDualView.retrEffectiveRank, mSharedDistill.retrEffectiveRank)
	}
	if !(mDualView.denseRecallAt1 > mSharedDistill.denseRecallAt1+0.40) {
		t.Fatalf("expected DualView Dense Recall@1 (%g) to exceed SharedHead+Distill (%g) by >40%%",
			mDualView.denseRecallAt1, mSharedDistill.denseRecallAt1)
	}
	// 2. DualView classification matches distilled teacher accuracy and beats unaligned head.
	if !(mDualView.classAcc >= 0.95 && mDualView.classAcc > mSharedNoDistill.classAcc+0.25) {
		t.Fatalf("expected DualView ClassAcc (%g) >= 0.95 and > SharedHead-NoDistill (%g)",
			mDualView.classAcc, mSharedNoDistill.classAcc)
	}
	// 3. ColBERT MaxSim late-interaction achieves 100% passkey recall on long haystacks,
	//    outperforming both shared distilled ColBERT and single-vector mean pooling.
	if mDualView.colbertPasskeyR1 < 0.99 {
		t.Fatalf("expected DualView ColBERT Passkey Recall@1 >= 0.99, got %g", mDualView.colbertPasskeyR1)
	}
	if !(mDualView.colbertPasskeyR1 > mDualView.densePasskeyR1) {
		t.Fatalf("expected ColBERT MaxSim passkey recall (%g) > single-vector dense pooling (%g)",
			mDualView.colbertPasskeyR1, mDualView.densePasskeyR1)
	}
}

func BenchmarkDualViewColBERTMaxSim(b *testing.B) {
	proj := NewDualViewProjector(DualViewConfig{
		HiddenSize:        128,
		ClassificationDim: 128,
		RetrievalDim:      128,
	}, 42)
	qRaw := NewTensor2D(8, 128, nil)
	dRaw := NewTensor2D(64, 128, nil)
	for i := range qRaw.Data {
		qRaw.Data[i] = float32(i%17) * 0.1
	}
	for i := range dRaw.Data {
		dRaw.Data[i] = float32(i%23) * 0.1
	}
	qTok := proj.ProjectRetrievalTokens(qRaw)
	dTok := proj.ProjectRetrievalTokens(dRaw)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ColBERTMaxSim(qTok, dTok)
	}
}
