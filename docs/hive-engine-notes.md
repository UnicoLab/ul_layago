# layago: M0-LAYA spike, M1 GEMM kernels and Go-side training

> Historical source notes retained from the supplied Hive engine. Paths,
> integration rules, commands and pending work describe that original snapshot;
> they are not the current standalone package instructions. Start with the
> [documentation index](README.md), [current usage](../README.md),
> [training guide](training.md) and [cross-project learning log](learning-log.md).

`internal/layago` is a cgo-free (`CGO_ENABLED=0`) Go replication, with Go
assembly only in the GEMM microkernels, of the
[Laya](https://github.com/NandhaKishorM/laya) typed-decision architecture
(Apache-2.0, source commit `573e5b62`, version 0.3.5) on the
ModernBERT/mmBERT encoder. It is the M0-LAYA spike of workstream W10 of the
Hive Intelligence Programme, plus the M1 assembly GEMM microkernels (arm64
NEON, amd64 AVX2+FMA), the harness for the M0-LAYA latency gate, and a pure-Go
trainer for Laya decision heads over a frozen multilingual trunk
(`train`, driven by `cmd/laya-train`; see
[Training in Go](#training-in-go-train-and-cmdlaya-train)).

The Hive binary imports only its inference packages (`laya`, `model`,
`tokenizer`, `tensor`, `safetensors`), and only through
`internal/routermodel`, which reads each Ask in shadow behind
`features.laya` (disabled by default, shadow at most, never changing an
answer). `train` and `cmd/laya-train` are offline: nothing in the binary
imports them. `internal/routermodel` consumes trained heads only through the
converged `hive.laya-artifact/v1` contract, which this trainer writes; the
compiled tokenizer format and readiness are later milestones.

Its outputs are nominations. A probability from an uncalibrated or
synthetically calibrated head is not evidence, carries no numeric authority and
must go through Hive's existing validators and exact-subject, unit and time
gates before anything uses it. The Laya checkpoints that were evaluated for
judgment on 2026-09-22 stay rejected for that use
(`docs/architecture/laya-judgment-evaluation.md`); they appear here only as
offline replication evidence.

## Packages

| Package | What it does |
|---|---|
| `safetensors` | Strict, fail-closed reader (read-only mmap on unix, `io.ReaderAt` fallback) and writer. F32/F16/BF16/F64 widen exactly to float32; I8 feeds the int8 path. Ranges must tile the buffer with no hole or overlap; duplicate or unknown header keys are rejected. Fuzzed. |
| `tokenizer` | Loads a Hugging Face `tokenizer.json` and reproduces tokenizers' ids and byte offsets for the mmBERT/Gemma-2 BPE family: added tokens (special, normalized, lstrip, rstrip, single_word), Replace/Prepend normalizers, Metaspace, BPE with byte fallback and fused unk, TemplateProcessing. Anything else is rejected at load. Fuzzed. |
| `tensor` | Tiled parallel GEMM over `nn.Linear` weights with an optional int8 weight path (symmetric per-row scales): the portable Go kernel (4x4 register tiles, 64x32 task blocks), which is the fallback and the oracle, and assembly microkernels for arm64 NEON (8x8) and amd64 AVX2+FMA (6x16) chosen per call by `Parallel.Kernel` (see [GEMM kernels](#gemm-kernels-m1)). Results are bit-identical across worker counts and batch positions for a given kernel. Also LayerNorm, exact-erf GELU, GeGLU, stable softmax and Hugging Face float32 rotary tables (portable Go). |
| `model` | ModernBERT forward pass: embeddings and bias-free LayerNorm, per layer pre-norm fused Wqkv, RoPE with a per-layer theta, global or sliding-window attention (keys with abs(i-j) <= local_attention/2), pre-norm GeGLU MLP, final norm. Reads `layer_types` when present (transformers 5) and otherwise `global_attn_every_n_layers`. Unpadded batching, context checks between layers, per-phase timings. |
| `laya` | The Laya inference contract: `build_sequence` (option budget shrinking, truncation and Python slicing quirks), the decision head (question-type embedding, two norm-first ReLU transformer layers, per-option `[MASK]` marker scorer with a softmax per question, act/escalate head), clamped temperatures per (type, option-count bucket), typed decoding for choice, score and noul (yes/no). `LoadCheckpoint` reads the published checkpoint layout; `LoadArtifact` reads a head trained by `tools/laya-train` ([below](#trained-artifacts-hivelaya-artifactv1)). |
| `train` | Go-side training and serving over a frozen trunk on the converged contracts: `hive.laya-dataset/v1` loading with the split invariants, `hive.laya-prompt/v1` rendering, dev halves by story and a sealed `locked_test`, a content-addressed, append-only and resumable feature store, a lexical catalog shortlist, Laya's decision head with zero head layers trained with hand-written gradients and AdamW, temperature scaling, fixed-sequence Learn-then-Test act thresholds, evaluation per split, language and task with held-out domains and languages, `hive.laya-artifact/v1` export (float32 and int8 directories) and served inference, and the real-weights latency bench. Standard library only. |

`importgraph_test.go` fails if any file imports anything but the standard
library, `golang.org/x/text`, `golang.org/x/sys` or `internal/layago` itself,
so the tree stays extractable as a standalone module.

## Parity evidence

The committed oracle is `testdata/parity/layago_reference.py`, an independent
float64 re-implementation that needs only Python and numpy. It generates a
tiny synthetic Laya checkpoint (d=64, 4 heads, 5 layers with global and
sliding layers, window 8, random weights stored as mixed F32/F16/BF16, a
synthetic BPE vocabulary) and the goldens. `crosscheck_torch.py` then checked
that oracle against the libraries it re-implements, from a pre-existing local
Python environment (nothing was downloaded); its output is
`testdata/parity/crosscheck_report.json`.

| Check | Reference | Result |
|---|---|---|
| Synthetic tokenizer, two layouts, 30 cases each | numpy reference; HF `tokenizers` 0.22.2 | Go exact on ids and byte offsets; reference exact against HF (60/60) |
| Real mmBERT `tokenizer.json` (256k vocabulary), 206 cases, 29 scripts, Laya prompts and fixture states, random strings | HF `tokenizers` 0.22.2 | 206/206 exact (`LAYAGO_TOKENIZER_JSON`) |
| Encoder, every hidden state, 4 sequences up to 49 tokens | numpy float64; `transformers` 4.57.6 `ModernBertModel` | Go vs numpy max abs 1.3e-5; numpy vs PyTorch 1.6e-5 |
| Laya head, 3 states x 7 questions (choice, score, noul, 1-option, 7-option shrink) | numpy float64; `laya` 0.3.5 `Agent` | Go vs numpy: logits 4.7e-6, probabilities 1.4e-6; identical sequences; numpy vs `laya`: logits 5.8e-6 |
| int8 path (encoder linears and embeddings) | numpy on dequantized weights | hidden states within 1e-4, logits 3.5e-6 |
| Real Laya multilingual checkpoint (mmBERT-base, 22 layers), Hive's frozen judgment fixture, 18 cases x 7 questions | PyTorch `laya` 0.3.5, CPU float32 | identical sequences and markers for all 126 questions; marker logits within 2.1e-5, probabilities within 2.9e-6, no choice flips (`LAYAGO_LAYA_CHECKPOINT`). The act head saturates at 1.0 on every input, so act parity there is vacuous. |
| Assembly GEMM kernels, 766 shapes (every residue of M mod 6 and 8, N mod 8 and 16, K mod 4, plus random M, N, K in 1..700), float32 and int8, with and without bias | the portable Go kernel; float64 | arm64 NEON: 3,184,773 of 3,184,773 elements bit-identical. amd64 AVX2 under Rosetta 2: with `GOAMD64=v3` bit-identical on every column the compiler fused (3,140,844 required, 3,161,507 identical), with `GOAMD64=v1` within tolerance. Every kernel within 3.5e-7 of Σ\|x·w\| of float64 |
| The encoder, Laya-head and int8 goldens above, per kernel | numpy float64 | NEON: the portable kernel's numbers exactly. AVX2 (v1, Rosetta 2): hidden states 1.4e-5, logits 3.8e-6, probabilities 2.0e-6, int8 logits 5.9e-6 |
| The real Laya multilingual checkpoint row above, per assembly kernel (`LAYAGO_LAYA_KERNEL`) | PyTorch `laya` 0.3.5, CPU float32 | NEON: logits 2.1e-5, probabilities 2.9e-6, the portable kernel's figures. AVX2 (v1, Rosetta 2): logits 2.1e-5, probabilities 3.8e-6. Both: no choice flips, identical sequences, the same correct count and mean top probability per question |
| The real checkpoint row, int8 (`LAYAGO_LAYA_PATH=int8`: per-row int8 embeddings and encoder linears, float32 head), NEON (the portable int8 kernel bit for bit) | PyTorch `laya` 0.3.5, CPU float32 | **Fails** the design's quantized bound (max \|Δp\| <= 0.005, no flip where PyTorch's margin exceeds 1e-3): logits within 0.48, probabilities within 0.091, 2 choice flips in 126 questions (PyTorch margins 0.023 and 0.045). int8 linears alone: 0.067 and 1 flip; int8 embeddings alone: 0.035 and none |
| Real jhu-clsp/mmBERT-base (MIT, revision `c5955035`, converted to safetensors outside the repository), full 22-layer encoder, every hidden state, EN/FR/DE/JA/AR questions and a 173-token mixed-language Laya prompt (sliding windows exercised) | `transformers` 4.57.3 `ModernBertModel`, CPU float32, eager attention (`testdata/parity/mmbert_base_golden.py`) | Token ids identical. NEON and portable (bit-identical to each other): hidden-state norms within 2.8e-6 relative, final sampled dimensions within 1.3e-5, every element of every state within 3.9e-6 of its row's largest magnitude (max abs 1.2e-2 on the massive-activation rows). AVX2 under Rosetta 2: 1.9e-6, 1.3e-5 and 5.1e-6 (`LAYAGO_MMBERT_DIR`) |
| The mmBERT-base `tokenizer.json` (sha256 `197d4cc5…`), 349 cases: the 206 above plus a 103-line business corpus in 60+ languages and scripts (`testdata/parity/multilingual_corpus.txt`) and more random strings | HF `tokenizers` 0.22.1 | 349/349 exact on ids and byte offsets |
| A head trained by `train` and exported, placed next to the tiny trunk's encoder tensors as a Laya checkpoint with `head_layers` 0 | `laya.LoadCheckpoint` + `Agent.Predict` | identical sequences and markers; logits within 6.0e-7, act probability within 1.5e-7 (111 items) |
| Served inference from exported `hive.laya-artifact/v1` directories (state → sequence → trunk → head → temperature → act), float32 and int8 | the trainer's own logits | bit-identical logits, act logits, probabilities, predictions and act decisions on all 444 instance-Q fixture items (float32) and every third (int8); both `parity_golden.json` re-serve bit for bit |
| The same artifacts (Q float32 and int8, R float32) in the converged loaders | `laya.LoadArtifact` (`w10/laya-train` `2cad0e69`, `w10/laya-integrate` `a509f540`); `routermodel.Load` with Hive's Q rendering (`w10/laya-integrate`) | accepted; token ids, state, question, temperature cell, decision and prediction as recorded; logits within 7.8e-7, probabilities within 2.4e-7 (`LAYAGO_CONVERGED_TREES`) |
| Go-trained heads on the real mmBERT-base and Laya-encoder trunks, float32 and int8 | the converged loaders, as above | accepted by both `laya.LoadArtifact` and `routermodel.Load`; logits within 4.8e-6, probabilities within 1.1e-6 |
| `hive.laya-prompt/v1` states and the dataset contract | the Python trainer's `render_state` and `dataset.load` | the 81 trainer-rendered states of `testdata/prompt` byte for byte; all 687 fixture states identical, the fixture accepted with the same data digest (`train/testdata/prompt_oracle.py`) |
| Act thresholds | the Python trainer's `ltt_threshold` | identical lambda, acted, errors, thresholds tested and start on four fixed cases |

The PyTorch run behind the three real Laya checkpoint rows (`testdata/parity/laya_checkpoint_golden.py`)
reproduces the 2026-09-22 evaluation record (transformers 5.17 on MPS) exactly:
the same correct count for each of the 7 questions and the same mean top
probability to six decimals. transformers 4.57 reads rope thetas only from
`global_rope_theta`/`local_rope_theta` and would default the local one to
10000, so that run states the checkpoint's `rope_parameters` values (160000)
in those keys; the Go config parser reads `rope_parameters` directly.

The quantization error of the int8 path on the tiny random model is large
(up to 0.3 absolute on final hidden states and 0.11 on probabilities) because
the weights are random with high gain; it is reported, not gated. The design's
int8 target (argmax agreement at 99.9 %, |Δp| <= 0.005) is gated on the real
checkpoint instead (`LAYAGO_LAYA_PATH=int8`), where per-row int8 misses it
(table above): int8 cannot carry the M0-LAYA gate until a finer quantization
passes that test.

Mutation checks: making the sliding window exclusive, swapping the GeGLU
halves, flipping the RoPE rotation sign, using GELU instead of ReLU in the
decision head, or flushing a pending unk before byte-fallback tokens each fail
the parity tests.

### Reproduce

```sh
go test ./internal/layago/...                       # goldens, fuzz seeds, import graph
python3 internal/layago/testdata/parity/layago_reference.py   # regenerate (numpy only)

# optional, against real artifacts already on disk (nothing is downloaded)
LAYAGO_TOKENIZER_JSON=/path/to/mmbert/tokenizer.json go test -run TestExternalTokenizerParity ./internal/layago/tokenizer
LAYAGO_LAYA_CHECKPOINT=/path/to/laya/multilingual go test -timeout 60m -run TestRealCheckpointParity ./internal/layago/laya
# LAYAGO_LAYA_KERNEL=generic|arm64_neon|amd64_avx2 picks the GEMM kernel (default: the best one)
# LAYAGO_LAYA_PATH=int8 quantizes the encoder and holds it to |Δp| <= 0.005 (fails today)

# the real mmBERT-base trunk (converted directory with config.json, model.safetensors, tokenizer.json)
LAYAGO_MMBERT_DIR=/path/to/mmbert-base go test -run TestRealMMBERTBaseParity ./internal/layago/model
# element-by-element: first write the full dump outside the repository
/opt/miniconda3/bin/python3 internal/layago/testdata/parity/mmbert_base_golden.py --model /path/to/mmbert-base \
    --out internal/layago/model/testdata/mmbert_base_golden.json --full /tmp/mmbert_base_full.safetensors
LAYAGO_MMBERT_DIR=/path/to/mmbert-base LAYAGO_MMBERT_FULL_GOLDEN=/tmp/mmbert_base_full.safetensors \
    go test -run TestRealMMBERTBaseParity ./internal/layago/model   # LAYAGO_MMBERT_KERNEL picks the kernel
LAYAGO_TOKENIZER_JSON=/path/to/mmbert-base/tokenizer.json LAYAGO_TOKENIZER_GOLDEN=testdata/mmbert_base_tokenizer_golden.json \
    go test -run TestExternalTokenizerParity ./internal/layago/tokenizer
```

## Trained artifacts (hive.laya-artifact/v1)

`tools/laya-train` trains Laya heads over a frozen trunk and exports them as
`hive.laya-artifact/v1` directories (contract in
`docs/architecture/intelligence-contracts.md`, "Model engine and data").
`laya.LoadArtifact(dir, baseDir, opts)` loads one over the base model files in
`baseDir`:

- it recomputes the sha256 of the head tensors, the base config, the base
  weights and `tokenizer.json`, and checks that the cls, sep, mask and pad
  tokens have the ids they had in training; any mismatch, a file name outside
  its directory, or an out-of-range temperature or threshold fails the load;
- it loads the first `layers_used` trunk layers (`model.Config.Truncate` and
  `model.LoadOptions.SkipDeeperLayers`: the student of the M0-LAYA rule is
  the checkpoint's first layers followed by its final norm);
- head linears may be I8 with a per-row `<name>_scale` sidecar, which feeds
  the int8 GEMM path unchanged;
- `Artifact.Predict` builds each question with the task's trained type,
  instructions and fixed options, applies the temperature of
  `<task>:<bucket>`, `<task>` or `*`, and the task's certified act threshold
  (`floor(p·10000 + 0.5) / 10000 ≥ lambda`; uncertified tasks escalate).

`testdata/artifact/tiny-q/{float32,int8}` are heads trained by
`python -m laya_train parity-fixture` on the generated fixture dataset over
the synthetic parity trunk truncated to 4 of its 5 layers.
`TestArtifactMatchesPyTorchGolden` replays their PyTorch CPU float32 goldens
through every kernel: token ids and markers exact, logits within 1.8e-5 and
probabilities within 2.9e-6 on arm64 (generic and NEON) and amd64 (generic
and AVX2 under Rosetta 2), float32 and int8 (tolerances 1e-4 and 2e-5).
`TestExternalArtifactParity` runs the same check on a real export:

```sh
LAYAGO_ARTIFACT_DIR=/path/artifact/float32 LAYAGO_ARTIFACT_BASE_DIR=/path/mmBERT-base-trunk \
  go test -timeout 60m -run TestExternalArtifactParity ./internal/layago/laya
```

## GEMM kernels (M1)

`tensor.Linear.Forward` runs one of three kernels, chosen per call by
`Parallel.Kernel`:

| Kernel | Architecture | Tile per call | Available when |
|---|---|---|---|
| `generic` | any | 4x4 dot products (Go) | always; the fallback and the oracle |
| `arm64_neon` | arm64 | 8x8, `FMLA` by element | always (ASIMD is part of arm64), unless built with `-tags purego` |
| `amd64_avx2` | amd64 | 6x16, `VBROADCASTSS` + `VFMADD231PS` | CPUID reports AVX2 and FMA and XGETBV shows the OS saves YMM state; otherwise `generic` |

The zero value (`auto`) takes the best kernel available, and a kernel the
process cannot run resolves to `generic`. Receipts record
`Parallel.EffectiveKernel()`, never the request. `-tags purego` removes the
assembly.

How it works:

- **Packing.** Each `Linear` is re-laid out once into panels of nr output
  columns, K-major and zero-padded, lazily on first use or by `Prepack` at
  load (`Weights.Prepack`, `Agent.Prepack`). The panels are a second copy of
  the linear weights: +441 MB in float32 and +110 MB in int8 for mmBERT-base,
  about +46 MB for the 6-layer student. The design's off-heap prepacked panels
  would replace the row-major copy and are later work. `Quantize` drops panels
  built from float32 weights; weights must not change after the first
  `Forward`.
- **Preemption.** One assembly call computes one tile over at most 4096 K
  steps (`MaxCallKSteps`); every loop over tiles, panels and task blocks is in
  Go. Go assembly is not asynchronously preemptible: the runtime's stop signal
  cannot interrupt it, and a NOSPLIT leaf never checks the stack guard the
  runtime poisons to request a stop. So every call goes through a small
  non-inlined Go function (`callF32`, `callQ8`) whose prologue does check it,
  and a goroutine asked to stop does so before its next tile. The receipt's
  longest call is 8 µs at p50 and 10 µs at p99.
  `TestEveryAssemblyCallIsAPreemptionPoint` (default suite) type-checks the
  package for arm64, amd64 and purego and fails if a kernel is ever called
  other than through such a wrapper; `TestAssemblyKernelsStopPromptlyForTheRuntime`
  (opt-in: `LAYAGO_PREEMPT_TEST=1`) times the same property and fails with a
  median stop of 6 ms when the wrappers are removed.
- **Sharing a processor.** The runtime preempts Go code only after about
  10 ms, so a pass also yields: `Parallel.Run`, which every heavy operation
  goes through (GEMM tasks, attention per head, GELU, LayerNorm), calls
  `runtime.Gosched` between two tasks once its goroutine has run 0.5 ms
  without yielding, and the encoder yields between layers. A goroutine that
  shares the processor then waits about 0.5 ms plus one task.
  `TestRunYieldsBetweenTasks` pins the mechanism; the opt-in
  `TestYieldsBoundTickerLatenessAtGOMAXPROCS1` times it (below).
- **Arithmetic.** Each accumulator lane is one output element, updated by one
  fused multiply-add per K step, in order: x[r][k] (broadcast, or taken by
  lane) times nr consecutive weights. No reduction across lanes reorders a
  sum. So wherever the Go compiler fused the portable kernel's `c += a*b`, the
  result is bit-identical: every element on arm64; on amd64 with
  `GOAMD64=v3`, every element except the last `Out mod 4` columns, whose
  one-column loop the compiler turns into `MULSS` from memory plus `ADDSS`
  (no mmBERT projection has such columns). Below v3 (the release image builds
  at the default v1) the portable kernel rounds twice per step and the AVX2
  kernel once, and they agree within tolerance.
- **int8.** Panels hold the int8 weights. The kernel sign-extends (`SXTL`,
  `VPMOVSXBD`) and converts (`SCVTF`, `VCVTDQ2PS`) each weight in registers,
  which is exact for |q| <= 127, and the tile store applies the per-row scale
  and then the bias as two separately rounded steps, exactly as the portable
  path does. s·Σ x·q is the dequantisation Σ x·(q·s) performed once per
  output instead of once per weight.
- **Edges.** A partial row tile is padded, not sent to another kernel, so a
  row's bits do not depend on its position in a stacked batch. Partial column
  panels are zero-padded and their extra columns discarded.
- **Generation.** `tensor/asmgen` prints both `.s` files. It is plain Go (no
  third-party generator). It emits as `WORD` encodings, each checked against
  the system assembler (`as -arch arm64`), the by-element `FMLA` no Go
  assembler accepts and the int8 path's `SXTL`, `SXTL2` and `SCVTF`, which
  Go 1.27 accepts but Go 1.26 (the `go.mod` minimum and the toolchain CI and
  the release image pin) does not. `go generate ./internal/layago/tensor`
  regenerates them, `TestGeneratedAssemblyIsCurrent` fails on drift, and
  `TestGeneratedAssemblyAssemblesWithThisToolchain` assembles both files for
  their own architecture with the toolchain running the test, so CI's amd64
  Go 1.26.7 job assembles the arm64 kernel too.
- **Not covered.** Attention products (`tensor.Dot`), GELU, LayerNorm, RoPE
  and `MatMulT` stay portable Go. With the GEMM faster they take a larger
  share of each pass; attention is already the largest phase at 256 tokens
  (see the M1 receipt).

Tests: `TestKernelsMatchGenericAndFloat64` (the 766 shapes above),
`TestKChunkingIsBitIdentical`, `TestLinearIsDeterministicAcrossWorkersAndTiles`
and `TestForwardIsDeterministicAndBatchInvariant` (every kernel),
`TestQuantizeDropsPackedPanels`, `TestConcurrentFirstForwardPacksOnce`,
`TestKernelSelection`, `TestEveryAssemblyCallIsAPreemptionPoint`,
`TestRunYieldsBetweenTasks`, the asmgen encoding and assembly tests, and the
golden tests per kernel. Mutation checks, each caught by at least one of
them: a wrong `FMLA` lane, ignoring the accumulate flag, fusing the int8
scale with the bias, reading the wrong row in the AVX2 kernel, advancing the
AVX2 weight pointer by half a step in the tail, calling a microkernel field
directly, dropping `//go:noinline` from a wrapper, calling a kernel value
through a local variable, dropping the serial path's yield, and spelling
`SCVTF` as a mnemonic again (fails under Go 1.26.7, including on amd64).

```sh
go test ./internal/layago/...                                        # arm64: NEON against the portable kernel, bit for bit
GOTOOLCHAIN=go1.26.7 go test ./internal/layago/...                   # the pinned toolchain (go.mod minimum 1.26.6)
ROSETTA_ADVERTISE_AVX=1 GOARCH=amd64 go test ./internal/layago/...   # Apple silicon: the AVX2 kernel under Rosetta 2 (v1)
ROSETTA_ADVERTISE_AVX=1 GOARCH=amd64 GOAMD64=v3 go test ./internal/layago/tensor
GOARCH=amd64 go test ./internal/layago/tensor                        # AVX2 not advertised: dispatch falls back to generic
go test -tags purego ./internal/layago/...
```

Rosetta 2 (macOS 26 on this host) executes AVX2 and FMA but advertises them in
CPUID only when `ROSETTA_ADVERTISE_AVX=1` is set; without it the dispatch
selects `generic`, which exercises the fallback. Timings under Rosetta mean
nothing, so the AVX2 kernel's speed is unmeasured until the gate run below.

## Latency receipts

### M0: portable kernel, saturated host

`testdata/receipts/m0-latency-2026-09-26-apple-m3-max-contended.json` holds
the first numbers, produced by `TestWriteM0LatencyReceipt`
(`LAYAGO_RECEIPT_OUT=... go test -timeout 120m -run TestWriteM0LatencyReceipt ./internal/layago/laya`):
the full mmBERT-base shape (22 layers, d=768, 12 heads, GeGLU 1152,
256k vocabulary, random weights) at 32, 64, 128 and 256 tokens, float32 and
int8, with 1, 8 and 16 intra-op workers, plus the Laya head and eight
concurrent single-worker requests.

Read them with care. They were taken on this development machine (Apple M3
Max, 12 performance and 4 efficiency cores, darwin/arm64, Go 1.27, portable Go
kernel) while other build, lint and test jobs saturated it: the one-minute
load average was 173.6 at the start and 207.6 at the end, on 16 cores.
Wall-clock latency therefore measures the queue for a core, not the
engine, and a pass with 8 workers received less than one core on average.
Process CPU time per pass is the less contended signal, and it scales with
the FLOP count as expected. Rows are not comparable with each other on wall
time: the 128-token encoder-plus-head pass beat the encoder alone only
because the contention changed between them.

| Tokens | GFLOP per pass | CPU s per pass, fp32 (int8) | GFLOP per CPU-second, fp32 | Wall p50 / p95 s, fp32, 8 workers | Wall p50 / p95 s, encoder + Laya head, 8 workers |
|---|---|---|---|---|---|
| 32 | 7.1 | 0.47 (0.49) | 15.2 | 1.36 / 2.18 | 1.82 / 2.35 |
| 64 | 14.4 | 0.87 (1.08) | 16.5 | 3.10 / 4.94 | 5.07 / 6.69 |
| 128 | 29.2 | 2.05 (2.25) | 14.2 | 9.63 / 10.95 | 5.65 / 7.86 |
| 256 | 59.3 | 4.92 (4.63) | 12.1 | 9.72 / 16.22 | 13.44 / 16.54 |

What this does and does not show:

- Compute per pass is about 0.22 GFLOP per token (2 FLOPs per multiply-add),
  so a 64-token pass is 14.4 GFLOP. Meeting the M0 fast-lane gate for 22
  layers (p95 <= 120 ms) at 64 tokens needs at least 120 GFLOP/s sustained,
  15 GFLOP/s per vCPU with perfect scaling over 8.
- Under this contention the portable kernel delivered 12 to 17 GFLOP per
  CPU-second. A GEMM microbenchmark run earlier the same day
  (`BenchmarkLinear`, host load not recorded) reached about 26 GFLOP/s on one
  worker and about 220 GFLOP/s with 16 for the 2304x768 projection.
- Tokenizing a 47-token Laya prompt with the real 256k vocabulary took about
  0.18 ms under the same contention (`BenchmarkEncodeQuestion`), inside the
  design's 0.5 ms budget.
- The production gate is 8 vCPU AVX2 on Cloud Run under open-loop load, and
  none of this was measured on that shape. The release image is built without
  `GOAMD64`, i.e. at v1, where the Go compiler does not fuse multiply-add, so
  the portable kernel will be slower per core there than on arm64. An AVX2
  microkernel, or a `GOAMD64=v3` build, has to exist before the gate run means
  anything (M1 adds the AVX2 kernel).
- The M0-LAYA decision (22 layers versus a smaller student) cannot be taken
  from this receipt. It should be re-run on a quiet host, and the gate itself
  run on the production shape.

### M1: portable versus NEON kernel on the same host

`testdata/receipts/m1-gemm-2026-09-26-apple-m3-max-contended.json`, produced
by `TestWriteM1KernelReceipt`
(`LAYAGO_KERNEL_RECEIPT_OUT=... go test -timeout 120m -run TestWriteM1KernelReceipt ./internal/layago/laya`),
compares the two kernels this machine can run on the same random
mmBERT-base weights. It is not the gate and says nothing about AVX2 speed.
The host was again shared with other agents' builds and lints: the one-minute
load average was 46 at the start, peaked at 122 and was 36 at the end, on 16
cores (12 performance, 4 efficiency). To keep the comparison fair the two
kernels alternate run by run, so both meet the same contention, and every
row carries the load it started under. Absolute wall times are therefore
contention, and even process CPU time is inflated when a thread lands on an
efficiency core; the ratios are the result.

Single-core GEMM on layer 0's projections (best of 15 alternating runs):

| Projection (N x K) | m | Portable fp32 / int8, GFLOP/s | NEON fp32 / int8, GFLOP/s |
|---|---|---|---|
| Wqkv 2304 x 768 | 64 | 17.8 / 15.2 | 62.4 / 49.5 |
| Wqkv 2304 x 768 | 256 | 17.3 / 16.3 | 68.9 / 54.9 |
| Wo 768 x 768 | 256 | 18.8 / 16.8 | 73.7 / 50.3 |
| Wi 2304 x 768 | 256 | 18.6 / 14.3 | 74.5 / 71.4 |
| WoM 768 x 1152 | 256 | 15.6 / 21.3 | 57.7 / 68.7 |

A `BenchmarkLinear` run at load average 12 earlier the same day measured
106 GFLOP/s for NEON float32 on the 64 x 2304 x 768 projection (82 for int8)
against 28 (26) for the portable kernel, about 80 % of this core's FMA peak
for NEON; under the receipt's contention the same ratio, about 3.5 to 4x,
holds at lower absolute numbers.

Encoder, 22 layers (process CPU ms per pass at p50, one worker; wall ms p50
with 8 workers):

| Tokens | Portable CPU ms | NEON CPU ms | CPU speedup | Portable / NEON wall, 8 workers | Portable time in the 4 projections |
|---|---|---|---|---|---|
| 32 | 419 | 132 | 3.2x | 260 / 84 | 90 % |
| 64 | 930 | 356 | 2.6x | 1187 / 586 | 86 % |
| 128 | 1953 | 838 | 2.3x | 1634 / 664 | 80 % |
| 256 | 4751 | 2262 | 2.1x | 1271 / 561 | 70 % |

int8 shows the same pattern (CPU speedup 2.9x at 32 tokens, 2.1x at 256).
What this does and does not show:

- The GEMM is 3.5 to 4x faster per core and the whole pass 2.1 to 3.2x: the
  rest of the pass (attention products, exact-erf GELU, LayerNorm, RoPE) is
  still portable Go, and by Amdahl's law it caps the gain at the portable
  pass's GEMM share (70 to 90 %). With NEON, attention is already the largest
  phase at 256 tokens (1688 ms against 1338 ms for the projections, one
  worker). That is the next target once the gate shows where the budget
  goes.
- NEON output equals the portable kernel's bit for bit on a full 64-token
  mmBERT-base pass, float32 and int8 (49,152 of 49,152 elements each).
- Preemption: the longest assembly call (one 8x8 tile over the 4096-step
  cap) took 8.3 µs at p50 and 10.2 µs at p99. With runtime.GC forced during
  8-worker 64-token passes, stop-the-world stopping latency was p50 0.08 ms
  and p99 0.39 ms with NEON against 0.12 and 0.79 ms with the portable
  kernel, and on a single tile 512 calls long p50 0.025 ms (tile time up to
  14 ms). Before every call became a preemption point (commit "make every
  assembly GEMM call a preemption point"), the first receipt run measured a
  p99 of 5.2 ms against the portable kernel's 0.9 ms, and the single-tile
  test a median of 6.3 ms. The ticker at GOMAXPROCS=1 woke 12 ms late at
  p50 (24 ms at p99) with NEON and 16 ms (32 ms) with the portable kernel:
  the runtime's 10 ms preemption, not the kernel, set that floor. Those
  ticker numbers predate the yields in `Parallel.Run` (commit "yield between
  parallel tasks every 0.5 ms"). With them, `TestYieldsBoundTickerLatenessAtGOMAXPROCS1`
  (one synthetic 64-token layer on NEON, one worker) measured the ticker at
  p50 0.08 to 0.6 ms against 24 to 30 ms without them, over three runs on
  this host at a load average near 200. Its p99 was 28 to 58 ms with the
  yields and 60 to 104 ms without, because the OS deschedules the busy thread
  itself: an idle process's ticker already had a p99 of 4 to 15 ms here. The
  design's p99 <= 2 ms is therefore unmeasured; the test judges it only where
  that idle floor is under 1 ms (gate step 2).
- An uncontended `BenchmarkEncoderMMBERTBase` run earlier the same day (load
  average 11, mean per pass, not a percentile) took 44 ms for a 64-token
  float32 pass on 8 workers with NEON against 139 ms portable. It shows the
  order of magnitude on this CPU only; the 120 ms gate is on 8 vCPU AVX2
  under open-loop load, where hyperthreads share FMA units and the AVX2
  kernel's speed is not yet measured.

## M0-LAYA gate procedure (8 vCPU AVX2)

The gate, as the programme plan states it (W10): numerical parity against
PyTorch, and p50/p95 at 32, 64, 128 and 256 tokens on the production shape,
8 vCPU AVX2, under open-loop load; ship 22 layers if the fast-lane p95 is at
most 120 ms, otherwise a fewer-layer student of the same engine. The fast
lane caps input at 64 tokens, so the verdict is read at 64 tokens; 32, 128
and 256 are reported for the quality lane and per-script caps. The procedure
below has not been run on the production shape; only short dry runs of the
harness on the Apple M3 Max (not receipts) checked that it works, and every
number in this README comes from that machine.

### Pre-registered decision rule

The rule is fixed before any gate run, in the harness (`gatePreregistration`
in `laya/gate_rule_test.go`, pinned by `TestGateRuleIsPreregistered`) and
here. Environment variables can take a run out of it but cannot change it.

1. **Load and budget.** A configuration passes at a depth when every
   64-token cell at each of 5, 10 and 20 rps per instance reads `pass`: p95
   from scheduled arrival at most 120 ms, with the queue drained. These are
   the design's fast-lane load rates ("an open-loop Poisson load of 5, 10
   and 20 rps per instance"), so there is no target rate to choose after
   reading the receipts. A run reads `not_a_gate_run` if any of these hold:
   - it lacks one of the three rates or changes the budget;
   - it has cells shorter than 120 s;
   - it times a kernel other than `amd64_avx2`;
   - it runs at a GOMAXPROCS other than 8;
   - it is built with a toolchain other than the release image's (`go1.26.7`,
     which `TestGateToolchainIsTheReleaseToolchain` checks against
     `deploy/docker/Dockerfile`).
2. **Candidates.** Only these serving configurations can carry a pass, and a
   pass binds production to that configuration:

   | Id | Path | Intra-op workers | Passes in flight | Run |
   |---|---|---|---|---|
   | C1 | fp32 | 8 | 1 | 1 |
   | C2 | int8 | 8 | 1 | 2 |
   | C3 | fp32 | 4 | 2 | 3 |
   | C4 | fp32 | 1 | 8 | 4 |

3. **Parity.** A pass counts only if its path meets the parity half on the
   real Laya checkpoint with the `amd64_avx2` kernel, checked by
   `TestRealCheckpointParity`. Both paths require no choice flip where
   PyTorch's top-2 margin exceeds 1e-3.

   | Path | Bound on max \|Δp\| | Status |
   |---|---|---|
   | fp32 | 1e-4 | Holds: AVX2 under Rosetta 2, NEON and portable. |
   | int8 | 0.005 | **Fails today**: max \|Δp\| 0.091 and 2 flips on NEON, which is the portable int8 kernel bit for bit. |

   C2 therefore stays informational until a better quantization passes that
   row (see [Deviations and limits](#deviations-and-limits)).
4. **Depth.** Ship 22 layers if a candidate passes at 22 layers with its
   parity holding. Otherwise run 6 at 6 layers and then at 4 layers. The
   student is the larger of those depths at which a candidate passes. A
   student still needs parity for its own trained export (the design's S6/S4
   goldens), and that export does not exist yet.

Every receipt carries `preregistration` and `decision`. The decision has
these fields:
- `verdict`: `pass`, `fail` or `not_a_gate_run`;
- `candidate`;
- `deviations`;
- `failing_cells`;
- `parity_prerequisite`.

The recorded M0-LAYA decision is this rule applied to the receipts. It is
never a rate, budget or configuration picked after reading them.

`TestM0LayaGate` (`laya/gate_test.go`) is the harness. Per request it runs the
encoder at the mmBERT-base shape (random weights: latency depends on shape,
not values) and the Laya head on one four-option question, the same work as
the M0 receipt's "encoder + laya_head" rows. Arrivals are Poisson at each
offered rate for a fixed time; at most `LAYAGO_GATE_INFLIGHT` passes run at
once and the rest queue. Latency counts from each request's scheduled
arrival, so queueing is included, and a cell whose backlog has not drained
by twice the offered time is marked saturated rather than reported on the
requests it finished. Each 64-token cell gets a verdict against the budget,
and the run a `decision` against the rule.

| Variable | Default | Meaning |
|---|---|---|
| `LAYAGO_GATE_OUT` | (unset: skip) | receipt path, or `-` to print it as one stdout line prefixed `LAYAGO_GATE_RECEIPT ` |
| `LAYAGO_GATE_TOKENS` | `32,64,128,256` | sequence lengths |
| `LAYAGO_GATE_RPS` | `5,10,20` | offered Poisson rates per instance |
| `LAYAGO_GATE_SECONDS` | `120` | offered duration per cell |
| `LAYAGO_GATE_WORKERS` | `GOMAXPROCS` | intra-op workers per pass |
| `LAYAGO_GATE_INFLIGHT` | `1` | passes that may run at once |
| `LAYAGO_GATE_KERNEL` | `auto` | `generic`, `amd64_avx2`, ...; a named kernel this CPU cannot run is refused |
| `LAYAGO_GATE_PATH` | `fp32` | `fp32` or `int8` |
| `LAYAGO_GATE_LAYERS` | `22` | first N mmBERT-base layers, for the student decision |
| `LAYAGO_GATE_BUDGET_MS`, `LAYAGO_GATE_BUDGET_TOKENS` | `120`, `64` | the cell verdicts (changing them makes the run `not_a_gate_run`) |
| `LAYAGO_GATE_SEED` | `1` | weights and arrival schedule |

### 1. Build, from the commit under test

```sh
git rev-parse HEAD                      # record it with the receipts
mkdir -p /tmp/layago-gate
export GOTOOLCHAIN=go1.26.7             # the release image's toolchain (deploy/docker/Dockerfile)
go version                              # must print go1.26.7; receipts record it as host.go_version
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go test -c -trimpath -o /tmp/layago-gate/laya.test   ./internal/layago/laya
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go test -c -trimpath -o /tmp/layago-gate/tensor.test ./internal/layago/tensor
```

The toolchain is pinned so the portable operations that dominate the
non-GEMM time (attention, GELU, LayerNorm) are compiled as production
compiles them; a newer local Go would time different code. Leave `GOAMD64`
unset: `deploy/docker/Dockerfile` builds the release binary at the default
(v1), and the AVX2 kernel is chosen at run time by CPUID either way; the
level only changes the portable fallback.

### 2. Package and run on the production shape

The production service runs on Cloud Run in `europe-west1`
(`deploy/terraform/main.tf`). Run the harness as a Cloud Run job of the same
shape, in the same region and with the same execution environment as the
service (the environment is not pinned in Terraform today, so record which
one ran):

```sh
cat > /tmp/layago-gate/Dockerfile <<'EOF'
FROM gcr.io/distroless/static-debian12:nonroot
COPY laya.test tensor.test /gate/
WORKDIR /gate
ENTRYPOINT ["/gate/laya.test"]
CMD ["-test.run", "^TestM0LayaGate$", "-test.v", "-test.timeout", "4h"]
EOF
IMAGE=europe-west1-docker.pkg.dev/<project>/<repository>/layago-gate:<commit>
docker build --platform linux/amd64 -t "$IMAGE" /tmp/layago-gate && docker push "$IMAGE"

gcloud run jobs create layago-m0-gate --region europe-west1 --image "$IMAGE" \
  --cpu 8 --memory 8Gi --task-timeout 4h --max-retries 0 \
  --set-env-vars '^;^GOMAXPROCS=8;LAYAGO_GATE_OUT=-;LAYAGO_GATE_WORKERS=8;LAYAGO_GATE_INFLIGHT=1'
gcloud run jobs execute layago-m0-gate --region europe-west1 --wait
```

8 GiB leaves room for the float32 model (786 MB embedding table, 441 MB of
linear weights and their 441 MB packed copy) and the harness; the receipt
reports cgroup `memory.current` and `memory.peak` so the real need can be
read off. A job keeps its CPU allocated for the whole run, like a service
with CPU always allocated; if the production service throttles CPU between
requests, treat a borderline verdict as provisional until it is confirmed
there.

Before timing anything the harness checks itself on that CPU. It stops if
the kernel it would time is not the one requested: with `auto` on amd64 when
the AVX2 kernel is unavailable, and with any named kernel this CPU cannot
run. It then compares the timed kernel with the portable one on a 64-token
pass and stops above 1e-3 (`kernel_self_check` in the receipt).

Run the kernel test suite once on the same hardware, including the two
opt-in timing tests: the stop-the-world latency test and the
GOMAXPROCS=1 ticker test, whose p99 <= 2 ms can only be judged on a host
this quiet.

```sh
gcloud run jobs create layago-kernel-check --region europe-west1 --image "$IMAGE" \
  --cpu 8 --memory 2Gi --command /gate/tensor.test --args=-test.v \
  --set-env-vars LAYAGO_PREEMPT_TEST=1
```

Then execute it. Its log must say `best kernel on this CPU/build:
amd64_avx2` and end in `PASS`. Keep its log with the receipts.

Execute the gate job once per row, overriding variables per execution with
`gcloud run jobs execute layago-m0-gate --region europe-west1 --wait --update-env-vars '^;^NAME=value;...'`
(the `^;^` prefix lets values contain commas):

| Run | Overrides | Purpose |
|---|---|---|
| 1 | none (22 layers, float32, 32/64/128/256 tokens, 5/10/20 rps, 120 s per cell) | C1, the gate |
| 2 | `LAYAGO_GATE_PATH=int8` | C2, informational while the int8 parity row fails |
| 3 | `LAYAGO_GATE_TOKENS=64;LAYAGO_GATE_WORKERS=4;LAYAGO_GATE_INFLIGHT=2` | C3 |
| 4 | `LAYAGO_GATE_TOKENS=64;LAYAGO_GATE_WORKERS=1;LAYAGO_GATE_INFLIGHT=8` | C4 |
| 5 | `LAYAGO_GATE_TOKENS=64;LAYAGO_GATE_KERNEL=generic` | portable baseline (`not_a_gate_run` by design) |
| 6 | `LAYAGO_GATE_LAYERS=6`, then `4`, each with the overrides of runs 1, 3 and 4 in turn until one passes (only if no candidate passes at 22 layers) | the student fallback |
| 7 | `LAYAGO_KERNEL_RECEIPT_OUT=-;LAYAGO_RECEIPT_HOST_NOTE=<Cloud Run job, execution environment>` with `--args=-test.run=^TestWriteM1KernelReceipt$,-test.v,-test.timeout=2h` | kernel receipt on the production shape |

With `-` as the output, each receipt is one log line starting
`LAYAGO_GATE_RECEIPT ` (or `LAYAGO_KERNEL_RECEIPT `) followed by compact JSON:

```sh
gcloud logging read 'resource.type="cloud_run_job" AND resource.labels.job_name="layago-m0-gate" AND textPayload:"LAYAGO_GATE_RECEIPT "' \
  --freshness 1d --format 'value(textPayload)' | sed 's/^LAYAGO_GATE_RECEIPT //' | python3 -m json.tool > m0-gate-<date>-cloudrun-8vcpu-<run>.json
```

Commit each receipt under `internal/layago/testdata/receipts/` with the commit
it was built from.

### 3. Read the result

The harness has already refused a wrong kernel and applied the rule, so
start from `decision` in each receipt. Then check by hand what it cannot
see:
- `host.cgroup_cpu_max`, where the environment exposes it, is a quota of 8
  CPUs, such as `800000 100000`;
- `host.go_version` is `go1.26.7`;
- the kernel-check log from step 2 ends in `PASS`.

Apply the pre-registered rule:
- The 22-layer engine ships if a candidate has `verdict` `pass` at 22 layers
  and its `parity_prerequisite` holds.
- Otherwise the student is the larger pre-registered depth at which a
  candidate passes.

Record the decision next to the receipts, together with every `fail` and
`not_a_gate_run`; the harness only reports.

## Training in Go (`train` and `cmd/laya-train`)

Laya heads are trained and served in Go, over a frozen multilingual trunk,
with the standard library only. Python appears only as a test oracle.

### The contracts

The trainer reads and writes the converged model contracts of
`docs/architecture/intelligence-contracts.md` ("Model engine and data";
they converged on `w10/laya-train` `2cad0e69` and `w10/laya-integrate`
`a509f540`, both merged into this tree with the W10 wave):
`hive.laya-dataset/v1` in, `hive.laya-prompt/v1` states and
`hive.laya-artifact/v1` out. The Python trainer (`tools/laya-train`) writes
the same artifacts, and Hive loads either through `laya.LoadArtifact` and
`routermodel.Load`, both in this tree. What that means in this package:

- **Dataset.** `LoadDataset` validates every line as `dataset.py`
  `parse_example` does, in the same order and without coercion: instance
  `Q`, `R` or `A` and an instance-prefixed task; `story_id` required; a
  `yes_no` label only as a JSON boolean, with optional `[false_text,
  true_text]` options; choice and score labels as option indices (a score
  label is a level index, not a number on a scale); `label_dist` soft labels
  on train only, with the label among the most probable options; LLM
  provenance ids for the LLM label sources. The split invariants (one split
  per story, held-out domains and languages absent from train and dev, by
  primary subtag) are checked over every instance, in the Python wording
  with file and line. The data digest is sha256 over the contract name and
  the sorted (file name, sha256) pairs.
- **Prompt.** States are rendered with `laya.RenderPromptState` (question,
  catalog, evidence, draft and sentence lines), the renderer `routermodel`
  serves with, and options with `laya.RenderOptions`, as the runtime renders
  them. Task instructions are the trainer's per-task defaults
  (`DefaultInstructions`, identical to the Python trainer's) unless
  `-instructions` overrides them; the manifest records them.
- **Splits.** dev is divided by a hash of the story (`DevHalf`, the Python
  `dev_half`) into `dev_cal`, which early-stops the scorer and fits the
  temperatures and the act head, and `dev_conf`, which only certifies the
  act thresholds. A story never feeds both. A dev leaving either half
  without a scorable example is refused; too-small cells are recorded as
  shortfalls (`training.dev_halves`) and a limitation. `locked_test` is
  sealed and shapes nothing, not even the task definitions, unless
  `-unseal-locked-test` is given with an `-access-log` that records every
  unsealing; a sealed one is published as its size and files only. A
  split-invariant violation or a rendered train example reappearing in an
  evaluated split is refused unless allowed, and then recorded.
- **Artifact.** One directory per numeric path, `float32/` and `int8/`,
  each with `model.safetensors` (Laya's tensor names; in `int8/` every
  linear weight is I8 with a per-row F32 `<name>_scale`), `manifest.json`
  (the converged sections: `prompt_format`, the base weights and config
  files, tensor prefix and layer types, `adapter: null`, `training.files`
  and the data digest, temperatures, thresholds, evaluation) and
  `parity_golden.json` in the converged format. The int8 head is
  recalibrated and recertified on its own outputs. Scores are compared with
  thresholds as `floor(p·10000 + 0.5) / 10000`.

### Pipeline

1. **Prepare.** Load, validate, check the split invariants, seal
   `locked_test`, define the tasks, render and lay out every example with
   Laya's `build_sequence`. An example whose option markers do not fit the
   limits is unscorable: never encoded or trained on, escalated and counted
   (`unscorable_rate`) in evaluation, as the runtime escalates it. When the
   trunk cannot encode all of train within a compute budget,
   `-train-fraction f -subsample-seed s` keeps ceil(f·n) train examples per
   (task, language, domain) stratum, ranked by a seeded sha256 of the id;
   dev and the held-out splits stay whole, the task definitions still come
   from every example, and the manifest keeps the dataset's files and digest
   and records the subsample (`training.train_subsample`, and a limitation).
2. **Features.** The frozen trunk runs once per distinct scorable
   sequence. The final hidden state at position 0 and at each option's
   `[MASK]` marker is written to an append-only store of sharded safetensors
   outside the repository (`hive.laya-feature-cache/v3`):
   - The store key covers the trunk's weights, config and tokenizer digests,
     its depth, numeric path and sequence limits, the kernel and the stored
     precision (float32 by default, which is exact; float16 on request).
     Each sequence is addressed by a digest of its token ids and markers, so
     a larger nested subsample, another split or another run encodes only
     the sequences the store lacks.
   - Shards are written atomically and only added. A stopped build resumes;
     a corrupt shard is removed and its sequences encoded again. One build
     writes a store at a time (an advisory lock); a view whose sequences are
     all stored reads the store without the lock.
   - Batches are encoded in parallel across cores. A shard's bytes do not
     depend on the worker count or the batch size.
   - `views/<sequence digest>.json` records each view's sequence lengths and
     the throughput of the build that last encoded for it: wall and process
     CPU seconds, tokens per second and per CPU-second, and the load
     average.
   - Q inputs may be laid out over a lexical catalog shortlist
     (`-shortlist k`): the k metrics and distinct scope values most similar
     to the question, most relevant first. A dropped gold option is never put
     back, and the manifest records the shortlist for the runtime. Every k
     reorders the catalog: a k at least the catalog's size keeps every entry
     but is not the whole-catalog layout (`-shortlist 0`).
3. **Head.** Laya's `DecisionModel` head with zero head transformer layers:
   the type embedding, the marker scorer (LayerNorm → Linear(d) → GELU →
   Linear(1), with a softmax per question) and the act head
   (Linear(d+4 → 256) → GELU → Linear(2) over position 0 and
   `laya.ActFeatures`). Laya's two head layers attend over every token, so
   they would need every token's state cached; the frozen trunk's own layers
   already contextualise each marker against the question, the other options
   and the state. `head.layers` is 0 in the manifest, which
   `laya.LoadArtifact` accepts.
   - Losses have hand-written gradients, against the label or its
     `label_dist`:
     - softmax cross-entropy for choice and score tasks; with a
       `label_dist` over evidence items to rank (`r_answer_rank`) this is the
       listwise softmax ranking loss;
     - on score tasks, plus 0.5 × the normalised squared error of the
       expected level (ordinal regression);
     - logistic loss on the yes/no margin.
   - Every parameter is checked against central finite differences, for
     one-hot and soft targets (`grad_test.go`).
   - AdamW keeps float64 master weights, with warmup and cosine decay.
     Mini-batches are shuffled with a seeded PCG. Early stopping restores the
     epoch with the best `dev_cal` NLL. The batch gradient is summed in fixed
     chunks and reduced in chunk order, so a run is bit-identical with 1 or
     8 workers (tested).
4. **Calibration and act.**
   - Temperatures per (task, option-count bucket), per task and overall
     (`*`) are fitted on `dev_cal` by the label's NLL, each only with at
     least `min_cell` items, clamped to [0.5, 5] and looked up in that order,
     then 1.0.
   - The act head is fitted on `dev_cal` as a class-balanced predictor of
     whether the scorer's answer is right.
   - Thresholds are certified per task on `dev_conf` by fixed-sequence
     Learn-then-Test over the 1e-4 grid: from lambda 1.0000 down, the exact
     binomial tail tests "error rate among acted items > alpha", stopping at
     the first p > delta, starting where at least
     max(`n_min`, ⌈3 ln(δ/2) / ln(1 − α)⌉) items act. Both scores (the
     calibrated top probability and the act head's P(act)) are candidates,
     and delta is split between them (Bonferroni), so the certificate holds
     whichever is read; P(act) is read only when its AUROC is at least 0.80
     and higher (the design's rule). On four fixed cases the result is
     identical to the Python trainer's `ltt_threshold`. Uncertified means
     escalate.
5. **Evaluation.** Every split (`train`, `dev_cal`, `dev_conf`,
   `heldout_domain`, `heldout_language`, and `locked_test` when unsealed),
   each overall, by language, by task and by language × task, with the
   Python trainer's metric definitions: accuracy, macro-F1, 10-bin ECE, NLL,
   Brier, coverage, abstention, selective accuracy and the unscorable rate.
   The held-out splits are the generalisation to new businesses and
   languages; `evaluation.generalisation` also splits every evaluated split
   into seen and unseen domains and languages.
6. **Export and serve.** `LoadArtifact` and `NewPredictor` serve an
   artifact with the trainer's own float64 head and reproduce its logits
   bit for bit on the feature kernel (`laya-train verify`); a trunk that
   differs in weights, config, tokenizer, special tokens, depth or limits is
   refused, as is a fixed-option task asked with other options.

### Proof against the converged loaders

The loaders that serve these artifacts in Hive, `laya.LoadArtifact` and
`routermodel.Load`, are in this tree. `TestConvergedLoadersServeGoArtifacts`
exports this trainer's artifacts and runs the loaders' own parity tests on
them in each module root `LAYAGO_CONVERGED_TREES` names (this checkout will
do); it is opt-in because it runs `go test` in each tree. A mirror of their
checks (`convergedManifestErrors`) runs in every test run. The results below
were recorded on the tiny trunk while the loaders were still on their
branches, which the first column names as provenance:

| Loader (where it was measured) | Go-exported artifact | Result |
|---|---|---|
| `laya.LoadArtifact`, `TestExternalArtifactParity` (`w10/laya-train` `2cad0e69`; `w10/laya-integrate` `a509f540`) | Q float32 and int8, R float32 | accepted (digests, prompt format, tasks, temperatures, thresholds, head tensors); token ids exact; logits within 7.8e-7, probabilities within 2.4e-7 of the Go goldens |
| `routermodel.Load` and Hive's own Q rendering, `TestRealArtifactParity` (`w10/laya-integrate` `a509f540`) | Q float32 and int8 | accepted (digests, adapter, data digest from `training.files`, Hive's Q task profiles); state, question, token ids, markers, temperature cell, decision and choice as recorded; logits within 7.8e-7 |
| Python trainer, `prompt_oracle.py` (`dataset.load`, `render_state` of `w10/laya-train`) | the 687-line fixture | loads it with no violation; the same data digest; all 687 states byte-identical |

```sh
# from the module root: this checkout holds both converged loaders
LAYAGO_CONVERGED_TREES=$PWD \
    go test -count=1 -run TestConvergedLoadersServeGoArtifacts -v ./internal/layago/train
```

```sh
go build -o /tmp/laya-train ./cmd/laya-train
/tmp/laya-train validate -data DATA.jsonl
/tmp/laya-train train -data DATA.jsonl -instance Q -model /path/to/mmbert-base \
    -cache /path/outside/repo/features -out /path/outside/repo/artifact-q   # writes artifact-q/float32 and artifact-q/int8
/tmp/laya-train verify  -artifact /path/outside/repo/artifact-q/float32 -model /path/to/mmbert-base
/tmp/laya-train predict -artifact /path/outside/repo/artifact-q/int8 -model /path/to/mmbert-base -data DATA.jsonl -limit 20
/tmp/laya-train bench   -model /path/to/mmbert-base -out receipt.json -depths 22,6,4 -workers 8,1
```

Evaluation beyond the manifest (`evalreport`, standard library only):

- `eval` decodes every item from the feature cache with each artifact under
  a root (`float32/`, `int8/`), refuses unless the rows reproduce the
  manifest's `evaluation.splits` exactly, and writes per-item predictions
  (probabilities at the fitted temperature and at 1, act decisions) for
  `dev_cal`, `dev_conf` and the held-out splits.
- `report` turns prediction files into accuracy and macro-F1 per task with
  95 % bootstrap intervals that resample whole stories, accuracy per
  language, top-1, worst-label and pooled one-vs-rest ECE before and after
  temperature, act coverage and selective risk with a Clopper-Pearson bound,
  confusion, majority-class baselines fitted on train, and paired
  comparisons (story-paired bootstrap, exact McNemar, Holm).
- `latency` times one served decision at a time (trunk pass, head over every
  option, temperature, act) at fixed lengths and on the items' own
  sequences end to end, for each artifact and optionally the int8 trunk.

```sh
/tmp/laya-train eval    -artifact ART -model TRUNK -data DATA.jsonl -cache FEATURES -out EVAL   # same -train-fraction as training
/tmp/laya-train report  -pred EVAL/a.float32.predictions.jsonl,EVAL/b.float32.predictions.jsonl -train-data DATA.jsonl \
    -pairs a/float32:b/float32 -out report.json -md report.md
/tmp/laya-train latency -artifact ART -model TRUNK -data dev.jsonl -tokens 32,64,128,0 -int8-trunk -out latency.json
```

### Runs on the real trunks (generated fixture, not the synthetic dataset)

The first run on the synthetic dataset itself (`hive.laya-dataset/v1` v2,
instance Q, both trunks, a stratified 20 % of train) is recorded in
[`proof/intelligence_bench/laya_q_v1`](../../proof/intelligence_bench/laya_q_v1/README.md).
The fixture runs below came before it.

The synthetic-data package's dataset still did not exist
(`scratchpad/w10/data` is absent). The loop was therefore run on the real
weights over this package's own contract fixture (`fixture_test.go`, 687
lines, written with `LAYAGO_TRAIN_FULL_FIXTURE_OUT`; the Python trainer's
loader accepts it with no violation). It is template data in the Python
fixture's shape, labels drawn from the spec before the text:

- instance Q: `q_operation` (4 fixed options), `q_metric_choice` (the
  catalog's 3 or 4 metrics, per example), `q_time_supported` (Laya's default
  criteria), `q_single_series` (`[false_text, true_text]` options) and
  `q_act_or_escalate`: 270 train, 129 dev (82 `dev_cal`, 47 `dev_conf` by
  story), 15 `heldout_domain` (energy, EN/FR/DE), 30 `heldout_language`
  (Japanese and Arabic, 15 each) and 43 sealed `locked_test`;
- instance R: `r_evidence_relevance` (three graded levels) and
  `r_answer_rank` (three evidence items, `label_dist` on train): 108 train,
  54 dev, 6 and 12 held out.

Every run: code `b4d6f2e3`, seed 1, the default configuration (30 epochs,
patience 4, `max_len` 256, `head_max_len` 192, float32 features), 22 frozen
layers. Receipts: `testdata/receipts/train-fixture-2026-09-28-*.manifest.json`
(the exported manifests). Accuracy / macro-F1 / 10-bin ECE; macro-F1 is the
Python trainer's, over every class key a split holds, so it mixes tasks.
This is plumbing and generalisation evidence of the `synthetic_engineering`
tier on a very small evaluation set (47, 15 and 30 items: a 95 % interval
on an accuracy of 0.7 at n=30 is about ±0.16), not a quality claim.

| Trunk (frozen, 22 layers), instance | best epoch | train | `dev_conf` | held-out domain (energy) | held-out languages (ja, ar) |
|---|---|---|---|---|---|
| jhu-clsp/mmBERT-base (raw MLM weights), Q | 30 of 30 | 0.885 / 0.906 / 0.191 | 0.745 / 0.546 / 0.151 | 0.733 / 0.507 / 0.127 | 0.700 / 0.537 / 0.144 (ja 0.733, ar 0.667) |
| convaiinnovations/laya `multilingual` encoder (mmBERT-base fine-tuned by Laya, already in the local Hugging Face cache), Q | 10 of 14 | 0.967 / 0.994 / 0.052 | 0.830 / 0.848 / 0.136 | 0.867 / 0.893 / 0.166 | 0.767 / 0.893 / 0.123 (ja 0.733, ar 0.800) |
| jhu-clsp/mmBERT-base, R | 4 of 8 | 0.417 / 0.270 / 0.064 | 0.350 / 0.250 / 0.169 | 0.167 / 0.100 / 0.433 | 0.250 / 0.214 / 0.179 |

- **Hive serves the Go-trained real-weights heads.** On every artifact
  above, float32 and int8, `laya.LoadArtifact` of both converged branches
  and, for Q, `routermodel.Load` with Hive's own rendering of each golden
  example's catalog, accept the artifact and reproduce its golden: logits
  within 4.3e-6 (mmBERT Q), 4.8e-6 (Laya encoder Q) and 1.4e-6 (R),
  probabilities within 1.1e-6. The trainer's own runtime re-serves every
  golden bit for bit (`laya-train verify`).
- **Every threshold escalates, and says why.** At alpha 0.05 and delta
  0.1 (split between the two scores) a threshold needs at least 176 acted
  `dev_conf` items; this fixture has 8 to 10 per task. The run records the
  10 shortfalls in `training.dev_halves` and a limitation, and Hive's
  loader lists every task as `uncertified_threshold`. Only the overall
  temperature was fitted (T = 1.19, 1.22 and 1.12); the task cells fall
  back to it below 30 `dev_cal` items.
- **The int8 heads cost little.** Each is recalibrated on its own outputs;
  against its float32 head it moves probabilities by at most 0.0024
  (mmBERT Q), 0.0037 (Laya encoder Q) and 0.0008 (R), with no argmax or
  decision flip on any of the 444 (Q) and 180 (R) items.
- **The Laya encoder is the stronger frozen trunk for Q**, as in the
  previous round (whose fixture and tasks differ, so the numbers are not
  comparable): its `[MASK]` states were trained for this marker scoring.
  That use still needs a founder decision, since the Laya checkpoints stay
  rejected for Hive judgment (`docs/architecture/laya-judgment-evaluation.md`).
- **Raw mmBERT does not learn R from 108 examples.** Both R tasks stay at
  chance (three options) on every evaluated split, and early stopping keeps
  epoch 4. Comparing a metric named in the question with the metric in an
  evidence sentence needs more data or a trained trunk; the listwise and
  ordinal losses are exercised, not validated, here.
- **Feature-cache throughput is contention-bound.** The mmBERT tokenizer
  gives a median of 120 tokens and a p95 of 186 per sequence. At a load
  average of 80 to 110 on 16 cores the cache ran at 1.8 to 2.9 items per
  second; per CPU-second it held at 131 to 138 tokens, the previous round's
  figure, so that is the rate to scale from.

Exact command for the synthetic dataset once it exists (instance Q, real
weights; the dataset's own splits carry the held-out domains and
languages):

```sh
/tmp/laya-train validate -data <scratchpad>/w10/data/<dataset>.jsonl
/tmp/laya-train train -data <scratchpad>/w10/data/<dataset>.jsonl -instance Q \
    -model /Users/piotrlaczkowski/hive-models/mmbert-base \
    -cache <scratchpad>/w10/features -out <scratchpad>/w10/artifacts/q-mmbert-base
# the same with -model <HF cache>/models--convaiinnovations--laya/snapshots/1c5edc17…/multilingual
```

### M0-LAYA on the real weights (this machine only)

`laya-train bench` times the frozen trunk's forward pass on the real
mmBERT-base weights, float32 and int8 (per-row weights, float compute), at
32/64/128/256 tokens and at the slice's typical length. Each length runs as
a single request with 8 (and 1) intra-op workers, and as 8 sequences batched
in one pass. It covers 22 layers and the first 6 and 4 layers of the same
engine. The receipt carries the process CPU time and the load average for
every cell, and applies the plan's rule mechanically (`decision`): ship 22
layers if the fast-lane single-request fp32 p95 at the slice's typical
length is at most 120 ms, else recommend the deepest timed student that
meets it. The served head adds 1.7 ms per four-option decision at d=768
(`BenchmarkHeadMMBERTBase`, portable float64 loops, contended).

The receipts are this machine's evidence only (Apple M3 Max, 16 cores,
darwin/arm64, NEON, Go 1.27), and every one was taken while other agents'
builds, lints and benchmarks shared the host:

| Receipt (`testdata/receipts/`) | 1-minute load average at those cells | fp32, 22 layers, 32 tokens, 8 workers: p50 / p95 / min ms | fp32, 22 layers, 64 tokens: p50 / p95 / min ms | fp32 6 layers, 64 tokens: p50 / p95 ms | Rule read mechanically |
|---|---|---|---|---|---|
| `m0-real-weights-2026-09-27-apple-m3-max-contended-run1.json` | 55–85 | 101 / 312 / 66 | 342 / 895 / 225 | 31.5 / 52.5 | fail → 6-layer student |
| `m0-real-weights-2026-09-27-apple-m3-max-contended-run2.json` (typical 45 tokens) | 17–82 | **32.5 / 34.1 / 25.5** (load 17) | 88 / 263 / 67.6 (load 22) | 45 / 91 (load 82) | fail → 4-layer student |
| `m0-real-weights-2026-09-28-apple-m3-max-contended-run3.json` (typical 45 tokens, fast-lane cells only) | 29–43 | 46.7 / 173 / 35.1 | **63.4 / 96.4 / 58.9** (load 30) | 81.7 / 166 (load 42) | fail (read at 45 tokens: 214 / 308 at load 43) → no timed depth |
| `m0-real-weights-2026-09-27-rosetta2-avx2.json` (`GOARCH=amd64`, AVX2 kernel under Rosetta 2) | 20–21 | 108 / 154 | 215 / 241 | 56 / 61 | fail → 6-layer student |

What these do and do not show:

- **The rule's outcome on this host is fail, in every receipt.** Each
  receipt's own `decision.verdict` is `fail`, run 3's included. The commit
  that added run 3 (`c65358fc`, "add the fast-lane receipt where 22 layers
  met 120 ms") overstates it: the rule reads the slice's typical length,
  45 tokens, and there run 3 measured 214/308 ms p50/p95 at a load of 43;
  its 32-token cell was 46.7/173 ms at a similar load. The one favourable
  cell, 64 tokens at 63.4/96.4 ms over 40 passes at a load of 30, is not
  what the rule reads.
- **None of these cells is a gate reading.** The cells have 8 to 40 timed
  runs. With 20 runs or fewer the nearest-rank p95 is the maximum, so a
  single slow or fast pass decides it. The harness now defaults to 100 runs
  per cell (`MinGateRuns`) and marks a verdict read from fewer as
  `gate_run: false` with the reason; the receipts above predate that field.
- **Contention dominates what failed.** The p95 that fails is the queue for
  a core. In run 2 the one cell timed at a load near 17, fp32 22 layers at
  32 tokens, ran at 32.5/34.1 ms p50/p95. The 64-token cell's fastest pass
  was 67.6 ms, twice that, as its FLOP count predicts (14.4 against 7.1
  GFLOP). The recommended student changes from run to run with the load,
  which a real verdict would not do, and in run 3 the 45-token cell was
  slower than the 64-token cell timed just before it, which only contention
  can do.
- **The pre-registered gate is unmeasured.** It is 8 vCPU AVX2 on Cloud Run
  under open-loop load, where hyperthreads share FMA units; see
  [the procedure](#m0-laya-gate-procedure-8-vcpu-avx2). The Rosetta receipt
  exercises the AVX2 kernel path, and its parity holds on the real weights
  (above), but translated timings say nothing about native AVX2 speed.
- **int8 is not faster on this path.** Weights are int8 but the arithmetic
  is float; the M1 kernel receipt measured it at 68–119 % of fp32 speed per
  core, depending on the projection. It also still fails the
  real-checkpoint parity bound, so it cannot carry the gate.
- **The shallow students are cheap:**
  - 6 layers at 64 tokens ran 18.5/20.6 ms p50/p95 on int8 at a load of 34;
  - 4 layers at 45 tokens ran 17.4/26.3 ms on fp32.
- **Recommendation (a judgment, not the rule's output).** Applied to these
  receipts, the rule says fail and recommends a student; that is recorded
  as the outcome on this host. The judgment is not to act on it yet: the
  failing cells are contention on a shared host with too few runs, and the
  least contended cells (32 tokens at a load of 17; 64 tokens at a load of
  30) are inside the budget. So keep the 22-layer trunk as candidate C1 and
  decide on the pre-registered gate: at least 100 timed runs per cell, on a
  quiet host or, for the decision itself, the production shape. If it fails
  there, the rule selects the 6-layer student. That student needs a
  distilled trunk, because the first 6 layers of the frozen model are not a
  trained student; the first-6-layer depths here only price the compute.

## Deviations and limits

- The tokenizer parses `tokenizer.json` at load time (0.6 to 3.3 s for the
  256k vocabulary on this host). The compiled binary format of the design is
  later work.
- Hugging Face quirks are mirrored on purpose and pinned by goldens: a pending
  unk lands after byte-fallback tokens, and text swallowed by an rstrip token
  is emitted again when it contains another added token.
- Only the tokenizer components listed above are supported, and only literal
  Replace patterns. NFC/NFKC, regex splits, ByteLevel, WordPiece and Unigram
  are rejected rather than approximated.
- `Predict` takes the state as text. Laya serialises dict states with
  `json.dumps(state, ensure_ascii=False)`; callers must do the equivalent.
  Criteria must be strings (Laya renders structured criteria as compact JSON).
- Sliding-window semantics follow transformers 4.57 (abs(i-j) <= window/2). The
  real-checkpoint run reproduces the transformers 5.17 record on 310-token
  sequences, so any difference there did not change those outputs.
- Only the four linear projections per layer (and the head's linears) use the
  assembly kernels. Go assembly is not asynchronously preemptible, so the
  kernels stay tile-sized with every loop in Go.
- Bit-identical output holds per (GOARCH, GOAMD64, kernel); across kernels it
  holds where the compiler fused the portable kernel (see
  [GEMM kernels](#gemm-kernels-m1)). The design's hand-off to avo for the
  assembly was replaced by an in-repo generator, since no new dependency or
  download was allowed.
- The packed panels are a second, on-heap copy of the linear weights.
- The int8 path misses the design's quantized bound on the real checkpoint:
  max |Δp| 0.091 with 2 choice flips (PyTorch margins 0.023 and 0.045) in
  126 questions, against 0.005 and none. Both halves contribute: per-row
  int8 on the encoder linears alone gives 0.067 and 1 flip, and on the
  embedding table alone 0.035. Candidates are finer scales (per group of K)
  or keeping the most sensitive tensors in float32, each measured with
  `LAYAGO_LAYA_PATH=int8`. Until one passes, int8 cannot carry the gate.
- Scheduling latency (design: ticker p99 <= 2 ms and GC stop-the-world p99
  <= 2 ms during inference, at GOMAXPROCS=1). Stop-the-world stopping is
  measured, at p99 0.39 ms with NEON. The ticker's p50 is now sub-millisecond
  with the `Parallel.Run` yields. Its p99 is not yet measured on a quiet
  host, and the committed M1 receipt's ticker rows predate the yields. One
  task is the floor with yields. At 256 tokens a global-attention head is
  one task of about 8 M multiply-adds in portable Go, several milliseconds,
  so the 2 ms bound can hold at the fast lane's 64 tokens without holding
  there.
- Training (`train`) departs from Laya's own recipe, and from the Python
  trainer's (which the artifacts do not depend on), in these ways:
  - The trained head has zero head transformer layers (`head.layers` 0).
    Laya's two attend over every token, which would need every token's
    state cached, and the trunk stays frozen. The Python trainer trains two.
  - Temperatures are keyed by (task, option-count bucket), as the plan and
    the contract ask, not by Laya's (question type, bucket).
  - The act head is a class-balanced correctness predictor fitted on
    `dev_cal`, not Laya's cost-weighted act objective (the Python trainer's
    is unweighted); its score is read only above AUROC 0.80, the design's
    rule, where the Python trainer takes the higher AUROC.
  - Score tasks add an expected-level regression term to the
    cross-entropy; the Python trainer adds spherical and RPS terms. Neither
    permutes training options.
- The trainer renders with `laya.RenderPromptState` itself (held to the 81
  trainer-rendered states by both packages' `TestPromptMatchesTrainer`).
  Only the catalog readers the shortlist and the floor use (`pyvalue.go`)
  still restate `laya`'s unexported helpers;
  `TestCatalogReadersAgreeWithTheRenderer` holds them to the renderer.
- float16 feature caches are not bit-exact against served inference. Only
  the default float32 cache is.
- The trainer's own runtime (`LoadArtifact`, `Predictor`) computes the head
  in float64 loops, bit-identical to training: 1.7 ms per four-option
  decision at d=768. Hive serves the same artifact with `laya.LoadArtifact`'s
  float32 GEMM, within the golden's tolerance.

## Not done yet

- The M0-LAYA gate itself: p50/p95 at 32/64/128/256 tokens on the production
  shape (8 vCPU AVX2, Cloud Run) under open-loop load, cold start and cgroup
  memory. The harness, the pre-registered rule and the procedure exist
  (above); nothing was measured on that shape, and the AVX2 kernel's speed
  is unmeasured.
- An int8 quantization that meets the real-checkpoint bound, and the
  ticker p99 on a dedicated host (see the last two deviations).
- Off-heap prepacked weights (replacing the row-major copy), K-blocking for
  small L1 caches if the gate host's GEMM efficiency calls for it, assembly
  or vectorised attention, GELU and LayerNorm, micro-batching.
- The compiled tokenizer, language router, shortlist, span and late-fusion
  heads. (Trained-head artifacts with pinned digests, per-task temperatures
  and act thresholds load through `LoadArtifact`.)
- Hive integration (`internal/routermodel`, config, readiness, receipts).
- Heads that certify act thresholds. The second run
  (`proof/intelligence_bench/laya_q_v2`) trained on the whole train split of
  dataset v4 with three seeds and English and French as the primary
  evaluation; its heads still certify no act threshold, and a hashed-feature
  floor (`laya-train floor`) beats them on English and French while the
  trunk wins on unseen languages. Also not done: head layers or a trained
  trunk, a distilled 6-layer student trunk, and real (non-synthetic)
  calibration. Every artifact trained so far carries the synthetic evidence
  tier and must not be used in primary mode.
- A `q_scope_choice` fixture. The trainer handles it (per-example options),
  but routermodel's `TestRealArtifactParity` maps only `q_metric_choice`
  predictions to Hive's option ids, so a `q_scope_choice` golden would fail
  that test on the `sc:` prefix rather than on the artifact; the fixture
  leaves the task out until the test maps it.
- One copy of the prompt helpers: `train/pyvalue.go` restates the catalog
  readers of `laya/prompt.go` (`pyStr`, `pyStrip`, `promptLabels`, `asList`,
  `asMap`, `promptList` as `catalogList`, and what they call) for the
  shortlist and the floor. Exporting them from `laya` would let the trainer
  drop them; until then `TestCatalogReadersAgreeWithTheRenderer` holds the
  two to the same output.
- A cross-check against the converged loaders in every test run: they are
  in this tree now, but `TestConvergedLoadersServeGoArtifacts` still runs
  them only when `LAYAGO_CONVERGED_TREES` names a module root. It should call
  `laya.LoadArtifact` and `routermodel.Load` directly.
