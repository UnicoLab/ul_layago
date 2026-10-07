# Shared frozen base and calibrated head adapters

The reusable serving path implements a lifecycle idea from the
[JEFF architecture review](research/jeff-architecture-review.md): separate a
verified shared base from task-specific decision weights and calibration.
Laya still uses a bidirectional encoder and contextualized option markers.
JEFF's decoder final-token readout and attention/MLP LoRA tensors are different
architectures and are not imported here.

## Supported lifecycle

`laya.LoadArtifactBase(artifactDir, baseDir, options)` verifies the first
artifact's encoder config, weights, tokenizer and special-token IDs and loads
one opaque `*laya.VerifiedBase`. `LoadArtifactWithBase(artifactDir, base)` verifies
each subsequent head and refuses different base, tokenizer, depth or sequence
format pins, even if dimensions happen to match. Sequence and head length
budgets remain independently pinned and enforced by each adapter, because
complete token sequences, not a shared context limit, determine encoder states. Every pinned file is parsed from the
same bytes/file handle that was hashed. Loaded files need not be reopened to
attach more heads. Treat loaded weights as immutable; low-level exported Agent
fields are not an invitation to modify a shared encoder.

`layago.OpenAdapters` loads an immutable map of heads once. `Predict` requires
an explicit adapter ID, uses the existing typed request validation and clipping
checks, and serializes computation through a shared cancellation-aware gate.
Head weights, task metadata, temperatures and act/escalate thresholds never
move between adapters. Same-name tasks in different heads are not equivalent
calibration distributions. A loaded artifact remains an unaccepted model
nomination requiring domain validation.

The existing `Open`, `LoadArtifact`, prediction contract and artifact-v1 files
remain compatible. There is no new artifact schema or trainable encoder adapter.
Both float32 and int8 heads can share the same frozen encoder. Complete heads
with transformer layers remain complete during inference; the zero-layer
trainer cannot silently warm-start from those layers. Unsupported non-null
artifact `adapter` payloads are explicitly rejected by both loader families.

## Cache scope and memory

The optional budget is zero by default. Cache entries hold final full-sequence
float32 encoder states before the selected head's type embeddings or head
transformer layers. The owning verified-base instance fixes encoder numeric
path, config/weights, tokenizer and sequence policy. Keys add the effective
kernel/architecture and all token IDs in order. The full encoded sequence
already includes the rendered question, options and supplied state. No causal
prefix, source substring, logits, probabilities or act decision is reused.

The mutex-protected LRU owns copies, returns copies, evicts least-recently-used
entries and skips an entry larger than its budget. `CacheStats` exposes hits,
misses, eviction count, entry count and bytes including conservative entry
overhead. That budget does not cover loaded weights, temporary inference
buffers, goroutine stacks or total RSS. Instrumented/capture-layer passes bypass
the cache, and failed/cancelled forwards do not publish partial states.
Concurrent low-level misses may compute the same sequence independently; no
request cancellation can poison another request's entry.

## Training and acceptance

Existing training caches already retain frozen marker and pooled features by
base/config/tokenizer/layout/numeric/kernel identity and exact sequences.
Independent heads can continue to train from that compatible frozen store with
hard or train-only soft targets. Warm starts clone their initial head; temperature
fitting on dev_cal and threshold certification on dev_conf are repeated.
Serving several heads does not establish domain quality or combine their risk
guarantees. There is no mixture/averaging or automatic adapter selection.

Included synthetic tests check head float32/int8 and complete-transformer
standalone/shared parity, actual existing synthetic training/export attachment,
private calibration/threshold ownership, mismatched pins, unsupported adapters,
exact sequence keys, bounded eviction, copying, instrumentation, cancellation,
concurrent requests and JSONL selectors. Race tests exercise the shared runtime.
These qualify mechanics, not real-domain accuracy or latency. True encoder LoRA,
low-rank head training, dynamic online replacement, prefix caching and learned
adapter routing require separate contracts and controlled experiments.
