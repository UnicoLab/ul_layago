# Hive findings applied to LayaGo and DelayLama

Source: the user's `hive-models/reports/laya/laya-report.html`, titled **Laya Training Report, W10 round 3, 2026-10-02**. This is synthetic Hive evidence, not a contract-extraction benchmark. Earlier summary/recommendation paragraphs still describe pending or killed runs that later sections resolve; the round-3 tables and §§7.7–7.8 are the relevant final decoder results. Instructions about founder approvals, queues or other resident services belong to that report's original context, not this package's operating instructions.

This is the report-specific assessment. The [shared learning log](learning-log.md)
combines it with the contract pilot, implementation checks and operational
lessons; [the index](README.md) links receipts and the reusable experiment format.

## What changes our priorities

- **Do not equate architecture with quality.** The frozen marker head lost to the lexical floor on Hive's locked tasks. The trained small-decoder J1 arm reached roughly 92.7% EN / 91.0% FR, while its larger zero-shot J0 competitor failed the overall gate. This makes a trained small reader worth a future *contract-specific* comparison; those accuracies cannot transfer to DelayLama.
- **Contrastive retrieval is a candidate generator, not proof.** The small embeddinggemma contrastive arm had attractive retrieval latency, but still missed the locked accuracy gate and showed offline/live decision flips. Keep exact source-span checks and an independently qualified semantic judgment behind any future ranker.
- **GLiNER smoke tests are not extraction qualification.** Section 12 tests loading and latency on one EN and one FR example. Decide variants had different span behavior, including no entities for some variants at the default threshold. This does not evaluate GLiNER2.5-Extract on contracts; measure exact spans and role/value association before integration.
- **More template data is not necessarily the remedy.** Saturating learning curves, shared templates, name/label shifts and confident cross-reference errors call for better splits, hard negatives, paraphrases and human-adjudicated contracts. Our own head confused missing information with contradiction and rejected valid price/notice wording. Mine new training examples from adjudicated corrections; do not mine the frozen challenge or locked test.
- **Calibration cannot repair a class-collapse or missing context.** Temperature alone cannot fix a label prior or distinguish contract completion from actual reception. Evaluate confusion matrices and false-support rates per language/task; any additive class-bias fit needs fresh independent certification and comparison before serving.
- **Ordering/retention are experiments, not invisible preprocessing.** The report's shortlist changed order, option count and length together. Keep input layout identical between cached training and live inference; fail on clipping by default. Isolate reorder-only and quote-window experiments and test compound claims/amendments.
- **Use story-level uncertainty and independent selection.** Paraphrases within a contract are correlated. Model selection must not reuse dev_conf certification labels as an unrestricted search target. Use a distinct selection split or nested grouped folds and re-certify the selected model on untouched data.

## Implemented in this package

The optional MLX cached-marker scorer uses the same typed losses, deterministic Go epoch permutations, train/dev_cal boundary, parameter layout and best-dev-NLL selection as the CPU path. Go retains encoder/cache, act fitting, temperatures, independent thresholds and artifact export. Actual Apple GPU forward/gradient parity and post-export Go NLL checks gate it. Float32 GPU arithmetic/optimizer moments are not bit-identical to Go's float64 masters.

`Config.ScheduleEpochs` / `schedule_epochs` now decouples cosine decay from the maximum early-stop cap. For a maximum of 30 epochs and an intended 8-epoch decay, set `max_epochs: 30, schedule_epochs: 8`; warmup uses the decay horizon and later steps stay at the floor. The default remains the original maximum-epoch horizon. This is an experiment control, not a demonstrated quality improvement.

The external supervisor monitors progress, process-group RSS, host memory pressure and wall duration, records child signals without assuming every SIGKILL is an OOM, and retains bounded logs, telemetry and a health report. MLX exposes bounded allocations and resumable epoch/AdamW checkpoints. A memory guard observes its process group rather than changing unrelated applications.

## Architecture experiments that remain

1. A stronger contract head with learned cross-token interaction. The complete public two-layer Laya head is already a serving baseline. Fine-tuning it needs a different training path and new forward/gradient/export parity. **This package currently caches markers and a pooled state, not every token**; the report's suggestion to reuse cached token states cannot be assumed here. Full-token caches need explicit disk/memory budgeting or activation recomputation.
2. A small LoRA-tuned typed decoder behind a resident worker, scored on the same contract corpus and exact input layout. Preserve the LLM extraction/grounding workflow, and compare quality, calibrated abstention, cold/warm latency and memory before considering a native Go port.
3. A lightweight contrastive clause shortlist using reusable candidate embeddings, plus a lexical baseline and an independent cross-reference reader. Its selection order must not become a hidden gold-label shortcut, and cached/live embedding parity is a release gate.

Do not stack these paths without measured complementary errors. Current DelayLama evidence review remains experimental; acceleration is not a legal-quality improvement or an automatic-acceptance policy.

## Current engineering receipt

[The 2026-10-02 acceleration receipt](validation/acceleration-20261002.json) pins
code revision, clean build, base/dataset/weights hashes, fitting config, all epoch
metrics, supervision outcomes and the contract challenge.

On one final sequential warm-cache comparison of the existing 600-example pilot,
the scorer epoch loops took **2.48 s in Go vs 0.31 s in MLX**; supervised total run
time was **4.91 s vs 3.54 s**. These are one-machine observations, not a general
8x training claim: GPU transfer/checkpoint/setup overhead counts outside the epoch
timer, and an earlier first GPU run took 10.23 s vs Go's 5.03 s. MLX accelerates the
head, not the frozen encoder's initial cache pass or serving.

The exported GPU head retained **64/72 held-out-domain**, **60/72 held-out-language**
and **9/12 French challenge** results, with no false-support predictions on that
small challenge. Full-run resume reproduced its exported weights hash. Go serving
still requires review; no fresh locked test was consumed for acceleration tuning,
and no extraction quality gain is asserted.
