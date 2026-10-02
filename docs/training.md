# Training recipe and acceptance gates

Read before building data or running training. The initial contract pilot lives in DelayLama's `delaylama/scripts/generate_laya_contract_data.py`; the generic warehouse fixture is only a pipeline smoke test.

See the [shared learning log](learning-log.md) for the measured motivation and
failure cases behind this recipe. Record new comparisons using the
[experiment template](experiments/TEMPLATE.md), then add receipts and findings to
the [documentation index](README.md).

## What the manuals establish

Reviewed on 2026-10-02:

- [Laya fine-tuning guide](https://github.com/NandhaKishorM/laya#fine-tuning), [Apple Silicon training script](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_mps.py), and [browser-agent specialization](https://github.com/NandhaKishorM/laya/blob/main/docs/finetune_browser_agent.md).
- [CLM training stages](https://github.com/Contrastive-LM/CLM#data-recipe), [fine-tuning experiment discipline](https://github.com/Contrastive-LM/CLM/blob/main/docs/FINETUNING.md), and [typed-decision trainer](https://github.com/Contrastive-LM/CLM/blob/main/train/finetune.py).
- [MLX documentation](https://ml-explore.github.io/mlx/build/html/index.html).
- The supplied Go engine's existing contracts, dataset validation, cache, gradients, calibration and export tests, archived in [Hive notes](hive-engine-notes.md#training-in-go-train-and-cmdlaya-train).

Laya's upstream full-model recipe loads pretrained weights, trains with proper-scoring rewards plus soft cross-entropy, calibrates temperatures and exports. Its Mac implementation uses PyTorch MPS with small micro-batches, accumulation and activation checkpointing. Its specialization example emphasizes real corrections and input format: synthetic phrasing and missing/truncated candidates can dominate apparent model quality.

CLM uses independent state/action embeddings and trainable projection heads. Its stages progress from broad paired data to hard negatives and task trajectories with replay. The fine-tuning implementation supports warm starts, frozen embedding caches, task-separated evaluation, and soft targets. These ideas inform our data and acceptance gates; **our Go marker scorer is not CLM's dual encoder and currently implements neither bidirectional InfoNCE nor upstream Laya RLCD**.

## Data, before training

Define a task and its label policy first. Appropriate contract pilots are clause-topic routing, candidate-span ranking, explicit support/contradiction/missing-information classification, and review-priority scoring. LLM extraction remains responsible for open-ended fields, contextual reasoning and explanations.

1. Construct facts and clean labels from a structured specification. Generate supported claims, close counterfactuals (amount, unit, date, party, obligation), missing evidence, amendments and negations. Do not use a model's self-agreement as gold.
2. Group every paraphrase, translation, quote window and counterfactual of a contract under one `story_id`. Split stories/contracts before rendering. A source or rendered train example appearing in evaluation fails preparation.
3. Separate held-out domains/languages from fitting data. Reserve independently authored templates and real, human-adjudicated contracts for the final challenge. Synthetic held-out domain identifiers alone do not establish real-domain generalization.
4. Add review corrections only with provenance and adjudication. A note/flag is not automatically a label. Keep earlier accepted training cases as replay; freeze all evaluation stories and never replay them into training.
5. Inspect sequence retention and option markers. Chunk long sources around exact citations; store source offsets in the application. Do not turn truncated or absent text into a negative label. `inspect` reports token retention; current low-level preparation preserves legacy clipping behavior, so inspect before accepting a dataset.

## JSONL contract

The preserved contract name is `hive.laya-dataset/v1`. Each line must supply `id`, `instance`, `task`, `task_type`, `input`, `options`, `label`, `label_source`, `split`, `lang`, `domain_id`, `story_id`, `generator_version`, `seed`.

Tasks are prefixed `q_`, `r_`, `a_` for Q/R/A. Input requires a string `question`. Q may contain `catalog`, R contains `evidence`, A may contain `draft`/`sentence`. Use `laya.RenderPromptState` for serving the same representation. Choice options are `{key,text}` and labels are indices; score labels are level indices; `yes_no` labels are JSON booleans. `label_dist` is allowed on training cases only. LLM-written examples require interaction provenance IDs.

Train only on `train`. Hash-separated `dev_cal` controls early stopping, temperature fitting and the act head; `dev_conf` certifies thresholds. A tiny dev split cannot certify safe coverage. `locked_test` remains sealed; only `evaluate --unseal-locked-test --access-log PATH` opens it and records the access. Holding out data does not prove its labels are correct.

## Run and compare

1. Record code revision, base/tokenizer digests, dataset digest, generation seed, fixed configuration and a baseline before changing architecture or objectives.
2. Run `validate`, then `inspect`. Use float32 frozen features and bounded cache concurrency. Compare the public full multilingual checkpoint, a majority baseline, and the specialized head on the same frozen challenge.
3. Fit the scorer with AdamW and early stopping; restore best dev_cal NLL weights. Fit the act head, fresh temperature calibration, then independent dev_conf thresholds. `--init-artifact` permits compatible zero-layer warm starts and discards inherited calibration; it is not an optimizer checkpoint.
4. Export to a new directory. Load with the production loader and verify numerical parity; evaluate the exported artifact rather than trusting training metrics alone.
5. Report accuracy/macro-F1 by task, language, domain and counterfactual family; NLL, ECE/Brier, abstention/coverage, selective accuracy and false support. Measure end-to-end warmed and cold latency with token counts. Inspect individual mistakes, including correct quote/wrong meaning.
6. Unseal the locked test only after selecting a candidate. Record failures/OOM/timeouts as failed runs, not zero-valued successful scores. A tested synthetic head starts in shadow/review mode. Promotion requires human-labelled contract evaluation and no unacceptable false-support regressions.

## Architecture experiments

Current inference accepts the original complete pretrained Laya head. Current training uses a cached frozen encoder and zero-transformer-layer scorer. Reuse the pretrained multilingual **encoder** to avoid relearning language representations; do not silently discard two pretrained head transformer layers and call that a compatible warm start.

Available now: an optional MLX cached-marker scorer with actual GPU parity checks; see [the package usage](../README.md#optional-apple-gpu-training-and-run-supervision-02). The common Go calibration/export boundary is unchanged. `schedule_epochs` allows decay over the intended horizon instead of tying it to the maximum early-stop cap.

Next controlled alternatives: full pretrained Laya head fine-tuning with upstream-compatible MPS/RLCD; a CLM-style dual-encoder retrieval/ranker for reusable clause candidates; an MLX encoder implementation or a stronger cross-token scorer. Keep common data, input formats and held-out stories; introduce one change per run and compare each artifact's own calibration/parity. Do not stack models merely because their architectures differ.

## Operations and acceleration

Implemented now: separate cache/head concurrency controls, durable feature shards, cancellation between batches, run deadlines, heap heartbeats, a soft Go memory budget, epoch histories and offline HTML charts. An external supervisor now survives child death, records exit code/signal and bounded logs, distinguishes progress stalls from liveness, samples process-group RSS and host memory pressure, terminates only its owned group, and persists status/telemetry plus a memory report. An observed SIGKILL is not automatically called an OOM. Fault tests cover stalls, crashes, memory-limit/pressure responses, unrelated-process isolation and descendant cleanup.

The optional MLX scorer requires an available Apple Metal GPU and caps MLX allocations/cache. Float32 forward/gradient parity is checked for every task/soft-target family; the exported best head must reproduce its development NLL in Go. Completed-epoch optimizer checkpoints pin immutable tensors and resume safely after an incomplete later write; data/config/code mismatches or corrupted tensors fail explicitly. Go act/calibration are recomputed after resume. Full GPU encoder training, full-head RLCD and CLM InfoNCE are still separate experiments.

Use a supervisor for either Go or MLX; only the MLX scorer currently resumes optimizer moments. RSS and pressure are sampled, so brief spikes may escape detection. Host OOM can also kill the supervisor; durable files make failure inspectable but do not guarantee survival. Stronger OS isolation/cgroup integration, portable notifications, Go optimizer-resume and full-token/GPU encoder parity remain work to qualify.

[Hive report findings](hive-report-insights.md) separate the report's final synthetic results from stale earlier sections and describe what transfers to contract analysis. Real contract quality, per-language calibration, exact-source grounding and independent tests remain promotion gates.
