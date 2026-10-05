# UL LayaGo

A reusable, pure-Go Laya runtime and frozen-encoder head trainer extracted from UnicoLab's Hive engine. Module: `github.com/UnicoLab/ul_layago`. Go components require **1.27**, use only the standard library, and provide cgo-free inference and CPU training. The optional MLX scorer trainer uses Python on Apple GPUs. Native ARM64 NEON and AMD64 AVX2 kernels have a portable `purego` fallback.

This is a private **0.x engineering package**. Real model weights and training caches stay outside Git. The included tiny checkpoints are synthetic numerical/parity fixtures, not useful domain models.

## Shared documentation and findings

Start with the [documentation index](docs/README.md). The [learning log](docs/learning-log.md)
collects measured results, unsuccessful approaches, integration lessons and open
questions from Hive and DelayLama. Its evidence links distinguish implementation
checks from model quality. Use the [experiment template](docs/experiments/TEMPLATE.md)
to record new investigations and add their findings to the log.

The latest [L125 finding](docs/learning-log.md#l125--separate-predicate-witnesses-from-proposed-graph-meaning-2026-10-05)
separates exact-source grounding from predicate/direction/modality and occurrence
roles, retaining failed trials and positive controls for future LayaGo evaluation.

## Build and use

```sh
make build
make check
make race
```

After configuring Git access to this private repository:

```sh
go env -w GOPRIVATE=github.com/UnicoLab/*
go get github.com/UnicoLab/ul_layago@v0.2.0
```

Existing `GOPRIVATE` entries should be retained when adding this pattern. Import the root package as `layago`. `Open(OpenOptions{Checkpoint: path, Workers: 4})` loads once; `Predict(ctx, Request{State: text, Questions: questions})` returns typed nominations, distributions, token usage and model metadata. Calls serialize safely and queued calls respect cancellation. Lower-level packages remain available for custom pipelines.

`bin/layago` serves one JSON object per input line and one response/error per output line. It loads once, preserves request IDs, keeps stdout machine-readable, and supports `--timeout`, `--workers`, `--kernel`. Example:

```sh
bin/layago --checkpoint /models/laya-multilingual --workers 4 <<'JSONL'
{"id":"example","state":"Payment is due within 30 days.","questions":[{"id":"topic","type":"choice","instructions":"Which clause topic is present?","options":[{"key":"payment"},{"key":"termination"}]}]}
JSONL
```

Checkpoint questions use `choice`, `score` or `yes_no` and explicit instructions/criteria. A trained artifact instead uses its pinned tasks:

```sh
bin/layago --artifact /models/my-head/float32 --base /models/laya-multilingual
# stdin: {"state":"<the exact trained prompt format>","questions":[{"id":"support","task":"r_contract_support"}]}
```

Fixed choices come from the manifest when omitted; incompatible options/instructions are refused. Use `laya.RenderPromptState` for dataset Q/R/A inputs. For R, the simple string format is `question: <claim>\nevidence:\n- source: <quote>`.

Inputs exceeding state, instruction or option budgets fail explicitly. `allow_truncation: true` reports the clipping and forces escalation. Uncalibrated checkpoints always escalate. Calibrated artifact `act` decisions apply only to their calibration distribution; **model nominations never establish source evidence or legal facts**. Responses always report `requires_domain_validation: true`. Checkpoint `model.sha256` hashes its weights file; artifact metadata hashes the manifest, whose pins cover the base and head.

## Train, calibrate, evaluate

Read [the training recipe](docs/training.md) first. The trainer reuses an existing frozen ModernBERT/mmBERT encoder, including the trained encoder inside a public multilingual Laya checkpoint. It fits the lightweight zero-transformer-layer marker scorer and act head; it does **not** retrain the encoder or reproduce upstream RLCD/CLM InfoNCE training.

```sh
python3 scripts/generate_synthetic.py --out .local/demo/data.jsonl --stories 40
bin/laya-train validate --dataset .local/demo/data.jsonl --instance R
bin/laya-train inspect --dataset .local/demo/data.jsonl --instance R \
  --base /models/laya-multilingual --instructions examples/evidence-support.instructions.json
bin/laya-train train --dataset .local/demo/data.jsonl --instance R \
  --base /models/laya-multilingual --cache .local/demo/cache --out .local/demo/model \
  --instructions examples/evidence-support.instructions.json --workers 4
bin/laya-train evaluate --dataset .local/demo/data.jsonl --instance R \
  --base /models/laya-multilingual --cache .local/demo/cache --artifact .local/demo/model/float32
python3 scripts/render_training_report.py --artifact .local/demo/model/float32 --out .local/demo/report.html
```

The fixture demonstrates mechanics, not deployment quality. Existing output directories are never overwritten. Cache shards are atomic, content-addressed and resumable. `--init-artifact` warm-starts a compatible previous head, pins its weights digest and fits fresh calibration; it is not optimizer-resume. Complete public two-transformer-layer heads remain usable through inference, but cannot be silently substituted into the zero-layer trainer.

`--max-duration` bounds each run. `--memory-limit` is a **soft Go GC budget**, not an OS RSS cap; mmap/prepacked weights also consume memory. Cache concurrency, batch size, shard size and head workers have separate bounds. Stderr emits cache progress, 5-second process heartbeats, epoch losses and calibration outcomes. Manifests store training history, dataset/base digests, code identity and split metrics. See [operations and acceleration](docs/training.md#operations-and-acceleration) and the optional supervisor/GPU commands below.

## Compatibility and provenance

The `hive.laya-dataset/v1`, `hive.laya-prompt/v1`, `hive.laya-artifact/v1` and cache contracts retain their names to load existing artifacts. They do not imply Hive is a dependency. [Origin manifest](docs/ORIGIN.json) records digests of the user-supplied source before import migration. [Archived Hive engineering notes](docs/hive-engine-notes.md) preserve historical receipts; those results are not new package or domain-quality benchmarks. [NOTICE](NOTICE) identifies the upstream architecture/reference.

## Optional Apple GPU training and run supervision (0.2)

The Go runtime stays dependency-free. On Apple Silicon, install the optional
Python requirements in a suitable environment (`scripts/requirements-mlx.txt`).
`--backend mlx` accelerates **cached marker scorer training**; encoder caching,
act fitting, calibration and serving remain Go. GPU availability is mandatory;
there is no silent CPU fallback. GPU forward/gradient tests are opt-in:

```sh
LAYAGO_MLX_PYTHON=python3 go test ./train -run TestMLX -count=1
```

Use the external supervisor to retain telemetry when the trainer dies:

```sh
python3 scripts/supervise_training.py --run-dir .local/runs/experiment-1 \
  --rss-limit-mib 6144 --stall-seconds 180 --max-seconds 1800 -- \
  bin/laya-train train --dataset /data/train.jsonl --instance R \
  --base /models/laya-multilingual --cache .local/cache --out .local/artifact-1 \
  --instructions /data/instructions.json --config /data/config.json --workers 4 \
  --backend mlx --mlx-script scripts/mlx_head.py --mlx-memory-limit 1073741824 \
  --mlx-work-dir .local/optimizer-work
```

The supervisor sets `LAYAGO_PROGRESS_FILE`, which the trainer uses for atomic
stage counters and an append-only progress journal. Heartbeats/log output do not
reset the stall deadline. Size the stall deadline above the slowest expected
cache shard or validation stage. `status.json`, bounded stdout/stderr, `telemetry.jsonl`
and `health-report.html` remain in the new run directory. RSS includes descendants;
macOS host pressure comes from `memory_pressure -Q`, Linux from `MemAvailable`.
Sampling can miss brief spikes; this is a sentinel, not an OS-enforced allocation
cap or an assurance that the supervisor itself can survive system-wide OOM.

After an interruption, use a **new** run/output directory with the same cache and
`--mlx-work-dir`. MLX restores completed-epoch parameters, AdamW moments, step,
shuffle schedule, history and best scorer. Data/config/implementation, MLX/Python versions and tensor
hashes must match; memory budgets may change. Atomic checkpoint state pins immutable
weights, so a death between tensor writes and state commit resumes the previous
complete epoch. A file lock prevents concurrent writers and releases on exit.
Encoder cache shards resume independently; Go act/calibration stages rerun.
There are no unbounded automatic retries. Supervisor process-group cleanup is
currently macOS/Linux only.

Set `schedule_epochs` in training config to size cosine decay independently of
`max_epochs`; zero retains the original behavior. See the
[Hive report assessment](docs/hive-report-insights.md) for motivation and
architecture experiments still to qualify.
