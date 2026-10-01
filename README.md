# UL LayaGo

A reusable, pure-Go Laya runtime and frozen-encoder head trainer extracted from UnicoLab's Hive engine. Module: `github.com/UnicoLab/ul_layago`. Go **1.27**, standard library only, cgo-free inference/training. Native ARM64 NEON and AMD64 AVX2 kernels have a portable `purego` fallback.

This is a private **0.x engineering package**. Real model weights and training caches stay outside Git. The included tiny checkpoints are synthetic numerical/parity fixtures, not useful domain models.

## Build and use

```sh
make build
make check
make race
```

After configuring Git access to this private repository:

```sh
go env -w GOPRIVATE=github.com/UnicoLab/*
go get github.com/UnicoLab/ul_layago@v0.1.0
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

`--max-duration` bounds each run. `--memory-limit` is a **soft Go GC budget**, not an OS RSS cap; mmap/prepacked weights also consume memory. Cache concurrency, batch size, shard size and head workers have separate bounds. Stderr emits cache progress, 5-second process heartbeats, epoch losses and calibration outcomes. Manifests store training history, dataset/base digests, code identity and split metrics. See [monitoring follow-up](docs/training.md#operations-and-acceleration) for the stronger supervisor/GPU path still to implement.

## Compatibility and provenance

The `hive.laya-dataset/v1`, `hive.laya-prompt/v1`, `hive.laya-artifact/v1` and cache contracts retain their names to load existing artifacts. They do not imply Hive is a dependency. [Origin manifest](docs/ORIGIN.json) records digests of the user-supplied source before import migration. [Archived Hive engineering notes](docs/hive-engine-notes.md) preserve historical receipts; those results are not new package or domain-quality benchmarks. [NOTICE](NOTICE) identifies the upstream architecture/reference.
