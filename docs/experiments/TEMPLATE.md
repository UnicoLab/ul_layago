# Experiment: <name>

Status: proposed | running | measured | rejected | adopted. Replace placeholders
before recording results; use `not measured` for missing values, never invented
zeroes. Link this record from the documentation index and relevant learning IDs.

## Question and predeclared decision

- Date/owner/project:
- Learning IDs and hypothesis:
- Intended application/task/label policy:
- Baseline and the one factor changed:
- Success criteria, unacceptable regressions and resource limits:
- Selection data and independent certification/final-test plan:

## Reproducibility and scope

- Code revision, clean/dirty state and patch identity if dirty:
- Dataset/generator/model/tokenizer/config hashes and model/head architecture:
- Story/document/template/entity split rules; languages and label counts:
- Rendering, candidate order, quote windows, token budgets and clipping policy:
- Training objective, soft labels, schedule, seed(s), calibration and threshold rules:
- Backend/precision, hardware/OS/runtime, workers/concurrency and other host load:
- Commands, cache state, immutable output/workspace identity and supervisor limits:

## Observations

| Measurement | Baseline | Candidate | Corpus/slice, count and uncertainty |
| --- | --- | --- | --- |
| Accuracy / macro-F1 / confusion | not measured | not measured | |
| False-support rate / contradiction and missing-information recall | not measured | not measured | |
| Exact span precision/recall and role/value association | not measured | not measured | |
| NLL / ECE / Brier / coverage / selective risk | not measured | not measured | |
| Cold/warm served p50/p95 and full document latency | not measured | not measured | |
| Cache / scorer / total training time | not measured | not measured | |
| RSS / device peak memory / tokens / concurrency | not measured | not measured | |

Record per-language/task/domain slices, seeds and paired error changes, not only
aggregate scores. Identify grouped uncertainty and repeated measurements.
Separate synthetic from real/adjudicated evidence. Attach reviewable, shareable
receipts with hashes; keep confidential source text, caches and weights outside Git.

## Verification and failures

- Token/rendering, forward/gradient, export/served and cached/live parity results:
- Clipping, queue/timeout, crash, checkpoint recovery and resource-limit results:
- Failed/incomplete runs, exit codes/signals, logs and actual progress:
- Locked-test accesses and selection boundary; no replay of evaluation stories:
- Missing measurements, deviations, contamination/confounds and invalidated runs:

## Interpretation and follow-up

Separate observed changes from their proposed cause. Explain whether the
success criteria were met, why the candidate is adopted/rejected/inconclusive,
and the limits of that decision. An adopted training/backend tool does not
automatically qualify its model for application acceptance. Record the next
isolated experiment and update dated learning-log conclusions without erasing
older results. Material new conclusions need new independent evidence.
