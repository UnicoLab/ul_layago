# Data preflight and safe preparation

Use these checks before the expensive encoder pass. They apply [Jeff's family/leak/shortcut ideas](research/jeff-architecture-review.md) to the existing Laya dataset contract; they do not certify labels or model accuracy.

## Check the current data

```sh
python3 scripts/audit_training_data.py data.jsonl \
  --json-out runs/audit-new.json --markdown-out runs/audit-new.md
bin/laya-train readiness --dataset data.jsonl --instance R --config config.json
bin/laya-train inspect --dataset data.jsonl --instance R --base /models/laya \
  --instructions instructions.json --config config.json
```

The audit checks declared story/family and source isolation, exact copies, conflicting labels, template remnants, metadata class mix and bounded long-source near-copies. Shared fixed option descriptions are not source leakage. Default reports omit source text but identifiers may still be sensitive. It skips locked logical inputs/labels after envelope decoding and explicitly marks that locked-source isolation was not examined. Deliberate locked inspection requires its unseal flag and append-only access log.

An optional `--surface-eval dev` or `dev_conf` fits a simple length/shape/option-position diagnostic on training data only, then reports it against the specified development fold and a train-majority floor. Strong surface predictability is a review warning, not semantic success. Inspecting `dev_conf` to change data/model choices consumes that fold's independence; use new confidence data afterward. Near-copy sampling/postings/comparison budgets and omissions are recorded; this is not exhaustive semantic deduplication.

Native `readiness` needs no base, tokenization or model. It reports raw row and unique-story support separately, the configured two-score budget, small-cell warnings and correlation limitations; it never reads source/gold for those calculations. The surrounding JSONL loader still parses/validates the existing dataset schema. `inspect` additionally reports token retention and adds this metadata plan. Neither means the data is certified.

At alpha .05, family delta .10, minimum20 and zero starting errors, two score candidates each receive delta .05. The configured starting count is **176**, not135. Counts above that are only a prerequisite, not a guarantee. For a positive starting-error budget, readiness uses a bounded binary search within available support and marks a lower bound when exact support is unavailable. The actual trained-artifact policy remains unchanged.

Training configuration is now validated before loading a base or building feature caches. The native CLI also rejects identical task/rendered-input/options copied between `dev_cal` and `dev_conf`, normalizing evidence IDs so renaming an evidence record does not hide a copy. This exact guard leaves locked sources out and is separate from the richer source/family audit. Different claims or near copies still require the audit and provenance review. Direct low-level library experiments should call `CalibrationOverlap` and respect that result; the CLI enforces it automatically.

## Prepare a separate candidate

```sh
python3 scripts/prepare_training_data.py --dataset data.jsonl \
  --out runs/prepared-new.jsonl --seed 7 --permutations 2 \
  --protected-access-log runs/protected-source-access.jsonl
bin/laya-train validate --dataset runs/prepared-new.jsonl --instance R
```

The access log is necessary when a baseline contains locked rows: protection checks intentionally inspect their source fingerprints, so record that access before processing. No metric, fitting target or test-tuning result is produced. The preparation API likewise requires an explicit protected-access log; callers remain responsible for their prior reads. Passing a log never permits replaying evaluation rows into training.

Original baseline bytes remain a prefix of a fresh output file. Only training `choice` rows gain deterministic distinct orders; hard labels and every soft-target mass move by option key. Score levels and false/true criteria stay ordered. Evaluation records are untouched. A shuffled task has per-example options: callers **must provide its option catalog at serving**, including the currently fixed-catalog DelayLama client if later adapted. Do not switch the deployed artifact to an augmented model without implementing/verifying that explicit catalog contract.

Optional replay uses `--replay FILE --replay-owner OWNER --replay-fraction FRACTION`. Rows must be training-only, declared owned/adjudicated, and bound to the exact source/options/labels by a receipt. All supplied evaluation/source/story overlaps are refused; entire linked families are selected within the row/fraction budget. Producer provenance is preserved. Receipt structure and hashes cannot prove a person made a correct judgment. A note or candidate-model probability is not an adjudicated replay target.

Files, rows, output bytes, permutation count and comparison work are bounded; output publication is exclusive and atomic. These are not process RSS limits. Use the existing external supervisor for large preparation/training jobs and retain failures rather than overwrite earlier output.

## What the existing pilot revealed

The new audit on the existing 615-row classifier pilot inspects525 nonlocked rows and skips90 locked rows. It finds14 exact source-copy and14 rendered-copy pair findings across development roles, plus2 heuristic near-copy findings. The native guard counts16 affected `dev_conf` rows and refuses training before a nonexistent base/cache could be opened. The two numbers count different things: pair findings versus certification rows in overlapping input groups.

Its genre/topic confidence cells contain60/62 rows from30/31 stories, below176 and not independent row samples. Those findings reinforce the existing no-promotion decision; the earlier artifact already escalates every prediction. Previous train/dev/locked exact-text checks did not inspect the two development halves. Preserve the original dataset, artifacts and reports; create properly source-family-grouped new data rather than silently repair the old evaluation.

[Retained engineering receipts](validation/jeff-training-tools-20261007/), [independent planted controls](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/assets/jeff-data-tool-independent/independent-final-audit.md), [data sources](research/contract-training-data.md), [next experiment curriculum](research/jeff-training-experiments.md).
