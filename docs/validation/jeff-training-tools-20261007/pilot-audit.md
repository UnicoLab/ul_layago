# Training data preflight

Valid rows: 525; invalid rows: 0.
Locked-test access: `sealed_skipped`; skipped: 90.

This is a bounded data diagnostic. It reports no trained-model metrics or certification.

## Findings

- `exact_rendered_state_copy`: 14
- `exact_source_copy`: 14
- `near_source_copy`: 2

## Calibration count readiness

| Task | dev_cal | dev_conf | Configured minimum | Shortfall |
|---|---:|---:|---:|---:|
| `r_contract_agreement_type` | 52 | 60 | 176 | 116 |
| `r_contract_clause_topic` | 66 | 62 | 176 | 114 |

Several paraphrases/translations from one story are correlated; example counts do not establish independent acted items or domain validity.

## Surface-only diagnostic

Fit: train only; evaluation: `dev_conf`. These are shortcut diagnostics, not trained-model metrics.

| Task | Dev rows | Surface accuracy | Train-majority accuracy | Shortcut warning |
|---|---:|---:|---:|---|
| `r_contract_agreement_type` | 60 | 0.250 | 0.167 | False |
| `r_contract_clause_topic` | 62 | 0.065 | 0.129 | False |

Near-copy coverage: `heuristic_bounded_not_exhaustive`; comparisons: 1122. Limits/skips are recorded in JSON.

## Limits

- Default locked rows are envelope-decoded then discarded; never rendered or used for audits/labels/metrics.
- Locked split isolation/duplicates are unexamined while sealed; group isolation covers only explicit id/story_id/group_id/meta family identifiers.
- Source duplication excludes fixed options/catalog/question instructions; complete rendered-state equality remains separately checked.
- Near-copy work is capped and sampled; no claim of exhaustive leakage detection.
- No source text is emitted; identifiers/digests/counts may still be sensitive.
- Preflight does not certify dataset correctness, model quality, domain validation or local routing certainty.
