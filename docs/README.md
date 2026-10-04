# LayaGo documentation

This is the shared reference for LayaGo engineering and model experiments across
projects. Findings describe the particular data, code and hardware tested; they
do not confer accuracy or calibration on another application's tasks.

## Where to start

| Document | Purpose |
| --- | --- |
| [Learning log](learning-log.md) | Cross-project findings, failure modes, evidence and next experiments |
| [Training recipe](training.md) | Data construction, fitting, calibration, evaluation and operations |
| [Hive report assessment](hive-report-insights.md) | Interpretation of the final Hive model comparisons and what transfers |
| [Package usage](../README.md) | Supported APIs, commands, GPU training and supervision |
| [Experiment template](experiments/TEMPLATE.md) | A reusable record for new hypotheses, results and decisions |
| [Archived Hive engine notes](hive-engine-notes.md) | Historical tokenizer/kernel/parity, latency and training details |
| [Origin manifest](ORIGIN.json) | Digests of the original supplied engine |

## Evidence records

| Record | What it establishes |
| --- | --- |
| [DelayLama inventory → Prompt Lab (L96)](validation/delaylama-inventory-prompt-lab-20261004.json) | Source/versioned history handoff, baseline isolation, failed frozen action-separation comparison; no new training |
| [DelayLama action-first and comparison review (L97)](validation/delaylama-action-first-comparison-review-20261004.json) | Separate action calls, failed causal/impact support, inventory recall limits and exact-run human judgments; no new training |
| [DelayLama original inventory (L95)](validation/delaylama-source-inventory-20261004.json) | Separate omission-review workflow, retained actor/date and non-occurrence failures, graph preservation; no new training |
| [DelayLama source action coverage (L94)](validation/delaylama-source-action-coverage-20261004.json) | Two rejected prompt revisions, 32 fresh draws, missed-action/deadline retention and correct quote/wrong premise; no new training |
| [DelayLama prompt evidence scope (L93)](validation/delaylama-prompt-evidence-scope-20261004.json) | Relevant experiment inventory, selected-output binding, retained assistant attribution errors and browser persistence; no new training |
| [DelayLama scalar associations, 2026-10-04](validation/delaylama-source-associations-20261004.json) | Original-premise date/value binding, exact Unicode offsets, frozen wrong-deadline regression, fresh chronology and retained export/recall limits; L90 |
| [DelayLama performer checks, 2026-10-04](validation/delaylama-event-performers-20261004.json) | Per-event attribution protocol, retained affected-relation failure, dated issuance replay, fresh directed-role checks and separate generic date-association error; L89 |
| [DelayLama results first, 2026-10-04](validation/delaylama-results-first-20261004.json) | Contract/event overviews, source preview isolation, actual saved model jobs and retained evidence/graph gaps; L84 |
| [DelayLama event representations, 2026-10-04](validation/delaylama-event-representations-20261004.json) | Whole Unicode identifiers, quote versus source-job representations, retained occurrence/role/schema gaps and failed full race sweep; L85 |
| [DelayLama startup and per-run profiles, 2026-10-04](validation/delaylama-startup-schema-20261004.json) | Captured task configuration vs requested intent, full registry, browser selection without submission and retained quality gates; L86 |
| [DelayLama event date roles, 2026-10-04](validation/delaylama-event-date-roles-20261004.json) | Required output protocol, structured fallback repair, frozen occurrence/date-role comparison, retained omitted-role draw and browser export observation limit; L88 |
| [DelayLama dated dispatch and UX, 2026-10-04](validation/delaylama-dated-dispatch-20261004.json) | Frozen registration comparison, fresh semantic failures, targeted performer recovery and readable chronology; L87 |
| [DelayLama review queue, 2026-10-04](validation/delaylama-review-queue-20261004.json) | Unsearched event/role work, scoped source-to-prompt workflow and retained model variability; L80 |
| [DelayLama observed actions, 2026-10-04](validation/delaylama-observed-actions-20261004.json) | Frozen-output occurrence registration, fresh chronology/source evidence and retained performer omissions; L79 |
| [DelayLama typed parties, 2026-10-04](validation/delaylama-domain-actors-20261004.json) | Fixed-output endpoint validation, typed role GUI, optional prompt comparisons and retained Planning occurrence failure; L78 |
| [DelayLama repeated occurrences, 2026-10-04](validation/delaylama-repeated-occurrences-20261004.json) | Title-only merge loss, SQLite/filedb repair, fresh workflow boundaries and original occurrence labels; L92 |
| [DelayLama chronology exports, 2026-10-04](validation/delaylama-chronology-export-20261004.json) | Human review/quote metadata, observed copies, unconfirmed downloads and replay boundaries; L91 |
| [DelayLama profile occurrences, 2026-10-04](validation/delaylama-profile-occurrence-20261004.json) | Fixed-output event registration, real source rechecks and prompt-history transfer, with retained performer omissions; L77 |
| [DelayLama captured job schema, 2026-10-04](validation/delaylama-analysis-schema-20261004.json) | Worker configuration isolation, actual-model notice typing/role failures and downstream evaluation priorities; L76 |
| [DelayLama event vocabulary, 2026-10-04](validation/delaylama-event-vocabulary-20261004.json) | Observed schema categories, bounded run metadata, shared-setting limits and contract-quality priorities; L75 |
| [DelayLama exact evidence resume, 2026-10-04](validation/delaylama-review-resume-20261004.json) | Complete-context navigation, citation-list binding, async races and actual saved-analysis browser checks; L74 |
| [DelayLama scoped navigation, 2026-10-04](validation/delaylama-review-navigation-20261004.json) | Contract/agent/layer isolation, graph reload and preservation checks; L73 |
| [DelayLama role meaning, 2026-10-04](validation/delaylama-role-meaning-20261004.json) | Synthetic directed role sets, incoming-cause schema, corrected proposal labels, preserved duplicate-event failure and actual GUI coverage; L72 |
| [Pretrained contract probe, 2026-10-01](validation/laya-pretrained-probe-20261001.json) | Twelve synthetic French clause/claim decisions from the full pretrained model |
| [Contract pilot, 2026-10-02](validation/contract-pilot-20261002.json) | Dataset splits, model pins, calibration shortfall, held-out/locked results and baseline comparison |
| [Acceleration comparison, 2026-10-02](validation/acceleration-20261002.json) | Paired warm-cache CPU/GPU timings, quality, resume and supervisor failure tests |
| [Final 0.2.0 export, 2026-10-02](validation/release-0.2.0-20261002.json) | Clean release-code GPU export, actual devices, additional run timing and challenge errors |
| [DelayLama live evidence recovery, 2026-10-02](validation/delaylama-evidence-recovery-20261002.json) | Eight targeted LLM checks and five agreements among seven available Laya second opinions; one synthetic contract, not certification |

The pretrained contract probe and contract pilot are preserved copies of
DelayLama's checked-in synthetic evaluation records. Original paths, source revision and SHA-256 digests appear in
[their provenance index](validation/contract-record-sources.json). They are
historical evidence, not newly repeated experiments. The Hive report's identity
is pinned in the acceleration receipt; its interpretation is attributed in the
assessment. Model weights, caches, private contracts and the full external report
are not required to read these findings.

## Keeping this reference useful

Record observations separately from explanations and proposed fixes. Link a
dated receipt and code/data/model identity before adding a quantitative claim.
Retain failed, interrupted and inconclusive runs. Mark experiments as proposed,
running, measured, rejected or adopted; an adopted engineering tool is not an
accepted model-quality result. Update a superseded conclusion with a dated link
instead of erasing the old evidence. Add new experiments to this index and cite
their finding IDs in the learning log. Keep confidential source text and model
weights outside Git; use reviewed, shareable aggregates where necessary.
