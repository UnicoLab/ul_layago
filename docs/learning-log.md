# Shared learning log

Updated 2026-10-02. Scope: the supplied Hive engine, the W10 round-3 Hive report,
the DelayLama contract pilot, and LayaGo 0.2.0. This log is an interpretation of
the linked evidence. Proposed explanations and experiments are explicitly
separate from observed results. The [documentation index](README.md) identifies
all preserved receipts and their provenance.

## Measured starting point

| Evaluation | Observation | Evidence and limits |
| --- | --- | --- |
| Full pretrained multilingual Laya, French contracts | 9/12 correct; zero false-support predictions | [Initial probe](validation/laya-pretrained-probe-20261001.json); four examples per class, synthetic only |
| Specialized Go contract head | Domain 64/72; Spanish 60/72; locked synthetic test 67/72; French challenge 9/12 | [Pilot receipt](validation/contract-pilot-20261002.json); small template families, no overall challenge gain |
| Contract acceptance calibration | 96 dev_conf examples vs 176 required at this run's settings; zero certified act coverage | Same receipt; 176 is configuration-specific, not a universal sample requirement |
| GPU contract head | Domain 64/72; Spanish 60/72; challenge 9/12, zero false supports | [Acceleration receipt](validation/acceleration-20261002.json), confirmed by [final export](validation/release-0.2.0-20261002.json); no new locked-test access for acceleration tuning |
| Warm-cache scorer epoch loops | Go 2.48 s vs MLX 0.31 s | One [paired comparison](validation/acceleration-20261002.json); excludes setup, transfer and checkpoint costs |
| Supervised total training time | Paired Go 4.91 s vs MLX 3.54 s; other GPU runs 10.23 s and 10.14 s | Paired receipt and final export; shared-machine variation prevents a general whole-training speedup claim |
| Hive trained small decoder | J1/LFM2-1.2B-LoRA locked EN 92.66%, FR 91.04%; passed its registered Hive gate | [Report assessment](hive-report-insights.md); synthetic Hive tasks, not contract extraction |
| Hive frozen marker / contrastive arms | Laya-Q locked EN/FR about 68.1%/67.2%; lexical floor 84.2%/81.6%; best contrastive arm 77.5%/77.3% | Same assessment; architecture names and retrieval speed did not predict task quality |

## Findings and implications

### L01 — Numerical parity and domain accuracy are different evidence

**Observed:** tokenizer, kernels, artifacts and MLX gradients have extensive
parity checks. The real contract challenge still has errors. Hive's real int8
encoder also failed its probability/choice-flip bound despite tiny fixtures
passing ([archived parity evidence](hive-engine-notes.md#parity-evidence)).
**Implication:** test token/layout identity, forward/gradient/export parity and
domain performance separately. Quantizing the head and quantizing the encoder
are different experiments; neither inherits the other's calibration or result.

### L02 — Reusing an encoder is different from reusing the complete head

**Observed:** the contract pilot reuses the pretrained multilingual encoder but
fits a separate zero-transformer-layer scorer. The complete public checkpoint
has two head transformer layers and remains a distinct serving baseline.
**Implication:** record which weights are retained and which are initialized.
Do not call dropping the transformer head a compatible full-head warm start.
LayaGo currently caches markers and a pooled state, not all token activations.
Full cross-token fine-tuning needs a new cache/training path and memory budget.

### L03 — Equal aggregate accuracy can hide useful fixes and regressions

**Observed:** the specialized head corrected the amended-date examples while
rejecting supported price/notice wording and predicting contradiction for a
claim about actual reception when only scheduled completion was supplied.
Both complete and specialized models scored 9/12. The final GPU challenge has
the same three errors, identified in its [receipt](validation/release-0.2.0-20261002.json).
**Implication:** inspect confusion matrices and error families, including
missing-information versus contradiction. Zero false supports in eight negative
challenge examples is an observation, not a reliability guarantee. A candidate
must improve document-level outcomes and important slices, not just total score.

### L04 — Template growth can saturate without adding reading capacity

**Observed:** the first contract generator produced duplicate rendered examples
and was rejected by preparation; distinct references fixed that check. Hive
also found shared templates/entities inflating development performance and a
nearly flat marker-head learning curve.
**Hypothesis:** cross-reference errors need better contextual interaction and
better examples, rather than more variants of the same template.
**Next test:** disjoint template/entity pools, near counterfactuals, amendments,
negation, amount/unit/role association and independently adjudicated real
contracts. Distinct references alone do not prove semantic independence.
Group translations, paraphrases and quote windows under the same story before
splitting; freeze the existing challenge and locked test rather than training
on their known mistakes.

### L05 — Calibration has a scope and a finite sample requirement

**Observed:** the contract pilot cannot certify act coverage. Hive showed
yes/no class bias and weaker French slices; temperature scaling alone did not
resolve these errors. Paraphrases in a story are correlated.
**Implication:** retain escalation, report task/language/label slices, and use
story-level uncertainty. A proposed additive class-bias calibrator is a new
experiment requiring independent certification. Use separate selection data or
nested grouped folds; repeated architecture searches on dev_conf consume its
independence. One selected candidate's locked result must not become training
data for the next candidate.

### L06 — Rendering, clipping and candidate order are part of the model

**Observed:** Hive changed candidate ordering, count and length in the same
shortlist experiment, making their effects inseparable. Cached/live contrastive
decisions differed in 12/315 examples in its parity probe.
**Implication:** pin rendering, tokenizer, options, order and retention with the
artifact. Inspect training retention; low-level preparation still preserves
legacy clipping behavior. Serving rejects oversized input by default and
explicit clipping forces escalation. Test reorder-only and quote-window changes
independently; never truncate a required option or silently change the serving
layout. Fix offline/live discrepancies before relying on a ranker.

### L07 — Similarity, source matching and semantic support are separate checks

**Observed:** contrastive models rank supplied candidates. Laya emits typed
decisions, not arbitrary extracted strings. GLiNER's Hive results are loading/
latency smoke tests on two examples; they do not establish contract extraction
precision/recall or evaluate GLiNER2.5-Extract here.
**Implication:** use retrieval to nominate clauses and an optional extractor to
nominate spans. Validate exact original-source offsets, then judge the entire
claim in context. A matching quote can support only one part of a compound
claim. Preserve role, currency/unit, dates, source/amendment identity and tested
date arithmetic. Classifier confidence and retrieval scores are nominations.

### L08 — Feedback must preserve meaning and provenance

**Observed:** DelayLama's optional Laya review follows LLM evidence recovery and
exact quote validation. It supplies a second opinion without rewriting extracted
facts, graph entries, offsets or acceptance. Review notes and corrections can
inform prompt iteration, but are not automatically adjudicated gold labels.
**Implication:** separate user notes, reviewed evidence, corrected labels and
prompt/model versions. Retraining should use reviewed corrections with replay
of earlier accepted training cases and frozen evaluation stories. Keep the UI
human-readable: show the nomination, contextual citation and exact highlight;
put embeddings, distributions and diagnostics in explicit developer views.

### L09 — Small trained readers deserve comparison; model size is insufficient

**Observed:** Hive's trained small-decoder arm passed its registered gate while
a larger zero-shot arm failed, with substantial yes/no bias. The contrastive
arm was fast but did not pass its quality gate.
**Next test:** compare a full cross-token Laya head and a small LoRA typed reader
on the same contract corpus. Start a decoder experiment behind a resident
worker; qualify quality, memory, calibrated abstention and cold/warm latency
before a native Go port. An oracle union of two models' correct answers is not
a deployable selector. Require measured complementary errors and a separately
evaluated selection policy before combining models.

### L10 — Early stopping and learning-rate decay need independent controls

**Observed:** Hive often stopped at 8–12 epochs of a 30-epoch schedule while
the learning rate remained high. LayaGo 0.2.0 adds `schedule_epochs` independently
of `max_epochs`; warmup uses that horizon and later steps stay at the floor.
**Status:** implemented experiment control. A contract quality improvement from
changing the schedule has not been measured. Compare it as one isolated arm.

### L11 — GPU acceleration must name the accelerated stage

**Observed:** MLX speeds cached-marker scorer fitting; Go still performs frozen
encoder/cache work, act fitting, calibration, export and serving. GPU float32
moments differ from Go's float64 masters. Exported development NLL is checked
again in Go, and artifact metadata names the actual devices and arithmetic.
**Implication:** report scorer, cache, total and serving times separately, with
cold/warm state, token counts, hardware, load and repetitions. The 82.71 s cache
statistic in the final artifact is historical cache-build metadata, not time
spent by that warm-cache run. A later 10.14 s GPU total and different inference
medians confirm that these runs cannot establish a serving speedup.

### L12 — Liveness, progress, memory and recoverability are distinct

**Observed:** the supervisor tests cover progress stalls despite heartbeats,
SIGKILL, RSS/host-pressure limits, monitor failure, deadlines, bounded logs and
owned-descendant cleanup. A SIGKILL alone does not identify an OOM. Completed
MLX epochs resume parameters, AdamW moments and best weights; an interrupted
state commit resumes the previous complete epoch.
**Implication:** retain durable status/telemetry and failure outcomes. Only
signal the owned process group. RSS sampling is not an OS allocation cap and a
system OOM can kill the supervisor. Code/data/runtime/hash mismatches refuse
resume; use new run/output directories and the same compatible workspace.
Only the MLX scorer resumes optimizer state; Go act/calibration rerun. Cache
resume and weight warm starts are different mechanisms. Avoid unlimited retries.

### L13 — Historical reports need explicit version and release boundaries

**Observed:** earlier Hive paragraphs describe pending/killed runs that the final
round-3 tables resolve. Archived engine paths and commands refer to Hive's
original layout. A workspace-guard compile error was pushed during acceleration
work, then immediately corrected before the successful release checks.
**Implication:** attribute source revision/section and retain failure history.
Build and inspect required check results before committing/tagging; do not let a
failed check fall through to publication. Verify the private module from a clean
external consumer as well as the checkout. Source-report approval/queue rules
belong to that report's context and are not package operating instructions.

### L14 — Exact source spans do not establish entailment

**Observed:** DelayLama's [live recovery receipt](validation/delaylama-evidence-recovery-20261002.json)
records eight targeted checks on one synthetic French contract using English
claim labels. The LLM recovery verdicts passed all eight checks with exact source
offsets and mandatory human review. The deployed float32 contract pilot agreed
with five of seven available second opinions. It classified the correct contract
price as insufficient and a retention-release clause as contradicting a claimed
reservation-removal deadline, where the expected policy is insufficient. One
missing-role-evidence case had no candidate quote and no Laya verdict.

**Implication:** keep the model advisory. This single integration fixture is not
independent certification and may overlap task development. Add independently
split hard negatives for retention release versus defect-removal deadlines,
contractor identity versus assigned defect duties, original versus amended dates,
and near-miss amounts/currencies. Measure false contradiction separately from
false support and insufficient recall. Log missing-quote coverage separately from
classification agreement; do not count an unavailable model as correct.

### L15 — Review feedback needs provenance before becoming training data

**Observed:** live prompt iteration produced an incompatible recommendation that
split an original and amended deadline across two application aliases. Providing
the actual target schema, field meanings and explicit reviewer correction yielded
a compatible additive refinement. Earlier incorrect proposals remained in the
session history. DelayLama now retires proposals when their review context changes
and requires new approval when evidence re-search replaces an accepted candidate.

**Implication:** preserve original documents, exact prompts, output snapshots,
review revisions, candidate spans and model identities. A matching span, generated
summary, accepted historical candidate or assistant suggestion is not automatically
a gold semantic label. Adjudicate feedback and bind approval to the exact candidate
before exporting training examples. Keep evaluation contracts/templates separate
from prompt-development and training feedback. Recheck historical suggestions
against the current output contract rather than treating history as authority.

### L16 — Preserve date precision and separate assessment targets

**Observed:** DelayLama's [Lumen regression receipt](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-02-claim-conclusions.md)
uses a second synthetic contract and deliberately seeded assessment outputs to
test merging and review. A granted time extension and an excluded cost assessment
for the same event both survive; neither conclusion replaces the other. Complete
English dates start the expected clock, while a month-only trigger stays unknown.
These seeded cases do not measure Laya or LLM extraction accuracy.

Actual Chronograph model analysis of the original contract stored `2026-06-01`
and marked it exact even though its own `date_text` and quotation say only
`June 2026`. DelayLama now flags that conflict and displays no established exact
day. The original graph record remains erroneous and needs upstream correction;
valid graph topology and a matching quote did not validate its date semantics.

**Implication:** synthetic examples and feedback exports should retain date
precision, original spans and the assessment target. Time, cost, notification and
party responsibility are distinct questions, even for the same event. Different
answers are not automatically contradictory. Never use generated graph timestamps,
model-declared precision or a last-write merged conclusion as gold labels. Add
hard negatives for month-only dates, plausible invented first days, sent notices
versus legally timely notices, and time entitlement versus cost entitlement.
Evaluate each target and source association independently; treat ambiguous or
missing evidence as insufficient and retain human review. This is a proposed
training/evaluation policy, not an implemented architecture or measured quality gain.

### L17 — Verify original-source semantics across the full extraction path

**Observed:** DelayLama's [original-source follow-up receipt](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-02-source-date-precision.md)
uses actual graph and four downstream model calls on the synthetic Lumen contract,
followed by event/claim reruns. The upstream date-storage fix preserves `June 2026`
as a month with period bounds and no exact occurrence. Scoped topology and
mid-month/out-of-period timeline checks pass; this does not migrate historical
graph data or establish the semantics of every model-inferred relation.

Two independent integration bugs mattered more than model choice: claim viability
was missing the original source, and output-format cleanup discarded later
behavioral instructions. After both fixes, reruns separated E1 time entitlement
from excluded compensation, preserved E2's one-day observed interruption and kept
undocumented notice status unknown. Passing quote matching still did not establish
actual costs: the model returned zero for both events without sufficient support.
The application now retains those raw values with a field-level source-review
warning across claim, analysis, review and timeline views. Nineteen focused
integration checks pass while those two semantic review issues remain explicit.

**Implication:** capture the effective composed prompt and original source, not
only a stored prompt version or derived graph summary, when creating training
examples. Separate observed duration, extension entitlement, incurred cost,
compensation entitlement and notice-record presence into distinct targets. Add
hard negatives for an excluded payment being misread as zero spend, an excluded
extension being misread as zero delay, and missing records being misread as proven
absence. Preserve precision and review warnings through downstream views and
feedback exports. A period sorting anchor, a matching quote and a passing topology
check are not semantic gold labels. These are data/evaluation requirements and
application findings, not evidence of improved Laya weights, architecture, MLX
acceleration or independent contract accuracy.

### L18 — A second opinion needs field-specific certification, not confidence

**Observed:** DelayLama's [field-review receipt](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-02-field-review.md)
follows an unsupported zero actual-cost finding through live evidence recovery
and two assistant/draft iterations. The LLM repeatedly called excluded
compensation support for zero actual cost. A conservative source guard now keeps
that case insufficient and preserves the raw model verdict; explicit incurred-zero
evidence and excluded compensation entitlement still pass their separate checks.

The optional Laya artifact `559cbfbd…` returned `contradicts` on all three focused
cost-field rechecks: missing actual cost (expected insufficient), explicitly zero
incurred cost (expected supports), and no compensation entitlement (expected
supports). The two supported cases received contradiction probabilities above
0.99. In a larger whole-event finding the head instead nominated the quote as
supporting despite the unsupported actual-cost field. This is a concrete failure
on these inputs, not an estimate of general accuracy. The application exposes
disagreement and keeps the head advisory. No retraining occurred in this receipt.

**Implication:** test the exact deployed claim serialization and source association
before promotion. Compare terse field/value claims with explicit natural-language
assertions on identical fixed evidence; isolate field-specific questions from
whole-event bundles. Include negative/zero/unknown quantities and distinguish
unsupported from explicitly contradicted claims. Align train, evaluation and
production formatting, but do not explain away a failure as formatting without
a controlled comparison. Measure false support and false contradiction separately;
high confidence is not certification. New adjudicated examples should distinguish
incurred cost from compensation, preserve unknown facts, and independently retain
exact citations for notices, duration and entitlements.

The real prompt revisions improved costs and source quotations, but an intermediate
draft introduced new citation errors. Save every draft and evaluate the full
output contract again after each revision. A better value or matching final span
does not establish all fields in the finding. Model suggestions and accepted
historical candidates remain review evidence, not automatic training gold.

### L19 — Completeness needs separate targets and independently judged evidence

**Observed:** DelayLama's [contract coverage audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-02-contract-coverage.md)
used eight real agents on an invented French contract. A single claim pass omitted
cost conclusions; improving one event with a prompt-only revision lost another
cost conclusion. Separate time/cost passes retained all four discussed targets,
and 22 targeted criteria passed. Partial failure remains visible instead of
turning retained rows into an apparently complete analysis.

Unknown formal-notice requirements were biased by a false schema default. An
amendment's completion change was copied into risk rewrite by normalization and
merging. The interface also hid all but the first supporting quote. These are
schema, application and presentation errors, not necessarily head quality failures.
Their fixes preserve null facts, distinct amendment effects and all evidence.

**Implication:** model-training examples should separate event occurrence, observed
delay, extension entitlement, incurred cost, requested/awarded compensation, future
rights, notice record and notice applicability. Include hard negatives where an
unrequested payment is mistaken for exclusion or zero actual cost. Record every
target, even negative conclusions, and score omissions separately from false
support. Preserve source versions, field-specific citations, IDs and explicit
scope; failed passes are incomplete examples, not gold negatives.

All 34 quotes matched source spans, but E1's notice-compliance assertion still
lacked the notice passage in that assessment's own citations. All 36 graph edges
were proximity/keyword suggestions; scoped endpoints do not prove semantic
relations. Name lookup proves only literal name presence. Evaluate clause-to-event
and field-to-span associations independently, including missing quote components
and amendment chains. Use independent held-out contracts after prompt or model
selection; this tuned short fixture is not a production recall estimate. No Laya
training or weight improvement occurred in this audit, and the optional verifier
remains advisory in light of the cost-field failures in L18.

## Open experiment priorities

These are proposed investigations, not implemented quality improvements.

| Priority | Investigation | Required comparison |
| --- | --- | --- |
| 1 | Better contract data and label policy | Independent contract/story/template splits; false support, contradiction/missing recall, role/span association and per-language performance |
| 2 | Cross-token head vs small LoRA reader | Existing full checkpoint and marker head baselines; identical task/input policy, model-specific calibration, memory and served latency |
| 3 | Contrastive or lexical clause shortlist | Recall of necessary clauses, amendment retention, cached/live parity, full pipeline quality and latency |
| 4 | GLiNER span candidates | Exact span precision/recall, role/value associations, multilingual PDF artifacts and end-to-end benefit |
| 5 | Schedule/class-bias calibration controls | Isolated registered arms; fresh independent certification after candidate selection |
| 6 | Larger operational benchmarks | Repeated cold/warm runs, varied lengths/concurrency, resource peaks, failure recovery and deployment hardware |

Full-head RLCD, CLM InfoNCE, GPU encoder training, Go optimizer-resume, stronger
OS isolation and a native decoder port remain separate engineering work. Use the
[experiment template](experiments/TEMPLATE.md) for each arm, report unsuccessful
results, and update the relevant finding with its dated evidence.
