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

### L20 — Association judgments and malformed evidence need separate failure paths

**Observed:** DelayLama's [field coverage follow-up](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-02-field-coverage.md)
rechecked each material field separately on invented source documents. Initial
responses lost a valid field contradiction in the overall summary and invented
an extension quote. Field candidates now preserve original asserted values and
independent source quotes; omitted/duplicate checks and nonmatching spans cannot
establish support. Display titles and assessment scope labels are excluded from
source-assertion coverage. A model can organize a finding without those labels
being literal source text.

One bounded corrective pass addresses malformed evidence, retains the earlier
checked and raw unverified response, and stops after another invalid response or
provider failure. Genuine semantic insufficiency does not trigger repeated
persuasion. Four final focused association tests and three cost regressions passed;
the wrong notice-date case needed the corrective pass. This is application behavior
on these inputs, not independent production accuracy or a Laya training gain.

**Implication:** distinguish model meaning errors, invalid span generation,
missing fields, application aggregation errors and display metadata in evaluation
and failure logs. Preserve each original claim/value, its candidate source spans,
all attempts, and whether correction changed a verdict. Score first-pass quality
and final quality separately with extra-call latency; avoid treating repeated
attempts as independent examples. Explicit source association is needed for
notice vs extension and incurred-cost vs entitlement distinctions.

Composite fields such as timely-notice compliance still need separately judged
rule, event, date, form and role premises. A matching field quotation is not a
certificate for every sub-assertion. The optional Laya head still disagreed on
supported and missing-evidence cases in this follow-up; it remains advisory.
Keep rejected raw model proposals out of training gold unless adjudicated, and
retain malformed evidence as a distinct negative/failure label rather than
turning it into a semantic contradiction automatically.

### L21 — Analysis navigation and semantic relation evidence are different layers

**Observed:** DelayLama's [saved analysis map](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-02-analysis-map.md)
projects 25 saved Aster findings into a readable catalog. Its 30 links comprise
25 navigation links, four explicit assessment/event references and one explicit
amendment/event reference. The original source graph separately contains 70
facts and 36 proximity/keyword suggestions. Its suggestions remain hidden by
default. No new extraction, semantic graph storage or Laya training occurred.

The exact finding opens its evidence review with all quotations and prior field
checks intact. Pending reviewer concerns and model disagreements stay visible in
the map, including disagreements on fields that all have candidate supporting
quotes. Duplicate event IDs, inferred matches, missing endpoints and explicitly
foreign project scope do not produce event links.

**Implication:** use graph structure to organize candidate analysis while retaining
the provenance and uncertainty of each link. An explicit reference can navigate
to an event without proving cause or entitlement. Do not label navigation edges,
co-occurrence suggestions or accepted UI decisions as positive relation-training
gold. Build relation examples from independently adjudicated endpoint roles,
source spans and the specific predicate's premises; preserve source versions,
target scope and amendment ordering.

Compound conclusions need several atomic associations: timely notice depends on
the trigger, date, rule, recipient and delivery method. Clause retrieval evaluation
must score retention of all necessary passages, not just the easiest matching
quote. Laya shortlist/field-verifier experiments should report association errors,
false support, necessary-clause recall and end-to-end latency on independent
contract families. Model disagreement remains a review signal; this interface
change is not evidence that the existing advisory contract head improved.

### L22 — Saved output, matching spans and supported conclusions need separate checks

**Observed:** DelayLama's [analysis availability audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-analysis-coverage.md)
ran eight real agents on the invented Aster source and passed 24 targeted
regression checks. An explicitly seeded failure simulation separately passed 11
application checks. Exact agent selection and immutable run timestamps prevent
a base-agent output or touched older file from becoming the latest public-agent
result. Unreadable latest runs remain visible rather than silently restoring an
older success. Atomic writes avoid publishing incomplete results.

Actual analysis-input hashes detect changed input even when a historical quote
still matches. Legacy input versions remain unknown. Availability/provenance
metadata is excluded from review identity; finding facts, source and scope are
retained. This preserved prior field rechecks and model disagreements. An initial
false partial-analysis warning came from the validator assuming a parties list;
the canonical identity schema uses a map. Derive validators from real schemas.

The live outputs still included two uncited identity findings. An E1 cost action
claimed closure from unchanged price and no payment requested or awarded; its
quotes did not explicitly establish closure or waiver. E2 used “Non compensable”
as cost impact, conflating entitlement with quantified incurred cost. Both were
flagged, and the E2 correction reached its owning-agent prompt workspace. No
default prompt was activated, no raw saved output was rewritten, and no Laya
training or weight improvement occurred. The verifier remains advisory.

**Implication:** keep availability, input freshness, span match, field association
and semantic entailment as distinct states and evaluation metrics. A passed
literal-zero cost check does not rule out other cost/entitlement dimension errors.
Support every closure, waiver or notice-compliance premise independently. Treat
uncited absence assertions separately from positively extracted source facts.
Do not turn saved outputs, matching quotes or browser flags directly into gold.
Adjudicate field/value/span examples and preserve disagreement and unknown labels.

For retrieval or verifier training, include unchanged-price passages as close
negatives for waiver/closure and compensation exclusions as close negatives for
incurred-cost amount. Split by independent contract/template families and source
versions; score omitted necessary clauses, wrong field associations, false support
and full-pipeline latency. Hashes describe supplied input under its existing
limits, not completeness of the original corpus. The short tuned fixture and
seeded failure tests are regression evidence, not production model certification.

### L23 — A supported entitlement sentence can still be wrong for a cost field

**Observed:** DelayLama's [financial review follow-up](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-financial-dimensions.md)
adds narrow financial-dimension and closure-consistency warnings to current and
historical outputs. Saved model values and files are preserved; path-specific
delivery annotations carry warnings into evidence review and prompt feedback
without discarding earlier notes. An import-selection bug also surfaced: matching
only an analysis label reused an older prompt-lab snapshot. Exact facts/source
comparison now creates a new snapshot when either changes, while identical
facts/source refresh their current warnings.

One real GUI recheck of the saved E2 weather event returned 12 field checks.
The LLM proposed support for **Non compensable** as cost impact using the matching
contractual compensation exclusion. A deterministic dimension warning kept that
field insufficient; the other 11 checks remained model candidate support, not
independent gold. Laya nominated support for the finding, and the disagreement
remained visible. Reload preserved the recheck and its exact feedback reached
the owning agent. No model training, improved-weight result or prompt activation
is claimed. The 355 Python/167 frontend tests and eight runtime API checks validate
application behavior, not held-out contract accuracy.

**Implication:** field support needs the requested dimension and association,
not just semantic compatibility with a statement somewhere in the document.
Label actual incurred cost, compensation eligibility, payment requested, payment
awarded and settlement/waiver separately. Include a genuine exclusion quote as
a hard negative for an incurred-cost amount. A source-matched model “supports”
verdict can be wrong even when it is well-formed and uses an authentic passage.
Laya must be tested on such close dimension negatives before governing final
conclusions. Preserve its disagreements rather than treating repeated model
agreement as new evidence.

The wording warnings are finite triage checks, not a semantic verifier. Other
phrasings, role assignments, citation references and composite conclusions still
need independent adjudication. Retain original assertions, source spans, proposed
and checked verdicts, input version, target dimension and all review attempts in
future training/evaluation records. UI labels or compact fingerprints must not
serve as snapshot identity for selecting those records.

### L24 — Passing selected checks can still hide a prompt regression

**Observed:** DelayLama's [multi-case benchmark](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-prompt-benchmarks.md)
records exact intent overrides, original sources, field expectations, model
outputs, failures and review notes. Four English/French event cases were run
with two drafts; both met 14/14 selected checks, with 8/8 and 6/6 source quote
matches respectively. Two contract-identity cases met 7/7 checks, including a
revised completion date, with 15/15 quote matches. These are six invented
regression cases, not independent final evaluation data. No extraction accuracy
improvement or general speed gain was measured, and no Laya weights were trained.

The full-result diff revealed a change outside the configured checks: the event
candidate changed an explicitly unknown notice status to empty text. Review
feedback about that field and unknown costs followed the original benchmark
case into the prompt assistant. The assistant reported nine history entries and
returned a model proposal without heuristic fallback. Results/reviews survived
reload; the proposed prompt was not activated.

**Implication:** a green aggregate requires a defined coverage policy. Selected
field checks do not measure complete recall, semantic quote support or unchanged
behavior in other fields. Preserve the full output alongside each scalar check,
and keep unknown wording, missing fields, explicit zero, compensation exclusion,
actual costs, granted time extensions and role associations distinguishable.
Treat incomplete/failed/cancelled model attempts as operational outcomes, not
negative training gold. Human notes and assistant recommendations require
adjudication before becoming training labels.

Build project/template/document-family splits before generating adaptive data.
Keep an independent final evaluation set outside both training and prompt
revision history. Record requested dimension, source/claim association and exact
span, along with model/prompt/schema/input versions and all attempts. For Laya
retrieval or verification arms, measure necessary-clause recall, false support,
contradictions, abstention and downstream extraction/graph quality against the
same LLM-only baseline. Matching quotes and agreeing model verdicts do not
replace independent labels. Keep Laya advisory until that comparison succeeds.

The GUI's current records do not pin model/schema versions or persist jobs on
the server. Sequential case timing includes application processing and has only
one observation per draft. The benchmark deliberately avoids project graph
writes; the earlier graph audits remain separate evidence. MLX and supervisor
work are unchanged by this application learning. See the [benchmark guide and
priorities](https://github.com/UnicoLab/delaylama/blob/main/docs/prompt-benchmarks.md)
for the reusable evaluation workflow and remaining engineering work.

### L25 — Observe the actual invocation before attributing prompt gains

**Observed:** DelayLama's [call-level provenance validation](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-runtime-provenance.md)
records the LLM binding selected by each JSON-document agent invocation, rather
than inferring it from the health panel's default model. It retains configured
and reported model identity, selected generation settings, driver versions,
output-contract hash, application prompt/input hashes and JSON repairs/errors.
Records live in request-local state so concurrent runs and parallel claim passes
do not share attempt history. Exports allowlist metadata and exclude credentials,
headers, arbitrary provider objects and endpoint paths/query.

Two invented English/French identity cases were run with baseline and candidate
drafts. Both met 7/7 selected checks with 15/15 quote matches. Each case request
used five LLM invocations for identity sections: four case requests represent
twenty logical invocations, not necessarily twenty billable transport requests.
The reported model was Gemini 3.1 Flash Lite. A metadata-free baseline correctly
withheld gained/lost prompt checks; the recorded baseline allowed them because
execution observations matched, despite intentionally different prompt hashes.
The English full-result diff still showed four changed sections outside/in excess
of the configured checks. No accuracy or speed gain was measured, no Laya weights
were trained and no agent default prompt was activated.

**Implication:** distinguish case requests, logical invocations, JSON repair and
internal transport retries. Keep all attempts and failures; do not mix them with
semantic training gold. Record the actual model role/binding used for each arm.
Require identical evaluation case definitions and compatible runtime observations
before attributing gained/lost selected checks to a prompt or auxiliary model.
Retain full outputs and human review even when attribution is withheld. For Laya
arms, additionally record checkpoint/tokenizer/calibration, retrieval policy,
training-data provenance and preprocessing versions; those are not supplied by
this LLM-only provenance feature.

Observed metadata is not immutable provider weights or a fully pinned environment.
Unreported defaults, routes on the same endpoint origin, backend revisions and
preprocessing can still differ. Repeat controlled trials with independent labels,
false-support/contradiction/necessary-clause recall, field/role/span association,
full graph quality and resource/latency measurements before promoting a Laya arm.
The 362 Python/208 frontend application tests verify the provenance workflow;
they are not held-out extraction or model-quality certification. Historical L24
accurately describes the earlier benchmark before this new runtime guard.

### L26 — Preserve every premise through chronology corrections

DelayLama's [2026-10-03 chronology audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-contract-chronology.md)
ran eight agents on one invented French contract, then one identity and two
amendment rechecks. The same expanded checks scored 26/29, 27/29, 28/29 and
28/29. Explicit original/current milestones and an amendment's own dated heading
improved, but correcting one missing quote could omit another premise. The final
raw amendment still lacked the original-deadline quotation despite 41 returned
quotes matching source text across the outputs. Source matching is not support
coverage, date-role association or recall.

A separate browser evidence search recovered the original dated provision. Its
field checks suggested seven supported fields and two gaps (`claims_waiver` and
`clause_refs`); overall evidence remained insufficient and no finding was
accepted. The synthetic-trained Laya advisory flagged a possible contradiction.
This disagreement is a review signal, not a measured Laya gain. Keep raw runs,
recovered passages and human decisions separately. Do not turn source absence,
model agreement or an inferred amendment reference into gold labels.

Practical training/evaluation consequences:

- Label span and role independently: original deadline, revised deadline,
  amendment signature and event occurrence are distinct even when every date
  appears somewhere in the source. Superseded status also needs the replacement
  premise, not just the original dated provision.
- Add paired multi-premise cases: original date plus replacement plus unchanged
  price; explicitly absent/unknown waiver language; implicit base-provision
  linkage versus explicit amendment reference. Measure coverage before and after
  a correction, with unrelated premises as preservation checks.
- Synthetic retrieval/entailment examples should include wrong-role exact spans,
  unsupported absence claims, partial support and conflicting amendments. Keep
  unknown fields unknown; successful passage retrieval must not raise the raw
  extraction score retrospectively.
- Evaluate graph effects separately. The light source graph in this pass had
  40 nodes/22 edges, 26 Date nodes and no typed Event nodes, while the saved
  analysis map had explicit event findings. Keyword weather/causal suggestions
  and duplicate per-episode dates do not establish event identity or causation.
  Graph supervision needs independently reviewed typed entities, direction,
  scope and evidence, rather than proximity-based pseudo-labels.

The application tests passed (374 Python, 218 frontend, production build), and
live notes/recovery persisted after reload. Those are application regressions on
one invented contract, not held-out model quality. No Laya training, architecture
or default-prompt activation occurred in this audit. Next run a fixed reviewed
field/span/role suite comparing LLM-only with Laya-assisted retrieval under the
same runtime, then report completeness, false-support rate, latency and memory.

### L27 — Typed events and matching quotes do not establish causal roles

Evidence: [DelayLama's live deep-source graph audit, 2026-10-03](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-source-graph-analysis.md).
One original invented French contract was submitted once to a real Chronograph
background job. The graph went from 40 nodes/22 edges and no Event nodes to
46 nodes/35 edges and five Event nodes. All five event quotations matched the
source text, but semantic errors remained: notice-to-party cause-of-delay links,
ambiguous responsible/affected party direction, and an amendment date absent from
the Event's own quotation. No typed milestone supersession chain was established.

The inspected engine promotes an event's related-concept links to `CAUSED_DELAY`
when its description/type/impact contains delay/impact words. Generated
justifications carry LLM provenance even though the relation type comes from
that keyword rule. A source-backed event plus a matching quote is insufficient
to label these directed edges as gold. DelayLama retains the raw graph, annotates
one date concern and five causal concerns, and hides known uncertain links until
requested for review. This qualifies delivery; it does not fix upstream graph
semantics or prove Laya improves them.

Training and evaluation consequences:

- Separate association, actor responsibility, affected party, notice compliance,
  temporal succession and causation. Label both endpoint roles and supporting
  premises. Include exact-source wrong-role edges as independently reviewed hard
  negatives, rather than copying model/keyword graph edges into training labels.
- Preserve month-only precision and uncertainty. Period boundaries are not
  observed event days, and an occurrence duration is not an established end date.
  Each exact-date field needs its own supporting date passage; nearby context
  cannot silently become the supplied evidence.
- Measure graph recall and relation/direction correctness separately from event
  quote match. Compare saved analysis, light source ingestion and deep-source
  output; they have different coverage and derivation methods.
- Keep original source hashes, submission intents, job handles and raw before/
  after snapshots. A timed-out write or missing status is unknown, not permission
  to resubmit or evidence of failed extraction. Operational receipts are distinct
  from model quality labels.

Application checks passed (392 Python, 224 frontend, production build), and reload
and an API restart retained the same completed run/job. Chronograph worker status
remains in memory; the saved receipt cannot restore a lost worker. This is one
synthetic case, not held-out accuracy, speed or memory evidence. No Laya training
or architecture change occurred. Next compare LLM-only against Laya-assisted
clause retrieval/role validation on an independently reviewed, fixed suite, with
false-support rate, premise recall, causal direction, latency and memory reported.

### L28 — Compare missing fields and stable source anchors, not just wording

Evidence: [DelayLama's finding comparison audit, 2026-10-03](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-finding-comparison.md).
A live contract-identity experiment on the same invented Aster original returned
19 matching quotations but withdrew `scope.scope_of_works`, while retaining the
nearby project description. Two other sections had no supporting quote. Exact
span presence therefore cannot establish field completeness, an absence claim,
or correct field/value association. Unchanged evidence gaps must remain visible
when reviewing iterations, rather than disappearing from a changed-values view.

The penalty label also changed from French prose to `late_delivery` while its
quoted provision, rate and cap stayed the same. Exact-label-only pairing showed
it as absent plus new. The application now allows a transparent pairing by a
unique identical quotation/reference set within the same list section. Explicit
IDs still take precedence; duplicated anchors, other sections, changed references
and conflicting IDs remain separate. Such UI pairing is not an independently
reviewed entity-equivalence label and must not become training gold.

The recorded model/settings matched, but the imported baseline did not retain
its original prompt. That comparison cannot attribute differences to a prompt
revision. Keep complete source, prompt, invocation record and review decisions
for every new arm. An accepted baseline recovery is not a new run's acceptance.

For Laya data, include paired cases where an exact relevant quotation exists but
a required field is missing, an old unsupported value is correctly withdrawn,
or a schema label changes without a new provision. Evaluate field completeness,
source absence and stable identity separately. These cases need independent
review; matching quotes, assistant revision rationales and UI change counts are
not quality certificates. No Laya training or measured accelerator gain occurred
in this interface audit.

The full live feedback loop sent four recorded entries to the assistant, whose
proposal was manually made conditional on explicit original evidence before
testing. The follow-up still omitted the works-scope field while preserving three
dated milestones. That failed refinement is retained; an assistant rationale or
successful request does not establish extraction improvement. The comparison
keeps a flagged baseline's note visible as a recheck concern, without transferring
its human decision to the new run. Follow-up evaluation should score whether the
specific reviewed defect was resolved, alongside preservation of unrelated facts.

### L29 — Missing values need field definitions, own spans and uncertainty checks

Evidence: [DelayLama missing-field recovery audit, 2026-10-03](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-missing-fields.md).
Checking only returned assertions missed an omitted works-scope field despite a
matching project-description quotation. The application now shows core scalar
gaps and searches them independently, retaining candidates outside the extraction.
An initial source pass required detailed specifications before recognizing a
broad renovation scope; an explicit task definition recovered the stated scope.
Both attempts remain visible. Definition quality is part of task quality, not a
model-size problem by default.

Four fixed development smoke cases exposed an omitted response entry and an
absence statement offered as a value. A bounded response repair now requires
every requested key; literal values need their own exact matched source spans,
and recognized English/French unknown/future-definition wording stays unresolved.
The final four cases passed, but they were used to develop the corrections and
are not independent accuracy evidence. General semantic association and silence
remain unproven; lexical guards do not certify those properties.

For Laya tasks, distinguish an explicit broad scope, a project title alone, a
deferred scope, and a matching passage with the wrong field role. Include price
vs penalty and invoice deadline vs installment schedule examples. Gold labels
need independently reviewed field definitions and spans, not UI-generated gaps,
assistant candidates or literal matching alone. Evaluate candidate recovery
separately from whether a later extraction actually adopts the correct value and
preserves unrelated facts. The Aster extraction still lacks its scope field; this
batch only recovered a review candidate and sent it into feedback. No Laya
training or measured speed/memory improvement occurred.

### L30 — A recovered candidate must survive a real rerun without hiding losses

Evidence: [DelayLama correction audit, 2026-10-03](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-review-correction.md).
The next Aster development rerun returned the missing broad works scope, with
its original quotation, after shared field guidance and a recorded correction
draft were introduced. This updates L29's earlier failed extraction observation;
the earlier failed runs and recovery attempts remain evidence. It is one reused
synthetic specimen, and both runtime guidance and prompt changed, so it proves
neither held-out improvement nor isolated prompt causality.

The same rerun changed five other findings and no longer returned a previously
reported penalized-milestone value. The UI now surfaces those changes beside the
targeted return and leaves their meaning for review. Never automatically restore
baseline values: some may be wrong. Preserve originals and independently assess
both the corrected target and collateral additions, withdrawals, wording and
clause association. Quote presence, candidate adoption and semantic correctness
are distinct measurements. In particular, a generic insurance requirement does
not establish the names of its insurer/beneficiary solely from nearby party names.

For Laya evaluation, use separate labels for candidate retrieval, explicit role
support, missing/contradicted facts, amendment precedence and preservation of
independently supported facts. Feedback-generated candidates and successful UI
checks must not become gold training labels without review. Persist the exact
source, prompt, runtime contract and returned model, and retain failed reruns.
No new Laya training or measured acceleration benefit occurred in this batch.

### L31 — Insurance roles need separate labels and their own evidence chain

Evidence: [DelayLama insurance-role audit, 2026-10-03](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-insurance-roles.md).
A generic insurance requirement had been expanded into named contracting parties
without source role support. Clarifying the legacy field as the party required
to arrange insurance, separately from carrier and beneficiary, improved six fixed
English/French development smoke cases: generic requirements, explicit roles,
reversed roles with a carrier, a role alias, and a negative obligation. All six
passed; they are not independent accuracy evidence. The repeated Aster rerun
retained its insurance type and left the two unsupported roles empty.

For Laya examples, label policyholder/obligated party, carrier, beneficiary and
unknown independently. Include nearby names without role assignment, reversed
parties, negative duties and role aliases needing a second identifying passage.
Name presence is a narrow review check, not entailment: both names can occur in
the wrong relationship. Do not train support labels from quotation matches or
assistant verdicts without independent role review.

A failed existing-field recheck must reach the correction draft even without a
human-authored note; otherwise the feedback loop silently loses its diagnosis.
Keep model suggestions, source warnings, user notes and original values distinct.
The same rerun changed generated milestone IDs while retaining dates/statuses.
Evaluate semantic preservation separately from record pairing; uncertain identity
should stay unpaired rather than becoming a guessed link or a claimed value loss.
No Laya training or measured accelerator improvement occurred in this batch.

### L32 — Make review findings distinct before curating model feedback

Evidence: [DelayLama results-first audit, 2026-10-03](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-results-first.md).
The contract desk repeated an assessment in linked and standalone lists, while
coverage and an expanded timeline displaced the actual contract results. A
compact source-aware summary, section shortcuts and a contextual inspector now
make each saved finding easier to inspect. Linked assessments retain all their
premises in one place; unresolved or ambiguous event references remain separate.
Repeated event IDs cannot establish a unique join. All 259 frontend tests and a
production-browser check passed, including an observed 389-pixel layout.

For training-data export, count the underlying versioned finding once, regardless
of how many views display it. Preserve the original document, array index/review
path, prompt/run identity, all relevant passages and uncertainty. A summary card,
source highlight, expanded reasoning panel or user navigation is not a support
label. A missing primary value means extraction omission/unknown, not proved
absence in the source. Ambiguous event links need adjudication before relational
labels are derived. No new Laya data, training or measured quality/speed benefit
was produced in this interface batch. Stable cross-run milestone identities and
durable server-backed review exports remain unfinished work.

### L33 — Separate source alignment from premise support

Evidence: [DelayLama milestone comparison audit, 2026-10-03](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-milestone-comparison.md)
and [prioritized contract improvement plan](https://github.com/UnicoLab/delaylama/blob/main/docs/contract-analysis-improvement-plan.md).
Two recorded Aster runs retained three scheduled dates/kinds but renamed all
generated milestone IDs. A source-aware comparison now aligns unique complete
date occurrences in uniquely located original passages, within the same milestone
kind and unchanged source. Repeated passages/anchors, ambiguous same-date
mentions, unsupported dates and conflicting known source identifiers remain
unpaired. Same-day obligations at different source positions remain distinct.
All 271 frontend tests and the production-browser review/navigation checks passed.

The aligned superseded deadline lost its replacement-amendment quote on the
rerun. Date identity was retained, but its status lost a supporting premise.
Quote-count reductions now remain visible in comparisons and correction checks,
even without a value change. A combined quote can legitimately replace two;
the count warning is not a semantic verdict. Source graph edges, canonical IDs,
old outputs and human review decisions were not rewritten or inferred.

For training/evaluation data, label dated span association separately from
contractual role, amendment applicability and supersession. Do not label two
findings semantically equivalent because a source alignment paired them. Include
reassigned generated IDs, reordered rows, changing quote windows, repeated
clauses, multiple obligations on one day and removed-premise negatives. Preserve
each run's original source, fields, exact passages and review provenance. A
retained value or quotation match is not a support label. Independently reviewed
contract families and server-backed versioned reviews remain priorities before
claiming improvements from Laya. No new Laya training or measured quality/speed
gain occurred in this batch.

### L34 — A matched amendment is not evidence that its conditions took effect

Evidence: [DelayLama milestone premise audit, 2026-10-03](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-milestone-premises.md)
and its preserved initial/final live responses. The initial development pass was
5/6: an LLM reported a completion date as superseded because a potential new date
was quoted, despite acknowledging an unmet permit condition. Both quotations
matched the source. Span presence and even an apparently correct explanation
therefore cannot stand in for a sound applicability verdict.

The application now requests separate original-date and replacement-applicability
premises, plus condition states and independent fulfillment passages when
recognized contingencies occur. A cropped replacement quote is checked against
its original sentence. Reported unmet/unknown or source-unverified fulfillment
cannot preserve unqualified support. The UI and correction loop retain these
checks as candidates, with actual response-model observations and bounded repair
history. Generated milestone record IDs are context, not source assertions.
Old outputs/default prompts/review identities were preserved. A full browser
correction rerun restored the missing amendment quote but changed a guarantee
beneficiary and other wording, illustrating why collateral changes need review.

The final 7/7 live development pass includes a fulfilled-condition control; it is
not independently held-out evidence. All 431 backend and 278 frontend tests
passed. Lexical cues do not certify semantic entailment, all-condition coverage,
negation scope, obligation identity or legal effect. Include paired fulfilled,
unmet and unknown conditions, cropped clauses, unrelated permit issuance,
multiple necessary conditions and contradictory provisions in future training.
Label source-span presence, field association, premise completeness, condition
fulfillment and applicability separately. Keep failures and family-level splits;
do not turn assistant verdicts or matched spans into automatic truth labels.
No new Laya training or measured quality/speed improvement occurred in this batch.

### L35 — Reproducible reviews need durable history, not automatic truth labels

Evidence: [DelayLama Studio backup audit, 2026-10-03](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-studio-backups.md).
Project-backed revisions preserved six original runs and 13 history entries
through browser migration, two-client conflict, restore, import and reload. Each
recorded source, prompt, result, finding-scoped review and runtime observation was
retained. Historical preview is separate from the active draft and acceptance.
The initial real-process SQLite initialization lock failure remains documented;
transactional schema creation fixed it without weakening the concurrency test.

For reusable training/evaluation pipelines, retain the exact source revision,
run prompt/config, output, reviewer decision and amendment/correction provenance.
Archive divergent reviewer copies before replacement; never silently select the
latest browser write as the authoritative label. Export/import should round-trip
unknown metadata and full histories. Explicit unsupported/conflicting/unknown
states must survive storage and later dataset preparation.

These are reproducibility capabilities, not training labels or measured model
benefits. User acceptance, source matches and assistant verdicts still require
independent adjudication and family-level splits before inclusion as truth.
Non-Studio reviews remain partly browser-local and snapshot backups do not resume
interrupted jobs. No new Laya training, accuracy or latency improvement was
measured in this batch.

### L36 — Original evidence and annotation identity must survive recovery

Evidence: [DelayLama project review audit, 2026-10-03](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-project-reviews.md).
Project/Extract/Claim Desk reviews now retain exact-source revisions, explicit
conflict resolution and source-recheck runtime. A fresh browser restored an exact
snapshot. Claim Desk feedback transferred its merged context to Studio without
pretending an imported agent analysis had the same output or historical prompt.

A grounding bug allowed generated summaries/legacy extractor context to act as
citation evidence. Registered originals are now separate from summaries and input
budgets; unavailable source cannot establish grounding by self-quotation. The live
insurance recheck found insufficient support for named roles despite a generic
quote match. Preserve that failure with its actual model/config/runtime before
preparing training examples.

Datasets need exact source revision, field/role association, full result and review
scope. Compact cache hashes are not authoritative annotation identities: collision
or source changes must not inherit accepted labels. Archive disagreements before
replacement. Notes, matched quotes, candidates and assistant verdicts are not
truth labels without independent adjudication and family-level splits. This batch
measured no new Laya training, quality or speed gain. Interrupted-job recovery,
cross-device hosting and retention remain separate work.

### L37 — Recovery records and exact spans do not establish field semantics

Evidence: [DelayLama workflow recovery audit, 2026-10-03](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-job-recovery.md).
A durable journal retained a live extraction and simulated interruption through
API restart. Identical submission identity/body returned one execution; changed
body was rejected. Lost-response lookup is distinct from model rerun. Preserve
partial output, actual process state and errors; observation timeout or quiet
progress does not prove death. Process tracking currently uses local PID checks,
not a distributed lease, checkpoint resume or OS memory limit.

Cèdre's synthetic development example passed checks for core fields and 12 exact
quote offsets, while `extension_causes` incorrectly contained a notice procedure.
Retain the unchanged error and flagged review. Training/evaluation should separate
entitlement grounds, notice obligation/procedure, evidence of actual notification,
late consequences and uncertainty even when they share clauses. Span presence and
correct adjacent dates do not make that field association a support label. These
are development findings, not independently adjudicated gains or Laya evaluation.
No new Laya training, quality or speed improvement was measured in this batch.

### L38 — Give negative/procedural facts their own slots; keep quotation policy explicit

Evidence: [DelayLama extension-ground audit, 2026-10-03](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-extension-grounds.md).
The notice-only failure was repaired by separating permitted extension grounds
from request deadlines/recipients/forms. A live counterexample then placed an
explicitly excluded weather ground in the positive slot. Providing a separate
exclusion field repaired placement in six short French/English development cases.
Preserve entitlement conditions, exclusions, request requirements, actual notice
and granted extensions as distinct labels; an exclusion or a duty to describe a
cause is not positive entitlement. These hand-written cases share a shell and do
not establish independent model accuracy or a Laya gain.

The same pass exposed accent/translation changes in quotations. Navigation can
locate similar text without making a valid support label. Strict verification now
accepts exact or whitespace-only matches, displays the actual original span, and
records its policy. One translated scope quotation remained rejected; a separate
reviewed source recheck supplied a verbatim English passage without rewriting the
saved failure. Keep raw output, reviewer decision, candidate, source identity and
policy separate. Never train on fuzzy matching, lexical warning absence, model
verdicts or reviewer acceptance alone as automatic entailment truth.

A Laya shortlist/reader arm should retain both positive and excluded clauses plus
procedural/actual-notice hard negatives. Measure span association and false support
alongside recall, latency and memory on independent families. Model selection,
per-language calibration and repeated-run variability remain necessary. This batch
measured no new Laya training, architecture, quality or speed improvement.

### L39 — Review completion and reviewer absence judgments are not training truth

Evidence: [DelayLama review-priority audit, 2026-10-03](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-review-priorities.md).
Acceptance records a decision; it does not erase quotation gaps, interpretation
warnings or missing fields. Missing outputs need separate reviewer states for
not-stated, not-applicable and extraction-missed. Cèdre distinguished an extension
request procedure from absent permitted grounds, and an explicit payment deadline
from an absent staged schedule. The original model outputs and failed reviews
remain separate from these judgments and immutable review histories.

Field determinations reach prompt experiments as untrusted review context, not
forced replacement values. Exact source/result identity is necessary for restoring
annotation context. It is insufficient to establish an absence or entailment label.
Do not train on acceptance, closed task queues, matching spans, empty searches or
reviewer not-stated judgments automatically. Independently adjudicate the field,
relevant passages, document coverage and contract family before assigning a label.

Useful failure labels include missing/invented facts, wrong role or field, date
precision, negation, amendment precedence, unsupported edges, incomplete coverage
and invalid quotations. Laya shortlisting and triage arms must compare against the
unchanged LLM and pretrained Laya on independently separated families; report false
support, necessary-clause recall, per-language quality, served latency and memory.
An optional GLiNER span/record arm needs the same benchmark and explicit incremental
benefit. This batch added no Laya training, architectural change, accuracy or speed
gain. MLX encoder training, full CLM objectives and stronger OS isolation remain
separate unproven engineering work.

### L40 — Structure repair can succeed while field interpretation regresses

Evidence: [DelayLama omitted-section audit, 2026-10-03](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-review-omissions.md).
A deliberately seeded incomplete output exposed omitted core sections. Real source
searches and a correction draft returned five target values with matching passages
and two scheduled milestones. This establishes a software review workflow, not
extraction recall or independent accuracy: the omissions were seeded, and the
contract was a development fixture.

The same reruns added an unstated contract classification and put an ordinary
delay-notice obligation in the extension-request slot. Further prompt feedback
and tighter shared guidance did not eliminate those failures. Another rerun
filled a positive payment-schedule field with an explicit absence statement.
A source recheck proposed support for that statement; a limited wording guard
kept the field uncertain and retained the model verdict. Separate an installment
schedule, invoice deadline, source absence, generic notice duty and an extension
request. Matching quotation text and repaired structure do not establish the
field association or the correctness of other outputs.

Keep wrong-slot and absence-as-positive examples as hard negatives after separate
adjudication. Do not train from source-search support, user notes, repaired fields,
review acceptance or lexical warning absence automatically. A useful Laya triage
arm should measure these errors alongside necessary-clause recall, multilingual
quality and full-pipeline latency on independent contract families. Retain all
failed reruns and collateral changes; evaluate whether a candidate fixes the
target while harming another field. No new Laya training, architecture, speed or
quality gain was measured in this batch.

### L41 — Evidence checks need an exact field/value owner and visible uncertainty

2026-10-03. DelayLama's [field-evidence audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-field-evidence-links.md)
found that individual insufficient/contradictory source checks could disappear
from result cards, graph navigation, timeline and Claim Desk when a separate
warning map was missing. Section acceptance did not justify that loss. A matching
section quotation also did not establish support for every displayed value.

The frontend now projects recorded checks onto one exact current field/value,
exposes own quotations beside that value and retains gaps after acceptance.
Duplicate checks, stale values, unmatched/missing quotations and unresolved
amendment premises or activation conditions remain concerns. A negative schedule
passage can match the original while still being inappropriate as a positive
payment schedule. Unrequested checks do not supply a neighboring field's verdict.
These are display and review fixes; the saved model outputs stay unchanged.

For future auxiliary models, keep field ownership, span presence, semantic
support, applicability and human adjudication as separate labels/observations.
Use wrong-slot, obsolete-value and unmet-condition examples only after independent
label review. A candidate classifier can propose a review priority or retrieve a
necessary clause, but must not erase uncertainty or create a factual graph edge.
Benchmark it against the unchanged LLM pipeline, including collateral errors,
necessary-clause recall, latency and memory. Do not train automatically from notes,
source-search support, text matches or reviewer acceptance.

The existing server-backed Érable Run 4 was inspected with its actual recorded
checks. All 374 frontend tests and the production build passed; 32 protected
source/output/code hashes matched. No new model calls, Laya training, architecture,
acceleration, quality or speed gain were measured in this batch. The independent
contract-family evaluation remains the prerequisite for promoting a Laya arm.

### L42 — Full workflow performance includes loading, state and failure recovery

DelayLama's 2026-10-03 frontend pass reduced the main production JavaScript chunk
from 599.72 to 454.80 kB by separating pure evidence helpers from secondary UI
panels. That is approximately 24% less main-chunk code, not a measured latency or
model-quality gain. Opening optional views still fetches their chunks. Benchmark
cold/warm end-to-end task time separately from model service latency and cache hits.

A real failed graph download stayed local to the panel, but retry did not recover
after restoring the file: the browser retained the failed native import. Fresh
React lazy state recovered a mocked loader in tests, which was insufficient proof
for the production failure. A deliberate test-tab reload recovered it; the product
now explains save/export-before-reload and never automatically discards the parent
workspace. Test actual failure mechanisms, including cached artifacts and stale
deployment versions, rather than inferring recovery from component tests.

Apply that distinction to model services: separate worker restart, checkpoint
availability, artifact cache state, request replay and current user selection.
Recovery must preserve the correct input/revision and avoid silently starting work
twice. A supervisor heartbeat alone does not prove checkpoint recovery or bounded
memory. These are evaluation requirements, not newly implemented Laya features.

All 379 frontend tests and the production build passed; 32 protected hashes and
the existing Érable Studio revision 10 were unchanged. No Laya training, model
calls, graph writes, new accuracy result or model-speed improvement was measured.
The [DelayLama audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-deferred-panels.md)
retains the native-import failure and limits alongside the successful UI checks.

### L43 — Compact review surfaces without hiding uncertainty or changing ownership

DelayLama's 2026-10-03 reading-layout pass removed competing height caps from
timeline/claim/document reading and compacted backup controls. In the same desktop
contract view, the review-conflict surface decreased from about 230 to 88 px.
Conflict choices, storage/backup failures and the prohibition on merging acceptance
remain visible; routine recovery explanations are available on demand. Explicit
timeline selection focuses the exact source inspector, with return to the original
entry. A controlled inspector retains its owner's navigation behavior.

UI density, evidence matching and task completion are distinct measurements.
Reducing a panel's height does not establish faster review, lower false acceptance,
better labels or extraction accuracy. Month-only dates and entitlement/cost warnings
remained visible in light/dark checks. Preserve uncertainty and source/revision
ownership when testing an auxiliary Laya triage or retrieval arm; a tidier screen
must not silently turn an assistant candidate into an accepted training label.

All 380 frontend tests and the production build passed; protected source/output
hashes and the existing Érable revision remained unchanged. The requested 390 px
browser override yielded only 582 px of effective DOM width, so mobile certification
remains incomplete. No Laya training or model quality/speed result was measured.
See the [reading-layout audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-reading-layout.md).

### L44 — Own-finding source coverage and the danger of overbroad labels

DelayLama added optional original-passage requirements to selected field checks.
Only quotations of the configured finding count, after source matching. A correct
value and a perfect aggregate citation-match count can coexist with missing own
premises. Three new self-authored English/French cases measured 8/10 selected
values and 6/14 required passages; an experimental prompt rerun measured 10/10
values and 8/14 passages. Both used fifteen recorded Gemini section invocations.
These are development observations, not independent model quality or speed gains.

Inspection distinguished an actually omitted role-to-name premise from broad
anchors that demanded unrelated neighboring text. A correct signed replacement
quote failed an anchor that also required an unchanged-price sentence; separate
valid obligation/absence quotes failed a combined single-quote requirement. The
original expectations and unsuccessful coverage results were retained. Define
minimal self-contained premise labels before model calls, then adjudicate meaning,
required-role association, absence and amendment applicability separately. Passing
literal coverage still cannot certify interpretation or complete recall.

For Laya retrieval/triage experiments, evaluate necessary-premise recall and role
association instead of rewarding arbitrary paragraph overlap. Preserve original
spans and retrieval omissions; a good semantic candidate can have different span
boundaries. Separate field correctness, span coverage, semantic support, reviewer
acceptance and operational timing. Review notes and these synthetic development
cases are not automatically independent training labels. Keep unseen contract
families outside candidate refinement and benchmark against the unchanged LLM arm.

The new GUI routes source gaps to the exact owner and restores requirement scores
from recorded output/source. A portable JSON fallback retained both complete runs
when native download/clipboard capture was unavailable; an isolated server-store
round-trip also preserved them. All 393 frontend tests and build passed. No Laya
weights, training recipe, default prompt or project graph pipeline changed here.
See the [source-expectations audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-source-expectations.md)
for full records, strict-label failures and scope limits.

### L45 — Make review gaps durable; inspect assistant proposals for example leakage

DelayLama now puts configured field expectations in the exact finding inspector,
separating returned values, required original text and the agent's own quotations.
Unmet literal expectations can be filtered and explicitly appended to a flagged
review note. Existing reviewer text is retained, repeated appends do not duplicate
the block, and acceptance does not erase a source-coverage gap. Context is
recovered after reload only for the exact original agent/case/prompt/source/output
and review scope. Ambiguous owners are withheld rather than guessed.

A live production prompt-improve API call received four prior history entries,
including both synthetic benchmark records, the selected case and feedback. It
reported success, no fallback and no saved/default prompt change. Its proposal
repeated a source-premise instruction already present and introduced the synthetic
party name `Delta Maintenance Ltd.` despite feedback explicitly prohibiting
case-specific names. Its self-reported confidence was 0.9; that is not evidence
of improvement. A universal two-quotation rule also needs adjudication when one
original clause already contains both necessary premises.

An advisory GUI check now reports newly introduced recorded party names, contract
references, prices and ISO dates that also occur in the original source. It survives
reload and distinguishes copied values from generic instructions. This is a small
literal review aid, not a comprehensive leakage detector or an activation gate.
Unknown/extraction-omitted values, paraphrases, semantic leakage, inappropriate
universal rules and regressions still need human review and independent evaluation.

For Laya data generation or prompt-assisted labeling, do not treat returned
assistant changes, confidence scores, accepted notes or synthetic benchmark gains
as correct training targets. Inspect source support and label boundaries; retain
failed proposals, avoid example-specific reusable rules and reserve independent
contract families. Treat prompt refinement as a hypothesis and evaluate it before
using generated labels or promoting a model. The unchanged LLM-only arm remains
the comparator for a future Laya-assisted retrieval/triage experiment.

All 405 frontend tests and build passed. A live selected-finding focus/scroll bug
was fixed after lazy panel loading. No Laya training, weights, architecture,
acceleration, new extraction runs or graph writes were performed in this pass.
See the [benchmark-review audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-benchmark-review.md).

### L46 — Preserve ambiguous event identity and financial dimensions end to end

A DelayLama full-stack audit exposed a join error: normalization made duplicate
original event IDs unique for display, then a reference to the duplicate original
could be treated as explicit and borrow the first event's date. The fix retains
original ambiguity across normalization/finalization/reload, leaves affected
assessments unlinked and prevents inferred notice clocks, risk/amendment chains
and analysis-map event edges. Raw findings and independent explicit trigger dates
remain available. Missing original-identity metadata in older normalized records
cannot be reconstructed from unique display IDs alone.

Eight agents ran on three new English synthetic contract documents, plus two on
a control contract. Thirty targeted workflow/source checks passed, including
source hashes, quote presence, amended deadlines, month-only event precision and
cross-contract graph isolation. Four original-source graph jobs completed. These
are development checks, not independently labeled extraction/graph accuracy.
One insurance finding still omitted its role-to-name premise, and six stored
Chronograph cause links remained suspect and hidden for review by default.

The model also put requested compensation into actual cost impact. A lexical
review guard now flags recognized requested/awarded numeric amounts, preserving
raw output and unresolved actual expenses. A requested amount, an award, incurred
cost, entitlement and closure/waiver are separate labels. The combined returned
cost status and recommendations still need semantic adjudication. Backend tests
and visible source warnings do not establish model correctness.

For Laya training and downstream evaluation, preserve original event identifiers,
normalization decisions and ambiguity labels; never generate a positive join
solely from a rewritten ID or title similarity. Keep event existence, reference
resolution, temporal precision, causal direction and financial stage as separate
targets. Synthetic data should include duplicate IDs, wrong references, separate
notice dates and requested-but-unawarded amounts. Reject ambiguous automatic
labels, reserve independently reviewed contract families and compare complete
LLM-only versus Laya-assisted workflows before promoting an accelerator.

All 532 backend tests, 409 frontend tests and production build passed. No Laya
training, weight, architecture or default-prompt change was made in this pass.
See the [event-reference full-stack audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-event-references.md).

### L47 — Remove duplicated diagnostics before adding a retrieval accelerator

DelayLama's claim flow sent prior saved analyses twice: as contract analysis and
reused graph context. Those structures also contained runtime/provenance counters
and evidence aggregates. A lossless projection of relevant analysis data now
removes known operational diagnostics/vectors, keeps exact quotations, original
ID ambiguity, field concerns and failures, and sends identical context once.
Original text and saved outputs remain unchanged; distinct graph suggestions
remain available and explicitly unverified.

A live paired Hazel sample used identical original inputs and recorded model
configuration. Each time/cost input decreased from 75,238 to 31,180 characters
(58.6%). Both runs returned the same four event/target/status tuples and six
original-matching quotes. One pair is not independent accuracy/recall evidence
or a demonstrated latency benefit. The combined requested/awarded financial label
remained semantically questionable even though the tuples matched.

The claim-preparation pipeline also passed generated event output into its
document slot. It now preserves the original execution text alongside contract
clauses. With no original, quotations cannot verify against generated detector
JSON and an analysis gap is displayed. A fresh pipeline and separate saved-project
run verified quotes from both kinds of originals. The latter still renamed source
event E4 to display E1, omitted the event date premise from its own quote and
called an extension finalized without an explicit settlement statement.

Before attributing gains to Laya retrieval, compare against this cleaned LLM-only
baseline. Optimize context duplication and diagnostic noise independently of
clause selection. Necessary premise recall, source/display identity, temporal
precision, financial stages and unsupported finality remain separate evaluation
targets. Quote presence and unchanged model labels cannot certify these meanings
or safely produce positive training labels.

All 539 backend tests, 410 frontend tests and production build passed; live
browser verification also fixed a citation card hidden behind the sticky header.
No Laya training, weights, architecture or default-prompt change was made.
See the [claim-context audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-claim-context.md).

### L48 — Source identity and missing premises need separate labels

A paired DelayLama Pine test exposed source event E4 being displayed as E1 and
the event's dated occurrence being omitted from its quotations. New output
guidance separates an exact original source reference from the analysis ID and
requests the dated identifier premise. With identical input and recorded
model/parameters, the candidate retained E4 and quoted that premise. This is one
synthetic regression, not an independent quality or latency result.

A fresh Cedar saved-project workflow returned four occurrences with E4, E12,
RX-9 and an unlabeled rain event. Eight time/cost assessments linked to their
current analysis IDs; source references remained separate facts, not guessed
aliases. Cross-row display/source conflicts stay ambiguous even after assigning
unique display IDs. A repeated original reference must not become a positive
identity/link label merely because one row is renamed. Source-reference absence
from an event's own quotes raises lexical review, not semantic certification.

The same run still cited rules without some assessments' own occurrence premises,
suggested no further action after a grant despite no-final-settlement wording,
and introduced a late-notice exception not established by the contract. Deep
source graph jobs completed but produced three causal edges targeting event
identifiers and five targeting parties with questionable role/direction. The
interface treats these as uncertain suggestions. Quote presence, empty review
diagnostics and operational completion cannot supply trustworthy support labels.

For Laya data, label original identity, unique analysis reference, actor/rule/
occurrence support, edge type/direction and grant/settlement status separately.
Include missing-premise and counterfactual examples across independent contract
families; group translations and paraphrases before splitting. Keep these known
development failures outside claims of held-out performance. Clause shortlisting
must retain every required premise and conflicting amendment, and escalate when
uncertain. Compare against the cleaned LLM-only baseline before attributing any
workflow improvement to retrieval or contrastive training.

Six extraction calls, two separate deep graph jobs, nineteen targeted receipt
checks, 552 backend tests and 414 frontend tests were recorded. No Laya training,
weights or architecture changed. See the
[source-identity audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-source-identity.md).

### L49 — Selecting original passages avoids transcription errors, not interpretation errors

DelayLama's Cedar claim test still omitted occurrence premises after stronger
instructions. Free quotation recovery then returned proposals rejected by exact
original matching. Selecting an index from bounded original passages instead
yielded seven source-matched occurrence candidates in the paired run and five in
a saved claim rerun. The selected text is copied from the original; the model
cannot replace it with a paraphrase, altered date or joined spans. Conclusions,
actions and original citations remain unchanged, and association warnings persist.

This supplies a useful comparator for Laya clause/occurrence shortlisting. Preserve
the original option text, provenance, identity and rendering; train/rank selection
separately from semantic support. Candidate reference/date matches cannot label
role association, applicability, entitlement or causation as true. Include a
decline option and missing-premise cases, and evaluate necessary-premise recall
across independent contract families before restricting full-source analysis.

The shipped lexical retrieval uses eight options per request and overlapping
2000-character long-line windows. These caps and recognized date/reference formats
do not establish exhaustive recall. Two final saved model quotes still contained
ellipses and stayed unverified. Actual expense was also incorrectly set to zero
in one fresh detector result and remained a review concern. A field's literal
match and an error-free job are not positive semantic labels.

Thirty recorded model calls retain failed attempts as well as successful option
selection. The paired final request took 5.78 seconds versus 4.78 seconds baseline;
two extra evidence calls were added. Neither speed nor independent semantic gains
are established. All 582 backend and 423 frontend tests/build passed. A live review
note and candidate concern reached the prompt lab; no prompt activation or Laya
training/architecture change occurred. See the
[claim-occurrence audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-claim-occurrence.md).

### L50 — Stage-specific facts expose errors hidden by a combined claim label

DelayLama now separates incurred cost, payment request, payment award, extension
grant, final settlement and claims waiver, with each record's own quotation and
fact date. Unknown, explicit absence and numeric zero are different labels.
Legacy results do not acquire inferred stages, and duplicate or missing types
stay review concerns. Requested payment cannot be a label for actual expense;
an extension or partial award cannot establish settlement or waiver.

An initial English run returned only 8/24 required stage records; a French run
returned 6/8. Stronger per-pass instructions returned all records, but the final
English run met only 21/24 configured status/value/date combinations. The saved
pipeline met 22/24; two no-settlement statements remained incorrectly unknown.
French status/value checks passed 8/8 while only five date checks passed. Matching
source quotes did not prevent an inferred zero award, undated facts acquiring a
grant/occurrence date or incomplete interpretation. Retain these as development
failures, not independent accuracy or automatic contrastive training labels.

For future Laya datasets, annotate financial/temporal stages and the role of each
date separately. Include requested-but-refused, partial awards, unquantified
expenses, explicit zero actual cost, unknown waiver and grant-without-settlement
examples. Rank/retrieve each fact's own necessary premise, with explicit decline,
contradiction and missing-type cases. Original matching does not label semantic
support; distinguish absence from a model's failure to return a record.

Twenty-six extraction-model call records and one assistant refinement request
were retained. The proposal remained inactive and unevaluated; Studio reruns
still need captured detector inputs to reproduce saved claim-event joins. All
604 backend and 430 frontend tests/build passed. No Laya training or architecture
change occurred. See the
[recorded-outcomes audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-recorded-outcomes.md).

### L51 — Freeze upstream inputs before evaluating a downstream prompt or head

DelayLama's claim Studio previously supplied original text without the event
register used by saved analysis. A prompt comparison could therefore also be an
input-path change. Experiments now bind events and optional contract analysis to
the exact original, retain each run's actual inputs and snapshot fresh detector
intent. Benchmarks freeze those inputs; stale/missing registers block claim
calls. Different inputs cannot be credited as prompt improvement. Duplicate IDs
remain ambiguous rather than selecting one occurrence.

The live saved-review rerun linked six time/cost assessments explicitly to three
events. Editing to an invented single-event case required fresh detection. Its
benchmark passed two request/award value checks and both own required passages,
with 14/14 literal quotation matches. The model nevertheless described a two-day
interruption as time entitlement despite no documented extension decision. That
flag reached an assistant revision with 11 full history entries. The candidate
remained inactive and unevaluated. Matching spans, explicit IDs and passing two
checks do not establish semantic correctness or complete recall.

For future Laya training/evaluation, snapshot source, task intent, upstream event
register, schema, language and runtime identity per example. Hold upstream
inputs fixed when isolating a prompt/head change; evaluate detector+reader arms
separately when changing the whole pipeline. Include missing/misclassified events,
ambiguous IDs and stale source bindings in orchestration tests, not automatic
positive training labels. Entitlement, interruption duration and granted
extension need independent labels and necessary original premises.

Clause shortlisting and review triage remain promising *experiments*. Their
acceptance needs independently reviewed necessary-premise recall, false support,
amendment retention, per-language calibration and complete pipeline latency
against the source-bound LLM baseline. A better input path is not evidence that
Laya or an architecture is more accurate or faster. Fourteen extraction-model
call records plus one assistant request were retained; 605 backend and 467
frontend tests and production build passed. No Laya model/training/acceleration
change occurred. See the
[claim experiment audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-claim-experiment-inputs.md).

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
