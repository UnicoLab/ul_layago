# Shared learning log

Updated 2026-10-04. Scope: the supplied Hive engine, the current Round 4 Hive report,
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

### L52 — A correct quotation can still carry the wrong status or association

DelayLama now defines regression checks through readable result fields and each
finding's own original passages. Model values remain observations; a reviewer
enters expected answers independently. Checks combine event ID, assessment target
and fact kind rather than relying on global labels or array positions. Repeated
settlement/waiver types also need parent-qualified identities in run comparisons;
shared quotes must never merge facts from different events or targets. Duplicate
parents and child kinds remain unresolved rather than choosing by order.

The initial live request/award checks passed 4/4 values and required passages, but
a refused payment was separately represented as a recorded zero-valued award.
Adding an independently chosen absence-status development check exposed a 4/5
value result despite 5/5 original-passage coverage. The exact failed finding and
note reached an assistant with 17 full history entries. Its proposed draft passed
5/5 values and required passages on the unchanged suite, but retained a non-matching
quotation and other field concerns. No default was activated. This selected
single observation is not independent generalization or semantic certification.

For Laya data, keep source occurrence, financial/temporal stage, status and field
association as distinct labels. Include refusal, silence, explicit zero, partial
award, non-payment refusal, duration versus grant and repeated-event negatives.
Reviewer notes and model outputs are candidate examples, not automatic gold
labels. Preserve all necessary premises and the original-bound upstream inputs;
split contract families/templates before training or prompt selection. Measure
both quotation retrieval and interpretation, plus omissions outside configured
checks. Parent context is part of a training/evaluation example's identity.

Clause shortlisting and status/support triage remain promising experiments, to be
compared with the source-bound LLM baseline on necessary-premise recall, wrong
associations, false support, abstention, per-language calibration and complete
workflow latency. The pipeline still needs independent held-out contracts and
scoped graph precision/recall. Twenty-four extraction-model call records and one
assistant request were retained; 605 backend and 502 frontend tests/build passed.
The attempted narrow browser override did not reach its requested width and was
not counted as responsive verification. No Laya training, architecture or
acceleration changes occurred. See the
[readable-check audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-readable-checks.md).

### L53 — Operational readiness and graph structure are separate from model quality

A DelayLama live probe reported a nonexistent project's graph as ready using
1,426 database-wide nodes and 318 episodes, although its scoped statistics were
zero. Readiness now requires successful project statistics; failed or malformed
scope checks fall back to original documents. A populated service, completed job
or nonempty retrieval cache cannot establish usable context for a selected
contract, much less correct extraction or model accuracy.

The analysis map also treated nested settlement/payment facts as assessments and
attached them all to the overview. It now preserves their actual structural
parent, event/target context, own original quotes and exact review path. Shared
outcome names are not global identities. Inherited foreign scopes are excluded.
This containment is recorded analysis structure, not a newly established causal
or legal relation. Keep those relation classes distinct in training examples,
retrieval metadata, evaluators and reviewer interfaces.

Browser checks on the existing Birch case reached E12's EUR 400 award, its cost
assessment and its original highlighted passage, preserving the existing note
distinguishing incurred/requested/awarded amounts. The Aster source graph still
has five questionable causal edges and one date concern. These remain negative
development examples requiring independent source review, not automatic labels.
617 backend and 505 frontend tests/build passed; project/graph reads, 32 protected
files and the original user Studio session were unchanged. No new extraction,
Laya training, acceleration or semantic-quality experiment occurred. See the
[scope and hierarchy audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-graph-hierarchy.md).

### L54 — Source-backed relations still need roles, premises and occurrence preservation

DelayLama's live synthetic graph evaluations found correct extracted events
merged into their companies during graph storage. Organisation cues in an event
title (including “insurance”) could trigger that demotion; the related-party
fallback then silently lost the occurrence. The compatibility fix preserves a
dated, uniquely original-backed model event candidate with actor/action support,
while bare party identities still demote. This preserves a candidate rather than
certifying its semantics. Test extraction, storage and presentation separately.

The engine also promoted event-to-concept associations to causal edges using
delay/impact keywords, mixing affected and responsible parties. Those links now
remain associations. A separate bounded LLM pass uses explicit endpoint references,
role/type guards and unique original quotations for directed proposals. The graph
inspector compares the relationship assertion and both endpoint passages; source
presence does not establish that all necessary premises are present.

Four development cases cover English access restriction, French crane omission,
a notice explicitly causing no delay, and a stop-work order. The final candidate
preserved the notice and recorded its recipient without a causal edge; each delay
case retained a cause-to-effect candidate. Responsible/affected role coverage is
still incomplete, one stop-order proposal was withheld, and earlier attempts
failed to preserve the insurance notice. Short causal quotes can omit endpoint
identities; the Aster notice quote omits its recipient, requiring additional
premises. Duplicate ID/description event nodes also remain. These are targeted
synthetic development checks, not independent held-out accuracy estimates.

For Laya examples, label occurrence identity, endpoint roles, causal direction,
source-span presence and necessary-premise sufficiency separately. Include
negated causation, report-versus-cause notices, action titles with company/insurance
words, partial aliases, duplicate occurrences and missing role links. A successful
job or quote lookup is not a positive support label. Keep full source-bound inputs
and contract-family splits; compare any clause shortlist or validation head against
the complete LLM workflow on omissions, false support, abstention and latency.
No Laya training, architecture or acceleration change occurred in this pass.
See the [relationship audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-event-relationships.md)
and [quality priorities](https://github.com/UnicoLab/delaylama/blob/main/docs/contract-quality-roadmap.md).

### L55 — Actor-free occurrences can disappear after correct extraction

A further live DelayLama development case extracted three dated events correctly,
but graph storage merged “Foundation work started one day late” into its related
contractor. The organization-word guard treated “foundation” as an organization,
and the quoted passage did not name an actor. The date was transferred to the
party, removing the endpoint needed for the causal link. Assess extraction,
identity/storage and retrieval separately: a correct model output can still be
lost before review.

A failing deterministic pipeline regression reproduces that party-date transfer.
The storage safeguard retains a dated, uniquely source-matched model candidate
with a verbatim title of at least four words after excluding registered actor
identities. This heuristic preserves a reviewable candidate; it does not certify
an occurrence or generalize to every short/paraphrased title. One post-fix
synthetic replay retained all three events, left the companies undated, and
retained permit-failure → late-start causation without treating the unrelated
notice as a delay cause. The responsible-for relationship remained absent.
The unsuccessful first graph and both outputs were retained. This is selected
development evidence, not an independent accuracy estimate.

For Laya evaluation, include actor-free occurrences, generic organization words
used for activities, bare long party names, paraphrased/short action titles,
related-party date contamination and required role links. Separate candidate
retention from truth/entailment and required-premise recall. A causal span that
mentions an event identifier can need a separate endpoint passage to resolve
its actor. Missing graph edges must not automatically become negative labels,
and matched quotes must not automatically become positive support labels.

Operational follow-up also fixed exact project/tenant job scoping and queued
cancellation before capacity becomes available, with queue-inclusive expiry and
race tests. Service-wide health counts support safe activation without exposing
job results. Cancellation can leave partial graph writes. These are lifecycle
properties, not extraction accuracy or speed improvements. The evidence-first
UI now compares a relationship and both endpoint passages before showing its
connection list; the diagram is optional. Full Go tests, focused race tests,
clean-release regressions, 631 backend and 514 frontend tests/build passed.
No Laya training, MLX or architecture change occurred in this pass. See the
[evidence-workspace audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-evidence-workspace.md).

### L56 — Source recovery is separate from graph completeness and semantic support

Three live DelayLama source rechecks used the existing ORCHID-96 development
case. A targeted role search recovered Willow as responsible and Granite as
affected with exact original quotes, although the responsible-for graph edge was
absent. A notice search left cause/effect values unresolved while retaining a
negative interruption quote. A causal connection search checked the relationship,
both endpoint facts and its assertion separately. No graph rows changed.
These selected synthetic searches are not independent quality certification.

The notice's milestone category received model support from an event/date quote
that does not establish that category. Exact text presence is insufficient for
classification entailment. Also, an unresolved effect value coexisted with an
explicit negative-effect quote. Label source presence, role/value association,
category policy, negative assertions and necessary-premise sufficiency separately;
do not train directly on model verdicts or treat unresolved searches as absence.
For evaluation, retain this mistaken category judgment alongside successful role
recovery and assess taxonomy-guided inference against literal fact grounding.

Each search now has a durable job and stable request identity, with scope guards,
source/finding hashes, reload recovery and a selectable human-readable prompt-lab
handoff. Lost response recovery must read/reuse the existing handle rather than
silently launch another search. Review feedback does not automatically become a
prompt edit, graph correction or training example. 634 backend and 518 frontend
tests plus the production build passed. No Laya training, MLX or architecture
change occurred. See the [graph-recheck audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-graph-rechecks.md).

### L57 — Prompt feedback can improve one field while regressing explicit negatives

DelayLama now imports a scoped graph search, exact checked originals and bounded
search history into the chosen prompt session without replacing its draft or
pretending a graph check is an extraction run. On the ORCHID-96 development case,
one live baseline retained O1/O2/O3 but inferred zero actual notice cost. One
assistant proposal incorrectly called graph omissions extraction failures despite
baseline responsible-party values, and requested affected_party outside the target
schema. An advisory UI warning exposed the unknown field. The proposal was retained
and manually reviewed to use existing output fields before candidate evaluation.

The same-source candidate withdrew unsupported zero cost but also withdrew
explicit no-interruption and no-extension/payment-request consequences/time impact.
It retained an inferred administrative-notice cause, and shortened O2's citation
so its responsible/affected-role premises were absent from that quote. The
comparison exposed withdrawals; a correction note was saved on the candidate.
Both runs, the original proposal and the failed aspects survive reload. These
are selected development observations, not independent accuracy/latency evidence
or grounds for default prompt promotion.

For Laya data and validation, separate unknown, explicitly negative, zero actual
impact, no request/award, and absent graph connections. Measure negative-assertion
recall and necessary role-premise sufficiency as well as span presence and field
precision. Track the target output ontology: a graph role label is not automatically
an available extraction field or proof of a failed extraction. Assistant proposals
and model support verdicts require review before becoming labels; retain successful
and unsuccessful aspects of each candidate. The tested workflow now preserves
explicit source/finding provenance, baseline/candidate outputs and full Studio
review history, with no graph writes or default activation. 636 backend and 524
frontend tests/build passed. No Laya training, MLX or architecture changes occurred.
See the [handoff evaluation](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-graph-prompt-handoff.md).

### L58 — Structured correction can leave unsupported assertions in narrative output

A third ORCHID-96 DelayLama development run kept the same original and editable
intent, with strengthened automatic field guidance. It restored explicit
no-interruption time/consequences and no extension/payment request/award outcomes,
retained unknown actual cost, and included O2's responsible/affected role passage.
However, the notice cause still described its content/purpose; reasoning and the
summary asserted no financial impact without an incurred-cost premise. These
issues were manually flagged and preserved. O1's one-day time impact was withdrawn
while O2 retained its stated one-day delay. One selected synthetic rerun does not
establish independent accuracy, causal effect of guidance, or speed improvement.

The application now surfaces quoted-negative omissions on fresh and historical
results without replacing facts. Shared English/French cases distinguish unknown,
numeric zero, explicit absence, conditional/future statements, PDF line breaks,
and broad cross-event quotations. This narrow wording guard is review triage;
it does not certify meaning or discover all omitted negatives. 670 backend and
548 frontend tests plus the production build passed. Earlier runs/feedback,
original graph fingerprints and protected artifacts remained unchanged.

For Laya datasets, label structured values, rationale and summary assertions
separately. A correct empty cost field cannot make an unsupported “no financial
impact” sentence a positive label. Include cause-versus-purpose negatives,
necessary role premises, cross-event duration propagation and explicit absence
versus unrecorded/unknown information. Measure assertion-level false support and
negative recall, retaining mixed outcomes rather than declaring whole runs good
or bad. Human-reviewed spans/associations are prerequisites for training labels;
automatic guard silence and model verdicts are not substitutes. No new Laya
training, MLX or architecture change occurred in this pass. See the
[negative-outcomes evaluation](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-negative-outcomes.md).

## L59 — Narrative judgments need separate evidence and retained failures (2026-10-03)

DelayLama now reviews actual summary/rationale strings at independent result paths,
without adding those projections to graph facts. A correct unknown-cost field was
accompanied by a rationale asserting no financial impact. The source established
no interruption and no payment requested/awarded, not absence of incurred costs.
The first live narrative checker nevertheless returned supports. After narrative
guidance plus a narrow financial-premise safeguard, a deliberate recheck repeated
the model's supports judgment; the safeguard retained it as model_verdict while
marking the final result insufficient. The original failed response and human
correction remain available. A separate summary check was insufficient for the
wrong reason, illustrating that negative verdicts also need explanation review.

Notes for all three flagged findings reached the prompt-feedback panel and
survived reload with both rationale searches. No assistant call, new extraction,
default prompt activation or graph rewrite occurred. 690 backend tests, 569
frontend tests and the production build passed; these are application regressions,
not independent model accuracy evidence. Fourteen shared English/French examples
cover recognized financial wording; this is a narrow review aid, not a general
semantic validator. Protected artifacts and graph fingerprints remained unchanged.

For training, separate exact-span presence, actor/event association, each material
assertion, whole-paragraph sufficiency and explanation correctness. Preserve
failed positive and failed negative judgments as human-review candidates, not
automatic labels. Split by contract/template/story before synthetic expansion;
use held-out amendments, conditionals, absence/unknown, payment/cost and
cause/purpose counterexamples. First evaluate Laya as a clause shortlist or review
priority layer, measuring necessary-premise recall, abstention, full-source fallback
and complete workflow latency. No training, MLX or architecture change occurred
in this pass. See the [narrative review evaluation](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-narrative-review.md).

## L60 — Readable stage separation without stronger model claims (2026-10-03)

DelayLama's overview now separates observed time impact, contract dates, assessed
entitlement and recorded extension grants; incurred costs, contract price,
retention, payment requests and awards remain separate. Empty output is displayed
as not returned, rather than inferred zero/absence. Overview navigation uses actual
result paths, and financial/time focus exposes the relevant fields before
secondary details. Flagged summaries and reasoning join the review queue without
being promoted to graph facts. Live retained event and contract-identity checks
verified navigation and display; 577 frontend tests/build passed. A passive
389-CSS-pixel resize preserved the financial view without horizontal overflow.
One active click during a viewport override reached an unexpected view; its cause
was not established. Narrow screenshot capture returned blank pixels, and a second
documented capture returned a clipped view inconsistent with the DOM width. Only
DOM geometry is verified; visual/mobile-interaction certification remains open.

This is a usability pass, not improved model accuracy or a training experiment.
Original results, drafts, protected sessions and graph fingerprints stayed
unchanged; no source-search, extraction or assistant call was made. Training
examples should retain unknown, not-returned, explicit absence and contradictory
status/value records as distinct annotations. Do not turn easier navigation,
completed review counts or quiet source guards into positive semantic labels.
Continue measuring necessary-premise recall, stage/role association and false
support on independent contract families before introducing shortlist models.
No Laya training, MLX or architecture change occurred. See the
[results-overview evaluation](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-results-overview.md).

## L61 — Endpoint-shaped evaluation and source recovery reveal different gaps (2026-10-03)

DelayLama's event inspector now shows recorded actor/performer, affected party,
cause, effect and notice recipient with direct original-evidence navigation.
An absent slot describes the loaded graph, not the source document. A live role
search recovered the explicitly responsible client and affected contractor while
leaving graph facts and prompt drafts unchanged. A source-only candidate can be
useful even when graph extraction omitted the role; it is not an accepted label.

Five declared English/French development originals compared the existing
relationship request with optional direct prompt execution plus a per-event role
audit. Gemini's Summarize wrapper prepends a request to summarize concisely. With
readable references, two baseline runs matched 10/19 and 9/19 expected edges;
both candidate runs matched 14/19. A baseline quotation and fact named the correct
contractor but selected the client's valid reference: literal matching and typed
endpoints did not detect the wrong identity. The candidate avoided that binding
in these samples, but retained actor/affected omissions. The two changes were
tested together, so neither individual effect is established.

Replacing reference labels with production-shaped opaque node numbers, on the
same originals, yielded 10/19 baseline versus 11/19 candidate matches. This is a
format diagnostic, not additional held-out data. An isolated real-engine HTTP
pipeline still stored only two of three Orchid events: the paraphrased actor-free
foundation title bypassed the existing occurrence safeguard and was merged away.
Responsibility and the expected causal edge remained absent. The weather case
also retained a date-named Event duplicate. These pipeline runs were not paired
end-to-end comparisons. The experimental patch is explicit opt-in, excluded from
default builds and was not activated on the user's running engine. Failed and
mixed results are retained in the
[role evaluation](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-graph-role-review.md).

For Laya/contrastive examples, include same-quote/fact pairs with swapped valid
actor references, reordered opaque registers, paraphrased occurrence titles,
actor-free passages, known actor aliases, date-anchor duplicates and notice
performer versus delay responsibility. Evaluate occurrence preservation, endpoint
association, role completeness, legal interpretation and necessary-premise recall
separately. A supplied gold endpoint register can hide upstream losses; compare
against actual extracted/stored registers and full workflow latency. Split by
contract/story/template before expansion and do not turn graph source matches,
model-supported roles or quiet guards into training positives. Human-reviewed
span/association labels remain necessary. 581 frontend tests/build, full clean Go
tests and patch compatibility checks passed; these are operational evidence.
No Laya training, MLX, GLiNER or architecture change occurred in this pass.

## L62 — Storage identity must be evaluated independently of extraction (2026-10-03)

DelayLama's paraphrased foundation event was lost after extraction because the
type guard and merge helper treated a related contractor as the event's identity.
A controlled original-source replay reproduced the failure before the change:
the contractor inherited the foundation date, and a bare actor in `events[]`
could inherit another event date. Merging now requires an unambiguous normalized
full identity rather than a related mention or substring. Actor mentions stay
undated; historical records were not rewritten. An original-linked dated
paraphrase can survive a conflicting actor keyword as a classification-review
candidate. This lexical heuristic is not semantic verification and can retain
false events; alias and date-anchor duplicates remain separate concerns.

The default fifth compatibility patch was reproduced on clean Chronograph v1.3.0,
then activated after idle checks and binary/database backups. Existing graph rows
and protected sources/sessions remained unchanged. Isolated real configured model
runs retained all three Orchid event passages with their dates and timeline
entries; Fir's notice-only control retained its recipient without causal edges.
One fresh synthetic Poplar variation was analyzed once through the running GUI:
all three dated events survived, actors stayed undated, and permit → foundation
had both original endpoint passages. Its job report showed three proposed/stored
events, two classification reviews and no storage failures. The GUI exposes these
operational counts separately from completeness, with highlighted original context.
Explicit client responsibility was still omitted, and a malformed `OPLAR-74`
document candidate appeared beside `POPLAR-74`. These are retained failures.
The real calls were integration acceptance, not paired independent accuracy trials.

For reusable training/evaluation, annotate occurrence identity, actor mention,
date role, span match, endpoint binding and semantic support separately. Include
paraphrased actor-free occurrences, a bare actor returned as an event, wrong
related-party order, ambiguous normalized identities, aliases, malformed IDs,
date-anchor duplicates and unrelated notice controls. A correct model event list
can still fail storage/retrieval, so measure end-to-end preservation alongside
false candidate retention and necessary-premise recall. Lexical source matches,
stored-event counts and review warnings are not automatic positive Laya labels.
Use human-reviewed spans/associations and independent contract/story/template
splits. Establish clause-shortlist recall and total workflow latency benefit
before narrowing the full-source LLM's input.

Full Go tests, agents/API race tests, 697 backend tests, 583 frontend tests and the
production build passed. Normal desktop screenshots were inspected; inconsistent
full-page capture geometry was discarded. No mobile certification, new Gantt view,
Laya training, MLX acceleration, GLiNER integration or architecture change occurred
in this pass. See the
[occurrence identity validation](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-03-occurrence-identity.md)
for operational receipts, failed-regression evidence and remaining limitations.

## L63 — Chronology position must retain date uncertainty (2026-10-04)

DelayLama now presents the active contract's loaded source event candidates as
recorded-day markers, labelled periods and an unresolved-date review list. Live
Poplar checks retained three original-backed day markers; Lumen's month-only rain
became a June 2026 period rather than an invented first-day occurrence or a
month-long interruption. Original excerpts, role gaps and both causal endpoint
passages remain one selection away. An actual CSV export matched the live graph's
three Poplar IDs/titles/dates. The browser's download observation timed out; the
fresh file itself confirmed completion, so the operation was not resubmitted.

For training/evaluation, distinguish a recorded day, calendar period, estimated
anchor, disputed date, ingestion timestamp and event duration. A period boundary
can organize a chart without establishing occurrence, chronological precedence,
notice trigger or a causal relationship. Include contradictory source date text,
reversed range bounds, missing/absent event quotes, date-only Event labels and
party dates as negatives. Exact text presence and plot position are not accepted
date-role labels. Unrecognized period expressions remain unresolved rather than
being guessed; evaluate date parsing by language and input format independently.

The UI separates source graph chronology from saved-analysis timelines and shows
pagination limits. Export retains unresolved entries and review metadata. All
graph tables and protected sessions/artifacts stayed unchanged during this pass;
no new analysis, source recheck or model promotion was submitted. 597 frontend
tests in 49 files and the production build passed. Keyboard selection, return
focus and actual original evidence were checked live. Mobile certification remains
open, and mixed-export Fast Refresh remounts during frontend code edits remain a
development UX follow-up. No Laya training, MLX acceleration, GLiNER integration
or architecture change occurred. See the
[source chronology validation](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-graph-chronology.md).

## L64 — Complete role cards improve coverage, not semantic certification (2026-10-04)

DelayLama's free-form relationship pass omitted explicit performers. A separate
opt-in experiment requests one card for every stored event, with performer,
affected, causal and recipient outcomes, exact registered names/references,
original quotes and explicit unresolved/not-applicable responses. Redundant
name/reference agreement rejects inconsistent joins; a consistently wrong role
can still pass it. Original matching does not certify an endpoint's meaning.

Six invented English/French stories were declared before model calls, with each
opaque register also reversed/renumbered. The baseline matched 18/30 declared
links; the first candidate matched 29/30 and the final candidate 28/30, each with
zero unexpected declared-set links. Two final proposals failed original quotation
matching. The variants differ slightly in prompt/type guards and diagnostics;
these are not isolated model-variance estimates. A paired full original → engine
extraction → storage → scoped REST trial matched 9/15 baseline links and 15/15
candidate links across the six stories, retaining each dated occurrence. Neither
register permutations nor paired runs add independent contract families.

For Laya training/evaluation, model cardinality and explicit unknown responses
should expose omitted roles; they must not become correctness labels. Include
wrong-but-valid refs, swapped party roles, register permutations, weather risk
versus physical causation, notice recipient versus sender, negation and unknown
roles. Evaluate exact event identity and necessary role premises separately from
literal quote presence. Indexed original spans are a useful next experiment for
transcription reliability; measure semantic false support independently.

The shipped GUI now selects one role's original-source search, including notice
recipient, keeps keyboard focus during submission and distinguishes a selected
saved search from the next draft's fields. One live recipient search returned an
original-matched candidate; an affected-party search stayed unresolved. Their graph
snapshots were unchanged, and reading older evidence did not submit another job.
600 frontend tests, 698 backend/SDK tests, the production build, full Go suite and
intelligence/agents race checks passed. The existing macOS temporary-path assertion
required a canonical TMPDIR for the full Go run; resolver behavior was unchanged.

The engine experiment remains excluded from default builds and the running engine.
Broader independent contract tests, long inputs, aliases, duplicate/false events
and consistent-but-wrong semantic associations remain open. User graph tables,
source files, protected sessions/artifacts were preserved; only the API was
refreshed for the new source-check field. No Laya training, MLX/GLiNER integration,
architecture change or automatic positive labels followed. See the
[role-register audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-role-register.md).

## L65 — Indexed evidence can still support the wrong direction (2026-10-04)

DelayLama tested an optional original-passage selection arm after L64. The model
chooses bounded consecutive IDs from the whole indexed nonblank original; exact
text and UTF-8 byte offsets are reconstructed locally. Invented, absent, reordered,
skipped, duplicate and oversized selections are rejected without quotation
fallback. Repeated wording can be located at a particular position. That position
still needs to be associated with the correct occurrence and role.

Eleven invented English/French originals, each with normal and reversed/renumbered
registers, produced 22 supplied-register development cases. Expectations were
frozen before model calls and not included in input. One sequential comparison
retained 51/60 declared links for quoted roles and 53/60 for indexed roles; both
matched 19/22 exact sets. Indexed selection accepted one reversed French causal
relationship despite real source text and consistent endpoint names/references.
It also withheld six repeated-notice proposals for nonconsecutive passage IDs.
The quoted variant withheld eight ambiguous repeated quotes and one mismatched
quote. Both exact-set quality gates failed; no production promotion followed.

Five additional full original → extraction → storage → scoped REST stories
retained 15/15 quoted links and 11/15 indexed links. These trials used the same
configured Gemini model with noop embeddings and separate upstream generations.
One French indexed run stored concept Event aliases `F1`/`F2` beside the fuller
occurrence nodes on the same dates/source passages. Its expected edge set matched,
but event cardinality failed. The role register covered two events while four
dated Event nodes existed. This duplication preceded passage selection and cannot
be attributed causally to the indexed arm from independent stochastic runs.

For Laya data and scoring, separate source location, occurrence identity, entity
binding, relation direction, whole-assertion support and human acceptance. A
source-matched model proposal is not an automatic positive label. Include sender
corrections, withdrawn statements, subsequent reporting without causation,
standing obligations without occurrences, repeated text and cross-paragraph
qualifications. Necessary-premise recall must include the identifying date/heading
and any negation or amendment, even when a local sentence looks sufficient.
Use independent contract-family splits; permutations add robustness checks, not
independent families. Preserve failed arms and raw suitable synthetic outputs.

Next register a bounded start/end-window or multiple source-located-span arm;
measure semantic false support separately from citation validity. Resolve
concept/occurrence identity with same-day distinct-event negatives before training
on graph-derived labels. Compare full extraction/review latency rather than
assuming token reductions improve the workflow. No Laya model training, MLX,
GLiNER or architecture changes occurred in this pass.

The graph GUI now preserves its selected contract, event and open connection
review during source edits to its two refactored components. Live corrected-notice
review displayed the reconstructed original paragraph with sender/recipient
context and retained model-suggestion status. 600 frontend tests, 704 backend/SDK
tests, the build, full Go suite, race checks and clean patch reproduction passed;
the experiment quality failures remain visible. User graph/session/source hashes
stayed unchanged. See the
[indexed-source audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-source-passages.md).

## L66 — Occurrence identity must precede graph-derived training labels (2026-10-04)

DelayLama's recorded French extraction had short Event concepts `F1`/`F2` beside
full descriptions of the same two source occurrences. SQLite replay produced four
dated Event nodes despite an otherwise matching expected relationship set. A
narrow default repair now enriches the existing concept when one explicit labelled
dated heading, a uniquely located shared original quote, the same exact day and
one matching Event concept agree. Existing IDs and relationship endpoints survive;
self-associations are excluded. It does not establish semantic correctness.

The captured replay now stores two occurrences with readable F1/F2 identifiers.
Filedb and SQLite regression tests fail before the change with four instead of
two and pass after it, including two distinct events on the same day. Actor,
related-name, missing/inferred/partial-date, repeated-quote, multi-heading and
duplicate-candidate cases withhold binding. Foreign project/tenant and historical
episode rows cannot be reused. An injected update failure remains reported and
does not create a duplicate replacement. Existing graph rows are not migrated.

Four frozen synthetic originals were also tested through fresh configured Gemini
extraction, actual storage and scoped REST before/after: eight jobs completed and
passed the declared event-count/day, literal-quote, scope and self-link checks.
Neither fresh arm emitted the short-identifier duplicate pattern, so these runs
show observed regression checks rather than a measured fresh extraction gain.
The replay contains normalized properties from a recorded generation; it is not
an independent gold-labelled extraction trial. Noop embeddings were used in the
fresh comparison. Role completeness and semantic direction remain unresolved.

For reusable Laya data, separate source span location, occurrence identity, actor
identity, role association, relation direction and human acceptance. Include
same-day distinct events, repeated headings/quotes and cross-paragraph corrections
as negatives. Deduplicate verified occurrence identities before sampling
graph-derived training examples, while preserving original evidence and rejected
bindings. Never promote successful storage or exact quotation into automatic
positive semantic labels. Report independent contract-family evaluation separately
from replay, register permutations and synthetic fixture cardinality.

The replay also exposed a malformed organization name `F1.\nArdoise SARL` from
cross-line source matching. Study source-label/actor boundaries without discarding
legitimate multiline company names. This is separate from occurrence binding and
remains open. Readable identifiers now support graph/chronology navigation and
CSV export without displaying internal vectors or identity-basis fields.

602 frontend tests, 704 backend/SDK tests, the production build, full Go suite,
agents/intelligence race checks and clean default/optional patch compatibility
passed. The activated default engine preserved every existing graph row,
configuration, protected source/artifacts and checked Studio sessions. No Laya
training, MLX, GLiNER or architecture change occurred in this pass. See the
[occurrence identity audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-occurrence-alias.md)
for frozen fixtures, replay tooling, limits and operational evidence.

## L67 — Preserve source layout separately from readable entity identity (2026-10-04)

DelayLama's deterministic legal-form supplement could emit `F1.\nArdoise SARL`
as an organization. The matcher treated a sentence-ending reference and the next
line's company as one capitalized name. A derived recorded-generation replay
removed exactly that previous supplement from model input so both engines could
regenerate their own deterministic additions. Actual SQLite output changed from
three organizations to the two source parties, retaining two event candidates.
This isolates supplement behavior; it is not raw upstream or independent gold
extraction. Filedb and SQLite SDK regression tests fail before and pass after.

The repair scans bounded tokens backwards from company endings. Soft wraps stay
inside company names; sentence/paragraph boundaries, another legal form and
explicit reference/known heading lines stop prefix binding. Dotted French forms,
initials, accents, apostrophes, ampersand companies and nonbreaking spaces are
tested. Names normalize whitespace for readable identity/deduplication; original
quotes and UTF-8 byte offsets remain unchanged. Ambiguous adjacent headings,
broader international forms and general entity aliases remain open.

For Laya span/identity data, save the raw document, raw span text/offsets and
normalized display identity separately. Generate CRLF, soft-wrap, nonbreaking
space, heading/reference, paragraph and page-break variants with transformation
maps back to the original span labels. Keep layout variants of a contract in the
same train/evaluation family. Include adjacent distinct companies, initials,
dotted suffixes and repeated wrapped/unwrapped mentions. Never silently train
on a display-normalized string as though its character positions were original.
Go byte offsets and JavaScript string indices need an explicit conversion or
original-text lookup policy; source presence still does not certify role meaning.

Five frozen French/English originals produced ten fresh configured Gemini jobs
with noop embeddings and full extraction/storage/scoped REST. Both arms returned
the expected party sets; fresh runs did not reproduce the malformed organization.
One baseline wrapped-name generation had an extra dated event. Four of five
baseline runs and five of five candidate runs passed all declared party/day,
scope, whitespace-normalized quote presence and self-link checks. Independent
upstream generations do not attribute that event-count difference to this repair.
Role omissions, semantic direction and real-contract recall remain unscored.

The initial bounded-window implementation allocated about 37 MB per extraction
on a synthetic 203 KB repeated-source workload. Direct token scanning reduced it
to about 1.7–1.8 MB while preserving regex output on all five frozen originals;
the prior baseline allocated about 1.1 MB. Timing was noisy on a shared host, so
there is no model/workflow speed or resident-memory claim. Retain unsuccessful
implementation measurements and distinguish allocations from peak memory.

Live source review exposed a separate UI hover/focus defect. Fact-specific styles
now keep readable text and visible keyboard focus in both themes; checked labels
exceed 4.5:1 contrast. The final Go suite, agents/intelligence race checks, clean
seven-default reproduction, optional patch compatibility, 13 CLI tests and the
frontend build passed. All user graph rows/configuration, protected files and
checked Studio sessions survived activation. QA graph comparisons after restart
must compare node/edge records by ID: a list-order comparison failed despite
unchanged records. No Laya training, MLX, GLiNER or architecture change occurred.
See the [source boundary audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-organization-boundaries.md)
for fixtures, measurements, GUI proof, remaining limits and activation evidence.

## L68 — Keep review attempts and labels bound to their own source (2026-10-04)

DelayLama's saved-search UI could replace a historical result with the latest
search on refresh or with a slower asynchronous response after another selection.
Selection now retains the job ID, cancels obsolete observations and preserves the
historical result while polling an active search. Human-readable history cards
retain scope/status/date; importing uses the selected checked originals and
bounded history. A client guard blocks new submissions when that history contains
active work. It is not a cross-client/server concurrency guarantee.

For Laya datasets, bind reviewer decisions to the document hash, finding identity,
actual checked run and prompt version. Do not turn all search suggestions or
iteration notes into supervised labels. Quote presence, semantic role/direction,
missing information and explicit human acceptance need separate annotations.
Repeated recovery attempts on one contract are correlated observations, not
independent evaluation examples. Historical context may improve prompt proposals;
its inclusion alone does not establish better extraction or justify input pruning.

Verification used 605 frontend tests, 15 backend source-search tests, a production
build and native browser navigation over real scoped HTTP endpoints. Three saved
search records were controlled fixtures on a copied actual extraction graph;
there were no new LLM or training calls. Selected-source handoff and unchanged
graph tables were checked. These are workflow results, not semantic model scores.
The next auxiliary-model comparison still needs independent contract families,
necessary-clause/amendment recall, false support and end-to-end latency, with a
full-source fallback. See the [review audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-source-search-history.md).

## L69 — Joint relationship labels need occurrence and direction checks (2026-10-04)

DelayLama's source search now checks the source endpoint, target endpoint and
exact directed relationship separately. Endpoint dates and source context join
the finding hash; embeddings do not. A model that silently reverses/corrects the
requested IDs cannot verify the reviewed connection, even when its explanation
and quotations sound plausible. Missing/duplicate bindings remain unresolved
with at most one existing format repair. Risk allocation is not event performance,
a later reporting notice is not its reported cause, and endpoint existence does
not establish a link.

In a paired 12-case synthetic development comparison, baseline and initial
candidate each returned 10/12 exact verdicts. The candidate removed the baseline's
one false-support case and retained all six supported positives, but both confused
unknown information with contradiction. Final aggregation and narrow EN/FR
all-epistemic quotation guards replayed the checked candidate records at 12/12;
this was not fresh model generation, raw-binding revalidation or independent
certification. Baseline/candidate recorded 12/14 calls respectively. Retain failed
judgments and distinguish model verdicts from source/binding checks and human
acceptance; do not turn guarded suggestions into training ground truth.

Two real saved searches on a copied extraction graph checked a correct causal
edge and one controlled reverse. The correct edge received candidate support;
the reverse remained insufficient because the model changed its requested
binding even after one repair. Native GUI and live Prompt Studio handoff preserved
the original document, both input history entries and the existing draft. The
assistant proposed causal-direction instructions without activation or an
extraction rerun. Saved review uncertainty still needs propagation into graph
role/timeline cards, and missing actor/affected roles remain extraction work.

For Laya data, annotate exact original spans, occurrence/date identity, role
association, directed whole-assertion support, unknown versus explicit denial,
and human acceptance separately. Evaluate on held-out contract/source families;
positive/negative variants and repeated rechecks are correlated. Auxiliary clause
shortlists need necessary-premise/amendment recall and a full-source fallback.
No Laya training, GLiNER, MLX, supervisor or architecture change occurred here.
See the [connection audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-connection-verification.md)
for fixtures, scores, operational checks, GUI proof and remaining limits.

## L70 — Review visibility and scope must survive retries (2026-10-04)

DelayLama now propagates saved source-review concerns into roles, graph connections,
fact lists, chronology and CSV notes. A later failed/pending attempt retains the
previous completed judgment; a newer finding-only search cannot clear a separate
role-search concern. Keep the newest completed record for each requested field
scope and inspect the actual unresolved job, rather than substituting the newest
result. Date/source/direction hashes and current original quotations guard display
support. A partial snapshot makes endpoint verification unavailable, not changed.

For reusable Laya datasets, preserve document/finding/prompt/run/field-scope identity,
model suggestions, exact source checks, unresolved outcomes and explicit reviewer
acceptance separately. Failed retries do not create negative semantic labels, and
an unknown relationship is not a contradiction. Broader or narrower search scopes
are not interchangeable evaluation examples. A fresh UI, source match and stored
review history cannot certify extraction completeness or independent correctness.

The live workflow reused two prior saved LLM searches on a copied synthetic graph
and added one declared failed retry with no model call. Original jobs and all graph
rows stayed unchanged. Changing/restoring the registered original invalidated and
restored both summaries without rerunning models. Native browser checks confirmed
roles, focused findings, chronology, both themes and refresh recovery. An API-restart
failure exposed missing project/source recovery and empty-desk graph-view resets;
both were repaired. Final suites passed 742 backend/SDK and 618 frontend tests.
No Laya training, GLiNER, MLX, supervisor or architecture change occurred. See the
[review propagation audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-source-review-projection.md).

## L71 — Correct task transport does not establish extraction completeness (2026-10-04)

DelayLama isolated a provider transport change from prompt/schema changes. Its
relation pass previously sent JSON extraction instructions through Gemini's
`Summarize`, which prepended a concise-summary instruction. Optional direct
`Complete` now preserves the task with the legacy interface fallback, the same
45-second bound, exact original matching and no second request after failure.
A wire-level provider test verifies the extraction prompt without a summary prefix.

On 18 frozen English/French development cases with supplied opaque endpoint
registers, baseline/direct retained 28/46 and 29/46 declared links respectively,
with zero unexpected links and 8/18 exact sets each. Both strict gates failed.
Four fresh full-source extraction/storage/REST pipelines per variant retained
5/9 links and 2/4 exact sets each. These are single paired observations with
correlated register permutations, not independent certification or a measured
completeness/speed gain. The running user engine was preserved; the transport
patch is in future default build preparation. Experimental role prompts remain
opt-in with their prior failed gates retained.

Raw responses exposed a specific French causal quotation failure: the model
inserted `[...]` into a required contiguous original quote. Exact matching
correctly withheld it. Actor/performance omissions were mostly unproposed,
not lost at storage. Training/evaluation should distinguish unproposed roles,
invalid quotations, wrong endpoint associations, storage failures and legitimate
unknowns. Never promote a copied span, empty output or successful job into a
semantic ground-truth label; event performance also differs from legal liability.

The GUI pass fixed Open Graph opening Claim Desk and source-analysis availability
remaining stale when originals became readable. Scoped navigation is consumed
once, refresh respects manual view selection, and changing source text observes
existing status without resubmission. Actual new F1/F2 graph evidence still shows
missing roles. All QA node/edge rows stayed unchanged through browser review.
622 frontend tests, build, Go/race and patch-reproduction checks passed. No Laya
training, GLiNER, MLX, supervisor or architectural change occurred here.

Next compare complete per-event role coverage on independent contract families,
then evaluate auxiliary clause/role triage with necessary-premise/amendment recall,
false support, latency and full-source fallback. Transport correctness and better
navigation are useful engineering fixes, but neither proves model quality. See
the [direct-execution audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-direct-relations.md).

## L72 — Role schema direction, occurrence labels and truthful coverage (2026-10-04)

DelayLama's event-role development experiment recovered all declared directed
sets in 17 final synthetic full-source runs (33 + 18 links), with zero unexpected
links. This does not certify extraction completeness or real contracts: the
sources are short English/French inventions and one run still created a second
Event whose entire name was “14 January 2032”. Its failed node-identity check is
retained. A date-value guard, deterministic SQLite/file-storage regressions and
a subsequent focused source replay address that specific failure. A new real
browser/API job also preserved two independent same-day occurrences.

Four separate contract-format sources were frozen before requests and excluded
from prompt tuning, but were not externally adjudicated. The register experiment
reverses/renumbers the same endpoints, so its permutations are correlated examples.
The original gold missed the performer of an observed proposal. Initial scores
and labels remain intact; final gold distinguishes the actual proposal from the
never-sent notice. Training labels must identify which action is actually observed,
including proposals/decisions, rather than equating “proposed to send” with a sent
notice or suppressing the performer's real proposal action.

A raw French response described L1 causing L2 but placed L2 in L1's ambiguous
`causes` slot. The decoder interpreted that as an incoming cause and reversed it.
An explicit `caused_by` schema and instruction fixed this particular direction
contract. Raw flat-prompt output also linked a valid wrong party reference.
Reference/name consistency and verbatim passages reduce mechanical errors;
they cannot certify that the named parties satisfy the asserted semantic roles.

The final proposal-aware read-only development pass retained 66/66 declared links
and 28/28 exact sets. The direct baseline retained 39/64 with two unexpected links
on the earlier 26 cases. These differing label versions are not a paired quality
percentage. Full-source saved baseline rows can be transparently rescored on the
33-link reference (17/33, one unexpected) without erasing the 32-link original.
Keep annotation revisions, model outputs, cohort hashes and failed gates visible.

Laya data should separate performance, legal liability, affected parties,
conditional duties, actual decisions, unsent/negated notices and incoming causal
identity. Hard negatives should swap actors, distinguish reports from causes,
and retain independent same-day events. A calendar value is not itself an action.
Score span support, endpoint association, direction, event class and completeness
separately. This pass did not retrain Laya, implement another architecture, change
MLX or the supervisor, or validate GLiNER. These are candidate data/benchmark
improvements, not claims that the current contract head handles them.

DelayLama now shows bounded coverage of registered candidates, combining
unresolved/not-applicable responses honestly instead of declaring facts absent.
Completed checks collapse; actor labels and contextual source highlights remain
readable. The real browser job's six directed roles survived persistence and
review without resubmission. Its closure was still classified as milestone and
its notice as decision: positive role scores do not qualify event classification.
The revised role prompt remains opt-in; the active user engine uses tested default
transport/date fixes without the new role prompt. Existing graphs were preserved.

753 backend tests, 626 frontend tests, build, full Go and extraction/storage race
checks passed. The Mac path test's root assertion was canonicalized without
changing runtime confinement; missing model keys now fail before replacing an
environment. See the [source audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-role-meaning.md)
and [shared receipt](validation/delaylama-role-meaning-20261004.json).

## L73 — Preserve navigation without transferring evidence decisions (2026-10-04)

DelayLama's graph reopened the claim desk after reload and carried graph layout
between contracts. Project/workspace/agent navigation now has separate scopes.
Graph layer, layout, finding and connection identifiers are remembered; source
and saved-analysis selections are independent. Delayed data never overwrites a
manual preference. A missing anchor cannot attach its saved connection to the
fallback finding. Navigation restoration neither accepts evidence nor launches
a new model job.

This matters for Laya-assisted review too: a shortlist, high score or restored
selection is not an adjudicated finding. Keep original spans, task semantics,
role association, model candidates and review decisions distinct. Store only UI
identifiers in navigation preferences; exact source/result identities govern
review notes and verdicts. The EvidenceBoard's exact selected finding/citation
position remains a separate, unimplemented resume improvement in this pass.

636 frontend tests and the build pass. Actual browser reloads used a copied
previously generated source graph and identical original, retaining the delivery
event, actor connection and highlighted source sentence. No model jobs were
submitted; node/edge hashes and saved terminal history were unchanged. User
state was preserved. Desktop layout was checked at 1910 × 1075. The viewport
provider scaled a requested 390 width to 582 CSS pixels; no phone-width
certification is claimed. No model quality, Laya training, MLX or supervisor
change was measured here. Existing event-class concerns remain unresolved.

See the [navigation audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-review-navigation.md)
and [shared receipt](validation/delaylama-review-navigation-20261004.json).

## L74 — A restored quote index must bind to the same evidence (2026-10-04)

The L73 EvidenceBoard resume item is now implemented. DelayLama binds the selected
finding and quote position to native SHA-256 of the complete canonical result,
source, review scope and agent. Preferences retain identifiers/digests only.
Runtime metadata and review notes do not change result identity. A changed
recovered citation list resets the index instead of showing a different quote
under the remembered position. Narrative rows resume separately from factual rows.

Explicit inspection requests are consumed after focus; later manual selection
wins when reopening Review. The timeline retains a separate exact-context anchor.
Late restore and digest-write races cannot replace a newer manual choice. Warm
citation digests support synchronous writes before reload. Unsupported browser
crypto/storage keeps review usable in the tab without a weak hash fallback.
The initial fallback may briefly display a restoration hint; filters remain
unpersisted. Position restoration does not accept evidence or send model jobs.

653 frontend tests (including seventeen new resume/race cases) and the production
build pass. Browser tests used copied original Aster synthetic-contract analyses:
deadline/quote three survived reload; price/quote two survived reopening after an
old explicit request, agent changes and final reload. Existing source/extraction
bytes and graph hashes remained unchanged; no QA model jobs were submitted.
No mobile, new extraction quality, Laya training or acceleration result is implied.
The same distinction applies to Laya candidates: navigating back to a scored span
must not turn it into a reviewed association or silently substitute different
support. Treat data identity and human decisions independently.

See the [resume audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-review-resume.md)
and [shared receipt](validation/delaylama-review-resume-20261004.json).

## L75 — A vocabulary observation does not identify a queued model run (2026-10-04)

DelayLama’s source graph uses the observed general vocabulary: milestone, delay,
decision, change and deadline. A previously retained notice/decision error must
be assessed against these permitted labels before concluding that better role
links or a new encoder solve event meaning. The contract-claims preset includes
more specific labels, but remains an unmeasured candidate. Exact quotations and
performer links cannot certify classification.

New analysis runs retain bounded pre-submission vocabulary observations and a
canonical schema digest without private prompt text. The GUI shows the current
vocabulary, absent standard notice/suspension labels, legacy unknown metadata
and changed configuration digests. Schema-read failures do not resubmit work.
780 backend tests, 661 frontend tests and a production build pass; the isolated
browser used existing actual-model synthetic-contract results, submitted no new
model jobs and preserved source/result bytes and graph fingerprints.

The engine’s schema remains shared and mutable. This observation does not prove
which settings a queued worker used. Before model comparisons or new Laya label
training, freeze task configuration and separate actual run identity from current
service settings. Labels for notices, suspensions, duties and proposals need an
explicit policy and reviewed independent examples. Score occurrence identity,
classification, roles and spans separately. A schema expansion cannot be counted
as improved extraction without measured benefit; Laya must retain full-source
fallback until necessary-clause and amendment recall gates pass.

See the [vocabulary audit and work priorities](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-event-vocabulary.md)
and [shared receipt](validation/delaylama-event-vocabulary-20261004.json).
No new Laya training, GLiNER comparison, MLX measurement or accuracy certification
occurred in this pass.

## L76 — Correct model labels can still become the wrong graph objects (2026-10-04)

DelayLama now captures the domain schema and provider references before each
source-analysis worker waits for capacity. Startup profiles reach the actual
intelligence component, provider replacement preserves the schema, and durable
job receipts remain distinct from today's vocabulary observations. Legacy jobs
remain unknown; private prompt bodies stay out of metadata. Deep-copy, queued
worker, actual model-input, concurrent setter and race tests pass. 802 backend,
665 frontend tests, the production build and clean patch reproduction pass.

A fresh full-source Gemini trial on a synthetic French contract returned correct
`suspension` and `notice_served` labels, exact source spans and the standing duty
as a separate clause. Storage nevertheless classified the sent notice as a dated
`Suspension` node rather than an Event. The suspension performer and notice
performer/recipient directed links were missing. One AFFECTED link was retained;
INVOLVES associations do not fill the omitted roles. Source matching and correct
upstream labels cannot certify end-to-end identity, role completeness or meaning.

Changing the isolated engine to general after completion and reloading preserved
the captured Construction Claims receipt and graph; no second job was submitted.
The tested default engine was activated without changing the user's profile,
source edits, graph rows, artifacts or Studio sessions. Optional role prompts
remain evaluation candidates. The snapshot covers domain schema, not every
provider setting or hook; documents are captured separately and may use mixed
schemas. Engine handles remain in memory.

Before retraining Laya, define labels independently for occurrence, category,
standing duty, span, actor role and graph identity. Preserve this storage failure
as a regression fixture. Compare fixed full-source outputs before regenerating;
then score independent contract families and downstream completeness. Laya may
shortlist clauses or provide a second opinion only after necessary-clause and
amendment recall gates; keep full-source fallback. No new training, GLiNER
comparison, MLX timing or real-contract accuracy certification occurred here.

See the [schema audit and priorities](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-analysis-schema.md)
and [shared receipt](validation/delaylama-analysis-schema-20261004.json).

## L77 — Profile words must not erase source-dated occurrence identity (2026-10-04)

A retained correct `notice_served` label became a non-Event Suspension node through
an entity-word guard. The same captured output now replays into two Events instead
of one Event/one standing mention: explicit original date, unique quote, permitted
category and sufficiently specific source overlap retain a marked candidate.
Standing actors/values, unsupported categories, ambiguous quotes and inferred dates
retain their guards. Month-only dates remain periods. This storage exception is
not semantic proof that an action happened or a duty was fulfilled.

Full Go tests, changed-package race checks, 802 backend and 668 frontend tests,
a production build and clean twelve-patch reproduction pass. A fresh source-model
run shows two correct days and exact citations, but has a different notice title
using the older actor-word branch. It cannot establish the new branch's isolated
benefit; the fixed-input SQLite replay does. Performers remain omitted in the graph.
A targeted recheck recovers the notice performer separately and its history/source
opens in Prompt Lab, without automatic graph writes, model revision or activation.

“Unresolved details” now distinguishes pending review from evidence absence.
Located additional values remain candidates even when every stored field has
model support. Review status, source presence, candidate role values, graph identity
and human acceptance must be separate training labels. For Laya, benchmark both
upstream classification and downstream registration/role association; encoder
scores cannot repair storage guards or establish full-source completeness.
No Laya retraining, GLiNER comparison or acceleration measurement occurred here.

See the [occurrence/review audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-profile-occurrence.md)
and [shared receipt](validation/delaylama-profile-occurrence-20261004.json).

## L78 — Typed party integration must precede model-quality attribution (2026-10-04)

Construction Claims used Contractor/Maître d’ouvrage and sixteen other party types,
but the relation validator and GUI recognized only generic actors. Correct roles
were discarded or hidden; a party could also pass a causal-target guard. The new
explicit family preserves specialized types, enables performer/recipient links,
keeps standing parties undated and rejects nonparty recipients. Unknown labels,
clauses and risks remain excluded. No historical graph migration is automatic.

Same recorded optional output retained 5/14 declared links before versus 14/14
after this guard fix, with no model calls. Fresh default/optional Gemini passes
retained 9/14 and 14/14 across four new synthetic supplied-register cases. Expected
sets were not sent to the model; default omissions are retained. This is manually
annotated development evidence, not independent unseen-contract certification.
Optional meaning remains opt-in. Two different full-source/browser originals show
working typed role evidence and a remaining Planning revision-registration failure;
their counts cannot be presented as a paired prompt comparison.

All Go/focused-race, 802 backend/671 frontend, build and thirteen-patch reproduction
checks pass. The default runtime was updated while preserving graph/configuration,
source edits and user artifacts. GUI candidates keep original quotations, endpoint
passages and performance/legality distinctions. Source presence, party role, stored
identity and human acceptance must be separate labels for Laya/GLiNER evaluations.
An encoder cannot fix downstream type guards. Prioritize occurrence registration,
independent role/false-link coverage and version-bound reviewed apply/undo before
using classification scores as quality claims. No new model training or timings.

See the [party integration audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-domain-actors.md)
and [shared receipt](validation/delaylama-domain-actors-20261004.json).

## L79 — Occurrence identity and source grounding need independent gates (2026-10-04)

The Planning revision failure retained in L78 was reproduced with the exact
recorded Gemini extraction and synthetic original. A downstream entity-word guard
had demoted a completed, source-dated schedule revision. A bounded new default
exception retains a marked Event when a unique original quote, explicit date,
allowed category, specific title overlap and supported affirmative past wording
agree in the same dated clause. Actors, scalar identities, inferred dates,
obligations, negated/future acts and ambiguous quotes remain outside this exception.
Sentence/paragraph scope and bounded date-text matching prevent adjacent acts and
“4 April”/“14 April” substring errors from supplying the premise.

Identical recorded output now stores two Events rather than one Event/one standing
mention; this SQLite replay makes no provider call. A fresh original-only full
stack/model/browser run exercises the new branch, shows both correct days and exact
source quotes, and still omits both performers. Twenty-five new storage cases,
direct guards, all Go/focused race, 802 backend/671 frontend, production build and
fourteen-default/four-optional reproduction checks pass. Existing graph/config,
protected edits and user histories remain intact. Optional role-meaning stays
opt-in. This is synthetic development evidence, not independent certification.

An exact quote can support a subject mention without proving action occurrence,
date role or performer association. Keep these as separate labels, alongside
entity/category and human acceptance. A lexical storage exception is not a
semantic classifier; unknown verbs/languages and older actor-word exceptions still
need evaluation. Train or shortlist with Laya only after downstream integration
checks, necessary-clause/amendment recall and calibrated fallback gates. Prioritize
role recovery and version-bound reviewed correction/undo before treating counts
as completeness. No new Laya/GLiNER training, speed or acceleration measurement.

See the [observed-action audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-observed-actions.md)
and [shared receipt](validation/delaylama-observed-actions-20261004.json).

## L80 — Review queues must surface unsearched work and retain model variability (2026-10-04)

DelayLama's saved-concern filter could miss source Events that had never been
checked. A dedicated contract queue now separates missing loaded performer links,
unmatched quotations, occurrence/date concerns, unsearched candidates and saved
search failures. Nearby-mention associations and vectors stay out of its event
review list. Scope, partial snapshots and bounded/unavailable history remain
visible; no-search and model support are not human acceptance states.

Eight new checks and 679 frontend tests pass. One fresh original-only full-stack
Gemini draw stored both performers plus affected/cause links, unlike earlier draws
on the same synthetic original. This is model variability, not proof that the
prompt improved or role recall is reliable. The live browser opened queue evidence,
ran one performer source check, retained its exact quote and imported originals
and search history into Prompt Lab. The graph remained unchanged. That live graph
had no missing performer slot; missing-slot preparation is covered by mounted UI
fixtures. No prompt revision, experiment run or activation was requested.

Keep successful and failed draws when constructing training/evaluation data.
Distinguish quote presence, role association, missing loaded link, search state,
source version and human acceptance. A review-history import is context for later
revision, not training or automatic improvement. Version-bound correction/apply/undo
and independent same-document prompt comparisons remain open. No new Laya/GLiNER
training, throughput, mobile or Docker measurements occurred here.

See the [queue workflow audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-review-queue.md)
and [shared receipt](validation/delaylama-review-queue-20261004.json).

## L81 — Close the review loop before treating feedback as training (2026-10-04)

DelayLama's next [delivery plan](https://github.com/UnicoLab/delaylama/blob/main/docs/next-improvements.md)
prioritizes explicit source-bound correction, a results-first workspace, coverage
states and frozen baseline/candidate prompt comparisons. The SQLite/HTTP
performer apply/undo prototype passed targeted concurrency, rollback, idempotency
and scope/change guards in an isolated checkout. It is not packaged, activated or
connected to the application; full application/job/source validation, GUI and live
workflow qualification remain. This is prototype evidence, not a released feature.

A model support verdict or matching quote is not an accepted training label.
Retain original suggestions, human decisions, exact source versions and applied /
undone history separately. Corrections can become adjudication candidates, with
independent contract/template splits, explicit role direction and missing-premise
labels. Keep benchmark contracts out of training and retain unsuccessful draws.

The supplied Hive report was rechecked: its current banner says Round 4 and §16
reports no live model promotion, despite the small LoRA reader passing its Hive
gate. The [assessment](hive-report-insights.md) now identifies that current report
instead of the earlier Round 3 label. Compare a trained reader, frozen head,
contrastive shortlist and optional span extractor on contract-specific tasks;
the Hive scores cannot select a DelayLama architecture by themselves. Existing
MLX scorer acceleration and supervision do not establish full-encoder acceleration
or new extraction performance. No training, model promotion or live graph change
was made for this planning update.

## L82 — Keep reviewed graph actions separate from extraction and training labels (2026-10-04)

DelayLama now connects its source-bound performer review prototype to the
application and browser: select an existing matching party and exact original
passage, explain the association, preview, explicitly confirm, apply, reload and
undo. The engine atomically writes an actor→Event link, journal and timeline under
a SQLite `BEGIN IMMEDIATE` lock. The initial deferred read-to-write prototype
failed concurrent apply; the final concurrency, rollback and race checks pass.
Exact project/tenant, source and endpoint fingerprints guard the write. A retry
after undo returns undone history instead of recreating the edge. Other storage
adapters refuse unsupported writes. User graph data and protected work were
preserved during activation.

An isolated recorded-model-output replay plus one new Gemini source search and
the live browser qualified the workflow. All 835 backend and 688 frontend tests,
the production build, Go packages and five patch variants pass. This establishes
review operation behavior, not independent extraction accuracy. Human action
history is readable in Prompt Lab and accompanies original search history.

Two actual assistant proposals nevertheless misattributed a graph omission to
the event-detector prompt without an evaluated target-agent run; neither clearly
acknowledged the undone link in its rationale. Additional explicit origin/action
guidance did not demonstrate a reasoning improvement. Both proposals were
rejected and retained, with no draft/default activation or benchmark run.
`responsible_party` is a defined extraction field: this was causal misattribution,
not an unsupported-field error.

For reusable training data, preserve model proposals, original source versions,
role direction, reviewer explanation, applied/undone state and actual target
input/output as different labels. Undo is an operation outcome, not automatically
a false extraction label. Source quote presence is not semantic entailment, and
a reviewed performer link is not legal liability. Only adjudicated task-specific
labels should enter training; keep the failed assistant draws as evaluation
examples. No new Laya/GLiNER training, architecture, speed or MLX measurement was
made here. Prioritize independently evaluated baseline/candidate comparisons
before promoting prompt or model changes.

See the [reviewed-correction audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-reviewed-performers.md)
and [shared receipt](validation/delaylama-reviewed-performers-20261004.json).

## L83 — Freeze analysis inputs and keep criteria separate from scores (2026-10-04)

DelayLama's selected-run comparison now captures the baseline original, language
and claim detector register together with a candidate prompt. Human criteria are
recorded before execution; results retain the baseline/attempt identity and their
own review scope. Failed or interrupted attempts remain in history without
automatic resubmission. Restored metadata must agree with both actual run records
before the GUI calls their inputs identical. Actual runtime differences remain
visible rather than being hidden by the captured-input label.

Two fresh event-detector and two fresh claim experiment runs qualified this path
in an isolated browser/API/graph stack. Edited editor sources did not replace the
baseline original or event inputs. Both event draws preserved dates and explicit
performers; this candidate included the original occurrence labels inside exact
citations. Both claim pairs retained E1/E2 time/cost associations with unknown
entitlement when premises were missing. No graph write, default activation,
semantic acceptance, prompt-assistant request or new training occurred. One
synthetic contract/draw pair cannot demonstrate reliable prompt gains.

All 706 frontend tests and the production build pass. The initial automatic-open
implementation was caught closing the comparison when result selection changed
and was fixed. An intermediate full run timed out on the busy host while edits
also overlapped it; the final stable-source run passed with four workers and
unchanged assertions/timeouts. Keep unsuccessful observations in the audit.

For training/evaluation data, retain captured inputs, source versions, candidate
and baseline prompts, criteria, unsuccessful attempts and actual execution
records separately. Human criteria are not objective labels; exact quotations
are not entailment, and equivalent raw date formats are not necessarily semantic
regressions. Repeated draws and adjudicated independent contracts remain the
gate for prompt/model promotion. No Laya/GLiNER or acceleration measurement was
added in this pass.

See the [frozen-comparison audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-frozen-comparison.md)
and [shared receipt](validation/delaylama-frozen-comparison-20261004.json).

## L84 — Keep result presentation, quote matching and graph completeness separate (2026-10-04)

DelayLama now puts contract identity and named parties first for the identity
agent, and pairs occurrence dates with reported performers for event extraction.
Every overview item opens its original review path; duplicate titles/IDs cannot
redirect review to a different array element. Entirely unreturned financial
fields stay accessible in a collapsed group without being turned into zero.
Project document previews no longer inherit analysis payloads, timeline flags or
saved-run headings. Agent-specific modes remain separate across tab/reload checks.

Two actual saved-workflow model jobs on one new synthetic French contract
returned the expected contract reference, employer/contractor roles, price,
completion and two event-date-performer triples. The browser confirmed party
quotation navigation, event timeline association, Project isolation and reload.
This qualifies presentation and binding on the case, not independent recall,
semantic support, model speed or prompt optimization.

The same draws retained uncited OPC/summary fields and event quotes lacking
G-1/G-2 labels. Guarantee provider/beneficiary assignments and construction-scope
paraphrases still require their own semantic support. The isolated source graph
had nine generic nodes, two HAS_DATE links, no typed Event nodes and a truncated
accented contract identifier. A saved analysis register, matching quotes and a
running graph service cannot establish a complete or correctly linked graph.

For reusable Laya training/evaluation, record exact Unicode source/reference
identities, event/actor/affected-party direction, field-specific evidence,
missing extraction, missing graph representation and human adjudication as
separate observations. Do not train a correct scalar extraction as incorrect
merely because graph registration failed, and do not call a generic same-line
entity/date edge an adjudicated performer relation. Preserve unsuccessful draws
and evaluate independent contracts/templates/languages before promoting a
retrieval, classification or span head. No new Laya/GLiNER training, architecture
or MLX measurement occurred in this pass.

See the [results-first audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-results-first.md)
and [shared receipt](validation/delaylama-results-first-20261004.json).

## L85 — Separate analysis findings, graph registration and source truth (2026-10-04)

DelayLama's new live trial reuses L84's two frozen saved model outputs and the same
original, then independently exercises quote ingestion and one fresh original-source
Gemini graph job. Quote indexing initially gives two saved event findings but zero
typed source Event nodes. Switching representations starts no model work; an
explicit source action owns one confirmed durable handle. A completed graph job
is an operational observation, not an extraction completeness label.

Whole-reference parsing is a concrete source identity prerequisite: the old ASCII
matcher made RABLE-2044 from QA-ÉRABLE-2044. The new bounded Unicode matcher retains
accents/decomposed marks and exact original UTF-8 offsets, withholding embedded or
oversized fragments. Eight positive and eleven negative convention cases pass in
two extraction modes; full Go, default/optional clean patch reproduction and live
quote ingestion qualify the change. No historical alias migration is claimed.

The source draw returns three Events (signature, work start, access closure) and
keeps the April 4 planning transmission as Planning. Only the closure has a quoted
Ville performer link. The transmission therefore remains absent from source-event
chronology and its Horizon performer is missing. Signature entry-into-force impact,
scheduled versus observed start and a nearby Article 3 clause association require
separate semantic adjudication. Three source Events versus two saved events is not
better recall: these representations contain different findings.

For Laya datasets/evaluation, label field extraction, source span presence,
occurrence identity, observed/planned date role, typed-node registration, actor vs
affected-party direction and clause association separately. Never turn a graph
storage/type omission into a negative extraction example, or a nearby heading into
positive clause entailment. Keep missing/ambiguous outcomes and original Unicode
identities. Split templates/contracts/languages before calibration or promotion;
this pass includes no new Laya/GLiNER training, MLX benchmark or classifier gain.

Configuration must also be observed: this CLI automatic path ignored the startup
schema preset and the REST preset inventory omitted supported profiles. The
isolated engine was explicitly set to construction_claims via REST and its worker
receipt confirmed that schema. User configuration stayed general. Full Go and
713 frontend tests pass; a full API race sweep timed out at ten minutes in the
malformed-body route/payload test, while changed-package and short API race checks
pass. Keep the unsuccessful check rather than silently widening its budget.

See the [implementation audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-event-representations.md)
and [shared receipt](validation/delaylama-event-representations-20261004.json).

## L86 — Freeze actual task configuration before evaluating quality (2026-10-04)

DelayLama's startup path previously discarded the configured pipeline and custom
configuration; the CLI silently fell back after opening errors. The REST catalog
listed four of seven registered profiles. New regressions fail on the sixteen-patch
baseline and pass after preserving configuration, validating the winning profile
before storage and enumerating the registry. Invalid profiles now exit without
creating either graph. Environment intent alone is not a reproducible experiment.

A contract-scoped selector now submits an explicit task-local profile only when
analysis is requested. One fresh original-source Gemini job captures
construction_claims from preset while the shared engine remains general. Choosing
the profile starts no work; reload retains the choice and one existing run.
Unavailable choices block new submission, and missing/mismatched job receipts
produce an observation warning without silently repeating accepted work.

The source result still has three Event nodes and one Planning transmission,
with only the closure's Ville performer. Startup/transport success therefore
does not remove the existing chronology/performer and semantic date/clause gates.
The two saved identity/event outputs were replayed from L84; they are not new model
draws. No new Laya/GLiNER training or acceleration measurement was performed.

For reusable training/evaluation, record requested configuration separately from
actual captured configuration, fail invalid winning settings, retain unsuccessful
probes, and compare independently labelled semantics after transport passes.
Go internal tests verify bounded pipeline options; the REST config endpoint does
not expose every option. Full Go, 843 backend and 719 frontend tests pass; final
copy is covered by 57 focused tests and build. Race checks are explicitly short;
the earlier full API race timeout remains unresolved.

See the [startup/profile audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-startup-schema.md)
and [shared receipt](validation/delaylama-startup-schema-20261004.json).

## L87 — Recover registration without manufacturing semantic labels (2026-10-04)

DelayLama retained a planning transmission as Planning because its occurrence
guard missed dated journal language in the historical present. A registered actor,
supported finite dispatch wording, action-specific title, explicit same-clause
date and original quote now preserve a reviewable Event candidate. Seventeen
new storage cases and direct subject/title checks retain duties, future/negative
wording, document labels, unrelated dates and missing actors as negatives. A
source supplement can register a newly named company; an initial test incorrectly
treated that actor as unregistered, so its assumption was corrected and the
helper's exact-boundary negative is tested independently.

The frozen actual L86 output changes three Events plus one Planning mention to
four Events without another provider draw. One fresh original-source job yields
five Events, including the correctly dated transmission but also a June completion
deadline whose planned/observed role is unresolved. No performers are proposed,
and the closure-to-March-start affected link lacks occurrence-specific support.
More stored candidates do not establish better semantic accuracy.

A targeted performer source check returns Horizon with the exact G-2 quote. The
GUI prepares a candidate link preview and requires human review; no link is applied
and graph rows remain unchanged. That same recheck also calls the transmission's
"decision" category supported merely from the record of transmission. Preserve
this failed semantic check, rather than using its support verdict as a training
label. Separate registration, quote location, date role, category, role direction
and human acceptance in datasets and promotion gates.

The chronology and performer-preview visual check exposed low-contrast hover
text and a faint citation highlight. Scoped hover styles and contextual yellow
marks restore readability in both themes. Full Go, agents/intelligence races,
22 Python patch tests, 56 frontend surface tests, build and five patch arms pass.
Earlier broad frontend/backend checks and the full API race timeout remain
historical evidence. No new Laya/GLiNER training or speed measurement occurs.

See the [dispatch/UX audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-dated-dispatch.md)
and [shared receipt](validation/delaylama-dated-dispatch-20261004.json).

## L88 — Separate date-role labels, output compliance and quote location (2026-10-04)

DelayLama's source graph stored a contractual completion deadline beside reported
actions without a date-role field. A new additive protocol transports observed,
planned, deadline or unknown plus a role quotation through model output,
conversion, storage, chronology and source review. Planned/deadline and explicit
unresolved assessments keep calendar metadata without actual-occurrence fields.
Legacy no-field rows remain unassessed; there is no historical migration.

The fallback extractor also treated JSON as plain lines, turning structured
arrays/properties into event names and losing evidence. Fenced/ordinary arrays
now retain fields; malformed structured output fails extraction. Baseline tests
retain the erroneous full-array event name and a nil-parser panic. Provider
transport, actual storage and original Unicode/ambiguity checks are separate
from model interpretation tests.

The first live Gemini draw omitted all five date roles despite an instruction.
Explicit mandatory event fields and a concrete output template supplied them in
a separate second draw: three reported candidates, one planned start and one
completion deadline. This establishes protocol feasibility on one synthetic
French original, not a measured quality improvement. Both actual outputs and
unsuccessful probes are retained. Identical second-output SQLite replays isolate
the storage repair: the older arm assigns occurrence fields to all five; the new
arm withholds them for the plan/deadline while retaining their calendar dates.
Both draws still omit performers, and directed relation proposals vary.

For Laya training, keep output compliance, quote location/containment, role meaning
and human adjudication as separate labels. A matching passage with a wrong role
can pass the location gate; a precise date does not certify occurrence. Never use
source-match flags or a source-recheck support verdict as gold semantic labels.
Split contract families/templates before training, and qualify date-role/premise
association and contradiction handling independently. No new Laya/GLiNER training,
MLX benchmark or architecture promotion was performed.

The GUI now filters roles and opens contextual exact role evidence in both themes.
A targeted deadline recheck retains `requires_review` and does not change graph
rows. Reload restores contract/graph/event/search. Production CSV generation on
actual graph data passes; the browser download observation timed out, so browser
file delivery remains unconfirmed. Full Go, focused agents/intelligence/provider
races, 845 backend and 722 frontend tests/build, plus five clean patch arms pass.
The earlier full API race timeout and bundle-size warning remain limitations.

See the [date-role audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-event-date-roles.md)
and [shared receipt](validation/delaylama-event-date-roles-20261004.json).

## L89 — Actor attribution needs its own protocol and evaluation denominator (2026-10-04)

DelayLama's relation prompt omitted real performers and confused action with
legal responsibility. Required performer cards for every registered Event now
carry explicit actor/event references and names, candidate/unresolved/not-applicable
status and original quotations. Actor eligibility comes from the ontology,
including OPC. Denial of liability does not erase action; a plan/deadline or unsent
notice does not establish performance. Source/name binding is a transport gate:
a deterministic wrong-actor counterexample still passes a located quotation.

Thirteen synthetic supplied-register development cases were frozen before the
candidate. The prior default retains 5/16 declared links plus one extra; initial
cards retain 16/16 plus one unsupported affected relation from adjacent actions.
Requiring an explicit affected premise yields 16/16 with zero extras, 13/13 exact
cases in another draw. Retain the failed arm. These correlated development cases
are not independent contract quality or a causal estimate of prompt gains.

One fresh French source/browser run retains four performers but still misses the
contractor's actual suspension. Separate English/French final source draws retain
six declared performers and two recipients, with exact unique source spans and
scoped model-candidate labels; an English deadline remains absent as an Event.
A schedule issuance was mistyped as OPC. Verb expansion alone failed. The final
strict dated actor-action collision branch changes the identical actual extraction
SQLite replay from 3/4 to 4/4 Event candidates, leaving actor identity undated.
The replay also exposes a wrong generic date association to an unrelated deadline.
Registration, role attribution, source recall and broad graph associations need
separate denominators and regression gates.

For reusable training, do not label matching quotations or model support verdicts
as gold meaning. Include wrong actors, negation, duties, plans, unsent notices,
independent adjacent actions and date/clause misbinding. Split contract families
and templates before training; check necessary-premise coverage before shortlisting.
No new Laya/GLiNER training, MLX benchmark or architecture promotion took place.
Full Go/focused race, 859 backend, 724 frontend and five clean patch arms pass;
the prior full API race timeout, bundle warning and browser CSV delivery limit
remain. The live default binary/API were activated with user data/settings preserved.

See the [performer audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-event-performers.md)
and [shared receipt](validation/delaylama-event-performers-20261004.json).

## L90 — Connectivity is not a scalar-binding label (2026-10-04)

DelayLama's generic orphan linker associated scaffold erection with an unrelated
completion deadline because the endpoint was nearby. Its justification asserted
that proximity established qualification; a later healing pass could repeat it.
Both scalar paths now require original premises. Event date candidates match the
explicit date text inside a unique original event passage. Other values/references
need literal same-sentence mentions. Repeated names require a unique supporting
passage. Original UTF-8 projection prevents case folding from shifting offsets.
Surviving associations stay unverified; same-sentence location is not meaning.

The identical actual extraction replay through SQLite removes one wrong deadline
edge while preserving all four Events and date-role metadata. Positive SQLite/
filedb tests retain a supported date with exact original offsets; negative cases
cover cross-sentence values, different dates, fabricated/repeated passages,
foreign scope and forged byte offsets. A fresh synthetic browser/model run retains
three actual action candidates, performers, notice recipient and a planned unsent
reminder, without that date edge. Its completion deadline remains absent as an
Event. Reload and production CSV contents pass; browser download observation
times out, so file delivery is still unconfirmed.

For training, retain scalar-binding mistakes even if both names/dates occur in
the document. Separate transport/location, association, date role and adjudicated
meaning; do not promote connected-ratio gains into gold labels. Conservative
location guards can lose cross-sentence candidates, requiring independent
necessary-premise recall and fallback evaluation. General content healing remains
a separate semantic gate. No new Laya/GLiNER training or acceleration measurement
was performed. Full Go/final focused race, 725 frontend, 29 patch checks and five
clean patch arms pass. The tested binary is active with user data/settings preserved.

See the [scalar association audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-source-associations.md)
and [shared receipt](validation/delaylama-source-associations-20261004.json).

## L91 — Review metadata must travel with exported findings (2026-10-04)

DelayLama chronology now previews all loaded or visible rows and retains source
warnings, human decisions/notes, quotations and references in saved-analysis CSV.
Long warnings expand in the preview without truncating exported values. Located
text remains a transport/location observation, separate from adjudicated support.
Use this distinction when curating training candidates: a flag/note is feedback,
not automatically a correct target label, and a quoted model output is not gold.

An isolated full-stack browser replayed actual earlier synthetic model outputs:
four source events (one when filtered) and five saved timeline entries (two when
filtered). A human correction note, flag, original quote and field warning survived
export and review-backup reload. Browser clipboard copies were observed. A single
persistent-link download observation timed out; file delivery is unconfirmed.
Textarea LF normalization is distinct from original CSV serialization. A requested
390×844 override exposed 582×1260 DOM pixels; report observed dimensions, not an
unverified device claim. Native dialog modality must end before restoring focus.

Final frontend 735 tests/59 files and build pass; bundle-size warning remains.
Every replay graph table stayed unchanged, no new analysis/training job ran, and
user data/settings were preserved. No Laya/GLiNER quality or acceleration result
was generated. Known missing occurrences and uncertain semantics remain separate
quality work; reviewed candidate exports make that work reproducible, not solved.

See the [export audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-chronology-export.md)
and [shared receipt](validation/delaylama-chronology-export-20261004.json).

## L92 — Repeated event labels are not occurrence identities (2026-10-04)

DelayLama traced a known omitted French suspension to raw model extraction, then
found an independent transport loss: title-only candidate/chunk merging retained
one of three repeated notices. Conservative merge keys now include literal date
precision, original/role passages, participants and category/date-role assessment.
The controlled pipeline retains all three on SQLite/filedb; matching chunk overlap
still deduplicates. Time/range/month wording is not collapsed to a first date.

A fresh synthetic two-pass source/browser run retains three notices, six declared
sender/recipient links and a planned never-sent reminder without actual performance.
The two same-day notices open their own passages; reload and observed clipboard
export work. Its titles already differ, so it does not measure the repair's accuracy
gain. Location checks bind unique original bytes and occurrence passages, without
requiring a source label the quote does not actually contain. All four candidates
retain classification warnings; thirteen total connections are not all certified.

For training/evaluation, preserve actor/action/date/occurrence bindings and split
repeated actions, same-day different actions, plans and duplicate overlap examples.
Do not use title-based deduplication to create labels or count recall. A missing
model action needs a recall experiment; a lost returned action needs a transport
repair. Candidate assertion keys still cannot adjudicate ambiguous identical text,
paraphrases or impact conflicts. Full Go/focused race, 29 patch checks and five
clean patch arms pass; tested runtime is active with user data/settings preserved.
No new Laya/GLiNER training or acceleration evidence was produced.

See the [repeated occurrence audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-repeated-occurrences.md)
and [shared receipt](validation/delaylama-repeated-occurrences-20261004.json).

## L93 — A graph omission is not an extraction-prompt diagnosis

DelayLama's prior warning treated any experiment in a session as a relevant
baseline. The new inventory requires an exact original, explicit target agent,
recorded prompt and readable completed output; imported/legacy/foreign/fallback
records and ambiguous benchmark identities cannot establish that baseline. The
selected result is separately bound to its run, including strict JSON type
comparison. The endpoint computes this inventory independently of model output.

An actual assistant still blamed the prompt before a relevant baseline existed.
A fresh unchanged-prompt run returned the declared performer; the next assistant
recognized it and acknowledged human undo, while overclaiming graph causality.
Keep both failures and corrected observations. Neither proposal was applied or
activated. Available records and matched quotations do not authenticate execution,
prove semantic accuracy or locate a pipeline failure. Human review-state snapshots
also do not represent every transition or current live graph state.

For training/prompt optimization, localize loss at extraction, merge, registration,
association, retrieval or rendering before constructing improvement labels. A
missing graph link is not automatically a negative extraction label. Preserve the
original/run/output association and unknown/partial history; evaluate candidate
changes separately rather than treating assistant recommendations as truth.
No new Laya/GLiNER training or acceleration evidence was produced.

See the [Prompt Lab audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-prompt-evidence-scope.md)
and [shared receipt](validation/delaylama-prompt-evidence-scope-20261004.json).

## L94 — More extraction instructions can lose facts without fixing recall

Two isolated action-granularity prompt revisions were rejected after 32 actual
Gemini draws on four frozen invented English/French sources. V1 appended atomic
instructions; V2 also removed a conflicting later milestone preference. Every
arm retained 14/24 selected action records with its own quoted actor/calendar
premise. Candidate separate action identities rose from 16 to 17/18, but those
additions lacked their own date/actor support. Deadline/plan retention fell from
11/14 to 9/14 and 8/14. The known contractor suspension was not returned as a
separate action in either candidate draw. These are coding-agent-reviewed
synthetic development bindings, not an independent entailment/recall benchmark.

Both V2 report-receipt findings quoted a report-delivery sentence. The quote was
original but did not establish that finding or retain its date/pronoun antecedent.
Signature impacts also asserted entry into force without an explicit premise.
No declared cancelled/unsent action became observed, but other fields were not
exhaustively scored. Location validation and a longer quote cannot prove meaning.

For Laya data, create delivery/receipt, instruction/suspension, signature/effective
and review/approval counterfactuals. Preserve action, actor, date, direction and
conditions; train insufficient-support labels for a true finding paired with
its wrong supporting passage. Keep missing-action recall separate from support
classification: a head cannot recover an unproposed action. Next test a dedicated
inventory and missing-item pass, then independently reviewed contracts and
necessary-clause recall before shortlist filtering. Both prompts remain optional
rejected experiments; no Laya/GLiNER training, architecture or acceleration result
is claimed. Engineering tests passed without changing the live application.

See the [source action audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-source-action-coverage.md)
and [shared receipt](validation/delaylama-source-action-coverage-20261004.json).

## L95 — A separate source inventory can find omissions without validating meaning

DelayLama now provides an explicit one-original review inventory, independent of
stored extraction and graph output. Indexed original passages reconstruct each
candidate's own action, actor and calendar context. Stable saved requests, reload
observation, captured-source history and changed-original warnings keep model
work bounded. It does not add graph facts or automatically rewrite prompts.

Sixteen development calls compared two protocol revisions on previously inspected
synthetic English/French sources. The selected indexed protocol retained 20/24
selected observed actions with their own actor/calendar passages and 8/14 selected
plans/deadlines, based on coding-agent manual review. It recovered the separate
contractor suspension and separated submission/review/return, but withheld the
French completion deadline and report delivery/receipt for actor/date premises.
A never-sent reminder and cancelled meeting were not proposed. Original-language
instructions were also violated. The full completeness gate failed; different
provider settings/tasks prevent a paired improvement claim against prior graph
extraction. No independent real-contract evaluation was added.

A fresh two-original graph run and two browser inventory jobs demonstrated the
workflow. The stored closure combines a contractor suspension in its quotation
and assigns owner plus contractor actor links; the inventory separates the
suspension. Overlapping text therefore remains a comparison lead, not matching
identity or proof of a particular prompt failure. All graph-table fingerprints
were unchanged by inventory scans and saved-job reuse. The full suites passed
908 backend/746 frontend tests; final feedback-content checks and build passed.
These engineering checks do not prove extraction quality.

For training, preserve omitted/withheld assertions in the denominator. Include
pronouns, shared dates, cancelled actions and semantic pairs such as delivery/
receipt, instruction/suspension and review/approval. A true action paired with an
incomplete supporting passage is insufficient support; a source match paired
with the wrong action label is not approval. Do not train from unreviewed inventory
proposals or assistant recommendations. Laya can prioritize review and route
clauses only after measuring necessary-premise recall, false support and calibrated
abstention. No Laya/GLiNER weight, architecture, training or acceleration result
was produced in this experiment.

See the [source inventory audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-source-inventory.md)
and [shared receipt](validation/delaylama-source-inventory-20261004.json).

### L96 — Inventory feedback needs an actual baseline and a failed comparison record

**Observed (DelayLama, 2026-10-04):** a selected saved inventory candidate now
opens the chosen agent's Prompt Lab with its exact original, source version,
saved scan history, withheld proposals and reviewer request. It preserves existing
drafts, notes and runs; original selection is explicit. The evidence panel treats
inventory leads separately from target-agent experiments. Context reads reject
changed originals and foreign project/document/candidate scopes. Importing does
not invoke a model or write graph facts.

One previously inspected invented French contract supplied a fresh extraction
baseline and frozen candidate comparison, with two assistant requests. Before
the baseline, the assistant still blamed prompt inadequacy despite the recorded
zero-baseline scope. After seeing a real baseline, it recommended separate
actions for different actors and claimed the refinement would force separation.
Both extraction runs returned two operational records, merging the contractor's
separate suspension into the owner's closure. Both summaries overstated the
owner's order as a suspension order. The candidate preserved planning non-request
and non-agreement wording more faithfully, but failed the recorded separation
and summary criteria. Exact source quotation checks and matching recorded model
settings did not certify semantics. The candidate was not activated.

**Implication:** measure assertion structure and role association alongside span
matching. Shared dates, actor transitions and actions stored as consequences
need explicit evaluation. Preserve unsuccessful assistant suggestions and frozen
comparisons; do not convert them or inventory candidates into approved labels.
Use adjudicated source-supported corrections for training, with independent
contract/story/template splits and a separate promotion gate. A structured
action-first extraction arm is a proposed comparison, not a measured quality
improvement. A Laya shortlist or support judge must preserve all necessary
premises, recognize false support and allow calibrated abstention.

The rendered workflow persisted the selected candidate and both runs through
reload. Full suites passed 920 backend and 755 frontend tests, with a successful
build. The user graph and protected artifacts were preserved; API-served replay
nodes, edges, episodes and timeline were identical. Historical QA SQLite hash
serialization could not be reconstructed, so no all-table QA hash claim is made.
No new Laya/GLiNER weights, training, architecture or MLX result was produced.

See the [inventory-to-prompt audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-inventory-prompt-lab.md)
and [shared receipt](validation/delaylama-inventory-prompt-lab-20261004.json).

### L97 — Separate action calls improve granularity without certifying field support

**Observed (DelayLama, 2026-10-04):** four previously inspected invented EN/FR
fixtures supplied four fresh normal extraction baselines and eleven focused
per-candidate calls. The focused arm replayed source inventory V2; it made no
fresh inventory calls. Manual review found 10/12 selected actions as distinct
records versus 8/12 for the baseline. This is a development binding count, not an
independent accuracy or whole-document completeness score. Repeated full originals
and extra calls do not establish a latency or cost benefit.

The quality gate failed: closure was inferred as the suspension cause, the owner
closure inherited the contractor suspension's one-day impact, and adjacent acts
still appeared as consequences. Charme report delivery/receipt were withheld by
the replayed inventory and were never analyzed downstream. Thus inventory recall
limits focused extraction, and an exact quote does not certify a field's meaning
or its association with the correct action. The experimental arm remains outside
the active extraction agent and does not update graph facts or default prompts.

Prompt Lab now preserves failed/passed/inconclusive human comparison judgments
with explanations and SHA-256 bindings to both recorded runs, prompts, original
and criteria. The browser confirmed persistence, stale-output warnings and
restoration with both conflicting copies retained. One fresh assistant request
received all 13 preceding entries and acknowledged failure, but still described
its next prompt suggestion overconfidently. It was not applied or activated.
These judgments are client-supplied scoped reviews, not authenticated executions,
automatic promotion gates or approved training gold.

**Implication for Laya:** evaluate a sufficient-context shortlist and a calibrated
support judge separately. Include false causal support, neighboring-actor impact
leakage, pronoun/shared-date chains, negation and missing-inventory actions. Keep
all stages' omissions in the pipeline denominator. Use adjudicated corrections
with independent contract/story/template splits; synthetic model nominations or
reported passes alone must not become positive support labels. Preserve failed
suggestions and report per-stage calls, resource use and end-to-end quality.

The suites passed 937 backend and 758 frontend tests; browser workflow and user
graph/artifact preservation checks passed. No new Laya/GLiNER weights, training,
architecture or MLX acceleration result was produced.

See the [action-first and review audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-action-first-comparison-review.md)
and [shared receipt](validation/delaylama-action-first-comparison-review-20261004.json).

### L98 — More connected context improves retrieval, not entailment

**Observed (DelayLama, 2026-10-04):** node context now pages every scoped
Chronograph collection, retains direction and edges between boundary nodes, and
can explore an entire connected component without arbitrary total graph/body/quote
ceilings. Exact original document hashes, offsets, clause references and conditions
accompany review. Generic graph `status=validated` was mistakenly available as a
contract fact: it is now excluded from model context. Curation is not approval,
legal effectiveness or authenticated semantic adjudication.

The multi-step extraction path previously reused generic cached context and cut
it at 6,000 characters. Current topic retrieval replaces that shortcut with whole
original paragraphs and connected facts, preserving complete structured input.
Model budgets remain separate from graph browsing and expose omissions. Source
rechecks bind their neighbourhood and originals in the captured finding digest.
Lexical anchors, passage hashes and graph paths do not establish entity identity,
causation, applicability or completeness.

Eight fresh event source checks, one browser graph recheck (two calls), and four
amendment core runs produced **fourteen recorded model calls** on inspected invented
fixtures. The semantic gate failed. A literal actor/date guard replay withheld a
wrong cause but still allowed another actor's duration to be assigned to closure.
Graph context exposed an already merged closure/suspension node; it did not repair
it. An explicit manual three-original graph fixture verified distributed-source
retrieval, not model graph extraction. The final amendment run used seven normal
identify seeds and received the complete 18,125-character topic bundle. It retained
approval conditions but still inferred legal ineffectiveness from unrecorded
approval. No actual approval was established. The proposed deadline no longer
becomes claim impact through application normalisation.

**Implication for Laya:** train/evaluate sufficient-premise retrieval separately
from occurrence/role binding and contractual applicability. Useful negatives
include both actors inside the same matched paragraph, same-date distinct actions,
missing approvals, conditional amendments, changed source versions and graph
curation flags. Passage keys identify source occurrences, not certified events.
Neighbourhood shortlists need qualification/amendment recall across disconnected
clauses and long contracts; report every budget omission and per-stage failure.
Independent adjudication and story/template/language splits remain required before
synthetic model suggestions can become support labels.

The full suites passed 976 backend and 767 frontend tests; final focused checks,
actual API guards, browser highlights/direction and user data preservation passed.
The model semantic failures remain documented. No Laya/GLiNER weights, training,
MLX or supervisor result was changed. See the
[graph context audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-graph-neighborhood-support.md),
[implementation design](https://github.com/UnicoLab/delaylama/blob/main/docs/contract-graph-context.md)
and [shared receipt](validation/delaylama-graph-neighborhood-support-20261004.json).

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


### L99 — Source action identity and field binding need separate quality gates

**Observed (DelayLama, 2026-10-04):** automatic graph sync previously used
`LearnStandard` indexing (deterministic regex/date nodes) while presenting completion
as full graph readiness. Full original analysis now uses durable per-source async
jobs and a captured construction-claims schema. Unknown outcomes keep existing
handles; startup and UI recovery observe them without replacement submissions.
Execution, source location and semantic accuracy are distinct measurements.

An individual action frame retains verbatim action/context/actor/impact premises,
original SHA-256 and original UTF-8 byte spans. Conservative merge keys preserve
conflicting frames. Same-paragraph or same-date actions must remain distinct.
Framed performer links require their own action and named-actor premise, rather
than selecting the paragraph's other actor or recipient. Impact attribution must
include the own action. Removing uncertain DateText exposed a further bug: the
legacy time helper still inferred the first surrounding date. A separate unresolved
association flag now suppresses that fallback. Keeping a located action with an
unresolved date is better than dropping the action or filling a guessed date.

Thirty-six original-analysis jobs completed in isolated SQLite projects, with no
seeded graph facts. The optional full-original source-first pass retained 22/22
declared actions on the final six synthetic fixtures, 20/22 expected performer links
and 14/22 expected dates. Only one of two declared impact phrases survived; the
other was omitted by the model, not rejected by the attribution guard. One declared
forbidden causal pair and forbidden approval edges were absent. **Semantic
qualification failed**; this extra pass remains off in normal builds/runtime.
Twenty-two extra inventory calls are reported in retained engine receipts; two
first-seen v4 terminal remote snapshots were missed before restart. Their durable
histories and scoped graphs remain. No total provider HTTP/retry count is claimed.
Final fixtures were reused during development, not independent certification.

**Implication for Laya:** learn/rank sufficient premise windows and classify
association uncertainty separately from action recall. Hard negatives should
include same-date compound verbs, pronoun/shared-subject antecedents, both named
actors in one paragraph, sender/recipient reversals, neighbour durations, fabricated
ellipsis quotations, planned/negated work and missing approvals. Keep full original
qualifiers and contiguous exact source spans. Source matching and positive model
judgment cannot become support labels without adjudication. Report incomplete
fields and stages; don't substitute a previous or nearby date as a successful
prediction. Story/template/language separation and real long-contract adjudication
remain required. Generic graph relationships still need semantic qualification.

The full backend suite passed 1,005 tests; the final frontend suite passed 774.
Default/optional Go suites, focused race checks, twelve clean idempotent patch arms,
browser evidence review and protected-data audits passed. Structural/UI passes do
not override the semantic failure. No Laya/GLiNER weights, training, MLX,
architecture or supervisor qualification changed. See the
[action-frame audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-source-action-frame.md),
[measured graphs](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/assets/source-action-frame/final-synthetic-graph.json)
and [shared receipt](validation/delaylama-source-action-frame-20261004.json).

### L100 — Indexed evidence is not an association quality gate

**Observed (DelayLama, 2026-10-04):** a second calendar fallback existed even
without explicit extracted calendar text. A model-generated action label containing
a neighbouring date populated a Date pointer before source validation. Reproduced
source-context and generated-label failures now clear canonical occurrence/valid-from
and calendar metadata for model-assessed blank calendar fields. Explicit calendar
text remains usable; historical graphs are not repaired automatically.

An isolated source-first experiment selected original line ranges for actor,
calendar, context and impact premises. The engine reconstructs exact contiguous
UTF-8 source text, retains antecedents and withholds invalid selections. This
avoids invented abbreviated supporting quotes. It still cannot establish actor,
date, causal or legal meaning from source location alone.

Six new SQLite projects completed full-original jobs; four reused development
stories and two English/French stories were frozen before first indexed-model
draws. Final checks retained 20/22 separate actions, 2/22 expected performer links,
20/20 explicit dates and 0/2 unknown-calendar actions. Both unknown-date actions
were proposed without the required date role and withheld. One own-action impact
passed, two declared forbidden causal pairs were absent with resolved endpoints,
and two located unperformed-plan candidates had no observed occurrence.
**Qualification failed** and the optional pass stays off. This is not a controlled
comparison with L99's different fixtures/model draws. Six terminal stage receipts
and six reported inventory calls are archived; provider HTTP/retries are uncounted.
The independent actor relationship stage remains the primary bottleneck. A noisy
heading absorbed into an organization name also remains unresolved.

**Implication for training:** preserve positively reported actions even when their
calendar is unknown, and measure action recall separately from calendar accuracy.
Use original-range selection as a location protocol, then train/rank sufficient
premises and actor association against adjudicated labels. Include generated-label
calendar contamination, recipient/performer swaps, pronouns, negative performance
and unknown-date returns as hard negatives. A candidate omission cannot count as
correct abstention on a required action. Neither copied text nor model judgment
should become a supervised support label without adjudication.

Backend 1,018 and frontend 774 tests pass; default/optional full Go suites and
focused default race checks pass. An initial concurrent frontend timeout/cascade
was retained; the affected file and complete one-worker suite passed. Parallel Go sweeps also timed out in the large API malformed-body sweep; its
standalone replay passed in 98.718 seconds, then final full suites ran serially.
A premature activation log check was corrected before committing. Structural
passes do not override semantic failure. No Laya/GLiNER weights, MLX, architecture,
training or supervisor qualification changed. See the
[indexed-premise audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-indexed-action-premises.md),
[captured graph](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/assets/indexed-action-premises/final-synthetic-graph.json)
and [shared receipt](validation/delaylama-indexed-action-premises-20261004.json).


## L101 — Original performer selection, decoder reasons and positive graph recall (2026-10-04)

**Measured and adopted structurally; full semantic gate still failed.** DelayLama
now allows the actor stage to select a unique, bounded original performer passage
instead of retyping it. References are issued only after source/context/action
checks. Invalid selections cannot fall back to other supplied text. Unknown-date
candidates retain their original action/actor premises without inheriting dates.
A selected passage is a location protocol, not proof of the actor's semantic role.

Eight full-original synthetic jobs retained 32/32 separate actions, 29/32 correct
actor checks with located evidence, 28/28 explicit calendars and 4/4 unknown-date
actions. Six reused stories and two new frozen joint-actor stories were kept
separate. Both joint-actor stories passed their declared checks; one reused French
story lost all required actors. A subsequent diagnostic response exposed a
protocol mismatch: its action-card reason and candidate assertion were supplied,
but each candidate's duplicate explanation was empty. A captured-response replay
failed before the decoder repair (0 accepted), then retained five source-bound
proposals with the model's original card reason. Explicit not-applicable cards
without candidates also became answered rather than invalid. No assertion or
explanation was synthesized; raw omitted fields remain unchanged in diagnostics.
A later known-case live inventory confirmation passed 3/3 declared actions,
actors and calendars. It is not an independent holdout result.

The default engine still merged a separately reported closure and suspension:
2/3 declared distinct actions on that known case. Two more English/French cases
were therefore frozen before inference, with topics dispersed across contract
procedures, access logs, drawing records and correspondence. They retained 12/12
actions, actors and original supporting passages, including two joint actions and
two unknown-calendar returns. Both own durations and six forbidden causal pairs
passed; two unperformed demolition proposals acquired neither observed occurrences
nor performer links. However **0/2 required explicit causal links were stored**.
Qualification failed; the source-first inventory remains excluded from normal
builds/runtime. No prompt revision or replacement draw followed these new failures.
Raw general relationship proposals were not captured, so the particular missing
causal premise/parser rejection remains undiagnosed.

**Evaluation correction:** a previous negative-cause checker covered aliases but
missed actual `CAUSED`/`CAUSED_DELAY` edge types. The evaluator now covers both,
requires original actor-relationship byte citations and source identity, and tests
positive causal recall with correct direction and a sufficient original passage.
Saved snapshots were rechecked without inference; original actor success/failure
partitions stayed the same. A clean absence of bad edges cannot establish a
complete graph. Predicate/type coverage is itself a regression target.

**Training implication:** separate action recall, calendar assignment, performer
identity, supporting-premise sufficiency, causal direction and legal effect in
labels and evaluation. Preserve model omissions separately from parser failures.
Synthetic hard negatives should include shared paragraphs, unknown return dates,
joint actors, notice-versus-cause swaps, acknowledged-but-not-approved documents,
and unperformed plans. Include positive explicit causation so a model that simply
omits all causal links cannot pass. Use adjudicated labels for Laya retrieval,
classification/ranking and premise selection; quoted text or an LLM card reason
must not become an automatic semantic support target.

The GUI now shows readable performer withholding diagnostics and a collapsed
model explanation beside the relationship citation and endpoint evidence. Browser
checks preserved undated performers, showed both joint actors and retrieved six
surrounding facts/five links from the registered original. No console errors were
recorded. Structural checks pass: 1,057 backend/775 frontend tests, both complete
Go suites, default focused race tests, 12 apply/idempotence combinations and ten
other variant intelligence/agent suites. Thirteen accepted jobs are terminal and
archived; twelve extra inventory calls were reported, not provider HTTP/retries.
Failed harness preflights and failed semantic gates are retained. User graph,
source files, job journal and Studio records were preserved. No Laya/GLiNER
weights, training, MLX, architecture or supervisor qualification changed.

See the [performer and context audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-performer-premise-selection.md),
[strict holdout graph](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/assets/performer-premise-selection/context-holdout-graph.json)
and [shared receipt](validation/delaylama-performer-premise-selection-20261004.json).


## L102 — Causal antecedents, output schemas and quoted-context fidelity (2026-10-04)

**Source protocol adopted structurally; semantic qualification remains failed.**
DelayLama captured raw relationship proposals and rejection partitions to resolve
L101's missing-cause diagnosis. In two known-case diagnostic jobs, the model
supplied the correct causal assertion but its quote omitted the earlier closure
antecedent. The strict own-action guard withheld both causes. Do not label this
as a model omission: distinguish absent assertion, insufficient supplied premise,
invalid selection, storage failure and retrieval/display failure.

The engine can now select one uniquely located endpoint context, or two endpoint
contexts and recover their enclosing **contiguous original span**, preserving
intervening qualifications. Both own-action anchors must fit the bounded original
span. References identify location, not semantic truth. Foreign/duplicate refs,
ambiguous originals, budget excess and fabricated/joined snippets remain invalid.
A captured parser replay adds a selection explicitly; it is not evidence that an
actual model obeys the new schema. Planned/deadline occurrences remain barred
from actual causal chains.

Twelve full-original synthetic jobs retained 56/56 separate actions, actor checks
with original citations and calendars; 62 performer links were present. Sixteen
forbidden causal pairs and eight unperformed-action checks passed. Ten stories
were reused and two cross-paragraph stories were frozen before inference. Only
2/4 required causes and 4/5 own durations passed. The recovered known-case causes
used fuller quotes: no relationship context selections were actually emitted.
The two new causes still failed, so the optional source-first inventory stayed off.

Inspection found that the final JSON template contradicted the new preceding
instructions by omitting selection fields. Aligning that template led all four
subsequent model causes to select the correct endpoint contexts. These jobs used
two now-known stories and two new frozen structural variants with conditional
restart and denied access between antecedent and effect. Three models nonetheless
joined quotes with `...`/`[...]`, so their causes remained withheld. The fourth
retained its cause, actual original envelope and an exact source fragment. Five
relationship selections were accepted overall, including notice recipients.
All 24 actions, actor checks and calendars passed, with 28 performer links and
12/12 forbidden causal pairs absent, but only 1/4 required causes and 2/4 own
durations passed. **No complete case passed the final gate.** French impact
output described causation without the stated five-day duration; a located impact
passage does not establish complete field extraction. No further prompt tuning
or model draw followed those two fresh variant outcomes in this audit.

**Training implications:** evaluate output-schema compliance separately from
semantic recall and citation sufficiency. Selection-only outputs can avoid
lossy model recopying; require adjudicated full original support for labels.
Represent duration/magnitude, cause, actor, actual-versus-planned occurrence,
conditional applicability and legal effect as separate targets. Include dispersed
positive causal cases as well as negated, hypothetical, notice and reverse-direction
hard negatives. A short anaphoric quote can be faithful text yet insufficient
support; concatenating excerpts can erase a decisive intervening exception.
For Laya, candidate retrieval/ranking should preserve each passage's identity and
enclosing-context requirements, never promote quoted text alone into support or
entitlement. Avoid training on these unreviewed LLM propositions as gold.

The GUI exposes readable withholding reasons, source excerpts, the complete
antecedent context and both endpoint originals. A real Hêtre browser review
retained conditional access/restart and compensation clauses; no console errors
were observed. Backend 1,063/frontend 777 tests, production build, final default/
optional full Go suites, default race checks, twelve patch combinations and ten
other variant suites are recorded in the linked receipt. Eighteen accepted remote
jobs ended done and were archived; eighteen extra inventory calls are recorded,
not provider HTTP/retries. Failed semantic gates and harness invocations remain
available. Existing user graph, journals, prompts, engine edits and Studio state
are preserved. No Laya/GLiNER weights, training, MLX, architecture or supervisor
qualification changed.

See the [causal context audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-04-causal-context-selection.md),
[failed final template gate](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/assets/causal-context-selection/template-synthetic-graph.json)
and [shared receipt](validation/delaylama-causal-context-selection-20261004.json).


## L103 — Selected source context, literal scalars and unknown-calendar recall (2026-10-05)

**Measured application engineering progress; extraction coverage still fails.**
DelayLama added selector-only relationship quotations, source-bound duration
fields and provider-template consistency. The decoder recovers one continuous
original envelope, preserving intervening qualifications; it rejects model-spliced
quotes. Durations have separate own-action and scalar byte citations. A legacy
impact summary can retain a validated duration while its unsupported model effect
stays withheld. Repeated attachment clears stale source support and warnings.
These are source-location/provenance controls, not semantic adjudication.

Fifty-eight real remote jobs on isolated synthetic documents finished and were
archived. All used Gemini `gemini-3.1-flash-lite`, captured construction-claims
schemas, complete registered originals and noop embeddings. Forty-eight jobs
used the optional source-first inventory; provider HTTP attempts/retries were
not separately counted. Expected labels were never supplied to the model.

| Protocol observation | Declared action / actor / calendar checks | Required causes | Typed durations / legacy impact checks | Gate |
| --- | --- | --- | --- | --- |
| Selector protocol, 14 cases | 68/68 / 68/68 / 67/68 | 6/6 | 7/7 / 2/7 | fails 5 cases |
| Duration compatibility, 16 cases | 80/80 / 80/80 / 79/80 | 8/8 | 9/9 / 9/9 | fails inherited date |
| Calendar protocol, 18 cases | 89/90 / 89/90 / 89/90 | 8/8 | 10/10 / 10/10 | fails omitted undated return |
| Normal default31, 4 cases | 20/24 each | 4/4 | 3/4 / 3/4 | fails every case |
| Normal default32, 6 cases | 33/36 each | 6/6 | 6/6 / 6/6 | English 3 cases pass, French 3 returns missing |

The development sequence adds three pairs of fixtures frozen before their
requests, then deliberately reuses cases after diagnosed changes. New stories
vary utility disconnections, elapsed hours/weeks/minutes and inherited dates.
They remain structural variations, not an independent multilingual contract
benchmark. All failed observations remain linked. No hidden replacement draw
was used to produce a passing result. Final optional negatives pass 30/30 causal
pairs, 20/20 forbidden duration checks and 14/14 unperformed-action checks.
Normal default32's 18/20 forbidden-duration checks fail for missing nodes, not
for an accepted foreign duration. Missing facts cannot count as validated absence.

The initial inherited-date output quoted "le même jour" rather than selecting
its antecedent's literal calendar value. The protocol now asks for the explicitly
linked original date, and unparseable relative values mark a partial receipt
instead of completing the field. Unknown date never permits nearby-date fallback.
Default provider typing also conflicted with undated occurrences. Correcting
that instruction recovered the English returns but did not solve French recall.
Some returned candidates still conflate unknown calendar with uncertain occurrence;
source matching cannot repair that model interpretation.

**Training/evaluation implications:** evaluate candidate recall, source span
recovery, actor versus recipient, explicit temporal association, elapsed duration,
occurrence status and causal direction as separate tasks. An original citation
can accompany an incomplete impact field. A correctly processed model inventory
can still omit an entire action: operational `completed` always retains
`completeness=unestablished`. Neighbor traversal cannot recover an omitted graph
node unless source coverage is audited independently.

For Laya, potential retrieval/ranking data should retain the full enclosing
original context, separate action/actor/date/scalar anchors and language. Build
hard negatives from a notice deadline presented as elapsed duration, a preceding
delivery date applied to a later undated return, model-spliced text, a recipient
presented as performer, and cause/effect reversal. Distinguish positive performance
from calendar availability. These are proposed dataset designs, not newly trained
heads. Unreviewed model output is not gold. Add out-of-family French and English
source material and reviewed real contracts before accepting a model quality gain.

Browser verification on the final default graph confirmed the unknown calendar
without a borrowed date, the return performer, the nine-hour duration, a separately
withheld effect, original relationship context in one disclosure, and traversal
from two steps (9 facts/13 links) to the full connected component (11 facts/20
links), with one registered original and no console errors. This is a scoped
functional test, not completeness or legal accuracy certification. Backend 1,082
and frontend 779 tests, production build, engine suites/race tests and patch
combinations are recorded in the shared receipt. An optional coverage rebase
initially reverted new default instructions; provider-request tests caught it.
The corrected extension preserves the entire default protocol and failed logs
remain available. Optional source-first extraction stays excluded from normal
builds/runtime. User data and edits were preserved.

No Laya/GLiNER weights, training, MLX acceleration, architecture or supervisor
qualification changed in this application audit.

See the [DelayLama source/graph audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-selector-duration-analysis.md),
[retained default32 semantic gate](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/assets/selector-duration-analysis/default32-graph.json)
and [shared receipt](validation/delaylama-selector-duration-analysis-20261005.json).

## L104 — Independent source coverage, graph date integrity and structured output (2026-10-05)

**Adopted application engineering:** DelayLama's normal deep analysis now runs an
independent original-source omission check before event storage and actor/relation
extraction. It passes the indexed original and already located action anchors,
without generated summaries or gold answers. It preserves initial candidates,
adds uniquely located missing actions, ignores identical anchors and withholds
any overlap instead of guessing semantic equivalence. It adds one bounded task
request, not an automatic application retry. Operational completion still retains
`completeness=unestablished`; skipped/partial/failed receipts are human-readable.
The experimental replacement inventory now shares its parser and bypasses normal
recovery; replacement remains off in the user stack.

**Measured:** all 42 accepted Gemini `gemini-3.1-flash-lite` jobs ended and were
archived. Initial recovery found 44/44 actions across eight synthetic cases,
including previously missed French undated returns. An intermediate ten-job run
exposed stale inferred dates on association edges although event nodes were
correctly undated. The fix copies audited event dates, not pre-audit temporary
metadata, and restricts copied properties to dates. SQLite/file storage and source
byte regressions verify this boundary. Ten new projects then passed 48/48 declared
checks, including 82 association edges.

Two actual two-chunk sources next failed both performer gates. In English, a
source-matched performer quote ended before the complete own-action anchor; the
actor link was withheld. In French, introductory prose preceding JSON caused the
legacy line parser to emit metadata fields as events and quoted actor strings as
parties. A correct source omission pass could recover the action, but polluted
endpoints still misdirected actor links. Structured output now accepts one complete
array with prose/a JSON fence, rejects malformed or multiple blocks, and never
converts its field lines into facts. Legacy prose requires a unique original
quotation. Partial actor premises can retain the contiguous original enclosure
only when they overlap the own action inside its existing source context and
contain the named candidate. Disjoint/invented premises remain rejected. Old
performer properties are cleared on reattachment.

The final twelve-job run passes 52/52 distinct source actions, exact actor sets,
performer citations and event/association calendars; 97 association edges were
inspected. Required causal links and literal own-action durations pass 6/6 each;
26 forbidden causal pairs, 36 forbidden durations and eight unperformed-action
checks pass. No JSON-field graph artifacts remain. Backend 1,081/frontend 780
checks, production build, final engine suite/race checks, twelve idempotent patch
combinations and ten ordered optional provider/intelligence/agent combinations
passed. The tested normal binary was activated while preserving user rows,
configuration, histories, sources, artifacts and Studio sessions. No existing
contract was reanalyzed or migrated.

**Evaluation boundaries:** the original performer quote gate wrongly required a
longer fixture recipient/modifier. Its correction requires the entire independently
verified node action anchor; source bytes/identity, actor sets, wrong-action and
clipped-anchor negatives remain mandatory. Its failed receipt is preserved.
Association-date and structured-field-artifact gates were added rather than
waiving the observed graph failures. A failed isolated restart left ten requests
on an older binary; those observations remain a separate phase. An initial
concurrent optional check is superseded by ordered checks after all patches
finished applying. New fixtures were frozen before first requests; later runs
repeat known cases. Long ~62 KB inputs entered the chunked branch with one chunk,
while ~92 KB/~121 KB inputs used two actual chunks. The smaller English fixture
has a French annex. Synthetic identifiers/length do not establish independent
real-contract gold, and fixture language is distinct from the test helper's
French-default project metadata.

**Implications for Laya training:** retrieval cannot repair a malformed structured
parser or stale graph metadata. Train/evaluate source coverage independently of
the combined summary/entity task. Keep byte spans and original qualifiers when
joining an actor premise to an action; a recipient-bearing extended action is a
useful hard case for quotation scope. Separate missing action, missing actor,
unknown calendar, unknown performance, unsupported effect and wrong causal
relation as different failure classes. Model-unreviewed generated output remains
candidate material, never gold. Useful negatives include JSON field/prose noise,
an actor quoted for a different action, a document date copied onto an undated
edge, and an overlapping bundled-action proposal. Preserve failed raw outputs,
source/schema versions and both evaluator protocols in any derived dataset.

The browser verified contextual highlighted quotations, human performer names,
uncertain calendars and source omission receipts. Entire-component traversal from
a recovered return read eleven facts/twenty links and one original. The final
French two-chunk graph contains two events/seven facts/thirteen links and the
correct actor, but its undated return still has model role `unknown`. The declared
gate does not qualify role interpretation, clause applicability, categories,
amendment effectiveness, conflicts or arbitrary long contracts. The 16 KB
relation envelope and source/input/output budgets remain explicit limits.

No new Laya/GLiNER weights, training, MLX architecture or supervisor qualification
was performed. These are measured application integration findings and proposed
training designs.

See the [DelayLama report and retained failures](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-source-action-recovery.md)
and [shared receipt](validation/delaylama-source-action-recovery-20261005.json).

## L105 — Performance is independent of calendar; source stages need separate receipts (2026-10-05)

**Observed (DelayLama):** the prior declared action gate checked identities,
performers, calendar values and links without requiring a positive performance
classification and its enclosing original premise. A read-only stronger gate on
archived default34 graphs found 37/52 declared performance proofs, with zero
model requests and zero graph writes. This does not revise the old gate's result;
it exposes a qualification dimension that was absent.

A reported return with an explicitly unknown date can be observed. An initial
parsed document date must neither block source classification nor reappear on
association edges. Clipped role quotations can match an adjacent date while
omitting their own action, conditions or negation. DelayLama's default patches
35–37 require an original own-action enclosure, allow exact duplicate inventory
proposals to reassess unknown undated roles, and add one separate original-context
classification task for remaining eligible unknown actions. Repeated/foreign
references and explicit conflicts are withheld. Calendar, actors and effects
cannot be invented by this classification task; inferred undated calendar
fallbacks are cleared. These are source-bound model candidates, not approved labels.

**Measured:** four additional 14-case full-stack phases produced 56 accepted,
terminal `done` jobs. Positive performance checks progressed 51/56, 55/56, 54/56,
then 56/56; all intermediate failures are retained. The final declared gate passed
56/56 distinct actions, performer sets/quotations, performance roles/original
premises, calendars and association-calendar checks. It inspected 111 association
edges, 6/6 required causes/durations/impacts, 28/28 forbidden causal pairs,
12/12 unperformed negatives and 36/36 foreign durations, with zero JSON-field
artifacts. Twelve cases were known regressions; two new synthetic cases were frozen
before requests. Reruns after fixes are not blind real-contract evaluation. Gold
was never sent to the model or seeded in graphs.

The run used Gemini `gemini-3.1-flash-lite`, one candidate pass, zero critic passes
and the construction-claims profile. Noop embeddings were confined to owned QA;
normal user embeddings remained Gemini. There were 56 inventory task requests and
two separate classification task requests, not a measured total pipeline/HTTP
count. Provider retries were not instrumented. Some fixture project language
metadata defaulted to French despite English originals and remains unqualified.

**Retained failure:** one of fourteen final inventory tasks returned an unreadable
expected structure despite native JSON mode. Its failed body was not retained,
so its exact shape and cause are unknown. A separate classification task resolved
an existing undated return. The browser displays both the failed source-coverage
receipt and completed performance receipt. Native JSON constrains syntax, not
semantic or output-schema validity. Full declared graph checks can pass while a
coverage task fails; never collapse these statuses into one success badge.

Backend 1,085 tests, frontend 783 tests, frontend build, the default Go suite,
intelligence/agents race checks and eleven optional-engine combinations passed.
Twelve patch combinations reapplied idempotently. The tested engine/API were
activated while preserving user graph rows, configuration, job history, saved
artifacts, Studio sessions and fourteen local edited engine source files. Existing
contracts were not reanalyzed. Owned QA services and temporary browser/clones were
cleaned after archiving handles and retaining private backups.

**Training/evaluation lesson:** positive and negative role labels need explicit
source-premise checks, separately from dates. Include passive returns with unknown
calendars, conditional actions, cancelled plans, actor/recipient distinctions,
wrong-role exact quotations and insufficient enclosing text. Require human
adjudication before converting model source candidates into training targets.
Retain stage failures and bounded failed outputs for diagnosis; do not train from
unlocated or conflicting proposals or infer full coverage from a completed graph.
Compare any Laya/GLiNER acceleration on the same approved premise suite before
adoption. This audit adds no new Laya quality or MLX training result. Real-contract
normative roles, dispersed premises, amendments and PDFs remain unqualified.

See the [DelayLama report and retained failures](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-performance-source-review.md)
and [shared receipt](validation/delaylama-performance-source-review-20261005.json).

## L106 — Diagnose output shapes, distinguish unused fields and preserve semantic counterexamples (2026-10-05)

**Observed (DelayLama):** a fresh long-French source inventory failed because its
1,538-byte Gemini response was a JSON action list rather than the requested
`events` object. Bounded failure diagnostics captured its exact bytes and digest;
this does not diagnose the older response whose body was unavailable. Whole-array
compatibility now uses the same source/role/count guards as the wrapper. Malformed
rows, prose/multiple values, invented quotations and over-limit inventories remain
withheld, without an application retry. Replay of the exact failed bytes on release
source completed with two located actions and zero network model requests.

Diagnostics retain at most 8,192 readable UTF-8 bytes in a scoped engine job
receipt. App history and evidence views receive only validated metadata and a
human-readable reason; failed text does not become graph facts or citations and
is not logged. Over-limit proposal lists are not retained in full, and their
wholly withheld counts preserve the receipt partition. Do not confuse an inventory
withheld before passage validation with a failed original quotation match.

Empty optional duration/effect values with populated line ranges had produced
spurious duration warnings. The indexed-source parser now treats these ranges as
unused when no scalar or explicit legacy premise was proposed. Nonempty scalars
retain their checks. This is not proof that the original has no duration/effect;
missing fields still require independent coverage assessment.

**Measured:** seventeen new accepted jobs ended `done`: a three-case diagnostic
phase followed by fourteen known synthetic regressions. The final declared gate
passed 56/56 distinct actions, performers/quotations, performance roles/premises,
calendars and association-calendar checks; 109 associations were inspected. It
passed 6/6 required causes/durations/impacts, 28/28 forbidden causes, 12/12
unperformed negatives and 36/36 foreign-duration checks, with no JSON artifacts.
Eleven final inventories completed, three remained partial and none failed. One
again returned an array and was accepted. There were seventeen inventory task
requests and three separate performance task requests, not a measured total
pipeline/HTTP call count. Provider retries were not instrumented.

Gemini `gemini-3.1-flash-lite` used one candidate pass, zero critic passes and the
construction-claims preset. Noop embeddings were confined to owned QA. Gold was
not model input or graph seed. Runs after changes are known-case regressions, not
blind real-contract qualification; English fixture language metadata still defaults
to French. Actual-model results used default40. Release default41 changes only
wholly withheld output-limit counts, validated through fault tests, full Go/race
suites, patch combinations and exact-response replay rather than another fresh
model run. Backend 1,088 tests, frontend 786 tests and build passed; twelve patch
combinations reapplied idempotently and eleven optional-engine suites passed.
Normal activation preserved user data/configuration and local engine source edits.

**Retained semantic failure:** fixed model output assigned "five days" to the
engineer's return in "Engineer returned the file. Inspector visited for five days."
A whole-line original premise contains both assertions, so the existing geometry
can accept a located duration **model candidate** without proving ownership. A
failed focused test and reproduction on final release source remain recorded;
zero network calls and zero graph writes were made by this counterexample.
The passing unmatched-literal regression does not qualify this ownership error.

**Learning:** never substitute quote location for scalar meaning or count only
cases where the first extractor already got the field right. Add independent
scalar ownership/ambiguity assessment, test omitted and conflicting duration
candidates, and require human-approved training labels. A clean output shape and
completed inventory do not certify legal applicability or complete coverage.
This audit adds no new Laya/GLiNER training or acceleration qualification.

See the [DelayLama diagnostic audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-source-response-diagnostics.md)
and [shared receipt](validation/delaylama-source-response-diagnostics-20261005.json).
