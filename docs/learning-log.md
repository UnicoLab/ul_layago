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

## L107 — Duration ownership needs a separate source task, not quote proximity (2026-10-05)

**Adopted engineering guard; synthetic model regressions measured.** DelayLama's
old quote geometry admitted an inspector's five-day duration as the engineer's
file-return duration when both appeared on one line. Unassessed durations and
dependent effect summaries are now withheld; candidates remain inspectable.
Independent whole-original review classifies ownership and elapsed/planned/deadline
kind, retaining several separate, byte-bound original attribution passages.
Duplicated original action/value targets share one check. Alternate source
candidates can recover an omitted first value; conflicting supported values are
withheld rather than silently overwritten or normalized.

The first 18-case full-stack run retained all 64 declared actions but withheld
three legitimate durations. Duplicate target handling and repeated clarification
wording caused protocol failures. Those failures are retained. The refined rerun
passed 64 action/performer/calendar checks, 12 independently assessed durations,
six impacts and six positive causal links; 38 negative durations, 28 negative
causes and 12 unperformed actions also passed. Eleven final source inventories
still reported partial coverage. These are known synthetic reruns, not complete
or blind real-contract qualification.

Eight source-only actual-model probes passed in each phase: English/French
cross-action rejection, dispersed context, planned time, a notice deadline,
unresolved dispute, and recovery of an omitted first candidate. Expected labels
never entered the model prompt and probes made no graph writes. Recorded tasks:
36 source inventories, three performance checks and 20 duration reviews across
36 full-stack jobs, plus 16 diagnostic duration probes. Provider HTTP retries and
total pipeline calls were not instrumented. Normal embedding/provider settings,
user graph records, saved sessions and local engine source edits were preserved.

**Learning for training and evaluation:** preserve actor/action occurrence identity
and original scalar spans separately from attribution labels. Include nearby
foreign durations, repeated clarifications, dispersed antecedents, different
kinds of time windows, disputes and missing first-extractor values. Quote presence
is not a semantic label. A rejected or unresolved assessment is not zero. Human
review is needed before using these observations as Laya/GLiNER supervision;
synthetic regression success alone does not qualify generated training data.

Backend 1,104 tests and frontend 789 tests/build passed. Engine suites, race checks,
12 idempotent patch combinations, 11 optional-engine suites and activation
preservation are recorded in the linked audit. Full contract coverage, real PDFs,
cross-document precedence, monetary scalars and free effect prose remain open.
No new Laya/GLiNER training, MLX or architecture qualification is claimed.

[DelayLama ownership audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-duration-ownership.md)
· [Shared receipt](validation/delaylama-duration-ownership-20261005.json).

**Additional retained operational finding:** the extended API race sweep timed
out after ten minutes because malformed complete-analysis bodies silently
started default reanalysis. Strict object/type/range/trailing-data validation now
returns 400 before model work or graph changes. The regular full API suite fell
from 150–177 seconds to about 18 seconds; corrected race results and the original
timeout are separate audit assets. Request-shape validation belongs before any
costly analysis or training action; a finished model pipeline cannot compensate
for unintended invocations.


## L108 — Relation identity must retain assertions, premises and source versions

DelayLama found that Chronograph silently invalidated any prior edge with the
same directed endpoints/type, while candidate/chunk/contract merging also used
only that tuple. Separate duties between the same parties and repeated original
sources could disappear. The corrected pipeline preserves assertion text,
original quote and source identity; explicit saved replacement commands remain
scoped, transactional on SQLite and distinct from legal supersession decisions.
Graph history stays available with operational labels for humans and the LLM.

The first English/French actual-model phase finished four jobs but returned no
converted concept relationships. The Gemini prompt did not declare endpoint
field names. After an explicit complete relation protocol, four known-source
replay jobs retained four matched concept assertions and two parallel source
pairs, without automatic invalidation. Failed and corrected phases are archived;
this does not establish a complete obligation inventory or legal correctness.

For Laya/GLiNER training and evaluation, a party pair/relation category is not a
unique assertion label. Keep conditions, exceptions, negation, occurrence identity
and versioned original premises. Graph invalidation/ingestion order must not be
used as gold for legal applicability. Train contradiction and applicability tasks
separately from entity pair matching, and calibrate abstention on ambiguous,
scoped and missing-premise cases. Prompt/schema omissions can mimic low model
recall; measure conversion and storage losses separately from extraction output.
No new classifier, retraining or acceleration result is claimed here.

The delivery passed 1,105 backend tests, 795 frontend tests/build, full Go and
scoped race tests, 12 idempotent patch combinations and 11 optional engine suites.
Real PDFs, normative completeness, ambiguous endpoint aliases and temporal query
timezone comparisons still need dedicated qualification.

[DelayLama assertion audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-parallel-assertions.md)
· [Shared receipt](validation/delaylama-parallel-assertions-20261005.json).


## L109 — Named endpoints, stored links and correct predicates are separate gates

Adopted engineering change, 2026-10-05. DelayLama reproduced a generic Organization
endpoint choosing the last organization, while unknown names could disappear at
storage. Chronograph patch 46 resolves exact names first and only unambiguous
aliases/types. Ambiguous, unknown and missing references are withheld with bounded,
source-hashed diagnostics. Initial concept storage separately reports missing
stored endpoints and write failures. UI counts distinguish event links from
concept connections and absent reports from zero.

Two frozen synthetic English/French sources produced three/two concept assertions,
all exact-name references, stored and source-matched. This is a conversion/storage
gate. The English certification quote still yielded an unsupported MANAGED_BY
predicate/direction; the French concept graph omitted several normative assertions,
even where separate Event candidates survived. A matched quote and correct named
endpoints do not establish the predicate, legal role, applicability or complete
inventory. Retain these failures when planning the next evaluation.

For Laya/GLiNER data: label entity identity separately from predicate direction,
modal force, payer/certifier roles, negation, conditions and actual performance.
Generic role/type labels need ambiguity negatives with multiple same-type parties.
Evaluate reference attachment, model recall, conversion survival, storage survival
and semantic edge accuracy separately. Train/evaluate abstention without treating
withheld references as gold source absence. Same-name homonyms/upstream merging
remain open. No new training, MLX or classifier-quality gain is claimed.

Final delivery passed 1,124 backend and 797 frontend tests/build, full Go and
intelligence/agents/API race checks, 12 idempotent patch combinations and eleven
optional engine suites. Browser review included a clearly separated manual
ambiguity fixture; it is not model-performance evidence.

[DelayLama reference audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-relation-references.md)
· [Shared receipt](validation/delaylama-relation-references-20261005.json).


## L110 — Preserve assertion meaning before aligning a relation vocabulary

Adopted engineering change, 2026-10-05. Eight Chronograph regressions showed
normalization reversing passive meaning, erasing negation or collapsing distinct
contractual predicates. Patch 47 preserves exact keys, uses a small equivalent
phrase table and otherwise formats the original label. Token/stem similarity and
preferred inverse aliases cannot decide assertion meaning. Versioned conversion
metadata is review evidence, not an entailment label. Source/target identity and
predicate direction must be evaluated independently.

Provider extraction and critic paths now treat relation vocabulary as suggested,
preserve modality/negation/conditions and request original premises. Cross-chunk
discovery receives original chunks, not a generated summary; provider budgets
still constrain coverage. Controlled provider tests verify request content, not
model adherence. The human graph review shows parties, precise predicate and
original quote, with expandable conversion details and warnings for unknown types.

Known English/French synthetic full-stack replays returned four of five expected
propositions each: appointment, conditional payment duty, certification and access
duty. Both omitted the separate engineer-does-not-pay relation. A separate A1/C1
critic probe recovered the English negative relation, but French still missed it.
English's negative quote needs an antecedent in the preceding original. These
are development observations, not an independent efficacy benchmark. Normative
and negative statements also survive as warned Event candidates, and a performer
link was observed on a negative candidate; an actor mention does not establish
actual performance. No global critic configuration or existing user graph changed.

For Laya/GLiNER training and evaluation, separate: entity/reference identity,
directed predicate, polarity, modal force, actor versus duty-holder/certifier/payer,
conditions/exceptions, quote scope and antecedent binding. Include passive/active
pairs, same-endpoint assertions with different qualifiers, and negatives against
closest-vocabulary collapse. Predict typed proposals/abstention with original span
support; keep normalization/calibration outside semantic truth. Measure source
assertion inventory recall separately from quote location, conversion survival,
storage survival and legal applicability. An extra critic call is an intervention
that needs language/family-specific evaluation, not an automatic quality guarantee.
No new training, MLX or classifier-quality gain is claimed here.

Final delivery passes 1,133 backend and 810 frontend tests/build, full Go and
race checks, twelve idempotent patch combinations and eleven optional suites.
The tested engine/API are active without changing existing sources, artifacts,
Studio sessions, configuration, graph rows or job journal.

[DelayLama predicate audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-relation-predicates.md)
· [Shared receipt](validation/delaylama-relation-predicates-20261005.json).


## L111 — Source polarity and modality must precede occurrence-role inference

Adopted engineering change, 2026-10-05. A source-located contractual duty and an
explicit negative assertion could survive as Event candidates and receive actual
performer interpretation. DelayLama patch 48 asks the existing original-source
inventory for a separate meaning classification: reported action, designation,
standing function, duty, permission, prohibition, negative/conditional statement,
plan, deadline or unknown. A generic recurring function must not be silently
strengthened into an obligation. A calendar is independent of whether work occurred.

The engine binds candidates to the own original action/context/hash; this verifies
location, not entailment. Normative/negative/conditional statements have separate
node types, no occurrence dates or performer metadata, and are excluded from
actual event-role links. Conflicting duplicate classifications become unknown;
missing/invalid kinds remain explicitly unassessed. Existing user graphs are not
silently rewritten. The UI labels statement reviews correctly, excludes them from
chronology and does not request performers for source-bound plans/deadlines.

Two three-case synthetic full-stack rounds passed the same twenty predefined type
checks. English/French were known development replays; the mixed case was new in
this continuation. Final storage retained seventeen statements, two performed
Events and one plan with correct own quotes/scope. No statement acquired actual
role links or dates. Mixed recall varied: two explicit no-report qualifications
were separate assertions in round one but only context in round two. A report
recipient's NOTIFIED edge does not certify a formal notice. These are bounded
engineering observations, not a multilingual model-quality or completeness result.
Full delivery passed 1,135 backend/816 frontend tests/build, Go/race checks,
twelve patch combinations and eleven optional suites. Actual browser checks
caught and fixed misleading statement Event badges and a planned performer gap.

For Laya/GLiNER training: supervise original-span identity, polarity, modality,
parties, conditions/antecedents, date role and applicability as separate targets.
Use minimal pairs (must/may/must not/did not), generic functions versus performed
acts, passive voice, conditional scope and explicit plans. Do not label a normative
subject as an actual performer or treat a quoted condition as established truth.
Evaluate occurrence precision, normative inventory recall and statement-to-party
attachment separately. Preserve independently frozen negative checks and source
qualifications so an improved type score cannot hide omitted assertions. Span
matching and nearest embeddings cannot substitute for semantic validation.
No new Laya training, MLX or classifier-quality gain is claimed by this change.

[DelayLama statement audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-source-statements.md)
· [Shared receipt](validation/delaylama-source-statements-20261005.json).

## L112 — Separate assertion discovery from statement-party attachment

Adopted engineering change, 2026-10-05. Adding party roles to the action inventory
increased task interference: the first four-case replay passed 26/26 types but
39/44 roles, with one extra role; a larger combined prompt regressed a designation
to an Event and still passed only 39/44 roles. Preserve these failures when
assessing an architectural change, rather than reporting only the best replay.

DelayLama patch 49 instead runs one bounded independent role task over the full
original and already typed statements. It cannot create/reclassify actions or
alter dates/performance. Subject, recipient, conditional approver and referenced
party have separate directed statement-to-party edges. Passive payment must retain
its complete governing predicate; a Vendor-submission modifier alone does not
represent the Buyer's payment duty. Invoice ownership can be explicitly established
in one article and reused by payment, review and negative-payment assertions in
later articles. Requests need their own direction: a party asked for corrections
is the request's addressee, not recipient of the resulting corrections.

Original line premises are reconstructed, uniquely located and bound to the own
assertion, named party/antecedent and original hash. Identity accepts exact names
or unique explicit aliases, then checks stored scope/type against a frozen concept
register. Neither original span presence nor unambiguous identity establishes
semantic role correctness or fulfilled approval. Missing/null/duplicate rows,
withheld proposals and storage failures have distinct receipts. Neighborhood
assembly must retain readable role and review metadata end to end; a correct
stored edge is ineffective if the context adapter drops its meaning.

The initial independent pass passed 32/33 types and 48/57 roles, with two extra
roles. Keeping complete own predicates and explicit article/invoice antecedents
then passed 33/33 type and 57/57 role checks in direct replay, restored full-stack
replay and the canonical delivery build. All five cases are synthetic development
fixtures, not an independent benchmark; four were known and the French article
chain was new when frozen. The final 79-node/117-edge graph retains quoted party
links without normative occurrence dates or actual performer edges. Full delivery
passed 1,155 backend/824 frontend tests/build, full Go/race checks, twelve patch
combinations and eleven optional suites. Existing user graph and work were
preserved. Broad event category accuracy remains unqualified: a French report
delivery still had an `approval` category candidate, separate from its correct
observed-action typing.

For Laya/GLiNER experiments, supervise complete predicate boundaries, polarity,
modality, own-assertion identity, antecedent chains and directed roles separately.
Include passive/active pairs, payments versus invoice review, negative payment,
requests, competing issuers, named approval conditions and ambiguous aliases.
Evaluate statement inventory recall, role precision/recall, source binding,
reference resolution and storage survival independently. Hold out contract
families and article arrangements before model/prompt tuning; repeated synthetic
replay cannot establish generalization. Train explicit abstention and preserve
qualification scope. No new training, MLX or classifier gain is claimed here.

[DelayLama statement-party audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-statement-parties.md)
· [Shared receipt](validation/delaylama-statement-parties-20261005.json).

## L113 — Keep source categories, qualifications and merge identity separate

Adopted engineering change, 2026-10-05. DelayLama's batch extraction called a
French report delivery approval while the source inventory proposed delivery.
Correct source discovery was ineffective because merging discarded its category.
Default patch 50 binds the own-action category to the original hash and byte span,
retains earlier disagreement, and keeps out-of-profile labels rather than choosing
a nearby vocabulary item. Conflicting source proposals stay unresolved. These are
model candidates; agreement and source presence do not establish legal meaning.

Quote identity is an independent failure mode. A leading calendar wrapper or
terminal full stop prevented otherwise identical source occurrences from matching.
The correction tolerates only those literal wrappers at the same original span,
without rewriting stored evidence. Question/exclamation marks and semicolons
remain significant. Partial verbs, negation, modality and qualifications must not
be stripped to manufacture equivalence. Different occurrences remain separate.

Explicit assertions that submission does not establish approval, or approval does
not establish completion, were initially omitted as mere context. Asking for them
as separate negative assertions improved these frozen checks. On seven synthetic
English/French cases, the first full-stack run passed 45/57 source-bound type,
57/57 statement-party and 6/12 category checks. Two corrected runs passed 57/57,
57/57 and 12/12, including the final delivery build. Five cases were known; two
were new when frozen. These repeated development replays are not an independent
quality benchmark. All failed observations and 21 terminal model jobs are retained.

Source-bound category disagreements appear beside the exact action and receive
review-queue priority. Eleven live neighborhood checks confirm that readable
category/review metadata reaches model context without raw receipts or vectors.
Delivery passed 1,156 backend/836 frontend tests/build, Go/race checks, twelve
patch combinations and eleven optional suites. Existing user data was preserved.

Category improvement does not establish event-role recall. The final French
delivery still withheld a proposed performer because the model selected the event
premise identifier where the performer protocol required its own identifier.
Preserve the failed selection and empty role slot; do not weaken source binding to
inflate recall. Long documents, independent families, cross-document amendments,
exceptions, applicability and complete prompt budgets remain unqualified.

For Laya/GLiNER experiments, separate category prediction, statement polarity,
qualification recall, span identity, endpoint resolution and typed role selection.
Use delivery-of-approval-request versus approval, qualified approval versus work
completion, negative qualification pairs, punctuation minimal pairs and repeated
occurrences. Evaluate pre-merge discovery, post-merge classification and stored
graph survival separately. Hold out contract families before tuning. No new
training, acceleration or classifier-quality gain is claimed by this change.

[DelayLama category audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-event-categories.md)
· [Shared receipt](validation/delaylama-event-categories-20261005.json).

## L114 — Evaluate source discovery, role selection and graph survival separately

**Measured engineering repair, 2026-10-05; synthetic development replays.**
DelayLama's previous French report example already had a valid named performer
and original passage, but the relationship protocol rejected selection of the
event's own registered context instead of its narrower performer identifier.
Accepting either engine-issued reference now requires the same uniquely located
complete actor/action premise inside that event. Copied text never rescues an
unknown or foreign selector. The source hash, occurrence, scope, actor identity,
date role and whole original quotation remain separate gates.

A later run supplied four correct French actor quotations that enclosed the
selected premise with an original date prefix. A unique enclosing quote is now
preserved only inside the same event's registered context. Partial overlap and
other-event context remain rejected. Treat these failures as selection/provenance
protocol failures, not as missing discovery or evidence of better model semantics.

A new joint-inspection fixture found a downstream storage failure: the inventory
already classified a roof survey as planned, but a title keyword demoted its
stored node to an Entity and lost plan metadata. Source-classified plans/deadlines
now retain their individual Event identity, review explanation and calendar role
without actual-performance links or occurrence dates. Score discovered candidates,
classified candidates and surviving graph assertions separately; a training
change cannot repair facts lost by later storage logic.

Across four cohorts, the seven known synthetic English/French cases passed
56/57, 57/57, 57/57 and 57/57 expected statement-type checks; named performers
passed 12/12, 8/12, 12/12 and 12/12. Abstentions passed 2/3, 3/3, 3/3 and 3/3.
The first run incorrectly assigned an ambiguous French approval request to the
letter courier. A prompt clarification requested abstention, but repeated success
on this case does not establish deterministic semantic safety. The four missing
actors in cohort two were the enclosing-quotation protocol failure. All failed
observations remain retained rather than being relabelled as success.

The additional fixture was new when frozen, then reused in cohorts two to four.
It passed 7/8, 7/8 and 8/8 statement-type checks, with all three performer groups
(four named assignments) correct each time. Its missing plan was a graph-survival
failure. The final build passes 65/65 combined type checks, 57/57 statement-party
checks, 12/12 action-category checks, 16/16 named performer assignments and 4/4
abstentions. These are bounded fixture expectations, not independent recall or
semantic/legal correctness scores. Thirty-one terminal model jobs are archived.

Five actual application neighborhood requests and browser checks preserve exact
highlighted originals, directed role links and readable context without vectors
or raw receipts. The final release passed 1,163 backend tests, 836 frontend tests,
the production build, full Go and race suites, twelve patch combinations and
eleven optional Go suites. Existing user graph rows and saved work were preserved.
Long/OCR contracts, unseen families, amendments, exceptions, applicability and
whole-request prompt budgets remain unqualified.

For Laya/GLiNER training and evaluation, use separate targets for source span,
statement kind/polarity, planned versus observed date role, actor identity,
directed participant role and evidence/context selection. Include explicit
abstention cases, letter courier versus author/requester, joint performers,
same-day distinct actions, duties without reported performance and planned
actions with an explicit non-performance assertion. Preserve minimal pairs and
qualifiers. Hold out contract families before prompt or head tuning; maintain
separate pre-storage and stored-graph metrics. These are proposed training uses;
this repair measures no new training, MLX acceleration or model-quality gain.

[DelayLama performer/plan audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-performer-context.md)
· [Shared receipt](validation/delaylama-performer-context-20261005.json).

## L115 — Reference retrieval cannot compensate for a failed extraction pass

**Measured context improvement and extraction failure, 2026-10-05.** DelayLama's
read-only neighborhood adapter now follows explicit numbered Article/Clause/
Section references directly in registered originals when the graph omits their
links. Unique source-bound targets can be followed recursively; cycles terminate,
parent passages retain nested exceptions and competing amendment headings remain
ambiguous. Document-qualified references and unsupported layouts need review.
This locates context without asserting semantic graph relations, precedence or
legal applicability. It is not a new Laya or GLiNER model measurement.

Two manually seeded synthetic projects each had one payment node and no graph
connections. The actual application API and browser retrieved Article 7, its
Article 7.1 exception and Article 9 from one original, then exposed both competing
Article 7 passages from a contract and conditional proposed amendment. Lookup
preserved the graph and made no model call. Wrong-document seed evidence did not
start expansion. Frontend validation maps original Unicode character boundaries
to browser UTF-16 positions and checks quotes, source identity/hash and bounds.
Astral-Unicode tests prevent shifted highlights or rejected valid citations.

A new two-document synthetic C35 fixture, frozen before model submission, found
an important separate failure. Both engine jobs completed, but the contract's
native-JSON source inventory returned extra closing structures. Strict parsing
rejected it and preserved the failed diagnostic. Only **4/13 expected statement
types** survived in the graph. The inspection actor and proposed-deadline
non-occurrence checks pass; the base deadline is absent. Original reference
lookup cannot restore missing structured extraction, and a completed job is not
a complete analysis. This failed development fixture remains a failure, not a
new quality benchmark or evidence of improved extraction recall.

The interface now highlights failed original-source checks outside collapsed run
details and opens their saved explanation without submitting again. Validation
passed 1,180 backend tests, 848 frontend tests and the production build; no engine
source changed or Go/race suite was repeated. User data and configuration were
preserved; both terminal model handles are archived. The first failed diagnostic
SHA-256 is `294896df026a7d0befbb04d00c30a26fb02e37950f4d266174f037f25d934998`.

Future training/evaluation should separate explicit reference detection, exact
source-span binding, target disambiguation, topical retrieval, statement discovery,
structured-output validity and graph survival. Use duplicate clause labels,
conditional unsigned amendments, negative qualifications, nested exceptions,
reference cycles, wrapped document qualifiers and Unicode minimal pairs. Preserve
the first failed response when testing recovery and keep the original gold.
Bounded format recovery is proposed work, not a measured success here. Long/OCR
contracts, implicit topics, plural/annex syntax, whole prompt budgets and legal
applicability remain unqualified. No new training or acceleration gain is claimed.

[DelayLama clause-context audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-clause-context.md)
· [Shared receipt](validation/delaylama-clause-context-20261005.json).

## L116 — Recover structure without certifying discovery (2026-10-05)

DelayLama now makes at most one fresh retry for an unreadable source inventory.
It preserves the first failed response, uses the same complete original and
source guards, and shares one 60-second deadline. Initial and retry prompts are
checked as complete inputs. No failed prefix is salvaged and no failed text,
graph facts or gold labels enter the fresh task. Cancellation/budget skips and
failed retries remain explicit; transport, source-location, semantic and output
limit failures do not trigger another request. This is engine/protocol work,
not a new Laya or GLiNER quality result.

The unchanged C35/unsigned-amendment development fixture reproduced the exact
first malformed 4,225-byte response, then exercised one readable retry. It
improved from 4/13 to **12/13 frozen statement-type checks**. Both deadline
non-occurrence checks and the inspection actor pass. Sixteen located/classified
inventory candidates do not imply all source assertions were discovered:
**“No completion is reported” is still absent as a separate negative statement.**
The base deadline's explicit date is still withheld because its model-selected
date passage was omitted. The gold was not changed. Full semantic qualification
still fails, and this is not an independent benchmark.

Twenty stored action/statement candidates and thirty displayed connections have
unique exact quotations, source hashes and original spans. This is a source
binding measurement, not validation of the predicate, meaning, applicability or
recall. The permission context exposes original Article 9, competing Article 7
passages, nested exceptions and the unsigned conditional amendment; ambiguity
remains explicit. The live UI distinguishes a readable retry from a completed
source/meaning check and keeps raw failed output outside supporting evidence.

Validation passed 1,201 backend tests, 851 frontend tests/build, full engine and
race checks, 12 idempotent patch combinations and 11 optional variant suites.
User data, configuration, source edits and journal/history remain preserved.
Two terminal QA engine handles are archived. Existing user contracts were not
reanalyzed. No new training, acceleration or architecture improvement is claimed.

For reusable extraction training/evaluation, keep separate gates for response
validity, source identity, assertion discovery, polarity/modality, own-action date
premise selection, party roles, graph survival and applicability. Include short
negative statements adjacent to deadlines, absence-of-record vs positive
occurrence, separate approval/completion denials, conditional unsigned amendments,
explicit dates with missing/wrong ranges, Unicode and dispersed clause references.
Measure missing assertions rather than conditioning recall on returned rows.
Preserve first failures and frozen expectations when evaluating recovery. Use
independent contract families before attributing an extraction gain to Laya.

[DelayLama recovery audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-source-format-recovery.md)
· [Shared receipt](validation/delaylama-source-format-recovery-20261005.json).

## L117 — Pair calendar premises; measure discovery and connectivity separately (2026-10-05)

DelayLama's source inventory now requests every assertion field through an optional
native Gemini JSON schema. Calendar text and its own original passage selector
form one nullable object. The complete prompt plus schema is budgeted initially
and for the existing bounded format retry. Required primitive fields, extra and
duplicate keys are checked locally before unchanged own-action source guards.
Unknown calendars do not borrow nearby dates. General absence-of-record
instructions ask for independent negative assertions without feeding evaluation
gold into the model task. This is protocol/extraction work, not a new Laya result.

The first four accepted source-inventory jobs failed provider requests and remain
archived. C35's retained first graph passed only 2/13 frozen type checks; the
additional English/French cases each passed 1/7. A controlled compatibility probe
accepted the nullable schema only after removing its provider array cap. The
unchanged engine still completely withholds more than 120 rows without retry or
truncation. Generic HTTP 400 responses do not prove an internal grammar cause or
that array limits are universally unsupported. Preserve rejected implementations
and compare identical fixtures rather than reporting only the successful draw.

The corrected full-stack variant passes **13/13 unchanged C35 type checks**,
including the previously missed completion negative. Its exact original deadline
calendar is restored through paired premises; that separate calendar diagnostic
was not part of the thirteen frozen type targets. Both deadlines remain
non-occurrences and the inspection retains its own engineer performer. Two new
translated synthetic development cases, frozen before their first submission,
each pass **7/7 type checks**, both date targets, actor and deadline non-occurrence
checks. These small cases are not independent benchmarks. Every final inventory
returned structure in one inventory call; all source audits remain partial.

Seventeen C35, seven English and ten French returned candidate citations, plus
44 connection citations, bind uniquely to current originals with exact hashes
and spans. This says nothing about extra candidate quality, missing assertions,
predicate meaning, applicability or relationship recall. Neighborhood reads
preserve all graph snapshots. The UI exposes the recovered negative separately,
its highlighted original, the deadline calendar and readable schema diagnostics.
It does not present request validity as complete analysis.

**A whole component is still not a whole contract.** The permission component
contains only eight of seventeen extracted source candidates; nine are outside,
including the isolated completion negative. The source reference layer retains
ambiguous Article 7 passages and a qualified amendment reference, but graph
connectivity cannot yet supply every relevant dispersed premise. Next work must
measure missing source/section/topic connections separately from fact extraction.
Structural adjacency must remain distinct from asserted dependency, causation,
contradiction and legal precedence; do not fabricate semantic links to fill a
connectivity metric.

For reusable training/evaluation, predict calendar value and source selector as
one target, train explicit null/abstention cases, and independently test another
action's calendar, missing range, Unicode offsets and date roles. Keep separate
response-format, assertion-recall, polarity, role, premise ownership, graph
survival, structural-context and directed-predicate gates. Include untranslated
independent long/dispersed/OCR families before claiming transfer or attributing
quality gains to a fast classifier. A strong model cannot repair a pipeline that
omits context or assigns source text to the wrong assertion.

Validation: 1,207 backend tests, 852 frontend tests/build, full Go/race checks,
twelve idempotent patch combinations and eleven optional variants passed. User
graph/configuration/journal/reviews/source edits remain unchanged; failed and
final QA handles are archived. No existing user contract was reanalyzed. No new
Laya training, MLX performance or architecture result is claimed.

[DelayLama structured-source audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-structured-source-inventory.md)
· [Shared receipt](validation/delaylama-structured-source-inventory-20261005.json).

## L118 — Separate source navigation from semantic truth (2026-10-05)

DelayLama's previous graph extracted seventeen source assertion/action candidates,
but only eight were reachable from the payment permission. A deterministic source
outline now links each located stored finding to its original section/document.
The controlled replay on a copy of that same graph expands membership from
**8/17 to 17/17**, without calling a model or modifying any existing fact or edge,
including properties and timestamps. Eleven source anchors and thirty-eight
containment links are added. Measure this as structural reachability, not recall,
semantic relationship accuracy or understanding of every relevant contract term.

Structural provenance, original episode/project/tenant identity, source hashes
and exact UTF-8 byte boundaries remain explicit. Unique own-action/source quotes
gate containment; foreign, stale, repeated, fabricated or unresolved quotes are
withheld. Nested headings retain distinct source positions. Source document and
section nodes cannot be entity-merged across documents just because labels match.
Missing parent storage stays partial instead of quietly assigning a different
parent. Per-analysis opt-in is captured independently for queued jobs, avoiding
shared mutable flags. The application independently validates current originals,
spans and endpoint bounds, while leaving heading interpretation unqualified.

Fresh full-stack model checks retain **13/13 frozen C35 type checks** and **7/7
English/French checks** each, expected dates, inspection actors and non-occurrence
deadlines. All four structural passes complete with exact original spans. The
fresh C35 graph has one additional semantic node/edge compared with the baseline;
do not credit this model variability to structural storage. The controlled replay
isolates the implementation's actual contribution. All source inventories remain
partial and these small translated synthetic cases are not independent benchmarks.

Source anchors/containment are hidden from the main fact list and semantic review
queue by default. Surrounding context exposes them separately, retains highlighted
original quotes and keeps competing or document-qualified clause references
unresolved. A full component can contain irrelevant statements; no structural
path proves dependency, causation, applicability or amendment precedence. Model
context retains structural ranges while original bodies are supplied separately,
avoiding document copies in every containment quote. Large-context budgets and
relevance selection still need qualification.

For LayaGo training and evaluation, maintain separate targets for formatting
boundaries, original span selection, assertion discovery/classification, premise
ownership, structural reachability and directed semantic predicates. Synthetic
families should include parallel clauses, repeated headings across documents,
unheaded text, nested exceptions, unsigned amendments, explicit and implicit
references, absence-of-record statements and unrelated nearby clauses. Freeze
expected records and relationships independently. Do not train a classifier to
turn proximity or containment into a causal label; include abstention and hard
negative pairs. A fast classifier may select context candidates, but this result
provides no new model training, accuracy, MLX or speed measurement.

Validation passes 1,231 backend tests, 854 frontend tests/build, Go/race checks,
twelve idempotent patch combinations and eleven optional suites. The real browser
checks semantic/structural separation and readable evidence. Existing user graphs,
configuration, journal, reviews and source edits are unchanged; no historical
backfill or reanalysis occurs. A separate source-only backfill workflow, independent
long/dispersed/OCR contracts, implicit topic links, predicate recall, legal
applicability and whole-context budgets remain work items.

[DelayLama source-structure audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-source-structure.md)
· [Shared receipt](validation/delaylama-source-structure-20261005.json).

## L119 — Backfill source context independently of extraction (2026-10-05)

Existing graphs need a source-only upgrade path: rerunning a model to add
formatting/navigation also changes extraction outputs and makes causal attribution
harder. DelayLama now previews current registered-original episodes and explicitly
builds their source containment. It pins the complete graph and original versions,
persists intent before submission, and reconciles lost responses by scoped
request identity. It sends episode/hash/request identifiers rather than document
bodies or prompts. A model-disabled engine performs all four tested jobs.

On a copy of the earlier C35 graph, the permission's reachable assertion/action
candidates expand **8/17 → 17/17**. All 29 stored findings connect; eleven source
anchors and thirty-eight containment links are added. English and French
record-negative copies connect 14 and 19 findings respectively, reaching all seven
and ten source candidates. Every original node, edge and episode is unchanged.
All 100 structural objects across the three fixtures independently locate against
current registered originals. These observations measure structural navigation,
not discovery of omitted assertions, semantic relationship accuracy or complete
understanding. No new model extraction, training, acceleration or speed gain is
measured; earlier frozen extraction results remain historical evidence.

Treat source-document identity, exact content, episode/project/tenant ownership
and spans as separate invariants. Check ownership/content after waiting for worker
capacity as well as before accepting a job. Exclude stale original versions and
refuse partial snapshots; never count malformed receipts as grounded evidence.
Retain partial writes/outcomes visibly. Reusing an accepted request must return
its existing handle. A lost or foreign handle must not trigger automatic model
work or silently become a completed result. Application journals survive page
reopening, but current engine workers are in memory: after engine restart, absent
handles remain unknown and durable worker recovery is still a work item.

The real browser verifies preview, explicit build, saved status/results and
identical handles on page reopening. Source context has its own readable card,
while structural containment remains separate from semantic review. The full
checks pass **1,249 backend tests**, **868 frontend tests/build**, Go/race suites,
twelve idempotent patch combinations and eleven optional variant suites. The
checked main stack is active and user graph/configuration/journal/reviews/source
edits are preserved. User contracts are not automatically backfilled.

For future LayaGo data and evaluations, keep structural reachability labels
separate from assertion recall, directed predicate labels and applicability.
Add hard negatives for nearby unrelated clauses, parallel statements, duplicate
headings and unsigned amendments. Train abstention when a predicate cannot be
established; a classifier should not turn containment into causation. Independently
qualify long/dispersed/OCR originals and context budgets before claiming broad
contract gains. This result closes the historical source-backfill workflow gap,
not those remaining semantic and model-quality gaps.

[DelayLama source-only backfill audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-source-structure-backfill.md)
· [Shared receipt](validation/delaylama-source-structure-backfill-20261005.json).

## L120 — Match original versions separately from surviving quotations (2026-10-05)

A short quote can survive an important document change. DelayLama's neighborhood
adapter also read a node-style source ID for relationships, ignoring their actual
episode ID. With two versions of the same source, an old relationship could then
look current. A controlled graph-copy/API comparison reproduces the old located
result and the corrected outdated result with identical assertion and quote.
The browser retains highlighted text next to the changed-version warning; no
model or graph write is involved.

Bind nodes and edges to their exact episode fields. Treat episode `source` as
identity and `name` as display text. Require project/tenant ownership and reconcile
explicit citation references with the bound episode. Do not select a different
original because it contains matching prose. Missing episodes, conflicting
references and unreadable hashes remain unavailable/unmatched; repeated text
remains ambiguous. An unbound legacy record cannot borrow the sole visible
episode's version. Source structure also independently verifies episode scope.

Indexed quote episodes need a separate representation: their content is an
excerpt, so its hash cannot be compared with the whole original's hash. Recognize
scoped extraction-source metadata and require the quote to occur in its own
excerpt and its registered original. New upserts pin the full original hash and
Unicode character bounds. Validate current excerpt boundaries independently.
Legacy excerpts remain readable with unknown original versions; changed original
hashes remain outdated even when the quote itself is unchanged. Corrupt span/hash
metadata cannot establish a current original version.

Three persisted indexed-excerpt fixtures pass live API/browser checks for legacy,
current and changed originals. The full source component separately shows 49
current matches and one changed source version. English/French copies retain
25/31 current semantic evidence candidates and 23/28 structural objects. These
are provenance checks on engine-shaped graph copies, not new extraction recall or
directed semantic-relationship measurements. Transport arguments from the actual
quote-upsert helper are checked with a mock client, including Unicode offsets.

Thirteen initial regressions fail against the old adapter. Final validation passes
**1,284 backend tests**, **875 frontend tests/build** and read-only full-stack
checks with no engine LLM/jobs. Review summarizes current/changed/unknown/missing
source evidence, displays readable enum values and hides internal IDs. User
records/configuration/journal/source edits remain unchanged after API refresh.
No new model training, MLX, classification-speed or architectural gain is claimed.

For LayaGo data, separate quote match, occurrence identity, original-version
binding, classification and predicate labels. Include hard negatives where the
quote stays fixed but a distant exception, consent requirement or unsigned
amendment changes. A classifier should not certify current applicability merely
from a surviving quote or nearest passage. Keep excerpt/full-document hashes
distinct. Independent long/dispersed/OCR cases, other result adapters, predicate
recall, relevance budgets and legal applicability remain qualification work.

[DelayLama context source provenance audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-context-source-provenance.md)
· [Shared receipt](validation/delaylama-context-source-provenance-20261005.json).

## L121 — Share provenance checks across every graph consumer (2026-10-05)

A correct neighborhood verifier is insufficient if the inspector, chronology or
review queue still accept any quote found in joined documents. DelayLama's main
graph had that gap: an older event could appear source-matched and be plotted,
while an older performer link suppressed the actor investigation. Fifteen new
frontend regressions reproduce the previous behavior and pass after integration.

The scoped graph API now supplies read-only evidence receipts from the same
exact-episode, project/tenant, original-reference, occurrence/version and indexed
excerpt verifier. Batch-fetch missing referenced episodes using scoped pages;
reject foreign/duplicate pages and ignored offsets. Missing or unavailable
provenance must stay unverified, including direct-engine fallback data. Detect
overlapping repeated quotations as well as ordinary repeats. A stored name/span
can remain navigation text without becoming proof of its source version.

Inspect the connection and each endpoint against their own original and captured
version. Keep surviving quotes readable with surrounding context, but distinguish
current, changed and unknown source versions. Withhold unverified dates and date
roles from chronology/CSV. A stale performer edge must not satisfy the review
queue's current actor-link coverage. Retain model proposals and historical counts
as proposals/checks at analysis time, with current-version review flags.

Validation passes **1,295 backend tests**, **890 frontend tests across 74 files**
and the production build. A copied C35 graph has 98 current, three changed and one
unknown node/edge evidence records, including containment objects; English/French
copies have 48/59 current records. These are provenance checks, not fresh extraction
recall or semantic relationship measurements. Full-stack browser checks retain
readable citations, withhold the old inspection date and keep performer
investigation available. No model inference or graph mutation occurred; temporary
services/tabs were closed and the user graph/artifacts/configuration were preserved.

For reusable LayaGo evaluation, test the complete downstream decision path rather
than only the classifier output. Separate text match, occurrence identity, source
version, directed role and applicability labels. Fixed quotes with changed distant
exceptions/consent/amendments are useful hard negatives. Version hashes and scope
are deterministic gates; a similarity score cannot substitute for them. Do not
count stale links as coverage positives or unknown versions as current. Maintain
independent long/dispersed/OCR and semantic-relationship benchmarks before claiming
extraction gains. No new training, acceleration or classification-speed gain is
claimed by this integration check.

[DelayLama main graph source provenance audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-main-graph-source-provenance.md)
· [Shared receipt](validation/delaylama-main-graph-source-provenance-20261005.json).

## L122 — Reuse exact-source indexes without widening acceptance (2026-10-05)

A correct source verifier can become too slow for long contracts if it rebuilds
Unicode or whitespace offset maps for every finding. DelayLama now captures one
original per scoped reader, lazily reuses its index, and carries the already-checked
own citation span into the inspector and chronology. Source identity, version,
quote, character bounds and repeat ambiguity remain separate checks. Mutating a
receipt invalidates its captured proof. Avoid global caches keyed only by document
ID, and keep derived lookup data outside model-facing facts and exports.

Local synthetic three-pass medians changed from **382.83 to 2.02 ms** for 500
frontend citations in a 106,285-character original, and **617.81 to 16.98 ms** for
300 backend quotations in a 49,584-character original. These are lookup kernels,
not whole-application/model-speed gains. Retained linear indexes trade memory for
less repeated work; process RSS does not demonstrate a memory reduction.

Validation passes **1,308 backend tests**, **905 frontend tests across 75 files**
and the production build. New regressions include 1,500 seeded Unicode differential
queries, repeated/overlapping quotations, source/version isolation and mutation
invalidation. A fresh isolated full stack serves a constructed 301-node/300-edge
long-document graph: all exact spans agree, older evidence remains flagged, late
facts load through pagination, a distant consent/negative-approval passage is
visible, and the old date stays off chronology. No model or graph write occurs in
these checks. User data/configuration/source edits are preserved after API refresh.

The QA health count disagreed with SQL/stats after direct SQL fixture insertion;
its cause remains unqualified. Actual scoped collections establish fixture counts,
not the health badge. Retain such discrepancies rather than treating a status
indicator as graph completeness or extraction accuracy.

For LayaGo and future extraction pipelines, benchmark deterministic grounding
separately from embedding, classification and semantic-relation quality. Reusing
exact-source indexes can accelerate a validator without accepting paraphrases or
substituting nearby documents. Independent long/dispersed/OCR evaluation, predicate
precision/recall and applicability remain outstanding; no new training, MLX or
architecture improvement is established by this check.

[DelayLama graph grounding and long-source lookup](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-source-index-performance.md)
· [Shared receipt](validation/delaylama-source-index-performance-20261005.json).

## L123 — Exact quotes and connected context do not validate predicates (2026-10-05)

Fresh DelayLama English/French dispersed-contract development fixtures retain all
23 frozen statement types, four performer sets and three deadline non-occurrences,
but only six of seven calendar/role checks pass. All 18 English and 11 French
stored source action/assertion candidates are reachable from their permission
seed. Neither result establishes complete semantic quality: the model inferred
causation from sequence, mapped two unsigned-amendment assertions to notice
requirements and included “Neither” in a party name. Fourteen concept citations
also retained a conflicting chunk-local original identity. These post-observation
diagnostics remain separate from the pre-request frozen expectations.

One original-source recheck, with two recorded calls, falsely endorsed the causal
link. The exact quote was real; its predicate was wrong. DelayLama now queues all
directed semantic candidates, including vocabulary-valid party/document links,
and exposes unresolved performance/date roles. Five regressions fail before their
respective fixes; 910 frontend tests and the production build pass. These are UI
repairs and fresh qualification findings, not engine semantic repairs.

The failed check, exact originals and a corrective human note reach the prompt
lab's server backup as two history entries, without activating a prompt or writing
the graph. Browser cache quota is reached; its visible warning remains unresolved.
The isolated services/tab are closed after terminal jobs, preserving user data.

For LayaGo training and evaluation, keep sequence-versus-explicit-cause and
unsigned-versus-notice-requirement as hard negatives. Separate predicate, direction,
modality, party boundaries and applicability from vocabulary membership, source
presence and connectivity. Chunk ownership, absolute offsets and hashes need
deterministic engineering checks rather than similarity guesses. Do not use an
LLM support verdict as independent semantic gold. This synthetic development run
establishes no new Laya training, MLX, speed gain or real/OCR generalization.

[DelayLama dispersed semantic analysis](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-dispersed-semantic-analysis.md)
· [Shared receipt](validation/delaylama-dispersed-semantic-analysis-20261005.json).

## L124 — A chunk is an extraction view of its original (2026-10-05)

DelayLama's next engine fix repairs a concrete source-ownership failure. Automatic
chunk extraction used `source:<document>#chunkN` citations while storage retained
an episode for `source:<document>`. Offsets could also locate only an entity name
rather than its longer supplied quote. Preserve the exact original byte interval
at splitting time, check each supplied quotation within its own chunk and rebase
it before merging. Do not infer original ownership from an arbitrary suffix or
unique matching text in another document. The explicit chunk API's known original
is its joined input, not an unknown pre-split document.

A controlled replay reconstructs the 14 concept candidates from L123's stored
synthetic evidence. Actual extraction/SQLite/containment changes current exact
grounding from 0/14 to 13/14 and links 13 findings to their own source sections.
The repeated Clause 88 remains ambiguous. This is not a new model draw or extraction
recall benchmark; the raw provider output is not reconstructed. Historical user
rows are not silently repaired. Source identity/span corrections establish no
improvement in semantic predicates, applicability or model classification.

Unicode-aware splitting preserves whole characters and complete source coverage,
and oversized overlap cannot stall advancement. Both chunk extraction paths,
forced fallback, foreign/missing/invented/repeated quotations and citation metadata
are covered. Full Go and focused race checks, 1,308 backend tests, 28 patch tests,
12 clean/idempotent patch arms and 11 optional variant suites pass. Model-disabled
full-stack inspection verifies all 13 current spans, their connected context and
the browser's visible ambiguity. The tested engine is activated with backed-up,
unchanged user graph tables/configuration and protected source edits.

For reusable LayaGo pipelines, make source intervals and coordinate conventions
explicit at every split/merge boundary. Keep provenance tests separate from model
quality. A source-matched wrong causal or notice predicate remains wrong; L123's
semantic counterexamples, entity naming and acknowledgement role are unresolved.
No new Laya training, MLX or architecture-quality result is established here.

[DelayLama chunk original provenance](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-chunk-original-provenance.md)
· [Shared receipt](validation/delaylama-chunk-original-provenance-20261005.json).

## L125 — Separate predicate witnesses from proposed graph meaning (2026-10-05)

DelayLama now inventories full-original relationships for registered endpoint
pairs without showing the proposed predicate, recorded fact, quotes or prior
verdict. Keep causal effect, chronology, notice obligation, notice delivery and
signature state distinct, with direction, negation and modality. Match original
identity and exact spans deterministically; treat a matching inventory as a model
candidate rather than proof of entailment. A second pass can share the first
provider's errors and still sees model-derived endpoint labels.

A controlled real-model/SQLite replay changes 13-case storage criteria from 4/13
to 13/13: four valid candidates retained, nine unsuitable candidates withheld.
The initial Event-only restriction reached 11/13 by overwithholding two valid
undated controls. Narrow the construction profile's guard to known standing
causes: a place/party/obligation must not substitute for an occurrence. Generic
Entity/Concept labels remain candidates, not certified occurrences. Preserve
failed draws and positive controls; a stricter validator can reduce useful recall.

Five native source jobs complete in an isolated full stack. Dispersed English and
French retain 23/23 frozen statement types and 6/7 date/role checks; reachable
context includes distant consent and unsigned-amendment passages. The first
positive source falsely stored a location as cause; the final native run retains
the actual closure→stoppage event link and withholds two standing proxies. Exact
quotes and connectivity alone never establish predicate correctness. A 12-case
source recheck remains 11/12 because an explicit no-duty inventory has unresolved
modality, and an acknowledgement role remains unknown. These synthetic development
checks are not independent legal gold or a general accuracy estimate.

Source-first statements, exact quotes and review history reach the prompt lab's
server backup without changing a graph or activating prompts. Neighbor navigation
is excluded from stable finding identity; authenticated legacy hashes remain
read-only. Operational counts include withheld candidates and verify original
hashes, but do not measure semantic accuracy. Full Go/race, 1,352 backend and 912
frontend checks, production build, 12 patch arms and 11 variant suites pass. User
data and source edits remain intact after controlled engine/API activation.

For LayaGo datasets, use separate tasks for kind, direction, polarity, modality
and endpoint role, preserving source/occurrence IDs and exact context. Retain
chronology-versus-cause, signature-versus-notice, delivery-versus-duty and
conditional-versus-observed hard negatives, with valid undated positive controls.
Do not train on automatic support verdicts as independent truth. Benchmark
precision/recall and escalation on held-out human-reviewed long/dispersed/OCR
sources before enabling acceptance. No new Laya training, MLX, speed or architecture
quality is established in this turn. Browser cache quota and broader predicate,
applicability and date-role qualification remain open.

[DelayLama source-first predicate audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-source-first-predicate-audit.md)
· [Shared receipt](validation/delaylama-source-first-predicate-audit-20261005.json).

## L126 — Date ownership, performance and document coverage are separate (2026-10-05)

DelayLama's independent performance pass previously skipped dated candidates even
when their role quote lacked the own action. Extend the separate original-context
task to those candidates, independently request date ownership, and retain the
existing calendar. Require exact registered enclosing context and preserve earlier
unresolved values on missing, invalid, duplicate or conflicting assessments.
A plan, deadline, condition or negated action can own a date without occurring.
A bare denial of performance is not evidence of a future plan. Keep an engine-only
review marker when a downstream domain classifier would demote a reviewed plan
into a standing identity; model-supplied flags must not authorize that override.

Fifteen frozen bilingual controls with injected raw proposals, real Gemini review
and production MemoryAgent/SQLite storage progress from 6/15 to 12/15, 13/15 and
15/15 as distinct implementation/prompt changes address dated skipping, ownership,
French plan classification and negation. Preserve every failed draw. This is not
native extraction recall or independent gold. Twelve graph source-recheck controls
match 12/12 after a harness route correction. The earlier generic pilot is 9/12
and did not exercise the graph guard. Express absence of an obligation differs
from record silence; delivery and nonservice do not establish or remove a duty.

Three fresh full-stack source jobs complete. The PR-681 acknowledgement now has
its own full action/date passage, recorded day and reported-occurrence assessment.
Initial strict action-string gates match 18/23 types and 6/7 calendars. A separate
unchanged-expectation original-span audit matches 19/23 and 7/7 because a French
deadline's shorter action is in the same unique source sentence. Do not conflate
that span-identity correction with semantic improvement. Four early English duty,
permission and negative statements genuinely have no candidates in this long
source draw, so an English permission-seeded component cannot be inspected.
The French component reaches all nine stored candidates; reachability alone does
not establish document coverage. Use section-level inventories with coverage
receipts next, preserving full-source cross-clause context.

Full Go/race, 1,353 backend and 914 frontend tests, production build, 12 clean
idempotent patch arms and 11 optional variant suites pass. Readable longer evidence
retains earlier model excerpts; main activation preserves all user graph tables,
24 journal snapshots, 32 artifacts, two Studio sessions and 14 user engine edits.
Temporary QA services/tab close only after terminal jobs and read-only graph checks.

For LayaGo datasets, separately label performance, own calendar association,
explicit plans, negation, obligation versus record silence, and document coverage.
Preserve source/occurrence IDs and exact context; include valid conditional dated
controls and wrong-action nearby-date negatives. A second pass from the same
provider is not independent semantic gold. No new Laya training, MLX, speed or
architecture-quality result is established. Party boundaries, unavailable and
ambiguous evidence, other predicates/applicability and browser cache quota remain
open; real/OCR and human-reviewed held-out qualification is still required.

[DelayLama independent context review](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-independent-context-review.md)
· [Shared receipt](validation/delaylama-independent-context-review-20261005.json).

## L127 — Inventory coverage must precede graph completeness (2026-10-05)

DelayLama's whole-original pass omitted four early duty, permission and negative
predicates while returning seven late statements. Graph connectivity cannot
recover statements that have no candidates. Add separate formatting-range
inventories with the full original available, preserving global line IDs and exact
UTF-8 intervals. Accept an own predicate only when its uniquely located start lies
inside the assigned interval; its original supporting context may extend outside.
Do not use extracted entities or existing actions as a completeness checklist.

Partition every original byte without semantic filtering or overlap. Group small
sections, split large ones at whole lines, preserve Unicode and expose input,
output and range budgets. DelayLama uses roughly 8,000-byte targets above 24,000
bytes, at most 64 ranges, one shared five-minute deadline and 120 rows per response.
Repeated full-original context adds inference cost. A failed or empty range does
not establish absence, and execution counts do not measure semantic completeness.
Duplicate kind disagreements stay unknown; a later pass cannot choose meaning
by response order. Keep new-row performance review separate from earlier existing
row update counters.

The exact retained English/French originals and expectations are frozen before
each new trial. Two fresh full-stack draws each accept three terminal source jobs.
Both match 23/23 original-span statement types and 7/7 original-span date-role
criteria, recovering all four early omissions. First-trial strict action-string
checks are 22/23 and 6/7 because a French deadline uses a shorter own-action span
inside the same unique source sentence. The final strict checks also match 23/23
and 7/7; retain both audits and unchanged expectations. Span identity does not
certify semantics. The historical L126 draw matched 19/23 original-span types.
This is not a paired provider seed or locked held-out accuracy measurement. The
first new prompt repeated outside-range statements 80 times because full-document
enumeration instructions competed with scope; deterministic ownership withheld
them. Explicit focus instructions and a final focus block reduce out-of-range
proposals to zero in the changed-prompt draw. Preserve both outcomes.

The recovered English permission reaches all 18 stored source statement candidates,
including distant consent and unsigned-amendment passages; French reaches all nine.
The browser locates Clause 88 without another model call. Reachability and exact
source spans still do not establish entailment, applicability or legal precedence.
The `Neither Northmere Employer Ltd` entity boundary and an associated-with model
suggestion from the permission remain, alongside withheld approver/entity roles
and unavailable/ambiguous evidence. Party identity and relation correctness are
the next graph-integrity targets.

Original-bound receipt guards check byte/line partitions, hashes, candidate counts
and per-range execution, stripping raw proposals. The interface groups empty
ranges and highlights ranges to inspect, with all details available on demand.
Full Go/race, 1,384 backend and 916 interface tests, build, 12 clean/idempotent patch
arms and 11 variant suites pass. Main activation preserves every user graph table,
24 journal snapshots, 32 artifacts, both Studio sessions and 14 engine-source edits.

For LayaGo training, distinguish source-window ownership from semantic relevance,
entity identity and document coverage. Include omitted early predicates, late
qualifications, passive/modality/negative statements, empty-range negatives and
outside-range but real source assertions. Never label a model-supported inventory
as independent gold. Evaluate human-reviewed held-out long/OCR cases and relation
precision separately. No new Laya training, MLX, speed or architecture-quality
claim is established here; General-profile defaults and browser cache quota remain
open.

[DelayLama source range inventory](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-source-range-inventory.md)
· [Shared receipt](validation/delaylama-source-range-inventory-20261005.json).

## L128 — Similar names are retrieval candidates, not party identities (2026-10-05)

A permission mentioning Northmere Employer Ltd pointed to Neither Northmere
Employer Ltd. The engine's deterministic supplement absorbed a coordinator, and
online reconciliation accepted containment/token overlap before checking all
exact candidates. A graph path with a verbatim quote still had the wrong actor ID.
Identity correctness is a separate gate from quotation presence and connectivity.

Scan every page in the same typed project/tenant scope before choosing a unique
exact display-normalized identity. Do not use shared content words as merge proof.
Ambiguous exact identities and legal-ending variants must remain unresolved.
DelayLama permits only a unique omitted legal ending on a multiword organization
base, preserving different explicit legal forms; this is still heuristic. Remove
coordinators only in paired unquoted syntax, retaining literal/quoted names and
original UTF-8 spans. Protect routing from malformed legacy nodes without silently
rewriting historical graphs.

New regressions fail on default 60 and pass on default 61. SQLite and file-storage
MemoryAgent controls retain all three statement party links despite a seeded bad
legacy actor. Exact priority, >1,000-row pagination, scope isolation, duplicate
identities, variant ambiguity, different legal forms and literal/quoted names have
separate controls. Three fresh full-stack jobs on unchanged frozen synthetic
English/French originals pass 23/23 expected types and 7/7 date roles. The permission
now connects both permitted party and conditional approver: 2/2 proposals pass,
0 withheld, and its association targets the correctly named actor. English context
reaches all 18 candidates and French all 11. The browser shows the original clause
highlight and preserves the bare-name evidence ambiguity.

Do not generalize this to perfect extraction: a separate global-role heuristic
labels Northmere contractor despite an employer declaration. Global role ownership,
actor/profile duplicates, repeated mention evidence, normalized stored statement
party names, applicability and precedence remain open. Full Go/race, 1,384 backend,
916 frontend, build, 12 clean/idempotent arms and 11 variants pass; every user graph
table, 24 journal rows, 32 artifacts, two Studio sessions and 14 engine source edits
survive activation.

For LayaGo, contrastive name similarity can retrieve candidate mentions, but entity
merge precision needs a different supervised target and abstention. Include close
subsidiary names, omitted versus different legal endings, grammatical coordinators,
literal brand names, role words attached to another nearby party and exact-name
collisions across projects/documents. Keep retrieval, identity, role ownership,
source matching and relationship direction evaluations separate. These synthetic
same-provider trials are not independently adjudicated held-out labels. No new Laya
training, MLX, acceleration or architecture-quality result is claimed.

[DelayLama party identity](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-party-identity.md)
· [Shared receipt](validation/delaylama-party-identity-20261005.json).

## L129 — Own-label ownership is separate from identity and contractual relationship (2026-10-05)

A correctly identified employer acquired another party's contractor label because
both declarations were on one line. Nearby role words and company-name terms are
retrieval cues, not ownership premises. A negated declaration could also become a
positive role. Default DelayLama patch 62 records only recognized own declarations,
direct party fields or scoped headings, retaining episode, original byte span,
quote and source hash. Negative, qualified and competing labels abstain; coordinated
labels do not select the first. Cross-source exact actor reuse keeps every role
premise instead of borrowing the actor's first episode. Damaged history is retained
and blocks primary role selection.

The API checks project/tenant, registered original, role episode, hash and exact
UTF-8 occurrence; the browser converts to UTF16 for highlight. Current text matching
is not role meaning, appointment validity, authority or applicability. Explicit
spans disambiguate repeated quotes; changed sources retain history without current
highlights. Human-readable role panels expose each own passage and qualifications;
duplicate evidence collapses only for the same checked occurrence. Global bare
role fields no longer enter neighborhood context. Role-derived connections remain
uncertain source-review candidates even when both labels match original text.

Full Go/race, 1,400 backend, 922 frontend/build, 12 clean/idempotent combinations
and 11 variant suites pass. SQLite/file controls retain competing cross-source
roles without a primary. Three fresh unchanged synthetic source jobs verify five
explicit English/French party declarations and retain the permission's permitted
party/conditional approver. English reaches 18/18 stored source candidates and
French 9/9. Original-overlap criteria pass 23 types and seven date roles; strict
prefix criteria remain 22/23 and 6/7 because a French deadline uses a shorter action
span inside the correct full original quote. Both results are retained.

Do not turn these controls into a semantic accuracy score. The French word
entreprise alone does not establish contractor, though an extracted Contractor
node type remains a model proposal. Model CONTRACTS_WITH links from unsigned
amendment non-signature evidence still need polarity/modality and applicability
qualification beyond the current causation/notice audit. Actor/profile duplicate
mentions, complex role wording, precedence and independent real/OCR contract tests
remain open. Every user graph table, 24 journal snapshots, 32 artifacts, two Studio
sessions and 14 engine source edits survive activation; no historical migration.

For LayaGo datasets, distinguish mention similarity, identity, label ownership,
polarity, modality and relationship entailment. Build hard negatives with adjacent
parties/roles, role-like company names, employer/payee objects, negated appointments,
conditional signatures, competing declarations across amendments, representatives
under party headings, multilingual legal forms and repeated original quotations.
Train candidate retrieval or ownership classification with an abstain/unknown
class; never promote a close embedding or a matched role phrase into legal meaning.
Keep evidence-version/coordinate validation outside learned confidence. These are
synthetic development learnings, not independently adjudicated training gold. No
new Laya training, MLX acceleration or architecture-quality result is claimed.

[DelayLama party roles](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-party-roles.md)
· [Shared receipt](validation/delaylama-party-roles-20261005.json).


## L130 — Own relationship occurrence and focused assertions (2026-10-05)

DelayLama exposed an unsigned-amendment CONTRACTS_WITH proposal that borrowed an
operative relationship from another part of the same original. Identity and exact
quote matching do not solve predicate ownership. An agreement between two parties
does not make every amendment, signature action or party label an operative link.

Default patch63 adds contractual_relationship to the independent blind inventory.
The model sees original text and endpoint identities; the proposed predicate,
quotation and previous verdict are withheld. Only afterward does validation bind
a unique exact focused assertion to the candidate’s own passage and its cited
context. Negative, proposed, conditional and unknown statements remain reviewable;
conflicting same-occurrence witnesses withhold support. Contract connections are
reciprocal, while causation and notice obligations remain directed.

Full Go/race, 1,419 backend, 923 frontend/build, twelve clean/idempotent combinations
and eleven variant suites pass. SQLite and file-store baseline controls fail on
default62. Four final frozen native jobs retain an operative SC315 link and reject
forbidden unsigned A123/A316 links; prior original-overlap criteria retain 23 types
and seven date roles. Strict prefix mismatches remain documented. A browser recheck
shows the focused assertion, exact highlight and contextual evidence without
changing the graph. All user records and fourteen engine source edits survive.

For LayaGo, separate pair similarity from relationship entailment and instrument
scope. Candidate retrieval can use embeddings, but eligibility should retain
original instrument/occurrence IDs, source version and exact assertion coordinates.
Build hard negatives with the same parties in an operative contract and unsigned
amendment, signature actions without operative-effect statements, proposed or
conditional appointments, denial/unknown wording, reversed directed predicates,
repeated quotations and broad context spanning multiple independent assertions.
Keep original source context and focused predicate spans as separate fields.
Train an abstain/unknown outcome and evaluate retained positives as well as false
links withheld; an empty graph is not an accurate graph.

These are synthetic development lessons, not adjudicated training gold. Quote
containment cannot certify semantics, precedence or legal effect. Broad candidate
quotes can still require instrument-scope review; omissions and model disagreement
remain possible. No new Laya training, MLX, OOM-supervision or speed claim follows.

[DelayLama contractual predicate report](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-contractual-predicates.md)
· [Shared receipt](validation/delaylama-contractual-predicates-20261005.json).


## L131 — Broad candidate scope and historical review qualification (2026-10-05)

A witness belonging to an operative contract can fit inside a broad candidate
quotation that also contains a separate unsigned amendment. Containment alone is
not occurrence identity. DelayLama default64 now withholds contractual quotes
spanning recognized source-unit boundaries, multiple focused relationship
assertions, or a separate signature assertion. A focused own assertion and its
wider context remain separate. Physical wrapped lines and an initial own heading
remain eligible. Formatting is not legal applicability or semantic scope.

A second lesson concerns history. Updating qualification in a graph overview does
not update a saved detail view that still renders the earlier verdict. Keep the
historical search intact and return a distinct current assessment against current
originals/finding. Use it in detail, copied feedback and prompt imports; retain
both assessments and complete available history for learning. Source/finding
changes invalidate current qualification rather than silently repairing history.
Distinguish a failed meaning/scope check from a failed quote or endpoint match.

Go/clean/race, 1,430 backend, 925 frontend/build, twelve fresh/idempotent patch
combinations and eleven variants pass. Frozen native originals retain the operative
SC315 candidate, exclude forbidden unsigned links, and retain all 23 expected
classifications/seven date roles in this draw. A separate controlled browser
recheck finds the operative passage but keeps a manually supplied broad A316
proposal unresolved. Status and sanitized prompt context agree; histories and
stored source graphs remain unchanged. Every user graph table, saved app artifact
and fourteen engine source edits survive activation.

For LayaGo datasets, annotate focused predicate spans, wider context, instrument
identity, assertion-unit boundaries, polarity/modality and qualification status
separately. Add hard negatives that combine an operative agreement and unsigned
amendment in one quote, including no-heading/same-paragraph cases, line-wrapped
positive controls and relevant dispersed qualifications. Learn candidate scope
with an abstain class; verify source versions/coordinates outside learned scores.
Keep historical failures and current policy assessments separately labelled.
These development cases are not independent legal training gold. No Laya training,
MLX speed, OOM supervision or architecture-performance result follows from them.

[DelayLama source-unit report](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-contractual-source-units.md)
· [Shared receipt](validation/delaylama-contractual-source-units-20261005.json).

## L132 — Keep graph generation, downstream experiments and historical support distinct (2026-10-05)

A feedback destination is part of evidence provenance. DelayLama was routing every
source-graph relationship to its execution-event prompt, while a generic field
extractor was labelled “Entities & relationships.” Neither application prompt
owns Chronograph’s native graph-generation stage. The handoff now suggests relevant
downstream comparisons only for assigned tasks and requires an explicit choice
otherwise. It names the selected experiment and explains what editing it changes.
A source-graph error alone is not an evaluated failure of that extraction prompt.

Known critical links must require their predicate check even when old finding
metadata is missing. An assessment of a different relationship cannot establish
the selected one. Current display and copied feedback remain unresolved while the
original historical verdict is retained. Do not carry an old qualification into a
new scope merely because both use similar names or matching passages.

937 frontend tests and the production build pass. An isolated read-only replay
preserves three native synthetic graphs and a separately supplied controlled edge.
Its one terminal review is not replayed: status and prompt context agree on the
unresolved broad A316 proposal and retain the focused operative SC315 witness.
A browser import into an explicitly chosen Amendments experiment preserves the
active prompt intent, exact original, historical review and current assessment.
No model run occurs. Project backup restoration retains these even when browser
local storage is unavailable/full. User graph tables, fourteen engine edits and
application histories remain unchanged.

For LayaGo experiments, record the producing stage/model, task/schema, source and
instrument IDs, predicate identity, historical assessment and current qualification
policy separately. Route feedback to a measured task; do not mislabel retrieval or
field extraction as relationship entailment. Keep failures as development evidence,
not automatically adjudicated gold. These checks establish workflow behavior, not
new extraction accuracy, complete instrument applicability or Laya/MLX performance.

[DelayLama graph-prompt scope report](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-graph-prompt-scope.md)
· [Shared receipt](validation/delaylama-graph-prompt-scope-20261005.json).

## L133 — Do not drop qualified relationship witnesses between graph and reasoning (2026-10-05)

A source-first inventory on a native graph edge is useful only if the context builder
preserves its meaning checks and witness provenance. DelayLama retained assertions
and quotes but dropped that inventory before handing neighborhoods to the LLM.
The application now rechecks its actual predicate/pair, project/tenant, current
original/version, unique witnesses and exact native UTF-8 coordinates; then carries
polarity, modality, focused assertion and current qualification into graph previews,
neighborhoods and model/topic context. Historical status stays distinct and unmodified.
Raw vectors and unverified model objects do not become visible facts or prompt context.

A topic can occur only on a relationship while endpoint names are generic parties.
Matching scoped relationships now nominate their actual endpoints. Foreign/dangling
links do not nominate anchors. This is lexical retrieval, not alias resolution or
entailment, and coverage/budget limits remain explicit.

1,461 backend and 959 frontend tests plus the production build pass. An isolated
read-only engine/API/browser replay preserves three synthetic native graphs. Its
native SC315 relationship has the same currently qualified original witnesses in
main graph, neighborhood, model and topic context. A manually broadened in-memory
quote mixing SC315 with unsigned A316 is unresolved while old support remains
historical. Browser inspectors show focused highlights and expandable context without
model noise. No new model jobs or user graph/history changes occurred.

LayaGo datasets should retain producer/task/schema, source/version and instrument
identities, directed endpoints, byte/character coordinate conventions, polarity,
modality and historical/current qualification policy separately. Evaluate retrieval
and predicate classification independently. Do not turn these development controls
into adjudicated legal gold or label every matching quote positive. This is evidence
transport and policy behavior, not new LLM/Laya accuracy, legal applicability,
complete recall, MLX performance or supervisor/training evidence.

[DelayLama native-predicate context report](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-native-predicate-context.md)
· [Shared receipt](validation/delaylama-native-predicate-context-20261005.json).

## L134 — Separate amendment wording, effect status and unknown flags (2026-10-05)

A waiver flag is not an applicability premise. DelayLama generated an entitlement
effect from that flag when a summary was missing, and its desk merger converted
unknown waiver values to false. It now requests scope/applicability review and
preserves unknown flags as null. Linked event impact chains retain each amendment's
own status record. Extraction requests status, affected scope, conditions, effective
date, rationale and own citations; it cannot borrow another amendment's approval,
signatures or date. UI review labels identify the owning amendment and surface
unmatched status quotes beside the assessment.

Two bilingual documents and eight synthetic expected statuses were frozen before
four production consolidation invocations (two before, two after). The baseline
already described the conditions correctly in prose. The new explicit statuses
match 8/8 in one draw; this is a structured-interface result, not a demonstrated
semantic accuracy gain. Only 7/8 status quotes match: one English unknown-status
quote joins nonadjacent text with an invented ellipsis. The original response is
retained, not repaired into gold or rerun for a preferred answer.

That error exposed a verifier skipping citation objects with other nonempty lists
such as conditions. Explicit own citations are now checked regardless of list
count. The same responses were requalified offline; the invented quote remains
unverified. A no-LLM owned-project replay verifies API/UI status delivery and the
exact nested correction note reaching the Amendments prompt experiment with the
full registered original and history. Final null-flag behavior is checked via the
current production API helper after listeners stopped. 1,468 backend and 964 frontend
tests plus build pass. Main API is refreshed; user graph/history/source edits remain
unchanged. Date-role and missing-date-premise concerns remain visible.

For LayaGo datasets, separate waiver wording from operative status, amendment ID,
affected term/incident/claim dimension, fulfilled/unfulfilled/unknown conditions,
effective date and retained rights. Treat missing records as unknown, not a negative
label. A valid classification with an invalid quote is not a grounded positive.
Keep exact witness spans and failed raw responses, and evaluate classification and
evidence selection separately. These are development controls, not independently
adjudicated legal gold, full precedence/coverage or new Laya/MLX performance.

[DelayLama amendment-applicability report](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-05-amendment-applicability.md)
· [Shared receipt](validation/delaylama-amendment-applicability-20261005.json).


## L135 — Keep event-reference provenance and each finding’s own context (2026-10-05)

DelayLama's event links now carry generated source paths and distinguish supplied
ID fields from narrative mentions. A sentence explicitly excluding E2 must remain
available as context, without being displayed as an applicable amendment. Duplicate
amendment IDs cannot select a peer's status, scope or citations; validated own paths
are locators and never semantic proof. Legacy unresolved links remain readable.
Risk rewrites are no longer copied into absent claim-impact fields. The event view
exposes own conditions, status and quotations; analysis-map mention edges and the
review queue preserve the uncertainty through prompt feedback.

A hand-authored synthetic extraction replay in an isolated full stack checked two
repeated-A1 records, express exclusions, same-event risk/claim separation and exact
parent/nested-status review routing. The own note, whole registered original and
relationship warning reached a saved two-entry Amendments session without a model
revision or prompt activation. 1,470 backend tests, 970 frontend tests and the build
passed. Three frozen native graphs remained unchanged; no new native extraction,
model call, training or Laya/MLX performance measurement occurred. An initial QA
harness argument-order error produced invalid QA-only output history, which was
retained while saving corrected latest fixtures in the same owned project.

For classifier/retriever evaluation, retain endpoint provenance and exclusion
candidates separately from positive applicability labels. Exact quotation, entity
identity, relationship assertion, effective status and affected claim dimension are
different gates. A model that improves retrieval or classification still needs an
independent semantic/source benchmark; these integration checks do not establish
held-out legal accuracy or native graph construction completeness.

Shared hashed receipt: [DelayLama event-link context qualification](validation/delaylama-event-link-context-20261005.json).


## L136 — Quote retrieval is separate from outcome-object and rule support (2026-10-05)

DelayLama claim input now retains all selected prior analyses and a distinct graph
bundle, including long-tail analysis. Unavailable graph refreshes become visible
saved gaps rather than silently replacing supplied context. Tests cover both
reasoning modes, empty/readiness/error/sentinel paths and exact context preservation.

A production time/cost consolidation on synthetic C124 received its whole original,
hand-authored event/prior controls and current context from its existing native graph.
All eight quotes matched, but the model called a suspension request a payment request,
and its refusal an absent extension/payment award. Missing client approval records
also became categorical entitlement conclusions. One documented two-request prompt
follow-up still failed outcome kinds and entitlement; both failed raw responses are
retained. Four total real requests completed without fallback. This is prompt
development, not independent legal gold or a successful accuracy benchmark.

Own-path backend/frontend review hints now expose unrecognized stage premises,
documentation gaps used as absence, and definite time/cost conclusions without own
dimension wording. They preserve answers and never certify a matching vocabulary.
An owned browser check carried the exact correction, warning, whole registered
original and two-entry history into a saved Claim viability prompt session. Its
source heading changes the hash versus the bare model original; the provenance
warning and current-detector gate remain visible. No revised model/detector run or
prompt activation occurred in the browser. 1,495 backend and 974 frontend tests plus
build pass. Three native graph snapshots and protected user state remain unchanged;
owned listeners stopped. No new native graph build or Laya/MLX experiment occurred.

For future LayaGo classifiers/retrievers, label the object of each request and
refusal, occurrence, decision stage, document-status uncertainty, and applicable
rule as separate dimensions. An exact witness span with the wrong object is a hard
negative, not a grounded positive. Missing records must not become absence labels.
Train/evaluate association, polarity and rule applicability independently of quote
location; use unseen independently reviewed cases before claiming quality gains.
These controls could guide a later dataset, but no training improvement is measured.

Shared hashed receipt: [DelayLama claim context and premise qualification](validation/delaylama-claim-context-handoff-20261005.json).

## L137 — Claim rechecks need separate object, state, rule and applicability premises (2026-10-05)

An exact witness span about D-174 did not establish a payment request: its original
antecedent described a suspension request. DelayLama's source recheck now asks for
separate outcome-object and recorded-state evidence, and for separate dimension,
rule type and event-applicability evidence for definite time/cost conclusions.
Coverage, source matching, enum binding and conservative wording checks gate review
candidates. A permission procedure or missing approval record cannot substitute for
an entitlement/exclusion rule or independently established condition noncompliance.
These gates do not certify the meaning of validly formatted model declarations.

Four frozen synthetic development controls were run once, then once more after
making the array schema and required metadata explicit. The initial valid payment
control was an **insufficient false negative** from missing metadata; it is retained.
The follow-up kept both C124 negatives unresolved while the explicit payment and
weather-cost-exclusion controls received candidate support. The raw time response
still misinterpreted a permission rule: qualification blocked promotion, not the
underlying semantic error. Five initial calls, four follow-up calls and one actual
browser recheck completed. All returned quotations matched; no fallback or semantic
persuasion retries occurred. This is format development, not independent legal gold
or held-out extraction accuracy.

The actual button kept the erroneous payment fact unchanged, with four gaps and no
qualified contradiction. Compact field disclosures retained independent passages.
Own field-prefixed premise explanations and exact quotes, the correction note, full
registered original and two history entries saved and restored through project
backups. The source-heading hash difference and detector gate stayed visible.
1,528 backend and 984 frontend tests plus build pass. Three existing native graph
snapshots and protected user state stayed unchanged; owned services were stopped.
No native graph construction, Laya training or MLX performance test occurred.

For future reusable LayaGo datasets, distinguish request object, decision stage,
recorded absence versus undocumented state, dimension, rule type and applicability.
Use genuine matching spans for the wrong object/rule as hard negatives. Measure
abstention/false negatives separately from false support, and retain initial failures
alongside development follow-ups. Train association and condition handling against
independent labels; graph neighbors and successful quote location are not legal
gold. A useful next evaluation would blind the object/rule inventory to the proposed
claim conclusion, following the existing graph predicate-audit separation, then test
unseen independently reviewed cases. No quality gain from training is measured here.

Shared hashed receipt: [DelayLama claim premise review](validation/delaylama-claim-premise-review-20261005.json).


## L138 — Fresh graphs and blind premise inventories still need semantic evaluation (2026-10-05)

**Evidence:** [DelayLama receipt](validation/delaylama-fresh-claim-premises-20261005.json)
and its linked report, frozen author-written controls, native reports, retained raw
model outputs and browser/source-map proofs. This is synthetic development evidence,
not independent legal gold. No LayaGo training or MLX/GLiNER run was performed.

A completed source graph was unavailable to saved-project analysis because explicit
native jobs did not mirror extraction's readiness record. Shared current-source
readiness and actual event-language/reference queries fixed the wiring. Generic
English queries had no French topic matches. Native witness metadata then overflowed
the context window even on short originals; keep whole original passages and graph
items, disclose omissions, and preserve unresolved missing context. Node reachability
and located quotations do not establish predicate meaning or applicability.

A source-first claim inventory saw only reported event identity/title/date candidates
and full originals, without prior verdicts, graph assertions or draft claims. One
shared inventory preceded independent time/cost passes. Source presence, own quotes,
metadata and event joins were checked; raw failures remained visible. The source map
made occurrence, permission request, refusal, conditions and documentation gaps easy
to compare. Three inventory calls were recorded. An inventory can still omit rules
or misinterpret a passage, even when every returned quote matches.

Six native jobs and five application jobs finished across three fresh projects with
six originals, twenty saved outputs and 48 recorded application model calls. Native
model calls are additional and were not aggregated. The combined C127/C128 follow-up
improved undocumented payment/extension statuses but still made categorical conclusions
from clauses that supplied no rule. A new C129 run preserved a signed two-day grant
and express weather-cost exclusion, but still classified undocumented payment stages
as absent and non-waiver text as a recorded waiver. Source-review flags expose these
inconsistencies while retaining the actual answers. No repeated preferred-answer
retry or manual correction was used. The shared pass adds one serial call; speed and
causal quality benefits were not independently established.

Role-inferred engine connections can assert a specific engineering mission merely
from a party label. Their structural-role status, qualification and explicit unresolved
assessment must survive every reasoning and inspection projection. These are review
suggestions, not semantic graph facts. A usable neighborhood is neither an exhaustive
contract reading nor an automatically correct legal model.

**Training/evaluation implications:** build distinct labels for occurrence versus
permission/obligation, request object versus decision object, undocumented versus
explicitly absent, non-waiver versus waiver, and rules that determine entitlement
versus procedures or non-determination. Test multilingual negation, modality, scope,
notice fulfillment and dispersed amendments using separate contracts/templates and
independently labelled expected states. Measure source location, retrieval recall,
object/state classification, entailment/applicability and resulting claims separately.
A LayaGo candidate router/classifier can nominate passages and uncertain states, but
must not turn lexical confidence or graph reachability into certified rights. Preserve
hard false positives, abstention failures and positive controls; report latency/memory
and end-to-end cost before claiming a faster or better extraction system.

**Next:** evaluate per-event multilingual retrieval on long unseen dossiers, reduce
irrelevant repeated graph metadata without dropping qualifiers, assess complete rule
and condition inventories independently, and prevent role co-occurrence from becoming
an asserted mission at the engine generation stage. Keep all incorrect model outputs
and source-review histories as failure examples separate from ground-truth labels.


## L139 — Source layout, semantic navigation and rule coverage need separate evaluation (2026-10-06)

**Evidence:** [DelayLama receipt](validation/delaylama-focused-claim-context-20261006.json)
and linked source report, retained context replay, actual model outputs and restored
feedback proofs. No LayaGo/GLiNER training or MLX measurement was performed.

Document containers dominated lexical anchor ranking and containment expanded whole
originals into graph reasoning. Focused semantic navigation now expands two steps;
all matching facts retain own evidence even outside four expansion anchors. Explicit
numbered clause references enter the model bundle without inventing an applicability
edge. Source outline/full-component exploration stays separate. Original passages and
source hashes remained equal across three retained native graph replays; serialized
graph context fell from 119,150/100,729/119,108 to 49,642/53,237/96,775 characters.
This does not establish lower total prompt tokens, latency, recall or better analysis.

A per-event inventory checklist distinguishes time/cost rule candidates, conditions
and documentation gaps. Procedure-only candidates cannot substitute for an entitlement
rule. Missing independent rules flag definite recognized conclusions while preserving
raw fields. Current-source quote matching qualifies displayed counts, not interpretation
or condition fulfillment. Exact event/target review links carried these concerns into
prompt feedback with the original, actual output and two history entries; restoration
worked across reload without accepting a finding or running an assistant revision.

Two development model workflows completed with ten application calls. The first was
incorrectly routed by the test launcher to an API with different graph scope; retain it
as failed integration evidence. The corrected explicit API origin received focused
context but still made four incorrect categorical time/cost conclusions from a procedure
that determined no entitlement. Those answers remain retained and flagged; **no accuracy
improvement was demonstrated**. Inspect actual input/runtime records before attributing
an extraction outcome to a graph or trained classifier.

Both requests asked for single-shot, but the active stack forced multi-step. Runtime
records take precedence over request labels. A copied engine restart retained graph data
but not the native server's memory-held completed-job registry; no job was resubmitted.
Durable recovery of interrupted native jobs needs further engineering and failure tests.

**Implications:** evaluate retrieval independently from source-state/object classification
and final entitlement decisions. Preserve disconnected rules, scope, exceptions, notice
fulfillment and source versions; shared party/containment proximity is not a legal link.
Use independent source-first decisions to study bias from earlier incorrect analyses.
Any LayaGo routing or classifier experiment needs actual invocation-scope/mode verification,
held-out human labels and retained abstention/negation failures before a quality or speed
claim. Do not use review flags or exact-quote ratios as semantic ground truth.


## L140 — Independent source decisions and actual runtime controls (2026-10-06)

**Evidence:** [DelayLama receipt](validation/delaylama-source-first-claim-decisions-20261006.json)
links original synthetic controls, actual outputs, failures, model request checks and
browser proofs. No LayaGo/GLiNER/MLX training or performance measurement occurred.

First claim decisions now use originals, event identity candidates and uniquely
source-located graph passages; prior conclusions, graph explanations and draft
verdicts are withheld. Graph paths nominate original context rather than certify
meaning. Prior records remain inspectable. This separation improved some development
answers but did not establish a complete or accurate contract model.

A dedicated reasoning profile initially looked correct in YAML, but the installed
Agentomatic driver silently dropped flat reasoning_effort. Actual request construction
and runtime inspection exposed it; extra_body forwarding fixed it. An initial
budgeted probe therefore used provider-default thinking, not the requested low
setting. Evaluation must record actual controls, input scope and response identity,
including silently ignored settings. JSON recovery must preserve original source/
schema constraints; known truncation must not become a completed analysis.

Seventy-one recorded application calls and two native source jobs were retained.
After forwarding was fixed, the new C130 control preserved four unresolved rights
and unknown undocumented outcomes. Harder C128 still rejected one cost target
without an applicable rule; C129 preserved a grant/exclusion but omitted event
targets. Earlier non-waiver polarity mistakes remain saved. Missing-target gaps,
exact-quote ratios and model uncertainty are observations, not ground-truth labels.

**Training implications:** separate retrieval, occurrence identity, request/decision
object, source polarity, rule applicability, condition fulfillment and recorded
outcomes. Include procedure-only clauses, permission refusals, explicit exclusions,
grants, non-waivers and undocumented states as distinct multilingual controls.
Preserve independent positive controls and abstention errors; do not train on a
model rejection as gold simply because its quote matches. Split unseen contracts/
templates and require human labels before semantic quality claims. A LayaGo router
can nominate passages, while an independently evaluated classifier can propose
states; neither should turn role proximity or contrastive similarity into entitlement.

**Open:** graph mission edges still inferred from role co-occurrence need generation-
time source grounding, long-dossier recall needs independent evaluation, and native
job recovery across restart remains unfinished. This work added no LayaGo architecture,
MLX acceleration or training supervisor changes.


## L141 — Party roles are not relationship witnesses (2026-10-06)

**Evidence:** [DelayLama receipt](validation/delaylama-role-relation-generation-20261006.json)
links before-fix regression failures, storage controls, fresh native outputs, actual
browser witnesses and preserved-state activation. No LayaGo training or speed claim.

A deterministic role pass cross-multiplied owners and contractors, assigned specific
missions from general project-manager roles, defaulted insurance/guarantee beneficiaries
and created utility nodes from stakeholder labels. A role declaration quotation
supported the label, not those endpoint relations or scopes. Exact text and structural
provenance did not repair that semantic gap.

Default compatibility patch65 separates role-only nominations from stored edges,
synthetic entities and known-relationship prompts. Source-backed party labels remain
candidates. Positive SQLite/file-store controls retain an explicit operative contract
with its own predicate witness while withholding an unsigned amendment. Two fresh
saved projects produced three native source jobs and twelve application calls: no
role-inferred edges, and the explicit SC315 contract/invoice certification persisted.
These development controls do not establish full semantic accuracy or recall.

A source-hashed receipt distinguishes withheld generation proposals from graph facts.
Wrong-source or malformed receipts cannot become a verified policy display. The
browser keeps that policy separate from source-extracted model relationships and
highlights the retained witness. No historical graph edge was deleted.

**Training implications:** use minimally differing examples with the same parties/roles:
a roster alone, an explicit bilateral agreement, a proposed unsigned amendment, a
narrow assignment, a negative assignment and an insurance/guarantee scope for a named
beneficiary. Include multiple possible counterparties so a classifier cannot choose
the first label, plus identical role words attached to different originals. Evaluate
endpoint/direction, predicate, modality, instrument and conditions separately from
role classification. Contrastive similarity can nominate contexts, but a party role
embedding must not certify a contract or legal relationship.

**Open:** duplicate native and extraction-seeded role declarations need source identity
reconciliation. Other heuristic/model predicates, long-dossier recall, precedence,
causality/fulfillment and durable native job recovery still need evaluation. Previous
claim abstention/omission/waiver failures are not resolved by this graph generation
change. No LayaGo architecture, GLiNER integration, MLX or supervisor change occurred.


## L142 — Occurrence copies and acceptance origins are separate (2026-10-06)

**Evidence:** [DelayLama receipt](validation/delaylama-party-role-provenance-20261006.json)
links source-offset tests, read-only retained graph replay, browser highlights,
failed and corrected live storage probes and preserved-state activation.

A full original and an extraction-indexed contiguous excerpt could contain the
same role declaration. The verifier recognized the former but left the latter
unchecked, falsely describing identical labels as competing. Exact excerpt lineage
(document identity/hash/range, scoped episode, local UTF-8 quote coordinates)
now projects to the same Unicode original occurrence. Only equal checked occurrence,
role, polarity/qualification and basis are grouped. Historical records remain;
missing verification is separate from a real recorded-label conflict.

A live nomination probe also exposed implicit human acceptance: the manual-node API
overwrote candidate properties with validated/human stamps. A dedicated unreviewed
creation route now stores pending records, strips supplied review stamps and rejects
scope overrides. The client refuses unsupported engines and never falls back to
manual acceptance or replays an uncertain write. Two bad initial synthetic records
remain as failure evidence; the corrected live probe has two pending nodes and zero
edges. No model/native source analysis or training was run.

**Training implications:** distinguish duplicate indexing from distinct source
occurrences, and source verification from semantic qualification. Construct controls
with identical role wording in separate instruments/versions, repeated Unicode
passages, shifted/corrupt excerpt ranges and explicit contradictions. Do not count
duplicate citations as independent support or an evidence-strength boost. A model
or LayaGo confidence score cannot become a human-review status; keep nomination,
source-location, interpretation and acceptance origins separate in datasets/UI.

**Open:** qualified legal roles and unrecognized multilingual labels still need
independent source review. Legacy implicit-human stamps require historical auditing;
source and graph histories were not migrated. Earlier claim, recall, causality and
native recovery errors remain open. This work introduced no LayaGo/GLiNER/MLX or
training-supervisor quality/performance improvement.


## L143 — Unit coverage, rule meaning and replacement polarity differ (2026-10-06)

**Evidence:** [DelayLama receipt](validation/delaylama-claim-rule-basis-20261006.json)
links frozen development controls, actual model outputs (including errors), source
traces, browser proof and a failed/corrected native replacement-generation control.

Batch target calls omitted notices and decisions. One event/target per unit returned
all 30 requested pairs across 15 reported events in five production workflows.
Source-first shared premises and structured role/subject/rule/condition traces made
an unsupported cost rejection unresolved in the permission control. This is measured
unit coverage, not exhaustive source-event recall or a legal accuracy benchmark.
The later event detector still omitted an unsigned draft found in the source graph.
Notices remained incorrectly labelled claimable, and a weather cost row misread
non-waiver language as a waiver and unsettled compensation as final settlement.
Those raw failures stay retained; coherent schema and located quotations do not
certify interpretation, polarity, related subject or condition fulfillment.

The first fresh native graph stored a positive SUPERSEDES edge from an ineffective
unsigned draft, even though its own exact quotation and fact denied replacement.
Default compatibility patch67 requires a blind own-source instrument-replacement
witness with matching direction, focused occurrence, positive polarity and actual
modality before storing this positive predicate. Proposed/denied/reverse/borrowed
witnesses are withheld with receipts. The live recheck had no draft replacement
edge and retained an operative replacement solely for completion date, with its
own model witness. Historical unqualified edges remain unresolved for inspection;
their records were not silently deleted or recertified.

**Training implications:** construct separate labels for reported event, supporting
notice, request and recorded decision; annotate the related claim subject and the
own association passage. Include exact rule/fulfillment evidence and unknown
conditions. Sample non-waiver versus waiver, undocumented versus affirmatively
absent, undated accounting facts, qualified replacement versus total supersession,
unsigned drafts and operative replacements. Include another positive instrument
near a negative candidate to test borrowed evidence, with direction-specific
contrasts. Evaluate source-event recall separately from requested-row coverage.
LayaGo can nominate passages or classes; its output must not become operative
legal relationships or human acceptance by confidence alone.

**Limits:** 66 production plus 17 prototype application calls and six native
analysis jobs; native internal call count is separate. No held-out human gold,
repeated-seed or causal model/prompt estimate. Relevant automated suites and
storage/API/UI checks passed; this is no LayaGo/GLiNER/MLX training or speed result.
Repeated registration refreshed merge candidate IDs and added indexing episodes;
merge lineage, broader predicate correctness, long-dossier recall and durable
native-job recovery remain open.


## L144 — Outcome object, polarity and target scope need independent extraction (2026-10-06)

**Evidence:** [DelayLama receipt](validation/delaylama-original-outcome-statements-20261006.json)
retains two source-only probes, two production revisions, a fresh recorded-payment
control, source-guard failures, full finalization tests and browser evidence.

Combining entitlement judgments and recorded facts allowed a cost assessment to
turn non-waiver into waiver and unsettled compensation into final settlement.
A source-only statement ledger now extracts the record/subject, object, instrument,
polarity, modality, dimension, scope and separate date evidence before entitlement.
Decision units cannot supply the primary fact rows. Missing kinds stay coverage
gaps; failed statement extraction does not fall back to a decision's invented facts.
Scoped negative statements remain scoped denials, and proposed/conditional amounts
stay distinct from operative outcomes. Literal presence and coherent schema remain
candidate checks, not semantic or legal certification.

The first fresh ledger correctly extracted a cost settlement and limited release,
but the merger copied them into time outcomes because both lived in one instrument.
Explicit time/cost dimension is now required for primary projection. Other or
undetermined dimensions remain related context. The recheck kept the cost-only
settlement/release separate from the completion grant; a draft zero stayed proposed.
The source-only root remains inspectable when all entitlement calls fail.

A display boundary also mattered: citation verification added match/source metadata
and broke naive full-object equality. Binding now ignores additive verifier fields
while retaining quotation/ref/coordinates identity. Actual quote/coordinate changes
still fail. Source-view hashes gate exact trace highlights; repeated cards share a
bounded four-view digest cache. These are behavior checks, not performance results.

**Training implications:** separate literal outcome statement extraction from claim
viability. Label object, record, subject, polarity, modality, dimension and scope;
include own focused assertions, explicit associations and independent dates. Add
cost-only releases in instruments that also grant time, qualified non-waivers,
undocumented stages, undated expenses, requested/awarded/settled amount differences,
unsigned zero proposals, and actual zero costs. Assess event recall separately from
requested-row coverage: the event detector omitted a payment request which the
ledger nevertheless extracted. Do not train confidence/source match as acceptance.

**Evidence limits:** 14 probe and 106 production application calls, two native source
analysis jobs, final 24 requested pairs across 12 reported events. Same synthetic
development controls, not independent legal gold or causal/held-out/repeated-seed
accuracy. Early date/object mistakes and rejected source rows remain retained.
Generic lexical/date review hints still create false warnings; full date-role,
subject/instrument applicability, long-dossier recall and native recovery remain
open. No LayaGo/GLiNER/MLX training or speed gain was measured this turn.


## L145 — Action recall, financial stages and tentative rights require separate checks (2026-10-06)

**Evidence:** [DelayLama validation report](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-06-review-wording-event-stages.md)
and [hashed receipt](validation/delaylama-review-wording-event-stages-20261006.json).

An initial event-extraction probe recovered an omitted payment request but copied
requested/awarded/settled figures into actual incurred expense. Explicit financial
stage fields and reported-action/context guidance separated those quantities in
subsequent development probes and saved workflows. A delay mentioned as the subject
of a request remained undated context in the isolated probe; the saved draw returned
only the three expressly recorded actions. Circulation and proposed effect must be
separate: their classification still varied across draws.

Review precision needs own focused predicate, independently captured date and
current field/record/target binding. Broad context can contain a non-waiver of other
claims alongside an operative limited release. Apply wording hints to the own
focused assertion, while preserving original context for human inspection. Location,
metadata consistency and absence of a warning never certify semantic support.

A fresh workflow still promoted a standing request procedure into a tentative cost
entitlement. “Potentially claimable” must undergo substantive rule coverage and
applicability checks too. The new review flagged the missing independent cost rule;
it did not repair the verdict or prove exclusion. The native graph separately
retained a mentioned delay with an anchor-inferred request date. Application context
handling does not repair native graph metadata.

**Training implications:** annotate action, normative duty, mention, negative fact
and proposal separately. Label requested/awarded/settled/proposed amounts apart from
incurred expense; distinguish action date from subject occurrence date and effect
date. Train rule/procedure/outcome separation before viability, including tentative
recovery labels. Evaluate event recall independently from requested-row coverage,
and native graph validity independently from application display. Do not train
inferred anchors, model confidence, source matches or review hints as acceptance.

**Evidence limits:** six isolated and 46 saved application calls; one native source
analysis job; 18 event-target rows across nine reported events in final saved runs.
1,662 backend and 1,029 frontend tests plus build passed. Retained failed probes,
semantic proposals, inferred native date and source-review false warnings remain.
These are synthetic development controls, not independent legal gold, causal,
held-out or repeated-seed accuracy. No LayaGo/GLiNER/MLX training or speed gain was
measured. The final notice-status guidance had deterministic checks only, without
a new model draw.


## L146 — Document calendar and concept occurrence must remain separate (2026-10-06)

**Evidence:** [DelayLama native-calendar validation](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-06-concept-calendar-context.md)
and [hashed receipt](validation/delaylama-concept-calendar-context-20261006.json).

A mentioned-delay concept inherited a payment-request date through document-anchor
fallback. A second deterministic regression showed that reparsing a normalized
inferred date could promote it to an exact occurrence timestamp. This happened in
native concept storage, independently of the application's context-only display.
Date provenance must survive every normalization, storage and retrieval boundary.

Default engine compatibility patch 68 retains first-document-date and supplied
inferred calendars as context metadata. They do not populate concept occurrence
or validity. Repeated normalization and timeline reconstruction cannot recover an
occurrence from a dated summary. Explicit own date text keeps its source precision;
a parser or label does not certify its semantic role. Historical rows remain intact.
The inspector distinguishes unestablished occurrence from document calendar context.

Actual source-first runs retained an undated English mentioned delay and a separately
reported undated French delay across two sources, while dated payment requests,
awards and draft circulations kept their own dates. Actual expense remained distinct
from requested/awarded amounts. Tentative entitlement from a standing request duty
still appeared in the English control; date correction did not repair that verdict.
A rejected source statement and normative deadline-role Event nodes remain explicit
qualification problems.

**Training implications:** label concept mention, actual action, statement modality,
date role, precision, document context and source association separately. Include
an undated subject discussed by several dated records; ensure transformations of
cached date strings cannot turn an inferred anchor into observation. Evaluate the
native store and timeline projection separately from application displays. Preserve
source text, original unknowns and failed proposals. Do not turn a context date,
connected path or exact text match into a domain-training acceptance label.

**Evidence limits:** 51 application calls, three native source jobs, 16 requested
pairs across eight reported events. Full Go and agents/intelligence/API race suites,
12 patch arms, 1,662 backend and 1,031 frontend tests plus build passed. English
control reused; French control frozen before submission. No independent legal gold,
held-out, repeated-seed or causal accuracy result; explicit model date associations,
semantic typing, source recall, applicability and native recovery remain open.
No LayaGo/GLiNER/MLX training or speed gain was measured.


## L147 — Procedures, substantive rights and recorded decisions need distinct output channels (2026-10-06)

**Evidence:** [DelayLama substantive-rule validation](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-06-substantive-rule-separation.md)
and [hashed receipt](validation/delaylama-substantive-rule-separation-20261006.json).

An original-first claim baseline still promoted a duty to request payment into
tentative cost entitlement. Its inventory had no procedure category, while the
numeric request/award stages supplied a strong competing signal. A dedicated
procedure kind and explicit provision-role classification removed those proposals
on the fixed negative control, while a positive substantive reimbursement clause
with fulfilled notice remained recognized. Substantive clauses with unknown
conditions were retained even when applicability remained unresolved.

Checking the whole application exposed the same error in another agent: Risk
allocation inferred liability from an award despite the corrected claim result.
Original-first risk inputs and basis classification removed that allocation. The
first recheck still put granted days into prospective claimable_time. Separate
recorded_decisions fields gave the true fact an appropriate channel and removed
the overload in subsequent draws. Reported, denied, proposed, conditional and
unknown states prevent treating a non-waiver or unsigned proposal as operative.

**Training implications:** classify statement function before predicting legal
consequence. Include filing deadlines with payment vocabulary, recorded awards
without allocation clauses, genuine reimbursement rules, fulfilled/unknown notice
conditions, and recorded grants that do not create general prospective rights.
Provide dedicated factual channels rather than forcing quantities into entitlement
fields. Evaluate every consuming agent; one correct decision does not make a
contradictory risk summary correct. Preserve raw failures, own quotation coordinates,
source scope and unknowns. Exact text and a coherent role are not acceptance labels.

**Evidence limits:** 75 baseline/revised source-only calls, 49 initial saved-workflow
calls and six risk rechecks after changes: 130 application calls, three native
source jobs, final 14 pairs across seven reported events. Claims kept Gemini 3.5
Flash / low reasoning; risk kept its fast profile. 1,668 backend and 1,037 frontend
tests plus build passed. These are synthetic development draws, not independent
legal gold, held-out/repeated-seed or isolated causal accuracy. Focused-statement
and subject-association rejections, literal heading-reference warnings, cross-agent
summary scope, comprehensive recall and applicability remain open. No LayaGo,
GLiNER, MLX, training-quality or speed gain was measured.


## L148 — Scope quotation location, preserve row identity and separate awards from payment (2026-10-06)

**Evidence:** [DelayLama scoped-quotation validation](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-06-scoped-quotation-recovery.md)
and [hashed receipt](validation/delaylama-scoped-quotation-recovery-20261006.json).

Global repetition had rejected an award association that occurred once in its own
exact citation. Matching inside independently located parent passages resolves that
physical occurrence, while distinct own locations remain ambiguous. A heading and
its contiguous body also retain an otherwise isolated record identity without
borrowing the next record's date. Source positions must survive both transformations
and Python scalar-to-browser UTF-16 conversion, including astral characters.

A bounded quotation selector can repair invented ellipses without changing the
statement's kind, subject, amount, date, polarity, modality or legal decision. It
selects prepared original spans only for unresolved evidence fields. Full raw
proposal and source-hash binding reject stale selections. A sole filtered request
with original row 2 initially received row 0; that response was correctly withheld.
Explicit original-index instructions recovered the quote in a separate diagnostic.
The earlier partial saved result remains unchanged. Selection success is source
location, not certification of the assertion or a permanent completeness result.

Full-stack browser review found another inference boundary: the LLM subtracted an
award from actual expense to recommend an unpaid balance. Revised prompt guidance
did not eliminate that action. A separate visible review warning now requires own
payment/allocation evidence; the raw action and gross entitlement remain inspectable.
A later draw also projected an unknown accounting date as an unknown incurred-cost
statement. Unknown time metadata must not erase a known amount or create apparent
contradictory expense coverage. Both failures remain training/validation examples.

**Training implications:** include repeated identifiers in requests and awards,
heading-only identities, verbatim versus invented focused quotes, negative scope,
nonzero filtered row IDs and Unicode positions. Label citation location separately
from subject association, statement function, polarity and date role. Distinguish
requested, awarded, actually paid, allocated and discharged values; derive an unpaid
balance only from supported payment allocation to the same obligation. Label unknown
accounting dates separately from quantified expense. Keep raw failures and partial
coverage, and evaluate new draws across agents and interface projections. Graph
reachability and exact quoting do not create semantic acceptance labels.

**Evidence limits:** 104 retained application call records, two native source jobs,
1,680 backend and 1,039 frontend tests plus build. The fresh two-source graph had
28 nodes and 43 edges; one three-step neighborhood retrieved both originals with
unestablished completeness. An `If Linden Owner Ltd` organization candidate, broad
payment types and a withheld structure item remain issues. One intermediate payment
recheck used the earlier copied prompt and is retained as a harness mistake. These
are synthetic development controls, not independent legal gold, held-out,
repeated-seed or causal accuracy evidence. No LayaGo/GLiNER/MLX architecture,
training-quality or speed gain was measured; no new model weights were trained.


## L149 — Missing details must not erase known outcomes (2026-10-06)

**Evidence:** [DelayLama outcome-detail validation](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-06-outcome-detail-scope.md)
and [hashed receipt](validation/delaylama-outcome-detail-scope-20261006.json).

A source-only extractor represented an unrecorded expense accounting date as an
unknown incurred-cost statement beside a quantified expense. Its output shape had
no way to identify which detail was unknown, creating an apparent whole-outcome
conflict. Separate amount/date/support scope is necessary even when original text
matching, polarity and subject identity succeed.

Source statements now distinguish outcome, value, date, support and unestablished
aspects. A recorded expense can retain its stated amount and empty date, with a
separate source-located date gap and missing invoice/receipt gap. Detail-only records
do not establish occurrence or whole-kind coverage. Genuine conflicting amounts
remain separate outcomes. Malformed aspects and definite facts put into gap facets
are withheld with raw proposals retained. Aspect/label are bound to the captured
statement; editing scope cannot reuse an earlier highlight/check.

The same distinction must survive every projection: Claim Desk, standalone results,
financial overview, analysis graph and source review. Missing support stays visible
in the entitlement decision without erasing a reported expense or becoming zero.
Historical gaps without an aspect remain unestablished; old arrays/review paths
are retained instead of moving feedback to a different finding. Explicitly scoped
gaps use concise labels and own evidence controls rather than redundant warning
boxes; format, binding and unknown-scope problems retain review concerns.

**Training implications:** label outcome occurrence, amount, accounting/statement
date and documentary support independently. Include a known expense with unknown
date, unavailable invoices for a reported amount, unspecified amounts, actual
payment versus award, and contradictory amounts with distinct own scope. A field
gap is not a negative whole-outcome label. Retain per-aspect quotation and subject
association. Evaluate typed output, incomplete-entitlement survival and all consuming
views, not just classification loss or sentence embedding similarity. These are
candidate training-design lessons, not new acceptance labels or model weights.

**Evidence limits:** two English development controls, 40 application calls (22 saved
claim recheck, 18 fresh six-agent workflow), two native source-analysis jobs, 1,688
backend and 1,044 frontend tests plus build. Claim profile stayed Gemini 3.5 Flash /
low reasoning. Fresh graph had 28 nodes and 42 edges; one three-step neighborhood
retrieved both originals with unestablished completeness. A source-path relocation
harness error remained terminal before a distinct corrected-source recheck. C140's
award-minus-expense recommendation, C141's related-notice time-right duplication,
native entity candidates and event quotation coverage remain issues. Synthetic
single draws are not independent legal gold, held-out/repeated-seed or causal
accuracy evidence. No LayaGo/GLiNER/MLX architecture/training or speed gain occurred.


## L150 — A source record and its claim subject need separate identities (2026-10-06)

**Evidence:** [DelayLama claim-subject validation](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-06-claim-subject-context.md)
and [hashed receipt](validation/delaylama-claim-subject-context-20261006.json).

A notice assessment named the underlying delay as its subject but the interface
linked only its record ID, making repeated time proposals resemble independent
rights. Source-bound subject context now keeps record, subject, target and role
separate. Unique identities, captured full basis, current whole-original hash and
own association span permit candidate navigation; they do not certify legal meaning.
Unknown or stale scopes stay unresolved. Different proposals, quantities and review
paths remain intact. Subject/target indexing is not verdict aggregation or summation.

The same distinction appears in graph navigation. The analysis graph has separate
record-reference and proposed-subject links, with the own association quotation.
The browser waits for the current original digest, sharing promises across cards.
An edited original blocks a stale link even if the quote still appears. Graph
inspection initially hid quotations on all structural references; the new subject
link now exposes its own quote with candidate wording. Long fact/review fields also
span the inspector rather than wrapping inside a narrow column.

A three-source control exposed a deeper error: “a notice alone does not grant
payment” became a substantive exclusion of the underlying delay claim. Matching
text and coherent relation fields did not make that interpretation correct. Scoped
guidance and a narrow English current-field warning distinguish notice contribution
from substantive exclusion or late-notice forfeiture. One recheck removed that
exclusion while leaving cost applicability unresolved with its provision retained.
The bad draw, changed cost wording and remaining procedure-only time judgment in
the saved case remain visible; navigation and a later draw are not adjudication.

**Training implications:** supervise record identity, assessed subject, event role,
association and rule scope separately. Include notice-only non-grant versus true
underlying exclusion, actual late-notice rules, grants versus future rights, and
several records reporting one subject's quantity. A repeated quantity is not another
right or independent evidence. Include changed whole-source versions with unchanged
local quotes, duplicate IDs/ambiguous associations and scalar-versus-UTF-16 positions.
Keep source-bound candidate links distinct from acceptance labels, and test evidence
visibility, indexing, counters and review navigation as well as extraction quality.

**Evidence limits:** 37 application calls, three native source-analysis jobs,
1,699 backend and 1,047 frontend tests plus build; claims retained Gemini 3.5 Flash /
low reasoning. The fresh graph had 28 nodes and 41 edges; one three-step neighborhood
retrieved all three originals with unestablished completeness. These English synthetic
single draws are not independent legal gold, held-out/repeated-seed, causal or complete
long-contract accuracy. Semantic entity/occurrence duplication, event quotation
coverage, related title/scope consistency, precise applicability/labels and payment
reconciliation remain open. No LayaGo/GLiNER/MLX architecture, new training/weights or
measured speed gain occurred.

## L151 — A field quote must establish its own occurrence and source kind (2026-10-06)

**Evidence:** [DelayLama event-field validation](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-06-event-field-evidence.md)
and [hashed receipt](validation/delaylama-event-field-evidence-20261006.json).

Occurrence-only quotations can leave an otherwise correct amount or duration
ungrounded. An independent source pass now checks populated event references,
dates, actors, causes, actual costs and observed durations. A field needs its own
source kind and occurrence scope; impacts additionally need the action's actor/date
premise and explicit attribution. Requested money, awarded payments and granted
extensions are distinct from incurred expense and actual elapsed delay. Matching
text, shape or model agreement does not certify any of those interpretations.

Short quotations repeated across requests, awards and expense records must not
default to their first occurrence. Require exact, unique own surrounding context
and one child match inside it. Invented ellipses and reconstructed passages remain
insufficient. Preserve original scalar positions and convert to UTF-16 for browser
highlights. Bind candidates to the complete original digest and captured event,
including financial-stage semantics; verification metadata enrichment alone should
not invalidate them. Never turn source-check candidates into human acceptance.

Integration checks matter: normal extraction kept its original in `document`, not
the separate `citation_document` field, initially skipping the new pass. Raw skipped
stages are retained. An initial UI projection also exposed private metadata and
changed legacy JSON expectations; protocol-gated selected fields restored those
tests. Per-field expandable summaries avoid flooding the event register with
repeated excerpts. Preview context stays within its own original document while
validation still binds to the whole original. Existing review concerns remain
separate from automatic candidates.

The fresh three-source control preserved EUR 1700 actual expense/two days actual
delay separately from a EUR 6200 request/seven granted completion days. Initial
field checks used invented ellipses and omitted actor/date premises. Explicit
contiguous-quote and occurrence instructions preceded one recheck that returned
own cost/duration candidates, with cause/reference gaps still visible. A separate
frozen deliberately wrong EUR 6200/seven-day occurrence proposal received
contradictions in one six-field batch; saved output stayed unchanged. These are
observed single development draws, not causal accuracy estimates.

**Training implications:** supervise source span selection, occurrence identity,
field kind, own-versus-related scope, actor/date anchors and explicit attribution
as separate tasks. Generate hard negatives with identical amounts in requests,
awards and incurred expense; granted versus observed days; headings separated
from facts; invented ellipses; duplicate references; changed originals with intact
local quotes; and ambiguous subquotes requiring own context. Distinguish an invoice
documenting expense from the act of issuing or paying it. Use unknown/missing fields
and incomplete premises as abstention examples, not false zero/absence labels.
Keep all failed/intermediate proposals and UI binding failures in experiment records.

**Evidence limits:** 77 application calls and three native source jobs, excluding
native internal calls; 1,705 backend and 1,051 frontend tests plus build. Event checks
retained Gemini 3.1 Flash Lite, temperature 0.1, 4096-token output budget and zero
transport retries. The native graph had 36 nodes/58 edges; a three-step neighborhood
retrieved all three originals, with unestablished completeness. Duplicate access
blocks and an unsupported invoice-issuance action remain in the native candidates.
This populated-field pass does not establish omitted-field recall, native entity
resolution, legal applicability or payment reconciliation. Synthetic English cases
are not independent legal gold, held-out/repeated-seed or long-contract accuracy.
No LayaGo/GLiNER/MLX architecture, training, weights or measured speed gain occurred.

## L152 — Ground nominations independently and distinguish documentary facts (2026-10-06)

**Evidence:** [DelayLama action-grounding qualification](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-06-action-grounding.md)
and [hashed receipt](validation/delaylama-action-grounding-20261006.json).

A generated initial event quote omitted an intervening awareness sentence. The
independent grounding model copied that reconstruction when it was included in
its input. Removing earlier generated excerpts, supplying only the nominated
label/calendar plus the full original, and showing a contiguous own-action example
preceded successful source selection. Matching source contexts are not permission
to fabricate missing quotation text. Invalid/ambiguous/missing cards remain review
leads. Exact own spans may reconcile nominations with inventory; partial overlap
must not silently merge different predicates or bundled actions.

An invoice documenting expense was also named as an issuance event. Positive
`factual_statement`/`SourceFact` is now separate from `observed_action`: documentary
support, a reported quantity or state does not establish issuing, delivering or
paying. The fact's own assertion supplies its display name; earlier labels survive
as proposals. Ungrounded nominations are `ActionCandidate` records outside executed
events, without occurrence calendars or actual performer/causal roles. Original
proposals remain inspectable. Source facts and proposals are not human acceptance.

**Training implications:** contrast documentary support versus actual issuance,
receipt versus sending, expense versus request/award, generic state versus a stated
change, and observed versus normative/deadline language. Include old generated
excerpts that reconstruct or omit intervening text, but keep them out of the
independent selector's evidence input. Supervise own assertion, enclosing context,
statement kind, occurrence identity and scalar/date ownership separately. Include
repeated quotations, overlapping clauses and source changes; retain abstentions
and rejected predictions instead of forcing fuzzy identity matches. Source matching
and candidate classifications need separate labels from semantic acceptance.

**Engineering learning:** clean-source equivalence caught a missed calendar patch
in the isolated development baseline. An invalid positional patch command had
failed while later shell commands continued. Earlier live runs and Go tests did
not qualify the full default series. The baseline was corrected, full Go/races
repeated, source files matched to the clean build, and a distinct qualified project
ran before activation. A variant idempotence probe also needed each arm's original
flags; the preflight refused writes when those were omitted. Preserve intermediate
failures, verify the actual compiled source set and avoid treating a passing suite
on the wrong baseline as release evidence.

**Evidence limits:** 124 application calls, 12 native source jobs (internal calls
excluded); 1,707 backend and 1,054 frontend tests plus build, full corrected Go,
relevant races, 12 patch arms and 11 variant suites. Final control grounded one
initial block and withheld overlapping inventory, preserving one performed block;
invoice documentation remained a source fact. The explicit issuance positive was
also retained. Early native draws used the incomplete development baseline; these
remain reported separately from corrected default69 qualification. Synthetic English
reused originals are not independent gold, held-out/repeated-seed, causal or complete
contract accuracy. Actor ownership, applicability, notice triggers/compliance,
source recall and payment reconciliation remain open. No LayaGo/GLiNER/MLX model,
training, architecture, weights or measured speed improvement occurred.


## L153 — Actor ownership follows the own predicate, not a party label (2026-10-06)

**Evidence:** [DelayLama report](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-06-event-actor-ownership.md)
with [hashed receipt](validation/delaylama-event-actor-ownership-20261006.json).

**Observed failure:** a notice receipt was assigned to the party supplying the
notice. A grant by the owner received actor support from an earlier owner blockage
passage. Shared party names and a possessive document label did not identify the
performer of the selected predicate. Authorship is not dispatch; beneficiary status
is not granting. Conversely, the recipient *does* perform receiving when that own
predicate is expressly reported.

**Implementation learning:** require separately quoted own occurrence/calendar and
actor-role premises. Preserve the requested actor, literal source performer proposal
and role separately. Unique original Markdown record headings can bound citation
ownership, while incidental “notice for R-1” headings cannot replace R-1's own
heading. Repeated headings, generated source wrappers, missing quotes or roles stay
unresolved. This narrow formatting guard is not alias resolution or semantic proof.
Read-time revalidation prevents old weak support badges surviving new requirements,
without rewriting model history or making a model call. Manual and automatic field
checks share the role guard; changed source/core details withhold current support.

Place a full role-bearing JSON example after older impact/cause examples. The first
live response copied the old shape and omitted role fields; a later draw called a
grantor a document author. Clear predicate examples improved these proposals but did
not eliminate conservative/incorrect evidence. A completed act of drafting needs
coverage even if the drafted document proposes an unapproved effect. Preserve the
act separately from the document's proposed effect and from sending/receiving.

**Another dimension failure:** an extracted seven-day extension occupied a
`payment_award` record with amount `7 days`. A narrow unit warning now exposes this
without rewriting the original. Explicit time-versus-money extraction guidance was
followed by a separate time-grant category and empty financial records in one final
run. Rates such as EUR 7 per day remain monetary expressions. Do not confuse granted
time, observed delay, requested money, incurred expense, payment award or payment.

**Training implications:** include minimal contrasts sharing actors, document names
and quantities but changing the predicate. Train sender/receiver/drafter/grantor
ownership, own record identity, action dates and scalar dimensions separately. Use
multiple actions by the same party, possessive notices, third-party drafting,
beneficiary grants, duration-as-money negatives and daily monetary rates. Include
missing/duplicate role cards, repeated headings, incidental references, changed
source versions and absent premises as abstention examples. Graph proximity and
matching parties must not become a positive label for an unsupported predicate.
Evaluate extraction recall separately from source-location/role coherence and human
semantic acceptance. Preserve withheld and wrong predictions for evaluation.

**Measured limits:** 143 application calls and three native source jobs (internal
calls excluded), six application workflows submitted once; 1,721 backend tests,
1,058 frontend tests across 93 files and production build passed. Native C147 graph
had seven separate dated action candidates with expected actor proposals; extraction
returned six and omitted a separate awareness entry. A negative receipt-actor batch
refused the possessive/sender substitution. Final time-grant extraction preserved
expected actors/dates, but four parent actor verdicts remained insufficient despite
complete premise cards. No reroll or automatic promotion forced support. Assigned
analysis IDs also changed; claim analysis was rerun against that final generation.
Stable cross-generation linkage, parent evidence ergonomics, aliases, notice
applicability, payment reconciliation and long-contract recall remain open. Reused
synthetic English development originals are not independent gold, repeated-seed,
held-out or multilingual accuracy. No LayaGo/GLiNER/MLX training, architecture,
weights or measured speed improvement occurred.


## L154 — Capture event inputs before joining claims across reruns (2026-10-06)

**Evidence:** [DelayLama report](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-06-claim-event-inputs.md)
with [hashed receipt](validation/delaylama-claim-event-inputs-20261006.json).

**Failure:** a detector rerun reused E1 for drafting instead of blockage. Reading an
older claim against the latest register joined it by that reused display ID or title
overlap. A shared original-source hash did not detect this namespace change. The
claim's own record and its declared underlying subject both need input binding.

**Implementation:** claim analysis captures the normalized event proposals actually
supplied and the original-source SHA. Each assessment and standalone fact/detail
records only its referenced record/subject inputs, with full-roster ambiguity markers;
the full roster is retained once. Material action/type, actor, date, original reference,
status, impact, financial-stage and citation text changes withhold current navigation.
Evidence-check/location metadata and unrelated added events alone do not detach an
unchanged proposal. Old references, verdicts, values and quotes remain unchanged.
Legacy uncaptured results require reassessment; no title-based migration or automatic
alias relink is inferred. The desk, subject summaries, analysis map and common evidence
review expose connection gaps. Snapshot diagnostics stay out of ordinary facts and
model analysis context. Matching proposals permit candidate navigation, not accepted
identity, legal support or complete source understanding.

**Persisted qualification:** 12 legacy assessments remained separate. One fresh
claim workflow, submitted once, made 29 application calls and captured 12 assessments
plus six standalone statements. All current captures matched. A controlled saved
output swapped only the first two detector display IDs; this was not another model
extraction. All 12 affected assessments detached, with visible earlier-version
warnings and no current claim record/subject edges in the analysis map. Raw claim
values, original documents and every native graph table stayed identical. Restoring
the old event inputs as another owned version restored all 12 links without a model
call. Tests also protect unrelated additions and check-only metadata changes.

**Training/evaluation implications:** separate semantic/action identity from generated
display IDs and version lineage. Include reordered registers, reused E-numbers,
changed related subjects, renamed or missing actions, ambiguous original references,
changed actor/date/impact/stage/quotation, stale source versions and absent captures.
Use unchanged metadata-only updates and unrelated new actions as positive controls.
A consistent source quote or original hash alone cannot label a cross-generation
join correct. Evaluation must detect wrong joins, withheld valid joins, ID instability
and source recall separately; guardrail abstention is not improved semantic extraction.
An empty-list subject initially bypassed the optional-subject check and now fails
closed. Preserve malformed/failed examples instead of coercing them into identity.

**Limits:** 1,733 backend tests, 1,062 frontend tests across 94 files and production
TypeScript/build passed; bundle warning remains. One reused synthetic English C147
case and one fresh claim draw are not independent gold, held-out, repeated-seed,
multilingual or long-contract accuracy. No new native source jobs or Go changes;
the graph reused qualified default69. Stable cross-generation identity resolution,
other notice/risk/amendment references, parent actor evidence, applicability, recall
and payment reconciliation remain open. This increment withholds stale links rather
than solving semantic entity resolution. No LayaGo/GLiNER/MLX training, architecture,
weights or measured speed improvement occurred.


## L155 — Separate display numbering from unchanged captured inputs (2026-10-06)

**Evidence:** [DelayLama report](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-06-claim-numbering.md)
with [hashed receipt](validation/delaylama-claim-numbering-20261006.json).

**Problem:** L154 correctly withheld reused display IDs, but a pure numbering change
also detached otherwise identical captured actions and could force unnecessary claim
analysis. Current navigation needs a separate number while the original analysis
retains its own references, source checks and reasoning.

**Implementation:** compare the full stored proposal snapshot with current snapshots,
excluding only the display ID, on unchanged original-source SHA. Require unique
matches, full-register duplicate-proposal information and injective record/subject
mapping. New compact bindings carry ambiguity markers; legacy rows can use their
actually saved full roster only after exact captured-row agreement. A selected subset
or today's roster cannot manufacture old uniqueness. Missing/duplicate records,
material/source changes, malformed or non-finite values and identity collapse remain
unresolved. Booleans are distinct from numbers; no title/alias/similarity matching is
used. Read-only numbering metadata exposes both old and current IDs without changing
saved fields or promoting semantic certainty.

The desk, rule/source panels, outcome/detail/standalone labels, subject summaries and
analysis-map record/subject edges now use current navigation while showing original
analysis references. Quotes, check IDs and free-text reasoning stay original. Quote
coherence is checked against the occurrence actually assessed, not an unrelated action
that reused its display label. Input equality permits candidate navigation, not
accepted meaning, attribution, applicability or entitlement.

**Measured controls:** legacy C147 E1/E2 swapping preserved 12 correctly mapped claims
using the real stored roster, with no model call. Changing the blockage actor proposal
then withheld 10 affected assessments while two unrelated assessments stayed linked.
One fresh claim run, submitted once with 31 application calls, recorded modern capture
markers and returned 12 assessments plus seven standalone statements. Renumbering all
six events to E101–E106 retained those connections without further model calls. Original
source/raw claims/native graph rows stayed identical during navigation controls; direct
numbering was restored as another owned version. The live rule panel retained its old
check ID and showed current subject E102 beside analysis E2. The analysis map had 96
items and 115 candidate references, with the renamed dispatch and blockage targets.

**Training/evaluation:** do not make generated E-numbers semantic identity labels.
Separate numbering-only positives, material-change negatives, old/current namespace
collisions, identical repeated proposals, missing old-roster evidence and record/subject
collapse. Preserve original IDs in evaluation artifacts. Measure correct connections,
wrong relinks, abstentions, recall and semantic judgments separately. An unchanged
snapshot is not a substitute for correct extraction or independent legal gold.

**Limits:** 1,738 backend tests, 1,064 frontend tests across 94 files and production
TypeScript/build passed; bundle warning remains. Captured citation identity covers
quote/clause/section; location-check metadata is excluded. Unknown/incomplete captures,
changed wording or aliases require reassessment; this is not general semantic entity
resolution. The fresh model also returned a malformed/unreadable rule-basis candidate,
whose concern remains visible. One reused synthetic English dossier and one new claim
draw are not held-out, repeated-seed, multilingual or long-contract accuracy. Other
notice/risk/amendment and event-field-check references, recall, applicability and
payment reconciliation remain open. No new native source jobs, Go changes or LayaGo/
GLiNER/MLX training, architecture, weights or measured speed gain occurred.


## L156 — Make reassessment an owned, recoverable workflow (2026-10-06)

**Evidence:** [DelayLama report](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-06-claim-reassessment.md)
with [receipt](validation/delaylama-claim-reassessment-20261006.json).

A review gap needs an actionable next step. The contract workspace now reassesses
claims directly using server originals and the latest source-bound event register.
It verifies project/source provenance, rejects stale/unknown/fallback registers,
uses the project's language and configured prompt, and preserves earlier results.
Stale inputs offer a direct route to event detection rather than silently reusing
an unrelated register or cached preview.

A synchronous click lock, existing-run lookup and durable request receipt prevent
unintended duplicate work. Unknown submission/progress recovers the same request or
handle. A saved analysis whose refresh fails retries only the view refresh. Project
changes suppress late UI writes and stop observation while server work remains
recoverable. Foreign run history/progress and offline mode fail closed. These are
orchestration guarantees, not guarantees about the model's semantic conclusions.

One real UI click on retained synthetic English C147 created one terminal saved
claim run with 28 application calls, 12 assessments and 12 current input bindings.
The desk/map refreshed in place; the previous analysis, source and register were
unchanged. The copied journal ended with 69 terminal jobs, 68 done and one inherited
error. There were no new native source jobs. All 1,073 frontend tests across 95 files
and production TypeScript/build passed, including nine new workflow tests. Backend
code was unchanged from its prior 1,738-test qualification. Bundle warning remains.

**Shared engineering lesson:** uncertainty about transport, observation or display
refresh must not become a new training/analysis run. Keep run identity separate from
result visibility, persist receipts before dispatch, distinguish terminal failures
from unknown status, validate ownership at every observation and retain old outputs.
These patterns also apply to training supervisors, evaluation jobs and GPU workers.

This is one development workflow, not held-out, repeated-seed, multilingual or
long-contract extraction accuracy. No LayaGo/GLiNER/MLX model, training, architecture,
weights or measured speed improvement occurred. Semantic quality and broader graph
qualification remain open.


## L157 — Balance retrieval and separate obligations from occurrences (2026-10-06)

**Evidence:** [DelayLama report](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-06-graph-notice-correctness.md)
and [receipt](validation/delaylama-graph-notice-correctness-20261006.json).

A passage budget can create systematic omissions before a model sees its input.
Direct topic matches now precede adjacent filler, with hierarchical rotation across
originals and query queues. A three-original/twelve-query stress case retained eight
passages per original in 24 slots and reported twelve direct omissions. Exact quotes,
hashes and offsets remained intact. Numeric page limits must not be interpreted as
continuation flags. Ranking and semantic relevance remain different questions.

Graph context needs unique fact, connection and document identities before selecting
witnesses. A source status must agree with its actual citations. The review panel now
shows retrieved versus displayed counts, keeps older context explicitly labeled after
failed refresh, and places human facts and quotations before expandable diagnostics.
Live source context showed fifteen semantic facts, eighteen relationships and three
originals with thirty-three current source matches; that verifies location and UI
connectivity, not their interpretation or the completeness of the contract graph.

A contractual obligation is not its triggering occurrence. An unbound event ID must
not borrow a date from a later register generation. Notice rows now withhold that
join and expose missing trigger support; explicitly supplied dates remain estimates.
Sending does not establish timely service, and estimated lateness does not establish
forfeiture. Conflicting/malformed dates, normalized invalid timestamps, alternative or
negative windows and calendar overflow require abstention. Business-day arithmetic
must not iterate millions of times on an erroneous model value.

For LayaGo training/evaluation, candidate ordering, budgets, identity collisions and
unit/date validity should be independently tested outside semantic model scoring.
Build hard examples with dispersed conditions, reused display IDs, distinct awareness/
dispatch/receipt, and missing or conflicting triggers. Group related variants before
splitting. A fast head can nominate or challenge candidates; it cannot supply missing
provenance, prove applicability or certify legal conclusions from shared words.

The increment passed 1,754 backend and 1,106 frontend tests plus production build.
One existing async assertion and an overly broad warning deduplication were corrected.
No new model calls or training occurred; this is an adopted engineering increment,
not evidence of improved LayaGo/GLiNER/MLX weights, speed, calibration or held-out
contract quality. Lexical recall, temporal applicability and reconciliation remain open.


## L158 — Locate the starting point before choosing a notice date (2026-10-06)

**Evidence:** [DelayLama report](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-06-notice-source-resolution.md)
and [receipt](validation/delaylama-notice-source-resolution-20261006.json).

A notice receipt is not the start of an awareness-based clock. Three retained
synthetic C147 analyses exposed distinct failures: initial selection of receipt with
placeholder/missing kind metadata, then correct awareness with a cropped “that day”
date quotation, then a successful bounded quote repair. The final automatic output
and real GUI retry both proposed awareness on 1 July 2098 and estimated 8 July using
the original seven-day rule. They remain unaccepted model candidates; source location
and coherent metadata are not semantic or legal certification.

Every obligation now carries five independent source checks: obligation identity,
starting-point rule, own occurrence, applicability and date. Capture source SHA plus
unchanged notice inputs, preserve alternatives, and recompute estimates instead of
trusting saved countdowns. Document slugs/display IDs are not occurrence references.
The dated antecedent and anaphoric action must remain in the actual own quotation;
merely putting “that day” into a date card cannot establish a complete calendar date.
A single formatting/quotation repair preserves its first checked output and must not
persuade insufficient or contradictory semantic judgments into support.

Engineering results: ten recorded application LLM calls (three extraction jobs and
one GUI retry), successful project-backed review restore while browser storage was
unavailable, no altered originals/earlier extraction files and no native source jobs.
1,859 backend and 1,210 frontend tests plus production build passed. Actual EN/FR
window parser parity passed 3,052 synthetic cases. Nested financial evidence no longer
inflates event counts or disrupts event-reference navigation. Those observations are
workflow/representation qualification, not a fresh held-out model-quality benchmark.

For LayaGo examples, vary awareness, drafting, dispatch and receipt dates independently;
vary the contractual starting point while keeping surrounding entities/records similar.
Group related variants before splitting. Include anaphora, incomplete dates, changed
rules, multiple plausible occurrences, spelled-number windows and contradictory
corroboration. Score rule selection, occurrence identity, date ownership and literal
quote validity separately. Preserve rejected/repaired draws rather than reporting only
the final successful candidate. No new LayaGo/GLiNER/MLX training, architecture, weights,
calibration or speed change was measured. Semantic recall/applicability remain open.

## L159 — Bind finalized inputs and retain failed review history (2026-10-06)

**Evidence:** [DelayLama report](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-06-notice-review-history.md)
and [receipt](validation/delaylama-notice-review-history-20261006.json).

A proposal bound before extraction finalization can invalidate itself after harmless
evidence defaults or original whitespace restoration. Capture the final canonical
source fields, using the same deterministic normalization before and after the
model pass. Preserve caller inputs and keep invalid extraction quotations unverified.
Exercise the actual finalization pipeline, not only the source-check helper.

Prior-notice durations need explicit direction: adding seven days to commencement
cannot implement “seven days before commencement.” Unsupported EN/FR prior-notice
forms now abstain. Check the qualifier attached to the duration, not unrelated later
headings. Python/TypeScript equality passed 3,073 synthetic parser cases, including
21 direction/context controls; this does not validate calendars or legal meaning.

Review history is learning data. Retain initial rejected date quotations, later
repairs, each explicit retry and the latest result separately. A dated antecedent
must appear in the own date quotation; “that day” alone is insufficient. Surface
this history in result/graph evidence and pass it into prompt feedback. Correction
targets must still use actual current field names/values, rather than adopting
model-supplied replacement values. Old result/source reviews cannot migrate merely
because the displayed notice name is unchanged.

The retained C147 workflow used 15 recorded application calls: one automatic
extraction and four GUI retries across two saved analysis contexts. Current review
revision 3 restored two searches while browser storage was unavailable; prompt
iteration received the note, earlier search, latest search and rejected repairs.
Forfeiture judgments differed between the two contexts (insufficient versus support
inferred from a condition). Both remain flagged/unaccepted. A matching quotation and
consistent date metadata do not settle semantic applicability.

For LayaGo training examples, preserve such disagreements and unsuccessful draws;
separate explicit forfeiture from notice conditions, and prior from subsequent
windows. Split related variants by source family. Score normalization/binding,
literal quote ownership and semantic judgment independently. This increment passes
1,883 backend/1,240 frontend tests plus build; it measures engineering workflow,
not held-out model accuracy. No new LayaGo/GLiNER/MLX training, architecture, weights
or speed gain was measured. Broader recall, exceptions, temporal applicability and
payment reconciliation remain open.

## L160 — Separate notice effects and retain original qualifications (2026-10-06)

**Evidence:** [DelayLama report](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-06-notice-consequence-evidence.md)
and [receipt](validation/delaylama-notice-consequence-evidence-20261006.json).

A conditional entitlement is not automatically an express loss-of-rights provision.
Likewise a penalty, negated loss wording, a future conditional bar and an actual lost
claim are separate targets. Train effect kind, polarity, explicit/inferred basis,
obligation scope and explicit association separately. Keep coexisting effects and
unknown classes; a false extraction flag cannot prove whole-contract absence.

Blind source checks should exclude earlier flags, consequences and rationale. Bind
checked proposals to finalized scope fields and the original source digest. Graph
neighbors may retrieve original passages; derived edges and display names cannot
substitute for the effect's own supporting quotation.

A correct short effect quote can omit a following waiver signature or document
requirement. Preserve the complete own original paragraph and exact span, and score
qualification inventory separately from literal quote coverage. Retaining that
paragraph does not prove recall of all dispersed exceptions or that a condition was
fulfilled. Keep oversized context whole and unresolved rather than silently clipping.

One bounded repair can fix structural quotation gaps while holding the original
proposition fixed. Do not let it rewrite a claim to match a new quote or turn a
semantic insufficient result into support. Preserve initial rejected draws, repaired
results, retries and transport failures in review feedback and training diagnostics.

Four frozen synthetic EN/FR development families qualified expected effect categories
in 1/4 initial, 1/4 intermediate and 4/4 final runs. Guidance changed in response to
observed failures, so these are regression probes, not held-out model accuracy.
Signed waiver/acceptance requirements were preserved in full paragraphs and final
French condition lists. C147 retained its conditional notice rule, raw false flag,
independent awareness trigger and date estimate; its false-flag support remained
insufficient, flagged and unaccepted. All 56 recorded application calls across these
runs and one manual retry are accounted for. Python/TypeScript parity passed 1,200
calendar estimates and 48 lexical loss controls; lexical matches are not semantics.

For LayaGo datasets, generate paraphrases, cross-references, conditional exceptions,
negation, multiple coexisting effects, changed rules and independent service facts.
Split related source families before synthetic expansion. Evaluate retrieval recall,
effect classification, scope/link ownership, qualification recall, quote/span validity
and calibrated abstention independently. UI engineering tests do not measure these
model objectives. No new LayaGo/GLiNER/MLX training, weights, architecture, throughput
or OOM supervision improvement was measured in this increment.

## L161 — Date representation, prior notice and stable review identity (2026-10-06)

**Evidence:** [DelayLama report](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-06-notice-direction.md)
and [receipt](validation/delaylama-notice-direction-20261006.json).

Retrieval can introduce a representation error even when the original date is
correctly quoted. Two fresh EN/FR prior-notice dossiers qualified in document-only
runs, but graph-ready runs copied normalized ISO graph dates into verbatim source
fields. Literal source checks withheld both estimates. A graph value can guide
navigation; it must not replace original wording or its supporting span. Keep
verbatim date, normalized date, date role, reference identity and occurrence status
as separate supervised targets. Test with and without derived navigation context.

Independent notice-trigger checks now receive checked original navigation passages
and explicit original-date instructions. Both final graph-aware runs qualified the
same planned references: seven calendar days before July 20 gives a July 13 estimate;
five weekdays before July 21 gives July 14. Original planned wording remains intact;
these examples do not establish actual occurrence, service or lost rights. All six
model draws, including failed ISO representations and repairs, remain available.

Direction belongs to the own duration/reference, not a distant “before” heading.
Match count, own unit and direction independently against original rule evidence.
Separate latest boundaries from earliest/strict/alternative intervals, keeping
unsupported forms unresolved. Calendar/month arithmetic and weekday counting are
engineering estimates, not proof of an applicable counting convention. Hourly
values based on dates alone need visible approximation labels.

A review interface should preserve exact finding identity during search/filtering.
Keeping the selected notice open prevents an identically named event from silently
replacing its context. Filtered-out or unavailable IDs need explicit explanations
and recovery controls, not guessed label matches. Otherwise feedback can target the
wrong finding even when source validation is correct.

5,292 actual Python/TypeScript calendar comparisons had zero mismatches. The scoped
native graph audit checked 31 exact citations, no dangling/cross-project nodes, and
planned action/date roles throughout these fixtures. Final suites passed 2,048
backend/1,341 frontend tests, build, two actual-output frontend replays and browser
checks. The 24 recorded notification-agent calls exclude consolidation/native graph
model calls. Two synthetic development dossiers are not a held-out quality benchmark.

For LayaGo evaluation, group date-format and temporal-direction variants by source
family before splitting. Score literal representation, normalization, reference
ownership, direction, planned/actual status, retrieval contamination, applicability
and abstention separately. Preserve failed attempts and graph/document-only pairs.
No new LayaGo/GLiNER/MLX training, architecture, weights or throughput was measured.
Broader dispersed-rule recall, calendar conventions and outcome reconciliation remain
open beyond these regressions.

## L162 — Reviewable temporal representations and missing contract context (2026-10-06)

**Evidence:** [DelayLama report](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-06-timeline-review.md)
and [receipt](validation/delaylama-timeline-review-20261006.json).

A qualified notice reference and its calculated due date are two distinct points,
not the start/end of a performed action. Bind both to current originals and the
unchanged notice; retain a missing-date review task when input identity, direction,
reference selection or manual overrides are unresolved. Graph/event names alone
cannot populate a clock. Preserve the exact owning finding and distinct marker key
through review, filters and export.

Temporal datasets should distinguish reported actions, plans/proposals, context,
schedule dates and derived notice estimates. Conflicting classifications require
abstention rather than choosing a familiar label. Known unrelated interval endpoints
and disputed schedule dates should remain raw review candidates without being plotted
as precise spans. Independent ownership of dispersed endpoints remains a separate
learning/evaluation target; an explicit pair of extracted fields is not semantic proof.

Exports need the checked own date and rule passages, not only a section's general
citations. Location counts do not prove the computed date's meaning or applicability.
Representation tests should include changed sources, stale captures, conflicting raw
triggers, explicit null reviews, duplicate identities and missing selected entries.
Strict UTC calendar construction is also needed when supporting years 1–99; the
JavaScript Date constructor's 1900 offset can otherwise misrepresent a valid source.

Missing identity values should not produce a wall of empty cards or erase a supplied
identity when claims are absent. Keep useful values prominent and incomplete sections
reviewable, without inferring that the underlying contract lacks those facts.

Browser verification reused the prior synthetic English/French dossiers and their
saved model outputs: separate July13/20 and July14/21 notice points, exact review
navigation/restoration, and source-complete CSV rows. No new model-quality benchmark,
LayaGo/GLiNER/MLX training, weights, architecture or speed measurement was performed.
Dispersed interval attribution, broader exception recall, calendar applicability,
actual service and outcome reconciliation remain open beyond this UI increment.

## L163 — Dispersed activity intervals and source-identity repair (2026-10-06)

**Evidence:** [DelayLama report](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-06-activity-intervals.md)
and [receipt](validation/delaylama-activity-intervals-20261006.json).

Treat the activity and its individual occurrences as separate representations. One
original work-package identity may connect a planned start/finish, an actual-source
start/completion, and an invoice, but those dates cannot substitute for one another.
Keep plans/proposals and execution candidates separate; an interval does not establish
actual duration, performance, delay attribution or entitlement. Bind current originals,
raw interval fields and the owning current event register, rather than using captured
metadata as current authority.

The four-document D150 development case exposed a repeatable quotation problem.
Normalized activity names could be literal in execution records but absent from the
quoted definition, which contained connecting words such as “is”. Repeated bare labels
also lacked a unique source location. Require identity labels and their supporting
quotations to agree verbatim, preserve whole own paragraphs, and repair quotation
structure using full located own passages. Do not change candidate values or persuade
insufficient semantic support. Retain every failed check and every development draw.

Four draws on the same frozen fixture progressed from an omitted plan/unresolved
actual identity, through both intervals discovered but withheld, to a qualified plan
with unresolved actual identity, then both qualified source proposals. A separate
manual retry qualified the unchanged first actual proposal and retained its initial
failure. These are development improvements, not independent recall/accuracy results.

Potential future dataset distinctions include original activity versus record/notice
identity, own start versus completion, planned versus performed-source wording, invoice
issuance/payment request versus actual expense, and document date versus reported work
date. Hard negatives should include all five model checks claiming support for an
invoice/notice boundary, mixed calendar days, missing qualifiers, duplicate identities,
and payment “completion” mistaken for work completion. Recognizable verbs and one
matching date are necessary vetoes, not semantic subject-ownership certification.
Human-reviewed role and predicate labels remain necessary for a reliable evaluation.

Independent graph audit confirmed one K88 entity connecting nine separate source event
nodes, with project/source hashes scoped correctly. Located quotations for 39/40 nodes
and 71/71 edges did not certify their meanings; a duplicate regex monetary node stayed
unsupported. Native first-date parsing also assigned September 1 to several indexed
quotation containers. Preserve such dates as indexing/audit context, distinct from
occurrence dates; DelayLama isolated them without mutating original nodes or native
records. The native parser issue remains open.

Review UX should expose both own boundaries, source premises, full context and repair
history without vectors or repeated raw fields. Qualified source candidates should not
remain in a missing-quotation queue, and accepting a generic quote should not qualify
an interval. Exact source/candidate identity must survive graph inspection, timeline
filters, manual retry and prompt iteration. Asynchronous source hashing needs explicit
readiness in presentation tests, without weakening the production binding.

Final DelayLama checks passed 2,126 backend and 1,458 frontend tests, a production build,
and 166 Python/TypeScript comparisons including all eight saved interval reports.
No new LayaGo/GLiNER/MLX training, weights, architecture or speed result was measured.
Held-out multilingual recall, linked conditions, repeated identifiers, calendar rules,
actual service, performance scope and outcome reconciliation remain separate targets.

## L164 — Boundary subject ownership and competing activity scopes (2026-10-06)

**Status:** Adopted engineering safeguards; synthetic development observations, no new model training or held-out accuracy claim. Evidence: [DelayLama scope receipt](validation/delaylama-activity-scope-20261006.json).

Five model checks saying “supports”, an exact quote and one correct date can all be present while payment, invoice, report or document completion is wrongly treated as work completion. Necessary English/French subject/object guards should distinguish administrative completion from actual work reported inside an administrative record. Preserve failures and original paragraphs; neither a blacklist nor recognized grammar certifies arbitrary semantic ownership. Synthetic hard negatives should deliberately supply all five positive model checks for a wrong subject, with paired positives containing genuine work completion inside invoices/reports/notices.

Different current-supported boundary pairs on a literal activity ID and role are scope alternatives, not automatically contradictions. They can be separate work cycles or schedule revisions. DelayLama now retains every pair and withholds exact duration bars pending cycle/revision/applicability review; plans remain separate from execution. Duplicate dates, stale checks and generic human acceptance cannot decide which pair applies. Current sibling lists and effective manual source checks must be revalidated; captured collection metadata is only a historical audit. Normalize identifiers conservatively and identically across runtimes.

A frozen three-document S151 full-stack draw retained two actual completions and payment as distinct actions. Both actual intervals independently qualified as source proposals; the planned interval remained unresolved because the model paraphrased its own end description. Preserve that failure, do not count a date discovered without an own checked boundary as successful extraction. Payment was also mislabeled as payment_award: distinguish completed transaction from monetary award/decision, incurred expense and settlement, with separate own-source stage labels. Exact date checks do not validate the stage.

The independent graph audit found 21 nodes/27 edges and 46 located node/edge proof sets, all matching original spans and hashes. Two reconstructed contract-edge quotations were unmatched and stayed unsupported; a payment/work association remained a model suggestion, not proof of performance. High quote-location coverage does not establish predicate meaning, scope or completeness.

Review UX should expose scope alternatives separately from missing quotations, retain all diagnostic history but prioritize a few readable concerns, and send sibling candidates/current source checks into prompt feedback as untrusted data. Withheld dates must be labeled differently from absent dates. Potential reusable training/evaluation units are action subject/object ownership, own boundary date, payment stage, revision/cycle applicability and graph predicate scope. Human-reviewed multilingual held-out data remains necessary before claims of extraction quality or speed improvement. No LayaGo weights, MLX architecture or training throughput changed in this round.

Final DelayLama validation passed 2,218 backend tests, 1,516 frontend tests across 111 files, a production build, and 268 cross-runtime comparisons with zero mismatches. Browser review notes and both boundary candidates transferred to Prompt Studio and saved project backups. These checks establish specific engineering behavior, not model accuracy or complete contract understanding.

## L165 — Original scope ownership, bounded discovery and shared review history (2026-10-06)

**Status:** Adopted application engineering safeguards and synthetic development observations. Evidence: [DelayLama scope-resolution receipt](validation/delaylama-activity-scope-resolution-20261006.json). No new LayaGo, GLiNER, MLX training, weights or throughput results.

A parent work package can contain several original execution cycles and several planned schedule versions. Source qualification must bind an exact identity to both same-occurrence boundary paragraphs, preserve the full original qualifications and revalidate the whole current inventory. An invoice tag, record identifier, nearby heading or identical sentence in another cycle cannot establish this scope. Distinct checked versions remain plans; a version with signature conditions is not selected as operative. Same-cycle competing dates and unnamed competing occurrences remain unresolved. Human acceptance cannot bypass source ownership or make a current unknown sibling disappear.

Separate compact proposal discovery from citation-rich qualification. A frozen five-document D220 dossier originally yielded only two of eight expected interval proposals. After compact source-only discovery and one-candidate five-premise evidence batches, all eight expected pairs qualified against unchanged originals. This is development evidence, not held-out accuracy or complete recall. Retain every original model draw, omitted format-budget candidate, failed premise and repair call rather than replace an unfavorable draw. A larger output token limit alone is not a completeness strategy.

Case-sensitive own quotations can fail when a cropped uppercase definition omits the unchanged lowercase scope label present in its full own paragraph. One controlled structural repair can expand those same current paragraphs only if every semantic premise is already coherently supported and the unchanged claim independently qualifies with exact scaffolds. Missing identity, semantic insufficient/contradictory checks, source denials, wrong occurrences and unnamed scopes are not persuasion targets. An actual provider copied an example request ID and omitted another batch item; strict binding rejected both. One exact original request per quotation-rich repair call and matching example IDs corrected that workflow while preserving all failures and raw metadata. Six scope proposals qualified; two remained unresolved. No candidate dates or kinds changed.

Graph quotation coverage is distinct from graph semantics and extraction recall. The native D220 graph has 55 nodes/88 edges with located originals, including two action points missing from the extracted event inventory. Subsequent field checks only validate returned rows; they cannot find omissions. The primary extraction finished normally despite about 90,000 graph-context characters versus 4,603 original-source characters, so response truncation was not observed and salience remains a hypothesis. Add bounded original-only action discovery and independent own-source qualification before field checks; never promote graph nodes or interval endpoints automatically into event truth. A missing direct cycle edge and proposal-to-plan label coarsening remain native graph targets.

Shared proof/history must survive extraction-to-Claim-desk-to-graph/timeline/review handoffs. Readable roles and both dates distinguish alternatives; successful own-source qualification should leave missing-quote queues while scope ambiguity stays visible. Initial failed source checks, manual retries, conditions and notes belong in prompt iteration as untrusted history. A live handoff exposed severe repetition when the same full inventory report was serialized for every row and every sibling, exceeding the 20 MiB session backup limit. Canonical shared attachments with explicit finding references should preserve complete evidence once; increasing limits or dropping failed histories would hide the underlying duplication.

Potential reusable synthetic tasks include own occurrence identity, original cycle/version identity, strict same-paragraph boundary ownership, conditional proposal versus actual performance, semantic predicate scope, negative routing tags and source-only omission discovery. Preserve uncertainty labels and use independently reviewed multilingual held-out data before measuring extraction or acceleration improvements.

The subsequent generic-guidance draw discovered eight interval proposals but only six qualified, with seven of ten action points. Earlier explicit dossier IDs were removed from production instructions before this retained draw; do not treat repeated development runs as held-out performance. One source batch returned an extra cycle alongside the requested candidate. Bounded structural recovery can select a unique exact match across all nine explicitly present raw fields, preserve every extra response, and independently revalidate its own five source premises; duplicates and partial/name/date-only matches remain unresolved. A separately labeled recorded-response replay qualified this candidate without changing the observed artifact.

Lossless shared references reduced actual feedback from 13.67 MB to 268.7 KB (98.03%); the browser’s complete project backup then succeeded. Keep the editable feedback human-readable while retaining exact reports, histories and raw source context for the assistant and export. Reload testing further exposed a local quota/checkpoint hazard: advancing the server marker after a failed browser snapshot write can make an older empty local snapshot overwrite a richer project head. Bind the marker to the exact successfully saved local content; revision numbers alone do not establish that two snapshots match. Immutable project revisions provide recovery, not permission to silently discard history.

Final validation: 2,399 backend tests, 1,636 frontend tests across 120 files, production build and 426 backend/frontend comparisons passed. Live recovery kept the rich revision intact, paused automatic save for divergent cache copies, and retained the browser version when the project copy was selected. These are engineering checks, not complete recall, semantic graph correctness or model-performance certification.

## L166 — Original-only action recall, exact output bindings and retained primary failures (2026-10-07)

**Status:** Adopted application safeguards and retained synthetic development observations. Evidence: [DelayLama event-recall receipt](validation/delaylama-event-source-recall-20261007.json). No LayaGo/GLiNER/MLX training, weights, architecture or speed results changed.

Field checking only returned events cannot find omitted actions. Survey bounded original passages independently of graph events and interval proposals, then qualify each missing occurrence using its own action, performer, calendar and complete qualifications. Shared participants, matching dates or a graph neighbor do not identify that occurrence. Keep ambiguous overlap unresolved. Two distinct records with the same submission sentence or equal-date work cycles must survive separately.

Action role and calendar availability are separate tasks. A positively reported transmission with no stated date remains a reported action with an unknown calendar; a routing-sheet date is not its transmission date. Cancellation is an action while the canceled performance remains unperformed. Completed drafting does not establish conditional execution. Synthetic role/association heads should include these independent labels and hard negatives, with uncertainty preserved rather than merged into a single unknown class.

A nomination hash is not an original record identifier. Source references can reside in exact owning headings, outside a short action quote. Permit only bounded same-record context with its literal own reference; never borrow a neighboring heading or body cross-reference. Preserve action spans. Mechanical serialization recovery may move an own heading from a broad main quote into context only when the full unchanged assertion and three supportive premise bindings independently qualify. Missing/contradictory votes, actor/date changes and wrong records are not persuasion targets.

Model output can alter superficial but frozen calendar serialization, such as dropping a French prefix or replacing a blank with 'not stated'. Provide exact per-request field and premise scaffolds after generic schema placeholders. Models still choose their verdicts and explanations independently; the scaffold is a requested assertion, not evidence. Do not silently normalize a failed response into a favorable match. Record the original failure and any narrowly allowed enum/hash/witness changes.

Five retained development draws on an independently frozen seven-document R740 dossier produced final action inventories of 12, 14, 12, 0 and 16 against 16 expected actions. The fourth run's primary extraction and repair both truncated at 4096 tokens, leaving an empty fallback and skipping the old source-survey gate. The final run preserved nine primary rows and added seven source-qualified drafts, with five failed initial attempts and their bounded repairs retained. All 16 expected occurrences appear; three nonexecution nominations remain separate. These repeated development draws are not held-out model accuracy and cannot establish semantic graph or legal completeness.

A confirmed unreadable primary can receive a distinct original-only source investigation while its failure flag, receipts and uncertainty remain. Unconfirmed transport outcomes, offline mode and stale source/register bindings do not permit this path. A separately labeled replay preserved the fourth draw's two truncated primary calls and performed 32 new action/field/interval calls without resubmitting the primary or replacing its saved result. Fourteen actions qualified; ROUND-B start and the undated transmission remained withheld. Field checks retained 28 supporting and 14 insufficient findings; all five interval proposals remained unresolved. Recovery execution is separate from complete recovery. New invalid/truncated outputs retain full model text alongside previews; historical traces cannot retroactively recover discarded text.

Located graph quotes still leave entity typing and predicates unresolved. Source-record labels typed as organizations, conditional wording included in party names, a contracting link targeting a works scope and an obligation link lacking its exact object need own-source review. Advisory checks should expose these concerns while preserving records and avoiding automatic entity merges or predicate certification. Multilingual hard negatives for reusable heads can target record/party distinctions, conditional spans, own performer versus recipient, directed relations, and dispersed qualifications; use independently reviewed held-out originals before claiming improvements.

Evidence presentation should show the complete owning paragraph around the exact highlight, rather than unrelated document wrappers or a clipped qualification. Distinguish unresolved source nominations from human review: all new drafts remain unaccepted even when zero nomination checks are unresolved. Retain failed histories, exact source versions and structured outputs when handing concerns into prompt iteration; portable exports must survive unavailable local storage, clipboard or downloads.

Final engineering validation: 2508 backend tests, 1709 frontend tests across 124 files, production build and 83 Python/TypeScript comparisons passed. The primary runtime and user data remained protected; 102 owned application jobs were terminal and the temporary stack/browser were closed. Native graph semantics, dispersed applicability, financial-stage interpretation, larger-source coverage and broader extraction quality remain open.

## L167 — Original-bound entity identity and graph-label anchoring (2026-10-07)

**Status:** Adopted application source/version safeguards; retained synthetic development failure and proposed blind-inventory follow-up. Evidence: [entity-review receipt](validation/delaylama-graph-entity-identity-20261007.json). No LayaGo/GLiNER/MLX training, architecture, weights or speed measurements changed.

Seal a graph node's type, roles, provenance, aliases and merge lineage, as well as its label. Operational timestamps and vectors may be excluded; semantic changes must invalidate saved evidence. Bind a declared registered original and its complete owning paragraph. The same literal identifier can be a company in a party register and a record elsewhere. Global name matching, a matching suffix, shared actor/date or nearby company wording cannot settle the selected mention's type.

Preserve identity, kind/type and name boundaries as distinct source premises. Necessary local contradiction guards must inspect complete own qualifications rather than a short captured fragment. Literal If/Si company names must survive; an unsigned proposed role or denied performance does not negate company existence. Record references, work cycles and performed actions remain different labels. Retain malformed replies, rejected quotes and stale alternatives without upgrading sanitized failures into support.

Mechanical source binding is not semantic classification accuracy. In a five-node live development draw, two source-record references and two companies matched independent expectations, but the model interpreted the conditional label “Si Opal Research SA” as a literal company name. All three structured supportive checks came from one response and cited exact full paragraphs. The raw failure remained unaccepted. Three premise votes are not independent model evidence, and suffix/legal-form vocabulary does not determine grammatical boundaries.

The next experiment should read names, types and conditional markers from opaque original passages before showing existing graph names, types, aliases or previous verdicts. Only afterward compare exact spans to the selected node. Keep disagreements and all alternatives. A narrower Opal company nomination cannot authorize changing the type of an unchanged Si Opal label. This blind protocol remains proposed; it has not been evaluated or deployed.

Reusable contrastive/span heads can target these original-local hard negatives: record versus company with identical surface text, literal If/Si legal names versus conditional operators, party existence versus operative role, and source-reference versus performed occurrence. Freeze independent original cases and expected boundaries before training. Evaluate name spans, types, roles and uncertainty separately. Repeated development draws are not held-out accuracy, and successful citation tests are not contract-analysis completeness.

Safe graph corrections require a durable source/node/episode/complete incident-edge capture under the writer lock, explicit human interpretation, idempotent apply and exact unchanged-poststate undo. Initial same-name type-only corrections avoid rename/FTS/alias/merge hazards; they do not reinterpret existing semantic connections, calendars or events. The engine primitive was implemented in an isolated clone and independently tested (82 race-audit entries, including 59 entity cases). It has not been activated or connected to application review in this increment.

Application engineering checks passed: 2580 backend tests, 1777 frontend tests across 125 files, production build and 80 backend/frontend comparisons with zero mismatches. The primary user graph, environment, job history, local engine edits and user artifacts remained protected. Wider graph semantics and extraction correctness remain open.

Portable review history must contain actual completed/failed payloads, not only a compact job index. Live browser testing exposed and fixed that export gap. A collapsed full JSON backup retained all registered originals and complete raw attempts; manual full selection was verified. Native download completion was not verified because the browser download event timed out. Treat clipboard/download failures as recoverability tests, retain selectable fallbacks and avoid reporting an attempted export as a completed export.

Source-unit guards must handle plain Article/Clause/Section headings and their bounded indentation as well as Markdown. Withhold quotes crossing those units, and stop a narrow own paragraph before the next indented unit so it cannot borrow another unit’s type qualifiers. Final mechanical revalidation reproduced the retained five live reports without new model calls; it did not repair the semantic conditional-name failure.

## L168 — Blind name inventory, occurrence association and reviewed type-only writes (2026-10-07)

**Status:** Verified engineering increment with retained model failures; the broader semantic-analysis goal remains active. Final checks are recorded in the [release receipt](validation/delaylama-graph-entity-blind-inventory-20261007.json). This follows L167's graph-label anchoring failure. The [independent first-run audit](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/assets/entity-blind-inventory-eval/first-run-independent-audit.md) preserves the unfavorable run. The [DelayLama release report](https://github.com/UnicoLab/delaylama/blob/main/docs/validation/2026-10-07-graph-entity-blind-inventory.md) is finalized with the live browser evidence, source failures and release limits. No LayaGo weights, architecture, MLX acceleration, training or throughput result changed.

Discover names, kinds and grammatical boundaries from opaque original source slots before supplying any selected graph label, type, aliases, roles or previous verdicts. Then perform a separate current-node association. Source-only discovery and association have different success criteria: a precisely located company nomination does not establish which occurrence a broad graph quotation intended. If the stored quotation contains multiple literal occurrences of its label, or several distinct compatible name spans are nominated, retain the source nominations but withhold the graph comparison. Do not select the first or most favorable occurrence. A narrow conditional party name and its covering conditional span may coexist; they are different assertions and cannot authorize a rename or merge.

Require the name, kind and boundary quotations to cover the same original name occurrence. Condition-marker evidence must cover its exact operator and named span; an arbitrary preceding word, another occurrence or a quotation omitting the operator cannot substitute. Literal If/Si/Falls legal names need their complete own qualifications and must survive operator-looking prefixes. Keep competing overlapping nominations unresolved, including duplicates. Preserve raw invalid fields, unsupported kinds, omitted premises and every failed quotation. Replaying a saved report must derive spans and checks again from its raw claims and current originals; cached positive statuses, supplied offsets or altered review/coverage markers cannot authorize a correction. Three separately structured premises from one model response are not three independent model judgments.

Original context mechanics matter. Treat headings and the first owning body paragraph as one exact discovery chunk, preserving blank separators, Unicode character offsets and all source text. Identical bodies in distinct headed records need their own heading context to remain separately locatable. Accept only outer whitespace trimming in a returned context; retain the complete original chunk and raw response. Cropped sentences, missing headings and rewritten inner whitespace remain unresolved. This is a mechanical paragraph partition, not a claim that adjacent text determines semantic ownership. Larger source budgets, later repeated paragraphs and dispersed qualifications still need explicit coverage and scope review; never silently clip or bridge records.

The independently frozen H512 first run had **0/30 source-qualified outcomes** under the original mechanics. A separate review of its retained raw replies found **26/30 exact nominations** and **22/30 name/kind/boundary interpretations** matching the frozen expectations. These are different success criteria in one small synthetic run: raw interpretation can appear plausible while source-context and association gates fail. Retain the original zero-qualified run and its actual invalid fields after fixes. The counts are not representative accuracy, complete recall, calibration, or evidence of a trained Laya head. Keep held-out labels out of prompts and training; use generic structural regression cases to repair protocol mechanics, then record any new run separately rather than combine favorable replies into a replacement result.

Same-name type correction requires explicit human interpretation after fresh source checks. Seal the complete job, current originals, semantic node preimage, source episode and complete incident graph. The native operation changes only type; names, aliases, merge lineage, temporal properties and existing semantic relations remain untouched. Event/action/source-outline conversions and rename/merge work require separate mechanisms. Revalidate the native receipt against the sealed preimage as well as the before/after delta: coordinated changes to both snapshots must not hide a changed date or provenance. Undo is conditional on the exact unchanged poststate. A historical applied receipt does not verify today's graph state.

Persist an operation's identity before dispatch. A transport/server failure is an unconfirmed outcome and must not trigger blind resubmission. A confirmed atomic invalid/conflict refusal is a rejected operation with its failure retained, while the historical status remains distinct. Inspect the same identity for uncertain outcomes; prepare a new reviewed preview after a known refusal. This distinction improves recovery without claiming an uncertain write succeeded or that a rejected undo proves the current node still has its prior type.

Reusable LayaGo research tasks are original name-span nomination, kind classification, operator-versus-literal-name boundaries, same-occurrence association and calibrated abstention. Training examples should pair same-text companies/records, repeated headed mentions, literal legal-name prefixes, conditional-party spans, unsigned roles and deliberately positive model votes attached to wrong quotations. Evaluate location, interpretation, association and abstention separately on independently sealed multilingual data. The present work supplies integration failure modes and validation patterns, not new model weights or an acceleration benchmark. Final verification passed 2,732 backend/app and independent-control tests, the full 1,866-test frontend suite, 90 final focused interface tests and production build. Browser/backend parity covers 29 source cases, 43 nominations and 29 comparisons with zero mismatches. The owned browser applied and undid a type correction, preserving all other node fields and retaining a complete selectable backup. The primary API was refreshed with its 24 journal snapshots and environment unchanged; the primary engine was not replaced. All owned test services stopped with 112 terminal jobs, and protected user data/source edits remained preserved. The optional atomic type capability requires the maintained engine review patch; broader semantic and coverage work remains open.

## L169 — Source-slot grounding, retained progress and semantic error separation (2026-10-07)

**Status:** Adopted engineering safeguards and measured targeted synthetic evaluation; the wider contract-analysis goal remains active. Evidence: [DelayLama source-slot receipt](validation/delaylama-entity-source-slots-20261007.json). Final application checks passed 2,805 backend/evaluation and 1,955 frontend tests, plus 182 final metadata/integration and 86 wording controls; production build passed with the existing large-bundle warning. Browser/backend parity covers 53 source cases, 70 nominations and 53 comparisons with zero mismatches. The primary API retained all 24 job snapshots and its environment; four owned test services stopped with 117 terminal app jobs and the temporary tab closed. Protected graph rows, source files and user Studio sessions were preserved. No new LayaGo/GLiNER training, weights, architecture, MLX acceleration, calibration or throughput result.

Exact quotation grounding and correct interpretation are different measurements. In the retained I513 retry, 34 raw nominations included 32 mechanically source-located items, while only 26 of 36 frozen name/kind/boundary expectations matched exactly. Of 32 known-kind targets, 24 were correctly interpreted and source-qualified; two of four unknown-kind targets were correctly withheld. Thirty-one targeted names had exact own source spans. These counts belong to eight manually authored English/French/German originals, not representative contract accuracy or a complete entity inventory. The earlier R740 five-node development audit likewise retained one wrong conditional-name interpretation despite three supporting, precisely located quotations. A matching passage and positive model votes do not certify its semantic type, grammar, operative role or graph predicate. Three structured premises from one response are not independent model judgments.

Blind original-only nomination needs an explicit opaque source slot, followed by a separate graph association. Locate every name/kind/boundary quotation inside that same supplied slot, preserve its full original context on the server and retain its absolute Unicode source positions. Identical text in different slots remains separate. A slot can locate repeated wording without choosing which occurrence a broad existing graph quotation meant. Same-name competing referents, missing recorded source mention and uncertain metadata must still withhold graph association; raw graph labels, aliases and prior types are proposals, not source evidence.

Survey bounded canonical windows rather than silently clipping long originals. Revalidate each window's exact source-slot list, request JSON hash, versioned guidance hash, runtime status and ordered raw replies. Known output truncation cannot be replayed as parsed evidence. The retry retained thirteen parsed source-window invocations, thirteen surveyed meaningful slots and five blank separators, with no failed meaningful window. Complete source surveying does not establish complete recall: three record identifiers were still omitted. Blank separators differ from meaningful regions skipped for whole-slot size or call budgets, and the interface should show those reasons separately. Keep all failures, oversize regions and declared omissions visible; located candidates from an independently checked region are reviewable without certifying the rest of the contract.

Progress callbacks are an event stream, not one terminal object per window. The original eight-recovery attempt failed because the evaluation runner saved a running receipt before dispatch, then refused the terminal receipt under the same filename. Only eight running receipts and failed original outputs survived; terminal model bodies were unavailable. Its preserved zero score is a retention-harness failure, not a measured classification result. Append immutable records under window ID plus sequence and status, preserving both running and terminal snapshots. A regression must require all originals to finish, two events per ready window and one per skipped window, terminal raw output retained, prior running records unchanged and progress hashes verified. Merely asserting that some progress exists can pass while every recovery actually failed. The corrected retry is separately retained and must never be described as original first-pass workflow success.

Current interpretation failures remain useful research targets: two honorary titles were included in person names; three record identifiers were omitted; all three conditional parties missed their explicit operator-boundary labels and requested wider conditional spans; one unknown recipient was forced into a record-reference kind; one physical device outside the frozen kind repertoire was forced into concept. The narrow parties were still correctly named organizations, so missing conditional metadata must not be confused with denying company existence or proving authorization. Unsupported ontology is also different from source absence: physical equipment can be explicitly described while the current inventory lacks a suitable kind. Version an expanded ontology in a later experiment instead of changing frozen unknown labels after observing predictions. Additional literal mentions outside targeted gold need separate adjudication, not automatic false-positive scoring or global identity merging.

Potential LayaGo tasks are source-local name-span nomination, entity-kind classification, person-title versus literal legal-name boundaries, grammatical operators versus legal-name prefixes, record ownership, same-occurrence association and calibrated abstention. Construct independent multilingual training data and paired hard negatives; include deliberately positive support votes attached to wrong semantic interpretations, repeated headed bodies and same-label entities with different roles. Measure exact spans, interpretation, quotation qualification, association, false support and selective risk separately. Keep I513 excluded from prompt selection and training; any result-informed later prompt/weight iteration requires a separately declared development evaluation and a new untouched final-test plan. No same-corpus paired baseline establishes that the new counts improve over H512 or R740.

The maintained same-name type-only engine route was subsequently activated on the primary local engine after source-baseline reproduction and compatibility checks. The activation receipt verifies a bit-identical preexisting source baseline plus only the maintained reviewed-entity-type patch, unchanged primary graph table fingerprints and configuration, and an invalid native preview correctly returning HTTP 400. Shadow reads compared 111 collections across 37 scopes with zero mismatches. This updates the deployment state recorded in L168 without erasing its earlier evidence. Binary/route compatibility does not validate entity meaning, existing graph relations or entire contract analysis; no primary graph mutation occurred during this activation.


## L170 — Source fragments, output budgets and contract-field meaning (2026-10-07)

**Status:** Adopted application safeguards and readable review; retained paired mixed results and subsequent development failures. Evidence: [fragment and scope receipt](validation/delaylama-entity-fragments-20261007.json). No new LayaGo weights, training architecture, MLX/GLiNER integration or calibrated performance result.

Separate original-only name discovery from name/kind/boundary interpretation and current graph association. Submit opaque source fragments without existing graph labels, aliases, kinds or previous votes. Bind each occurrence to current original version, complete owning reference and independent Unicode name span. Use bounded overlapping fragments with explicit core ownership; retain complete originals once rather than duplicating an oversized parent for every name. Related passages are retrieval leads, not verified co-reference or operative applicability. Resolve cross-batch conflicts and unknown types without favoring later positive votes.

Freeze whole helper/provider code dependencies as well as the headline guidance and expected annotations. The first same-corpus paired draw across eight English/French/German synthetic originals and 79 targets located 25→70 occurrences and interpreted 19→44 exact name/kind/boundary combinations, but correct qualified known meanings changed 17/76→14/76. Five rich interpretation batches truncated under the unchanged 2048-token limit. Report 76 known/3 unknown denominators, aggregate and ordinary/dense/long strata, raw failures and correction trails together. Extra nominations outside nonexhaustive targeted annotations require adjudication rather than automatic false-positive labels.

A persisted, versioned output-sizing heuristic reduced new rich batches to three names at 2048 output tokens. Replay must reconstruct the recorded policy rather than current model settings; legacy absent policies retain prior grouping. The policy cannot guarantee complete output, even for one name at a tiny limit. In the subsequent declared DEVELOPMENT draw, exact semantic interpretations rose to 67/79 and semantic truncations fell to zero, but correct qualified known meanings fell to 11/76. All three unknowns were withheld. Costs rose to 116 v4 attempts/332,808 reported tokens versus 14 baseline attempts/16,206 tokens. This is improved targeted interpretation with a grounding/cost regression, not established usable accuracy or efficiency.

Most correct but withheld meanings used a provided own fragment identifier in a citation slot restricted to separately registered related context, or omitted an own name quote. Do not silently remap/discard invalid citations or fabricate an enclosing proof. A future versioned context registry can explicitly register supplied own and related originals with authoritative doc/version hashes, exact references, completeness, unique quotations and collision/staleness rejection. It is currently a design, not implementation. Short predicate quotes without demonstrated subject association need a separate unproven premise rather than weakened three-premise gates.

Matching quotations alone cannot establish contract-field meaning. An actual ContractIdentity run promoted an invoice to agreed price and six observed work dates to contractual milestones. Necessary field warnings preserve raw values and reviews while separating same-value agreed consideration from transactions, and applicable schedule provisions from observed execution. Another agreed amount or planned date nearby cannot rescue the wrong field. Wrong-scope identity dates stay readable but are withheld from contractual timeline plotting; event chronology stays separate. One same-original development rerun removed these two error categories, while still omitting a deliverable deadline and overstating summary/applicability. Preserve those remaining errors rather than reporting full contract understanding.

Prompt review notes become evidence for the assistant when a revision is requested; they are not automatic training or activation. A fallback after an actual model call must identify that reason truthfully, preserve complete feedback beyond 400 characters and keep base prompt coverage. Retain exact raw output and runtime even when a required additive refinement is missing. Explicit no-model, failed/unreadable/truncated attempt and legacy unknown states must stay distinct. Proposed changes and recommendations remain unverified until compared on original-bound runs.

For reusable heads, consider independent multilingual hard negatives covering invoice versus consideration, observed versus required/planned date, conditional versus literal legal name, record versus actor, source-local subject association and dispersed agreement applicability. Train only on separately reviewed data; keep this evaluation excluded. Evaluate spans, semantics, proof qualification, selective false support and uncertainty independently. Source binding and passing engineering tests do not prove correct native graph predicates, entity identity, complete context or legal analysis.

The third development draw returned the explicit validation-sheet deadline without restoring the invoice price or observed milestones, but retained a summary binding overclaim. The assistant accepted ten full history entries and returned an unapplied additive revision with complete raw response/runtime retained. Final engineering checks passed 3,007 backend tests, 2,155 frontend tests, build and 62/168/140/42 source/nomination/semantic/comparison parity cases with zero mismatch. Primary user state stayed intact, the absent port 5173 frontend was restored, and 119 owned app jobs were terminal when three temporary services and the test tab closed. These checks do not establish semantic completeness.

Independent audit confirms all 35 quotation slices in their own originals, unchanged earlier experiments, all 200 frozen helper hashes and eight final necessary-pattern controls; the four initial control failures remain retained. Raw LOT-58 scope is still inferred from execution, and signature absence/PD-9 applicability/risk allocation remain unestablished. Five phases per actual run were recorded, but extraction provider bodies and tokens were unavailable and are not reconstructed. Those omissions differ from the separately retained exact prompt-assistant response.

## L171 — Explicit supplied contexts, independent replay and successful-response observability (2026-10-07)

**Status:** Implemented application safeguards and measured development comparison, with substantial evidence and semantic gaps retained. Evidence: [supplied-context and contract-scope receipt](validation/delaylama-qualification-contexts-20261007.json). This supplies reusable design and training ideas; no new LayaGo weights, training, architecture, MLX/GLiNER integration, calibration or throughput result.

Version an explicit supplied-context registry rather than silently remapping a model's quotation identifiers. Register own fragments and related passages with canonical opaque identity, exact document/version hash, Unicode reference and bounded text. Keep selected own passage, another supplied own passage and related retrieval distinct. Reject missing, colliding, ambiguous, stale, foreign and unsupplied references. Three own name-covering name/kind/boundary premises remain necessary; a located ancillary qualification neither rescues a missing premise nor proves co-reference or an operative role. Legacy history must retain its original protocol instead of being reinterpreted by current defaults.

A browser's self-consistent hashes are insufficient if it trusts the supplied registry itself. Independent controls caught acceptance of a forged related context with recomputed hashes and normalization of a present null quotation list to an absent list. Independently reconstruct canonical retrieval and reject malformed protocol fields. Preserve initial failures and subsequent passing controls. Source binding and cross-language parser parity do not certify relation truth, complete retrieval or semantic understanding.

On the already-observed eight-original EN/FR/DE development corpus, semantic v2 obtains 70/79 exact discoveries, 72/79 exact names, 68/79 exact name/kind/boundary interpretations and 21/76 correctly qualified known targets; all three unknowns remain withheld. The unchanged baseline has 25/79 discoveries, 19/79 exact interpretations and 17/76 qualified known targets. The previous separate v4 development draw was 67 interpretations and 11 qualified known targets. These changed-code, changed-guidance stochastic draws establish neither causal registry improvement nor new held-out accuracy. Retain ordinary/dense/long strata and cost together: dense40/40 meanings yield only3 qualified; long6/6 yield only3. Current v4 consumes116 observed client invocations/returns and366,430 reported tokens, versus14 and16,206 baseline. Provider aliases are counted once; client invocations are not wire HTTP attempts, SDK retries remain unobserved and summed active durations do not measure acceleration.

All44 correct known meanings still withheld lack an own kind quote covering their selected name. Forty fail for this alone, three additionally fabricate noncontiguous ellipsis qualifications and one also meets a finite German prefix-registration guard. Do not fabricate enclosing text or weaken proof requirements to improve the reported number. An explicit subject-association extension should separately represent the named subject, predicate statement, source link and unresolved competing referents. Its training/evaluation needs pronouns, multiple actors, negation, other agreements, repeated records and deliberately misleading neighboring affirmations.

Two quote-qualified matched targets still have incorrect conditional boundaries. Add a necessary same-occurrence If/Si/Falls ambiguity warning without stripping words, changing raw proposals or automatically writing graph relations. Preserve registered literal prefix names. Independent preimplementation controls exposed a nonbreaking-space gap before horizontal Unicode-space support was added. Keep the original observed metrics and a later offline counterfactual separate; a current warning does not retroactively make the model response correct. The165 unmatched extras include160 source-qualified interpretations and155 procedural references; nonexhaustive gold requires adjudication, not automatic false-positive scoring.

Successful extraction responses need the same observability as failed responses. Retain exact client content, system/user messages, actual phase-specific requested schemas and hashes, portable content-block/invalid-Unicode encodings, allowlisted usage counters and overlapping phase durations. Do not reconstruct missing earlier raw replies or expose arbitrary provider metadata and secrets. Different legitimate section schemas are distinct phases rather than mixed model configurations; malformed status arrays must not become valid strings through coercion. Legacy execution records remain separately comparable. Raw request/reply traces belong outside ordinary user findings.

Contract-field training needs both semantic and applicability distinctions: actual work versus own agreed scope, a separate proposed package versus overall agreement conditions, signature absence versus explicit unsigned status, an invoice versus agreed consideration, and an observed date versus a local duty or contract schedule. Same-value own provisions can support a candidate; different values and unrelated positive clauses cannot rescue it. Human acceptance cannot erase fresh source gaps. The owned ContractIdentity Run4 removes several previous scope/price/date/binding/risk errors but still wrongly promotes a proposal's condition and states Non signé. Run5 leaves overall conditions empty and preserves the local deadline, but retains an unsupported unsigned assertion in its time reasoning. Both raw failures and exact quotations stay reviewable. These remain same-original development cases, not proof of general contract understanding.

Run4 records36,335 tokens and Run5 records36,686, once per response, with five exact phase/request/schema/reply pairs each. Run4's source copy existed before dispatch but its manifest was recorded afterward; Run5 has a172-file pre-dispatch manifest. Client capture is not raw HTTP bytes, billing or proof of SDK retry policy. A native type-only preview for the own source record leaves all seven graph tables unchanged; semantic review, human decision and graph application remain separate. New controls and measured failures should inform separately reviewed synthetic training generation while this observed evaluation corpus stays excluded from training and final held-out claims.

An independent test using the installed retry wrapper exposed that its default thinking/whitespace normalization precedes capture. Run4/Run5 retain exact post-framework content; missing pre-normalization originals are not reconstructed. New extraction and source-recovery calls explicitly retain the provider return before normalization, while answer parsing still strips thinking through the standard text helper. Metadata records this capture stage. Real-wrapper deterministic tests must accompany mocked-provider tests, because a mocked invoke can conceal destructive normalization and dropped usage.

The actual missing-evidence button initially returned support for the unsupported unsigned reasoning. Its original normalized response and exact quotes remain preserved; its full provider body and usage were unavailable. Current browser replay withholds candidate support without rewriting that historical positive verdict. After narrative signature-gap checks, a repeated real search still returned raw support but became insufficient in field validation; the interface exposes the gap and retains the first search. The second search captures1,636 original provider-return characters, exact requests and2,790 tokens counted once. Plural signature gaps, explicit same-source unsigned assertions and local uncertainty frames need separate positive/negative controls. Repeating a flawed interpretation through a source critic does not make it true; quote verification, semantic inference and usable field support remain separate.

A third source search with explicit signature-gap task guidance improves its explanation to missing signature evidence, but still returns positive whole-finding and field votes for the flawed unsigned reasoning. Validation again withholds it; all three searches remain retained. Its pre-dispatch-sealed copy records1,284 raw characters and2,768 tokens once per response. This is a changed-guidance development example, not causal improvement or a reliable critic. Training hard negatives should preserve accurate partial statements paired with a single unsupported status assertion, so the model cannot certify the whole finding by citing only its correct parts.

Final mechanical checks pass3,456 backend tests,2,667 frontend tests across154 files and production build with the existing bundle advisory. Python/browser replay agrees on215 source cases,361 semantic rows,49 comparisons and151 direct guards. Independent positive Unicode operator controls pass44 backend/41 browser cases while protecting registered names. Known source-search output limits and unreadable Unicode are withheld with portable raw capture and no automatic retry; deterministic tests cover these paths, not new real truncated-model draws. The primary API retains24 journal snapshots and its environment; user graph/source/artifact/Studio preservation checks pass. Three owned services stop with120 terminal jobs, including one retained earlier error. The observed model metrics and current semantic gaps remain unchanged.


### L172 — Fresh named-kind proof, cross-actor controls and native quotation fidelity (2026-10-07)

**Status:** Implemented application evidence recovery and reviewed graph integration. Evidence: [named-kind recovery receipt](validation/delaylama-named-kind-recovery-20261007.json). No new LayaGo/GLiNER weights, training, architecture, MLX, calibration or acceleration measurement.

A new source check can repair a missing named kind quotation without rewriting the original interpretation. Preserve primary name/kind/boundary, all raw replies and the original comparison. Return an independently copied contiguous own quotation covering the same selected physical name occurrence, with unaccepted affirmative/same-subject facets. Require all three original-bound name/kind/boundary premises and recheck competing alternatives. Never manufacture an enclosing quote, concatenate a list or let a pronoun, related document, hypothesis or another actor supply kind proof. The browser must independently replay requests, exact returned bodies, source versions and occurrence references; cached positive projections are not authority.

On eight previously observed EN/FR/DE originals, only this fresh stage runs. Discovery70/79, exact names72/79 and exact semantic annotations68/79 are unchanged. Correct known targets with qualified source proof improve21/76→37/76, all16 additions in the dense register3/40→19/40; long-source qualification remains3/6 and unknown withholding3/3. Fourteen parsed stop returns report24,753 tokens counted once. Eighteen invented ellipsis quotes, two competing French title boundaries and three unsupported finite subject descriptions remain withheld despite positive raw votes. These are development outcomes, not held-out accuracy, improved recall, source-critic reliability, causal training benefit or speed evidence.

Independent controls exposed a false predicate association: an actor hired by a company, employed by one or photographed at its office is not thereby an organization. Direct finite type predicates/appositions and explicit collective lists are necessary source cues, not complete entailment. Synthetic training should pair the same type words with wrong actors, negation, conditional statements, repeated names and unresolved title boundaries. Preserve long list citations without generated ellipses. Keep the observed evaluation originals and gold excluded from training/final held-out claims.

Cross-language integration found proposal IDs diverging on Python1.0 versus browser1. New recovered comparisons use an explicit binary64 identity protocol; unknown, null, absent or coherently downgraded markers cannot borrow a positive legacy fallback. Preserve old protocol IDs and original fixtures; add separate current comparison expectations. Numeric/parsing parity tests should include bool-versus-number, negative zero, unsafe integers, Unicode key ordering, duplicate JSON keys and unrepaired truncated answers.

Full-stack review also found the app trimming native recorded quotation whitespace. Engine source guards correctly rejected both a candidate without an original episode and a source-bound packet missing its trailing newline. The adapter now resolves agreeing native raw source text exactly once, verifies the same trimmed wording and offsets, and expands only outer whitespace while retaining complete original context. Inner normalization, repeated/conflicting records and source crossings remain rejected. A real owned-graph UI preview/apply/undo preserves all979 characters, changes only node_type, restores the exact before-node on undo and leaves original episode content/incident edges unchanged. No engine guard is weakened.

Final application checks pass3,583 backend tests,2,764 frontend tests/158 files and build. Python/browser replay agrees on64 recovery/comparison/seal/parser/subject controls; independent real-driver scripted tests pass39. Selected-mention UI hides unrelated targets, highlights the exact name, separately shows type wording from the same contiguous quote and exposes full raw attempts/progress/export. All classifications remain unaccepted interpretations; roles, retrieval closure, complete extraction and perfect contract analysis remain open.


### L173 — Fresh proof replay reuse and truthful service presentation (2026-10-07)

**Status:** Application performance and UX improvement with unchanged source interpretations. Evidence: [replay and health receipt](validation/delaylama-replay-reuse-20261007.json). No new model call, weights, training, architecture, MLX, calibration, memory or inference-speed measurement.

A proof validator can reuse freshly reconstructed primary state and source quotation indexes inside one synchronous invocation without trusting persisted positives or keeping a cross-request cache. The old40-target review redundantly replayed its primary thirty times and derived fifteen plans. Revalidate once for this invocation, compare every batch to the current canonical plan and keep all source/request/raw-response/subject/conflict gates. Construct new state at every public entry. Same-object source/context/runtime mutations, altered return bodies, poisoned cached projections and interleaved independent sources must still be rejected or rederived.

An independently frozen65-case set matches exactly in two candidate passes, preserving IDs, raw evidence and withheld states. Seventeen new isolation/staleness controls and thirty-nine driver controls pass. Independent full-public replay totals8.684/8.884s→1.353/1.407s; native40-case2.412/2.441s→0.141/0.139s. A separate warmed interleaved native-envelope benchmark gives2.070s→0.131s. These measure local source-proof processing, not LLM inference, GPU acceleration, memory, billing or general throughput. Do not substitute a private microfunction for the public workflow when making timing claims. Four live current HTTP checks285–696ms have no paired old HTTP baseline.

Retain benchmark harness mistakes rather than concealing them. The first integrity digest failed on a deliberate lone-surrogate input and was corrected only in harness encoding. A file initially named baseline-public-replay.json actually pinned the candidate helper; preserve it as a candidate and explicitly exclude it from baseline claims. The valid old baseline uses the immutable8005273 helper and baseline-v1-public-replay.json. Source expectations, production guidance, raw draws and extraction scores stay unchanged.

Reduce progress noise without hiding failed or unscanned coverage: one collapsed completed-search summary, separate expanded phase counts, actual live phase, and visible unknown/pending/failure concerns. Service configuration is not availability or successful generation. A remote model should not be marked healthy solely because the app API responds; an unrelated local catalog failure should not imply that default remote model failed. Match explicit endpoints, retain uncertain endpoint matching, and do not assume every other profile is unused. Health should show service/configuration diagnostics rather than an unrelated stale contract result; returning to analysis must preserve that result.

Final application checks pass3,608 backend tests,2,791 frontend tests/161 files, build and64 Python/browser parity checks. Earlier source-qualified targets37/76, semantic annotations68/79 and retained evidence gaps remain unchanged. No new LayaGo training or extraction-quality claim accompanies this validation-processing improvement.


### L174 — Exact model-selected citation spans and measured prompt iteration (2026-10-07)

**Status:** Application source-proof and review development; no new LayaGo weights, training, MLX, contrastive architecture, calibration or inference-speed measurement. Evidence: [citation development receipt](validation/delaylama-exact-kind-spans-20261007.json).

A model may understand a late name in a long typed list yet fail to transcribe its complete supporting passage. Allow a fresh explicit reference with literal beginning and ending selectors, then resolve one contiguous span of the actual supplied original covering the separately sealed full-name occurrence. Keep raw selectors, source offsets and the resolved text distinct from a model-copied quotation. Never infer an enclosure around a historical nameless quote, invent ellipses, choose the nearest pair or borrow another actor/document. Literal ordered nonoverlapping selector pairs must be unique at that occurrence. Stop ambiguity enumeration at two candidates to avoid quadratic memory use on repetitive inputs.

Offering selectors alone failed: the first actual draw returned39 copied quotations and no boundary references;27 invented ellipsis quotes stayed withheld. Correct known source-qualified targets were21/76→28/76, weaker than earlier copied-quotation development37/76. Seal that failed result before revising guidance. The explicit next guidance prioritizes boundary references for long passages and illustrates their actual output shape. Its independent observed-source draw returns39 boundary references and qualifies21/76→55/76, dense register3/40→37/40. All34 additions are dense-list source proof; discovery70/79, names72/79, semantic annotations68/79 and unknown withholding3/3 remain unchanged. These are separate previously observed development draws, not causal or held-out accuracy evidence. Five short French/German references overlap and remain unresolved; do not reinterpret them as copied quotations to improve a score.

Both draws retain14 parsed/stop client returns and36 immutable journal events. Usage counted once is24,971 and26,923 tokens; SDK retries and billing remain unobserved. Keep exact captured guidance1 through a strict version registry when guidance2 becomes the producer default. Missing/null/unknown or coherently downgraded version/capture/request/source metadata must not borrow a favorable legacy fallback. Generated proof views remain untrusted; current browser and server independently replay actual raw histories.

A source-resolved span still must pass all three own named-premise checks, affirmative same-subject interpretation, existing finite kind/boundary guards and competing-alternative review. Synthetic data should cover direct predicates, typed collectives, wrong actors, negation, hypotheses, repeated names, malformed/overlapping selectors, long lists, Unicode and unknown kinds. Keep measured originals and gold excluded from training/final held-out claims. Source selection is not entailment, identity acceptance, contractual appointment or legal effect.

Separate full-stack review exercises a source-derived citation for the40th list name and native type-only preview/apply/undo. It restores the exact before-node, leaves the original episode and77 raw job events unchanged, and retains all prior unsupported wording. Selected-view duplicates from child scans should collapse to the current physical occurrence, while raw nominations remain exportable. A short name-centred preview, separate type-prefix context and complete contiguous citation are clearer than displaying the entire preceding list by default.

Actual replay exposed a producer input-budget bug: after a flush, a reused fragment retained a pre-flush zero added-byte value, so related-source retrieval exceeded the declared source-text budget. Four inherited batches contain28,428 or28,451 bytes despite a24,000-byte cap. Preserve their raw histories and old backend outputs, but disclose browser withholding rather than reporting zero total parity differences. New live producers explicitly capture fresh-group-bytes-v1, recount UTF-8 own/related text, seal per-batch totals and validate marker/body/runtime at replay. The cap concerns source text rather than full JSON or tokens. Forward only an explicitly captured marker through compatibility helpers; absent historical markers must not inherit the current default.

A marked-driver integration test also exposed passing legacy context into the projection after already validating compact context. Use that validated compact context consistently; retain the initial failing trace and corrected receipt. A separate final live draw binds all accounting fields and the same physical late-name source proof. Final application checks pass3,777 backend tests,2,915 frontend tests/167 files and build. Old65 public outputs and64 browser controls remain exact. New actual41-source/41-comparison parity has zero unexpected mismatches plus two explicitly documented historical budget refusals. Preserve that distinction from a zero-refusal or perfect-analysis claim.


### L175 — Lossless evidence history and measured source windows before model optimization (2026-10-07)

**Status:** Application evidence-flow development and audits; no new weights, training, MLX, architecture, calibration, inference speed or extraction-quality gain. Evidence: [dossier/input receipt](validation/delaylama-evidence-dossiers-20261007.json).

A prompt-improvement history must retain raw failed attempts separately from accepted interpretations, document versions and human reviews. A “checked-only” recursive summary can accidentally discard keys named raw_response/raw_model_response even when nested inside the actual model data. Keep the ordinary summary but add a typed, sealed dossier of exact persisted source-search snapshots. Query all matching records in one read-only transaction instead of the most recent 200/first20 index. Explicitly disclose unavailable, unreadable and bounded memory/history coverage; exact hashes do not establish semantic truth or live authentication. A missing stored snapshot may be exported as a labelled DTO fallback but cannot authorize a source baseline.

Raw history can multiply in size when a selected record is copied again into history, and each parsed record duplicates its exact serialized text. An actual development import exceeded20MiB. Store raw once, reference the selected job by ID+rawSHA, independently reconstruct portable parsed views, and avoid another copy of the same history in the App handoff. V2 retains both actual raw captures unchanged and reduces the compact dossier to6,920,899bytes; the complete saved successful Studio session is7,154,344bytes. Keep oldv1 readable and failures retained; do not raise limits or truncate evidence to hide the issue. Browser cache quota and successfully saved project revisions are separate outcomes. Model-context limits are also separate from complete exported history; a real assistant consuming this large dossier was not evaluated.

Extraction-input accounting must measure the actual constructed source windows, including identify, sections, repairs and known specialist builders. Preserve exact Unicode scalar boundaries and UTF8 sizes; source text may itself contain DOCUMENT:/GRAPHCONTEXT labels, so do not parse headings to guess scope. Track original/provided hashes, exact request ranges, citation-reference identity and transformed retrieval/draft channels. Negative/zero budgets retain actual Python slicing. Unknown custom builders and no-model fallback remain unrecorded. Two independent tests caught mutable metadata aliases and a foreign document part claiming coverage for another declared original; deepcopy snapshots and validate constructor source/slice agreement.

The actual framework boundary matters: a new str subclass passed direct scripted calls but failed dataclass state serialization because deepcopy required constructor keyword metadata. Preserve that real failed attempt without reconstructing lost provider replies. Use plain exact strings in recorded/dispatched messages, support independent temporary-object copies, test dataclass→JSON transport, and rerun the actual stack as a separate attempt. Five live section calls then succeeded, each supplying the same250/979sourcecharacters; five repetitions do not mean1,250uniquecharacters. Allphase/system/request/raw/source identities reproduce; reportedusage32,113tokens countedonce is notbilling orwire-attempt measurement.

Even that successful run completes a name one character beyond its supplied prefix and sometimes describes global document absence or register genre as contract identity. Full-original quote matching does not establish what was supplied, correct contract scope or complete analysis. Input-window warnings make this auditable; they do not solve semantic interpretation. Failed citation recovery must retain precise reasons and force all affected proposals noncurrent, not borrow a previous positive view. Complete input/history, source location, same-referent evidence, operative role and instrument/version applicability remain distinct.

The preserved observed55/76development result still has21source-proof gaps: seven name/heading boundaries, three operator boundaries, one ontology policy, three invalidshortselectors, one registeredprefixgrammar gap, three bad additionalqualification citations and three dispersedreferent/type proofs. Do not delete failed qualifications or merge identical strings to improve counts. A future model/contrastive supplement should nominate explicit mention/name/kind/boundary/qualification witnesses and separately source-bound referent bridges, then re-enter deterministic checks. Training examples need other actors, titles, literal prefixes, negation, hypothetical names, repeated occurrences, displaced definitions, unavailable versions and nonoperative registers. This increment does not establish new heldout accuracy, contract correctness or graph closure.

Application validation:3,838backendtests;2,993frontendtests/177files;build;67independentmechanicschecks;actualAPI/browser/Studioaudit;old64kind-readerparityzero mismatches. Primarygraph, protectedenginefiles,24jobs,artifactsandtwoStudiohistories preserved; parallel tracks were backend evidence, UI and independent validation.


### L176 — Contract-role witnesses and reviewed comparisons before extraction quality claims (2026-10-07)

**Status:** Application guard, evidence/review UX and actual end-to-end development tests. No new LayaGo weights, training, MLX, architecture, calibration, inference speed or extraction-quality gain. Evidence: [application receipt](validation/delaylama-contract-party-scope-20261007.json).

A supplier-register name, organization type, performed work or signature label does not establish that entity as a contracting party. Preserve the LLM's raw named role and own quotations, then independently locate the same named actor, requested role and agreement cue in an own source statement. Separate reported/listed names from unaccepted contract-linked candidates. An extracted contract number is itself a proposal; quote location and a UI-only original hash namespace do not establish a registered source ID, operative appointment, signature, authority or applicability. Do not let cached acceptance or a sanitized remaining quotation wash a failed role attempt into source support.

Own actor boundaries matter even in simple finite guards. A name beginning If/Si/No needs an own explicit naming witness before treating its prefix as literal. A negative quantifier immediately preceding the same recognized naming cue still negates the proposed role; an unrelated actor's negative clause must not veto an own positive clause. Likewise Contract/Agreement/Contrat inside a company name is not an external agreement cue. Retain failing pre-fix traces and controls, preserve failed extra citations, and never turn the guard's supported constructions into a universal semantic detector. Valid tables, captions, signature blocks, indirect formulations and dispersed referent links can remain conservatively withheld.

Global absence assertions need actual input accounting. Reproduce request/source identities and exact source windows, not a cached coverage percentage or headings appearing inside the original. A global absence claim from partial or unrecorded input needs review; an excerpt-local statement is different. A quoted source absence statement must itself be located in a validated supplied window. Complete input still does not establish absence, contractual attribution or model attention. Warnings attached to summary/value or parent rationale need exact-path inheritance in the UI and must survive human acceptance or later source rechecks. Historical raw analyses are not silently rewritten by a newer annotation protocol.

Prompt fill/quote/latency heuristics are diagnostics. In a preserved read-only legacy reproduction, a quoted wrong Contractor row scored 1.0, correctly empty unsupported parties scored 0.5214, and an unrelated dictionary received schema-valid 1.0/composite 0.7. A dictionary is not formal schema validation; no quotes are not 100% verified; retained cached quote flags are not current-original verification. Recount actual citations, preserve numeric zero versus unavailable values, reject booleans/strings/nonfinite or negative metrics at both the scorer and HTTP boundary, and clear stale measurements when a fresh diagnostic is unavailable.

A prompt recommendation should retain two exact runs, their original/language/prompts/outputs, predeclared criteria and explained human outcome. Store outputs once as exact captured JSON and replay the bundle rather than trusting a cached pass. Bind the consumer to the requested agent and indexed version as well as the bundle's internal agent; otherwise a coherently reviewed record misfiled under another agent can nominate a foreign prompt. Updated failed judgments must supersede earlier passes within the same context, and different originals, criteria or baselines have no universal winner. Viewing a candidate is read-only; explicit saving/activation is separate. An unchanged reviewed candidate can retain its comparison, while an edited draft must not inherit it or baseline stars. A retained client comparison remains human-reported, not provider-authenticated, semantically certified or a causal prompt-effect estimate.

The actual new A17 development fixture was sealed before model dispatch and distinguishes explicit Client/Contractor roles, register membership, a hypothetical supplier appointment, a separate B18 agreement and a missing filed signed copy. Two same-source full-input runs recovered the explicit A17 roles, price EUR 6400 and deadline 17 November 2097 without borrowing B18's EUR 9900 or asserting unsigned status. Each has five parsed section calls supplying the complete 946-character/948-byte original. Independently checked reported usage counted once is 24,778 baseline and 25,144 candidate tokens; this is not billing or wire-attempt accounting. The baseline summary overstates unestablished other-party roles as negative; the candidate improves that wording but invents an operative suspensive condition with nonfulfilled status from hypothetical appointment/approval. Record the comparison as inconclusive, retain both failures, save inactive version v019 and refuse automatic nomination; leave active v018 unchanged.

A narrow subsequent source review flags that hypothetical approval's applicability without rewriting its type, status, rationale or quote. A separate final-code live repeat still returns the hypothetical condition, now with a different status, while its applicability warning is visible. That change in model wording is variability, not proof the deterministic guard improved extraction. Existing agreed-price wording coverage also conservatively warns on the valid fixture phrase 'requires ... for EUR 6400'. The observed earlier 55/76 source-proof result and its 21 remaining referent/qualification gaps stay unchanged; no held-out contract-quality score is claimed.

For future trainable nominators, keep role classification, agreement/version binding, identity, qualification and source location as separate targets. Synthetic contrastive examples should vary named actors and distractors, literal operator prefixes, negation, quoted naming cues, agreement words inside entity names, register versus appointment, conditional role versus operative agreement condition, missing recorded copy versus unsigned status, unknown versus negative roles, and partial supplied windows. Train span/provenance proposals and abstention before trusting a classifier's label or graph proximity. Preserve source-first review loops and test whether a trained supplement improves independently evaluated outcomes rather than structural fill.

Application release validation: 4,251 backend tests, 3,108 frontend tests in 183 files, production build, 82 Python/browser role-reader parity cases with zero mismatches, independently sealed mechanics controls and actual API/browser/Studio audits. Frozen variants, initial failures and exact traces are retained. Primary graph, 14 protected engine sources, 24 jobs, 32 artifacts, application environment and both Studio histories were preserved. Parallel tracks covered backend extraction/recovery, results UX and independent validation.

A real HTTP negative-input probe also caught a framework response-formatting failure: Pydantic correctly rejected NaN/Infinity, but the default FastAPI error details retained the nonfinite input and Starlette could not serialize it, producing 500. Keep invalid inputs rejected and make only their diagnostic representation JSON-safe; the main:app application hook now responds 422. Independent live requests verified 18 invalid cases and three valid zeros, preserving inactive v019/default bytes; 25 focused handler tests pass. Both 379-file model-run manifests remain unchanged, and the later 380-file HTTP-only variant called no new models. Preserve validation-path failures as well as model failures.


### L177 — Separate agreement prerequisites, proposals and fulfillment proof (2026-10-07)

**Status:** Application source-review, producer guidance, pipeline transport and native UI development. No new LayaGo weights, training, MLX, contrastive architecture, calibration or inference-speed claim. Evidence: [condition/pipeline receipt](validation/delaylama-condition-source-review-20261007.json).

A possible appointment, proposed work package, own agreement prerequisite and another agreement's condition need separate output and review. Keep proposed arrangements outside the operative-condition list without deleting earlier raw rows. Distinguish contract effectiveness from agreed-work commencement/performance with an explicitly proposed condition-kind field. An authorization can also establish a work prerequisite; quoting it only in authorizations can leave the structured condition analysis incomplete. Preserve the exact original criterion and identifier rather than a translated or shortened description.

Rule, instrument and status witnesses are different learning targets. A broad exact quotation can be a retained source container from which bounded original statements are derived; each needs its actual owning paragraph and occurrence. Cropped quotations cannot erase their own draft or requirement denial. A shared approval word cannot bridge another actor, condition ID or instrument. Nonappointment does not establish a missing approval record, and a missing record does not establish unfulfillment. A located status statement can remain unbound to its condition. Graph neighbors and reference navigation are useful leads, not automatic entailment or legal effect.

Scripted controls caught 22 initially withheld source-positive cases, five unsafe negated/wrong-actor candidates, capitalized Condition keyword gaps and a foreign source-ID mismatch between Python and browser. All36 pre-frozen EN/FR controls now match finite expected classifications; 27 shared and12 during-iteration qualification packets also match across implementations. The final French draft-effectivity case remains unresolved; the earlier two-gap trail and narrative correction are retained. These are mechanical development outcomes, not model accuracy or held-out semantic coverage.

An extraction section must return only requested top-level keys. If none are present, importing the full parsed dictionary leaks foreign fields. Preserve raw replies and missing/ignored-key traces; complete key presence is neither formal schema validation nor semantic completeness. A critic's partial corrected dictionary cannot replace a full identity and erase its parties or price. Retain original, critic and unaccepted proposal separately; complete container coverage still does not prove source meaning.

Test the installed pipeline engine, ingestor and real class-agent input adapters. Its resolver accepts paths, not inline or-coalescing or literal template strings. Ingestor results wrap their document payload; agent/transform results are direct dictionaries. The first actual critic pipeline reported success with zero model calls and no document. Explicit source preparation, readable-output validation, direct identity mappings and correct collection fix that transport failure. Metadata-only general dictionaries must not count as populated identity. A condition evaluator exposes fewer builtins than a transform: moving the readiness predicate into a transform preserves evaluator scope and restores healthy critic dispatch. Keep failures visible rather than calling a no-call execution successful analysis.

Actual development evaluation retained three initial draws, two post-observation repeats and a six-call repaired pipeline. Initial A17 and D31 separate hypothetical appointments usefully; F77 quotes R2 in authorizations but omits its condition row. All initial proposals incorrectly report approval not_recorded merely from nonappointment. Generic subsequent guidance changes the two repeated proposal statuses to unknown and retains F77 R2 as an agreed-work prerequisite. Yet its direct criterion is not literal and lacks the status quotation, so current review stays unresolved. The pipeline quotes both rule and record gap and qualifies the own prerequisite; its status remains an unbound witness under finite grammar. The critic returns no correction. A17's proposal instrument still uses a heading. Do not turn these partial improvements into a causal or universal quality claim.

All31 observed calls preserve exact requests, replies, schema and source channels; reported usage counted once totals180,354 tokens, not cost or wire-attempt accounting. The real critic consumes the full normalized source plus6055characters of rendered prior-analysis context derived from227203rawcharacters. This is bounded transformed context, not the entire raw identity/history; a separate fake single-shot prefix test exercises another branch. Model coverage checklists, gaps[] and complete source input do not certify exhaustive review or absence. Existing55/76 entity source proof and21 gaps remain unchanged.

Native rendering needs an actual browser. Eight result children existed but facts had zero layout height. A persistent coverage/body wrapper alone still failed. Removing inline-size containment from this results root and retaining normal flow/media responsiveness renders the saved analysis on first mount. Preserve failed screenshots and source variants; DOM tests cannot prove native heights. Clearly identify the displayed run's original when the editor contains another document, and avoid repeated generic warnings above a dedicated evidence pane. Keep context around a short exact highlighted quote, full owning text and raw export available.

Future trainable nominators should learn proposal versus own prerequisite, effectiveness versus performance, literal criterion/ID, instrument, actor/object, negation, as-of status and explicit referent bridges as separate targets. Synthetic contrastive examples need near-identical other-condition/status distractors, register/nonappointment versus approval-record gaps, cropped qualifications, dispersed clauses and unavailable originals. Re-enter deterministic source checks and independently evaluate held-out meaning before trusting a learned label. This increment supplies learnings and auditable failure cases, not new trained heads or automatic graph correction.

Release validation:4457backend tests,3159frontend tests,build,75Python/browser parity packets,95initial independent mechanics checks and11additional real ingestor/class-agent transport checks. Counts overlap broader suites. Original model failures, pipeline failures and native layout failures remain retained. Primary graph, protected engine sources,24jobs,32artifacts, environment and both Studio histories were preserved. Work proceeded in parallel across backend, UI and independent audit.


### L178 — Original criterion witnesses, complete critic facts and faithful pipeline views (2026-10-07)

**Status:** Versioned application source/context and review-channel development; observed runs remain separately auditable. Evidence: [prerequisite/context receipt](validation/delaylama-prerequisite-context-20261007.json). No new LayaGo weights, training, MLX, contrastive architecture, calibration or speed result.

A reported condition description and an exact original prerequisite are different objects. Keep raw paraphrases and even contradictory descriptions unchanged while representing a separately source-located criterion, literal ID, instrument, actor/object and owning occurrence. A positive source-derived rule or status must not be displayed as support for an unsupported raw assertion. Preserve this precision-policy distinction and unresolved reported support rather than calling all control labels successful semantic classification.

A quotation can omit an explicit connector in its own paragraph. Reconstruct that paragraph from the current original and bound occurrence, then locate the uniquely named clause target. Preserve seed, paragraph, connector and target with original SHA and Unicode spans. A referenced status still needs the same typed subject, literal ID and instrument, not nearby type words or graph proximity. Duplicate headings, reused definitions, other actors, other instruments, draft/denied requirements and invalid extra citations remain unresolved. Record absence is not unfulfillment. A server-derived reference quotation must never be relabelled an earlier generator citation or proof the model read it. Keep v1 replay defaults and opt into v2 explicitly for new source paths.

A source critic must receive the complete supplied business analysis separately from graph rendering. Preserve unknown nested business fields, source-review overlays and quotations; exclude only declared top-level execution fields with hashes and counts. Retain the exact original outside the critic channel and label the facts packet unaccepted. Duplicate keys, nonfinite/unreadable input and unavailable analysis must not collapse into a partial packet called complete. Record exact packet identities in actual requests. Complete transmission is not exhaustive attention, correct review or the full raw execution history.

The initial two q9 pipelines plus an exact manual C92 retry retained 18 calls and 151,966 reported tokens counted once. F77 gained a source-bound record-gap witness under v2; C92's nominal modifier was initially outside grammar, although its direct quoted status remained a source candidate. The critic's packets contained 32,542/29,609/31,493 code points of complete business facts, independently from 831/719/719 original-source code points. All proposed critic corrections remained unaccepted and unresolved; an unsupported F77 pending authorization/applicant proposal stays visible. These are observed development examples, not causal or held-out quality evidence.

Post-observation refinements are separate variants. Adding literal access/security modifiers made the next six-call C92 criterion locatable but still withheld its status because Written began the sentence. That draw reported 50,656 tokens, bringing the first 24 calls to 202,622. A further six-call variant permits only the first lexical character's capitalization; actor, object remainder and ID case stay exact. It also withholds No/Aucune grant/refusal false positives while retaining explicit record gaps. The final live C92 draw located its criterion and explicit met report. The final six-call draw reports 52,300 tokens and a 35,014-code-point critic packet. Across the five workflows, 30 client invocations report 254,922 tokens once (239,128 input/15,794 output). Retain all variants; these are neither SDK wire-attempt/billing totals nor causal model-learning evidence.

A successful pipeline transport envelope is not 48 independent business findings. Render the original identity once, partial/different critic proposals separately as unaccepted, and each generic analysis report under its own namespace. Keep ingestion/steps/runtime out of ordinary facts while preserving the exact envelope in export/history. A duplicate corrected identity must not multiply findings or silently replace the original. DOM existence and transport success do not establish correct native layout, source scope or semantics; retain actual browser failures and fixes separately.

Independent development checks include 38 exact prerequisite/reference controls and six precision probes with two explicitly preserved policy discrepancies, plus sealed critic/transport and first-character/negative-subject controls. All 27 v1 compatibility fixtures remain unchanged. The first source-freeze checkpoint passed 112 independent mechanics tests and 4,648 backend tests. Counts overlap broader suites and do not measure model accuracy. The latest critic still proposes an unsupported administrative classification; its corrected condition has no own quotation or instrument and remains unresolved. Future trainable nominators should learn source criterion, reported-claim relation, instrument, subject, connector, status and uncertainty as separate reviewed targets; this observed evaluation and its labels must remain excluded from held-out claims and automatic training gold.


A later local fallback audit found that missing criterion IDs could bypass the typed negative-subject guard. The final v2 refinement retains No/Aucun/Aucune met/unmet observations as unbound, and rejects a nonempty declared ID when its status statement's named IDs are different or ambiguous. Record gaps remain distinct; no negative status is inverted. An anonymous positive status needs one unique physical own rule containing the exact raw criterion, all original claims currently located, and a direct own-subject/instrument cited statement. It invents no ID, expands no paraphrase and borrows no reference. The source witness has a separate exact_reported_criterion_status origin; the raw criterion and all failed evidence remain unchanged.

The first narrow fix and initial independent 20-pass/6-discrepancy result remain preserved. All 26 final missing-ID expectations match; the latest independent mechanics suite passes 138 tests and the final backend suite passes 4,720. Earlier 112/4,648 counts are retained checkpoints. All ten prior public collections from the five workflows remain exactly equal to the variant-3 cold replay; this is no new model run. The observed totals remain 30 client calls and 254,922 once-counted reported tokens. V1 default replay and prior entries remain unchanged, and the two precision-policy discrepancies are still disclosed.


A final browser probe demonstrated reintroduction when the exact raw criterion itself began with No/Aucune: anonymous supplementation could restore a candidate removed earlier. Reapplying the identical guard after supplementation closes that path; preserve the 4ffa checkpoint and final bfa41df8 source separately. All 50 preceding fixture outputs and v1 replay remain exact. The latest 394 focused tests and 170 browser parity packets pass; the full frontend suite passes 3,360 tests and build checks. The independent eight-control census retains six literal expectation matches and two conservative record-gap gaps, without any unsafe met/unmet affirmation. Those two semantic expectations remain explicit known limitations with separate safety checks; they are not relabelled successful classification. Final CI reports 4,736 backend passes with two strict known-gap xfails and one existing warning; independent mechanics reports 146 passes with those two xfails. Separate unmarked safety checks ensure unsafe affirmation cannot be ignored as an expected failure. These overlapping counts are not 146 exact source-label matches; earlier 138/4,720 results remain checkpoints. None of these mechanics refinements changes the 30 observed model calls or their token accounting.
