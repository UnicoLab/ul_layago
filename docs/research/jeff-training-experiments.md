# Contract training experiments informed by JEFF

2026-10-07. This is a prospective experiment plan and an independent audit of data-tool mechanics. It does not report new training or model-quality results. JEFF source was inspected at immutable commit `3720e7c66d73b79a6e90189f3d5b0d6ecf03ddc0`; the relevant transferable practices are family-based splitting, shortcut diagnostics, choice/soft-target permutation and protected replay. See [architecture review](jeff-architecture-review.md), [JEFF adapter-kit guide](https://github.com/firelex/jeff/blob/3720e7c66d73b79a6e90189f3d5b0d6ecf03ddc0/examples/adapter-kit/README.md), and [training contract](../training.md).

The separately authored earlier classifier challenge is already observed. Its sources, labels, errors and policy uncertainties must remain evaluation-only: they cannot become training examples, prompt-tuning targets, replay or calibration data. Improvements suggested by that study need newly authored stories and a new locked test. The existing evidence head was evaluated separately; its results are not a newly trained evidence adapter.

## Curriculum and source labels

Keep the first two tasks stable: one exclusive primary-clause topic, allowing `mixed_or_unclear`, and explicit agreement type, allowing `unknown` and explicitly combined `mixed`. Add harder story families rather than quietly changing their meanings.

| Family | Necessary contrasts and annotation policy |
|---|---|
| Explicit genre | A statement describing this instrument versus an invoice, register, proposed arrangement, excluded genre or another agreement. A company name containing “Lease” or “Contract” is not a genre statement. |
| Primary topic | Comparable passage length and formatting across topics. Two equally developed subjects are mixed; a bare reference to another clause does not establish that clause's subject in the supplied passage. |
| Own party or role | An explicitly appointed actor versus a listed supplier, witness, performed-by actor, hypothetical appointment or a denied role. Preserve subject, instrument and whether the source expresses a proposal. |
| Prerequisite and status | An effectiveness condition, agreed work prerequisite, proposed arrangement and another instrument's condition. Keep the existence of a rule distinct from fulfillment; no record of approval is not refusal or proof of nonfulfillment. |
| Evidence support | Correct subject/object/ID/instrument, wrong subject, wrong instrument, negation, hypothetical language and missing evidence. Annotation must identify which literal statement supports a claim; nearby related text alone is insufficient. |
| Dispersed evidence | Explicit same-original clause references and bounded source excerpts, ambiguous repeated IDs, unresolved pronouns and other-original references. Supply every excerpt used to adjudicate; missing context should produce insufficient evidence rather than an invented bridge. |
| Language and presentation | EN/FR/DE natural authoring, translations tied to the same family, tables/headings/prose, punctuation and long passages. Reserve a separately authored language stratum if testing transfer. |

For each source/claim, store the literal source, source identity, independent story/family identity, task definition and option keys, expected label, cited evidence span, and adjudication rationale. Two reviewers should independently adjudicate scope-sensitive labels; unresolved local-versus-global claims stay pending or are excluded under a policy declared before scoring. Soft targets should reflect recorded adjudicated alternatives, not probabilities copied from the candidate being trained. Preserve original producer provenance separately from reviewer metadata.

Synthetic generation can create close contrasts and then diversify wording, but a generator's answer key is not independent truth. Do not let one template, length, language, generation round, party suffix or option position determine a class. Keep explicit positive examples alongside unknown and negative controls so conservative withholding is not mistaken for useful recall.

## Split, augmentation and replay

Assign entire families before any augmentation. A family includes an original contract story, translations, excerpts, paraphrases and semantic counterfactuals even when their labels differ. Exact source fingerprints must ignore changed evidence IDs, questions and labels; a shared fixed question or option description alone is not a source leak. Near-copy diagnostics are bounded and cannot replace family provenance or human review.

Use train only for fitting; development selection only for choosing a checkpoint; separate calibration and confidence folds for temperatures and selective thresholds; locked independent families for the final challenge. Calibration and confidence rows drawn from the same story are correlated. Hold out domains/languages by family rather than merely hiding a few rows from already seen contracts. A source copied into two splits blocks preparation even when its labels were changed.

`prepare_training_data.py` adds only train-choice variants. It remaps the selected hard key and every soft probability by option key; ordered score levels remain unchanged. Shuffled examples require explicit per-example options at inference. Replay requires a source/label-bound adjudication receipt and explicit ownership, uses only train rows, refuses protected source/story/input overlaps, and selects whole families within its row budget. A structurally valid receipt does not establish a human actually reviewed the data. Semantic translations with unrecorded provenance can still evade exact hashes.

The audit tool's default locked mode parses JSON envelopes, then skips locked logical inputs and labels. This is not byte-level secrecy and cannot certify unseen locked source isolation. Any deliberate source-level inspection of protected rows must have a durable access record and remain outside fitting. A surface probe evaluated on `dev_conf` consumes information from that fold; subsequent decisions based on it require a newly independent confidence set.

## Predeclared next comparison

Freeze a new challenge and all task definitions before inspecting its outputs. Proposed minimum: 24 independent contract families per language for EN/FR/DE, with all intended strata represented; family counts, row counts and pending annotations must be reported separately. Select the checkpoint on development data only. Compare majority/structural shortcuts, the complete pretrained checkpoint, the current exported head, and the proposed head on identical source/task inputs. A separate matched LLM comparison is optional; it must use source-only requests and independent call capture.

For an experimental head to advance from evaluation to shadow review, declare these gates before scoring:

- Zero invalid task/schema/option mappings, source-family split violations or unreported clipping; no source or label from a protected family in replay. Float32 serving must reproduce the exported head within its documented numeric tolerance, and changed source/adapter selections must invalidate caches.
- At least a five percentage-point macro-F1 improvement over the current task head, with the lower bound of a family-cluster bootstrap interval above zero. Report per-language/stratum confusion, unknown/mixed recall and the result even if the gate fails.
- Unknown/mixed recall at least 90% in each represented language, with a one-sided 95% upper bound on false specific/support decisions below 5% for the declared independent safety population. Report denominator and effective family sampling; correlated paraphrases cannot be counted as independent evidence for this bound.
- For selective operation, risk at most 5% on an independently calibrated confidence population and coverage at least 20% on the predeclared challenge. Report coverage-risk curves, NLL, Brier and ECE, including escalation-all outcomes. Thresholds cannot be adjusted using the locked challenge.
- Repeat unique-input and identical-input runs with queueing, tokenization, encoder, scoring, provenance capture and LLM fallback separated. Publish p50/p95, peak memory and cancellation outcomes; a cached encoder-state hit is not general inference acceleration and cached-head training is not encoder fine-tuning.

These are prospective decision gates, not achieved claims or authority to auto-accept contract facts. Source quotation location, graph association, instrument applicability and human acceptance remain independent of classifier confidence. Even a passing head should first operate in shadow/review mode; a classifier label cannot drop source evidence or promote a contract role, fulfillment status or graph correction.

## Calibration count correction

The current native readiness path allocates the family error budget over two score families: `delta_score = delta_family / 2`. At alpha 0.05, family delta 0.10, zero starting errors and minimum 20, its minimum is `ceil(3 * log(0.05) / log(0.95)) = 176`, not the single-budget figure 135. Existing artifact role reports already record 176. This corrects earlier explanatory documentation; it does not retrospectively change an artifact's measurements. Count readiness is not threshold certification, and 176 correlated rows are not 176 independently sampled stories.

## Independent data-tool mechanics receipt

Fresh planted controls were sealed at 2026-10-07 21:44:01.598367Z before this auditor read the tool implementations. The tools were already being authored, so this is not a claim of sealing before all implementation. No prior classifier-challenge source was reused and no model, encoder or training process was invoked.

The final independent suite exercises exact source copies despite changed questions/IDs/labels, shared fixed descriptions, story splits, duplicate row IDs, malformed logical locked rows, a poison mapping that forbids locked payload reads, a deliberately perfect length shortcut, hard/soft choice permutation, ordinal preservation, protected replay, stale/foreign adjudication and real CLI byte-prefix/no-overwrite behavior. The replay-positive test adds a documented distinct train anchor solely to provide a nonzero replay budget; the frozen protected-only baseline otherwise has a zero replay allowance. Invalid replay inputs are rejected rather than silently filtered.

Initial reports blocked leaks correctly, but lacked a separate conflicting-label diagnostic and called the shortcut case `checks_passed_with_limits`. Those observations remain retained. The final audit reports conflicting hard keys only for identical task, complete rendered input and option catalog; different questions are not automatically conflicting labels. Surface/metadata warnings now require review. These changes improve diagnostic transparency; they do not establish semantic data quality. See the independent receipt and tests in DelayLama's `docs/validation/assets/jeff-data-tool-independent/`.

### Protected preparation access addendum

A subsequent audit found that preparation's first version read, hashed and logically validated protected locked rows without recording that access. The earlier 21-test checkpoint is preserved. Preparation now refuses locked rows by default before logical validation/source indexing. `--protected-access-log` explicitly authorizes source-protection processing and writes/fsyncs an intent before opening input content, then file hashes before logical validation. In-memory `prepare(..., protected_access_log=...)` logs the supplied canonical rows before logical validation; it cannot establish when or how the caller previously read them. Logs retain no source text or gold labels, and input/output aliases or unrelated existing files are refused.

Ten new independent controls and the original 21 adapted controls pass on this increment. Positive protected replay now requires the explicit log; invalid replay tests also supply it so they continue testing source isolation rather than passing solely because authorization is absent. Initial failures and the original frozen expectations remain retained. One initial auditor expectation was too strict: an explicitly supplied nonexistent log path may safely create a new durable log; absence of authorization means `None`, rather than filesystem nonexistence. This correction is recorded separately and is not an implementation defect. No locked rows were trained, calibrated or used for model evaluation in these tests.

Default CLI envelope detection still materializes file bytes and decodes outer JSON; the refusal does not provide byte-level confidentiality. Protected-source hashing and logical validation require the recorded opt-in. Explicit logging is access accountability, not authorization to train on those rows or proof of semantic label quality.
