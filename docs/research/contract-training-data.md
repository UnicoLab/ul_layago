# Data that can improve LayaGo contract decisions

Reviewed 2026-10-07. These are sources and experiment designs, not imported data or demonstrated quality gains. Keep source revision, rights/attribution, original contract identity and annotation provenance in a dataset ledger. The existing classifier challenge is observed and remains evaluation-only; new promotion requires new independently authored/adjudicated contracts.

| Source | Useful task | Adaptation boundary |
|---|---|---|
| [ContractNLI](https://stanfordnlp.github.io/contract-nli/) | Support/contradiction/not-mentioned and evidence selection | 607 NDAs and 17 fixed hypotheses, with document evidence spans; CC BY 4.0. Its document-level label does not automatically apply to a short excerpt missing the decisive exception or supporting span. |
| [CUAD](https://www.atticusprojectai.org/cuad/) | Clause presence and source-span candidates | 510 commercial contracts, 41 annotated clause categories. Use the original spans and [labeling guidance/licence](https://www.atticusprojectai.org/legal/). A missing extraction annotation is not an explicit contradiction. |
| [ACORD](https://www.atticusprojectai.org/acord/) | Clause relevance/ranking before source/LLM review | Expert 1–5 relevance grades over 126,000+ query/clause pairs and 114 queries; author lists CC BY 4.0. Relevance is not truth or evidentiary support. A separate ordered-score or ranking head fits those labels better. |
| [LEDGAR through LexGLUE](https://huggingface.co/datasets/coastalcph/lex_glue) | Broad legal clause vocabulary and topic adaptation | Check the pinned original dataset's terms and contract/source identity. Its legal category catalog needs an explicit mapping or separate specialist head; it does not directly label our eight topics or seven agreement types. |
| New owned EN/FR/DE contracts and adjudicated review cases | Actual DelayLama genre, role, prerequisite and support tasks | Required for the application-specific domain and language gap. Keep unresolved scope cases pending, and split by original contract across all excerpts/translations. A review note or LLM agreement alone is not gold. |

Jeff also used public legal sources, but its data-source document is version-specific and records removal of MAUD after contract-level overlap was discovered. Do not treat public benchmark familiarity as unseen-contract generalization. [Jeff source ledger at the inspected revision](https://github.com/firelex/jeff/blob/3720e7c66d73b79a6e90189f3d5b0d6ecf03ddc0/docs/data-sources.md). No dataset above was downloaded or trained in this increment.

## Recommended next data construction

Start with separate, stable task definitions and adjudicated positive/negative/unknown policies. Build source-first fact specifications, then diversify their language and presentation. Introduce each of these close contrasts within a family, assigning the family to a split before any rendering:

- Same genre words in this agreement, a quoted other instrument, a company name, excluded scope and a hypothetical proposal. Unknown examples should look like plausible business text without a sentence conveniently announcing “type not stated.”
- A single dominant clause topic, two equally developed subjects, and a bare reference to an unavailable clause. Vary length, tables, headings and formatting across every class.
- Evidence explicitly supporting the full claim, a direct scoped contradiction, and missing evidence. Vary actor, amount/unit, effective amendment, instrument, negation, condition and exception one at a time; retain affirmative support to avoid a model that merely rejects everything.
- Dispersed supporting spans and exceptions from the same original, plus unresolved/foreign references. Include every span needed for annotation. A clipped excerpt should be unscorable or insufficient for that excerpt, not inherit the full-document truth label.

LLMs can paraphrase a constructed specification or nominate hard cases. Record producer/model/prompt provenance and recheck preserved facts, scope and source spans. Humans adjudicate real examples and disputes. Soft targets can encode documented ambiguity over admissible choices; they should not be copied from the candidate's confidence. Class balancing is a training ablation, while calibration/locked evaluation should preserve the declared deployment distribution.

## Training sequence to measure

1. Audit the unaugmented family-split data. Compare majority, source-surface shortcuts and the full pretrained reader before fitting another head.
2. Fit a specialist marker head over the existing frozen multilingual encoder. Compare canonical versus train-only option permutations on identical untouched evaluation input. Re-encoding shuffled prompts is necessary; cached states from the original order cannot be relabelled as though the input were unchanged.
3. Add fresh hard contrasts and optional adjudicated replay from earlier **training** families. Keep calibration and test originals protected. Try a small replay fraction as a measured ablation; a published recipe's fraction is not a proven optimum for LayaGo.
4. Restore the best development-NLL scorer, refit temperature and the act head on their allowed fold, and certify on independent confidence data. Distinguish raw row count from independent source-family support. Do not relax the target risk solely to obtain coverage.
5. Evaluate production-exported float32 and independently recalibrated int8 artifacts, recording unknown false-specific types, false support, support recall, macro-F1/NLL/ECE/Brier, language/family strata, coverage and cold/warm latency. Keep LLM fallback until domain quality and a trusted routing policy are qualified.

A stronger cross-token head or actual encoder LoRA is a later controlled comparison, after data quality and the simple baseline. Current cached-head MLX accelerates the scorer; it does not implement full GPU encoder training or Jeff's Qwen LoRA. The package now supplies data preflight, protected preparation and native count readiness so these comparisons can be reproducible. [Prospective acceptance plan](jeff-training-experiments.md).
