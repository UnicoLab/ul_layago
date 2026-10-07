# JEFF-inspired training tools: engineering validation

2026-10-07–08. This increment adds data audits, protected training-only preparation and native calibration readiness. It does not train a new head, download a dataset, change the deployed classifier or establish an accuracy improvement.

## Findings that change the next experiment

The existing 615-row classifier pilot has copied source passages across its internal development roles. The final Python audit inspects525 rows and skips90 locked rows, finding14 exact source-copy pairs and14 rendered-copy pairs, plus2 bounded near-copy findings. The native guard counts16 affected confidence rows and blocks before loading a nonexistent base/cache. These count different objects. Existing confidence cells are60/62 rows from30/31 stories and remain below the actual176 starting count. Preserve the old model, data and measurements; build new family-isolated development/calibration data.

The earlier explanatory135 count was incorrect: two score candidates share delta/2, giving176 at the default risk configuration. Existing artifact metadata already recorded176. Readiness now reports the actual policy, correlated-story counts and token/domain limitations without loading a model. Positive-error planning is bounded by available support, including an excessive configured error budget; a reported lower bound is explicitly marked.

## Verification scope

Normal and purego Go tests, race tests, vet, cgo-free builds and package Python checks cover the modified package. Separately authored planted controls cover source/story leakage, a length shortcut, hard/soft choice permutation, ordinal preservation, protected replay, byte-prefix/no-overwrite behavior and protected-access accounting. The final independent31 controls comprise21 adapted original controls and10 access controls. The committed `release-receipt.json` pins final sources and verification logs.

The fresh logged CLI smoke starts with37 existing engineering-fixture rows, preserves29 evaluation rows and appends6 training choice variants. Its43-row output passes native Go dataset validation. Those shuffled tasks require explicit option catalogs at serving; the current fixed-catalog DelayLama client must be adapted before any later deployment of such an artifact. The retained input is an engineering fixture, not legal gold.

`preparation-smoke.json` and `prepared-go-validation.json` are the earlier preparation checkpoint: it inspected locked source/schema without an access record. They are preserved and are not retroactively called logged. `preparation-smoke-final.json`, `prepared-go-validation-final.json` and `protected-access-final.jsonl` are the fresh logged run. The final CLI records durable intent before reading input content and hashes before logical validation. Its default envelope detection still reads/deserializes outer JSON; this is logical access accountability, not byte secrecy. In-memory logging cannot establish callers' previous reads. No protected rows were fitted, calibrated or evaluated by a model.

The first package check during concurrent test adaptation failed with stale access-log expectations; its raw log is retained. Final tests use explicit logs for unsafe replay too, so they test isolation rather than passing only because authorization is absent. The independent auditor also retains initial reporting gaps, its draft import failure and its separately explained expectation correction. Mechanical test success does not prove semantic labels or independent sampling.

See [workflow](../../data-preflight.md), [source-specific data](../../research/contract-training-data.md), [prospective curriculum](../../research/jeff-training-experiments.md) and [independent retained controls](https://github.com/UnicoLab/delaylama/tree/main/docs/validation/assets/jeff-data-tool-independent).
