// Package train trains and serves Laya typed-decision heads in pure Go
// (CGO_ENABLED=0, standard library only) over a frozen multilingual encoder
// trunk (jhu-clsp/mmBERT-base, or any ModernBERT checkpoint the model
// package loads), reading and writing the converged model contracts of
// docs/architecture/intelligence-contracts.md ("Model engine and data"):
// hive.laya-dataset/v1 in, hive.laya-prompt/v1 states, hive.laya-artifact/v1
// out. The Python trainer (tools/laya-train) reads and writes the same
// contracts, and Hive serves either trainer's artifacts through
// laya.LoadArtifact and routermodel.Load.
//
// The pipeline, deterministic for a fixed seed:
//
//  1. Dataset (LoadDataset): every line of every file is validated as
//     dataset.py parse_example validates it, in the same order and with no
//     coercion (a yes_no label is a JSON boolean; a choice or score label is
//     an option index; label_dist is a soft label on train only, with the
//     label among its most probable options). The split invariants (one split
//     per story, held-out domains and languages absent from train and dev)
//     are checked over every instance, and the data digest is sha256 over
//     the contract name and the sorted (file name, sha256) pairs.
//  2. Preparation (Prepare): one instance per head. A sealed locked_test
//     shapes nothing, not even the task definitions; unsealing needs an
//     access log. Tasks take the trainer's instructions; every example is
//     rendered with hive.laya-prompt/v1 (laya.RenderPromptState) and laid out
//     with Laya's build_sequence. dev is divided by story into dev_cal and
//     dev_conf (DevHalf). A split-invariant violation or a rendered train
//     duplicate in an evaluated split is refused unless allowed, and then
//     recorded.
//  3. Features (BuildCache): the frozen trunk runs once per distinct
//     scorable sequence; the final hidden state at position 0 and at each
//     option's [MASK] marker go to an append-only store of sharded
//     safetensors outside the repository. The store is keyed by the trunk's
//     weights, config and tokenizer digests, depth, sequence limits, kernel
//     and stored precision, and each sequence by its token ids and markers,
//     so a larger nested subsample or another split encodes only what the
//     store lacks. Shards are atomic, an interrupted build resumes, batches
//     run in parallel and a shard's bytes do not depend on the worker count.
//     A Q input may be laid out over a lexical catalog shortlist
//     (CatalogShortlist) that never puts back a gold option it dropped.
//  4. Training (Train): Laya's decision head with zero head transformer
//     layers, with hand-written gradients (softmax cross-entropy against the
//     label or label_dist, which over graded items is the listwise softmax
//     ranking loss; an expected-level regression term on score tasks;
//     logistic loss for yes/no), AdamW with float64 master weights, seeded
//     mini-batches and early stopping on the dev_cal NLL; yes_no examples
//     may be class-balanced per task. Every dev fit may be restricted to
//     chosen languages (Config.FitLangs) while train keeps every language.
//     The act head is a
//     class-balanced correctness predictor fitted on dev_cal. Temperatures
//     per (task, option-count bucket), task and "*" are fitted on dev_cal by
//     NLL; act thresholds are certified per task on dev_conf by
//     fixed-sequence Learn-then-Test over the 1e-4 grid, with delta split
//     between the two candidate scores. Every split is evaluated per
//     language, task and language × task; held-out domains and languages
//     are their own splits. The int8 head is recalibrated and recertified
//     on its own outputs.
//  5. Artifact (Export): one hive.laya-artifact/v1 directory per numeric
//     path (float32/, int8/), each with model.safetensors (Laya's tensor
//     names; int8 linears as I8 with per-row F32 "_scale"), manifest.json
//     and parity_golden.json. LoadArtifact and Predictor serve it with the
//     trainer's own float64 head and reproduce its logits bit for bit on the
//     feature kernel; laya.LoadArtifact reproduces them within the golden's
//     tolerance (TestConvergedLoadersServeGoArtifacts).
//
// # Outputs are nominations
//
// A probability from a head trained or calibrated on synthetic data is not
// evidence and carries no numeric authority. Every artifact is of the
// synthetic_engineering evidence tier, and a consumer must refuse primary use
// while it is.
package train
