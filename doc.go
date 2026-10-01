// Package layago exposes a reusable, cgo-free Laya predictor over local checkpoints
// or verified trained artifacts. Predictions are typed nominations; a model score
// does not establish documentary evidence. The public facade detects input clipping,
// preserves pinned task instructions, and serializes concurrent requests safely.
//
// Subpackages laya, model, tensor, tokenizer and safetensors provide the numerical
// runtime. train fits/calibrates lightweight decision heads over frozen encoders;
// evalreport builds evaluation receipts. JSON-lines inference and offline training
// are available through cmd/layago and cmd/laya-train. Standard library only.
package layago
