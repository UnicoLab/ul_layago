#!/usr/bin/env python3
"""Record PyTorch ModernBertModel hidden states for the real mmBERT-base weights.

This is the test oracle behind TestRealMMBERTBaseParity
(internal/layago/model). It loads the same model.safetensors, config.json and
tokenizer.json the Go encoder reads (jhu-clsp/mmBERT-base, MIT, converted to
safetensors outside the repository), tokenizes EN/FR/DE/JA/AR inputs and one
long mixed-language Laya-layout prompt with Hugging Face tokenizers, and runs
transformers' ModernBertModel on CPU in float32, one sequence per forward
(no padding), with output_hidden_states.

It writes two files:

- a compact JSON golden (committed): token ids, and per token the L2 norm of
  every hidden state (embedding output, each layer's output, the final
  normed state) plus 16 sampled dimensions and the row sum of the final
  state;
- optionally (--full) every hidden state in full as a safetensors file,
  which stays outside the repository and which the Go test compares element
  by element when LAYAGO_MMBERT_FULL_GOLDEN points at it.

Nothing is downloaded (local_files_only):

    /opt/miniconda3/bin/python3 internal/layago/testdata/parity/mmbert_base_golden.py \\
        --model /path/to/mmbert-base \\
        --out internal/layago/model/testdata/mmbert_base_golden.json \\
        --full /tmp/mmbert_base_full.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

# transformers otherwise imports TensorFlow/Keras when it is installed, which
# pulls in pandas and pyarrow; only PyTorch is needed here.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

CASES = [
    ("en", "Why did weekly net revenue drop in the north region compared with last week?"),
    ("fr", "Pourquoi le chiffre d'affaires net hebdomadaire a-t-il baissé dans la région nord ?"),
    ("de", "Warum ist der wöchentliche Nettoumsatz in der Region Nord im Vergleich zur Vorwoche gesunken?"),
    ("ja", "北部地域の週次純売上が先週と比べて下がったのはなぜですか？"),
    ("ar", "لماذا انخفضت الإيرادات الصافية الأسبوعية في المنطقة الشمالية مقارنة بالأسبوع الماضي؟"),
]

# A long Laya-layout prompt (> 2 x 64 tokens) so the sliding-window layers
# (keys within |i-j| <= 64) differ from global attention.
LONG = (
    "choice question: Which operation does the question ask for? "
    "<mask> lookup: read the current value of a metric "
    "<mask> compare: compare a metric across periods or segments "
    "<mask> explain: explain why a metric changed "
    "<mask> forecast: project a metric forward "
    "<eos> "
    '{"question": "Pourquoi le chiffre d\'affaires hebdomadaire a-t-il baissé dans la région nord ?", '
    '"context": "Weekly net revenue (EUR) for the north region fell 12.5% week over week; '
    "Der Umsatz im Norden sank; 北部の売上は減少しました; انخفضت الإيرادات في الشمال; "
    'orders were flat while the average basket value decreased.", '
    '"dashboard": "retail-ops", "period": "2026-W38", "previous_period": "2026-W37"}'
)

SAMPLE_DIMS = list(range(0, 768, 48))


def r7(x: float) -> float:
    """Round to 7 significant digits (float32 carries about 7.2)."""
    return float("%.7g" % x)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--full", type=Path)
    parser.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    args = parser.parse_args()

    import numpy as np
    import tokenizers
    import torch
    import transformers
    from tokenizers import Tokenizer
    from safetensors.torch import load_file
    from transformers import ModernBertConfig, ModernBertModel

    torch.manual_seed(0)
    tok = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    # The converted file carries no {"format": "pt"} metadata, which
    # from_pretrained insists on, so build the model from config.json and
    # load the "model." tensors strictly (the MLM head is not used).
    config = ModernBertConfig.from_json_file(str(args.model / "config.json"))
    config._attn_implementation = args.attn
    model = ModernBertModel(config).to(torch.float32)
    tensors = load_file(str(args.model / "model.safetensors"))
    state = {k[len("model."):]: v for k, v in tensors.items() if k.startswith("model.")}
    model.load_state_dict(state, strict=True)
    model.eval()

    cases = []
    full = {}
    for idx, (lang, text) in enumerate(CASES + [("mixed-long", LONG)]):
        ids = tok.encode(text, add_special_tokens=True).ids
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                        output_hidden_states=True)
        states = [h[0].numpy().astype(np.float32) for h in out.hidden_states]
        final = out.last_hidden_state[0].numpy().astype(np.float32)
        cases.append({
            "lang": lang,
            "text": text,
            "ids": ids,
            # layer_norms[k][t]: L2 norm of hidden state k (0 = embeddings) at token t
            "layer_norms": [[r7(np.linalg.norm(row.astype(np.float64))) for row in s] for s in states],
            "final_norms": [r7(np.linalg.norm(row.astype(np.float64))) for row in final],
            "final_sums": [r7(row.astype(np.float64).sum()) for row in final],
            "final_dims": [[r7(row[d]) for d in SAMPLE_DIMS] for row in final],
        })
        for k, s in enumerate(states):
            full["case%d.layer%02d" % (idx, k)] = s
        full["case%d.final" % idx] = final

    manifest = {}
    mpath = args.model / "MANIFEST.json"
    if mpath.exists():
        manifest = json.loads(mpath.read_text())
    golden = {
        "model_id": manifest.get("model_id", ""),
        "revision": manifest.get("revision", ""),
        "weights_sha256": sha256(args.model / "model.safetensors"),
        "config_sha256": sha256(args.model / "config.json"),
        "tokenizer_sha256": sha256(args.model / "tokenizer.json"),
        "producer": "transformers %s ModernBertModel (torch %s, tokenizers %s), CPU float32, attn_implementation=%s, "
                    "one sequence per forward" % (transformers.__version__, torch.__version__,
                                                  tokenizers.__version__, args.attn),
        "sample_dims": SAMPLE_DIMS,
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(golden, ensure_ascii=False, separators=(",", ":")) + "\n")
    print("wrote %s: %d cases, %d tokens" % (args.out, len(cases), sum(len(c["ids"]) for c in cases)))
    if args.full:
        from safetensors.numpy import save_file

        save_file(full, str(args.full), metadata={"producer": golden["producer"],
                                                   "weights_sha256": golden["weights_sha256"]})
        print("wrote %s: %d tensors" % (args.full, len(full)))


if __name__ == "__main__":
    main()
