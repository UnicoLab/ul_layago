#!/usr/bin/env python3
"""Optional cross-check of the numpy reference against the real Python stack.

The numpy reference (layago_reference.py) is the committed parity oracle and
needs nothing but numpy. This script checks that oracle itself against the
libraries it re-implements, when they happen to be installed locally:

* Hugging Face `tokenizers` (Rust): ids and byte offsets of every
  tokenizer_golden.json case, for both synthetic tokenizer layouts;
* `transformers` ModernBertModel (SDPA, float32, CPU): every hidden state of
  encoder_golden.safetensors;
* the `laya` package (0.3.5, pass its checkout with --laya-src): the full
  Agent on checkpoint/, comparing token sequences, markers, raw marker
  logits, act probabilities and decoded answers with laya_golden.json.

It downloads nothing and writes a JSON report to --report. Example:

    PYTHONDONTWRITEBYTECODE=1 /path/to/venv/bin/python \\
        internal/layago/testdata/parity/crosscheck_torch.py \\
        --laya-src /path/to/laya --report /tmp/crosscheck.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import layago_reference as ref  # noqa: E402


def check_tokenizers(report: dict) -> None:
    import tokenizers
    from tokenizers import Tokenizer

    golden = json.loads((HERE / "tokenizer_golden.json").read_text())
    files = {
        "synthetic": HERE / "checkpoint" / "tokenizer" / "tokenizer.json",
        "gemma_style": HERE / "tokenizer_gemma_style.json",
    }
    result = {"library": "tokenizers %s" % tokenizers.__version__}
    for name, path in files.items():
        tok = Tokenizer.from_file(str(path))
        mismatches = []
        for case in golden[name]:
            text = case["text"]
            enc = tok.encode(text, add_special_tokens=False)
            b = [len(text[:i].encode()) for i in range(len(text) + 1)]
            offsets = [[b[s], b[e]] for s, e in enc.offsets]
            if enc.ids != case["ids"] or offsets != case["offsets"]:
                mismatches.append({"text": text, "hf_ids": enc.ids, "ref_ids": case["ids"],
                                   "hf_offsets": offsets, "ref_offsets": case["offsets"]})
        result[name] = {"cases": len(golden[name]), "exact": len(golden[name]) - len(mismatches),
                        "mismatches": mismatches}
    report["tokenizers"] = result


def check_encoder(report: dict) -> None:
    import torch
    import transformers
    from transformers import ModernBertConfig, ModernBertModel

    cfg_json = json.loads((HERE / "checkpoint" / "encoder" / "config.json").read_text())
    config = ModernBertConfig(**{k: v for k, v in cfg_json.items() if k != "layer_types"})
    config._attn_implementation = "sdpa"
    model = ModernBertModel(config).eval()
    weights = ref.read_safetensors(HERE / "checkpoint" / "model.safetensors")
    state = {k[len("encoder."):]: torch.tensor(v, dtype=torch.float32)
             for k, v in weights.items() if k.startswith("encoder.")}
    model.load_state_dict(state, strict=True)
    golden = ref.read_safetensors(HERE / "encoder_golden.safetensors")
    worst_layer, worst_final = 0.0, 0.0
    with torch.no_grad():
        for c, ids in enumerate(ref.ENCODER_CASES):
            out = model(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                        output_hidden_states=True)
            for k, hs in enumerate(out.hidden_states):
                diff = np.abs(hs[0].numpy() - golden["case%d.hidden%d" % (c, k)]).max()
                worst_layer = max(worst_layer, float(diff))
            diff = np.abs(out.last_hidden_state[0].numpy() - golden["case%d.final" % c]).max()
            worst_final = max(worst_final, float(diff))
    report["encoder"] = {
        "library": "torch %s, transformers %s (ModernBertModel, sdpa, float32, cpu)" % (
            torch.__version__, transformers.__version__),
        "cases": len(ref.ENCODER_CASES),
        "max_abs_diff_any_hidden_state": worst_layer,
        "max_abs_diff_last_hidden_state": worst_final,
    }


def laya_question(q: dict) -> dict:
    out = {"type": q["type"], "instructions": q["instructions"]}
    if q["type"] == "choice":
        out["criteria"] = {o["key"]: o.get("description") for o in q["options"]}
    elif q["type"] == "score":
        out["criteria"] = list(q["levels"])
    elif q.get("false") or q.get("true"):
        out["criteria"] = {k: q[k] for k in ("false", "true") if q.get(k)}
    return out


def check_laya(report: dict, laya_src: Path) -> None:
    sys.path.insert(0, str(laya_src))
    import torch

    import laya
    from laya.common import QTYPES, build_sequence, collate_items

    agent = laya.load(str(HERE / "checkpoint"), device="cpu")
    golden = json.loads((HERE / "laya_golden.json").read_text())
    questions = {q["id"]: laya_question(q) for q in golden["questions"]}
    worst = {"logit": 0.0, "prob": 0.0, "act_probability": 0.0}
    sequence_mismatches, decision_mismatches = [], []
    for case in golden["cases"]:
        items = []
        for ans in case["answers"]:
            q = agent._to_internal(questions[ans["id"]])
            ids, markers = build_sequence(agent.tok, case["state"], q, agent.cfg["max_len"], agent.cfg["head_max_len"])
            if ids != ans["ids"] or markers != ans["markers"]:
                sequence_mismatches.append({"state": case["state"][:40], "question": ans["id"]})
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
        batch = collate_items([items], agent.tok.pad_token_id)
        with torch.no_grad():
            logits, act = agent.model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"],
                                      batch["marker_mask"], batch["qtype"])
        act = torch.softmax(act.float(), -1).numpy()
        response = agent.predict(case["state"], questions)
        for r, ans in enumerate(case["answers"]):
            k = len(ans["markers"])
            worst["logit"] = max(worst["logit"], float(np.abs(logits[r, :k].numpy() - ans["logits"]).max()))
            worst["act_probability"] = max(worst["act_probability"], abs(float(act[r, 0]) - ans["act_probability"]))
            got = response["answers"][ans["id"]]
            probs = list(got["probabilities"].values()) if "probabilities" in got else [1 - got["noul"], got["noul"]]
            worst["prob"] = max(worst["prob"], float(np.abs(np.array(probs) - np.array(ans["probs"])).max()))
            for field in ("choice",):
                if field in ans and got.get(field) != ans[field]:
                    decision_mismatches.append({"question": ans["id"], "laya": got.get(field), "ref": ans[field]})
    report["laya"] = {
        "library": "laya %s (Agent and DecisionModel from source), torch %s, cpu float32" % (
            laya.__version__, torch.__version__),
        "cases": len(golden["cases"]),
        "questions": len(golden["questions"]),
        "sequence_mismatches": sequence_mismatches,
        "choice_mismatches": decision_mismatches,
        "max_abs_diff_raw_logit": worst["logit"],
        "max_abs_diff_act_probability": worst["act_probability"],
        "max_abs_diff_probability_after_laya_rounding": worst["prob"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--laya-src", type=Path, help="checkout of github.com/NandhaKishorM/laya at 573e5b62")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report: dict = {"reference": "layago_reference.py goldens"}
    check_tokenizers(report)
    check_encoder(report)
    if args.laya_src:
        check_laya(report, args.laya_src)
    args.report.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=1, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
