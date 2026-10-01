#!/usr/bin/env python3
"""Record PyTorch Laya outputs on the real multilingual checkpoint.

Runs the laya 0.3.5 Agent (from a source checkout) on CPU in float32 over
Hive's frozen judgment fixture with the exact questions and state rendering
of scripts/evaluation/evaluate_laya.py, and writes the token sequences,
marker positions, raw marker logits, act probabilities and decoded answers
that TestRealCheckpointParity compares the Go port against. The checkpoint
stays rejected for Hive judgment; this is offline replication evidence only.

transformers 4.x reads rope thetas from global_rope_theta/local_rope_theta
and silently defaults the local one to 10000, while the checkpoint's
transformers 5 config declares both in rope_parameters (160000). The script
therefore runs a patched copy of encoder/config.json that states the same
values in the 4.x keys; weights and tokenizer are used as they are.
Nothing is downloaded:

    PYTHONDONTWRITEBYTECODE=1 /path/to/venv/bin/python \\
        internal/layago/testdata/parity/laya_checkpoint_golden.py \\
        --checkpoint /path/to/laya/multilingual --laya-src /path/to/laya \\
        --work /tmp/laya-patched --out internal/layago/laya/testdata/laya_multilingual_golden.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def harness():
    source = (ROOT / "scripts" / "evaluation" / "evaluate_laya.py").read_text()
    namespace: dict = {"json": json}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "QUESTIONS" for t in node.targets):
            exec(compile(ast.Module([node], []), "evaluate_laya.py", "exec"), namespace)
        if isinstance(node, ast.FunctionDef) and node.name in ("state_for", "ece"):
            exec(compile(ast.Module([node], []), "evaluate_laya.py", "exec"), namespace)
    return namespace


def patched_checkpoint(src: Path, work: Path) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    (work / "encoder").mkdir(exist_ok=True)
    (work / "tokenizer").mkdir(exist_ok=True)
    cfg = json.loads((src / "encoder" / "config.json").read_text())
    rope = cfg.get("rope_parameters", {})
    cfg["global_rope_theta"] = rope["full_attention"]["rope_theta"]
    cfg["local_rope_theta"] = rope["sliding_attention"]["rope_theta"]
    (work / "encoder" / "config.json").write_text(json.dumps(cfg, indent=1))
    for rel in ("model.safetensors", "rl_agent_config.json"):
        link = work / rel
        if not link.exists():
            os.symlink((src / rel).resolve(), link)
    for rel in ("tokenizer.json", "tokenizer_config.json"):
        target = work / "tokenizer" / rel
        # laya rewrites tokenizer_config.json in place, so copy it.
        target.write_bytes((src / "tokenizer" / rel).read_bytes())
    return work


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--laya-src", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.laya_src))
    import numpy as np
    import torch
    import transformers

    import laya
    from laya.common import QTYPES, build_sequence, collate_items, serialize_state

    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    ns = harness()
    ckpt = patched_checkpoint(args.checkpoint, args.work)
    agent = laya.load(str(ckpt), device="cpu")
    fixture = json.loads((ROOT / "testdata" / "adaptive_intelligence" / "judgment_baseline_v1.json").read_text())
    questions = ns["QUESTIONS"]
    cases = []
    for case in fixture["cases"]:
        state = serialize_state(ns["state_for"](case, "json"))
        items = []
        for qid, qdef in questions.items():
            q = agent._to_internal(qdef)
            ids, markers = build_sequence(agent.tok, state, q, agent.cfg["max_len"], agent.cfg["head_max_len"])
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
        batch = collate_items([items], agent.tok.pad_token_id)
        with torch.no_grad():
            logits, act = agent.model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"],
                                      batch["marker_mask"], batch["qtype"])
        act = torch.softmax(act.float(), -1).numpy()
        response = agent.predict(state, questions)
        answers = []
        for r, qid in enumerate(questions):
            k = len(items[r]["markers"])
            z = logits[r, :k].numpy().astype(np.float64)
            got = response["answers"][qid]
            answers.append({
                "id": qid,
                "ids": items[r]["ids"],
                "markers": items[r]["markers"],
                "logits": z.tolist(),
                "act_probability": float(act[r, 0]),
                "choice": got["choice"],
                "probabilities_rounded": list(got["probabilities"].values()),
                "expected": case["expected"][qid],
            })
        cases.append({"id": case["id"], "state": state, "answers": answers})
        print(case["id"], [a["choice"] for a in answers], flush=True)
    weights = (args.checkpoint / "model.safetensors").resolve()
    golden = {
        "producer": "laya %s Agent from source, torch %s, transformers %s, cpu float32" % (
            laya.__version__, torch.__version__, transformers.__version__),
        "checkpoint": "convaiinnovations/laya multilingual, revision 1c5edc17 (offline replication evidence only)",
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "config_patch": "encoder/config.json global_rope_theta/local_rope_theta copied from rope_parameters",
        "questions": [{"id": qid, "type": q["type"], "instructions": q["instructions"],
                       "options": [{"key": k, "description": v} for k, v in q["criteria"].items()]}
                      for qid, q in questions.items()],
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(golden, ensure_ascii=False, separators=(",", ":")) + "\n")
    print("wrote", args.out, len(cases), "cases")


if __name__ == "__main__":
    main()
