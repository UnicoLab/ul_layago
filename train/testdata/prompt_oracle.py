"""Test oracle: the Python trainer's loader and renderer on the Go fixture.

Loads a hive.laya-dataset/v1 file with tools/laya-train's dataset.load (the
converged loader: every contract check, the split invariants and the data
digest) and renders every example with prompt.render_state, then compares
with the states and digest the Go trainer wrote. Nothing is downloaded; the
Python trainer is tools/laya-train of this checkout (or of any other that
has it).

    go test -count=1 -run TestFixtureSampleIsCurrent ./internal/layago/train \
        LAYAGO_TRAIN_FULL_FIXTURE_OUT=/tmp/fixture.jsonl LAYAGO_TRAIN_STATES_OUT=/tmp/go_states.jsonl
    python3 internal/layago/train/testdata/prompt_oracle.py \
        --laya-train /path/to/checkout/tools/laya-train --data /tmp/fixture.jsonl --go /tmp/go_states.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--laya-train", required=True, help="tools/laya-train directory of a converged checkout")
    ap.add_argument("--data", required=True)
    ap.add_argument("--go", required=True, help="JSONL of {id, state, digest} written by the Go test")
    args = ap.parse_args()
    sys.path.insert(0, args.laya_train)
    from laya_train import dataset, prompt  # noqa: PLC0415

    ds = dataset.load([args.data])
    go = {}
    digest = None
    with open(args.go, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            go[rec["id"]] = rec["state"]
            digest = rec["digest"]
    mismatches = 0
    for ex in ds.examples:
        state = prompt.render_state(ex)
        if go.get(ex.id) != state:
            mismatches += 1
            print("MISMATCH", ex.id, repr(go.get(ex.id)), repr(state))
    print(
        json.dumps(
            {
                "examples": len(ds.examples),
                "go_states": len(go),
                "state_mismatches": mismatches,
                "python_digest": ds.digest,
                "go_digest": digest,
                "digest_equal": ds.digest == digest,
                "split_violations": ds.split_violations,
                "dev_halves": {h: sum(1 for e in ds.examples if dataset.eval_split(e) == h) for h in ("dev_cal", "dev_conf")},
            },
            indent=1,
        )
    )
    return 0 if mismatches == 0 and ds.digest == digest and len(go) == len(ds.examples) else 1


if __name__ == "__main__":
    sys.exit(main())
