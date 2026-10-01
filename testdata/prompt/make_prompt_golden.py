"""Writes prompt_golden.jsonl: hive.laya-prompt/v1 states rendered by the trainer.

The Go port (internal/layago/laya/prompt.go) must render the same state from
the same hive.laya-dataset/v1 input, byte for byte; TestPromptMatchesTrainer
holds it to this file. Every line is one example in contract JSON with the
state the trainer's prompt.render_state gives it:

    {"example": {...hive.laya-dataset/v1...}, "state": "..."}

The examples are the trainer's own fixture (every example the committed tiny-q
parity goldens use, then a fixed sample across languages, domains, tasks and
splits), its R and A examples, and edge cases of the rendering rules. Nothing
is downloaded and no model runs. Regenerate from the repository root:

    /opt/miniconda3/bin/python3 internal/layago/testdata/prompt/make_prompt_golden.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "tools" / "laya-train"))

from laya_train import dataset, fixture, prompt  # noqa: E402

ARTIFACT = REPO / "internal" / "layago" / "testdata" / "artifact" / "tiny-q"
SAMPLE_EVERY = 9


def golden_ids() -> list[str]:
    ids: list[str] = []
    for variant in ("float32", "int8"):
        g = json.loads(
            (ARTIFACT / variant / "parity_golden.json").read_text(encoding="utf-8")
        )
        for case in g["cases"]:
            if case["id"] not in ids:
                ids.append(case["id"])
    return ids


def edge_cases(base: dict) -> list[dict]:
    """Inputs that exercise each rule of prompt.render_state."""

    def q(eid: str, inp: dict, **extra) -> dict:
        return dict(base, id=eid, input=inp, **extra)

    nbsp, fs, emsp = " ", "\u001c", " "
    return [
        # The trainer's own rendering test input: label objects in sorted
        # language order with duplicates dropped, blank dimensions dropped,
        # scope dimensions sorted, a label list, a single label string, an
        # empty unit, a bare string metric and a question trimmed at both ends.
        q(
            "edge-trainer-test",
            {
                "question": "  Quel était le CA à Paris ?  ",
                "catalog": [
                    {
                        "key": "retail.revenue",
                        "labels": {
                            "fr": "Chiffre d'affaires",
                            "en": "Revenue",
                            "de": "Revenue",
                        },
                        "unit": "EUR",
                        "dimensions": ["store", " "],
                        "scope_values": {
                            "store": ["Paris", "Lyon"],
                            "channel": ["web"],
                        },
                    },
                    {"key": "retail.orders", "labels": ["Orders"]},
                    {"labels": "Footfall", "unit": ""},
                    "plain metric",
                ],
            },
        ),
        # Python's str.strip() whitespace: NBSP, an em space and the ASCII
        # information separators (U+001C..U+001F), which Go's unicode.IsSpace
        # does not include.
        q(
            "edge-python-whitespace",
            {
                "question": fs + nbsp + " Umsatz in Köln? " + emsp + fs,
                "catalog": [
                    {
                        "key": fs + " sales.net " + nbsp,
                        "labels": {"de": fs + "Nettoumsatz" + fs, "en": " Net sales "},
                        "unit": emsp + "EUR" + emsp,
                        "dimensions": [fs, " city "],
                        "scope_values": {"city": [" Köln ", fs, "Bonn"]},
                    }
                ],
            },
        ),
        # Interior newlines and tabs are kept; only the ends are trimmed.
        q(
            "edge-interior-controls",
            {
                "question": "Revenue\tlast week\nin Paris?",
                "catalog": [{"key": "a\tb", "labels": ["x\ny"]}],
            },
        ),
        # Non-string JSON values go through Python's str(): integers, floats,
        # booleans and null; non-string labels are dropped.
        q(
            "edge-json-scalars",
            {
                "question": "Orders in 2024?",
                "catalog": [
                    {
                        "key": None,
                        "labels": ["Orders", 7, None, "Orders", ""],
                        "unit": 3,
                        "dimensions": [
                            2024,
                            True,
                            None,
                            1.5,
                            0.1,
                            1e16,
                            1e-5,
                            100.0,
                            -0.0,
                        ],
                        "scope_values": {
                            "year": [2024, 2025.0, False, None],
                            "empty": [" "],
                            "z": "not a list",
                        },
                    },
                    {"key": "k", "label": "Fallback label"},
                    {
                        "key": "k2",
                        "labels": None,
                        "label": "ignored: labels is present",
                    },
                    {"key": "", "labels": {}},
                    {},
                    12,
                ],
            },
        ),
        # Keys only, labels only, unit only.
        q(
            "edge-partial-metrics",
            {
                "question": "What changed?",
                "catalog": [
                    {"key": "only.key"},
                    {"labels": ["Only label"]},
                    {"unit": "EUR"},
                    {"dimensions": ["d"]},
                ],
            },
        ),
        # An empty or missing catalog renders the question line alone.
        q("edge-empty-catalog", {"question": "Why?", "catalog": []}),
        q("edge-no-catalog", {"question": "Why?"}),
        # Scripts beyond Latin, and an emoji.
        q(
            "edge-scripts",
            {
                "question": "上个月的收入是多少？ 📈",
                "catalog": [
                    {
                        "key": "retail.revenue",
                        "labels": {
                            "zh": "收入",
                            "ar": "الإيرادات",
                            "ru": "Выручка",
                            "hi": "राजस्व",
                        },
                        "unit": "CNY",
                        "dimensions": ["门店"],
                        "scope_values": {"门店": ["上海", "北京"]},
                    }
                ],
            },
        ),
    ]


def other_instances(base: dict) -> list[dict]:
    out = list(fixture.mixed_instance_examples())
    out.append(
        dict(
            base,
            id="edge-r-plain",
            instance="R",
            task="r_answer_rank",
            task_type="choice",
            options=["ev1", "2"],
            label=0,
            input={
                "question": " Why? ",
                "evidence": [
                    {"id": " ev1 ", "text": " a "},
                    "plain item",
                    {"text": "no id"},
                    {"id": 5},
                ],
            },
        )
    )
    out.append(
        dict(
            base,
            id="edge-a-partial",
            instance="A",
            task="a_causal_licensed",
            task_type="yes_no",
            label=True,
            input={"question": "Why?", "draft": "  ", "sentence": " It rose. "},
        )
    )
    return out


def main() -> None:
    examples = fixture.generate(7)
    by_id = {e["id"]: e for e in examples}
    chosen = [by_id[i] for i in golden_ids()]
    seen = {e["id"] for e in chosen}
    for n, e in enumerate(examples):
        if n % SAMPLE_EVERY == 0 and e["id"] not in seen:
            chosen.append(e)
            seen.add(e["id"])
    base = dict(examples[0])
    for key in ("options", "label"):
        base.pop(key, None)
    base.update(task="q_time_supported", task_type="yes_no", label=False)
    chosen += edge_cases(base)
    chosen += other_instances(base)

    lines = []
    for raw in chosen:
        ex = dataset.parse_example(raw)
        lines.append(
            json.dumps(
                {"example": raw, "state": prompt.render_state(ex)},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    out = HERE / "prompt_golden.jsonl"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{out.relative_to(REPO)}: {len(lines)} examples")


if __name__ == "__main__":
    main()
