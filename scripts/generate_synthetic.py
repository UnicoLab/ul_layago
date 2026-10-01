"""Deterministic evidence-support engineering fixture; not a production quality benchmark."""
import argparse
import json
from pathlib import Path

OPTIONS = [{"key": "supports", "text": "Every material part is explicitly supported."},
           {"key": "contradicts", "text": "A material part is explicitly contradicted."},
           {"key": "insufficient", "text": "The source does not establish the claim."}]
INSTRUCTIONS = {"r_evidence_support": "Does the evidence support every material part of the claim in the question?"}


def generate(stories=40, seed=20261002):
    if stories < 12:
        raise ValueError("at least 12 stories per split are needed for dev halves")
    rows = []
    for split_no, split in enumerate(("train", "dev", "heldout_domain", "locked_test")):
        for n in range(stories):
            serial = split_no * stories + n + seed
            count = serial % 83 + 10
            site = f"Depot-{serial}"
            lang = "fr" if n % 2 else "en"
            source = (f"Le dépôt {site} contient {count} caisses." if lang == "fr" else f"Warehouse {site} contains {count} crates.")
            claims = ((f"Le dépôt {site} contient {count} caisses.", f"Le dépôt {site} contient {count+3} caisses.", f"Les caisses du dépôt {site} sont rouges.") if lang == "fr" else (f"Warehouse {site} contains {count} crates.", f"Warehouse {site} contains {count+3} crates.", f"The crates at warehouse {site} are red."))
            for label, claim in enumerate(claims):
                rows.append({"id": f"fixture-{serial}-{label}", "instance": "R", "task": "r_evidence_support", "task_type": "choice", "input": {"question": claim, "evidence": [{"id": "source", "text": source}]}, "options": OPTIONS, "label": label, "label_source": "template", "split": split, "lang": lang, "domain_id": "unseen-warehouse" if split == "heldout_domain" else "warehouse", "story_id": f"fixture-story-{serial}", "generator_version": "ul-fixture/1", "seed": seed})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stories", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    rows = generate(args.stories, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as file:
        file.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    print(json.dumps({"examples": len(rows), "output": str(args.out)}))


if __name__ == "__main__":
    main()
