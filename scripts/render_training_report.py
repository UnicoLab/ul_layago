"""Render a self-contained training report from an exported artifact manifest."""
import argparse
import html
import json
import math
from pathlib import Path


def plot(history, keys):
    width, height = 720, 220
    finite = [float(row[key]) for row in history for key in keys if math.isfinite(float(row[key]))]
    maximum = max(finite + [0.01])
    colors = ("#2563eb", "#dc2626", "#059669")
    lines = []
    for color, key in zip(colors, keys):
        points = " ".join(f"{40+i*640/max(1,len(history)-1):.1f},{190-float(row[key])*160/maximum:.1f}" for i, row in enumerate(history) if math.isfinite(float(row[key])))
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/><text x="40" y="{18+keys.index(key)*18}" fill="{color}">{html.escape(key)}</text>')
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(", ".join(keys))} by epoch"><path d="M40 40 V190 H690" fill="none" stroke="#94a3b8"/>{"".join(lines)}<text x="40" y="215">Epoch 1</text><text x="630" y="215">Epoch {len(history)}</text></svg>'


def render(manifest):
    training = manifest["training"]
    history = training["history"]
    metadata = {"contract": manifest["contract"], "instance": manifest["instance"], "code": manifest["code_version"], "base": manifest["base"], "dataset": training.get("data_digest"), "best_epoch": training.get("best_epoch"), "limitations": manifest["limitations"]}
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>LayaGo training report</title><style>body{font:16px system-ui;background:#f1f5f9;color:#0f172a;max-width:1000px;margin:40px auto;padding:24px}section{background:white;border:1px solid #cbd5e1;border-radius:12px;padding:20px;margin:20px 0}svg{width:100%;background:#f8fafc}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}h1{margin-bottom:8px}</style><h1>LayaGo training report</h1><p>Synthetic engineering results require domain validation. Calibration coverage is distribution-specific.</p><section><h2>Loss by epoch</h2>'+plot(history,["train_loss", "dev_cal_nll"])+"</section><section><h2>Validation accuracy</h2>"+plot(history,["dev_cal_accuracy"])+"</section><section><h2>Evaluation and calibration</h2><pre>"+html.escape(json.dumps({"evaluation": manifest["evaluation"], "calibration": manifest["calibration"], "act": manifest["act"]}, indent=2))+"</pre></section><section><h2>Run identity and limitations</h2><pre>"+html.escape(json.dumps(metadata, indent=2))+"</pre></section></html>"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    text = render(json.loads((args.artifact / "manifest.json").read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as file:
        file.write(text)
    print(args.out)


if __name__ == "__main__":
    main()
