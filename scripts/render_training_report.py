"""Render a self-contained training report from an exported artifact manifest."""
import argparse
import html
import json
import math
from pathlib import Path


def escape(value):
    return html.escape(str(value))


def plot(history, keys):
    def recorded(row, key):
        return isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))
    finite = [float(row[key]) for row in history for key in keys if recorded(row, key)]
    maximum = max(finite + [0.01])
    colors = ("#2563eb", "#dc2626", "#059669")
    lines = []
    for index, (color, key) in enumerate(zip(colors, keys)):
        points = " ".join(f"{40+i*640/max(1,len(history)-1):.1f},{190-float(row[key])*160/maximum:.1f}" for i, row in enumerate(history) if recorded(row, key))
        if not any(recorded(row, key) for row in history):
            lines.append(f'<text x="40" y="{70+index*20}" fill="#64748b">{escape(key)}: not recorded in this artifact</text>')
            continue
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/><text x="40" y="{18+index*18}" fill="{color}">{escape(key)}</text>')
    return f'<svg viewBox="0 0 720 220" role="img" aria-label="{escape(", ".join(keys))} by epoch"><path d="M40 50 V190 H690" fill="none" stroke="#94a3b8"/>{"".join(lines)}<text x="5" y="65">{maximum:.2f}</text><text x="10" y="190">0</text><text x="40" y="215">Epoch 1</text><text x="630" y="215">Epoch {len(history)}</text></svg>'


def percent(value):
    return '—' if value is None else f'{100*float(value):.1f}%'


def render(manifest):
    training = manifest["training"]
    history = training["history"]
    metadata = {"contract": manifest["contract"], "instance": manifest["instance"], "code": manifest["code_version"], "base": manifest["base"], "dataset": training.get("data_digest"), "best_epoch": training.get("best_epoch")}
    evaluation = manifest['evaluation'].get('splits', {})
    rows = []
    for split in ('train', 'dev_cal', 'dev_conf', 'heldout_domain', 'heldout_language', 'locked_test'):
        if split not in evaluation:
            continue
        metrics = evaluation[split].get('all', {})
        rows.append('<tr>'+''.join(f'<td>{escape(value)}</td>' for value in [split.replace('_', ' '), metrics.get('n', '—'), percent(metrics.get('accuracy')), percent(metrics.get('macro_f1')), percent(metrics.get('coverage')), percent(metrics.get('ece'))])+'</tr>')
    limitations = ''.join(f'<li>{escape(text)}</li>' for text in manifest['limitations'])
    details = html.escape(json.dumps({"evaluation": manifest["evaluation"], "calibration": manifest["calibration"], "act": manifest["act"], "identity": metadata}, indent=2))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>LayaGo training report</title>
<style>body{{font:16px system-ui;background:#f1f5f9;color:#0f172a;max-width:1040px;margin:32px auto;padding:20px}}section,.cards article{{background:white;border:1px solid #cbd5e1;border-radius:14px;padding:20px;margin:18px 0}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}.cards article{{margin:0}}.cards strong{{display:block;margin-top:6px;font-size:24px}}svg{{width:100%;background:#f8fafc;border-radius:8px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}h1{{margin-bottom:8px}}h2{{font-size:19px}}p,li{{line-height:1.6}}.table{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:12px 8px;border-bottom:1px solid #e2e8f0;text-align:left}}summary{{cursor:pointer;font-weight:600}}.badge{{color:#92400e;background:#fef3c7;padding:5px 10px;border-radius:8px}}@media print{{body{{background:white;margin:0}}section{{break-inside:avoid}}}}</style></head><body>
<header><span class="badge">Synthetic engineering · review required</span><h1>LayaGo training report</h1><p>Training progress and evaluation for a frozen-encoder decision head. Model nominations do not establish source evidence.</p></header>
<div class="cards"><article>Selected epoch<strong>{escape(training.get('best_epoch', 'Not recorded'))}</strong></article><article>Frozen encoder<strong style="font-size:17px">{escape(manifest['base'].get('model_id', 'Pinned base'))}</strong></article><article>Decision policy<strong style="font-size:17px">Calibrated act / escalate</strong><small>Coverage applies only to the calibration distribution.</small></article></div>
<section><h2>Loss by epoch</h2>{plot(history,['train_loss','dev_cal_nll'])}</section>
<section><h2>Validation accuracy by epoch</h2>{plot(history,['dev_cal_accuracy'])}</section>
<section><h2>Evaluation at a glance</h2><div class="table"><table><thead><tr><th>Split</th><th>Examples</th><th>Accuracy</th><th>Macro F1</th><th>Act coverage</th><th>Calibration error</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p>Locked tests stay sealed during fitting. Any separate post-export test receipt is reported separately.</p></section>
<section><h2>Limits of this run</h2><ul>{limitations}</ul></section>
<section><details><summary>Inspect calibration, per-language metrics and run identity</summary><pre>{details}</pre></details></section></body></html>'''


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
