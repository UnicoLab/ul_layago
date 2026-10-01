#!/usr/bin/env python3
"""Write the mmBERT tokenizer golden used by TestExternalTokenizerParity.

The golden pins the sha256 of a real mmBERT/Gemma-2 tokenizer.json (the
Laya multilingual checkpoint's copy; the file itself is not committed) and
the ids and byte offsets Hugging Face tokenizers produces for a corpus of
multilingual business questions, the Laya prompts Hive's frozen judgment
harness sends (scripts/evaluation/evaluate_laya.py), tokenizer edge cases
and seeded random strings. It needs the `tokenizers` package and a local
tokenizer.json; nothing is downloaded:

    python3 internal/layago/testdata/parity/mmbert_tokenizer_golden.py \\
        /path/to/tokenizer.json internal/layago/tokenizer/testdata/mmbert_tokenizer_golden.json

The mmBERT-base golden (mmbert_base_tokenizer_golden.json, for the
jhu-clsp/mmBERT-base tokenizer.json) adds the multilingual business corpus in
testdata/parity/multilingual_corpus.txt and more random strings:

    python3 internal/layago/testdata/parity/mmbert_tokenizer_golden.py \\
        /path/to/mmbert-base/tokenizer.json \\
        internal/layago/tokenizer/testdata/mmbert_base_tokenizer_golden.json \\
        --extra internal/layago/testdata/parity/multilingual_corpus.txt --random 160
"""

from __future__ import annotations

import ast
import hashlib
import json
import random
import sys
from pathlib import Path

import tokenizers
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[4]

MULTILINGUAL = [
    "Why did weekly revenue drop in the north region?",
    "Pourquoi le chiffre d'affaires hebdomadaire a-t-il baissé dans la région nord ?",
    "Warum ist der Wochenumsatz in der Region Nord gesunken?",
    "¿Por qué bajaron los ingresos semanales en la región norte?",
    "Perché il fatturato settimanale è calato nella regione nord?",
    "Por que a receita semanal caiu na região norte?",
    "Waarom daalde de wekelijkse omzet in de regio noord?",
    "Dlaczego tygodniowy przychód spadł w regionie północnym?",
    "Почему еженедельная выручка упала в северном регионе?",
    "Чому щотижневий дохід впав у північному регіоні?",
    "لماذا انخفضت الإيرادات الأسبوعية في المنطقة الشمالية؟",
    "מדוע ירדו ההכנסות השבועיות באזור הצפון?",
    "چرا درآمد هفتگی در منطقه شمال کاهش یافت؟",
    "उत्तरी क्षेत्र में साप्ताहिक राजस्व क्यों गिरा?",
    "উত্তর অঞ্চলে সাপ্তাহিক আয় কেন কমেছে?",
    "வடக்கு பிராந்தியத்தில் வாராந்திர வருவாய் ஏன் குறைந்தது?",
    "ทำไมรายได้รายสัปดาห์ในภาคเหนือจึงลดลง",
    "为什么北部地区的每周收入下降了？",
    "北部地域の週次売上が下がったのはなぜですか？",
    "북부 지역의 주간 매출이 왜 감소했나요?",
    "Tại sao doanh thu hàng tuần ở khu vực phía bắc giảm?",
    "Kuzey bölgesinde haftalık gelir neden düştü?",
    "Γιατί μειώθηκαν τα εβδομαδιαία έσοδα στη βόρεια περιοχή;",
    "Kwa nini mapato ya kila wiki yalishuka katika eneo la kaskazini?",
    "በሰሜን ክልል ሳምንታዊ ገቢ ለምን ቀነሰ?",
    "မြောက်ပိုင်းဒေသတွင် အပတ်စဉ်ဝင်ငွေ အဘယ်ကြောင့် ကျဆင်းသနည်း",
    "ហេតុអ្វីបានជាចំណូលប្រចាំសប្តាហ៍នៅភាគខាងជើងធ្លាក់ចុះ?",
    "今月のGMVは?",
    "Q3 CAC vs LTV — northwest, EUR, week 38",
]

EDGE = [
    "",
    " ",
    "  leading spaces",
    "trailing  ",
    "Hey   friend!",
    "tab\there\nnew\n\nlines\n\n\n",
    "choice question: Is the <mask> ready?",
    "a <mask>b<mask> c",
    "a  <mask>",
    "<pad><eos>x",
    "emoji 😀🎉 test",
    "ZWNJ می‌خواهم",
    "<table><tr><td>1</td></tr></table>",
    "<unused0> <unused1>",
    "revenue_net_eur_wk dropped 12.5% vs last week",
    '{"metric_key": "orders", "value": 1200.0}',
    "▁▁ already metaspace",
    " nbsp　ideographic",
    "full-width ＡＢＣ１２３",
    "NFD é vs é",
    "mixed 123 ٣٤٥ ४५६",
    "\t\t\tindented\t\t",
    "x" + "\n" * 35 + "y",
    "<start_of_turn>user\nhi<end_of_turn>",
]


def laya_prompts() -> list[str]:
    """Question prompts, option texts and a few rendered states from the frozen harness."""
    source = (ROOT / "scripts" / "evaluation" / "evaluate_laya.py").read_text()
    tree = ast.parse(source)
    namespace: dict = {"json": json}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "QUESTIONS" for t in node.targets):
            exec(compile(ast.Module([node], []), "evaluate_laya.py", "exec"), namespace)
        if isinstance(node, ast.FunctionDef) and node.name == "state_for":
            exec(compile(ast.Module([node], []), "evaluate_laya.py", "exec"), namespace)
    out = []
    for q in namespace["QUESTIONS"].values():
        out.append("%s question: %s" % (q["type"], q["instructions"]))
        for key, desc in q["criteria"].items():
            out.append(" %s: %s" % (key, desc))
    fixture = json.loads((ROOT / "testdata" / "adaptive_intelligence" / "judgment_baseline_v1.json").read_text())
    for case in fixture["cases"][:3]:
        out.append(json.dumps(namespace["state_for"](case, "json"), ensure_ascii=False))
        out.append(namespace["state_for"](case, "narrative"))
    return out


POOLS = [
    "abcdefghijklmnopqrstuvwxyz",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "  \t\n\n 　",
    ".,:;?!-_'\"(){}[]%/+=<>",
    "éèàüöçñßøå",
    "中文日本語한국어",
    "العربيةעבריתفارسی",
    "हिन्दीবাংলাதமிழ்ไทย",
    "😀🎉👍🏽🇫🇷",
    "́̈‌‍",
]
FRAGMENTS = ["<mask>", "<pad>", "<unused3>", "<b>", "</td>", "\n\n", "▁▁", "\t\t"]


def random_strings(n: int) -> list[str]:
    rng = random.Random(20260926)
    out = []
    for _ in range(n):
        parts = []
        for _ in range(rng.randint(1, 12)):
            if rng.random() < 0.15:
                parts.append(rng.choice(FRAGMENTS))
            else:
                pool = rng.choice(POOLS)
                parts.append("".join(rng.choice(pool) for _ in range(rng.randint(1, 6))))
        out.append("".join(parts))
    return out


def extra_strings(path: str) -> list[str]:
    """One JSON-encoded string per line (blank lines ignored)."""
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def main() -> None:
    # Positional: tokenizer.json, output. Optional: --random N (default 120)
    # and --extra FILE (JSON strings, one per line) to widen the corpus.
    args = sys.argv[1:]
    n_random, extra = 120, []
    while len(args) > 2:
        flag, value = args[-2], args[-1]
        if flag == "--random":
            n_random = int(value)
        elif flag == "--extra":
            extra = extra_strings(value)
        else:
            sys.exit("usage: mmbert_tokenizer_golden.py tokenizer.json out.json [--random N] [--extra FILE]")
        args = args[:-2]
    path, out_path = args[0], args[1]
    tok = Tokenizer.from_file(path)
    corpus = MULTILINGUAL + EDGE + laya_prompts() + extra + random_strings(n_random)
    cases = []
    for text in corpus:
        enc = tok.encode(text, add_special_tokens=False)
        b = [len(text[:i].encode()) for i in range(len(text) + 1)]
        cases.append({"text": text, "ids": enc.ids, "offsets": [[b[s], b[e]] for s, e in enc.offsets]})
    golden = {
        "source_file_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "producer": "huggingface tokenizers %s (Rust), encode(add_special_tokens=False)" % tokenizers.__version__,
        "cases": cases,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(golden, ensure_ascii=False, separators=(",", ":")) + "\n")
    print("wrote %d cases, %d ids" % (len(cases), sum(len(c["ids"]) for c in cases)))


if __name__ == "__main__":
    main()
