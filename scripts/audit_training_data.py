#!/usr/bin/env python3
"""Bounded, data-only preflight for hive.laya-dataset/v1 JSONL.

No model, tokenizer, network, or training is used. The optional surface probe
fits only train labels and evaluates only explicitly requested dev/dev_conf.
Mixed-file JSON must be decoded to see split; sealed rows are then immediately
discarded, before input/label validation, rendering, hashes or statistics.
Thus this is logical isolation, not encrypted or byte-level secrecy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re

PROTOCOL = "layago-training-data-audit-v1"
SPLITS = {"train", "dev", "heldout_domain", "heldout_language", "locked_test"}
MAX_LINE = 2_000_000
MAX_BYTES = 256_000_000
MAX_ROWS = 50_000
MAX_FINDINGS = 250
MAX_REPORT_BYTES = 4_000_000
TOKEN = re.compile(r"\w+", re.UNICODE)
LEFTOVER = re.compile(r"\{\{[^{}]{1,80}\}\}|\$\{[^{}]{1,80}\}|\b(?:TODO|INSERT_HERE|REPLACE_ME)\b|<\|[^|]{1,60}\|>")


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def dev_half(story):
    return "dev_conf" if hashlib.sha256(("laya-dev-half\x00" + story).encode()).digest()[0] & 1 else "dev_cal"


def calibration_target(alpha=.05, delta=.1, n_min=20, start_errors=0):
    """Go LTTStart(alpha, delta/2, nMin, startErrors), bounded, not certification."""
    if type(alpha) not in (int, float) or type(delta) not in (int, float) or not (0 < alpha < 1 and 0 < delta < 1):
        raise ValueError("alpha and delta must be finite probabilities")
    if type(n_min) is not int or not 1 <= n_min <= 100_000 or type(start_errors) is not int or not 0 <= start_errors <= 100:
        raise ValueError("bounded positive n_min and nonnegative start_errors required")
    share = delta / 2
    if share==0:raise ValueError('score-family delta is too small')
    if start_errors == 0:
        estimate=3 * math.log(share) / math.log1p(-alpha)
        if not math.isfinite(estimate) or estimate>100_000:raise ValueError('calibration target exceeds bounded computation')
        return max(n_min, math.ceil(estimate))
    for n in range(start_errors + 1, 100_001):
        # Stable recurrence from P(X=0), enough for this bounded readiness check.
        term = math.exp(n * math.log1p(-alpha)); total = term
        for k in range(start_errors):
            term *= (n - k) / (k + 1) * alpha / (1 - alpha); total += term
        if total <= share:
            return max(n_min, n)
    raise ValueError("calibration target exceeds bounded computation")


def scalar(value):
    if isinstance(value, (dict, list)):
        raise ValueError("nested scalar rendering is unsupported, as Go RenderPromptState")
    return str(value)


def metric_line(m):
    if not isinstance(m, dict):
        return "- " + scalar(m)
    key = scalar(m.get("key", "")).strip()
    raw = m.get("labels", m.get("label"))
    vals = [raw] if isinstance(raw, str) else [raw[k] for k in sorted(raw)] if isinstance(raw, dict) else raw if isinstance(raw, list) else []
    labels = list(dict.fromkeys(v.strip() for v in vals if isinstance(v, str) and v.strip()))
    parts = [((key + ": ") if key else "") + " | ".join(labels)] if labels else [key] if key else []
    if isinstance(m.get("unit"), str) and m["unit"].strip():parts.append("unit: " + m["unit"].strip())
    dims = [scalar(v).strip() for v in m.get("dimensions", [])] if isinstance(m.get("dimensions"), list) else []
    if any(dims):parts.append("dimensions: " + ", ".join(v for v in dims if v))
    scope = m.get("scope_values")
    if isinstance(scope, dict):
        scopes = []
        for k in sorted(scope):
            values = [scalar(v).strip() for v in scope[k]] if isinstance(scope[k], list) else []
            if any(values):scopes.append(k + "=" + "/".join(v for v in values if v))
        if scopes:parts.append("scope: " + ", ".join(scopes))
    return "- " + "; ".join(parts)


def render_state(row):
    """Scalar Q/R/A port of laya.RenderPromptState; no options/gold/metadata."""
    obj = row["input"]; lines = ["question: " + obj["question"].strip()]
    if row["instance"] == "Q":
        if obj.get("catalog"):
            lines.append("catalog:"); lines.extend(metric_line(m) for m in obj["catalog"])
    elif row["instance"] == "R":
        if obj.get("evidence"):lines.append("evidence:")
        for i, item in enumerate(obj.get("evidence") or []):
            ident, text = (item.get("id", i), item.get("text", "")) if isinstance(item, dict) else (i, item)
            lines.append("- " + scalar(ident).strip() + ": " + scalar(text).strip())
    else:
        for k in ("draft", "sentence"):
            if isinstance(obj.get(k), str) and obj[k].strip():lines.append(k + ": " + obj[k].strip())
    return "\n".join(lines)


def source_parts(row):
    """Fixed questions, choice options and Q catalogs are NOT source evidence.

    Without explicit R evidence or A draft/sentence, Q/question-only rows get
    no source-copy check (their full rendered state is still checked exactly).
    """
    obj = row["input"]
    if row["instance"] == "R":
        return [scalar(v.get("text", "")) if isinstance(v, dict) else scalar(v) for v in obj.get("evidence", [])]
    if row["instance"] == "A":return [obj[k] for k in ("draft", "sentence") if isinstance(obj.get(k), str)]
    return []


def parse_row(raw, catalog=None):
    if not isinstance(raw, dict):raise ValueError("row must be an object")
    for key in ("id", "split", "lang", "instance", "task", "story_id", "domain_id", "generator_version", "label_source"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():raise ValueError("missing string " + key)
        raw[key].encode("utf-8")
    if raw["split"] not in SPLITS or raw["instance"] not in {"Q", "R", "A"}:raise ValueError("unknown split or instance")
    if not re.fullmatch(r"[a-z]{2,3}(?:-[A-Za-z0-9]{1,8})*", raw["lang"]):raise ValueError("invalid language")
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", raw["task"]) or not raw["task"].startswith(raw["instance"].lower() + "_"):raise ValueError("invalid task")
    if type(raw.get("seed")) is not int:raise ValueError("integer seed required")
    if raw["label_source"] not in {"spec", "template", "llm_paraphrase", "llm_translation", "claude_author"}:raise ValueError("unknown label_source")
    provenance=raw.get('provenance')
    if isinstance(provenance,dict):provenance=provenance.get('llm_interaction_ids',provenance.get('llm_interaction_id'))
    if isinstance(provenance,str):provenance=[provenance]
    if provenance is None:provenance=[]
    if not isinstance(provenance,list) or any(not isinstance(v,str) or not re.fullmatch('[0-9a-f]{64}',v) for v in provenance):raise ValueError('invalid provenance')
    if raw['label_source'] in {'llm_paraphrase','llm_translation','claude_author'} and not provenance:raise ValueError('model-generated text needs provenance')
    inp = raw.get("input")
    if not isinstance(inp, dict) or not isinstance(inp.get("question"), str) or not inp["question"].strip():raise ValueError("input question required")
    list_key = {"Q": "catalog", "R": "evidence"}.get(raw["instance"])
    if list_key and list_key in inp and not isinstance(inp[list_key], list):raise ValueError("input list required")
    for k in ("draft", "sentence"):
        if raw["instance"] == "A" and k in inp and not isinstance(inp[k], str):raise ValueError("draft/sentence must be strings")
    typ = raw.get("task_type"); options = raw.get("options", [])
    if options is None:options=[]
    if typ not in {"choice", "score", "yes_no"} or not isinstance(options, list) or len(options) > 256:raise ValueError("invalid task_type/options")
    keys = []; texts = []
    for i, v in enumerate(options):
        if isinstance(v, str) and v.strip():key, text = (v, "") if typ == "choice" else (str(i), v)
        elif isinstance(v, dict):
            key = v.get("key") if typ == "choice" else str(i); text = v.get("text", "")
            if not isinstance(key, str) or not key.strip() or not isinstance(text, str) or (typ != "choice" and not text.strip()):raise ValueError("invalid option")
        else:raise ValueError("invalid option")
        key.encode('utf-8');text.encode('utf-8')
        keys.append(key);texts.append(text)
    if typ == "yes_no":
        if type(raw.get("label")) is not bool or len(keys) not in (0, 2):raise ValueError("yes_no needs boolean label")
        keys = ["false", "true"]; texts = texts or ["false", "true"]; label = int(raw["label"])
    else:
        label = raw.get("label")
        if len(keys) < 2 or len(set(keys)) != len(keys) or type(label) is not int or not 0 <= label < len(keys):raise ValueError("unknown label index or option")
    if catalog is not None:
        allowed = catalog.get(raw["task"])
        if not isinstance(allowed, list) or not all(isinstance(k, str) for k in allowed) or set(keys) != set(allowed):raise ValueError("unknown task/option in explicit catalog")
    dist = raw.get("label_dist")
    if dist is not None:
        if raw["split"] != "train" or not isinstance(dist, list) or len(dist) != len(keys) or any(type(p) not in (int, float) or not math.isfinite(p) or p < 0 for p in dist) or abs(sum(dist) - 1) > 1e-6 or dist[label] != max(dist):raise ValueError("invalid soft label")
    rendered = render_state(raw); rendered.encode("utf-8")
    parts = source_parts(raw)
    if any(not isinstance(p, str) for p in parts):raise ValueError("source text must be string")
    return {"raw": raw, "keys": keys, "texts": texts, "label": label, "label_key": keys[label], "rendered": rendered, "parts": parts,
            "role": dev_half(raw["story_id"]) if raw["split"] == "dev" else raw["split"]}


@dataclass(frozen=True)
class NearConfig:
    min_words: int = 20
    max_words: int = 20_000
    max_shingles: int = 256
    anchors: int = 8
    max_posting: int = 128
    max_comparisons: int = 20_000
    max_parts: int = 2_000
    threshold: float = .85

    def __post_init__(self):
        for key in ("min_words", "max_words", "max_shingles", "anchors", "max_posting", "max_comparisons", "max_parts"):
            value=getattr(self,key)
            if type(value) is not int or not 1<=value<=20_000:raise ValueError("bounded positive near-copy limits required")
        if self.min_words<5 or self.max_words<self.min_words or self.max_shingles>256 or self.anchors>self.max_shingles or self.max_parts>2_000 or self.max_posting>128 or type(self.threshold) not in (int,float) or not 0<self.threshold<=1:
            raise ValueError("invalid near-copy configuration")


class Findings:
    def __init__(self):self.counts = Counter();self.items = []
    def add(self, kind, **data):
        self.counts[kind] += 1
        if len(self.items) < MAX_FINDINGS:self.items.append({"kind": kind, **data})
    def report(self):return {"counts": dict(sorted(self.counts.items())), "items": self.items, "retained": len(self.items), "omitted": sum(self.counts.values()) - len(self.items)}


def identity(row):
    r = row["raw"]
    return {"id": r["id"][:128], "task": r["task"], "split": r["split"], "role": row["role"]}


def near_copies(rows, findings, config):
    index = defaultdict(list); stored = []; comparisons = 0; skipped = Counter(); sampled = 0; pairs = set()
    for row in rows:
        for text in row["parts"]:
            if len(stored)>=config.max_parts:skipped["part_budget"]+=1;continue
            words = TOKEN.findall(text.casefold())
            if len(words) < config.min_words:skipped["short_exact_only"] += 1;continue
            if len(words) > config.max_words:skipped["oversize_words"] += 1;continue
            hashes = {digest(" ".join(words[i:i+5])) for i in range(len(words)-4)}
            if len(hashes) > config.max_shingles:sampled += 1
            hashes = frozenset(sorted(hashes)[:config.max_shingles]); anchors = sorted(hashes)[:config.anchors]
            candidates = sorted({i for h in anchors for i in index[h]})
            for i in candidates:
                old, other = stored[i]
                if row["role"] == other["role"]:continue
                if comparisons >= config.max_comparisons:skipped["comparison_budget"] += 1;break
                comparisons += 1
                score = len(hashes & old) / len(hashes | old)
                pair = tuple(sorted((row["raw"]["id"], other["raw"]["id"])))
                if score >= config.threshold and pair not in pairs:
                    pairs.add(pair);findings.add("near_source_copy", left=identity(other), right=identity(row), sampled_jaccard=round(score, 6))
            at = len(stored);stored.append((hashes, row))
            for h in anchors:
                if len(index[h]) < config.max_posting:index[h].append(at)
                else:skipped["posting_budget"] += 1
    return {"config": config.__dict__, "comparisons": comparisons, "sampled_parts": sampled, "skipped": dict(skipped), "coverage": "heuristic_bounded_not_exhaustive"}


def features(row):
    text = "\n".join(row["parts"]) or row["rendered"]
    return [math.log1p(len(text)), math.log1p(len(TOKEN.findall(text))), math.log1p(text.count("\n")),
            sum(c.isdigit() for c in text) / max(1, len(text)), len(row["keys"])] + [math.log1p(len(t)) for t in row["texts"][:8]]


def surface_probe(rows, eval_role):
    if eval_role is None:return {"status": "not_requested", "fit_split": "train", "allowed_evaluation": ["dev", "dev_conf"]}
    if eval_role not in {"dev", "dev_conf"}:raise ValueError("surface evaluation must explicitly be dev or dev_conf")
    result = [];remaining=5_000;omitted=0
    for task in sorted({r["raw"]["task"] for r in rows})[:128]:
        fit = [r for r in rows if r["raw"]["task"] == task and r["role"] == "train"]
        test = [r for r in rows if r["raw"]["task"] == task and (r["raw"]["split"] == "dev" if eval_role == "dev" else r["role"] == eval_role)]
        omitted+=max(0,len(test)-remaining);test=test[:remaining];remaining-=len(test)
        if not fit or not test:
            result.append({"task": task, "status": "no_train_or_requested_dev", "train": len(fit), "evaluation": len(test)});continue
        width = max(len(features(r)) for r in fit)
        def vector(r):return (features(r) + [0] * width)[:width]
        x = [vector(r) for r in fit]
        means = [sum(v[j] for v in x)/len(x) for j in range(width)]
        scales = [max(1e-9, math.sqrt(sum((v[j]-means[j])**2 for v in x)/len(x))) for j in range(width)]
        counts = Counter(r["label"] for r in fit); major = min(counts, key=lambda k: (-counts[k], k))
        centroids={k:[0.0]*width for k in counts}
        for v,r in zip(x,fit):
            for j in range(width):centroids[r['label']][j]+=(v[j]-means[j])/scales[j]/counts[r['label']]
        correct = majority = 0
        for row in test:
            v = [(a-m)/s for a,m,s in zip(vector(row),means,scales)]
            predicted = min(centroids, key=lambda k: (sum((a-b)**2 for a,b in zip(v,centroids[k])), -counts[k], k))
            correct += predicted == row["label"];majority += major == row["label"]
        result.append({"task":task,"status":"evaluated","train":len(fit),"evaluation":len(test),"correct":correct,
                       "accuracy":correct/len(test),"train_majority_accuracy":majority/len(test),"target":"option_index",
                       "shortcut_warning":len(test)>=10 and correct/len(test)-majority/len(test)>=.2,
                       "training_label_positions":dict(sorted(counts.items())),"train_feature_means":means,"train_feature_scales":scales})
    return {"status":"evaluated","fit_split":"train","evaluation_role":eval_role,"method":"train_standardized_nearest_label_position_centroid",
            "features":["source_length","word_count","line_count","digit_fraction","option_count","first_eight_ordered_option_lengths"],
            "evaluation_budget":5_000,"evaluation_rows_omitted":omitted,"task_budget":128,"tasks_omitted":max(0,len({r['raw']['task'] for r in rows})-128),
            "interpretation":"Surface shortcut diagnostic only, not model quality, calibration or certification; dev reuse can bias later decisions.","tasks":result}


def audit_rows(raw_rows, *, catalog=None, surface_eval=None, near_config=NearConfig(), unseal_locked=False, access_recorded=False, alpha=.05, delta=.1, n_min=20, start_errors=0):
    if type(unseal_locked) is not bool or type(access_recorded) is not bool:raise ValueError("explicit boolean access flags required")
    if not isinstance(near_config,NearConfig):raise ValueError("validated near-copy configuration required")
    if unseal_locked and not access_recorded:raise ValueError("explicit unsealing needs durable access log")
    findings = Findings();rows=[];locked=0;invalid=0;seen=defaultdict(dict); exact=defaultdict(dict);labels={}
    for line, raw in enumerate(raw_rows, 1):
        if line>MAX_ROWS:raise ValueError('row budget exceeded')
        if isinstance(raw, dict) and raw.get("split") == "locked_test" and not unseal_locked:
            locked += 1;continue  # No input/gold/story/task reads on this branch.
        if len(rows) + invalid >= MAX_ROWS:raise ValueError("row budget exceeded")
        try:row=parse_row(raw,catalog)
        except (ValueError, TypeError, UnicodeError, OverflowError):
            invalid+=1;findings.add("invalid_row",line=line);continue
        rows.append(row);r=row["raw"]
        meta=r.get('meta') if isinstance(r.get('meta'),dict) else {}
        groups=[(k,v) for k,v in meta.items() if k in {"group_id","family_id","template_family_id"} and isinstance(v,str) and v]
        if isinstance(r.get('group_id'),str) and r['group_id']:groups.append(('group_id',r['group_id']))
        for field,value in [("id",r["id"]),("story_id",r["story_id"])]+groups:
            try:value.encode('utf-8')
            except UnicodeError:findings.add('invalid_group_identifier',row=identity(row));continue
            old=seen[field].get(value)
            if old is not None and (field=="id" or old["raw"]["split"]!=r["split"]):findings.add("duplicate_id" if field=="id" else "cross_split_"+field,left=identity(old),right=identity(row),group_sha256=digest(value))
            else:seen[field][value]=row
        for kind,texts in (("source",row["parts"]),("rendered_state",[row["rendered"]])):
            for text in texts:
                if not text.strip():continue
                h=digest(text);key=h if kind=='source' else (r["task"],h);old=exact[kind].get(key)
                if old is not None and old["role"]!=row["role"]:findings.add("exact_"+kind+"_copy",left=identity(old),right=identity(row),sha256=h)
                else:exact[kind][key]=row
                if kind=='rendered_state':
                    label_key=(r['task'],h,digest(canonical([row['keys'],row['texts']])))
                    previous=labels.get(label_key)
                    if previous is not None and previous['label_key']!=row['label_key']:
                        findings.add('conflicting_hard_labels',left=identity(previous),right=identity(row),rendered_state_sha256=h)
                    else:labels[label_key]=row
        if any(LEFTOVER.search(t) for t in [row["rendered"]]+row["texts"]):findings.add("template_leftover",row=identity(row))
    fit_domains={r['raw']['domain_id'] for r in rows if r['raw']['split'] in {'train','dev'}}
    fit_langs={r['raw']['lang'] for r in rows if r['raw']['split'] in {'train','dev'}}
    for row in rows:
        r=row['raw']
        if r['split']=='heldout_domain' and r['domain_id'] in fit_domains:findings.add('heldout_domain_overlap',row=identity(row),domain_sha256=digest(r['domain_id']))
        if r['split']=='heldout_language' and r['lang'] in fit_langs:findings.add('heldout_language_overlap',row=identity(row),language=r['lang'])
    counts={};
    for field in ("split","lang","domain_id","task","label_source","generator_version"):
        counts[field]=dict(sorted(Counter(r["raw"][field] for r in rows).items()))
    counts["role"]=dict(sorted(Counter(r["role"] for r in rows).items()))
    counts["task_labels"]={t:dict(sorted(Counter(r["label_key"] for r in rows if r["raw"]["task"]==t).items())) for t in sorted(counts["task"])}
    # Metadata is NOT a model feature. Perfect metadata/round associations are
    # warnings about construction/split design, not evidence the model saw it.
    metadata=[]
    for task in sorted(counts["task"]):
        fit=[r for r in rows if r["raw"]["task"]==task and r["role"]=="train"]
        for field in ("generator_version","lang","domain_id","seed"):
            cells=defaultdict(Counter)
            for r in fit:cells[str(r["raw"][field])][r["label_key"]]+=1
            if len(cells)>1 and len(cells)<=100 and len({r['label_key'] for r in fit})>1 and all(sum(c.values())>=2 and len(c)==1 for c in cells.values()):
                metadata.append({"task":task,"field":field,"values":len(cells),"train_examples":len(fit),"warning":"perfect_train_label_association; metadata not used by probe"})
    target=calibration_target(alpha,delta,n_min,start_errors)
    ready=[]
    for task in sorted(counts["task"]):
        cal=sum(r["raw"]["task"]==task and r["role"]=="dev_cal" for r in rows)
        conf=sum(r["raw"]["task"]==task and r["role"]=="dev_conf" for r in rows)
        ready.append({"task":task,"dev_cal":cal,"dev_conf":conf,"dev_cal_unique_stories":len({r['raw']['story_id'] for r in rows if r['raw']['task']==task and r['role']=='dev_cal'}),
                      "dev_conf_unique_stories":len({r['raw']['story_id'] for r in rows if r['raw']['task']==task and r['role']=='dev_conf'}),
                      "configured_start":target,"shortfall":max(0,target-conf),"certified":False})
    near=near_copies(rows,findings,near_config)
    blocking={k:n for k,n in findings.counts.items() if k in {'invalid_row','invalid_group_identifier','duplicate_id','conflicting_hard_labels','heldout_domain_overlap','heldout_language_overlap'} or k.startswith('cross_split_') or k.startswith('exact_')}
    surface=surface_probe(rows,surface_eval)
    warnings=bool(findings.counts or metadata or any(t.get('shortcut_warning') for t in surface.get('tasks',[])))
    return {"protocol":PROTOCOL,"dataset_contract":"hive.laya-dataset/v1","status":"blocked" if blocking else "review_required" if warnings else "checks_passed_with_limits", "blocking_findings":blocking,"valid_rows":len(rows),"invalid_rows":invalid,
        "locked_test":{"access":"explicit_unsealed" if unseal_locked else "sealed_skipped","skipped_rows":locked,
                       "input_or_gold_transformed":unseal_locked,"byte_level_secrecy":False},
        "counts":counts,"findings":findings.report(),"near_copy":near,"metadata_shortcuts":metadata,
        "surface_probe":surface,"calibration_readiness":{"alpha":alpha,"delta_family":delta,"score_delta":delta/2,"n_min":n_min,"start_errors":start_errors,"tasks":ready,"meaning":"count lower bound only; no threshold certification or model metrics",
            "row_correlation_warning":"Several paraphrases/translations from one story are correlated; example counts do not establish independent acted items or domain validity."},
        "limitations":["Default locked rows are envelope-decoded then discarded; never rendered or used for audits/labels/metrics.",
            "Locked split isolation/duplicates are unexamined while sealed; group isolation covers only explicit id/story_id/group_id/meta family identifiers.",
            "Source duplication excludes fixed options/catalog/question instructions; complete rendered-state equality remains separately checked.",
            "Near-copy work is capped and sampled; no claim of exhaustive leakage detection.","No source text is emitted; identifiers/digests/counts may still be sensitive.",
            "Preflight does not certify dataset correctness, model quality, domain validation or local routing certainty."]}


def strict_json(text):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise ValueError("duplicate JSON key")
            result[k]=v
        return result
    def constant(_):raise ValueError("nonfinite JSON")
    return json.loads(text,object_pairs_hook=pairs,parse_constant=constant)


def load_rows(paths, *, unseal_locked=False):
    hashes=[];total=0;rows=[]
    for path in paths:
        h=hashlib.sha256();count=0
        with Path(path).open("rb") as f:
            while True:
                line=f.readline(MAX_LINE+1)
                if not line:break
                total+=len(line)
                if len(line)>MAX_LINE or total>MAX_BYTES:raise ValueError("input byte/line budget exceeded")
                h.update(line)
                if line.strip():
                    row=strict_json(line.decode("utf-8"))
                    if isinstance(row,dict) and row.get('split')=='locked_test' and not unseal_locked:row={'split':'locked_test'}
                    rows.append(row);count+=1
                if len(rows)>MAX_ROWS:raise ValueError("input row budget exceeded")
        hashes.append({"name":Path(path).name,"sha256":h.hexdigest(),"rows":count})
    return rows,hashes


def markdown(report):
    lines=["# Training data preflight", "", f"Valid rows: {report['valid_rows']}; invalid rows: {report['invalid_rows']}.",
           f"Locked-test access: `{report['locked_test']['access']}`; skipped: {report['locked_test']['skipped_rows']}.",
           "", "This is a bounded data diagnostic. It reports no trained-model metrics or certification.", "", "## Findings", ""]
    lines += [f"- `{kind}`: {n}" for kind,n in report['findings']['counts'].items()] or ["- No findings in the checks performed."]
    lines += ["", "## Calibration count readiness", "", "| Task | dev_cal | dev_conf | Configured minimum | Shortfall |", "|---|---:|---:|---:|---:|"]
    lines += [f"| `{x['task']}` | {x['dev_cal']} | {x['dev_conf']} | {x['configured_start']} | {x['shortfall']} |" for x in report['calibration_readiness']['tasks']]
    lines += ['',report['calibration_readiness']['row_correlation_warning'],'','## Surface-only diagnostic','']
    if report['surface_probe']['status']=='evaluated':
        lines += ['Fit: train only; evaluation: `'+report['surface_probe']['evaluation_role']+'`. These are shortcut diagnostics, not trained-model metrics.','', '| Task | Dev rows | Surface accuracy | Train-majority accuracy | Shortcut warning |','|---|---:|---:|---:|---|']
        lines += [f"| `{t['task']}` | {t['evaluation']} | {t['accuracy']:.3f} | {t['train_majority_accuracy']:.3f} | {t['shortcut_warning']} |" for t in report['surface_probe']['tasks'] if t['status']=='evaluated']
    else:lines += ['Not requested. Use explicit `--surface-eval dev` or `dev_conf`; no locked/heldout evaluation is available.']
    lines += ['','Near-copy coverage: `'+report['near_copy']['coverage']+'`; comparisons: '+str(report['near_copy']['comparisons'])+'. Limits/skips are recorded in JSON.']
    lines += ["", "## Limits", ""]+["- "+x for x in report['limitations']]
    return "\n".join(lines)+"\n"


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths",type=Path,nargs="+");p.add_argument("--json-out",type=Path,required=True);p.add_argument("--markdown-out",type=Path,required=True)
    p.add_argument("--catalog",type=Path,help="explicit JSON task -> option key list")
    p.add_argument("--surface-eval",choices=("dev","dev_conf"),help="explicit train-fitted surface probe; never locked or heldout")
    p.add_argument("--unseal-locked",action="store_true");p.add_argument("--access-log",type=Path)
    p.add_argument("--alpha",type=float,default=.05);p.add_argument("--delta",type=float,default=.1);p.add_argument("--n-min",type=int,default=20);p.add_argument("--start-errors",type=int,default=0)
    args=p.parse_args(argv)
    if args.json_out.resolve()==args.markdown_out.resolve() or any(out.resolve()==inp.resolve() for out in (args.json_out,args.markdown_out) for inp in args.paths):p.error("outputs must be distinct from input")
    if args.json_out.exists() or args.markdown_out.exists():p.error("refusing to replace an existing report")
    if args.unseal_locked:
        if args.access_log is None:p.error("unsealing requires --access-log")
        protected=args.paths+[args.json_out,args.markdown_out]+([args.catalog] if args.catalog else [])
        if any(args.access_log.resolve()==x.resolve() for x in protected):p.error("access log must be separate")
        event={"event":"training_data_audit_locked_access","timestamp":datetime.datetime.now(datetime.timezone.utc).isoformat(),"protocol":PROTOCOL,"files":[str(x.resolve()) for x in args.paths],"purpose":"explicit data preflight, no training/model metrics"}
        args.access_log.parent.mkdir(parents=True,exist_ok=True)
        with args.access_log.open("a",encoding="utf-8") as f:f.write(canonical(event)+"\n");f.flush();os.fsync(f.fileno())
    rows,hashes=load_rows(args.paths,unseal_locked=args.unseal_locked)
    catalog=strict_json(args.catalog.read_text(encoding="utf-8")) if args.catalog else None
    if catalog is not None and not isinstance(catalog,dict):p.error("catalog must be a task mapping")
    report=audit_rows(rows,catalog=catalog,surface_eval=args.surface_eval,unseal_locked=args.unseal_locked,access_recorded=args.access_log is not None,alpha=args.alpha,delta=args.delta,n_min=args.n_min,start_errors=args.start_errors)
    report["files"]=hashes
    payload=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n";md=markdown(report)
    if max(len(payload.encode()),len(md.encode()))>MAX_REPORT_BYTES:raise ValueError("report byte budget exceeded")
    for path,body in ((args.json_out,payload),(args.markdown_out,md)):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("x",encoding="utf-8") as f:f.write(body)
    print(canonical({"protocol":PROTOCOL,"valid_rows":report["valid_rows"],"findings":report["findings"]["counts"],"json_out":str(args.json_out),"markdown_out":str(args.markdown_out)}))
    return 1 if report["blocking_findings"] else 0


if __name__ == "__main__":raise SystemExit(main())
