"""Prepare a separate bounded training JSONL; never train, evaluate or overwrite input.

Choice permutations are train-only and remap hard/soft targets by option key.
They make fixed-choice tasks per-example: serving must supply explicit options.
Replay is opt-in, training-only, structurally adjudicated and family-selected.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import tempfile
import datetime
import stat
import uuid
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

PROTOCOL = "ul.layago-training-preparation/v1"
ADJUDICATION_PROTOCOL = "training-label-adjudication-v1"
SPLITS = {"train", "dev", "heldout_domain", "heldout_language", "locked_test"}
LABEL_SOURCES = {"spec", "template", "llm_paraphrase", "llm_translation", "claude_author"}
SOURCE_IDS = {"source_id", "document_id", "source_file_id", "source_document_id"}
SOURCE_HASHES = {"source_sha256", "document_sha256", "original_sha256", "source_file_sha256", "original_document_sha256"}
FAMILY_IDS = {"story_id", "family_id", "source_family_id", "parent_story_id", "contract_id"}
MAX_FILE_BYTES = 256 << 20
MAX_LINE_BYTES = 1 << 20
MAX_OUTPUT_BYTES = 512 << 20
MAX_ACCESS_LOG_BYTES = 16 << 20
ACCESS_PROTOCOL = "ul.layago-protected-data-access/v1"
ACCESS_PURPOSE = "training_preparation_source_protection_not_fit_or_evaluation"
SHA = re.compile(r"[0-9a-fA-F]{64}\Z")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _pairs(values):
    result = {}
    for key, value in values:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"nonfinite JSON value {value}")


def _nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def choice_keys(row):
    options = row.get("options")
    if not isinstance(options, list) or not 2 <= len(options) <= 128:
        raise ValueError("choice options must contain 2..128 keyed choices")
    keys = []
    for option in options:
        if isinstance(option, str):
            key = option
        elif isinstance(option, dict):
            key = option.get("key")
            if "text" in option and not isinstance(option["text"], str):
                raise ValueError("choice option text must be a string")
        else:
            raise ValueError("choice options must be strings or objects")
        keys.append(_nonempty(key, "choice key"))
    if len(set(keys)) != len(keys):
        raise ValueError("choice keys must be unique")
    return keys


def validate_row(row):
    """Validate transform-sensitive fields; Go validate remains authoritative."""
    if not isinstance(row, dict):
        raise ValueError("each row must be a JSON object")
    for key in ("id", "instance", "task", "task_type", "label_source", "split",
                "lang", "domain_id", "story_id", "generator_version"):
        _nonempty(row.get(key), key)
    if row["split"] not in SPLITS or row["instance"] not in {"Q", "R", "A"}:
        raise ValueError("unknown split or instance")
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", row["task"]) or not row["task"].startswith(row["instance"].lower()+"_"):
        raise ValueError("task must belong to its instance")
    if row["label_source"] not in LABEL_SOURCES:
        raise ValueError("unsupported label_source; preserve existing dataset vocabulary")
    if type(row.get("seed")) is not int:
        raise ValueError("row seed must be an integer")
    if not isinstance(row.get("input"), dict) or not isinstance(row["input"].get("question"), str):
        raise ValueError("input must contain a string question")
    if "meta" in row and not isinstance(row["meta"], dict):
        raise ValueError("meta must be an object when supplied")
    kind, label = row["task_type"], row.get("label")
    if kind == "choice":
        n = len(choice_keys(row))
    elif kind == "score":
        options = row.get("options")
        if not isinstance(options, list) or len(options) < 2:
            raise ValueError("score options need at least two ordered levels")
        n = len(options)
    elif kind == "yes_no":
        if type(label) is not bool:
            raise ValueError("yes_no labels must be booleans")
        if row.get("options") is not None and (not isinstance(row["options"], list) or len(row["options"]) not in (0, 2)):
            raise ValueError("yes_no criteria are ordered false/true")
        n = 2
        label = int(label)
    else:
        raise ValueError("unknown task_type")
    if kind != "yes_no" and (type(label) is not int or not 0 <= label < n):
        raise ValueError("hard label must be an in-range integer index")
    soft = row.get("label_dist")
    if soft is not None:
        if row["split"] != "train" or not isinstance(soft, list) or len(soft) != n:
            raise ValueError("soft targets must match training choices only")
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in soft):
            raise ValueError("soft targets must be finite nonnegative numbers")
        if abs(sum(soft)-1) > 1e-6 or soft[label] < max(soft):
            raise ValueError("soft targets must sum to one and agree with the hard label")
    if len(canonical(row)) > MAX_LINE_BYTES:
        raise ValueError("canonical row exceeds the bounded row size")


def _append_access(path, event, forbidden_paths=()):
    """Append/fsync one bounded record; unrelated files and aliases are refused."""
    if not isinstance(path, (str, os.PathLike)) or not str(path).strip():
        raise ValueError("an explicit protected-access log path is required")
    path = Path(path)
    resolved = path.resolve()
    forbidden = [Path(value) for value in forbidden_paths]
    if any(resolved == value.resolve() for value in forbidden):
        raise ValueError("access log must not alias input or output")
    if path.is_symlink():
        raise ValueError("access log must not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("access log must be a private regular file")
        for other in forbidden:
            try:
                other_info = other.stat()
            except FileNotFoundError:
                continue
            if (info.st_dev, info.st_ino) == (other_info.st_dev, other_info.st_ino):
                raise ValueError("access log must not alias input or output")
        if info.st_size > MAX_ACCESS_LOG_BYTES:
            raise ValueError("access log exceeds its bounded size")
        os.lseek(fd, 0, os.SEEK_SET)
        existing = bytearray()
        while len(existing) < info.st_size:
            block = os.read(fd, min(65536, info.st_size-len(existing)))
            if not block:
                break
            existing.extend(block)
        for line in bytes(existing).splitlines():
            if not line.strip():
                continue
            try:
                previous = json.loads(line.decode("utf-8"), object_pairs_hook=_pairs,
                                      parse_constant=_constant)
            except (ValueError, UnicodeError) as error:
                raise ValueError("existing file is not a protected-access log") from error
            if not isinstance(previous, dict) or previous.get("protocol") != ACCESS_PROTOCOL:
                raise ValueError("existing file is not a protected-access log")
        if existing and not existing.endswith(b"\n"):
            raise ValueError("existing access log has an incomplete final record")
        record = {"protocol": ACCESS_PROTOCOL, "purpose": ACCESS_PURPOSE,
                  "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "gold_payload_in_log": False, **event}
        encoded = canonical(record)+b"\n"
        if len(encoded) > 65536 or info.st_size+len(encoded) > MAX_ACCESS_LOG_BYTES:
            raise ValueError("access log record exceeds its bounded size")
        remaining = memoryview(encoded)
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise OSError("access log write did not complete")
            remaining = remaining[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    return record


def _access_start(path, scope, input_paths=(), forbidden_paths=()):
    request_id = uuid.uuid4().hex
    _append_access(path, {"event": "authorized_read_started", "request_id": request_id,
                         "scope": scope, "input_paths": [str(p) for p in input_paths],
                         "source_hash_status": "not_read_yet"}, forbidden_paths)
    return request_id


def _access_hash(path, request_id, scope, snapshots, forbidden_paths=()):
    return _append_access(path, {"event": "source_snapshot_hashed",
                                 "request_id": request_id, "scope": scope,
                                 "snapshots": snapshots}, forbidden_paths)


def read_jsonl(path, max_rows, *, protected_access_log=None, forbidden_paths=()):
    """Without a log, inspect split envelopes only before refusing locked rows.

    Explicit access is durably recorded before opening the dataset or hashing
    it. It authorizes source protection/validation, never fitting/evaluation.
    """
    path = Path(path)
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"{path}: input exceeds {MAX_FILE_BYTES} bytes")
    forbidden = [path, *forbidden_paths]
    request_id = None
    if protected_access_log is not None:
        request_id = _access_start(protected_access_log, "jsonl_file_read",
                                   [path], forbidden)
    raw = path.read_bytes()
    if request_id is not None:
        _access_hash(protected_access_log, request_id, "jsonl_file_read",
                     [{"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                       "bytes": len(raw)}], forbidden)
    rows = []
    for line_no, line in enumerate(raw.splitlines(), 1):
        if len(line) > MAX_LINE_BYTES:
            raise ValueError(f"{path}:{line_no}: row exceeds {MAX_LINE_BYTES} bytes")
        if not line.strip():
            continue
        try:
            # This first decode observes only split; no label validation,
            # canonical row hashing or source/family indexing happens yet.
            row = json.loads(line.decode("utf-8"))
            if isinstance(row, dict) and row.get("split") == "locked_test" and request_id is None:
                raise ValueError("locked_test source protection requires --protected-access-log; no logical locked-row processing was performed")
            rows.append((line_no, line))
            if len(rows) > max_rows:
                raise ValueError(f"more than {max_rows} input rows")
        except (ValueError, TypeError, UnicodeError) as error:
            raise ValueError(f"{path}:{line_no}: {error}") from error
    validated = []
    for line_no, line in rows:
        try:
            row = json.loads(line.decode("utf-8"), object_pairs_hook=_pairs,
                             parse_constant=_constant)
            validate_row(row)
        except (ValueError, TypeError, UnicodeError) as error:
            raise ValueError(f"{path}:{line_no}: {error}") from error
        validated.append(row)
    return validated, raw


def _walk(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key, child
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def identities(row):
    """Conservative declared-family/source/input fingerprints, not semantic dedup."""
    result = {("family", row["story_id"]), ("input", digest(row["input"]))}
    for key, value in _walk(row):
        if key in FAMILY_IDS:
            result.add(("family", _nonempty(value, key)))
        elif key in SOURCE_IDS:
            result.add(("source_id", _nonempty(value, key)))
        elif key in SOURCE_HASHES:
            if not isinstance(value, str) or not SHA.fullmatch(value):
                raise ValueError(f"{key} must be a source SHA256")
            result.add(("source_sha256", value.lower()))
    def declared_sources(value, source_context=False):
        if isinstance(value, dict):
            if source_context and "sha256" in value:
                sha = value["sha256"]
                if not isinstance(sha, str) or not SHA.fullmatch(sha):
                    raise ValueError("source ledger sha256 must be a SHA256")
                result.add(("source_sha256", sha.lower()))
            for key, child in value.items():
                declared_sources(child, source_context or key in {
                    "source", "sources", "original", "originals",
                    "source_versions", "document", "documents", "evidence"})
        elif isinstance(value, list):
            for child in value:
                declared_sources(child, source_context)
    declared_sources(row)
    inp = row["input"]
    texts = [inp.get(key) for key in ("document", "source", "source_text", "original_text")]
    evidence = inp.get("evidence")
    if isinstance(evidence, str):
        texts.append(evidence)
    elif isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                texts.append(item.get("text"))
    for text in texts:
        if isinstance(text, str) and text.strip():
            normalized = " ".join(text.split())
            result.add(("source_text", hashlib.sha256(normalized.encode()).hexdigest()))
    return result


def _isolation(rows):
    owners = {}
    fit_domains, fit_langs = set(), set()
    held_domains, held_langs = set(), set()
    for row in rows:
        split = row["split"]
        for token in identities(row):
            previous = owners.setdefault(token, split)
            if previous != split:
                raise ValueError(f"source/story/input overlap across {previous} and {split}")
        if split in {"train", "dev"}:
            fit_domains.add(row["domain_id"])
            fit_langs.add(row["lang"].split("-")[0].lower())
        elif split == "heldout_domain":
            held_domains.add(row["domain_id"])
        elif split == "heldout_language":
            held_langs.add(row["lang"].split("-")[0].lower())
    if fit_domains & held_domains or fit_langs & held_langs:
        raise ValueError("held-out domain/language also appears in fitting data")


def adjudication_binding(row):
    """Seal the reviewed labels/input/source ledger, without changing producer provenance."""
    fields = ("instance", "task", "task_type", "input", "options", "label",
              "label_dist", "label_source", "provenance", "split", "story_id",
              "domain_id", "lang")
    return digest({"row": {key: row.get(key) for key in fields},
                   "source_bindings": sorted(identities(row))})


def _adjudicated(row, owner):
    review = row.get("meta", {}).get("replay_adjudication")
    if not isinstance(review, dict) or review.get("protocol") != ADJUDICATION_PROTOCOL or review.get("status") != "adjudicated":
        raise ValueError("replay requires an explicit adjudication receipt")
    if review.get("owner_id") != owner:
        raise ValueError("replay ownership does not match --replay-owner")
    for field in ("reviewer_id", "record_id"):
        _nonempty(review.get(field), field)
    if review.get("binding_sha256") != adjudication_binding(row):
        raise ValueError("replay adjudication does not bind the current labels/source")


def _lineage(row, kind, seed, *, _clone=True, **extra):
    if "training_preparation" in row.get("meta", {}):
        raise ValueError("already prepared rows require an explicit new reviewed baseline")
    original_sha = digest(row)
    new = copy.deepcopy(row) if _clone else dict(row)
    new["meta"] = dict(row.get("meta", {}))
    new["seed"] = seed
    new.setdefault("meta", {})["training_preparation"] = {
        "protocol": PROTOCOL, "kind": kind, "seed": seed,
        "original_id": row["id"], "original_row_sha256": original_sha,
        "original_seed": row["seed"], "original_generator_version": row["generator_version"], **extra}
    return new


def _next_order(order):
    order = list(order)
    i = len(order)-2
    while i >= 0 and order[i] >= order[i+1]:
        i -= 1
    if i < 0:
        return None
    j = len(order)-1
    while order[j] <= order[i]:
        j -= 1
    order[i], order[j] = order[j], order[i]
    order[i+1:] = reversed(order[i+1:])
    return tuple(order)


def _permutation_previews(row, seed, permutations):
    if row["split"] != "train" or row["task_type"] != "choice" or not permutations:
        return
    keys = choice_keys(row)
    original = tuple(range(len(keys)))
    wanted = min(permutations, math.factorial(len(keys))-1)
    orders, seen = [], {original}
    for attempt in range(64):
        order = tuple(sorted(original, key=lambda i: digest(
            [PROTOCOL, seed, row["id"], attempt, keys[i]])))
        if order not in seen:
            orders.append(order); seen.add(order)
        if len(orders) == wanted:
            break
    # Bounded deterministic collision fallback; no factorial enumeration.
    order = original
    while len(orders) < wanted:
        order = _next_order(order)
        if order is None:
            break
        if order not in seen:
            orders.append(order); seen.add(order)
    for index, order in enumerate(orders):
        variant = _lineage(row, "choice_option_permutation", seed, _clone=False,
                           variant=index, option_keys_before=keys,
                           option_keys_after=[keys[i] for i in order])
        variant["id"] = row["id"]+"::perm-"+digest(
            [PROTOCOL, seed, digest(row), list(order)])[:24]
        variant["options"] = [row["options"][i] for i in order]
        variant["label"] = order.index(row["label"])
        if row.get("label_dist") is not None:
            variant["label_dist"] = [row["label_dist"][i] for i in order]
        yield variant


def permute_choices(row, seed, permutations):
    return [copy.deepcopy(preview) for preview in _permutation_previews(row, seed, permutations)]


def _families(rows):
    parent = list(range(len(rows)))
    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]; n = parent[n]
        return n
    owners = {}
    for index, row in enumerate(rows):
        for token in identities(row):
            if token in owners:
                a, b = find(index), find(owners[token])
                parent[a] = b
            else:
                owners[token] = index
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[find(index)].append(row)
    return list(groups.values())


def prepare(rows, *, seed=20261008, permutations=1, replay=(),
            replay_owner=None, replay_fraction=.2, max_rows=100000,
            protected_access_log=None, baseline_byte_count=None):
    """Prepare independent rows; locked input requires a durable access log.

    In-memory logging covers processing here, not caller prior reads.
    """
    if type(seed) is not int or not 0 <= seed <= 2**63-1:
        raise ValueError("preparation seed must be an integer in 0..2^63-1")
    if type(permutations) is not int or not 0 <= permutations <= 8:
        raise ValueError("permutations must be an integer in 0..8")
    if type(max_rows) is not int or not 1 <= max_rows <= 1000000:
        raise ValueError("max_rows must be in 1..1000000")
    if type(replay_fraction) not in (int, float) or not math.isfinite(replay_fraction) or not 0 <= replay_fraction <= .5:
        raise ValueError("replay fraction must be finite in 0..0.5")
    rows, replay = list(rows), list(replay)
    access = None
    if any(isinstance(row, dict) and row.get("split") == "locked_test"
           for row in [*rows, *replay]):
        if protected_access_log is None:
            raise ValueError("locked_test processing requires an explicit protected_access_log")
        request_id = _access_start(protected_access_log, "in_memory_processing_only")
        # The caller supplied decoded rows already. This receipt does NOT
        # establish how/when the caller previously read/deserialized them.
        snapshot = {"representation": "canonical_supplied_rows", "sha256": digest([rows, replay])}
        access = _access_hash(protected_access_log, request_id,
                              "in_memory_processing_only", [snapshot])
    if not rows or len(rows) > max_rows or len(replay) > max_rows:
        raise ValueError("nonempty bounded baseline and replay required")
    ids = set()
    for row in [*rows, *replay]:
        validate_row(row)
        if row["id"] in ids:
            raise ValueError("row IDs must be unique across baseline and replay")
        ids.add(row["id"])
    _isolation(rows)
    protected = {token for row in rows if row["split"] != "train" for token in identities(row)}
    baseline = {token for row in rows for token in identities(row)}
    held_domains = {r["domain_id"] for r in rows if r["split"] == "heldout_domain"}
    held_langs = {r["lang"].split("-")[0].lower() for r in rows if r["split"] == "heldout_language"}
    if replay:
        _nonempty(replay_owner, "explicit replay owner")
    for row in replay:
        if row["split"] != "train":
            raise ValueError("replay may contain only explicitly adjudicated train rows")
        _adjudicated(row, replay_owner)
        if identities(row) & protected:
            raise ValueError("replay overlaps protected source/story/input")
        if identities(row) & baseline:
            raise ValueError("replay source/story/input already belongs to baseline")
        if row["domain_id"] in held_domains or row["lang"].split("-")[0].lower() in held_langs:
            raise ValueError("replay would leak a held-out domain/language")
    if baseline_byte_count is not None and (type(baseline_byte_count) is not int or baseline_byte_count < 0):
        raise ValueError("baseline_byte_count must be a nonnegative integer")
    estimated_bytes = (sum(len(canonical(row))+1 for row in rows)
                       if baseline_byte_count is None else baseline_byte_count)
    if estimated_bytes > MAX_OUTPUT_BYTES:
        raise ValueError("baseline exceeds the output byte budget")
    def charge(preview):
        nonlocal estimated_bytes
        size = len(canonical(preview))+1
        if size > MAX_LINE_BYTES:
            raise ValueError("derived row exceeds the bounded row size")
        if estimated_bytes+size > MAX_OUTPUT_BYTES:
            raise ValueError("preparation additions exceed the output byte budget; reduce augmentation/replay")
        estimated_bytes += size
    output = copy.deepcopy(rows)
    for row in rows:
        for preview in _permutation_previews(row, seed, permutations):
            if len(output)+1 > max_rows:
                raise ValueError("choice augmentation would exceed max_rows; reduce variants")
            charge(preview)  # Check bytes before deep-copying/accumulating each addition.
            output.append(copy.deepcopy(preview))
    nonreplay_train = sum(r["split"] == "train" for r in output)
    fraction = Fraction(str(replay_fraction))
    replay_cap = min(max_rows-len(output), int(nonreplay_train*fraction/(1-fraction)))
    families = _families(replay)
    families.sort(key=lambda group: digest([PROTOCOL, seed, sorted(r["id"] for r in group)]))
    selected, skipped = [], []
    for family in families:
        if len(selected)+len(family) > replay_cap:
            skipped.append({"story_ids": sorted({r["story_id"] for r in family}),
                            "rows": len(family), "reason": "whole_family_exceeds_remaining_budget"})
            continue
        for row in sorted(family, key=lambda r: r["id"]):
            copied = _lineage(row, "adjudicated_training_replay", seed, _clone=False)
            copied["id"] = row["id"]+"::replay-"+digest([PROTOCOL, seed, digest(row)])[:24]
            charge(copied)
            selected.append(copy.deepcopy(copied))
    output.extend(selected)
    output_ids = [r["id"] for r in output]
    if len(set(output_ids)) != len(output_ids):
        raise ValueError("derived row ID collision; baseline was not modified")
    catalogs = defaultdict(set)
    for row in output:
        if row["task_type"] == "choice":
            keys = choice_keys(row)
            texts = [option.get("text", "") if isinstance(option, dict) else ""
                     for option in row["options"]]
            catalogs[row["task"]].add(canonical(list(zip(keys, texts))))
    tasks = sorted(task for task, orders in catalogs.items() if len(orders) > 1)
    report = {"protocol": PROTOCOL, "seed": seed,
              "protected_access": access,
              "caller_prior_source_access_established": False, "baseline_rows": len(rows),
              "protected_rows_unchanged": sum(r["split"] != "train" for r in rows),
              "choice_variants": len(output)-len(rows)-len(selected),
              "per_example_choice_tasks": tasks, "replay_rows": len(selected),
              "replay_fraction_requested": replay_fraction,
              "replay_fraction_actual": len(selected)/(nonreplay_train+len(selected)) if nonreplay_train+len(selected) else 0,
              "replay_cap_rows": replay_cap, "replay_families_skipped": skipped,
              "output_rows": len(output), "max_rows": max_rows,
              "estimated_output_bytes": estimated_bytes, "max_output_bytes": MAX_OUTPUT_BYTES,
              "serving_warning": "Choice-order variants make task options per_example: callers must supply the explicit catalog/options; fixed-option inference must not be silently reused.",
              "limits": ["Adjudication metadata is structurally checked, not proof of human judgment.",
                         "Semantic paraphrase/translation leaks require explicit shared family/source IDs.",
                         "No training, calibration or locked-test evaluation performed. Run Go validate and inspect before training."]}
    return output, report


def write_output(path, raw_baselines, additions):
    """Publish a new complete file atomically; existing files are never replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    digestor = hashlib.sha256()
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="."+path.name+".prepare-", delete=False) as target:
            temp_name = target.name
            def write(chunk):
                nonlocal count
                count += len(chunk)
                if count > MAX_OUTPUT_BYTES:
                    raise ValueError(f"prepared output exceeds {MAX_OUTPUT_BYTES} bytes")
                target.write(chunk); digestor.update(chunk)
            for raw in raw_baselines:
                write(raw)
                if raw and not raw.endswith(b"\n"):
                    write(b"\n")
            for row in additions:
                write(canonical(row)+b"\n")
            target.flush(); os.fsync(target.fileno())
        os.link(temp_name, path)  # Exclusive atomic publication, even against a race.
        return {"output": str(path), "output_bytes": count,
                "output_sha256": digestor.hexdigest()}
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True,
                        help="Baseline JSONL; repeatable, preserved in source order")
    parser.add_argument("--out", type=Path, required=True, help="NEW prepared JSONL")
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--permutations", type=int, default=1, help="0..8 extra distinct train-choice orders per row")
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--replay", type=Path, action="append", default=[])
    parser.add_argument("--replay-owner", help="Explicit owner ID in adjudication receipts")
    parser.add_argument("--protected-access-log", type=Path,
                        help="Append-only durable authorization for locked source protection, not fitting/evaluation")
    parser.add_argument("--replay-fraction", type=float, default=.2,
                        help="Maximum replay fraction of final TRAIN rows, 0..0.5")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("output must be a new file; existing files are never overwritten")
    try:
        originals, raw_baselines, files = [], [], []
        forbidden = [*args.dataset, *args.replay, args.out]
        if sum(path.stat().st_size for path in [*args.dataset, *args.replay]) > MAX_FILE_BYTES:
            raise ValueError("combined dataset/replay files exceed the input byte budget")
        for path in args.dataset:
            rows, raw = read_jsonl(path, args.max_rows, protected_access_log=args.protected_access_log,
                                    forbidden_paths=forbidden)
            originals.extend(rows); raw_baselines.append(raw)
            files.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
        replay = []
        for path in args.replay:
            rows, raw = read_jsonl(path, args.max_rows, protected_access_log=args.protected_access_log,
                                    forbidden_paths=forbidden); replay.extend(rows)
            files.append({"replay_path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
        prepared, report = prepare(originals, seed=args.seed, permutations=args.permutations,
                                   replay=replay, replay_owner=args.replay_owner,
                                   replay_fraction=args.replay_fraction, max_rows=args.max_rows,
                                   protected_access_log=args.protected_access_log,
                                   baseline_byte_count=sum(len(raw)+(bool(raw) and not raw.endswith(b"\n")) for raw in raw_baselines))
        report.update(inputs=files)
        report.update(write_output(args.out, raw_baselines, prepared[len(originals):]))
    except (OSError, ValueError, TypeError, UnicodeError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
