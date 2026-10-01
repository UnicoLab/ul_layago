#!/usr/bin/env python3
"""Independent numpy reference for the layago M0 parity suite.

Needs only Python 3.10+ and numpy (no torch, transformers or tokenizers). It

1. builds a tiny synthetic Laya checkpoint under checkpoint/ in the published
   Laya layout: rl_agent_config.json, model.safetensors (random weights stored
   as a mix of F32, F16 and BF16), tokenizer/tokenizer.json,
   tokenizer/tokenizer_config.json and encoder/config.json;
2. re-implements, from the semantics of Hugging Face tokenizers (BPE with
   Metaspace/Replace, byte fallback and added tokens), transformers 4.57
   ModernBertModel (SDPA path) and laya 0.3.5 (build_sequence,
   DecisionModel, Agent.system_one), the tokenizer, the encoder forward pass,
   the decision head and the typed decoding, all in float64; and
3. writes the goldens the Go tests compare against: tokenizer_golden.json,
   encoder_golden.safetensors plus encoder_golden.json, and laya_golden.json.

The int8 goldens use the same deterministic per-row quantisation rule as
tensor.QuantizeRows and run the float64 forward pass on the dequantised
weights, so the Go int8 kernels are checked tightly and quantisation error is
reported separately.

Regenerate from the repository root with:

    python3 internal/layago/testdata/parity/layago_reference.py

The outputs are deterministic for a given numpy version (default_rng(20260926)).
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import struct
import unicodedata
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 20260926

# --------------------------------------------------------------------------
# safetensors (manual, little-endian)
# --------------------------------------------------------------------------


def bf16_bits(a: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 bits with round-to-nearest-even."""
    bits = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return rounded.astype(np.uint16)


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def write_safetensors(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    header, blobs, offset = {}, [], 0
    for name in sorted(tensors):
        dtype, arr = tensors[name]
        if dtype == "F32":
            raw = np.ascontiguousarray(arr, dtype="<f4").tobytes()
        elif dtype == "F16":
            raw = np.ascontiguousarray(arr, dtype="<f2").tobytes()
        elif dtype == "BF16":
            raw = bf16_bits(np.asarray(arr, dtype=np.float32)).astype("<u2").tobytes()
        elif dtype == "I8":
            raw = np.ascontiguousarray(arr, dtype=np.int8).tobytes()
        else:
            raise ValueError(dtype)
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    hdr = json.dumps(header, separators=(",", ":")).encode()
    hdr += b" " * ((8 - len(hdr) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        for blob in blobs:
            f.write(blob)


def read_safetensors(path: Path) -> dict[str, np.ndarray]:
    data = path.read_bytes()
    (n,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + n])
    buf = data[8 + n :]
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        lo, hi = info["data_offsets"]
        raw = buf[lo:hi]
        shape = info["shape"]
        dt = info["dtype"]
        if dt == "F32":
            arr = np.frombuffer(raw, dtype="<f4").astype(np.float64)
        elif dt == "F16":
            arr = np.frombuffer(raw, dtype="<f2").astype(np.float64)
        elif dt == "BF16":
            arr = bf16_to_f32(np.frombuffer(raw, dtype="<u2")).astype(np.float64)
        elif dt == "I8":
            arr = np.frombuffer(raw, dtype=np.int8).copy()
        else:
            raise ValueError(dt)
        out[name] = arr.reshape(shape)
    return out


# --------------------------------------------------------------------------
# synthetic tokenizer.json
# --------------------------------------------------------------------------

META = "▁"  # ▁
SPECIALS = ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>"]
MISSING_BYTE = 0xF0  # four-byte characters cannot byte-fall-back and become <unk>
BASE_CHARS = (
    META
    + "abcdefghijklmnopqrstuvwxyz"
    + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    + "0123456789"
    + ".,:;?!-_'\"(){}[]%/+=<>*#&@"
    + "\t\n"
    + "é中"
)
MERGE_CORPUS = """
choice question: does the current admitted measurement have exactly the same metric key unit grain and scope
score question: rate the evidence quality level 0 level 1 level 2 level 3
noul question: is the statement true false: no, the statement does not hold true: yes, the statement holds
matched unmatched unknown material low unknown investigate routine evidence_gap mandatory
{"metric_key": "orders", "unit": "count", "grain": "week", "scope": {"region": "north"}, "value": 1200.0}
the revenue of the retail domain changed by 12 percent versus the previous week and the forecast
""".split()


def train_merges(words: list[str], n: int) -> list[tuple[str, str]]:
    seqs = [list(META + w) for w in words]
    merges: list[tuple[str, str]] = []
    for _ in range(n):
        counts: dict[tuple[str, str], int] = {}
        for s in seqs:
            for a, b in zip(s, s[1:]):
                counts[(a, b)] = counts.get((a, b), 0) + 1
        if not counts:
            break
        best = min(counts, key=lambda p: (-counts[p], p))
        merges.append(best)
        merged = best[0] + best[1]
        for idx, s in enumerate(seqs):
            out, i = [], 0
            while i < len(s):
                if i + 1 < len(s) and (s[i], s[i + 1]) == best:
                    out.append(merged)
                    i += 2
                else:
                    out.append(s[i])
                    i += 1
            seqs[idx] = out
    return merges


def build_tokenizers() -> tuple[dict, dict, int]:
    vocab: dict[str, int] = {}

    def add(tok: str) -> int:
        if tok not in vocab:
            vocab[tok] = len(vocab)
        return vocab[tok]

    for t in SPECIALS:
        add(t)
    for b in range(256):
        if b != MISSING_BYTE:
            add("<0x%02X>" % b)
    for ch in BASE_CHARS:
        add(ch)
    merges = train_merges(MERGE_CORPUS, 150)
    for a, b in merges:
        add(a + b)
    for tok in ["\n\n", "<unused0>", "<b>", "</b>"]:
        add(tok)
    model_size = len(vocab)
    extra = {"new york": model_size, "zz": model_size + 1, "<r>": model_size + 2}
    total = model_size + len(extra)

    def added(content, id_, special=False, lstrip=False, rstrip=False, normalized=False, single_word=False):
        return {
            "id": id_,
            "content": content,
            "single_word": single_word,
            "lstrip": lstrip,
            "rstrip": rstrip,
            "normalized": normalized,
            "special": special,
        }

    added_tokens = [added(t, vocab[t], special=True, lstrip=(t == "<mask>")) for t in SPECIALS]
    added_tokens += [added(t, vocab[t]) for t in ["\n\n", "<unused0>", "<b>", "</b>"]]
    added_tokens += [
        added("new york", extra["new york"], normalized=True),
        added("zz", extra["zz"], single_word=True),
        added("<r>", extra["<r>"], rstrip=True),
    ]
    template = {
        "type": "TemplateProcessing",
        "single": [
            {"SpecialToken": {"id": "<bos>", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "<eos>", "type_id": 0}},
        ],
        "pair": [
            {"SpecialToken": {"id": "<bos>", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "<eos>", "type_id": 0}},
            {"Sequence": {"id": "B", "type_id": 0}},
            {"SpecialToken": {"id": "<eos>", "type_id": 0}},
        ],
        "special_tokens": {
            "<bos>": {"id": "<bos>", "ids": [vocab["<bos>"]], "tokens": ["<bos>"]},
            "<eos>": {"id": "<eos>", "ids": [vocab["<eos>"]], "tokens": ["<eos>"]},
        },
    }
    decoder = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": META}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
        ],
    }
    mmbert_style = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": META},
        "pre_tokenizer": {"type": "Metaspace", "replacement": META, "prepend_scheme": "always", "split": True},
        "post_processor": template,
        "decoder": decoder,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": True,
            "byte_fallback": True,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": [list(m) for m in merges],
        },
    }
    # A second layout exercising the other supported components: a Gemma-2
    # style Sequence[Prepend, Replace] normalizer without a pre-tokenizer,
    # legacy "a b" merges, unfused unk and ignore_merges.
    gemma_style = json.loads(json.dumps(mmbert_style))
    gemma_style["normalizer"] = {
        "type": "Sequence",
        "normalizers": [
            {"type": "Prepend", "prepend": META},
            {"type": "Replace", "pattern": {"String": " "}, "content": META},
        ],
    }
    gemma_style["pre_tokenizer"] = None
    gemma_style["model"]["merges"] = ["%s %s" % m for m in merges]
    gemma_style["model"]["fuse_unk"] = False
    gemma_style["model"]["ignore_merges"] = True
    return mmbert_style, gemma_style, total


# --------------------------------------------------------------------------
# reference tokenizer (Hugging Face tokenizers semantics)
# --------------------------------------------------------------------------

WHITE_SPACE = set("\t\n\v\f\r \x85\xa0     　") | {
    chr(c) for c in range(0x2000, 0x200B)
}


def is_word_char(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "LM" or unicodedata.category(ch) in ("Nl", "Nd", "Pc") or ch in "‌‍"


class Norm:
    """A normalized byte string with per-byte original ranges."""

    def __init__(self, b: bytes, align: list[tuple[int, int]]):
        self.b, self.align = b, align

    @classmethod
    def of(cls, text: str, base: int) -> "Norm":
        align, pos = [], base
        for ch in text:
            n = len(ch.encode())
            align += [(pos, pos + n)] * n
            pos += n
        return cls(text.encode(), align)

    def slice(self, a: int, b: int) -> "Norm":
        return Norm(self.b[a:b], self.align[a:b])

    def replace(self, pat: bytes, content: bytes) -> "Norm":
        out, align, last = bytearray(), [], 0
        while True:
            i = self.b.find(pat, last)
            if i < 0:
                break
            out += self.b[last:i]
            align += self.align[last:i]
            out += content
            align += [self.align[i + len(pat) - 1]] * len(content)
            last = i + len(pat)
        out += self.b[last:]
        align += self.align[last:]
        return Norm(bytes(out), align)

    def prepend(self, s: bytes) -> "Norm":
        if not self.b:
            return self
        return Norm(s + self.b, [self.align[0]] * len(s) + self.align)

    def orig(self, a: int, b: int) -> tuple[int, int]:
        return (self.align[a][0], self.align[b - 1][1])


class RefTokenizer:
    def __init__(self, spec: dict):
        self.normalizers = self._flatten(spec.get("normalizer"), "normalizers")
        self.pre = self._flatten(spec.get("pre_tokenizer"), "pretokenizers")
        model = spec["model"]
        self.vocab = model["vocab"]
        self.id_to_tok = {i: t for t, i in self.vocab.items()}
        self.unk = self.vocab.get(model["unk_token"]) if model.get("unk_token") else None
        self.fuse_unk = model.get("fuse_unk", False)
        self.byte_fallback = model.get("byte_fallback", False)
        self.ignore_merges = model.get("ignore_merges", False)
        self.merges = {}
        for rank, m in enumerate(model["merges"]):
            a, b = m if isinstance(m, list) else m.split(" ")
            self.merges[(self.vocab[a], self.vocab[b])] = (rank, self.vocab[a + b])
        self.added = spec["added_tokens"]
        self.raw_tokens, self.norm_tokens = [], []
        for idx, a in enumerate(self.added):
            if a["normalized"]:
                content = self.normalize(Norm.of(a["content"], 0)).b
                self.norm_tokens.append((content, idx))
            else:
                self.raw_tokens.append((a["content"].encode(), idx))
        self.raw_tokens.sort(key=lambda x: -len(x[0]))
        self.norm_tokens.sort(key=lambda x: -len(x[0]))
        self.template = spec.get("post_processor")

    @staticmethod
    def _flatten(node, key):
        if node is None:
            return []
        if node["type"] == "Sequence":
            out = []
            for n in node[key]:
                out += RefTokenizer._flatten(n, key)
            return out
        return [node]

    def normalize(self, n: Norm) -> Norm:
        for step in self.normalizers:
            if step["type"] == "Replace":
                n = n.replace(step["pattern"]["String"].encode(), step["content"].encode())
            elif step["type"] == "Prepend":
                n = n.prepend(step["prepend"].encode())
            else:
                raise ValueError(step["type"])
        return n

    def _split_added(self, b: bytes, tokens) -> list[tuple[int, int, int]]:
        if not b:
            return [(0, 0, -1)]
        out, start_offset, i = [], 0, 0
        while i < len(b):
            hit = next(((content, idx) for content, idx in tokens if b.startswith(content, i)), None)
            if hit is None:
                i += 1
                continue
            content, found = hit
            tok = self.added[found]
            start, stop = i, i + len(content)
            i = stop  # the scan resumes after the literal match, like aho-corasick
            if tok["single_word"]:
                before = b[:start].decode()
                after = b[stop:].decode()
                ok_start = start == 0 or not is_word_char(before[-1])
                ok_stop = stop == len(b) or not is_word_char(after[0])
                if not (ok_start and ok_stop):
                    continue
            if tok["lstrip"]:
                prefix = b[:start].decode()
                stripped = prefix.rstrip("".join(WHITE_SPACE))
                start = max(len(stripped.encode()), start_offset)
            if tok["rstrip"]:
                rest = b[stop:].decode()
                stop += len(rest.encode()) - len(rest.lstrip("".join(WHITE_SPACE)).encode())
            if start_offset < start:
                out.append((start_offset, start, -1))
            out.append((start, stop, found))
            start_offset = stop
        if start_offset < len(b):
            out.append((start_offset, len(b), -1))
        return out

    def _pre_tokenize(self, n: Norm) -> list[Norm]:
        pieces = [n]
        for m in self.pre:
            assert m["type"] == "Metaspace"
            rep = m["replacement"].encode()
            nxt = []
            for p in pieces:
                p = p.replace(b" ", rep)
                scheme = m.get("prepend_scheme", "always")
                if scheme == "always" and not p.b.startswith(rep):
                    p = p.prepend(rep)
                elif scheme == "first" and not p.b.startswith(rep) and p.align and p.align[0][0] == 0:
                    p = p.prepend(rep)
                if m.get("split", True):
                    nxt += self._split_merged_with_next(p, rep)
                else:
                    nxt.append(p)
            pieces = nxt
        return pieces

    @staticmethod
    def _split_merged_with_next(n: Norm, rep: bytes) -> list[Norm]:
        if not n.b:
            return [n]
        out, start, i = [], 0, 0
        while i < len(n.b):
            if n.b.startswith(rep, i):
                if i > start:
                    out.append(n.slice(start, i))
                start = i
                i += len(rep)
            else:
                i += 1
        if start < len(n.b):
            out.append(n.slice(start, len(n.b)))
        return out

    def _bpe(self, word: Norm, ids: list, offsets: list) -> None:
        w = word.b.decode()
        if not w:
            return
        if self.ignore_merges and w in self.vocab:
            ids.append(self.vocab[w])
            offsets.append(word.orig(0, len(word.b)))
            return
        syms = []  # [id, size]
        pending = None
        for ch in w:
            size = len(ch.encode())
            if ch in self.vocab:
                if pending is not None:
                    syms.append(pending)
                    pending = None
                syms.append([self.vocab[ch], size])
                continue
            if self.byte_fallback:
                codes = ["<0x%02X>" % byte for byte in ch.encode()]
                if all(c in self.vocab for c in codes):
                    # Hugging Face does not flush a pending unk here.
                    syms += [[self.vocab[c], 1] for c in codes]
                    continue
            if self.unk is None:
                continue
            if pending is not None and self.fuse_unk:
                pending[1] += size
            elif pending is not None:
                syms.append(pending)
                pending = [self.unk, size]
            else:
                pending = [self.unk, size]
        if pending is not None:
            syms.append(pending)
        # merge_all: lowest rank first, leftmost on ties, with stale checks
        prev = list(range(-1, len(syms) - 1))
        nxt = list(range(1, len(syms) + 1))
        if syms:
            nxt[-1] = -1
        heap = []
        for i in range(len(syms) - 1):
            m = self.merges.get((syms[i][0], syms[i + 1][0]))
            if m:
                heapq.heappush(heap, (m[0], i, m[1]))
        while heap:
            rank, pos, new_id = heapq.heappop(heap)
            if syms[pos][1] == 0 or nxt[pos] == -1:
                continue
            r = nxt[pos]
            m = self.merges.get((syms[pos][0], syms[r][0]))
            if not m or m[1] != new_id:
                continue
            syms[pos] = [new_id, syms[pos][1] + syms[r][1]]
            syms[r] = [syms[r][0], 0]
            nxt[pos] = nxt[r]
            if nxt[r] != -1:
                prev[nxt[r]] = pos
            if prev[pos] >= 0:
                m = self.merges.get((syms[prev[pos]][0], syms[pos][0]))
                if m:
                    heapq.heappush(heap, (m[0], prev[pos], m[1]))
            if nxt[pos] != -1:
                m = self.merges.get((syms[pos][0], syms[nxt[pos]][0]))
                if m:
                    heapq.heappush(heap, (m[0], pos, m[1]))
        at = 0
        for sid, size in syms:
            if size == 0:
                continue
            ids.append(sid)
            offsets.append(word.orig(at, at + size))
            at += size

    def encode(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        ids, offsets = [], []
        raw = text.encode()
        for s, e, idx in self._split_added(raw, self.raw_tokens):
            if idx >= 0:
                ids.append(self.added[idx]["id"])
                offsets.append((s, e))
                continue
            n = self.normalize(Norm.of(raw[s:e].decode(), s))
            for ns, ne, nidx in self._split_added(n.b, self.norm_tokens):
                piece = n.slice(ns, ne)
                if nidx >= 0:
                    ids.append(self.added[nidx]["id"])
                    offsets.append(piece.orig(0, len(piece.b)))
                    continue
                for word in self._pre_tokenize(piece):
                    self._bpe(word, ids, offsets)
        return ids, offsets

    def token_id(self, tok: str) -> int:
        for a in self.added:
            if a["content"] == tok:
                return a["id"]
        return self.vocab[tok]


TOKENIZER_CORPUS = [
    "",
    " ",
    "hello world",
    "Hey   friend!",
    "  leading and trailing  ",
    "choice question: is the <mask> ready?",
    "a <mask>b<mask> c",
    "x  <mask>",
    "<pad><eos>x<bos>",
    "Café au lait é",
    "中文 and 中",
    "emoji 😀😀 then é",
    "mixed 😀é😀x",
    "ü ö ä",
    # An unk (😀 cannot byte-fall-back: <0xF0> is absent) followed by a
    # byte-fallback character in the same word: Hugging Face emits the byte
    # tokens before the pending unk.
    "😀ü",
    "a😀üb😀",
    "tab\there\nnew\n\n\nlines",
    "<b>bold</b> text <unused0>",
    "I love new york city",
    "newyork is not new  york",
    "zz top, azz, zz_x, (zz)",
    "<r>   after rstrip <r>",
    # rstrip swallows "\n\n", which Hugging Face then re-emits as its own
    # added token with overlapping offsets.
    "a<r>  \n\nb",
    "zz 0 000 00\x00<r>  \n\n\n0",
    " nbsp　ideographic",
    '{"metric_key": "orders", "value": 1200.0}',
    "revenue_net_eur_wk dropped 12.5% vs last week",
    "▁▁ literal metaspace",
    "level 0: low level 1: high",
    "true: yes, the statement holds",
]


# --------------------------------------------------------------------------
# reference model (float64)
# --------------------------------------------------------------------------


def erf_vec(x: np.ndarray) -> np.ndarray:
    return np.vectorize(math.erf, otypes=[np.float64])(x)


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + erf_vec(x / math.sqrt(2.0)))


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray | None, eps: float) -> np.ndarray:
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    y = (x - mean) / np.sqrt(var + eps) * w
    return y + b if b is not None else y


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.exp(x - x.max(axis=axis, keepdims=True))
    return z / z.sum(axis=axis, keepdims=True)


def rope_tables(dim: int, theta: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """transformers' default rope in float32: inv_freq, positions and angles."""
    half = dim // 2
    inv = []
    for i in range(half):
        e = np.float32(2 * i) / np.float32(dim)
        p = np.float32(math.pow(theta, float(e)))
        inv.append(np.float32(1.0 / float(p)))
    cos = np.zeros((n, dim))
    sin = np.zeros((n, dim))
    for pos in range(n):
        for i in range(half):
            ang = np.float32(pos) * inv[i]
            c, s = float(np.float32(math.cos(float(ang)))), float(np.float32(math.sin(float(ang))))
            cos[pos, i] = cos[pos, half + i] = c
            sin[pos, i] = sin[pos, half + i] = s
    return cos, sin


def rotate_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def modernbert_forward(ids: list[int], w: dict, cfg: dict) -> list[np.ndarray]:
    """Returns [embeddings, layer_0 ... layer_{n-1}, final_norm(last)]."""
    d, nh = cfg["hidden_size"], cfg["num_attention_heads"]
    hd = d // nh
    inter = cfg["intermediate_size"]
    eps = cfg["norm_eps"]
    L = len(ids)
    p = "encoder."
    x = w[p + "embeddings.tok_embeddings.weight"][ids]
    x = layer_norm(x, w[p + "embeddings.norm.weight"], None, eps)
    states = [x]
    for li, kind in enumerate(cfg["layer_types"]):
        lp = p + "layers.%d." % li
        h = x if li == 0 else layer_norm(x, w[lp + "attn_norm.weight"], None, eps)
        qkv = (h @ w[lp + "attn.Wqkv.weight"].T).reshape(L, 3, nh, hd)
        q, k, v = (qkv[:, j].transpose(1, 0, 2) for j in range(3))  # [nh, L, hd]
        theta = cfg["global_rope_theta"] if kind == "full_attention" else cfg["local_rope_theta"]
        cos, sin = rope_tables(hd, theta, L)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        scores = q @ k.transpose(0, 2, 1) / math.sqrt(hd)
        if kind == "sliding_attention":
            dist = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])
            scores = np.where(dist <= cfg["local_attention"] // 2, scores, -np.inf)
        attn = softmax(scores) @ v  # [nh, L, hd]
        attn = attn.transpose(1, 0, 2).reshape(L, d)
        x = x + attn @ w[lp + "attn.Wo.weight"].T
        h = layer_norm(x, w[lp + "mlp_norm.weight"], None, eps)
        up = h @ w[lp + "mlp.Wi.weight"].T
        x = x + (gelu(up[:, :inter]) * up[:, inter:]) @ w[lp + "mlp.Wo.weight"].T
        states.append(x)
    states.append(layer_norm(x, w[p + "final_norm.weight"], None, eps))
    return states


def mha(x: np.ndarray, in_w, in_b, out_w, out_b, nhead: int) -> np.ndarray:
    L, d = x.shape
    hd = d // nhead
    qkv = x @ in_w.T + in_b
    q, k, v = (qkv[:, j * d : (j + 1) * d].reshape(L, nhead, hd).transpose(1, 0, 2) for j in range(3))
    attn = softmax(q @ k.transpose(0, 2, 1) / math.sqrt(hd)) @ v
    return attn.transpose(1, 0, 2).reshape(L, d) @ out_w.T + out_b


def laya_head(h: np.ndarray, qtype: int, markers: list[int], w: dict, head_layers: int):
    d = h.shape[1]
    nhead = max(1, d // 64)
    x = h + w["type_emb.weight"][qtype]
    for i in range(head_layers):
        p = "head.layers.%d." % i
        a = layer_norm(x, w[p + "norm1.weight"], w[p + "norm1.bias"], 1e-5)
        x = x + mha(
            a,
            w[p + "self_attn.in_proj_weight"],
            w[p + "self_attn.in_proj_bias"],
            w[p + "self_attn.out_proj.weight"],
            w[p + "self_attn.out_proj.bias"],
            nhead,
        )
        a = layer_norm(x, w[p + "norm2.weight"], w[p + "norm2.bias"], 1e-5)
        f = np.maximum(a @ w[p + "linear1.weight"].T + w[p + "linear1.bias"], 0.0)
        x = x + f @ w[p + "linear2.weight"].T + w[p + "linear2.bias"]
    m = x[markers]
    s = layer_norm(m, w["scorer.0.weight"], w["scorer.0.bias"], 1e-5)
    s = gelu(s @ w["scorer.1.weight"].T + w["scorer.1.bias"])
    logits = (s @ w["scorer.3.weight"].T + w["scorer.3.bias"])[:, 0]
    pr = softmax(logits)
    k = max(2, len(markers))
    top = sorted(pr, reverse=True) + [0.0]
    ent = -(pr * np.log(np.maximum(pr, 1e-9))).sum() / math.log(k)
    feats = np.array([top[0], top[0] - top[1], ent, k / 255.0])
    a = np.concatenate([x[0], feats])
    a = gelu(a @ w["act_head.0.weight"].T + w["act_head.0.bias"])
    act = a @ w["act_head.2.weight"].T + w["act_head.2.bias"]
    return logits, act


# --------------------------------------------------------------------------
# laya port (build_sequence, render_options, decoding)
# --------------------------------------------------------------------------


def render_options(q: dict) -> list[str]:
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [k if v is None or v == "" else "%s: %s" % (k, v) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, c) for i, c in enumerate(crit)]
    crit = crit or {}
    f, tr = crit.get("false"), crit.get("true")
    return [
        "false: " + (f if f not in (None, "") else "no, the statement does not hold"),
        "true: " + (tr if tr not in (None, "") else "yes, the statement holds"),
    ]


def build_sequence(tok: RefTokenizer, sp: dict, state: str, q: dict, max_len: int, head_max_len: int):
    mask_tok = sp["mask_token"]
    opts = render_options(q)
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = tok.encode("%s question: %s" % (q["t"], ins))[0]
    opt_ids = [[sp["mask"]] + tok.encode(" " + o.replace(mask_tok, " "))[0][:48] for o in opts]
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]
    ids = [sp["cls"]] + head_ids + [sp["sep"]]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(sp["sep"])
    room = max(0, max_len - len(ids) - 1)
    st = tok.encode(state.replace(mask_tok, " "))[0][:room]
    ids = ids + st + [sp["sep"]]
    return ids[:max_len], [m for m in markers if m < max_len]


def clamp_temperature(t: float) -> float:
    if t != t or t in (float("inf"), float("-inf")):
        return 1.0
    return min(5.0, max(0.5, float(t)))


def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (["choice", "score", "noul"][qtype], size)


QT = {"choice": 0, "score": 1, "noul": 2}


def internal_question(q: dict) -> dict:
    if q["type"] == "choice":
        crit = {o["key"]: o.get("description") for o in q["options"]}
    elif q["type"] == "score":
        crit = list(q["levels"])
    else:
        crit = {k: q[k] for k in ("false", "true") if q.get(k)}
    return {"t": q["type"], "ins": q["instructions"], "crit": crit}


def predict(tok, sp, w, enc_cfg, agent_cfg, state: str, questions: list[dict]):
    out = []
    for q in questions:
        iq = internal_question(q)
        ids, markers = build_sequence(tok, sp, state, iq, agent_cfg["max_len"], agent_cfg["head_max_len"])
        opts = render_options(iq)
        if len(markers) != len(opts):
            raise ValueError("options exceed head_max_len")
        h = modernbert_forward(ids, w, enc_cfg)[-1]
        qt = QT[q["type"]]
        logits, act = laya_head(h, qt, markers, w, agent_cfg["head_layers"])
        k = len(markers)
        temps = [clamp_temperature(t) for t in agent_cfg["temperature"]]
        by_opt = {kk: clamp_temperature(v) for kk, v in agent_cfg["temperature_by_options"].items()}
        t = by_opt.get(temp_bucket(qt, k), temps[qt])
        p = softmax(logits / t)
        ans = {
            "id": q["id"],
            "ids": ids,
            "markers": markers,
            "logits": logits.tolist(),
            "temperature": t,
            "probs": p.tolist(),
            "act_probability": float(softmax(act)[0]),
        }
        if q["type"] == "choice":
            keys = [o["key"] for o in q["options"]]
            ans["choice"] = keys[int(p.argmax())]
        elif q["type"] == "score":
            ans["score"] = float((np.arange(k) * p).sum())
        else:
            ans["noul"] = float(p[1])
        if q["type"] == "noul":
            ans["confidence"] = max(float(p[1]), 1.0 - float(p[1]))
        elif k < 2:
            ans["confidence"] = 1.0
        else:
            ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
            ans["confidence"] = float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))
        out.append(ans)
    return out


# --------------------------------------------------------------------------
# fixture generation
# --------------------------------------------------------------------------

D, HEADS, INTER = 64, 4, 96
LAYER_TYPES = ["full_attention", "sliding_attention", "sliding_attention", "full_attention", "sliding_attention"]
LOCAL_ATTENTION = 8

# Weight storage dtypes: a deliberate mix so the Go loader converts all three.
DTYPE_RULES = [
    ("tok_embeddings", "BF16"),
    ("attn.Wqkv", "F16"),
    ("attn.Wo", "BF16"),
    ("mlp.Wi", "F16"),
    ("mlp.Wo", "F32"),
    ("in_proj_weight", "F16"),
    ("out_proj.weight", "BF16"),
    ("linear1.weight", "F16"),
    ("linear2.weight", "BF16"),
    ("scorer.1.weight", "F16"),
    ("act_head.0.weight", "F16"),
]


def storage_dtype(name: str) -> str:
    for key, dt in DTYPE_RULES:
        if key in name:
            return dt
    return "F32"


def make_weights(rng: np.random.Generator, vocab: int, n_act: int) -> dict[str, np.ndarray]:
    w: dict[str, np.ndarray] = {}

    def lin(name, out, inp, bias=False, gain=1.0):
        w[name + ".weight"] = rng.normal(0, gain / math.sqrt(inp), (out, inp)).astype(np.float32)
        if bias:
            w[name + ".bias"] = rng.normal(0, 0.1, (out,)).astype(np.float32)

    def norm(name, bias=False):
        w[name + ".weight"] = (1 + 0.1 * rng.normal(0, 1, (D,))).astype(np.float32)
        if bias:
            w[name + ".bias"] = rng.normal(0, 0.1, (D,)).astype(np.float32)

    p = "encoder."
    w[p + "embeddings.tok_embeddings.weight"] = rng.normal(0, 1, (vocab, D)).astype(np.float32)
    norm(p + "embeddings.norm")
    for i in range(len(LAYER_TYPES)):
        lp = p + "layers.%d." % i
        if i > 0:
            norm(lp + "attn_norm")
        lin(lp + "attn.Wqkv", 3 * D, D, gain=1.5)
        lin(lp + "attn.Wo", D, D)
        norm(lp + "mlp_norm")
        lin(lp + "mlp.Wi", 2 * INTER, D)
        lin(lp + "mlp.Wo", D, INTER)
    norm(p + "final_norm")
    w["type_emb.weight"] = rng.normal(0, 0.5, (3, D)).astype(np.float32)
    for i in range(2):
        hp = "head.layers.%d." % i
        w[hp + "self_attn.in_proj_weight"] = rng.normal(0, 1.5 / math.sqrt(D), (3 * D, D)).astype(np.float32)
        w[hp + "self_attn.in_proj_bias"] = rng.normal(0, 0.1, (3 * D,)).astype(np.float32)
        lin(hp + "self_attn.out_proj", D, D, bias=True)
        lin(hp + "linear1", 4 * D, D, bias=True)
        lin(hp + "linear2", D, 4 * D, bias=True)
        norm(hp + "norm1", bias=True)
        norm(hp + "norm2", bias=True)
    norm("scorer.0", bias=True)
    lin("scorer.1", D, D, bias=True)
    lin("scorer.3", 1, D, bias=True, gain=3.0)
    lin("act_head.0", 256, D + 4, bias=True)
    lin("act_head.2", n_act, 256, bias=True, gain=2.0)
    w["temperature"] = np.ones(3, dtype=np.float32)
    return w


def quantize_rows(wm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror of tensor.QuantizeRows."""
    q = np.zeros(wm.shape, dtype=np.int8)
    scale = np.ones(wm.shape[0], dtype=np.float32)
    for r in range(wm.shape[0]):
        row = wm[r].astype(np.float64)
        m = float(np.abs(row).max())
        if m == 0:
            continue
        s = np.float32(m / 127.0)
        scale[r] = s
        q[r] = np.clip(np.rint(row / float(s)), -127, 127).astype(np.int8)
    return q, scale


def dequantized(w: dict) -> dict:
    out = dict(w)
    for name, arr in w.items():
        if not name.startswith("encoder."):
            continue
        if name.endswith("tok_embeddings.weight") or any(
            k in name for k in ("attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo")
        ):
            q, s = quantize_rows(arr)
            out[name] = q.astype(np.float64) * s.astype(np.float64)[:, None]
    return out


QUESTIONS = [
    {
        "id": "relevance",
        "type": "choice",
        "instructions": "Does the current admitted measurement have the same metric key, unit, grain and scope?",
        "options": [
            {"key": "matched", "description": "all declared coordinates match"},
            {"key": "unmatched", "description": "at least one coordinate differs"},
            {"key": "unknown", "description": "coordinates are absent"},
        ],
    },
    {
        "id": "plain_choice",
        "type": "choice",
        "instructions": "Pick the <mask> best label.",
        "options": [{"key": "low"}, {"key": "high"}],
    },
    {
        "id": "quality",
        "type": "score",
        "instructions": "Rate the evidence quality.",
        "levels": ["no evidence", "weak", "adequate", "strong"],
    },
    {"id": "is_true", "type": "noul", "instructions": "Is the statement about orders true?"},
    {
        "id": "is_material",
        "type": "noul",
        "instructions": "Is the change material?",
        "false": "the change is below the minimum",
        "true": "the change meets the minimum",
    },
    {
        "id": "many_options",
        "type": "choice",
        "instructions": "Choose the investigation priority for this evidence.",
        "options": [
            {"key": "mandatory", "description": "a mandatory policy reason is explicitly present"},
            {"key": "investigate", "description": "supported material change requires investigation"},
            {"key": "evidence_gap", "description": "missing or insufficient evidence"},
            {"key": "routine", "description": "supported low change"},
            {"key": "unknown", "description": "the state does not support any class"},
            {"key": "defer"},
            {"key": "escalate", "description": "hand over to a person"},
        ],
    },
    {"id": "single", "type": "choice", "instructions": "Only one option exists.", "options": [{"key": "only"}]},
]

STATES = [
    "orders rose",
    '{"domain": "retail", "objective": {"metric_key": "orders", "unit": "count", "grain": "week"}, '
    '"current": {"value": 1200.0}, "previous": {"value": 1000.0}, "note": "<mask> 中文 é 😀 ü"}',
    "the revenue of the retail domain changed by 12 percent versus the previous week and the forecast "
    "was issued before the observation with a standard deviation that supports a residual",
]

ENCODER_CASES = [
    [2, 40, 41],
    [2] + list(range(300, 317)) + [1],
    [2] + [(7 * i + 11) % 400 + 5 for i in range(47)] + [1],
    [4],
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    out = args.out
    ckpt = out / "checkpoint"
    (ckpt / "tokenizer").mkdir(parents=True, exist_ok=True)
    (ckpt / "encoder").mkdir(parents=True, exist_ok=True)

    tok_spec, gemma_spec, vocab_size = build_tokenizers()
    (ckpt / "tokenizer" / "tokenizer.json").write_text(json.dumps(tok_spec, ensure_ascii=False, indent=1) + "\n")
    (out / "tokenizer_gemma_style.json").write_text(json.dumps(gemma_spec, ensure_ascii=False, indent=1) + "\n")
    tok_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "cls_token": "<bos>",
        "sep_token": "<eos>",
        "mask_token": "<mask>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "model_max_length": 512,
        "clean_up_tokenization_spaces": False,
    }
    (ckpt / "tokenizer" / "tokenizer_config.json").write_text(json.dumps(tok_config, indent=1) + "\n")
    enc_cfg = {
        "architectures": ["ModernBertModel"],
        "model_type": "modernbert",
        "vocab_size": vocab_size,
        "hidden_size": D,
        "intermediate_size": INTER,
        "num_hidden_layers": len(LAYER_TYPES),
        "num_attention_heads": HEADS,
        "hidden_activation": "gelu",
        "norm_eps": 1e-5,
        "norm_bias": False,
        "attention_bias": False,
        "mlp_bias": False,
        "local_attention": LOCAL_ATTENTION,
        "global_attn_every_n_layers": 3,
        "layer_types": LAYER_TYPES,
        "global_rope_theta": 160000.0,
        "local_rope_theta": 10000.0,
        "max_position_embeddings": 512,
        "pad_token_id": 0,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "cls_token_id": 2,
        "sep_token_id": 1,
        "attention_dropout": 0.0,
        "embedding_dropout": 0.0,
        "mlp_dropout": 0.0,
    }
    (ckpt / "encoder" / "config.json").write_text(json.dumps(enc_cfg, indent=1) + "\n")
    agent_cfg = {
        "encoder": "synthetic-modernbert-d64",
        "head_layers": 2,
        "max_len": 72,
        "head_max_len": 40,
        "act_costs": {"escalate": 0.5},
        "temperature": [1.3, 0.8, 1.0],
        "temperature_by_options": {"choice:3-5": 1.7, "score:3-5": 0.2, "noul:2": 7.5},
    }
    (ckpt / "rl_agent_config.json").write_text(json.dumps(agent_cfg, indent=1) + "\n")

    rng = np.random.default_rng(SEED)
    raw_w = make_weights(rng, vocab_size, len(agent_cfg["act_costs"]) + 1)
    write_safetensors(ckpt / "model.safetensors", {k: (storage_dtype(k), v) for k, v in raw_w.items()})
    w = read_safetensors(ckpt / "model.safetensors")  # the stored (rounded) values, as float64

    # Tokenizer goldens for both layouts.
    tok = RefTokenizer(tok_spec)
    gemma = RefTokenizer(gemma_spec)
    tok_golden = {}
    for name, t in (("synthetic", tok), ("gemma_style", gemma)):
        tok_golden[name] = [{"text": s, "ids": t.encode(s)[0], "offsets": [list(o) for o in t.encode(s)[1]]}
                            for s in TOKENIZER_CORPUS]
    (out / "tokenizer_golden.json").write_text(json.dumps(tok_golden, ensure_ascii=False, indent=1) + "\n")

    # Encoder goldens: every hidden state, float32-rounded.
    enc_tensors, enc_meta = {}, []
    wq = dequantized(w)
    for c, ids in enumerate(ENCODER_CASES):
        states = modernbert_forward(ids, w, enc_cfg)
        for k, s in enumerate(states[:-1]):
            enc_tensors["case%d.hidden%d" % (c, k)] = ("F32", s.astype(np.float32))
        enc_tensors["case%d.final" % c] = ("F32", states[-1].astype(np.float32))
        final_q = modernbert_forward(ids, wq, enc_cfg)[-1]
        enc_tensors["case%d.final_int8" % c] = ("F32", final_q.astype(np.float32))
        enc_meta.append({"ids": ids, "int8_max_abs_vs_fp": float(np.abs(final_q - states[-1]).max())})
    write_safetensors(out / "encoder_golden.safetensors", enc_tensors)
    (out / "encoder_golden.json").write_text(json.dumps({"cases": enc_meta}, indent=1) + "\n")

    sp = {
        "cls": tok.token_id("<bos>"),
        "sep": tok.token_id("<eos>"),
        "mask": tok.token_id("<mask>"),
        "pad": tok.token_id("<pad>"),
        "mask_token": "<mask>",
    }
    laya_cases = []
    for state in STATES:
        laya_cases.append(
            {
                "state": state,
                "answers": predict(tok, sp, w, enc_cfg, agent_cfg, state, QUESTIONS),
                "answers_int8": predict(tok, sp, wq, enc_cfg, agent_cfg, state, QUESTIONS),
            }
        )
    digest = hashlib.sha256((ckpt / "model.safetensors").read_bytes()).hexdigest()
    (out / "laya_golden.json").write_text(
        json.dumps(
            {
                "producer": "layago_reference.py (numpy %s, float64)" % np.__version__,
                "model_sha256": digest,
                "questions": QUESTIONS,
                "cases": laya_cases,
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n"
    )
    print("vocab", vocab_size, "model sha256", digest)
    for case in laya_cases:
        print([(a["id"], len(a["ids"]), a.get("choice", a.get("score", a.get("noul")))) for a in case["answers"]])


if __name__ == "__main__":
    main()
