"""Real-corpus loader: source/text files on this machine, tokenized with the Qwen3-0.6B BPE
tokenizer (NOT North's tokenizer; token-level statistics are an approximation of North's)."""
from __future__ import annotations
import os, random, glob, hashlib

EXTS = {".py", ".cu", ".cuh", ".md", ".sh", ".c", ".h", ".rs", ".ts", ".txt", ".cpp", ".lua", ".go"}
SKIP = {"node_modules", ".git", "__pycache__", ".venv", "venv", "build", "results", "data"}
CORPORA = {
    "kittens-vm": "~/kittens-vm",
    "north-star-src": "~/north-star/src",
    "qwentpu_v2": "~/qwentpu_v2",
    "cakebench": "~/cakebench-src",
    "bobliu-os": "~/bobliu-os",
    "doom-src(C)": "~/doom-src",
}


def tokenizer():
    from tokenizers import Tokenizer
    p = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/tokenizer.json"))[0]
    return Tokenizer.from_file(p)


def load_docs(root, max_tokens=150_000, max_doc_tokens=6000, min_doc_tokens=200, seed=0):
    tk = tokenizer()
    root = os.path.expanduser(root)
    files = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP]
        for f in fns:
            if os.path.splitext(f)[1] in EXTS:
                p = os.path.join(dp, f)
                try:
                    if os.path.getsize(p) < 200_000:
                        files.append(p)
                except OSError:
                    pass
    files.sort()
    random.Random(seed).shuffle(files)
    docs, total, seen = [], 0, set()
    for p in files:
        try:
            txt = open(p, encoding="utf-8").read()
        except Exception:
            continue
        h = hashlib.md5(txt.encode()).hexdigest()
        if h in seen:
            continue                      # drop exact duplicate files
        seen.add(h)
        ids = tk.encode(txt).ids[:max_doc_tokens]
        if len(ids) < min_doc_tokens:
            continue
        docs.append(ids); total += len(ids)
        if total >= max_tokens:
            break
    return docs
