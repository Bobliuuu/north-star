"""Shared harness utilities: workloads, exact-length prompt construction, JSONL + metadata + spend log."""
from __future__ import annotations

import hashlib, json, os, pathlib, platform, subprocess, sys, time

HERE = pathlib.Path(__file__).resolve().parent
MODEL_ID = "CohereLabs/North-Mini-Code-1.0"
MODEL_REV = "d11e61a842617a22dc328552fa5bb86231ee4f37"
A100_80_USD_PER_S = 0.000694  # Modal list price ~$2.50/h; CPU/mem side costs are negligible and not added

STDLIB_FILLER = ["argparse", "dataclasses", "difflib", "json/decoder", "pathlib", "shutil", "subprocess",
                 "tarfile", "textwrap", "logging/__init__", "http/server", "unittest/case", "typing", "enum", "heapq"]


def load_workloads(root=None):
    root = pathlib.Path(root) if root else HERE / "workloads"
    rows = []
    for name in ("code", "terminal", "prose"):
        rows += [json.loads(l) for l in (root / f"{name}.jsonl").read_text().splitlines() if l.strip()]
    return rows


def by_split(rows, split):
    return [r for r in rows if r["split"] == split]


def _filler_text(rotate):
    import sysconfig
    lib = pathlib.Path(sysconfig.get_paths()["stdlib"])
    mods = STDLIB_FILLER[rotate % len(STDLIB_FILLER):] + STDLIB_FILLER[: rotate % len(STDLIB_FILLER)]
    parts = []
    for m in mods:
        p = lib / (m + ".py")
        if p.exists():
            parts.append(f"# ===== file: {m}.py =====\n" + p.read_text(errors="ignore"))
    return "\n".join(parts)


def short_ids(tok, task):
    return tok.apply_chat_template(task["messages"], add_generation_prompt=True, tokenize=True)


def _ids(tok, msgs):
    out = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    return list(out["input_ids"]) if hasattr(out, "keys") else list(out)


def build_ids(tok, task, target_len=None, rotate=0):
    """Chat-templated token ids for `task`. If target_len is given, a repository-context filler
    (CPython stdlib sources, in the first user turn) is inserted so the length is EXACTLY target_len."""
    msgs = [dict(m) for m in task["messages"]]
    if target_len is None:
        return _ids(tok, msgs)
    base = _ids(tok, msgs)
    if len(base) > target_len:
        raise ValueError("task longer than target")
    full = tok(_filler_text(rotate), add_special_tokens=False)["input_ids"]
    n = max(0, target_len - len(base))
    while True:
        m2 = [dict(m) for m in msgs]
        m2[0]["content"] = "Repository context:\n" + tok.decode(full[:n]) + "\n\n---\nTask:\n" + msgs[0]["content"]
        ids = _ids(tok, m2)
        if len(ids) >= target_len:
            break
        n += max(1, target_len - len(ids))
        assert n < len(full), "filler exhausted"
    excess = len(ids) - target_len
    if excess:  # drop tokens from the middle of the filler region (exact length, filler content is irrelevant)
        mid = len(ids) // 2
        ids = ids[: mid - excess] + ids[mid:]
    assert len(ids) == target_len
    return ids


def env_meta():
    import torch, transformers
    m = dict(python=sys.version.split()[0], torch=torch.__version__, transformers=transformers.__version__,
             model=MODEL_ID, model_rev=MODEL_REV, time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if torch.cuda.is_available():
        m.update(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                 gpu_mem_gb=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
    try:
        m["pip_freeze_sha"] = hashlib.sha256(subprocess.check_output([sys.executable, "-m", "pip", "freeze"])).hexdigest()[:12]
    except Exception:
        pass
    return m


class Jsonl:
    def __init__(self):
        self.lines = []

    def add(self, **kw):
        self.lines.append(json.dumps(kw, sort_keys=True, default=float))

    def text(self):
        return "\n".join(self.lines) + "\n"


def spend_row(run, gpu, seconds, extra=""):
    usd = seconds * A100_80_USD_PER_S if "A100" in gpu else float("nan")
    return f"| {time.strftime('%Y-%m-%d %H:%M')} | {run} | {gpu} | {seconds:.0f} | {usd:.2f} | {extra} |\n"


def append_spend(results_dir, run, gpu, seconds, extra=""):
    p = pathlib.Path(results_dir) / "SPEND.md"
    if not p.exists():
        p.write_text("# SPEND\n\nModal A100-80GB list price assumed $0.000694/GPU-s ($2.50/h). Budget cap this task: $30. "
                     "`seconds` = wall time of the GPU container function (includes model load). Cumulative total at bottom of each report.\n\n"
                     "| time (local) | run | GPU | GPU-s | est USD | note |\n|---|---|---|---|---|---|\n")
    with p.open("a") as f:
        f.write(spend_row(run, gpu, seconds, extra))
