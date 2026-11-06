"""Self-edit loop: proposer(history) -> recipe -> validate -> train -> gates -> commit/reject."""
from __future__ import annotations

import itertools
import json
import re

import torch

from .adapter import EditedDraft, RouterAdapters
from .distill import train
from .gates import fingerprint, run_gates
from .recipe import Recipe, RecipeError, SCHEMA_TEXT
from .store import EditStore


class GridProposer:
    """Deterministic baseline proposer (no model). Cycles a small grid; source='grid'."""
    def __init__(self, rs=(2, 3), ranks=(4, 8), lrs=(1e-3, 3e-3), steps=100, **base):
        self.grid = [dict(expert_r=r, rank=k, lr=l, steps=steps, **base) for r, k, l in itertools.product(rs, ranks, lrs)]
        self.i = 0

    def propose(self, history):
        d = self.grid[self.i % len(self.grid)]
        self.i += 1
        return dict(d), dict(source="grid")


def extract_json(text):
    """First balanced {...} object in text, parsed; None if none."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for j in range(start, len(text)):
            depth += text[j] == "{"
            depth -= text[j] == "}"
            if depth == 0:
                try:
                    return json.loads(text[start:j + 1])
                except json.JSONDecodeError:
                    break
        start = text.find("{", start + 1)
    return None


def history_text(history, k=6):
    lines = []
    for h in history[-k:]:
        n = h.get("numbers") or {}
        lines.append(json.dumps(dict(recipe=h.get("recipe"), passed=h.get("passed"),
                                     agree_base=n.get("agree_base"), agree_edit=n.get("agree_edit"),
                                     failed=[g for g, v in (h.get("gates") or {}).items() if not v],
                                     error=h.get("error")), default=float))
    return "\n".join(lines) or "(none yet)"


class LLMProposer:
    """North proposes its own recipe. generate_fn(prompt:str)->str must run the model in VERIFIER mode.
    On unparsable/invalid output falls back to `fallback` and tags source='fallback' (never counts as self-authored)."""
    def __init__(self, generate_fn, mc, fallback=None):
        self.gen, self.mc = generate_fn, mc
        self.fallback = fallback or GridProposer()

    def prompt(self, history):
        return ("You are North. You are improving your own cheap DRAFT (top-r experts) by training a small router "
                "adapter that distills your full model. Goal: raise draft/verifier top-1 agreement on a held-out "
                f"split without more active experts. {SCHEMA_TEXT}\nPast rounds:\n{history_text(history)}\n"
                "Reply with ONE JSON recipe object and nothing else.")

    def propose(self, history):
        try:
            text = self.gen(self.prompt(history))
            d = extract_json(text)
            if d is None:
                raise RecipeError("no json object in reply")
            Recipe.from_dict(d).validate(self.mc)
            return d, dict(source="llm", raw=text[:500])
        except Exception as e:  # noqa: BLE001 - any failure -> recorded fallback
            d, meta = self.fallback.propose(history)
            return d, dict(source="fallback", error=f"{type(e).__name__}: {e}")


def run_round(m, proposer, store: EditStore, train_seqs, train_from, gate_seqs, gate_from, history, cache=None,
              fp_mode="full", log=print, ckpt=False, **gate_kw):
    d, meta = proposer.propose(history)
    try:
        recipe = Recipe.from_dict(d).validate(m.config)
    except RecipeError as e:
        entry = dict(passed=False, error=f"invalid recipe: {e}", recipe=d, **meta)
        store.reject(None, {}, meta["source"], reason=entry["error"])
        history.append(entry)
        return entry
    fp0 = fingerprint(m, fp_mode)
    adapters = RouterAdapters(m.config, recipe)
    with EditedDraft(m, adapters, recipe.expert_r) as patch:
        tr = train(m, adapters, patch, train_seqs, recipe, log=log, ckpt=ckpt)
        patch.tokens = patch.slots_active = 0
        with torch.no_grad():
            rep = run_gates(m, patch, adapters, recipe, train_seqs, train_from, gate_seqs, gate_from, fp0, fp_mode,
                            cache, **gate_kw)
    rep["train"] = dict(first=tr["first"], last=tr["last"], seconds=tr["seconds"])
    entry = dict(recipe=recipe.to_dict(), passed=rep["passed"], gates=rep["gates"], numbers=rep["numbers"],
                 train=rep["train"], **meta)
    if rep["passed"]:
        entry["version"] = store.commit(adapters, recipe, rep, meta["source"])
    else:
        store.reject(recipe, rep, meta["source"])
    history.append(entry)
    log(f"round: {meta['source']} passed={rep['passed']} gates={rep['gates']} "
        f"agree {rep['numbers']['agree_base']:.3f}->{rep['numbers']['agree_edit']:.3f}")
    return entry


def run_loop(m, proposer, store, train_seqs, train_from, gate_seqs, gate_from, rounds, **kw):
    history, cache = [], {}
    for _ in range(rounds):
        run_round(m, proposer, store, train_seqs, train_from, gate_seqs, gate_from, history, cache=cache, **kw)
    return history
