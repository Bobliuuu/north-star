"""Self-distillation: train adapters so the edited top-r DRAFT matches the frozen VERIFIER distribution (KL).
Base model is frozen (requires_grad False restored afterwards); only adapter params are in the optimiser."""
from __future__ import annotations

import contextlib
import time

import torch
import torch.nn.functional as F

from .adapter import EditedDraft, RouterAdapters
from .recipe import Recipe


def logits_of(m, x):
    h = m.model(x, use_cache=False).last_hidden_state
    return m.lm_head(h) * m.config.logit_scale


@contextlib.contextmanager
def frozen(m):
    prev = [(p, p.requires_grad) for p in m.parameters()]
    for p, _ in prev:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, r in prev:
            p.requires_grad_(r)


@contextlib.contextmanager
def grad_checkpointed(m):
    """Layer-level activation checkpointing via instance forwards (removed on exit)."""
    from torch.utils.checkpoint import checkpoint
    layers = list(m.model.layers)
    # Layers may already carry an instance forward (EditedDraft layer-skip patch). Chain on top of it
    # and restore exactly what was there, instead of asserting/popping (that crashed r4s, 2026-11-06).
    prev = [L.__dict__.get("forward") for L in layers]
    for L in layers:
        orig = L.forward
        L.forward = (lambda *a, _o=orig, **k: checkpoint(_o, *a, use_reentrant=False, **k))
    try:
        yield
    finally:
        for L, p in zip(layers, prev):
            if p is None:
                L.__dict__.pop("forward", None)
            else:
                L.forward = p


def sample_windows(seqs, window, batch, gen):
    out = []
    for _ in range(batch):
        s = seqs[int(torch.randint(len(seqs), (1,), generator=gen))]
        assert len(s) >= window, f"sequence shorter than window ({len(s)} < {window})"
        o = int(torch.randint(len(s) - window + 1, (1,), generator=gen))
        out.append(s[o:o + window])
    return torch.tensor(out)


def kl_loss(teacher_logits, draft_logits, T=1.0):
    t = F.softmax(teacher_logits.float() / T, -1)
    return F.kl_div(F.log_softmax(draft_logits.float() / T, -1), t, reduction="batchmean") * (T * T) / teacher_logits.shape[1]


def train(m, adapters: RouterAdapters, patch: EditedDraft, seqs, recipe: Recipe, log=print, ckpt=False):
    """Returns dict(loss=[...], first, last, seconds). `patch` must be installed on `m`."""
    dev = next(m.parameters()).device
    adapters.to(dev)
    gen = torch.Generator().manual_seed(recipe.seed)
    opt = torch.optim.AdamW(adapters.parameters(), lr=recipe.lr, weight_decay=0.0)
    losses, t0 = [], time.time()
    ctx = grad_checkpointed(m) if ckpt else contextlib.nullcontext()
    with frozen(m), ctx:
        for step in range(recipe.steps):
            x = sample_windows(seqs, recipe.window, recipe.batch, gen).to(dev)
            with torch.no_grad(), patch.verifier():
                tl = logits_of(m, x)
            opt.zero_grad(set_to_none=True)
            with patch.draft():  # backward INSIDE draft scope: checkpoint recompute must see draft mode
                dl = logits_of(m, x)
                loss = kl_loss(tl, dl, recipe.temperature)
                loss.backward()
            opt.step()
            losses.append(loss.item())
            if step % max(1, recipe.steps // 10) == 0:
                log(f"  step {step} kl={losses[-1]:.4f}")
    return dict(loss=losses, first=losses[0], last=losses[-1], seconds=time.time() - t0)
