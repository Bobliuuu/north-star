"""Teacher caching and edited-vs-teacher metrics (KL in nats/token, top-1 agreement) on teacher-forced sequences."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from weightedit.distill import logits_of


def _dev(m):
    return next(m.parameters()).device


@torch.no_grad()
def teacher_cache(m, seqs, lo):
    """Logits of the ORIGINAL model at positions lo..L-2 (predicting tokens lo+1..L-1), kept on CPU in model dtype."""
    out = []
    for s in seqs:
        x = torch.tensor([s], device=_dev(m))
        out.append(logits_of(m, x)[0, lo:-1].cpu())
    return out


@torch.no_grad()
def eval_vs_teacher(m, seqs, lo, cache):
    """Per-sequence arrays: kl[pos], agree[pos] (bool). The edited model runs in its current state."""
    kls, ags = [], []
    for s, t in zip(seqs, cache):
        x = torch.tensor([s], device=_dev(m))
        d = logits_of(m, x)[0, lo:-1].float()
        t = t.to(d.device).float()
        lt, ld = F.log_softmax(t, -1), F.log_softmax(d, -1)
        kls.append((lt.exp() * (lt - ld)).sum(-1).cpu().numpy())
        ags.append((t.argmax(-1) == d.argmax(-1)).cpu().numpy())
    return kls, ags


def summarize(kls, ags):
    kl_all, ag_all = np.concatenate(kls), np.concatenate(ags)
    return dict(kl=float(kl_all.mean()), agree=float(ag_all.mean()), n_pos=int(kl_all.size),
                kl_by_seq=[float(k.mean()) for k in kls])


def boot_reduction_lower(kl_a, kl_b, n=1000, seed=0, q=5):
    """Paired cluster bootstrap over sequences of the relative KL reduction (a - b)/a; returns q-th percentile."""
    a = np.array([k.mean() for k in kl_a]); b = np.array([k.mean() for k in kl_b])
    rng = np.random.default_rng(seed)
    red = []
    for _ in range(n):
        ix = rng.integers(0, len(a), len(a))
        red.append((a[ix].mean() - b[ix].mean()) / max(a[ix].mean(), 1e-12))
    return float(np.percentile(red, q))
