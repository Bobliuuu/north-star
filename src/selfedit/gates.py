"""Pre-registered gates (docs/SELF_EDIT.md). Defaults are the registered thresholds."""
from __future__ import annotations

import numpy as np
import torch

from .quant import tensor_hash

REG = dict(kl_max=0.03, agree_min=0.96, bytes_min=0.15)


def nonexpert_fingerprint(m):
    """Sampled hash of every parameter/buffer EXCEPT routed-expert weight tensors."""
    import hashlib
    h = hashlib.sha256()
    for n, t in list(m.named_parameters()) + list(m.named_buffers()):
        if n.endswith(".experts.gate_up_proj") or n.endswith(".experts.down_proj"):
            continue
        h.update(n.encode()); h.update(tensor_hash(t).encode())
    return h.hexdigest()


def apply_gates(num, bytes_saved, nonexpert_same, unlisted_same, listed_changed, source, cfg=None):
    c = {**REG, **(cfg or {})}
    g = dict(quality=bool(num["kl"] <= c["kl_max"] and num["agree"] >= c["agree_min"]),
             bytes=bool(bytes_saved >= c["bytes_min"]),
             nonexpert=bool(nonexpert_same and unlisted_same and listed_changed))
    g["selfauthored"] = source == "llm"
    return g


def claim_check(north_res, random_kls, greedy_kl=None):
    """North 'improved itself' iff the arm passed quality+bytes+nonexpert AND selfauthored AND its KL is strictly below
    >= 3 of the random same-K recipes' KLs. Returns dict with the reasons."""
    base_ok = all(north_res["gates"][k] for k in ("quality", "bytes", "nonexpert", "selfauthored"))
    beats = int(sum(north_res["kl"] < r for r in random_kls))
    out = dict(gates_ok=base_ok, beats_random=beats, n_random=len(random_kls), north_kl=north_res["kl"],
               random_median_kl=float(np.median(random_kls)) if random_kls else None)
    out["claim"] = bool(base_ok and beats >= 3)
    if greedy_kl:
        out["kl_ratio_vs_greedy"] = north_res["kl"] / greedy_kl
    return out
