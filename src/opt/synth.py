"""SYNTHETIC acceptance generator -- FOR TESTS AND REGIME SCANS ONLY.

!!! Everything produced here is INVENTED. The functional forms below are arbitrary monotone shapes chosen so
!!! the optimizer has a non-trivial trade-off to resolve. They are NOT measurements of North and must never
!!! be reported as acceptance results. Every row is stamped provenance="SYNTHETIC" and the solver propagates
!!! the flag into every report. Real calibration comes from Phase 1/2 teacher-forced runs (see calibration.py).
"""
from __future__ import annotations

import json
import random

from .calibration import AcceptEntry

BANNER = "SYNTHETIC ACCEPTANCE -- invented for tests/regime scans; not North measurements"

FAMILIES = {
    # name: (base per-token acceptance, penalty per attn-budget halving, per dropped expert, per skipped layer)
    "attn-tolerant":   (0.90, 0.01, 0.040, 0.012),
    "attn-sensitive":  (0.90, 0.08, 0.030, 0.012),
    "expert-sensitive": (0.90, 0.02, 0.090, 0.015),
    "skip-sensitive":  (0.90, 0.02, 0.040, 0.035),
}
SKIP_SETS = {"none": (), "late6": tuple(range(40, 46)), "mid12": tuple(range(16, 40, 2))}
ATTN_BUDGETS = (0, 2048, 512)
RS = (2, 3, 4, 6, 8)


def _alpha(fam, attn_budget, r, nskip, ctx_ref=32768):
    base, pa, pe, ps = FAMILIES[fam]
    a = base
    if attn_budget:
        import math
        a -= pa * max(0.0, math.log2(ctx_ref / attn_budget)) / 2
    a -= pe * (8 - r) + ps * nskip
    return min(0.995, max(0.05, a))


def make_pmf(alpha, gmax, decay=0.985):
    """Truncated geometric-like pmf with mild per-position decay (invented)."""
    pmf = []; surv = 1.0
    for j in range(gmax):
        a = alpha * (decay ** j)
        pmf.append(surv * (1 - a)); surv *= a
    pmf.append(surv)
    return tuple(pmf)


def synthetic_entries(workload="code", family="attn-tolerant", gmax=8, carry_hit=0.6, seed=0,
                      i_understand_this_is_synthetic=False, noise=0.0):
    if not i_understand_this_is_synthetic:
        raise RuntimeError(BANNER + ": pass i_understand_this_is_synthetic=True (tests / regime scans only)")
    rng = random.Random(seed)
    out = []
    for ab in ATTN_BUDGETS:
        for r in RS:
            for name, sk in SKIP_SETS.items():
                a = _alpha(family, ab, r, len(sk))
                if noise:
                    a = min(0.995, max(0.05, a + rng.gauss(0, noise)))
                out.append(AcceptEntry(workload, (ab, r, tuple(sorted(sk))), make_pmf(a, gmax), 0, carry_hit,
                                       "SYNTHETIC", f"synth:{family}:seed{seed}"))
    return out


def write_jsonl(entries, path):
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps({"workload": e.workload, "attn_budget": e.key[0], "r": e.key[1],
                                "skip_layers": list(e.key[2]), "accept_len_pmf": list(e.pmf),
                                "n_samples": e.n_samples, "carry_hit_given_full": e.carry_hit_given_full,
                                "provenance": e.provenance, "source": e.source, "_banner": BANNER}) + "\n")
