"""Cross-token overlap analysis (pure numpy). Inputs are routing traces [L_present][T][8] (score-ordered ids).

For a cycle: verify block = tokens s..s+K-1 routed natively (top-8). FUTURE draft tokens = positions s+K..s+K+Td-1 routed by the
draft mode (first r of its own ranking; skip modes drop layers). Teacher-forced (all drafts accepted), which is the case where
the next cycle's draft tokens sit right after the verify block.
Metrics per (layer, window):
  pair_in_union   fraction of the draft tokens' (token, expert) loads whose expert is in the verify block's expert union
  uniq_in_union   |U_D & U_V| / |U_D|
  byte_saving     |U_D & U_V| / (|U_V| + |U_D|)   = fraction of expert-weight bytes a fused pass avoids vs two passes
  same_pos_*      same metrics when the draft tokens are the verify block's OWN tokens (the earlier, easier teacher-forced metric)
  shuf_*          control: verify block taken from a distant random position of the same layer
"""
import numpy as np


def window_stats(V, D, r, rng=None, Vshuf=None):
    uv = set(V.ravel().tolist())
    dd = D[:, :r]
    ud = set(dd.ravel().tolist())
    inter = uv & ud
    pair = float(np.mean([e in uv for e in dd.ravel().tolist()]))
    return dict(pair_in_union=pair, uniq_in_union=len(inter) / len(ud), byte_saving=len(inter) / (len(uv) + len(ud)),
                n_union_v=len(uv), n_union_d=len(ud))


def analyze(full, draft, layers_draft, r, K, Td, seed=0):
    """full: [L,T,8] native routing; draft: [Ld,T,8] draft-mode routing for layers `layers_draft` (indices into L).
    Returns dict of means over all (layer, window)."""
    rng = np.random.default_rng(seed)
    L, T, _ = full.shape
    acc = {k: [] for k in ("pair_in_union", "uniq_in_union", "byte_saving", "same_pair", "same_uniq", "same_byte", "shuf_pair", "shuf_uniq", "shuf_byte", "n_union_v", "n_union_d")}
    per_layer = {}
    for li, l in enumerate(layers_draft):
        for s in range(0, T - K - Td + 1):
            V = full[l, s:s + K]
            D = draft[li, s + K:s + K + Td]
            a = window_stats(V, D, r)
            same = window_stats(V, draft[li, s:s + K][:Td] if Td <= K else draft[li, s:s + K], r)
            # control: random distant verify block
            cand = [x for x in range(0, T - K + 1) if abs(x - s) > K + Td]
            sh = window_stats(full[l, (rng.choice(cand) if cand else 0):][:K], D, r)
            acc["pair_in_union"].append(a["pair_in_union"]); acc["uniq_in_union"].append(a["uniq_in_union"]); acc["byte_saving"].append(a["byte_saving"])
            acc["same_pair"].append(same["pair_in_union"]); acc["same_uniq"].append(same["uniq_in_union"]); acc["same_byte"].append(same["byte_saving"])
            acc["shuf_pair"].append(sh["pair_in_union"]); acc["shuf_uniq"].append(sh["uniq_in_union"]); acc["shuf_byte"].append(sh["byte_saving"])
            acc["n_union_v"].append(a["n_union_v"]); acc["n_union_d"].append(a["n_union_d"])
            per_layer.setdefault(int(l), []).append(a["byte_saving"])
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out["per_layer_byte_saving"] = {l: float(np.mean(v)) for l, v in per_layer.items()}
    out["n_windows"] = len(acc["pair_in_union"])
    return out
