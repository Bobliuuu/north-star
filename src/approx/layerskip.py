"""Layer-skip sets for LayerSkip / CLaSp-style drafts (axis C). Layer 0 (dense, RoPE-global) is never skipped.

NOTE on faithfulness: North was not trained with layer dropout / early-exit loss, so these are TRAINING-FREE
skip sets (as in CLaSp / CAS-Spec), not LayerSkip's trained early exits.  `early_exit_set` is the LayerSkip
*inference shape* only.  `clasp_select` is a re-implementation of CLaSp's dynamic programme from memory of the
abstract (training-free DP over skipped layers maximising similarity to the full-model hidden state); the
paper text was not read, and the layer-application callable is supplied by the caller (see docstring)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def early_exit_set(num_layers: int, exit_after: int):
    """LayerSkip inference shape: run layers [0, exit_after) then jump to final norm/LM head."""
    return frozenset(range(max(1, exit_after), num_layers))


def uniform_skip_set(num_layers: int, n_skip: int, region=(1, None)):
    lo, hi = region[0], region[1] if region[1] is not None else num_layers
    cand = list(range(max(1, lo), hi))
    if n_skip >= len(cand):
        return frozenset(cand)
    step = len(cand) / n_skip
    return frozenset(cand[int(i * step + step / 2)] for i in range(n_skip))


def middle_block_set(num_layers: int, n_skip: int):
    start = (num_layers - n_skip) // 2
    return frozenset(range(max(1, start), max(1, start) + n_skip))


def candidate_skip_sets(num_layers: int, counts=(4, 8, 12)):
    out = {}
    for n in counts:
        out[f"late{n}"] = early_exit_set(num_layers, num_layers - n)
        out[f"middle{n}"] = middle_block_set(num_layers, n)
        out[f"uniform{n}"] = uniform_skip_set(num_layers, n)
    return out


def clasp_select(h_full, layer_apply, n_skip: int, protect=(0,)):
    """CLaSp-style DP. h_full: list of L+1 vectors [D] (full-model hidden state before layer l, and after the last).
    layer_apply(l, x) -> x after layer l (caller-defined; exact use needs the layer + KV state of that token).
    dp[l][j] = state after l layers having skipped j of them, kept as the candidate (from 'run layer l on dp[l-1][j]'
    or 'skip layer l from dp[l-1][j-1]') with the highest cosine similarity to h_full[l+1].  Greedy-per-cell, as in
    CLaSp's O(L*M) DP.  Returns frozenset of skipped layers."""
    L = len(h_full) - 1
    dp = {(0, 0): (h_full[0], frozenset())}
    for l in range(L):
        for j in range(0, n_skip + 1):
            cands = []
            if (l, j) in dp:
                x, s = dp[(l, j)]
                cands.append((layer_apply(l, x), s))
            if j >= 1 and (l, j - 1) in dp and l not in protect:
                x, s = dp[(l, j - 1)]
                cands.append((x, s | {l}))
            if not cands:
                continue
            tgt = h_full[l + 1]
            best = max(cands, key=lambda c: F.cosine_similarity(c[0].float().flatten(), tgt.float().flatten(), dim=0).item())
            dp[(l + 1, j)] = best
    return dp[(L, n_skip)][1]


def delta_layer_apply_factory(h_full):
    """APPROXIMATION: assume each layer's residual increment is input-independent: apply(l,x)=x+(h[l+1]-h[l]).
    Makes the DP runnable from one verifier forward's hidden states without KV/attention replay."""
    return lambda l, x: x + (h_full[l + 1] - h_full[l])
