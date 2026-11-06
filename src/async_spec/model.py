"""Cost model for Phase 5 (Saguaro-style async self-speculation) on one device.

Everything here is ARITHMETIC on measured inputs; nothing is a hardware result.
Provenance tags: MEASURED (read from results/*.jsonl or RESULTS.md tables),
INFERRED (arithmetic on measured numbers), INVENTED (swept, no measurement).

Round structure (greedy, exact): draft gamma steps, verify K=gamma+1 tokens,
commit 1+a tokens (a = accepted draft tokens).
Serial:      T_s = gamma*d + V
Async:       while verify of block n runs, the SAME device also drafts block n+1
             for k outcome branches (batched k-wide, gamma steps). On a branch
             hit the next draft is already done; on a miss the work is wasted
             and the draft is redone serially after verify.

Bandwidth sharing: a job of standalone time t keeps HBM busy for beta*t
(beta = bandwidth-busy fraction, MEASURED 0.60 for the megakernel, INFERRED
~0.45 for the graphed PyTorch executor). Two concurrent jobs demanding
beta_V + beta_B > 1 of HBM both slow by s = max(1, beta_V + beta_B). Only
non-bandwidth time (launch/latency gaps) can be hidden. Optimistic: assumes
perfect interleaving and no L2/SM contention beyond bandwidth.
"""
from __future__ import annotations
import itertools

# MEASURED: megakernel H100 batch-size step times (ms), results/megakernel_H100_bs_*.log
MK_BS_MS = {1: 3.32, 2: 4.10, 4: 5.29, 8: 7.39}


def batch_factor(k: int, table=MK_BS_MS) -> float:
    """Cost of k independent sequences in one step relative to 1 (>=1).
    Piecewise linear on the measured BS curve; linear extrapolation beyond 8
    with the measured 4->8 slope (extrapolation is INVENTED)."""
    if k <= 1:
        return 1.0
    xs = sorted(table)
    base = table[1]
    if k >= xs[-1]:
        slope = (table[xs[-1]] - table[xs[-2]]) / (xs[-1] - xs[-2])
        return (table[xs[-1]] + slope * (k - xs[-1])) / base
    for a, b in zip(xs, xs[1:]):
        if a <= k <= b:
            t = (k - a) / (b - a)
            return (table[a] + t * (table[b] - table[a])) / base


def length_dist(apos):
    """apos[i] = P(accepted >= i+1) (MEASURED; sums to mean_accepted).
    Returns P(a=j) for j=0..gamma."""
    g = len(apos)
    ge = [1.0] + list(apos) + [0.0]
    return [ge[j] - ge[j + 1] for j in range(g + 1)]


def outcome_probs(apos, q_rej=0.05, rej_decay=0.5, extra_full=2):
    """Joint probabilities of (accept length, bonus token) outcomes a branch
    could target, sorted descending.
    full-accept guess: draft's own next token; hit prob proxy = measured
    conditional accept ratio apos[-1]/apos[-2] (INFERRED proxy).
    reject guesses: verifier's correction token differs from draft token;
    q_rej for the best alternative, geometric decay for more (INVENTED)."""
    g = len(apos)
    pa = length_dist(apos)
    prev = apos[-2] if g > 1 else 1.0
    q_full = apos[-1] / prev if prev > 0 else 0.0
    out = []
    # full accept: first guess q_full; further guesses share what is left (INVENTED split)
    rem = 1 - q_full
    for m in range(extra_full):
        pm = q_full if m == 0 else rem * 0.3 * (0.5 ** (m - 1))  # 0.3 split INVENTED
        out.append((pa[g] * pm, ("full", m)))
    for j in range(g):
        for m in range(3):
            out.append((pa[j] * q_rej * (rej_decay ** m), (j, m)))
    out.sort(key=lambda x: -x[0])
    return out


def hit_prob(apos, k, **kw):
    if k <= 0:
        return 0.0
    return min(1.0, sum(p for p, _ in outcome_probs(apos, **kw)[:k]))


def round_times(gamma, d, V, k, h, beta_v, beta_b, fk=None):
    """Return (T_serial, T_async_expected) in ms per round.
    d: draft step ms (k=1), V: verify ms, k: branches, h: branch hit prob."""
    Ts = gamma * d + V
    if k <= 0:
        return Ts, Ts
    fk = batch_factor(k) if fk is None else fk
    Bt = gamma * d * fk            # standalone time of k-wide branch drafting
    s = max(1.0, beta_v + beta_b)  # bandwidth-sharing slowdown
    Vc = V * s                     # verify wall time under contention
    if Bt * s <= Vc:
        hit_t = Vc
    else:
        hit_t = Vc + (Bt - Vc / s)  # branch finishes alone after verify
    miss_t = Vc + gamma * d
    Ta = h * hit_t + (1 - h) * miss_t
    return Ts, Ta


def speedup(gamma, d, V, k, apos, beta, fk_scale=1.0, **kw):
    h = hit_prob(apos, k, **kw)
    fk = 1 + (batch_factor(k) - 1) * fk_scale if k > 0 else 1.0
    Ts, Ta = round_times(gamma, d, V, k, h, beta, beta, fk)
    return Ts / Ta, h, Ts, Ta


def upper_bound(gamma, d, V):
    """h=1 and branch work free and fully hidden: T_a = V."""
    return (gamma * d + V) / V
