"""Evaluate a tage proposer on recorded (prompt_ids, out_ids) pairs.

Protocol: proposer history is warmed with the prompt (committed, not scored), then decoding of out_ids is simulated
with an oracle verifier (greedy: a draft token is accepted iff it equals North's recorded token). A step drafts d
tokens only if len(draft) >= min_draft; otherwise it is a plain decode step (draft []).

Cost model (relative to plain decode step = 1.0): verify with block K = d+1 tokens costs COST(K), interpolated from
MEASURED megakernel numbers in RESULTS.md H5 (plain 3.32 ms; K=3/5/9 -> 4.7/5.8/7.9 ms).  Draft compute is zero
(CPU table lookup); CPU lookup latency is NOT modelled (INFERRED speedups, not wall-clock)."""
from __future__ import annotations

import numpy as np

PLAIN_MS = 3.32
COST_PTS = [(1, 3.32), (3, 4.7), (5, 5.8), (9, 7.9)]


def cost_ms(K, pts=None):
    pts = pts or COST_PTS
    xs, ys = zip(*pts)
    if K > xs[-1]:  # linear extrapolation from the last segment
        s = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return ys[-1] + s * (K - xs[-1])
    return float(np.interp(K, xs, ys))


# Named step-cost models: relative cost of a verify step with K tokens (K=1 plain decode = 1.0).
#  megakernel : MEASURED on our H100 megakernel (RESULTS.md H5) -> cost_ms(K)/PLAIN_MS
#  flat2/flat5: INFERRED stand-ins for the vLLM/RTX regime of inference-experiment (decode step bound by expert-weight
#               reads, so extra verified tokens are nearly free): 1 + c*(K-1), c = 2% / 5%
COST_MODELS = {
    "megakernel": lambda K: cost_ms(K) / PLAIN_MS,
    "flat2": lambda K: 1 + 0.02 * (K - 1),
    "flat5": lambda K: 1 + 0.05 * (K - 1),
}


def run_one(proposer, prompt, out, kmax=8, min_draft=1, keep_state=False):
    proposer.start_doc(keep_state=keep_state)
    for t in prompt:
        proposer.commit(t)
    pos, n = 0, len(out)
    steps = tokens = drafted = accepted = verified_steps = 0
    ms = 0.0
    kcount = {}
    while pos < n:
        draft = proposer.propose(kmax, 0.0)[0][: n - pos]
        if len(draft) < min_draft:
            draft = []
        a = 0
        while a < len(draft) and draft[a] == out[pos + a]:
            a += 1
        K = len(draft) + 1
        ms += cost_ms(K)
        kcount[K] = kcount.get(K, 0) + 1
        steps += 1
        verified_steps += bool(draft)
        drafted += len(draft)
        accepted += a
        take = min(a + 1, n - pos)
        for j in range(take):
            proposer.commit(out[pos + j])
        tokens += take
        pos += take
    return dict(tokens=tokens, steps=steps, drafted=drafted, accepted=accepted, verified_steps=verified_steps, ms=ms,
                kcount=kcount)


def speedup_from(kcounts, tokens, model="megakernel"):
    """tokens / sum_steps cost(K) under a named cost model (plain decode = 1.0 per token)."""
    f = COST_MODELS[model]
    return tokens / sum(n * f(K) for K, n in kcounts.items())


def evaluate(make, records, **kw):
    """records: list of dict(id, kind, prompt_ids, out_ids). Returns per-record rows + pooled summary."""
    rows = []
    for r in records:
        s = run_one(make(), r["prompt_ids"], r["out_ids"], **kw)
        s.update(id=r["id"], kind=r.get("kind"), speedup=s["tokens"] * PLAIN_MS / s["ms"])
        rows.append(s)
    T = sum(x["tokens"] for x in rows); M = sum(x["ms"] for x in rows)
    agg = dict(tokens=T, steps=sum(x["steps"] for x in rows), tokens_per_step=T / sum(x["steps"] for x in rows),
               draft_rate=sum(x["verified_steps"] for x in rows) / sum(x["steps"] for x in rows),
               precision=sum(x["accepted"] for x in rows) / max(1, sum(x["drafted"] for x in rows)),
               speedup_pooled=T * PLAIN_MS / M, speedup_mean=float(np.mean([x["speedup"] for x in rows])),
               speedup_min=float(min(x["speedup"] for x in rows)), n=len(rows))
    return rows, agg
