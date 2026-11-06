"""Plain greedy decode and greedy self-speculation (draft = same weights, cheaper mode).

Convention (both paths): with n committed tokens t[0:n], the *verifier* cache holds KV for t[0:n-1];
the last committed token is the pending input.  The *draft* cache holds KV for t[0:m], m <= n-1,
produced by the DRAFT computation (fixed layer skip / expert top-r), kept separate from the verifier's.

Round (gamma drafts):
  1. draft catch-up: feed t[m:n] (>=1 tokens) to draft cache -> logits -> d1 = argmax
  2. d2..dgamma: feed d_{i-1} single-token draft steps (draft cache grows by gamma-1)
  3. verify: feed [t[n-1], d1..dgamma] (gamma+1 tokens) in ONE full-model forward -> pred[0..gamma]
     a = longest prefix with d_i == pred[i-1];   commit d1..d_a + pred[a]  (pred[a] is the correction
     token if a<gamma else the bonus token) => a+1 committed tokens per round.
  4. rollback: verifier len = n+a  (== new_n - 1);  draft len = n + min(a, gamma-1).
Greedy only: no RNG is consumed (RNG accounting is a Phase-1b sampling item, not implemented).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .refmodel import FULL, DraftMode, RefNorth


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _top2_margin(logits_row):
    t = torch.topk(logits_row.float(), 2)
    return (t.values[0] - t.values[1]).item()


@torch.no_grad()
def plain_greedy(ref: RefNorth, prompt, n_new, eos_id=None, chunk=1024, timed=True):
    """prompt: list[int]. One token per forward. Returns dict(tokens, margins, prefill_s, step_s)."""
    dev = ref.device
    P = len(prompt)
    cache = ref.new_cache(1, P + n_new + 2, chunk=chunk)
    ids = torch.tensor([prompt], device=dev)
    _sync(dev)
    t0 = time.perf_counter()
    ref.prefill(ids[:, : P - 1], cache, FULL, chunk, logits="none") if P > 1 else None
    _sync(dev)
    prefill_s = time.perf_counter() - t0
    toks, margins, step_s = [], [], []
    cur = prompt[-1]
    for _ in range(n_new):
        t1 = time.perf_counter()
        lg = ref.forward(torch.tensor([[cur]], device=dev), cache, cache.len, FULL, logits="last")
        cache.len += 1
        row = lg[0, 0]
        nxt = int(row.argmax())  # host sync
        step_s.append(time.perf_counter() - t1)
        margins.append(_top2_margin(row))
        toks.append(nxt)
        cur = nxt
        if eos_id is not None and nxt == eos_id:
            break
    return dict(tokens=toks, margins=margins, prefill_s=prefill_s, step_s=step_s)


@dataclass
class RoundStat:
    accepted: int
    committed: int
    t_catchup: float
    t_draft: float
    t_verify: float
    t_rollback: float


class SelfSpec:
    def __init__(self, ref: RefNorth, draft: DraftMode, gamma: int, chunk: int = 1024, override=None):
        """override(pos, i, tok) -> tok lets tests force draft tokens (pos = absolute position, i = block index)."""
        self.ref, self.draft, self.gamma, self.chunk, self.override = ref, draft, gamma, chunk, override

    @torch.no_grad()
    def generate(self, prompt, n_new, eos_id=None):
        ref, dev, g = self.ref, self.ref.device, self.gamma
        P = len(prompt)
        max_len = P + n_new + g + 2
        vcache = ref.new_cache(1, max_len, chunk=self.chunk)
        dlayers = [i for i in range(ref.cfg.num_hidden_layers) if i not in self.draft.skip_layers]
        dcache = ref.new_cache(1, max_len, layers=dlayers, chunk=self.chunk)
        t = list(prompt)
        ids = torch.tensor([prompt], device=dev)
        _sync(dev)
        t0 = time.perf_counter()
        if P > 1:
            ref.prefill(ids[:, : P - 1], vcache, FULL, self.chunk, logits="none")
        _sync(dev)
        t_vpre = time.perf_counter() - t0
        t0 = time.perf_counter()
        if P > 1:
            ref.prefill(ids[:, : P - 1], dcache, self.draft, self.chunk, logits="none")
        _sync(dev)
        t_dpre = time.perf_counter() - t0
        stats: list[RoundStat] = []
        out: list[int] = []
        done = False
        while len(out) < n_new and not done:
            n = len(t)
            # 1. catch-up (may be >1 token); 2. gamma-1 more draft steps
            t1 = time.perf_counter()
            feed = torch.tensor([t[dcache.len : n]], device=dev)
            m0 = dcache.len
            # catch-up could exceed ring chunk only on first round if P huge; prefill() chunks it
            lg = ref.prefill(feed, dcache, self.draft, self.chunk, logits="last")
            nxt = int(lg[0, 0].argmax())
            if self.override:
                nxt = self.override(n, 0, nxt)
            _sync(dev)
            t_catch = time.perf_counter() - t1
            drafts = [nxt]
            t1 = time.perf_counter()
            for i in range(1, g):
                lg = ref.forward(torch.tensor([[drafts[-1]]], device=dev), dcache, dcache.len, self.draft, logits="last")
                dcache.len += 1
                nx = int(lg[0, 0].argmax())
                if self.override:
                    nx = self.override(n + i, i, nx)
                drafts.append(nx)
            _sync(dev)
            t_dr = time.perf_counter() - t1
            # 3. verify
            t1 = time.perf_counter()
            vin = torch.tensor([[t[n - 1]] + drafts], device=dev)
            vl = ref.forward(vin, vcache, n - 1, FULL, logits="all")
            pred = vl[0].argmax(-1).tolist()
            t_ver = time.perf_counter() - t1
            a = 0
            while a < g and drafts[a] == pred[a]:
                a += 1
            new = drafts[:a] + [pred[a]]
            # 4. rollback
            t1 = time.perf_counter()
            vcache.len = n + a
            dcache.len = n + min(a, g - 1)
            t_rb = time.perf_counter() - t1
            if eos_id is not None and eos_id in new:
                new = new[: new.index(eos_id) + 1]
                done = True
            t.extend(new)
            out.extend(new)
            stats.append(RoundStat(a, len(new), t_catch, t_dr, t_ver, t_rb))
        out = out[:n_new]
        return dict(tokens=out, rounds=stats, verifier_prefill_s=t_vpre, draft_prefill_s=t_dpre,
                    gen_s=sum(s.t_catchup + s.t_draft + s.t_verify + s.t_rollback for s in stats))
