"""Exact North serving engine (one H100, static batches): the measured H24/H25 configuration behind one call.

    eng = Engine(model, Lmax=4096)                    # captures graphs + cost tables once
    outs, stats = eng.generate([ids_a, ids_b], max_new=192)

Policy (each rule is a measurement, see RESULTS.md):
  * every executor is exact v2: attention || MoE streams, H100 tiles, attention chunk scaled with Lmax (H24);
  * B == 1           -> FastNorth + PLD (sam2, g=12) with pld.CostPolicy over this engine's measured T=1..13 graph costs; the proposer
                        index is built in a background thread during prefill (H25: never below 0.98x of plain on the mix, 1.08x at 32K/128K);
  * 2 <= B <= spec_bmax -> batched exact executor + cost-capped PLD (H19 `_pick_T`). Default spec_bmax=1, i.e. OFF: over all 16 pool
                        pairs at B=2 it gives 1.035x geomean but loses on 5 (worst 0.943x, H26); pass spec_bmax=2 to trade worst case for mean;
  * B > spec_bmax    -> batched exact executor, plain (speculation loses from B=4 even capped, H20/H24).
Guarantee: every path returns exactly the greedy tokens of the SAME executor's plain decode (exact mode). `generate_plain` is that
reference. Batched rows are not guaranteed bitwise equal to the B=1 executor (different kernels), bench_serve.py measures it.
Not here: continuous batching (requests joining mid-decode), sampling, prefix caching."""
import time
import threading
import torch

import pld
import batch_exact
import fast_kernels_batch as fkb
import fast_kernels_exact as fx
import fast_kernels_x2 as f2
from northstar.refmodel import RefNorth
from graph_north import GraphNorth, time_graph
from fast_north import FastNorth

TMAX = 13


def _pick_T(dr, q, cost, done):
    """Batched cost-aware round width (same rule as bench_batch._pick_T, H19)."""
    tmax = 1 + max(len(d) for d in dr)
    best, bT = -1.0, 1
    for T in range(1, tmax + 1):
        if T not in cost:
            break
        tok = 0.0
        for b, d in enumerate(dr):
            if done[b]:
                continue
            m = min(len(d), T - 1); qb = q[b]
            tok += 1.0 + (qb * (1 - qb ** m) / (1 - qb) if qb < 1 else m)
        v = tok / cost[T]
        if v > best:
            best, bT = v, T
    return bT


class Engine:
    def __init__(self, model, Lmax, bmax=32, spec_bmax=1, proposer="sam2", gamma=12, speculate=True):
        self.m, self.Lmax, self.bmax, self.spec_bmax = model, Lmax, bmax, spec_bmax
        self.name, self.g, self.speculate = proposer, gamma, speculate
        eos = model.config.eos_token_id
        self.eos = eos if isinstance(eos, int) else eos[0]
        # batched executor configuration (process-global module switches)
        fkb.TILE_GRID = 6
        batch_exact.STREAMS = True
        fx.LIN_CFG.update(f2.TUNED_LIN_CFG)
        self.ref = RefNorth(model)
        self.X = FastNorth(self.ref, Lmax, exact=True)
        self.X.set_x2(**f2.recommended(Lmax))
        self.BX = batch_exact.BatchExact(self.X)
        if Lmax <= 16384:
            gb = GraphNorth(self.ref, Lmax, "bmm")
            self._prefill = lambda ids, kv: gb.eager_prefill(ids, kv)
        else:
            from longctx_north import LCNorth
            ln = LCNorth(self.ref, Lmax)
            self._prefill = lambda ids, kv: ln.prefill(ids, kv, chunk=1024 if Lmax <= 40000 else 512)
        self.kvs = []
        self.steps1, self.cost1 = None, None
        self.bsteps, self.bcost = {}, {}
        self.dev = self.ref.device

    # ------------------------------------------------------------------ capture (lazy, once per shape)
    def _kv(self, n):
        while len(self.kvs) < n:
            self.kvs.append(self.X.new_kv())
        return self.kvs[:n]

    def _dummy(self, T, b=0):
        return [1000 + 37 * b + t for t in range(T)]  # distinct tokens: realistic expert union for the cost table

    def _single(self):
        if self.steps1 is None:
            kv = self._kv(1)[0]
            Ts = range(1, TMAX + 1) if self.speculate else (1,)
            self.steps1 = {T: self.X.make_step(T, kv, logits="all" if T > 1 else "last") for T in Ts}
            self.cost1 = {}
            for T in Ts:  # timed at positions 0..T-1 of slot 0: overwritten by the next prefill
                self.steps1[T].set(self._dummy(T), 0); self.cost1[T] = time_graph(self.steps1[T], n=10, warm=2)[0]
        return self.steps1

    def _batch(self, B):
        if B not in self.bsteps:
            kvs = self._kv(B)
            Ts = range(1, TMAX + 1) if (self.speculate and B <= self.spec_bmax) else (1,)
            self.bsteps[B] = {T: self.BX.make_step(B, T, kvs) for T in Ts}
            self.bcost[B] = {}
            for T in Ts:
                self.bsteps[B][T].set([self._dummy(T, b) for b in range(B)], [0] * B)
                self.bcost[B][T] = time_graph(self.bsteps[B][T], n=8, warm=2)[0]
        return self.bsteps[B]

    def warmup(self, sizes=(1,)):
        for B in sizes:
            self._single() if B == 1 else self._batch(B)
        torch.cuda.synchronize()

    # ------------------------------------------------------------------ prefill (+ proposer index in parallel)
    def _prefill_all(self, prompts, kvs, build):
        th, pool = (None, None)
        if build:
            pool = [None] * len(prompts)

            def work():
                for i, p in enumerate(prompts):
                    q = pld.make(self.name); q.extend(list(p)); pool[i] = q
            th = threading.Thread(target=work, daemon=True); th.start()
        t0 = time.perf_counter()
        for p, kv in zip(prompts, kvs):
            self._prefill(list(p[:-1]), kv)
        torch.cuda.synchronize()
        pre = time.perf_counter() - t0
        if th is not None:
            th.join()
        return pre, time.perf_counter() - t0 - pre, pool

    def _check(self, prompts, max_new):
        assert 1 <= len(prompts) <= self.bmax, len(prompts)
        for p in prompts:
            assert len(p) + max_new + 16 < self.Lmax, (len(p), max_new, self.Lmax)

    # ------------------------------------------------------------------ public API
    @torch.no_grad()
    def generate(self, prompts, max_new=192):
        """prompts: list of token-id lists (each ending with the generation prompt). Returns (outputs, stats)."""
        self._check(prompts, max_new)
        B = len(prompts)
        spec = self.speculate and B <= self.spec_bmax
        if B == 1:
            steps = self._single()
            kv = self._kv(1)[0]
            pre, wait, pool = self._prefill_all(prompts, [kv], spec)
            p = list(prompts[0])

            def verify(tcur, drafts):
                stp = steps[1 + len(drafts)]; stp.set([tcur[-1]] + drafts, len(tcur) - 1)
                return stp.run()[0].tolist() if drafts else [int(stp.run()[0, 0])]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            if spec:
                out, rounds = pld.spec_loop(pool[0], p, max_new, self.g, verify, self.eos, policy=pld.CostPolicy(self.g, self.cost1), prebuilt=True)
            else:
                out, rounds = self._plain1(steps[1], p, max_new), []
            dec = time.perf_counter() - t0
            return [out], dict(B=1, path="single+pld_cost" if spec else "single_plain", prefill_s=pre, index_wait_s=wait, decode_s=dec,
                               tokens=len(out), decode_tok_s=len(out) / dec, rounds=len(rounds) or len(out),
                               mean_T=(sum(d + 1 for d, _ in rounds) / len(rounds)) if rounds else 1.0)
        steps = self._batch(B)
        kvs = self._kv(B)
        pre, wait, pool = self._prefill_all(prompts, kvs, spec)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        if spec:
            outs, info = self._spec_b(steps, prompts, max_new, pool, self.bcost[B])
        else:
            outs, info = self._plain_b(steps[1], prompts, max_new), {}
        dec = time.perf_counter() - t0
        n = sum(len(o) for o in outs)
        return outs, dict(B=B, path="batched+pld_cost" if spec else "batched_plain", prefill_s=pre, index_wait_s=wait, decode_s=dec, tokens=n,
                          decode_tok_s=n / dec, **info)

    @torch.no_grad()
    def generate_plain(self, prompts, max_new=192):
        """Reference: plain greedy decode on the same executor generate() uses for this batch size."""
        self._check(prompts, max_new)
        B = len(prompts)
        if B == 1:
            steps = self._single(); kv = self._kv(1)[0]
            pre, _, _ = self._prefill_all(prompts, [kv], False)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out = self._plain1(steps[1], list(prompts[0]), max_new)
            dec = time.perf_counter() - t0
            return [out], dict(B=1, path="single_plain", prefill_s=pre, decode_s=dec, tokens=len(out), decode_tok_s=len(out) / dec)
        steps = self._batch(B); kvs = self._kv(B)
        pre, _, _ = self._prefill_all(prompts, kvs, False)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        outs = self._plain_b(steps[1], prompts, max_new)
        dec = time.perf_counter() - t0
        n = sum(len(o) for o in outs)
        return outs, dict(B=B, path="batched_plain", prefill_s=pre, decode_s=dec, tokens=n, decode_tok_s=n / dec)

    # ------------------------------------------------------------------ decode loops
    def _plain1(self, step, p, max_new):
        out, cur, P = [], p[-1], len(p)
        for i in range(max_new):
            step.set([cur], P - 1 + i); cur = int(step.run()[0, 0]); out.append(cur)
            if cur == self.eos:
                break
        return out

    def _plain_b(self, step, prompts, max_new):
        B = len(prompts)
        cur = [p[-1] for p in prompts]; pos = [len(p) - 1 for p in prompts]
        out = [[] for _ in range(B)]; done = [False] * B
        for _ in range(max_new):
            step.set([[c] for c in cur], pos, [[not d] for d in done])
            pred = step.run().tolist()
            for b in range(B):
                pos[b] += 1
                if done[b]:
                    continue
                tk = pred[b][0]; out[b].append(tk); cur[b] = tk
                if tk == self.eos or len(out[b]) >= max_new:
                    done[b] = True
            if all(done):
                break
        return out

    def _spec_b(self, steps, prompts, max_new, props, cost):
        B = len(prompts)
        qa = [1.0] * B; qr = [1.0] * B
        tcur = [list(p) for p in prompts]; out = [[] for _ in range(B)]; done = [False] * B
        rounds, Tsum = 0, 0
        while not all(done):
            dr = [(props[b].propose(self.g)[0] if not done[b] else []) for b in range(B)]
            T = _pick_T(dr, [qa[b] / (qa[b] + qr[b]) for b in range(B)], cost, done)
            dr = [d[:T - 1] for d in dr]
            T = 1 + max(len(d) for d in dr)
            ids, pos, val = [], [], []
            for b in range(B):
                row = [tcur[b][-1]] + dr[b]
                ids.append(row + [row[-1]] * (T - len(row))); pos.append(len(tcur[b]) - 1)
                val.append([(not done[b]) and t < len(row) for t in range(T)])
            steps[T].set(ids, pos, val)
            pred = steps[T].run().tolist()
            rounds += 1; Tsum += T
            for b in range(B):
                if done[b]:
                    continue
                d = dr[b]; a = 0
                while a < len(d) and d[a] == pred[b][a]:
                    a += 1
                new = d[:a] + [pred[b][a]]
                if self.eos in new:
                    new = new[:new.index(self.eos) + 1]; done[b] = True
                qa[b] += a; qr[b] += int(a < len(d))
                tcur[b] += new; out[b] += new; props[b].extend(new)
                if len(out[b]) >= max_new:
                    done[b] = True
        return [o[:max_new] for o in out], dict(rounds=rounds, mean_T=Tsum / max(1, rounds))
