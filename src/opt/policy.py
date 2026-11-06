"""Policy cache keyed by workload bucket. Solve offline (or occasionally); serving does an O(1) dict lookup."""
from __future__ import annotations

import json
import time

from . import config as C
from . import solve as SV


def quantize_ctx(ctx, grid=(2048, 8192, 32768, 131072)):
    for g in grid:
        if ctx <= g:
            return g
    return grid[-1]


def policy_from_result(res, bucket, ev, search):
    cfg = res.cfg
    return {
        "bucket": bucket.key, "config": {"gamma": cfg.gamma, "attn_budget": cfg.attn_budget, "r": cfg.r,
                                         "skip": list(cfg.skip), "mode": cfg.mode},
        "pred_tokens_per_cycle": res.N, "pred_cycle_us": res.T_us, "pred_tok_s": res.tok_s,
        "schedules": [{"prob": p, "makespan_us": s.makespan_us, "status": s.status, "variant": s.variant,
                       "start_us": s.start_us, "work_us_by_source": s.src_us} for p, s in res.schedules],
        "search": search, "cost_model": ev.cm.fingerprint(), "calibration": ev.calib.fingerprint(),
        "acceptance_synthetic": ev.calib.any_synthetic(), "solved_at": time.time(),
    }


class PolicyCache:
    def __init__(self, path=None):
        self.path = path; self.table = {}
        if path:
            try:
                with open(path) as f:
                    self.table = json.load(f)
            except (OSError, ValueError):
                self.table = {}

    def key(self, bucket):
        return C.Bucket(bucket.workload, quantize_ctx(bucket.ctx), bucket.batch, bucket.routing).key

    def lookup(self, bucket):
        return self.table.get(self.key(bucket))

    def solve(self, bucket, make_evaluator, space, force=False):
        """Return cached policy for the bucket or run the joint solve and store it."""
        k = self.key(bucket)
        if k in self.table and not force:
            return self.table[k]
        qb = C.Bucket(bucket.workload, quantize_ctx(bucket.ctx), bucket.batch, bucket.routing)
        ev = make_evaluator(qb)
        t0 = time.perf_counter()
        cands = SV.enumerate_configs(ev, space)
        best, lam, st = SV.dinkelbach(ev, cands)
        if best is None:
            raise RuntimeError(f"no feasible configuration for {k}")
        res = ev.opt_result("joint", best)
        st["total_wall_s"] = time.perf_counter() - t0
        pol = policy_from_result(res, qb, ev, st)
        self.table[k] = pol
        self.save()
        return pol

    def save(self):
        if self.path:
            with open(self.path, "w") as f:
                json.dump(self.table, f, indent=1)
