"""Enumeration + Dinkelbach + comparators.

Objective: maximise  N(c) / T(c)   with N = expected committed tokens per cycle (calibration table) and
T = expected cycle time (sum over scenarios of probability * optimal makespan) subject to memory and latency caps.
Ratio handled by Dinkelbach's iteration (lam <- N/T of the incumbent, accept c iff N - lam*T > 0). Exact schedule
solves are lazy: lower bounds LB(c) <= T(c) prune candidates whose N - lam*LB <= 0 (they cannot improve the incumbent),
so most configurations never reach CP-SAT.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

from . import config as C
from . import dag as D
from . import sched as S


@dataclass
class Space:
    gammas: tuple = (1, 2, 3, 4, 6, 8)
    modes: tuple = C.MODES
    skips: dict = field(default_factory=dict)       # name -> tuple(layers); informational, calibration decides


@dataclass
class Result:
    name: str
    cfg: object
    N: float
    T_us: float                  # expected cycle time
    tok_s: float
    n_solves: int = 0
    solve_s: float = 0.0
    schedules: list = field(default_factory=list)   # [(prob, Schedule)]
    notes: dict = field(default_factory=dict)

    def summary(self):
        return {"name": self.name, "cfg": self.cfg.label() if self.cfg else None, "N": round(self.N, 4),
                "T_us": round(self.T_us, 1), "tok_s": round(self.tok_s, 3), "cpsat_solves": self.n_solves,
                "solve_s": round(self.solve_s, 3), **self.notes}


class Evaluator:
    def __init__(self, cm, bucket, calib, n_groups=6, mem_cap_kib=None, lat_cap_us=None, time_limit_s=10.0,
                 workers=4):
        self.cm, self.bucket, self.calib = cm, bucket, calib
        self.G, self.mem_cap, self.lat_cap = n_groups, mem_cap_kib, lat_cap_us
        self.tl, self.workers = time_limit_s, workers
        self._dags = {}; self._opt = {}; self._ser = {}; self._lb = {}
        self.n_cpsat = 0; self.cpsat_s = 0.0

    # ---- acceptance ----
    def entry(self, cfg):
        return self.calib.get(self.bucket.workload, cfg.draft_key)

    def N(self, cfg):
        return self.entry(cfg).expected_committed(cfg.gamma)

    def h(self, cfg):
        e = self.entry(cfg)
        return e.carry_prob(cfg.gamma) if cfg.mode != "two_pass" else 0.0

    # ---- DAGs ----
    def dags(self, cfg):
        if cfg not in self._dags:
            self._dags[cfg] = D.scenarios(self.cm, self.bucket, cfg, self.h(cfg), self.G)
        return self._dags[cfg]

    def static_mem_ok(self, cfg):
        if self.mem_cap is None:
            return True
        return all(sum(b.nbytes for b in d.buffers) / 1024 <= self.mem_cap for _, d in self.dags(cfg))

    def T_serial(self, cfg):
        if cfg not in self._ser:
            self._ser[cfg] = sum(p * S.serial(d).makespan_us for p, d in self.dags(cfg))
        return self._ser[cfg]

    def T_lb(self, cfg):
        if cfg not in self._lb:
            self._lb[cfg] = sum(p * S.lower_bound_us(d) for p, d in self.dags(cfg))
        return self._lb[cfg]

    def optimise(self, cfg):
        """CP-SAT optimal schedules for each scenario. Returns (T_us, [(p, Schedule)]) or (inf, []) if infeasible
        (memory) or latency cap violated (worst-case scenario)."""
        if cfg in self._opt:
            return self._opt[cfg]
        scheds = []; T = 0.0
        for p, d in self.dags(cfg):
            sc = S.cpsat(d, self.mem_cap, self.tl, self.workers)
            self.n_cpsat += 1; self.cpsat_s += sc.solve_s
            scheds.append((p, sc))
            T += p * sc.makespan_us
        if any(not s.feasible for _, s in scheds) or (
                self.lat_cap is not None and max(s.makespan_us for _, s in scheds) > self.lat_cap):
            out = (float("inf"), scheds)
        else:
            out = (T, scheds)
        self._opt[cfg] = out
        return out

    def serial_result(self, name, cfg):
        sch = [(p, S.serial(d, self.mem_cap)) for p, d in self.dags(cfg)]
        T = sum(p * s.makespan_us for p, s in sch)
        N = self.N(cfg)
        bad = any(not s.feasible for _, s in sch) or (
            self.lat_cap is not None and max(s.makespan_us for _, s in sch) > self.lat_cap)
        if bad:
            T = float("inf")
        return Result(name, cfg, N, T, N / T * 1e6 if T < float("inf") else 0.0, 0, 0.0, sch)

    def opt_result(self, name, cfg):
        T, sch = self.optimise(cfg)
        N = self.N(cfg)
        return Result(name, cfg, N, T, N / T * 1e6 if T < float("inf") else 0.0, 0, 0.0, sch)


def enumerate_configs(ev: Evaluator, space: Space, modes=None, restrict=None):
    modes = modes or space.modes
    out = []
    for key in sorted(ev.calib.keys(ev.bucket.workload)):
        ab, r, sk = key
        for g in space.gammas:
            e = ev.calib.get(ev.bucket.workload, key)
            if g > e.gmax or ev.bucket.batch * (g + 1) > ev.cm.table.max_tokens or r > ev.cm.table.max_r:
                continue
            for m in modes:
                cfg = C.Config(g, ab, r, sk, m)
                if restrict is not None and not restrict(cfg):
                    continue
                if ev.static_mem_ok(cfg):
                    out.append(cfg)
    return out


def dinkelbach(ev: Evaluator, cands, lam0=0.0, eps=1e-12):
    """Maximise N/T over cands with lazy exact schedule solves. Returns (best_cfg|None, lam [tok/us], stats)."""
    t0 = time.perf_counter(); n0 = ev.n_cpsat
    N = {c: ev.N(c) for c in cands}
    LB = {c: ev.T_lb(c) for c in cands}
    # incumbent from the serial schedule (a feasible schedule, so the optimum can only be better)
    lam = lam0; best = None
    for c in cands:
        ts = ev.T_serial(c)
        if N[c] / ts > lam:
            lam = N[c] / ts
    solved = {}; iters = 0
    while True:
        pick = None; pv = eps
        for c in cands:
            if c in solved:
                continue
            v = N[c] - lam * LB[c]
            if v > pv:
                pick, pv = c, v
        if pick is None:
            break
        iters += 1
        T, _ = ev.optimise(pick)
        solved[pick] = T
        if T < float("inf") and N[pick] / T > lam + 1e-15:
            lam = N[pick] / T; best = pick
    if best is None and solved:
        fin = {c: N[c] / T for c, T in solved.items() if T < float("inf")}
        if fin:
            best = max(fin, key=fin.get); lam = max(lam, fin[best])
    stats = {"candidates": len(cands), "exact_solves_configs": len(solved), "pruned_by_bound": len(cands) - len(solved),
             "dinkelbach_iters": iters, "cpsat_calls": ev.n_cpsat - n0, "wall_s": time.perf_counter() - t0,
             "lambda_tok_per_s": lam * 1e6}
    return best, lam, stats


def nospec_reference(ev: Evaluator) -> Result:
    """Plain autoregressive decode (one verify-width-1 forward, no draft) -- context for all speedups."""
    cfg = C.Config(0, 0, 8, (), "two_pass")
    ds = D.build_cycle(ev.cm, ev.bucket, cfg, "B", ev.G)
    sc = S.cpsat(ds, None, ev.tl, ev.workers)
    return Result("no_speculation", cfg, 1.0, sc.makespan_us, 1e6 / sc.makespan_us, 1, sc.solve_s, [(1.0, sc)])


def run_comparators(ev: Evaluator, space: Space, baseline: C.Config):
    """fixed baseline / draft-only / schedule-only / independent / joint. Returns dict name -> Result."""
    out = {}
    t0 = time.perf_counter()
    if ev.entry(baseline) is None:
        raise KeyError(f"baseline {baseline.label()} not in calibration table")
    # 1. fixed baseline: fixed draft config, mode two_pass, in-order stream
    out["fixed_baseline"] = ev.serial_result("fixed_baseline", baseline)
    # 2. draft-only: optimise draft parameters under the baseline schedule policy (two_pass, serial)
    t = time.perf_counter()
    dcands = [c for c in enumerate_configs(ev, space, modes=("two_pass",))]
    best = max(dcands, key=lambda c: (ev.serial_result("x", c).tok_s))
    out["draft_only"] = ev.serial_result("draft_only", best)
    out["draft_only"].solve_s = time.perf_counter() - t
    # 3. schedule-only: baseline draft params, optimise schedule (mode + CP-SAT)
    n0, s0 = ev.n_cpsat, ev.cpsat_s; t = time.perf_counter()
    sc = [c for c in enumerate_configs(ev, space, restrict=lambda c: c.draft_key == baseline.draft_key
                                       and c.gamma == baseline.gamma)]
    b, lam, st = dinkelbach(ev, sc)
    out["schedule_only"] = _finish(ev, "schedule_only", b, n0, s0, t, st)
    # 4. independent: draft-only choice, then schedule optimisation of THAT draft config
    n0, s0 = ev.n_cpsat, ev.cpsat_s; t = time.perf_counter()
    dc = out["draft_only"].cfg
    ic = [c for c in enumerate_configs(ev, space, restrict=lambda c: c.draft_key == dc.draft_key and c.gamma == dc.gamma)]
    b, lam, st = dinkelbach(ev, ic)
    out["independent"] = _finish(ev, "independent", b, n0, s0, t, st)
    # 5. joint
    n0, s0 = ev.n_cpsat, ev.cpsat_s; t = time.perf_counter()
    allc = enumerate_configs(ev, space)
    b, lam, st = dinkelbach(ev, allc)
    out["joint"] = _finish(ev, "joint", b, n0, s0, t, st)
    # joint is a superset of every other comparator by construction; guard against solver-gap artefacts
    for k in ("schedule_only", "independent"):
        if out[k].tok_s > out["joint"].tok_s + 1e-9:
            out["joint"].notes["warning"] = f"{k} > joint by {out[k].tok_s/out['joint'].tok_s-1:.2e} (CP-SAT gap)"
    return out


def _finish(ev, name, cfg, n0, s0, t0, stats):
    r = ev.opt_result(name, cfg)
    r.n_solves = ev.n_cpsat - n0; r.solve_s = time.perf_counter() - t0
    r.notes = {"search": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats.items()}}
    return r


def speedup_table(res: dict, ref="fixed_baseline"):
    b = res[ref].tok_s
    return {k: round(v.tok_s / b, 4) for k, v in res.items()}
