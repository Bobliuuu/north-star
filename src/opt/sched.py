"""Schedulers for a cycle DAG: serial in-order baseline, lower bounds, and CP-SAT optimal schedule.

CP-SAT model: one interval per task (optional intervals when a task has several implementation variants, e.g. attention
tiles), precedence from the DAG, two cumulative resources (HBM bandwidth %, SM %; capacity 100 each) so overlapped tasks
are NOT additive, and a memory cumulative over buffer lifetimes (producer start .. last consumer end). Minimise makespan.
Durations are integer tenths of a microsecond.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ortools.sat.python import cp_model

SCALE = 10


@dataclass
class Schedule:
    makespan_us: float
    start_us: dict
    variant: dict
    status: str                  # SERIAL | OPTIMAL | FEASIBLE | INFEASIBLE
    bound_us: float = 0.0
    solve_s: float = 0.0
    src_us: dict = field(default_factory=dict)   # work (sum of chosen-variant durations) by provenance

    @property
    def feasible(self):
        return self.status != "INFEASIBLE"


def _work_by_source(dag, variant):
    out = {}
    for t in dag.tasks:
        v = next(x for x in t.variants if x.name == variant[t.id])
        for k, us in v.src_us:
            out[k] = out.get(k, 0.0) + us
    return out


def serial(dag, mem_cap_kib=None) -> Schedule:
    """In-order single-stream execution, default (first) variant: what a plain PyTorch/CUDA-graph stream does."""
    t = 0.0; start = {}; var = {}
    for task in dag.tasks:
        v = task.variants[0]
        start[task.id] = t; var[task.id] = v.name; t += v.us
    st = "SERIAL"
    if mem_cap_kib is not None and sum(b.nbytes for b in dag.buffers) / 1024 > mem_cap_kib:
        st = "INFEASIBLE"
    return Schedule(t, start, var, st, t, 0.0, _work_by_source(dag, var))


def lower_bound_us(dag) -> float:
    """max(critical path with fastest variants, bandwidth-area bound, SM-area bound). Valid for any schedule."""
    fin = {}
    for t in dag.tasks:
        d = min(v.us for v in t.variants)
        fin[t.id] = max((fin[x] for x in t.deps), default=0.0) + d
    cp = max(fin.values())
    bw = sum(min(v.us * v.bw for v in t.variants) for t in dag.tasks) / 100.0
    sm = sum(min(v.us * v.sm for v in t.variants) for t in dag.tasks) / 100.0
    return max(cp, bw, sm)


def cpsat(dag, mem_cap_kib=None, time_limit_s=10.0, workers=4, rel_gap=0.001, hint=True) -> Schedule:
    t0 = time.perf_counter()
    m = cp_model.CpModel()
    ser = serial(dag)
    horizon = int(ser.makespan_us * SCALE) + 1
    start = {}; end = {}; choice = {}
    bw_iv, bw_d, sm_d = [], [], []
    for t in dag.tasks:
        s = m.NewIntVar(0, horizon, f"s_{t.id}"); start[t.id] = s
        if len(t.variants) == 1:
            v = t.variants[0]; sz = max(1, int(round(v.us * SCALE)))
            e = s + sz; end[t.id] = e
            iv = m.NewFixedSizeIntervalVar(s, sz, f"i_{t.id}")
            bw_iv.append(iv); bw_d.append(v.bw); sm_d.append(v.sm)
            choice[t.id] = [(v, None)]
        else:
            lits = []
            ev = m.NewIntVar(0, horizon, f"e_{t.id}"); end[t.id] = ev
            ch = []
            for v in t.variants:
                b = m.NewBoolVar(f"b_{t.id}_{v.name}"); lits.append(b)
                sz = max(1, int(round(v.us * SCALE)))
                iv = m.NewOptionalFixedSizeIntervalVar(s, sz, b, f"i_{t.id}_{v.name}")
                bw_iv.append(iv); bw_d.append(v.bw); sm_d.append(v.sm)
                m.Add(ev == s + sz).OnlyEnforceIf(b)
                ch.append((v, b))
            m.AddExactlyOne(lits); choice[t.id] = ch
    for t in dag.tasks:
        for d in t.deps:
            m.Add(start[t.id] >= end[d])
    m.AddCumulative(bw_iv, bw_d, 100)
    m.AddCumulative(bw_iv, sm_d, 100)
    if mem_cap_kib is not None and dag.buffers:
        ivs, dem = [], []
        for b in dag.buffers:
            be = m.NewIntVar(0, horizon, f"be_{b.name}")
            for et in b.end_tasks:
                m.Add(be >= end[et])
            m.Add(be >= start[b.start_task] + 1)
            sz = m.NewIntVar(1, horizon + 1, f"bsz_{b.name}")
            m.Add(sz == be - start[b.start_task])
            ivs.append(m.NewIntervalVar(start[b.start_task], sz, be, f"buf_{b.name}"))
            dem.append(max(1, b.nbytes // 1024))
        m.AddCumulative(ivs, dem, int(mem_cap_kib))
    mk = m.NewIntVar(0, horizon, "mk")
    for t in dag.tasks:
        m.Add(mk >= end[t.id])
    m.Minimize(mk)
    if hint:
        for t in dag.tasks:
            m.AddHint(start[t.id], int(ser.start_us[t.id] * SCALE))
            for v, b in choice[t.id]:
                if b is not None:
                    m.AddHint(b, 1 if v.name == t.variants[0].name else 0)
    sol = cp_model.CpSolver()
    sol.parameters.max_time_in_seconds = time_limit_s
    sol.parameters.num_workers = workers
    sol.parameters.relative_gap_limit = rel_gap
    sol.parameters.random_seed = 1
    r = sol.Solve(m)
    wall = time.perf_counter() - t0
    if r not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return Schedule(float("inf"), {}, {}, "INFEASIBLE", 0.0, wall)
    st_ = {t.id: sol.Value(start[t.id]) / SCALE for t in dag.tasks}
    var = {}
    for t in dag.tasks:
        if len(t.variants) == 1:
            var[t.id] = t.variants[0].name
        else:
            var[t.id] = next(v.name for v, b in choice[t.id] if sol.Value(b))
    return Schedule(sol.ObjectiveValue() / SCALE, st_, var, "OPTIMAL" if r == cp_model.OPTIMAL else "FEASIBLE",
                    sol.BestObjectiveBound() / SCALE, wall, _work_by_source(dag, var))
