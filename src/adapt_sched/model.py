"""Phase 7 study: serve full North on ONE GPU while a background job
(kernel-candidate benchmarking, as in src/selfimprove) runs. MODELED, NOT a hardware result.

Provenance tags: MEASURED (results/ and RESULTS.md), INFERRED (arithmetic on measured),
INVENTED (swept, no measurement).

Fluid contention model (same style as src/async_spec): a serving step of standalone time t keeps
HBM busy for BETA_S*t; the background at intensity u demands BETA_B*u more. Total demand > 1 slows
both by s = max(1, BETA_S + BETA_B*u), times an SM-contention factor (1 + C_SM*u) (INVENTED).
Optimistic about interleaving: no L2 effects, no kernel-granularity head-of-line blocking,
no memory-capacity limit (accounted separately in memory_headroom_gb). Decode only; prefill is
not modeled.
"""
from __future__ import annotations
import random, math
from dataclasses import dataclass

# MEASURED: megakernel H100 step ms by batch (results/megakernel_H100_bs_*.log, real weights, ctx 2K)
MK_BS_MS = {1: 3.32, 2: 4.10, 4: 5.29, 8: 7.39}
# MEASURED: HF eager decode ms median by batch, ctx 8192, A100 (results/ctx_batch_run1.jsonl)
HF_BS_MS = {1: 85.8, 2: 96.6, 4: 116.3}
# MEASURED: selfimprove eval_wall_s per candidate over the 8 structured-run proposals
# (results/selfimprove_kernel_structured/proposals.jsonl): 119.9..154.0
EVAL_WALL_S = 130.5
BETA_S_MK = 0.60   # MEASURED (RESULTS.md H1: 1.97 ms HBM floor / 3.25 ms)
BETA_S_HF = 0.45   # INFERRED (async_spec)
# memory (GB)
WEIGHTS_GB = 61.0          # MEASURED/derived, docs/NORTH_ARCH.md
GPU_GB = 79.3              # MEASURED, results/baseline.meta.json
MK_VRAM_FRAC = 0.85        # MEASURED setting: --frac-vram-utilization used for megakernel runs
KV_GB_PER_128K_SEQ = 3.5   # derived, docs/NORTH_ARCH.md global-layer KV


def step_ms(b: int, table=MK_BS_MS) -> float:
    """Piecewise-linear measured step time; linear extrapolation beyond last point (INVENTED)."""
    if b <= 1:
        return table[1]
    xs = sorted(table)
    if b >= xs[-1]:
        sl = (table[xs[-1]] - table[xs[-2]]) / (xs[-1] - xs[-2])
        return table[xs[-1]] + sl * (b - xs[-1])
    for a, c in zip(xs, xs[1:]):
        if a <= b <= c:
            return table[a] + (b - a) / (c - a) * (table[c] - table[a])


@dataclass(frozen=True)
class Cfg:
    table: tuple = tuple(MK_BS_MS.items())
    beta_s: float = BETA_S_MK
    beta_b: float = 0.5          # INVENTED sweep: HBM-busy fraction of the benchmark kernels
    c_sm: float = 0.10           # INVENTED sweep: extra SM-contention slowdown per unit u
    rho: float = 0.5             # INVENTED: offered load as fraction of batch-Bmax capacity
    tokens_per_req: int = 200    # INVENTED
    bmax: int = 8
    n_cand: int = 6              # equal GPU-time budget: n_cand * eval_s standalone seconds
    eval_s: float = EVAL_WALL_S  # MEASURED standalone seconds per candidate
    f_timing: float = 0.5        # INVENTED: share of eval that is timing-sensitive
    noise_mult: float = 2.0      # INVENTED: extra reps needed for timing phase under contention
    slo_slack: float = 1.25      # INVENTED: SLO = slack * unloaded step time at Bmax
    horizon_s: float = 3000.0
    seed: int = 0
    policy: str = "none"         # none|sequential|static|adaptive
    u_static: float = 0.5
    seq_per_window: int = 1      # candidates per pause window (sequential)
    seq_gap_s: float = 60.0      # serving window between pauses (sequential)
    seq_start_s: float = 60.0
    ctl_dt: float = 1.0          # adaptive control interval
    ctl_target: float = 0.95     # aim p99 step <= target*SLO
    timing_idle_only: bool = True  # adaptive: timing phase runs only when serving is idle (controlled window)


def slo_ms(c: Cfg) -> float:
    return c.slo_slack * step_ms(c.bmax, dict(c.table))


def wpct(samples, q):
    """weighted percentile of (value, weight) pairs."""
    if not samples:
        return float("nan")
    s = sorted(samples)
    tot = sum(w for _, w in s)
    acc = 0.0
    for v, w in s:
        acc += w
        if acc >= q * tot:
            return v
    return s[-1][0]


def simulate(c: Cfg) -> dict:
    table = dict(c.table)
    rng = random.Random(c.seed)
    cap_tok_s = c.bmax / (step_ms(c.bmax, table) / 1e3)
    lam = c.rho * cap_tok_s / c.tokens_per_req
    arr, t = [], 0.0
    while True:
        t += rng.expovariate(lam)
        if t > c.horizon_s:
            break
        arr.append(t)
    SLO = slo_ms(c)
    cc_s = c.eval_s * (1 - c.f_timing)
    tm_s = c.eval_s * c.f_timing
    st = {"cc": cc_s, "tm": tm_s, "done": 0, "spent": 0.0}
    have_bg = c.policy != "none"
    total = c.n_cand if have_bg else 0

    def bg_advance(standalone_s, clean):
        """Spend standalone_s of background GPU work. Timing-phase progress made while
        contended is worth 1/noise_mult (extra reps), but is still charged in full."""
        left = standalone_s
        while left > 1e-12 and st["done"] < total:
            if st["cc"] > 1e-12:
                d = min(left, st["cc"]); st["cc"] -= d; left -= d; st["spent"] += d
            else:
                eff = 1.0 if clean else 1.0 / c.noise_mult
                d = min(left * eff, st["tm"]); st["tm"] -= d
                spent = d / eff; left -= spent; st["spent"] += spent
            if st["cc"] <= 1e-12 and st["tm"] <= 1e-12:
                st["done"] += 1; st["cc"], st["tm"] = cc_s, tm_s

    def bg_active():
        return have_bg and st["done"] < total

    t = 0.0
    qi = 0
    queue, active = [], []
    steps, req_lat = [], []
    bg_finish = None
    u = {"static": c.u_static, "adaptive": 0.5}.get(c.policy, 0.0)
    win_walls = []
    next_ctl = c.ctl_dt
    next_window = c.seq_start_s if c.policy == "sequential" else math.inf
    pause_total = 0.0
    u_sum = u_t = 0.0

    while t < c.horizon_s:
        if c.policy == "sequential" and bg_active() and t >= next_window:
            n = min(c.seq_per_window, total - st["done"])
            dur = n * c.eval_s
            bg_advance(dur, True)
            if active:
                steps.append((dur * 1e3, len(active)))  # in-flight tokens see the whole gap
            t += dur; pause_total += dur
            next_window = t + c.seq_gap_s
            if not bg_active():
                bg_finish = t
            continue
        while qi < len(arr) and arr[qi] <= t:
            queue.append(arr[qi]); qi += 1
        while queue and len(active) < c.bmax:
            active.append([c.tokens_per_req, queue.pop(0)])
        if not active:
            nxt = arr[qi] if qi < len(arr) else c.horizon_s
            if c.policy == "sequential" and bg_active():
                nxt = min(nxt, next_window)
            nxt = min(max(nxt, t + 1e-6), c.horizon_s)
            if bg_active() and c.policy != "sequential":
                bg_advance(nxt - t, True)       # idle serving: background alone, clean timing
                if not bg_active() and bg_finish is None:
                    bg_finish = nxt
            t = nxt
            continue
        b = len(active)
        t0 = step_ms(b, table) / 1e3
        uu = u if (bg_active() and c.policy in ("static", "adaptive")) else 0.0
        if uu > 0 and c.policy == "adaptive" and c.timing_idle_only and st["cc"] <= 1e-12:
            uu = 0.0   # timing-sensitive phase yields to serving: only runs in clean idle gaps
        s = max(1.0, c.beta_s + c.beta_b * uu) * (1 + c.c_sm * uu) if uu > 0 else 1.0
        wall = t0 * s
        if uu > 0:
            bg_advance(uu * wall / s, False)
            if not bg_active() and bg_finish is None:
                bg_finish = t + wall
        u_sum += uu * wall; u_t += wall
        t += wall
        steps.append((wall * 1e3, b))
        win_walls.append(wall * 1e3)
        nxt_active = []
        for r in active:
            r[0] -= 1
            if r[0] <= 0:
                req_lat.append(t - r[1])
            else:
                nxt_active.append(r)
        active = nxt_active
        if c.policy == "adaptive":
            if t >= next_ctl:
                if win_walls:
                    ws = sorted(win_walls)
                    p99 = ws[min(len(ws) - 1, int(0.99 * len(ws)))]
                    if p99 > c.ctl_target * SLO:
                        u = u * 0.5
                    elif p99 < 0.8 * SLO:
                        u = min(1.0, u + 0.05)
                    win_walls = []
                next_ctl = t + c.ctl_dt
            if u < 0.02:
                u = 0.02   # probe floor so the controller can recover (INVENTED)
    p99 = wpct(steps, 0.99)
    totw = max(1, sum(w for _, w in steps))
    viol = sum(w for v, w in steps if v > SLO) / totw
    rl = sorted(req_lat)
    return {
        "policy": c.policy, "p50_step_ms": wpct(steps, 0.5), "p99_step_ms": p99,
        "slo_ms": SLO, "slo_violation_frac": viol,
        "max_step_ms": max((v for v, _ in steps), default=float("nan")),
        "tokens_over_slo": sum(w for v, w in steps if v > SLO), "tokens_total": totw,
        "req_p99_s": rl[int(0.99 * (len(rl) - 1))] if rl else float("nan"),
        "req_p50_s": rl[len(rl) // 2] if rl else float("nan"),
        "reqs_done": len(rl), "reqs_arrived": len(arr),
        "cands_done": st["done"], "cands_target": total,
        "bg_finish_s": bg_finish, "bg_gpu_s_consumed": st["spent"],
        "pause_s": pause_total, "mean_u_while_serving": (u_sum / u_t if u_t else 0.0),
    }


def memory_headroom_gb(batch: int, ctx: int, vram_frac: float = MK_VRAM_FRAC,
                       kv_gb_per_128k=KV_GB_PER_128K_SEQ) -> dict:
    """MEASURED/derived weights+KV constants -> INFERRED headroom for a co-resident benchmark.
    The megakernel claims vram_frac*GPU for itself; the rest is what a benchmark can use."""
    kv = batch * ctx / 131072 * kv_gb_per_128k
    claimed = vram_frac * GPU_GB
    return {"weights_gb": WEIGHTS_GB, "kv_gb": kv, "serving_claimed_gb": claimed,
            "kv_budget_inside_claim_gb": claimed - WEIGHTS_GB,
            "kv_fits_in_claim": kv <= claimed - WEIGHTS_GB,
            "benchmark_headroom_gb": GPU_GB - claimed}


def breakeven_horizon_s(cost_gpu_s: float, serving_busy_frac: float, gain_frac: float) -> float:
    """Wall seconds of serving after which saved serving GPU time repays cost_gpu_s.
    Saved per wall second = gain_frac * serving_busy_frac (gain_frac = fractional decode-time cut)."""
    r = gain_frac * serving_busy_frac
    return math.inf if r <= 0 else cost_gpu_s / r
