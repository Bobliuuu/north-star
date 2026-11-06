"""Build results/adapt_sched_model_SIMULATED.json. CPU only, deterministic.
Usage: python src/adapt_sched/run.py   (MODELED, not a hardware result)."""
from __future__ import annotations
import json, pathlib, sys
from dataclasses import replace, asdict
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from adapt_sched.model import (Cfg, simulate, MK_BS_MS, HF_BS_MS, BETA_S_HF, EVAL_WALL_S,
                               memory_headroom_gb, breakeven_horizon_s)

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = Cfg()
POLICIES = [
    ("sequential_1perwindow", dict(policy="sequential", seq_per_window=1)),
    ("sequential_allatonce", dict(policy="sequential", seq_per_window=BASE.n_cand)),
    ("static_u0.25", dict(policy="static", u_static=0.25)),
    ("static_u0.5", dict(policy="static", u_static=0.5)),
    ("static_u1.0", dict(policy="static", u_static=1.0)),
    ("adaptive_timing_idle_only", dict(policy="adaptive", timing_idle_only=True)),
    ("adaptive_timing_shared", dict(policy="adaptive", timing_idle_only=False)),
]


def run_one(args):
    tag, cfg = args
    return tag, simulate(cfg)


def table(cfgmod, pool):
    jobs = [("none", replace(BASE, **{**cfgmod, "policy": "none"}))]
    jobs += [(n, replace(BASE, **{**cfgmod, **kw})) for n, kw in POLICIES]
    return {tag: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
            for tag, r in pool.map(run_one, jobs)}


def main():
    out = {
        "LABEL": "MODELED / SIMULATED. Not a hardware result. No GPU was run.",
        "inputs": {
            "MEASURED": {"megakernel_step_ms_by_batch": MK_BS_MS, "hf_eager_step_ms_by_batch_ctx8192_A100": HF_BS_MS,
                         "eval_wall_s_per_candidate": EVAL_WALL_S, "beta_s_megakernel": 0.60,
                         "selfimprove_kernel_candidates_passing_speed_gate": "0 of 8 (speedup 0.96-0.98 on dev)"},
            "INFERRED": {"beta_s_graphed_pytorch": BETA_S_HF},
            "INVENTED_swept": ["beta_b", "c_sm", "rho", "tokens_per_req", "f_timing", "noise_mult",
                               "slo_slack", "n_cand", "policy knobs", "payback gain"],
        },
        "base_cfg": asdict(BASE), "nominal": None, "one_at_a_time": {}, "beta_b_x_rho": {},
    }
    with ProcessPoolExecutor() as pool:
        out["nominal"] = table({}, pool)
        sweeps = {
            "beta_b": [0.1, 0.3, 0.5, 0.8],
            "rho": [0.3, 0.5, 0.7, 0.85],
            "slo_slack": [1.1, 1.25, 1.5],
            "f_timing": [0.2, 0.5, 0.8],
            "noise_mult": [1.0, 2.0, 4.0],
            "c_sm": [0.0, 0.1, 0.3],
        }
        for k, vs in sweeps.items():
            out["one_at_a_time"][k] = {str(v): table({k: v}, pool) for v in vs}
        out["one_at_a_time"]["executor"] = {
            "megakernel_bmax8": out["nominal"],
            "hf_eager_bmax4": table(dict(table=tuple(HF_BS_MS.items()), beta_s=BETA_S_HF, bmax=4), pool),
        }
        for bb in [0.1, 0.3, 0.5, 0.8]:
            for rho in [0.3, 0.5, 0.7, 0.85]:
                out["beta_b_x_rho"][f"beta_b={bb},rho={rho}"] = table(dict(beta_b=bb, rho=rho), pool)
    # memory accounting
    out["memory_accounting"] = {
        f"batch={b},ctx={c}": memory_headroom_gb(b, c)
        for b, c in [(1, 8192), (4, 8192), (4, 32768), (8, 32768), (4, 131072), (8, 131072)]}
    out["memory_accounting"]["NOTE"] = ("megakernel claims 0.85*79.3 GB; weights 61.0 GB; only ~12 GB is outside its "
                                        "claim for a co-resident benchmark (compile + candidate buffers: INVENTED size).")
    # payback (Phase 7 gate: total saved inference time must repay cost over a stated horizon)
    cost = BASE.n_cand * EVAL_WALL_S
    pb = {}
    for g in [0.01, 0.03, 0.10]:
        for busy in [0.3, 0.6, 0.9]:
            pb[f"gain={g},serving_busy={busy}"] = round(breakeven_horizon_s(cost, busy, g) / 3600, 2)
    out["payback_hours_for_6_candidates_INVENTED_gain"] = pb
    out["payback_note"] = ("Measured yield so far is 0 of 8 candidates passing the speed gate, so the measured "
                           "gain is 0 and payback is infinite; the table assumes one accepted candidate with the "
                           "stated INVENTED fractional decode-time cut. Cost counts background GPU-seconds only, "
                           "not serving slowdown.")
    p = ROOT / "results" / "adapt_sched_model_SIMULATED.json"
    p.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    print("wrote", p)
    for n, r in out["nominal"].items():
        print(f"{n:24s} p99={r['p99_step_ms']:9.2f}ms viol={r['slo_violation_frac']:.3f} "
              f"max={r['max_step_ms']:.0f}ms over={r['tokens_over_slo']}/{r['tokens_total']} req_p99={r['req_p99_s']:8.2f}s done={r['cands_done']}/{r['cands_target']} "
              f"finish={r['bg_finish_s']} gpu_s={r['bg_gpu_s_consumed']:.0f}")


if __name__ == "__main__":
    main()
