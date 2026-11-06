"""Build results/async_spec_model_SIMULATED.json from measured inputs.
Usage: python src/async_spec/run.py  (CPU only, deterministic)."""
from __future__ import annotations
import json, pathlib, statistics as st, sys, collections
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from async_spec.model import speedup, upper_bound, MK_BS_MS

ROOT = pathlib.Path(__file__).resolve().parents[2]
KS = [0, 1, 2, 3, 4, 6, 8, 12, 16]
BETAS = [0.45, 0.6, 0.8, 1.0]
QREJ = [0.0, 0.05, 0.2, 0.4]
FKS = (0.5, 1.0, 2.0)


def load(f):
    return [json.loads(l) for l in open(ROOT / "results" / f)]


def inputs():
    a100 = collections.defaultdict(list)
    for r in load("spec.jsonl"):
        if r["phase"] == "1_selfspec":
            a100[r["config"]].append(r)
    apos = {c: [st.mean(r["accept_rate_per_pos"][i] for r in rs)
                for i in range(len(rs[0]["accept_rate_per_pos"]))]
            for c, rs in a100.items()}
    h = collections.defaultdict(list)
    for r in load("h100_spec.jsonl"):
        if r["phase"] == "g_spec":
            h[r["config"]].append(r)
    cfgs = {}
    for c, rs in h.items():
        g = rs[0]["gamma"]
        cfgs[c] = dict(gamma=g,
                       d_ms=st.mean(r["ms_draft_per_round"] for r in rs) / g,
                       V_ms=st.mean(r["ms_verify_per_round"] for r in rs),
                       mean_acc_h100=st.mean(r["mean_accepted"] for r in rs),
                       apos=apos[c], mean_acc_a100=sum(apos[c]),
                       measured_serial_speedup_vs_plain=st.mean(r["speedup_wall"] for r in rs))
    return cfgs


def main():
    cfgs = inputs()
    out = dict(
        label="SIMULATED / MODELED. Not a hardware result. Arithmetic on measured costs; knobs tagged INVENTED are swept.",
        provenance=dict(
            measured=["H100 graphed-PyTorch draft/verify ms per round (results/h100_spec.jsonl)",
                      "per-position accept rates (results/spec.jsonl, A100; H100 mean accepted differs <0.2/round)",
                      "megakernel BS=1/2/4/8 step ms %s (results/megakernel_H100_bs_*.log)" % MK_BS_MS,
                      "megakernel BS1 bandwidth-busy fraction 0.60 (RESULTS.md H1: 1.97 ms floor / 3.25 ms)"],
            inferred=["graphed-PyTorch beta~0.45 (experts+lm_head bytes / K=1 step; RESULTS.md H2)",
                      "megakernel projection: V(K=3/5/9)=4.7/5.8/7.9 ms, d=2.4 ms (RESULTS.md H5)",
                      "full-accept bonus guess hit prob = apos[-1]/apos[-2]"],
            invented=["beta sweep 0.45..1.0", "reject-bonus guess prob q_rej sweep", "branch-count k sweep",
                      "batch-factor extrapolation beyond BS=8", "fk_scale sweep (branch-batch cost x0.5/1/2)",
                      "perfect interleave, no SM/L2 contention (optimistic)"]),
        configs=cfgs, executors={}, kill={})
    execs = {"h100_graphed_pytorch_MEASURED": {c: (v["gamma"], v["d_ms"], v["V_ms"]) for c, v in cfgs.items()},
             "megakernel_projection_INFERRED": {"r4_g2": (2, 2.4, 4.7), "r4_g4": (4, 2.4, 5.8), "r4_g8": (8, 2.4, 7.9)}}
    plain_ms = {"h100_graphed_pytorch_MEASURED": 10.66, "megakernel_projection_INFERRED": 3.32}
    for en, ec in execs.items():
        res = {}
        for c, (g, d, V) in ec.items():
            apos = cfgs[c]["apos"]
            grid = []
            for beta in BETAS:
                for q in QREJ:
                    for fs in FKS:
                        for k in KS:
                            sp, h, Ts, Ta = speedup(g, d, V, k, apos, beta, fs, q_rej=q)
                            tok = 1 + cfgs[c]["mean_acc_h100"]
                            vp = (tok / Ta) / (1 / plain_ms[en])
                            grid.append(dict(vs_plain=round(vp, 4), beta=beta, q_rej=q, fk_scale=fs, k=k, hit=round(h, 4),
                                             T_serial_ms=round(Ts, 3), T_async_ms=round(Ta, 3), speedup=round(sp, 4)))
            nominal = [x for x in grid if x["q_rej"] == 0.05 and x["fk_scale"] == 1.0 and x["k"] > 0]
            pos = [x for x in grid if x["speedup"] > 1.0]
            res[c] = dict(gamma=g, d_ms=round(d, 3), V_ms=round(V, 3),
                          upper_bound_h1_free_branches=round(upper_bound(g, d, V), 3),
                          serial_spec_vs_plain=round((1 + cfgs[c]["mean_acc_h100"]) / (g * d + V) * plain_ms[en], 4),
                          best_vs_plain_any=max(grid, key=lambda x: x["vs_plain"]),
                          best_vs_plain_nominal=max(nominal, key=lambda x: x["vs_plain"]),
                          best_any_point=max(grid, key=lambda x: x["speedup"]),
                          best_nominal=max(nominal, key=lambda x: x["speedup"]),
                          n_points=len(grid), n_points_gt1=len(pos),
                          betas_with_gain=sorted({x["beta"] for x in pos}), grid=grid)
        out["executors"][en] = res
    for en, res in out["executors"].items():
        kill = {}
        for c, r in res.items():
            for beta in BETAS:
                pts = [x for x in r["grid"] if x["beta"] == beta and x["q_rej"] == 0.05
                       and x["fk_scale"] == 1.0 and x["k"] > 0]
                b = max(pts, key=lambda x: x["speedup"])
                kill.setdefault(c, {})[str(beta)] = dict(best_k=b["k"], best_speedup=b["speedup"],
                                                         hit=b["hit"], killed=b["speedup"] <= 1.0,
                                                         best_vs_plain=max(p["vs_plain"] for p in pts))
        out["kill"][en] = kill
    be = {}
    for en, ec in execs.items():
        for c, (g, d, V) in ec.items():
            apos = cfgs[c]["apos"]
            tok = 1 + cfgs[c]["mean_acc_h100"]

            def best(beta, key):
                vals = []
                for k in range(1, 17):
                    sp, h, Ts, Ta = speedup(g, d, V, k, apos, beta, 1.0, q_rej=0.05)
                    vals.append(sp if key == "serial" else (tok / Ta) * plain_ms[en])
                return max(vals)
            row = {}
            for key in ("serial", "plain"):
                lo, hi = 0.0, 1.0
                if best(0.0, key) <= 1.0:
                    row[key] = None  # never beats baseline even with free bandwidth
                    continue
                for _ in range(40):
                    mid = (lo + hi) / 2
                    lo, hi = (mid, hi) if best(mid, key) > 1.0 else (lo, mid)
                row[key] = round(lo, 3)
            be.setdefault(en, {})[c] = dict(max_beta_to_beat_serial_spec=row["serial"],
                                           max_beta_to_beat_plain_decode=row["plain"])
    out["breakeven_beta_nominal_qrej0.05_fk1_best_k"] = be
    p = ROOT / "results" / "async_spec_model_SIMULATED.json"
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)
    print(json.dumps(be, indent=1))
    for en, kill in out["kill"].items():
        print(en)
        for c, bb in kill.items():
            r = out["executors"][en][c]
            print(" ", c, {b: (v["best_k"], v["best_speedup"], v["hit"]) for b, v in bb.items()},
                  "UB", r["upper_bound_h1_free_branches"], "bestany", r["best_any_point"]["speedup"], "serial_vs_plain", r["serial_spec_vs_plain"],
                  "bestVsPlainAny", r["best_vs_plain_any"]["vs_plain"], "nominal", r["best_vs_plain_nominal"]["vs_plain"],
                  "n>1", r["n_points_gt1"], "/", r["n_points"])


if __name__ == "__main__":
    main()
