"""Sensitivity of the joint policy to cost-model error.

For each draw: sample per-kind multiplicative errors -> a 'true' cost model. Compare
  nominal policy c* (chosen under the nominal model), re-scheduled optimally under truth, vs
  truth-optimal policy (joint solve under the true model; optionally restricted to a shortlist for speed).
Regret = 1 - tok_s(c* | truth) / tok_s(best | truth). Also reports nominal prediction error and whether the argmax moved.
"""
from __future__ import annotations

import random
import statistics as st

from . import costmodel as CM
from . import solve as SV


def perturbation_study(cm, bucket, calib, space, draws=10, sigma_scale=1.0, seed=0, shortlist=25, **ev_kw):
    rng = random.Random(seed)
    ev0 = SV.Evaluator(cm, bucket, calib, **ev_kw)
    cands = SV.enumerate_configs(ev0, space)
    best0, _, _ = SV.dinkelbach(ev0, cands)
    nom = ev0.opt_result("nominal", best0)
    # shortlist: top candidates by nominal serial-or-LB optimism so truth-optimal search stays tractable
    ranked = sorted(cands, key=lambda c: -ev0.N(c) / ev0.T_lb(c))
    short = list(dict.fromkeys([best0] + ranked[:shortlist]))
    sig = CM.default_sigmas(cm.table, sigma_scale)
    rows = []
    for d in range(draws):
        mult = CM.draw_multipliers(rng, sig)
        tm = CM.PerturbedCostModel(cm, mult)
        ev = SV.Evaluator(tm, bucket, calib, **ev_kw)
        ev_best, _, _ = SV.dinkelbach(ev, short)
        truth_best = ev.opt_result("truth", ev_best)
        mine = ev.opt_result("nominal_under_truth", best0)
        rows.append({"mult": {k: round(v, 3) for k, v in mult.items()},
                     "regret": 1.0 - mine.tok_s / truth_best.tok_s,
                     "pred_err": nom.tok_s / mine.tok_s - 1.0,
                     "argmax_moved": ev_best != best0, "truth_cfg": ev_best.label()})
    reg = sorted(r["regret"] for r in rows)
    return {"nominal_cfg": best0.label(), "nominal_tok_s": nom.tok_s, "sigma_scale": sigma_scale, "draws": draws,
            "sigmas": sig, "regret_mean": st.mean(reg), "regret_p90": reg[min(len(reg) - 1, int(0.9 * len(reg)))],
            "regret_max": reg[-1], "argmax_moved_frac": sum(r["argmax_moved"] for r in rows) / len(rows),
            "pred_err_abs_mean": st.mean(abs(r["pred_err"]) for r in rows),
            "shortlist": len(short), "rows": rows}
