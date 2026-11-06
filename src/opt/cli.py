"""python3 -I -m opt.cli [--calib file.jsonl | --synthetic FAMILY] ...  (run from north-star/src)

Without --calib the run uses SYNTHETIC acceptance (loudly labelled in the output) because no measured
accept-length table exists yet. Cost model is fit from results/*.jsonl microbench rows.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

from . import config as C
from . import costmodel as CM
from . import solve as SV
from . import synth
from .calibration import AcceptTable
from .perturb import perturbation_study
from .policy import PolicyCache

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASELINE = C.Config(4, 0, 4, (), "two_pass")


def load_table(paths=None):
    paths = paths or sorted(glob.glob(os.path.join(ROOT, "results", "*.jsonl")))
    return CM.MicrobenchTable.from_jsonl(*paths)


def regime_scan(table, ctxs, families, routings, tl=3.0):
    rows = []
    for fam in families:
        cal = AcceptTable(synth.synthetic_entries(family=fam, i_understand_this_is_synthetic=True))
        for ro in routings:
            for ctx in ctxs:
                cm = CM.NorthCostModel(table, routing=ro)
                ev = SV.Evaluator(cm, C.Bucket("code", ctx, 1, ro), cal, time_limit_s=tl)
                res = SV.run_comparators(ev, SV.Space(), BASELINE)
                ref = SV.nospec_reference(ev)
                row = {"family": fam, "routing": ro, "ctx": ctx, "no_spec_tok_s": ref.tok_s,
                       **{k: v.summary() for k, v in res.items()}}
                j, i = res["joint"].tok_s, res["independent"].tok_s
                row["joint_over_independent"] = j / i - 1
                rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib")
    ap.add_argument("--out", default=os.path.join(ROOT, "src", "opt", "results", "regime_scan_SYNTHETIC.json"))
    ap.add_argument("--ctx", default="2048,8192,32768,131072")
    ap.add_argument("--draws", type=int, default=12)
    a = ap.parse_args(argv)
    table = load_table()
    ctxs = [int(x) for x in a.ctx.split(",")]
    t0 = time.perf_counter()
    out = {"banner": synth.BANNER if not a.calib else "measured calibration", "cost_model": CM.NorthCostModel(table).provenance_note()}
    if a.calib:
        cal = AcceptTable.from_jsonl(a.calib)
        ev = SV.Evaluator(CM.NorthCostModel(table), C.Bucket("code", ctxs[0]), cal)
        out["comparators"] = {k: v.summary() for k, v in SV.run_comparators(ev, SV.Space(), BASELINE).items()}
    else:
        out["regime_scan"] = regime_scan(table, ctxs, list(synth.FAMILIES), ["uniform", "corr75", "identical"])
        gains = [r["joint_over_independent"] for r in out["regime_scan"]]
        out["joint_beats_independent_gt1pct"] = f"{sum(g > 0.01 for g in gains)}/{len(gains)}"
        cal = AcceptTable(synth.synthetic_entries(family="attn-tolerant", i_understand_this_is_synthetic=True))
        out["perturbation"] = perturbation_study(CM.NorthCostModel(table), C.Bucket("code", 32768), cal, SV.Space(),
                                                 draws=a.draws, time_limit_s=3.0)
        for s in (0.5, 2.0):
            p = perturbation_study(CM.NorthCostModel(table), C.Bucket("code", 32768), cal, SV.Space(),
                                   draws=a.draws, sigma_scale=s, time_limit_s=3.0)
            out[f"perturbation_sigma_x{s}"] = {k: v for k, v in p.items() if k != "rows"}
    out["wall_s"] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
