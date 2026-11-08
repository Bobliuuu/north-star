"""Project PLD tok/s on the megakernel (S3 measured block cost curve) vs fast-executor PLD / fast plain. CPU only, INFERRED.
Writes results/mk_block_projection.json. Run: python3 scripts/project_mk_pld.py"""
import json, sys, pathlib, collections, statistics
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
import pld, mk_projection as mp

R = ROOT / "results"
mk = {k: mp.fill(v, 8) for k, v in mp.mk_curves().items()}  # MEASURED S3 (T<=8 only; the cap is 8)


def mk_curve(ctx_tokens):
    return mk[2048] if ctx_tokens < 4096 else mk[8192]


def load_tab(path, cell_key=None):
    out = {}
    for l in open(path):
        d = json.loads(l)
        if d.get("phase") == "cost_table":
            out[d["cell"]] = mp.fill({int(k): v for k, v in d["cost_ms"].items()}, 13)
    return out


def exact_tab():
    c = collections.defaultdict(dict)
    for l in open(R / "exactspec_ex1.jsonl"):
        d = json.loads(l)
        if d.get("phase") == "cost" and d["exec"] == "exact":
            c[d["ctx"]][d["T"]] = d["ms"]
    return {"natural": mp.fill(c[2048], 13), "ctx2048": mp.fill(c[2048], 13), "ctx8192": mp.fill(c[8192], 13)}


FAST = {"fast_fp2": load_tab(R / "fastpld_fp2.jsonl"), "fast_fp3": load_tab(R / "fastpld_fp3.jsonl"), "exact_ex1": exact_tab()}


def cell_of_len(n):
    return "natural" if n < 1000 else ("ctx2048" if n < 4096 else "ctx8192")


def replay(trace, g):
    gen, pids = trace["gen"], trace["prompt_ids"]
    def verify(tcur, drafts):
        k = len(tcur) - len(pids)
        return [gen[k + i] if k + i < len(gen) else -1 for i in range(len(drafts) + 1)]
    return pld.spec_loop(pld.make("sam2"), pids, len(gen), g, verify)[1]


def summarize(rows):
    """rows: per-task dicts of tok/s. geomean ratios."""
    g = mp.geomean
    f = lambda num, den: round(g(r[num] / r[den] for r in rows), 3)
    caps = {str(c): round(g(r[f"mk_T{c}"] / r["fast_pld"] for r in rows), 3) for c in mp.CAPS}
    best = [max(mp.CAPS, key=lambda c: r[f"mk_T{c}"]) for r in rows]
    return {"n": len(rows),
            "fast_pld_over_fast_plain": f("fast_pld", "fast_plain"),
            "mk_plain_over_fast_plain": f("mk_plain", "fast_plain"),
            "mk_pld_best_T_over_fast_pld": round(g(r["mk_best"] / r["fast_pld"] for r in rows), 3),
            "mk_pld_best_T_over_fast_plain": f("mk_best", "fast_plain"),
            "mk_pld_T4_over_fast_pld": f("mk_T4", "fast_pld"),
            "mk_pld_T4_over_fast_plain": f("mk_T4", "fast_plain"),
            "mk_pld_T8_over_fast_pld": f("mk_T8", "fast_pld"),
            "mk_pld_T4_over_mk_plain": f("mk_T4", "mk_plain"),
            "mk_pld_best_T_over_mk_plain": f("mk_best", "mk_plain"),
            "plain_advantage_mk_over_fast_needed_for_T4_to_tie_fast_pld": round(f("fast_pld", "fast_plain") / f("mk_T4", "mk_plain"), 3),
            "plain_advantage_mk_over_fast_needed_for_best_T_to_tie_fast_pld": round(f("fast_pld", "fast_plain") / f("mk_best", "mk_plain"), 3),
            "mk_T_cap_over_fast_pld": caps,
            "tasks_mk_T4_beats_fast_pld": sum(r["mk_T4"] > r["fast_pld"] for r in rows),
            "tasks_mk_best_beats_fast_pld": sum(r["mk_best"] > r["fast_pld"] for r in rows),
            "best_T_histogram": dict(sorted(collections.Counter(best).items()))}


def row(rounds_by_cap, mkc, fc, g12):
    r = {"fast_plain": 1000 / fc[1], "mk_plain": 1000 / mkc[1], "fast_pld": mp.tokps(g12, fc)}
    for c in mp.CAPS:
        r[f"mk_T{c}"] = mp.tokps(rounds_by_cap[c], mkc)
    r["mk_best"] = max(r[f"mk_T{c}"] for c in mp.CAPS)
    return r


out = {"label": "INFERRED projection; mk curve MEASURED (results/mk_block_s3_vast.json), acceptance MEASURED, combination is a model", "mk_curve_ms": {str(k): v for k, v in mk.items()}, "scenarios": {}}
traces = [json.loads(l) for l in open(R / "pld_pld3.jsonl") if '"phase": "trace"' in l]
tr_rounds = []
# ctx2048/ctx8192 traces have no prompt_ids (padded prompt not saved) -> aggregate path from fp3 live sam2 g12 rows (drafted not saved: bracket dbar)
ctx_live = collections.defaultdict(list)
for l in open(R / "fastpld_fp3.jsonl"):
    d = json.loads(l)
    if d.get("phase") == "live" and d["proposer"] == "sam2" and d["gamma"] == 12 and d["cell"] != "natural":
        ctx_live[(d["cell"], d["task"], d["kind"])].append(d)
for t in traces:
    if t.get("prompt_ids") is None:
        continue
    rc = {c: replay(t, c - 1) for c in mp.CAPS}
    tr_rounds.append((t, rc, replay(t, 12)))

mix = collections.defaultdict(lambda: collections.defaultdict(list))
plain = collections.defaultdict(list)
for l in open(R / "realtraffic_rt1.jsonl"):
    d = json.loads(l)
    if d["phase"] == "plain": plain[d["task"]].append(d["tok_s"])
    if d["phase"] == "live" and d["proposer"] == "sam2" and d["mode"] == "fixed": mix[d["task"]]["rows"].append(d)
meas = [statistics.mean(x["spec_tok_s"] for x in v["rows"]) / statistics.mean(plain[t]) for t, v in mix.items()]
out["h15_measured_sam2_g12_geomean"] = round(mp.geomean(meas), 3)

for sc, tabs in FAST.items():
    S = {"fast_curve_ms_T1_T4_T8_ctx2K": {k: round(tabs["ctx2048"][k], 2) for k in (1, 4, 8)}}
    # (1) H13 task sets: exact replay of sam2 on the saved traces
    cells = collections.defaultdict(list); kinds = collections.defaultdict(list)
    agg_err = []
    for t, rc, g12 in tr_rounds:
        cell = t["cell"]; mkc = mk_curve(cell == "ctx8192" and 8192 or 2048); fc = tabs[cell]
        r = row(rc, mkc, fc, g12); cells[cell].append(r); kinds[(cell, t["kind"])].append(r)
        # validate the aggregate model on the same task
        nd = [x for x in g12 if x[0] > 0]
        if nd:
            ra = {c: mp.agg_rounds(len(g12), len(nd), sum(x[0] for x in nd), sum(x[1] for x in nd), c - 1) for c in mp.CAPS}
            ra_g12 = mp.agg_rounds(len(g12), len(nd), sum(x[0] for x in nd), sum(x[1] for x in nd), 12)
            ma = row(ra, mkc, fc, ra_g12)
            agg_err.append((ma["mk_T4"] / ma["fast_pld"]) / (r["mk_T4"] / r["fast_pld"]) - 1)
            agg_err.append((ma["mk_best"] / ma["fast_pld"]) / (r["mk_best"] / r["fast_pld"]) - 1)
    S["h13_cells"] = {k: summarize(v) for k, v in cells.items()}
    # ctx cells: aggregate model, dbar bracketed (dbar=abar means every drafted token accepted; dbar=12 means always a full-length draft)
    for bracket in ("dbar_eq_abar", "dbar_12"):
        cc = collections.defaultdict(list); ck = collections.defaultdict(list)
        for (cell, task, kind), xs in ctx_live.items():
            per = []
            for x in xs:
                nv = x["verify_rounds"]; acc = x["mean_accepted_per_verify"] * nv
                dr = acc if bracket == "dbar_eq_abar" else 12 * nv
                dr = max(dr, acc)
                ra = {c: mp.agg_rounds(x["rounds"], nv, dr, acc, c - 1) for c in mp.CAPS}
                per.append(row(ra, mk_curve(8192 if cell == "ctx8192" else 2048), tabs[cell], mp.agg_rounds(x["rounds"], nv, dr, acc, 12)))
            r = {k: statistics.mean(p[k] for p in per) for k in per[0]}
            cc[cell].append(r); ck[(cell, kind)].append(r)
        S["h13_" + bracket] = {k: summarize(v) for k, v in cc.items()}
        S["h13_" + bracket + "_by_kind"] = {f"{k[0]}/{k[1]}": summarize(v) for k, v in ck.items()}
    S["h13_natural_by_kind"] = {k[1]: summarize(v) for k, v in kinds.items() if k[0] == "natural"}
    S["aggregate_model_validation_on_h13_tasks"] = {"n": len(agg_err), "median_abs_rel_err": round(statistics.median(abs(e) for e in agg_err), 4), "max_abs_rel_err": round(max(abs(e) for e in agg_err), 4), "mean_signed_rel_err": round(statistics.mean(agg_err), 4)}
    # (2) H15 99-prompt mix from per-task aggregates
    rows, bycat = [], collections.defaultdict(list); mod = []
    for task, v in mix.items():
        per = []
        for x in v["rows"]:
            plen = x["prompt_len"] + 96
            fc = tabs[cell_of_len(plen)] if sc != "exact_ex1" else tabs[cell_of_len(plen)]
            mkc = mk_curve(plen)
            ra = {c: mp.agg_rounds(x["rounds"], x["draft_rounds"], x["drafted"], x["accepted"], c - 1) for c in mp.CAPS}
            g12 = mp.agg_rounds(x["rounds"], x["draft_rounds"], x["drafted"], x["accepted"], 12)
            per.append(row(ra, mkc, fc, g12))
        r = {k: statistics.mean(p[k] for p in per) for k in per[0]}
        rows.append(r); bycat[v["rows"][0]["kind"]].append(r)
        mod.append(r["fast_pld"] / r["fast_plain"])
    S["h15_mix_99"] = summarize(rows)
    S["h15_mix_99"]["model_fast_pld_over_fast_plain_vs_measured"] = {"model": round(mp.geomean(mod), 3), "measured_h15_exact_executor": out["h15_measured_sam2_g12_geomean"]}
    S["h15_by_category"] = {k: summarize(v) for k, v in sorted(bycat.items())}
    S["breakeven_iid_ctx2K"] = mp.breakeven(mk[2048], tabs["ctx2048"])
    out["scenarios"][sc] = S
json.dump(out, open(R / "mk_block_projection.json", "w"), indent=1)
for sc, S in out["scenarios"].items():
    print(sc, S["fast_curve_ms_T1_T4_T8_ctx2K"], S["aggregate_model_validation_on_h13_tasks"])
    for key in ("h13_cells", "h13_dbar_eq_abar", "h13_dbar_12"):
      for k in S[key]:
        print("  ", key, k, {a: b for a, b in S[key][k].items() if a.startswith(("fast_pld_over", "mk_plain", "mk_pld_best_T_over_fast", "mk_pld_T4_over_fast", "tasks"))})
    print("   mix", {a: b for a, b in S["h15_mix_99"].items() if not isinstance(b, dict)}, S["h15_mix_99"]["model_fast_pld_over_fast_plain_vs_measured"])
    print("   BE", S["breakeven_iid_ctx2K"]["a_intervals_mk_best_T_le8_ge_fast_pld_best_T_le13"], S["breakeven_iid_ctx2K"]["a_intervals_mk_cap4_ge_fast_pld_best_T_le13"], S["breakeven_iid_ctx2K"]["a_vs_fast_plain"], S["breakeven_iid_ctx2K"]["a_vs_mk_plain"], S["breakeven_iid_ctx2K"]["mk_over_fastpld_ratio_at_a"])
