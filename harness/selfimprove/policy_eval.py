"""Policy-table track: tunables in src/opt/policy_tables/spec_space.json, scored by the Phase-3 optimizer (src/opt).

    python policy_eval.py check|quality|bench      (NS_SEED, NS_SPLIT per the harness protocol)

The table is parsed as DATA (strict key/range whitelist; nothing from it is executed). Dev metric = Phase-3 PREDICTED tok/s
from the cost model + acceptance calibration; it is a model prediction, not a measurement, and the output says which
calibration it used (a SYNTHETIC calibration is allowed only for dry runs; config.make_config refuses it for a live run).
  check   : 8 fresh-seed buckets; self-consistency of the solver result for the table (feasible, finite, Dinkelbach
            lambda == N/T, prediction <= lower-bound ceiling, optimal >= serial schedule). Honest scope: this is solver self-consistency, NOT a
            measurement of the policy's real speed.
  quality : 24 HELD-OUT buckets (never in the prompt); item = predicted speedup over no-speculation. Non-inferiority gate.
  bench   : time_ms = mean over the split's buckets of (1e6 / predicted tok/s)  (ms per 1000 tokens). dev vs held-out buckets.
"""
import common, hashlib, json, math, os, random, sys

sys.path.insert(0, os.path.join(common.WS, "src"))
from opt import config as C, costmodel as CM, solve as SV, synth
from opt.calibration import AcceptTable

TABLE = os.path.join(common.WS, "src", "opt", "policy_tables", "spec_space.json")
CALIB = os.path.join(common.WS, "results", "accept_calibration.jsonl")
ALLOWED = {"gammas": list, "modes": list, "n_groups": int}

DEV = [("code", 2048, 1, "corr75"), ("code", 8192, 1, "corr75"), ("code", 8192, 1, "uniform"), ("code", 32768, 1, "corr75")]
HELDOUT_BENCH = [("code", c, 1, r) for c in (5000, 12000, 20000) for r in ("corr75", "identical")]
HELDOUT_QUALITY = [("code", c, b, r) for c in (3072, 5000, 12000, 20000) for r in ("corr75", "uniform", "identical") for b in (1, 2)]


class TableError(ValueError):
    pass


def load_table():
    raw = open(TABLE, "rb").read()
    t = json.loads(raw)
    if not isinstance(t, dict) or set(t) != set(ALLOWED):
        raise TableError(f"keys must be exactly {sorted(ALLOWED)}")
    for k, ty in ALLOWED.items():
        if type(t[k]) is not ty:
            raise TableError(f"{k}: wrong type")
    g = t["gammas"]
    if not (1 <= len(g) <= 12) or any(type(x) is not int or not 1 <= x <= 12 for x in g) or sorted(set(g)) != g:
        raise TableError("gammas: 1..12 distinct ints, ascending, at most 12")
    md = t["modes"]
    if not md or any(x not in C.MODES for x in md) or len(set(md)) != len(md):
        raise TableError(f"modes: non-empty subset of {C.MODES}")
    if not 2 <= t["n_groups"] <= 12:
        raise TableError("n_groups: 2..12")
    return t, hashlib.sha256(raw).hexdigest()


def load_calib():
    if os.path.exists(CALIB):
        return AcceptTable.from_jsonl(CALIB)
    return AcceptTable(synth.synthetic_entries(i_understand_this_is_synthetic=True))


_CM_TABLE = None
def solve(bucket, t, calib):
    global _CM_TABLE
    if _CM_TABLE is None:
        import glob
        _CM_TABLE = CM.MicrobenchTable.from_jsonl(*sorted(glob.glob(os.path.join(common.WS, "results", "*.jsonl"))))
    w, ctx, b, ro = bucket
    bk = C.Bucket(w, ctx, b, ro)
    ev = SV.Evaluator(CM.NorthCostModel(_CM_TABLE, routing=ro), bk, calib, n_groups=t["n_groups"], time_limit_s=2.0, workers=2)
    cands = SV.enumerate_configs(ev, SV.Space(gammas=tuple(t["gammas"]), modes=tuple(t["modes"])))
    best, lam, _ = SV.dinkelbach(ev, cands)
    if best is None:
        return dict(tok_s=0.0, ok=False)
    res = ev.opt_result("joint", best)
    ref = SV.nospec_reference(ev)
    ceil_tok_s = ev.N(best) / ev.T_lb(best) * 1e6
    ser_tok_s = ev.serial_result("s", best).tok_s
    return dict(tok_s=res.tok_s, nospec=ref.tok_s, ser=ser_tok_s, lam=lam * 1e6, ceil=ceil_tok_s, cfg=best.label(),
                ok=math.isfinite(res.tok_s) and res.tok_s > 0)


def cached(bucket, t, thash, calib):
    path = os.path.join(common.SCRATCH, "policy_cache.json")        # deterministic function of (table, bucket): safe to memoise
    try: cache = json.load(open(path))
    except Exception: cache = {}
    k = f"{thash}|{bucket}"
    if k not in cache:
        cache[k] = solve(bucket, t, calib)
        try: json.dump(cache, open(path, "w"))
        except OSError: pass
    return cache[k]


def main(mode):
    try:
        t, thash = load_table()
    except (TableError, ValueError, OSError) as e:
        sys.stderr.write(f"invalid policy table: {e}\n")
        if mode == "check": common.emit({"cases": []})
        elif mode == "quality": common.emit({"items": []})
        else: common.emit({"time_ms": None})
        return
    calib = load_calib()
    if mode == "check":
        rng = random.Random(common.SEED * 31 + 5)
        cases = []
        for i in range(8):
            bk = ("code", rng.randint(1500, 120000), rng.choice((1, 2)), rng.choice(("corr75", "uniform", "identical")))
            r = solve(bk, t, calib)       # fresh bucket: never cached
            if not r["ok"]:
                cases.append({"name": f"b{i}", "rel_l2": 1e9, "max_abs": 1e9, "finite": False}); continue
            gap = abs(r["lam"] - r["tok_s"]) / r["tok_s"]
            over = max(0.0, r["tok_s"] - r["ceil"]) / r["tok_s"]
            worse = max(0.0, r["ser"] - r["tok_s"]) / r["ser"] if r["ser"] > 0 else 1.0   # optimal schedule must not lose to the serial one
            e = max(gap, over, worse)
            cases.append({"name": f"b{i}", "rel_l2": e, "max_abs": e, "finite": True})
        common.emit({"cases": cases, "calibration": calib.fingerprint()})
    elif mode == "quality":
        items = []
        for bk in HELDOUT_QUALITY:
            r = cached(bk, t, thash, calib)
            items.append(r["tok_s"] / r["nospec"] if r["ok"] else 0.0)
        common.emit({"items": items, "calibration": calib.fingerprint()})
    else:
        bks = DEV if common.SPLIT == "dev" else HELDOUT_BENCH
        ts = [cached(bk, t, thash, calib) for bk in bks]
        if not all(r["ok"] for r in ts):
            common.emit({"time_ms": None}); return
        common.emit({"time_ms": sum(1e6 / r["tok_s"] for r in ts) / len(ts), "calibration": calib.fingerprint(),
                     "mean_pred_tok_s": sum(r["tok_s"] for r in ts) / len(ts), "split": common.SPLIT})


main(sys.argv[1] if len(sys.argv) > 1 else "bench")
