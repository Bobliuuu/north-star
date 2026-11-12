"""Summarize a North-in-North run: python harness/selfimprove_bench/nin_analyze.py results/nin_<tag>_*.json [--plot out.png]"""
import json, sys, statistics as st

def main(p, plot=None):
    r = json.load(open(p)); fin = r["final"]; rows = []
    print(f"{r['rounds']} rounds x {r['per_round']} evals/round; North calls {r['north']['calls']}, gen {r['north']['gen_s']:.0f}s; total {r['secs']:.0f}s")
    print("| arm/session | evals | best dev | its held | passes gate | passing configs | best-so-far by round (final-measured) |\n|---|---|---|---|---|---|---|")
    summ = {}
    for name, recs in r["arms"].items():
        cfgs = [fin[x["cid"]] for x in recs if x["cid"] in fin and fin[x["cid"]].get("dev_speedup")]
        def ok(c):
            d, h = c["dev_speedup"], c.get("held_speedup") or 0
            return d > 1.03 and h >= 1 + 0.25 * (d - 1) and (c.get("check") or {}).get("passed")
        best = max(cfgs, key=lambda c: c["dev_speedup"], default=None)
        curve, b = [], 1.0
        for i, x in enumerate(recs):
            c = fin.get(x["cid"], {}); b = max(b, c.get("dev_speedup") or 0)
            if (i + 1) % r["per_round"] == 0: curve.append(round(b, 3))
        npass = sum(1 for c in cfgs if ok(c))
        summ[name] = dict(best=best["dev_speedup"] if best else None, held=best.get("held_speedup") if best else None, npass=npass, curve=curve)
        print(f"| {name} | {len(recs)} | {summ[name]['best']:.3f} | {summ[name]['held'] or 0:.3f} | {'yes' if any(ok(c) for c in cfgs) else 'no'} | {npass}/{len(cfgs)} | {curve} |")
    print("\nfallbacks:", json.dumps(r["fallbacks"]))
    for arm in ("north", "north_in_north"):
        v = [s["best"] for k, s in summ.items() if k.startswith(arm + "#")]
        if v: print(f"{arm}: best dev per session {[round(x,3) for x in v]} mean {st.mean(v):.3f}")
    if plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4)); col = dict(standard="tab:gray", north="tab:blue", north_in_north="tab:red")
        for k, s in summ.items():
            a = k.split("#")[0]; ax.plot(range(1, len(s["curve"]) + 1), s["curve"], color=col[a], alpha=.8, label=a if k.endswith("#0") else None)
        ax.axhline(1.03, ls="--", c="k", lw=.8); ax.set_xlabel("round (3 measured evals each)"); ax.set_ylabel("best dev speedup so far"); ax.legend(); fig.tight_layout(); fig.savefig(plot, dpi=150)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[sys.argv.index("--plot") + 1] if "--plot" in sys.argv else None)
