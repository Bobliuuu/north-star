"""Summarize results/pld_<tag>.jsonl -> markdown tables (live wall speedups and simulated grid, by cell and task kind)."""
import json, sys, collections, statistics as st

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
live = [r for r in rows if r["phase"] == "live"]
sim = [r for r in rows if r["phase"] == "sim"]
gm = lambda xs: st.geometric_mean(xs) if xs else float("nan")


def table(rs, keyf, valf, title):
    g = collections.defaultdict(list)
    for r in rs:
        g[keyf(r)].append(valf(r))
    print("\n###", title)
    for k in sorted(g):
        print("| %s | n=%d | geomean %.3f | min %.3f | max %.3f |" % (k, len(g[k]), gm(g[k]), min(g[k]), max(g[k])))


for cell in sorted({r["cell"] for r in live}):
    L = [r for r in live if r["cell"] == cell]
    table(L, lambda r: "%s g%d" % (r["proposer"], r["gamma"]), lambda r: r["speedup_wall"], f"LIVE {cell}: wall speedup by config (all tasks)")
    for cfg in sorted({(r["proposer"], r["gamma"]) for r in L}):
        pass
    table(L, lambda r: "%-9s %s g%d" % (r["kind"], r["proposer"], r["gamma"]), lambda r: r["speedup_wall"], f"LIVE {cell}: by kind x config")
    print("\nidentical to plain: %d/%d; first_div values: %s" % (sum(r["identical"] for r in L), len(L), sorted({r["first_div"] for r in L if r["first_div"] is not None})[:20]))
    print("mean accepted/verify (rounds with a draft), tok/round, draft_hit_rate by config:")
    g = collections.defaultdict(list)
    for r in L:
        g[(r["proposer"], r["gamma"])].append(r)
    for k, v in sorted(g.items()):
        print("| %s g%d | acc/verify %.2f | tok/round %.2f | hit %.2f | sim-vs-live %.3f/%.3f |" % (k[0], k[1], st.mean(r["mean_accepted_per_verify"] for r in v), st.mean(r["tok_per_round"] for r in v),
              st.mean(r["draft_hit_rate"] for r in v), gm([r["sim_speedup"] for r in v]), gm([r["speedup_wall"] for r in v])))
for cell in sorted({r["cell"] for r in sim}):
    S = [r for r in sim if r["cell"] == cell]
    table(S, lambda r: "%s g%d" % (r["proposer"], r["gamma"]), lambda r: r["speedup"], f"SIM {cell}: speedup by proposer x gamma")
