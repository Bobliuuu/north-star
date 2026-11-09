"""Tables for RESULTS.md from results/batchspec_<tag>.jsonl.  python3 scripts/analyze_batchspec.py results/batchspec_bs1.jsonl"""
import json, sys, math, statistics as st
rows = [json.loads(l) for l in open(sys.argv[1])]
G = [r for r in rows if r["phase"] == "group"]
gm = lambda x: math.exp(sum(map(math.log, x)) / len(x))
print("| B | groups | plain agg tok/s | plain per-seq | spec agg tok/s | spec per-seq | ratio geomean (agg) | ratio median | min / max | sum-tokens/sum-time ratio | spec==plain same B | spec==plain-B1 | plain B==plain B1 | mean T / round | verify-slot utilisation | accepted/drafted |")
print("|" + "---|" * 16)
for B in sorted({r["B"] for r in G}):
    g = [r for r in G if r["B"] == B]
    pa = st.mean(r["plain_agg_tok_s"] for r in g); sa = st.mean(r["spec_agg_tok_s"] for r in g)
    ps = st.mean(x for r in g for x in r["plain_seq_tok_s"]); ss = st.mean(x for r in g for x in r["spec_seq_tok_s"])
    rat = [r["spec_agg_tok_s"] / r["plain_agg_tok_s"] for r in g]
    tw = lambda k: sum(sum(r["n_" + k]) for r in g) / sum(r[k + "_wall"] for r in g)
    eq = [x for r in g for x in r["spec_eq_plain_same_B"]]
    e1 = [x for r in g for x in r["spec_eq_B1"] if x != "na"]
    p1 = [x for r in g for x in r["plain_eq_B1"] if x != "na"]
    print("| %d | %d | %.0f | %.0f | %.0f | %.0f | %.3fx | %.3fx | %.2f / %.2f | %.3fx | %d/%d | %d/%d | %d/%d | %.2f | %.2f | %.2f |" % (
        B, len({r["group"] for r in g}), pa, ps, sa, ss, gm(rat), st.median(rat), min(rat), max(rat), tw("spec") / tw("plain"), sum(eq), len(eq), sum(e1), len(e1), sum(p1), len(p1),
        st.mean(r["mean_T"] for r in g), st.mean(r["slot_util"] for r in g), sum(r["accepted"] for r in g) / max(1, sum(r["drafted"] for r in g))))
print("\ncost (graph ms, median; 'padded' = control on copies of one token, only in runs after bs1):")
for r in rows:
    if r["phase"] == "cost": print("B", r["B"], r.get("inputs", "leftover warmup inputs (bs1: NOT representative for T>1)"), {k: round(v, 2) for k, v in r["ms"].items()},
                                   "padded", {k: round(v, 2) for k, v in r.get("ms_padded", {}).items()})
print("\nlive per-step cost (ms; plain step = plain_wall / max n; spec round = spec_wall / rounds; dev = inside graph replay + .tolist() sync):")
print("| B | plain ms/step | spec ms/round | mean T | spec dev ms/round | spec host ms/round |")
print("|---|---|---|---|---|---|")
for B in sorted({r["B"] for r in G}):
    g = [r for r in G if r["B"] == B]
    dev = [r["dev_wall"] * 1e3 / r["rounds"] for r in g if "dev_wall" in r]; host = [r["host_wall"] * 1e3 / r["rounds"] for r in g if "host_wall" in r]
    print("| %d | %.2f | %.2f | %.2f | %s | %s |" % (B, st.mean(r["plain_wall"] * 1e3 / max(r["n_plain"]) for r in g), st.mean(r["spec_wall"] * 1e3 / r["rounds"] for r in g),
          st.mean(r["mean_T"] for r in g), "%.2f" % st.mean(dev) if dev else "n/a", "%.2f" % st.mean(host) if host else "n/a"))
