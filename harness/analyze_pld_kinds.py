"""Live wall speedup per task kind (geomean over tasks) for each config, across one or more result files; plus overall."""
import json, sys, collections, statistics as st

rows = []
for f in sys.argv[1:]:
    rows += [json.loads(l) for l in open(f) if l.strip() and '"phase": "live"' in l]
gm = st.geometric_mean
for cell in ("natural", "ctx2048", "ctx8192"):
    L = [r for r in rows if r["cell"] == cell]
    cfgs = sorted({(r["proposer"], r["gamma"]) for r in L})
    kinds = sorted({r["kind"] for r in L})
    print(f"\n#### {cell} (MEASURED live wall speedup vs plain graphed decode, geomean over tasks; n per kind in header)")
    print("| config | all | " + " | ".join(f"{k} (n={len({r['task'] for r in L if r['kind']==k})})" for k in kinds) + " | worst task | identical |")
    print("|" + "---|" * (len(kinds) + 5))
    for c in cfgs:
        R = [r for r in L if (r["proposer"], r["gamma"]) == c]
        cells = []
        for k in kinds:
            v = [r["speedup_wall"] for r in R if r["kind"] == k]
            cells.append("%.2f" % gm(v) if v else "-")
        print("| %s g%d | %.3f | %s | %.3f | %d/%d |" % (c[0], c[1], gm([r["speedup_wall"] for r in R]), " | ".join(cells), min(r["speedup_wall"] for r in R), sum(r["identical"] for r in R), len(R)))
