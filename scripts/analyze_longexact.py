"""Summarize results/longexact_*.jsonl: per ctx geomean tok/s and same-executor ratios, identity counts. Usage: analyze_longexact.py results/longexact_le1.jsonl results/longexact_le2.jsonl"""
import sys, json, math, collections
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
gm = lambda v: math.exp(sum(map(math.log, v)) / len(v))
for cell in sorted({r["cell"] for r in rows if "cell" in r and r["kind"] == "live"}, key=lambda c: int(c[3:])):
    tr = {r["task"]: r for r in rows if r["kind"] == "trace" and r["cell"] == cell}
    print(f"== {cell}: tasks {len(tr)}")
    for ex in ("exact", "old"):
        lv = [r for r in rows if r["kind"] == "live" and r["cell"] == cell and r["exec"] == ex]
        if not lv: continue
        pk = "plain_exact_tok_s" if ex == "exact" else "plain_old_tok_s"
        bytask = collections.defaultdict(list)
        for r in lv: bytask[r["task"]].append(r)
        ratios, ratios_x, spec, spec_x, plain = [], [], [], [], []
        for t, rs in bytask.items():
            p = tr[t][pk]; s = sum(r["spec_tok_s"] for r in rs) / len(rs); sx = sum(r["spec_tok_s_excl_index"] for r in rs) / len(rs)
            ratios.append(s / p); ratios_x.append(sx / p); spec.append(s); spec_x.append(sx); plain.append(p)
        ident = sum(r["identical_same_exec"] for r in lv)
        idx = sum(r["index_s"] for r in lv) / len(lv)
        print(f" {ex}: plain {gm(plain):.1f} tok/s | spec {gm(spec):.1f} (excl prompt-index {gm(spec_x):.1f}) | ratio {gm(ratios):.3f}x (excl index {gm(ratios_x):.3f}x) min {min(ratios):.2f} max {max(ratios):.2f} | identical {ident}/{len(lv)} | mean index_s {idx:.2f} | tok/round {sum(r['tok_per_round'] for r in lv)/len(lv):.2f}")
        print("   per-task ratio:", {t: round(a, 2) for t, a in zip(bytask, ratios)})
