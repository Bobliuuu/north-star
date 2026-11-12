"""Summarize a serving-engine validation (results/serve_<tag>.jsonl from harness/bench_serve.py).  python scripts/analyze_serve.py sv1"""
import json, math, sys
tag = sys.argv[1] if len(sys.argv) > 1 else "sv1"
gm = lambda v: math.exp(sum(map(math.log, v)) / len(v))
rows = [json.loads(l) for l in open(f"results/serve_{tag}.jsonl")]
for sec in ("mix", "pool", "long", "b2"):
    rs = [r for r in rows if r["phase"] == sec]
    if not rs:
        continue
    for B in sorted({r["B"] for r in rs}):
        g = [r for r in rs if r["B"] == B]
        sp = [r["speedup"] for r in g]
        print(f"{sec} B={B} n={len(g)} path={g[0]['path']}: engine {gm([r['tok_s'] for r in g]):.1f} vs plain {gm([r['plain_tok_s'] for r in g]):.1f} tok/s,"
              f" geomean {gm(sp):.3f}x, wins {sum(x > 1.02 for x in sp)} losses {sum(x < 0.98 for x in sp)} worst {min(sp):.3f} best {max(sp):.3f},"
              f" exact {sum(r['exact'] for r in g)}/{len(g)}, max index wait {max((r['index_wait_s'] or 0) for r in g):.3f}s")
eq = [r for r in rows if r["phase"] == "pool_vs_single"]
if eq:
    print("batched rows == B=1 outputs:", {r["B"]: f"{sum(r['rows_equal_B1'])}/{len(r['rows_equal_B1'])}" for r in eq})
print("errors:", sum(r["phase"] in ("error", "fatal") for r in rows))
