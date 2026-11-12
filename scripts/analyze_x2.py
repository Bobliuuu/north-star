"""Summarize an exact-v2 run (results/x2_<tag>.jsonl from harness/bench_x2.py).  python scripts/analyze_x2.py results/x2_x3.jsonl"""
import json, math, sys, statistics as st


def gm(v):
    return math.exp(sum(math.log(x) for x in v) / len(v))


rows = [json.loads(l) for l in open(sys.argv[1])]
print("## Step graph ms (median of 30 replays)\n")
steps = [r for r in rows if r["phase"] == "step"]
for ctx in sorted({r["ctx"] for r in steps}):
    print(f"ctx {ctx}:")
    print("| variant | " + " | ".join(f"T={T}" for T in (1, 2, 4, 8, 13)) + " |")
    print("|---|" + "---|" * 5)
    for v in dict.fromkeys(r["variant"] for r in steps):
        c = {r["T"]: r["ms_med"] for r in steps if r["ctx"] == ctx and r["variant"] == v}
        print(f"| {v} | " + " | ".join("%.3f" % c[T] for T in (1, 2, 4, 8, 13)) + " |")
inv = [r for r in rows if r["phase"] == "invariance"]
print("\ninvariance (T=13 block rows == T=1):", {f'{r["variant"]}@{r["ctx"]}': r["rows_equal"] for r in inv})
for r in rows:
    if r["phase"] == "v1_vs_s":
        print("v1 vs s/ts logits:", r)
tr = [r for r in rows if r["phase"] == "traffic"]
if tr:
    print(f"\n## Real traffic ({len(tr)} tasks, 192 new tokens, greedy)\n")
    v1 = [r["v1_tok_s"] for r in tr]; v1b = [r["v1b_tok_s"] for r in tr]; s = [r["s_tok_s"] for r in tr]; ts = [r["ts_tok_s"] for r in tr]; sp = [r["ts_spec_tok_s"] for r in tr]
    print("| path | mean tok/s | median | geomean vs v1 |")
    print("|---|---|---|---|")
    for name, v in (("v1 exact plain (H15/H17 path)", v1), ("v1 exact plain, 2nd rep", v1b), ("s: v1 kernels + concurrent streams", s), ("ts: tuned tiles + streams (exact v2 plain)", ts), ("ts + PLD sam2 g12 (exact v2 spec)", sp)):
        print(f"| {name} | {st.mean(v):.1f} | {st.median(v):.1f} | {gm([a / b for a, b in zip(v, v1)]):.3f}x |")
    sr = [a / b for a, b in zip(sp, ts)]
    print(f"\nspec vs v2 plain: geomean {gm(sr):.3f}x, wins(>1.02) {sum(x > 1.02 for x in sr)}, ties {sum(0.98 <= x <= 1.02 for x in sr)}, losses(<0.98) {sum(x < 0.98 for x in sr)}, worst {min(sr):.3f}x, best {max(sr):.3f}x")
    print("exactness: spec == v2 plain", sum(r["spec_identical"] for r in tr), "/", len(tr), "| s plain == v1 plain", sum(r["s_equals_v1"] for r in tr), "/", len(tr),
          "| v2 (ts) plain == v1 plain", sum(r["ts_equals_v1"] for r in tr), "/", len(tr))
    errs = [r for r in rows if r["phase"] in ("task_error", "fatal")]
    print("errors:", len(errs))
    kinds = sorted({r["kind"] for r in tr})
    print("\n| category | n | v1 plain | v2 plain | v2 spec | spec/v2 plain |\n|---|---|---|---|---|---|")
    for k in kinds:
        g = [r for r in tr if r["kind"] == k]
        print(f'| {k} | {len(g)} | {st.mean(r["v1_tok_s"] for r in g):.0f} | {st.mean(r["ts_tok_s"] for r in g):.0f} | {st.mean(r["ts_spec_tok_s"] for r in g):.0f} | {gm([r["ts_spec_tok_s"] / r["ts_tok_s"] for r in g]):.2f}x |')
