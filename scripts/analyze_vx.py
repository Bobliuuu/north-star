"""Same-host vLLM vs exact v1/v2 (modal/vllm_vs_x2.py).  python scripts/analyze_vx.py vx1"""
import json, math, sys, statistics as st
tag = sys.argv[1] if len(sys.argv) > 1 else "vx1"
gm = lambda v: math.exp(sum(map(math.log, v)) / len(v))
R = lambda f: [json.loads(l) for l in open(f)]
vl = [r for r in R(f"results/vx_{tag}_vllm.jsonl") if r["phase"] == "rep" and r["cell"] == "mix99"]
vbest = {}
for r in vl:
    vbest[r["task"]] = max(vbest.get(r["task"], 0), r["decode_tok_s"])
for part in ("ours1", "ours2"):
    rows = R(f"results/vx_{tag}_{part}.jsonl")
    meta = rows[0]
    tr = [r for r in rows if r["phase"] == "traffic"]
    ks = [r["task"] for r in tr if r["task"] in vbest]
    o = {r["task"]: r for r in tr}
    print(f"## {part}: {len(tr)} tasks ({meta.get('torch')}, triton {meta.get('triton')}, {meta.get('gpu')})")
    print("| arm | geomean tok/s | vs vLLM plain (task geomean) | wins / losses vs vLLM |")
    print("|---|---|---|---|")
    vg = gm([vbest[k] for k in ks])
    print(f"| vLLM 0.33 plain (same container, decode-only, best of 2) | {vg:.1f} | 1.000x | - |")
    for key, name in (("v1_tok_s", "ours exact v1 plain"), ("s_tok_s", "ours v1 kernels + streams"), ("ts_tok_s", "ours exact v2 plain"), ("ts_spec_tok_s", "ours exact v2 + PLD sam2 g12")):
        rat = [o[k][key] / vbest[k] for k in ks]
        print(f"| {name} | {gm([o[k][key] for k in ks]):.1f} | {gm(rat):.3f}x | {sum(x > 1.02 for x in rat)} / {sum(x < 0.98 for x in rat)} |")
    print("exactness: v2 spec == v2 plain", sum(r["spec_identical"] for r in tr), "/", len(tr), "| v2 plain == v1 plain", sum(r["ts_equals_v1"] for r in tr), "/", len(tr),
          "| v1 rep2/rep1 tok/s", round(gm([r["v1b_tok_s"] / r["v1_tok_s"] for r in tr]), 4))
    print()
