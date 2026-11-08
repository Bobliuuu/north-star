"""Summarize results/vllm_<tag>.jsonl vs our executors (RESULTS.md H13/H14/H15 data). Usage: python3 scripts/analyze_vllm.py v1 > results/vllm_v1_summary.md"""
import json, math, sys, collections
tag = sys.argv[1] if len(sys.argv) > 1 else "v1"
R = lambda f: [json.loads(l) for l in open(f)]
gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")
v = R(f"results/vllm_{tag}.jsonl")
meta = {r["cfg"]: r for r in v if r["phase"] == "meta"}
reps = collections.defaultdict(list)
for r in v:
    if r["phase"] == "rep": reps[(r["cfg"], r["cell"], r["task"])].append(r)
def best(cfg, cell, key):  # best over reps (H13 convention), per task
    return {t: max(x[key] for x in rs) for (c, ce, t), rs in reps.items() if c == cfg and ce == cell}
def gens(cfg, cell):
    return {t: rs[0]["gen"] for (c, ce, t), rs in reps.items() if c == cfg and ce == cell}
cfgs = [c for c in ["plain", "ngram_k8_n3", "ngram_k12_n2to12", "ngram_k5_n2to5"] if c in meta]
out = []; P = out.append
P(f"# vLLM baseline summary ({tag}); vLLM {meta['plain']['vllm']}, kwargs {meta['plain']['kw']}\n")
# ours
fp2 = [x for x in R("results/fastpld_fp2.jsonl") if x["phase"] == "trace"]
fp3 = [x for x in R("results/fastpld_fp3.jsonl") if x["phase"] == "trace"]
ex2 = R("results/exactspec_ex2.jsonl"); ex2t = {x["task"]: x for x in ex2 if x["phase"] == "trace" and x["cell"] == "natural"}
nat_ids = set(ex2t)
def mean_live(rows, prop, g, cell="natural"):
    d = collections.defaultdict(list)
    for x in rows:
        if x["phase"] == "live" and x["proposer"] == prop and x["gamma"] == g and x.get("cell", "natural") == cell: d[x["task"]].append(x["spec_tok_s"])
    return {t: sum(a) / len(a) for t, a in d.items()}
ours_nat = {
 "ours exact plain (H14)": gm([x["plain_exact_tok_s"] for x in ex2t.values()]),
 "ours exact sam2 g12 (H14)": gm(list(mean_live(ex2, "sam2", 12).values())),
 "ours old-fast plain (H13 fp2 / fp3)": (gm([x["plain_fast_tok_s"] for x in fp2 if x["cell"] == "natural"]), gm([x["plain_fast_tok_s"] for x in fp3 if x["cell"] == "natural"])),
}
rt = R("results/realtraffic_rt1.jsonl")
d = collections.defaultdict(list)
for x in rt:
    if x["phase"] == "plain": d[x["task"]].append(x["tok_s"])
rt_plain = {t: sum(a) / len(a) for t, a in d.items()}
def rt_live(prop, g, mode="fixed"):
    d = collections.defaultdict(list)
    for x in rt:
        if x["phase"] == "live" and x["proposer"] == prop and x["gamma"] == g and x["mode"] == mode: d[x["task"]].append(x["spec_tok_s"])
    return {t: sum(a) / len(a) for t, a in d.items()}
P("## Geomean decode tok/s (per-task best of 2 reps; decode-only = (n-1)/(t_full - t_prefill+1tok))\n")
P("| cell | config | n tasks | decode tok/s | e2e tok/s (incl. prefill) | vs vLLM plain | exact == vLLM plain output | mean n |\n|---|---|---|---|---|---|---|---|")
for cell in ("natural", "mix99"):
    pl = best("plain", cell, "decode_tok_s"); gp = gens("plain", cell)
    for c in cfgs:
        dec = best(c, cell, "decode_tok_s"); e2e = best(c, cell, "e2e_tok_s"); g = gens(c, cell)
        ts = [t for t in dec if t in pl]
        ratio = gm([dec[t] / pl[t] for t in ts])
        ident = sum(g[t] == gp[t] for t in ts if t in gp)
        nn = sum(len(g[t]) for t in ts) / max(1, len(ts))
        P(f"| {cell} | vLLM {c} | {len(dec)} | {gm(list(dec.values())):.1f} | {gm(list(e2e.values())):.1f} | {ratio:.3f}x | {ident}/{len(ts)} | {nn:.0f} |")
P("\n## Ours (from existing data, NOT re-run; different launches/hosts)\n")
for k, x in ours_nat.items():
    P(f"- natural (21), {k}: " + (" / ".join(f"{y:.1f}" for y in x) if isinstance(x, tuple) else f"{x:.1f}") + " tok/s geomean")
P(f"- mix99 (H15), exact plain: {gm(list(rt_plain.values())):.1f}; exact sam2 g12: {gm(list(rt_live('sam2',12).values())):.1f}; exact pld3 g8: {gm(list(rt_live('pld3',8).values())):.1f} tok/s geomean")
P("\n## Task-level ratios vs ours (vLLM decode tok/s / ours), geomean over tasks\n")
plv = best("plain", "natural", "decode_tok_s")
P(f"- natural, vLLM plain / ours exact plain: {gm([plv[t]/ex2t[t]['plain_exact_tok_s'] for t in plv if t in ex2t]):.3f}x; / ours old-fast plain fp2: {gm([plv[x['task']]/x['plain_fast_tok_s'] for x in fp2 if x['cell']=='natural' and x['task'] in plv]):.3f}x; fp3: {gm([plv[x['task']]/x['plain_fast_tok_s'] for x in fp3 if x['cell']=='natural' and x['task'] in plv]):.3f}x")
pm = best("plain", "mix99", "decode_tok_s")
P(f"- mix99, vLLM plain / ours exact plain: {gm([pm[t]/rt_plain[t] for t in pm if t in rt_plain]):.3f}x (n={len([t for t in pm if t in rt_plain])})")
for c in cfgs[1:]:
    for cell, ours in (("natural", {"exact sam2 g12": mean_live(ex2, 'sam2', 12), "exact pld3 g8": mean_live(ex2, 'pld3', 8)}), ("mix99", {"exact sam2 g12": rt_live('sam2', 12), "exact pld3 g8": rt_live('pld3', 8)})):
        dc = best(c, cell, "decode_tok_s")
        for on, od in ours.items():
            ts = [t for t in dc if t in od]
            if ts: P(f"- {cell}, vLLM {c} / ours {on}: {gm([dc[t]/od[t] for t in ts]):.3f}x (n={len(ts)})")
P("\n## Per-category (mix99), vLLM spec / vLLM plain geomean\n")
kinds = collections.defaultdict(list)
for t in pm: kinds[t.rsplit("_", 2)[0]].append(t)
P("| category | " + " | ".join(cfgs[1:]) + " |\n|---|" + "---|" * len(cfgs[1:]))
for k, ts in sorted(kinds.items()):
    cells = []
    for c in cfgs[1:]:
        dc = best(c, "mix99", "decode_tok_s"); cells.append(f"{gm([dc[t]/pm[t] for t in ts if t in dc]):.2f}")
    P(f"| {k} | " + " | ".join(cells) + " |")
errs = [r for r in v if r["phase"] == "task_error"]
P(f"\ntask errors: {len(errs)}")
print("\n".join(out))
