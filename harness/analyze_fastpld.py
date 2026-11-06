"""python harness/analyze_fastpld.py results/fastpld_fp2.jsonl [more.jsonl ...] -> markdown tables (tok/s, speedups, variance, identity, margins)."""
import json, math, sys, collections


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


rows = []
for f in sys.argv[1:]:
    for i, l in enumerate(open(f)):
        d = json.loads(l); d["_file"] = f; rows.append(d)
live = [r for r in rows if r["phase"] == "live"]
trace = [r for r in rows if r["phase"] == "trace"]
plainrep = [r for r in rows if r["phase"] == "plain_rep"]
cells = sorted({r["cell"] for r in live}, key=lambda c: (c != "natural", c))
cfgs = sorted({(r["proposer"], r["gamma"]) for r in live})
print("## Wall tok/s (geomean over tasks; each task = mean of its reps; reps pooled across files)")
print("| cell | n tasks | graphed PyTorch plain | fast plain | " + " | ".join("%s g%d" % c for c in cfgs) + " |")
print("|---|---|---|---|" + "---|" * len(cfgs))
for c in cells:
    tt = [t for t in trace if t["cell"] == c]
    # per task means across all files/reps
    pb = collections.defaultdict(list); pf = collections.defaultdict(list)
    for r in plainrep:
        if r["cell"] == c: (pb if r["exec"] == "base" else pf)[r["task"]].append(r["tok_s"])
    tasks = sorted(pb)
    row = ["%.1f" % gm([sum(pb[t]) / len(pb[t]) for t in tasks]), "%.1f" % gm([sum(pf[t]) / len(pf[t]) for t in tasks])]
    for cf in cfgs:
        d = collections.defaultdict(list)
        for r in live:
            if r["cell"] == c and (r["proposer"], r["gamma"]) == cf: d[r["task"]].append(r["spec_tok_s"])
        row.append("%.1f" % gm([sum(v) / len(v) for v in d.values()]))
    print("| %s | %d | %s |" % (c, len(tasks), " | ".join(row)))
print("\n## Speedup (geomean over tasks of mean-over-reps tok/s ratio) vs graphed PyTorch plain (93.7 orig baseline) and vs fast plain")
print("| cell | config | vs graphed PyTorch plain | vs fast plain | tok/round | rep-to-rep spread (max |r1-r2|/mean over tasks) | reps/task | worst task vs fast | best task vs fast |")
print("|---|---|---|---|---|---|---|---|---|")
for c in cells:
    pb = collections.defaultdict(list); pf = collections.defaultdict(list)
    for r in plainrep:
        if r["cell"] == c: (pb if r["exec"] == "base" else pf)[r["task"]].append(r["tok_s"])
    for cf in cfgs:
        d = collections.defaultdict(list); tpr = collections.defaultdict(list)
        for r in live:
            if r["cell"] == c and (r["proposer"], r["gamma"]) == cf: d[r["task"]].append(r["spec_tok_s"]); tpr[r["task"]].append(r["tok_per_round"])
        sb = {t: (sum(v) / len(v)) / (sum(pb[t]) / len(pb[t])) for t, v in d.items()}
        sf = {t: (sum(v) / len(v)) / (sum(pf[t]) / len(pf[t])) for t, v in d.items()}
        spread = max((max(v) - min(v)) / (sum(v) / len(v)) for v in d.values())
        print("| %s | %s g%d | %.2fx | %.2fx | %.2f | %.1f%% | %d | %.2fx | %.2fx |" % (c, cf[0], cf[1], gm(sb.values()), gm(sf.values()), gm([sum(v) / len(v) for v in tpr.values()]),
              100 * spread, min(len(v) for v in d.values()), min(sf.values()), max(sf.values())))
print("\n## By kind (geomean speedup vs graphed PyTorch plain / vs fast plain; n tasks)")
kinds = sorted({r["kind"] for r in live})
print("| cell | config | " + " | ".join(kinds) + " |")
print("|---|---|" + "---|" * len(kinds))
for c in cells:
    pb = collections.defaultdict(list); pf = collections.defaultdict(list)
    for r in plainrep:
        if r["cell"] == c: (pb if r["exec"] == "base" else pf)[r["task"]].append(r["tok_s"])
    for cf in cfgs:
        out = []
        for k in kinds:
            d = collections.defaultdict(list)
            for r in live:
                if r["cell"] == c and (r["proposer"], r["gamma"]) == cf and r["kind"] == k: d[r["task"]].append(r["spec_tok_s"])
            if not d: out.append("-"); continue
            sb = [(sum(v) / len(v)) / (sum(pb[t]) / len(pb[t])) for t, v in d.items()]
            sf = [(sum(v) / len(v)) / (sum(pf[t]) / len(pf[t])) for t, v in d.items()]
            out.append("%.2f / %.2f (n=%d)" % (gm(sb), gm(sf), len(d)))
        print("| %s | %s g%d | %s |" % (c, cf[0], cf[1], " | ".join(out)))
print("\n## Token identity (all reps of all files; counts of live runs)")
print("| cell | config | identical to fast-plain | identical to graphed-plain | identical to HF (where run) | runs |")
print("|---|---|---|---|---|---|")
for c in cells:
    for cf in cfgs:
        rs = [r for r in live if r["cell"] == c and (r["proposer"], r["gamma"]) == cf]
        hf = [r for r in rs if r["identical_hf"] != "na"]
        print("| %s | %s g%d | %d/%d | %d/%d | %s | %d |" % (c, cf[0], cf[1], sum(r["identical_fast"] for r in rs), len(rs), sum(r["identical_base"] for r in rs), len(rs),
              ("%d/%d" % (sum(r["identical_hf"] for r in hf), len(hf))) if hf else "n/a", len(rs)))
print("\n## Plain executors vs each other / HF (traces)")
for c in cells:
    tt = [t for t in trace if t["cell"] == c]
    print("- %s: fast-plain == graphed-plain on %d/%d tasks; fast-plain == HF on %s; graphed-plain == HF on %s" % (c, sum(t["fd_fast_vs_base"] is None for t in tt), len(tt),
          "%d/%d" % (sum(t["fd_fast_vs_hf"] is None for t in tt if t["fd_fast_vs_hf"] != "na"), sum(t["fd_fast_vs_hf"] != "na" for t in tt)) if any(t["fd_fast_vs_hf"] != "na" for t in tt) else "n/a",
          "%d/%d" % (sum(t["fd_base_vs_hf"] is None for t in tt if t["fd_base_vs_hf"] != "na"), sum(t["fd_base_vs_hf"] != "na" for t in tt)) if any(t["fd_base_vs_hf"] != "na" for t in tt) else "n/a"))
print("\n## Logit margins at divergences (teacher-forced on common prefix; a = reference token, b = diverging token; logit_a_minus_b = how far a beat b)")
print("| cell | task | what | at token | top2 margin (fast T=1) | logit a-b (fast T=1) | top2 margin (HF) | logit a-b (HF) |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    if r["phase"] == "margin":
        f, h = r.get("fast_t1", {}), r.get("hf", {})
        print("| %s | %s | %s | %d | %.3f | %.3f | %s | %s |" % (r["cell"], r["task"], r["what"], r["fd"], f.get("top2_margin", float("nan")), f.get("logit_a_minus_b", float("nan")),
              "%.3f" % h["top2_margin"] if h else "n/a", "%.3f" % h["logit_a_minus_b"] if h else "n/a"))
errs = [r for r in rows if r["phase"] in ("task_error", "margin_error", "fatal")]
print("\nerrors:", len(errs), [e.get("task") or e.get("sec") for e in errs])
