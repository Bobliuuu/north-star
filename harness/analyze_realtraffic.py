"""Summarize results/realtraffic_<tag>.jsonl -> markdown (stdout). Speedup per task = mean spec tok/s / mean plain tok/s over that task's reps
(same executor, same host loop). Geomeans are over tasks. python3 harness/analyze_realtraffic.py results/realtraffic_rt1.jsonl"""
import json, math, sys, collections, statistics as st

rows = [json.loads(l) for l in open(sys.argv[1])]
plain = collections.defaultdict(list); live = collections.defaultdict(list); info = {}
for r in rows:
    if r["phase"] == "plain": plain[r["task"]].append(r); info[r["task"]] = (r["kind"], r["prompt_len"], r["n"])
    if r["phase"] == "live": live[(r["task"], (r["proposer"], r["gamma"], r["mode"]))].append(r)
cfgs = sorted({k[1] for k in live}, key=lambda c: (c[2] != "fixed", c))
gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")
tasks = sorted(info)
cfgname = lambda c: "%s g%d %s" % c


def sp(t, c):
    L = live.get((t, c));
    if not L: return None
    ptps = st.mean(r["tok_s"] for r in plain[t]); return st.mean(r["spec_tok_s"] for r in L) / ptps


S = {c: {t: sp(t, c) for t in tasks if sp(t, c)} for c in cfgs}
print("## Tasks %d; skipped: %s\n" % (len(tasks), sorted(r["task"] for r in rows if r["phase"] == "skip")))
lens = sorted(info[t][1] for t in tasks); q = lambda p: lens[int(p * (len(lens) - 1))]
print("Prompt tokens: min %d p25 %d median %d p75 %d max %d; generated n (mean %.0f, <48 tokens: %d tasks)\n" % (lens[0], q(.25), q(.5), q(.75), lens[-1], st.mean(info[t][2] for t in tasks), sum(info[t][2] < 48 for t in tasks)))
print("| config | geomean | arith mean | median | wins(>1.02) | ties | losses(<0.98) | worst | best | exact (runs) | tokens in draft rounds | accepted draft tok / generated |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
for c in cfgs:
    v = list(S[c].values()); L = [r for (t, cc), rs in live.items() if cc == c for r in rs]
    ex = sum(r["identical"] for r in L)
    print("| %s | %.3fx | %.3fx | %.3fx | %d | %d | %d | %.3fx | %.3fx | %d/%d | %.1f%% | %.1f%% |" % (cfgname(c), gm(v), st.mean(v), st.median(v), sum(x > 1.02 for x in v), sum(0.98 <= x <= 1.02 for x in v), sum(x < .98 for x in v), min(v), max(v), ex, len(L),
          100 * sum(r["tok_in_draft_rounds"] for r in L) / sum(r["n"] for r in L), 100 * sum(r["accepted"] for r in L) / sum(r["n"] for r in L)))
kinds = sorted({info[t][0] for t in tasks})
print("\n### Per-category geomean speedup vs exact plain (n tasks, median prompt tokens)\n")
print("| category | n | med prompt | " + " | ".join(cfgname(c) for c in cfgs) + " |\n|---|---|---|" + "---|" * len(cfgs))
for k in kinds:
    ts = [t for t in tasks if info[t][0] == k]
    print("| %s | %d | %d | " % (k, len(ts), sorted(info[t][1] for t in ts)[len(ts) // 2]) + " | ".join("%.2fx" % gm([S[c][t] for t in ts if t in S[c]]) for c in cfgs) + " |")
print("\n### Worst 8 tasks per config\n")
for c in cfgs:
    w = sorted(S[c].items(), key=lambda kv: kv[1])[:8]
    print("- %s: " % cfgname(c) + ", ".join("%s %.2fx (P=%d,n=%d)" % (t, x, info[t][1], info[t][2]) for t, x in w))
print("\n### Variance: 2-rep subset (rep-to-rep spread of spec tok/s, and of the speedup ratio using each rep's own plain)\n")
for c in cfgs[:2]:
    sp_, d = [], []
    for t in tasks:
        L = live.get((t, c))
        if L and len(L) == 2 and len(plain[t]) == 2:
            a = sorted(L, key=lambda r: r["rep"]); p = sorted(plain[t], key=lambda r: r["rep"])
            s0, s1 = a[0]["spec_tok_s"] / p[0]["tok_s"], a[1]["spec_tok_s"] / p[1]["tok_s"]
            d.append(abs(s0 - s1) / ((s0 + s1) / 2)); sp_.append((s0, s1))
    if d: print("- %s: %d tasks, per-task rep-to-rep ratio difference mean %.1f%% max %.1f%%; geomean rep0 %.3fx rep1 %.3fx" % (cfgname(c), len(d), 100 * st.mean(d), 100 * max(d), gm([a for a, _ in sp_]), gm([b for _, b in sp_])))
print("\n### Sensitivity: geomean excluding tasks with n<48 generated tokens, and excluding the 22 rep2-subset tasks (single-rep tasks only)\n")
for c in cfgs:
    v1 = [S[c][t] for t in S[c] if info[t][2] >= 48]; v2 = [S[c][t] for t in S[c] if len(plain[t]) == 1]
    print("- %s: n>=48 %.3fx (%d tasks); single-rep tasks %.3fx (%d)" % (cfgname(c), gm(v1), len(v1), gm(v2), len(v2)))
print("\n### Non-identical runs (exactness failures)\n")
bad = [(t, c, r["rep"], r["fd"]) for (t, c), rs in live.items() for r in rs if not r["identical"]]
print(bad[:30] if bad else "none")
