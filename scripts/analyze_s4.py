"""Summarize S4 (PLD on the megakernel vs fast executor). usage: python scripts/analyze_s4.py results/s4_mk_vast.jsonl results/s4_fast_vast.jsonl
Per (exec,mode,gamma,cell,task) tok/s is the mean over passes (passes interleave mk/fast launches ABAB). Cell value = geomean over tasks.
Ratios are per-task ratios (same task, same pass-mean), then geomean over tasks."""
import json, sys, math, collections

runs = []
for p in sys.argv[1:]:
    runs += [json.loads(l) for l in open(p) if l.strip()]
runs = [r for r in runs if r.get("kind") == "run"]
gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")
key = lambda r: (r["exec"], r["mode"], r.get("gamma", 0))
tps = collections.defaultdict(list)    # (key,cell,task) -> [tok_s per pass]
cuda = collections.defaultdict(list)
for r in runs:
    tps[(key(r), r["cell"], r["task"])].append(r["tok_s"])
    if "tok_s_cuda" in r:
        cuda[(key(r), r["cell"], r["task"])].append(r["tok_s_cuda"])
mean = lambda v: sum(v) / len(v)
cells = sorted({r["cell"] for r in runs}, key=lambda c: (c != "natural", c))
keys = sorted({key(r) for r in runs})
tasks = {c: sorted({r["task"] for r in runs if r["cell"] == c}) for c in cells}
npass = sorted({r["pass"] for r in runs})
print("passes", npass, "runs", len(runs))
print("\nMEASURED geomean wall tok/s over tasks (mean of passes)")
print("%-22s" % "config" + "".join("%12s" % c for c in cells))
for k in keys:
    row = []
    for c in cells:
        v = [mean(tps[(k, c, t)]) for t in tasks[c] if (k, c, t) in tps]
        row.append("%12.1f" % gm(v) if v else "%12s" % "-")
    print("%-22s" % ("%s-%s%s" % (k[0], k[1], ("-g%d" % k[2]) if k[2] else "")) + "".join(row))
print("\nMEASURED kernel-time-only tok/s (megakernel; sum of cuda-event ms, excludes python host loop)")
for k in keys:
    if k[0] != "mk":
        continue
    row = []
    for c in cells:
        v = [mean(cuda[(k, c, t)]) for t in tasks[c] if (k, c, t) in cuda]
        row.append("%12.1f" % gm(v) if v else "%12s" % "-")
    print("%-22s" % ("mk-%s%s" % (k[1], ("-g%d" % k[2]) if k[2] else "")) + "".join(row))


def ratio(num, den, c):
    v = [mean(tps[(num, c, t)]) / mean(tps[(den, c, t)]) for t in tasks[c] if (num, c, t) in tps and (den, c, t) in tps]
    return gm(v), len(v), (min(v) if v else float("nan")), (max(v) if v else float("nan"))


print("\nMEASURED ratios (geomean over tasks; [min,max] per task)")
mkp, fp = ("mk", "plain", 0), ("fast", "plain", 0)
pairs = []
for k in keys:
    if k[1] == "pld":
        own = mkp if k[0] == "mk" else fp
        pairs.append(("%s-pld-g%d / %s-plain" % (k[0], k[2], k[0]), k, own))
for k in keys:
    if k[0] == "mk" and k[1] == "pld":
        pairs.append(("mk-pld-g%d / fast-plain" % k[2], k, fp))
pairs.append(("mk-plain / fast-plain", mkp, fp))
for k in keys:
    if k[0] == "mk" and k[1] == "pld":
        for k2 in keys:
            if k2[0] == "fast" and k2[1] == "pld" and k2[2] == k[2]:
                pairs.append(("mk-pld-g%d / fast-pld-g%d" % (k[2], k[2]), k, k2))
for name, a, b in pairs:
    print("%-34s" % name + "".join("  %s: %.3fx (n=%d, %.2f-%.2f)" % ((c,) + ratio(a, b, c)) for c in cells))

print("\nOUTPUT IDENTITY (same executor: PLD vs its own plain greedy, every run)")
ident = collections.defaultdict(lambda: [0, 0, 0, 0])
for r in runs:
    if r["mode"] != "pld":
        continue
    k = (r["exec"], r.get("gamma", 0), r["cell"])
    ok = r.get("identical_to_mk_plain", r.get("identical_to_fast_plain"))
    d = r.get("fd_vs_mk_plain", r.get("fd_vs_fast_plain"))
    ident[k][0] += 1; ident[k][1] += bool(ok)
    if not ok and r["exec"] == "mk":
        m = max(r.get("margin_plain_at_fd", 99), r.get("margin_pld_at_fd", 99)) if "margin_plain_at_fd" in r else 99
        ident[k][2] += (m <= 0.25)  # divergence at a bf16 near-tie in at least... (both margins <=0.25 would be min; we use the larger)
        ident[k][3] += 1
for k in sorted(ident):
    n, ok, nt, nd = ident[k]
    print("%s g=%d %-9s identical %d/%d" % (k[0], k[1], k[2], ok, n))
mm = [r for r in runs if r["exec"] == "mk" and r["mode"] == "pld" and not r["identical_to_mk_plain"] and r.get("fd_vs_mk_plain") is not None and "margin_plain_at_fd" in r]
print("mk divergent PLD runs: %d (of %d PLD runs). At the first divergent token, logged top1-top2 margin of the plain row / the PLD row:" % (len(mm), sum(1 for r in runs if r["exec"] == "mk" and r["mode"] == "pld")))
for th in (0.25, 0.5, 1.0, 2.0):
    print("  max(margin_plain, margin_pld) <= %.2f: %d" % (th, sum(1 for r in mm if max(r["margin_plain_at_fd"], r["margin_pld_at_fd"]) <= th)))
print("  max margin > 2.0: %d ; largest max-margin %.3f" % (sum(1 for r in mm if max(r["margin_plain_at_fd"], r["margin_pld_at_fd"]) > 2.0), max([max(r["margin_plain_at_fd"], r["margin_pld_at_fd"]) for r in mm] or [0])))
# cross-executor plain identity
by = collections.defaultdict(dict)
for r in runs:
    if r["mode"] == "plain" and r["pass"] == min(npass):
        by[(r["cell"], r["task"])][r["exec"]] = r["gen"]
ne = sum(1 for v in by.values() if "mk" in v and "fast" in v and v["mk"] == v["fast"])
nb = sum(1 for v in by.values() if "mk" in v and "fast" in v)
print("mk plain == fast plain (first pass): %d/%d" % (ne, nb))
acc = collections.defaultdict(list)
for r in runs:
    if r["mode"] == "pld":
        acc[(r["exec"], r["gamma"], r["cell"])].append(r["tok_per_round"])
print("\nMEASURED tokens/round (mean over runs)")
for k in sorted(acc):
    print("%s g=%d %-9s %.3f" % (k[0], k[1], k[2], mean(acc[k])))
