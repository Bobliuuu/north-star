"""CPU analysis of results/selfimprove_bench_*.jsonl. A config 'passes' when it meets the same gate shape as the live loop:
dev speedup > 1.03, held-out speedup >= 1 + 0.25*(dev-1) (no shape specialisation), and correctness PASS."""
import json, sys, statistics, collections

GAIN = 1.03


def passes(r):
    d, h = r.get("dev_speedup"), r.get("held_speedup")
    if r.get("build_rc") or not d or not h:
        return False
    return d > GAIN and h >= 1 + 0.25 * (d - 1) and bool((r.get("check") or {}).get("passed"))


def summarize(rows):
    by = collections.defaultdict(list)
    for r in rows:
        for a in r["arm"].split("|"):   # a config shared by several arms counts for each of them
            by[a].append(r)
    out = {}
    for arm, rs in sorted(by.items()):
        sp = [r["dev_speedup"] for r in rs if r.get("dev_speedup")]
        out[arm] = dict(n=len(rs), compile_fail=sum(1 for r in rs if r.get("build_rc")), n_pass=sum(passes(r) for r in rs),
                        best_dev=max(sp) if sp else None, median_dev=statistics.median(sp) if sp else None,
                        frac_slower=(sum(1 for s in sp if s < 1.0) / len(sp)) if sp else None)
    return out


def summarize_llm(rows, samples, base_key=()):
    """Per-model, SAMPLE-level: every model sample counts (invalid/empty/failed samples count as 'no usable edit')."""
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from selfimprove_bench import arms
    meas = {arms.key(r["edits"]): r for r in rows}
    out = {}
    for model, ss in sorted(samples.items()):
        got = [meas.get(arms.key(e)) if e else None for e in ss]
        used = [g for g in got if g and g.get("dev_speedup")]
        sp = [g["dev_speedup"] for g in used]
        out[model] = dict(samples=len(ss), no_edit=sum(1 for e in ss if not e), distinct=len({arms.key(e) for e in ss if e}),
                          n_pass=sum(passes(g) for g in used), distinct_pass=len({arms.key(g["edits"]) for g in used if passes(g)}),
                          best_dev=max(sp) if sp else None, median_dev=statistics.median(sp) if sp else None,
                          frac_slower=(sum(1 for s in sp if s < 1.0) / len(sp)) if sp else None)
    return out


if __name__ == "__main__":
    rows = [json.loads(l) for f in sys.argv[1:] for l in open(f) if l.strip()]
    s = summarize(rows)
    print(f"{'arm':10s} {'n':>3s} {'cfail':>5s} {'pass':>4s} {'best_dev':>8s} {'median':>7s} {'%slower':>8s}")
    for arm, v in s.items():
        if arm.startswith("llm:"):
            continue
        print(f"{arm:10s} {v['n']:3d} {v['compile_fail']:5d} {v['n_pass']:4d} {v['best_dev'] or 0:8.3f} {v['median_dev'] or 0:7.3f} {100 * (v['frac_slower'] or 0):7.0f}%")
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from selfimprove_bench import arms
    ls = summarize_llm(rows, arms.llm_samples(os.path.join(os.path.dirname(__file__), "..", "..", "results")))
    if ls:
        print(f"\n{'model (sample level)':34s} {'n':>3s} {'noedit':>6s} {'dist':>4s} {'pass':>4s} {'dpass':>5s} {'best':>6s} {'median':>7s} {'%slower':>8s}")
        for m, v in sorted(ls.items(), key=lambda kv: -(kv[1]['best_dev'] or 0)):
            print(f"{m:34s} {v['samples']:3d} {v['no_edit']:6d} {v['distinct']:4d} {v['n_pass']:4d} {v['distinct_pass']:5d} {v['best_dev'] or 0:6.3f} {v['median_dev'] or 0:7.3f} {100 * (v['frac_slower'] or 0):7.0f}%")
