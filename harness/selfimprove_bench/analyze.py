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
        by[r["arm"]].append(r)
    out = {}
    for arm, rs in sorted(by.items()):
        sp = [r["dev_speedup"] for r in rs if r.get("dev_speedup")]
        out[arm] = dict(n=len(rs), compile_fail=sum(1 for r in rs if r.get("build_rc")), n_pass=sum(passes(r) for r in rs),
                        best_dev=max(sp) if sp else None, median_dev=statistics.median(sp) if sp else None,
                        frac_slower=(sum(1 for s in sp if s < 1.0) / len(sp)) if sp else None)
    return out


if __name__ == "__main__":
    rows = [json.loads(l) for f in sys.argv[1:] for l in open(f) if l.strip()]
    s = summarize(rows)
    print(f"{'arm':10s} {'n':>3s} {'cfail':>5s} {'pass':>4s} {'best_dev':>8s} {'median':>7s} {'%slower':>8s}")
    for arm, v in s.items():
        print(f"{arm:10s} {v['n']:3d} {v['compile_fail']:5d} {v['n_pass']:4d} {v['best_dev'] or 0:8.3f} {v['median_dev'] or 0:7.3f} {100 * (v['frac_slower'] or 0):7.0f}%")
