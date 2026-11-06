"""python -m suffixspec.run_eval results/north_greedy_dump.jsonl [--out results/suffixspec_eval.json]"""
import argparse, itertools, json, sys
from tage.baselines import PromptLookup, SuffixFreq, SuffixLast
from .evalsim import evaluate

PROPOSERS = {
    "pld(3..2)": lambda: PromptLookup(3, 2),
    "pld(4..2)": lambda: PromptLookup(4, 2),
    "suffix-last": lambda: SuffixLast(1),
    "suffix-last(min3)": lambda: SuffixLast(3),
    "suffix-freq": lambda: SuffixFreq(1),
}


def load(path):
    return [json.loads(l) for l in open(path) if l.strip() and not l.startswith("#")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump"); ap.add_argument("--out")
    a = ap.parse_args()
    recs = load(a.dump)
    res = []
    print(f"{len(recs)} records, {sum(len(r['out_ids']) for r in recs)} output tokens")
    print(f"{'proposer':18s} kmax min_d tok/step draft% prec  speedup(pooled) (mean) (min)")
    for name, mk in PROPOSERS.items():
        for kmax, md in itertools.product((4, 8, 16), (1, 2, 3, 4)):
            rows, agg = evaluate(mk, recs, kmax=kmax, min_draft=md)
            res.append(dict(proposer=name, kmax=kmax, min_draft=md, **agg))
            print(f"{name:18s} {kmax:4d} {md:5d} {agg['tokens_per_step']:8.3f} {agg['draft_rate']*100:5.1f} "
                  f"{agg['precision']:.3f} {agg['speedup_pooled']:8.3f} {agg['speedup_mean']:7.3f} {agg['speedup_min']:6.3f}")
    best = max(res, key=lambda r: r["speedup_pooled"])
    print("BEST", json.dumps(best))
    if a.out:
        json.dump(dict(results=res, best=best), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
