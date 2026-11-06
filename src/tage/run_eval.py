"""python -m tage.run_eval [--max-tokens N] -> JSON + table on stdout. Oracle-verifier simulation."""
import argparse, json, sys, time
from .corpus import CORPORA, load_docs
from .predictor import TagePredictor, TageConfig
from .baselines import PromptLookup, SuffixLast, SuffixFreq, SuffixCal
from .sim import simulate


def make_all():
    return {
        "pld(n=3..1)": (lambda: PromptLookup(3, 1), dict(kmax=8, theta=0.0)),
        "pld(n=3..2)": (lambda: PromptLookup(3, 2), dict(kmax=8, theta=0.0)),
        "suffix-last": (lambda: SuffixLast(1), dict(kmax=8, theta=0.0)),
        "suffix-last(min3)": (lambda: SuffixLast(3), dict(kmax=8, theta=0.0)),
        "suffix-freq": (lambda: SuffixFreq(1), dict(kmax=8, theta=0.0)),
        "suffix+gate th=0.3": (lambda: SuffixCal(), dict(kmax=8, theta=0.3)),
        "suffix+gate th=0.5": (lambda: SuffixCal(), dict(kmax=8, theta=0.5)),
        "tage k=8 fixed": (lambda: TagePredictor(), dict(kmax=8, theta=0.0)),
        "tage k=8 theta=0.3": (lambda: TagePredictor(), dict(kmax=8, theta=0.3)),
        "tage k=8 theta=0.5": (lambda: TagePredictor(), dict(kmax=8, theta=0.5)),
        "tage k=16 theta=0.5": (lambda: TagePredictor(), dict(kmax=16, theta=0.5)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=100_000)
    ap.add_argument("--corpora", default=",".join(CORPORA))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    results = {}
    for cname in a.corpora.split(","):
        docs = load_docs(CORPORA[cname], max_tokens=a.max_tokens)
        ntok = sum(map(len, docs))
        results[cname] = {"docs": len(docs), "tokens": ntok, "modes": {}}
        print(f"\n== corpus {cname}: {len(docs)} files, {ntok} Qwen3-BPE tokens", flush=True)
        for mode, keep in (("warm-session", True), ("cold-per-file", False)):
            rows = {}
            print(f"-- {mode}")
            print(f"{'method':22s} tok/step draft% first-hit tok-prec  a=.05  a=.15  a=.30")
            for name, (mk, kw) in make_all().items():
                t = time.time()
                r = simulate(mk(), docs, keep_state=keep, two_ahead=name.startswith('tage'), **kw).summary()
                rows[name] = r
                print(f"{name:22s} {r['tokens_per_step']:7.3f} {r['draft_rate']*100:5.1f} {r['proposal_hit_rate_first']:9.3f} {r['token_precision']:8.3f} "
                      f"{r['speedup_alpha0.05']:5.2f} {r['speedup_alpha0.15']:6.2f} {r['speedup_alpha0.3']:6.2f}", flush=True)
                if 'twoahead' in r:
                    print('     two-ahead:', {k: round(v, 3) for k, v in r['twoahead'].items()}, flush=True)
            results[cname]["modes"][mode] = rows
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
