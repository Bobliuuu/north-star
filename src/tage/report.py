"""python -m tage.report results/tage/*.json -> markdown tables (one per mode)."""
import json, sys

def main(paths):
    data = {}
    for p in paths:
        for c, v in json.load(open(p)).items():
            data[c] = v   # later files override same corpus
    methods = None
    for mode in ("warm-session", "cold-per-file"):
        print(f"\n### {mode}\n")
        print("| corpus (files/tokens) | method | tok/step | draft% | first-tok hit | token precision | speedup a=.05 | a=.15 | a=.30 |")
        print("|---|---|---|---|---|---|---|---|---|")
        for c, v in data.items():
            for m, r in v["modes"][mode].items():
                print(f"| {c} ({v['docs']}/{v['tokens']}) | {m} | {r['tokens_per_step']:.3f} | {100*r['draft_rate']:.1f} | {r['proposal_hit_rate_first']:.3f} | {r['token_precision']:.3f} | {r['speedup_alpha0.05']:.2f} | {r['speedup_alpha0.15']:.2f} | {r['speedup_alpha0.3']:.2f} |")
    print("\n### accepted-length distribution (fraction of steps; accepted draft tokens per step), warm-session\n")
    for c, v in data.items():
        for m in ("pld(n=3..2)", "suffix-last", "tage k=8 theta=0.3", "tage k=8 theta=0.5"):
            d = v["modes"]["warm-session"][m]["accept_len_dist"]
            print(f"- {c} / {m}: " + ", ".join(f"{k}:{x}" for k, x in d.items()))
    print("\n### two-ahead outcome coverage (TAGE, warm-session)\n")
    print("| corpus | config | P(full accept) | P(bonus predicted given full) | P(alt hit given reject) | outcome coverage top1 | top1+alt |")
    print("|---|---|---|---|---|---|---|")
    for c, v in data.items():
        for m in ("tage k=8 theta=0.3", "tage k=8 theta=0.5"):
            t = v["modes"]["warm-session"][m].get("twoahead")
            if t:
                print(f"| {c} | {m} | {t['P_full_accept']:.3f} | {t['P_bonus_pred_given_full']:.3f} | {t['P_alt_hit_given_reject']:.3f} | {t['outcome_coverage_top1_only']:.3f} | {t['outcome_coverage_top1_plus_alt']:.3f} |")

if __name__ == "__main__":
    main(sys.argv[1:])
