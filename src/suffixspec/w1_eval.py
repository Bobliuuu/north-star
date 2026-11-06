"""Evaluate context drafters on NORTH's sampled W1 (OpenHands) generations.
  python -m suffixspec.w1_eval results/w1_gen_g1.jsonl results/w1_prompts.npz [--out results/w1_suffix_eval_g1.json]
Modes: cold = history is that request's prompt + own output only; warm = proposer tables carried across requests in
file order (inference-experiment: warm inflates acceptance; only first point of a fresh boot is clean).
Speedups are INFERRED from named step-cost models (evalsim.COST_MODELS); draft CPU latency not modelled."""
import argparse, itertools, json
from collections import Counter

import numpy as np
from tage.baselines import PromptLookup, SuffixLast

from .evalsim import COST_MODELS, run_one, speedup_from

PROPOSERS = {
    "pld(3..2)": lambda: PromptLookup(3, 2),
    "suffix-last": lambda: SuffixLast(1),
    "suffix-last(min3)": lambda: SuffixLast(3),
    "suffix-last(min6)": lambda: SuffixLast(6),
}


def load(gen_path, npz_path):
    z = np.load(npz_path)
    recs = [json.loads(l) for l in open(gen_path) if l.strip()]
    for r in recs:
        r["prompt_ids"] = z[r["id"]].tolist()
    return recs


def agg(rows):
    T = sum(x["tokens"] for x in rows); S = sum(x["steps"] for x in rows)
    kc = Counter()
    for x in rows:
        kc.update(x["kcount"])
    d = sum(x["drafted"] for x in rows)
    out = dict(n=len(rows), tokens=T, steps=S, tokens_per_step=T / S, precision=sum(x["accepted"] for x in rows) / max(1, d),
               draft_rate=sum(x["verified_steps"] for x in rows) / S)
    for m in COST_MODELS:
        out[f"speedup_{m}"] = speedup_from(kc, T, m)
    return out


def run_grid(recs, kmaxs=(8, 16, 32), mins=(1, 3, 8)):
    res = []
    for name, mk in PROPOSERS.items():
        for mode in ("cold", "warm"):
            for kmax, md in itertools.product(kmaxs, mins):
                p = mk()
                rows = [run_one(p, r["prompt_ids"], r["out_ids"], kmax=kmax, min_draft=md, keep_state=(mode == "warm"))
                        | dict(id=r["id"], capped=r.get("capped", False)) for r in recs]
                a = agg(rows)
                a.update(proposer=name, mode=mode, kmax=kmax, min_draft=md)
                loops = [x for x in rows if x["capped"]]; rest = [x for x in rows if not x["capped"]]
                if loops:
                    a["loop"] = agg(loops)
                if rest:
                    a["nonloop"] = agg(rest)
                q = max(1, len(rows) // 3)  # first / last third of the stream (warm-session drift)
                a["first_third_tps"] = agg(rows[:q])["tokens_per_step"]; a["last_third_tps"] = agg(rows[-q:])["tokens_per_step"]
                res.append(a)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("gen"); ap.add_argument("npz"); ap.add_argument("--out")
    a = ap.parse_args()
    recs = load(a.gen, a.npz)
    n_out = sum(len(r["out_ids"]) for r in recs); capped = [r for r in recs if r.get("capped")]
    share = sum(len(r["out_ids"]) for r in capped) / max(1, n_out)
    print(f"{len(recs)} requests, {n_out} output tokens, capped(loop) {len(capped)} = {len(capped)/len(recs):.1%} of requests, "
          f"{share:.1%} of output tokens")
    res = run_grid(recs)
    print(f"{'proposer':18s} mode kmax min tok/step prec  su(mega) su(flat2) su(flat5) | first/last third tps")
    for r in res:
        print(f"{r['proposer']:18s} {r['mode']:4s} {r['kmax']:4d} {r['min_draft']:3d} {r['tokens_per_step']:7.3f} {r['precision']:.3f} "
              f"{r['speedup_megakernel']:8.3f} {r['speedup_flat2']:8.3f} {r['speedup_flat5']:8.3f} | {r['first_third_tps']:.2f}/{r['last_third_tps']:.2f}")
    best = {m: max((r for r in res if r["mode"] == "cold"), key=lambda r: r[f"speedup_{m}"]) for m in COST_MODELS}
    print("BEST cold per cost model:", {m: (b["proposer"], b["kmax"], b["min_draft"], round(b[f"speedup_{m}"], 3)) for m, b in best.items()})
    if a.out:
        json.dump(dict(n_requests=len(recs), n_out_tokens=n_out, capped=len(capped), capped_token_share=share, grid=res), open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
