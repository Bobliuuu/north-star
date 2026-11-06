"""Phase 1 on real North: plain greedy vs greedy self-speculation (same weights, draft = layer skip and/or expert top-r).
Greedy verification must reproduce plain greedy; divergences are measured and classified, never hidden."""
import statistics as st
import time

import numpy as np
import torch

from common import Jsonl, build_ids, load_workloads
from northstar.refmodel import FULL, DraftMode, RefNorth
from northstar.spec import SelfSpec, plain_greedy

NEAR_TIE = 0.25  # bf16 logit units; margins below this at the divergence point are classed as numerical near-ties


def skip_set(k, lo=4, hi=46):
    return frozenset(int(x) for x in np.unique(np.linspace(lo, hi, k).round())) if k else frozenset()


CONFIGS = [  # (name, skip_k, expert_r, gamma)
    ("skip8_g4", 8, None, 4), ("skip16_g4", 16, None, 4),
    ("r4_g4", 0, 4, 4), ("r2_g4", 0, 2, 4), ("skip8_r4_g4", 8, 4, 4),
    ("r4_g2", 0, 4, 2), ("r4_g8", 0, 4, 8),
]


@torch.no_grad()
def run(m, tok, n_new=64, ntasks=8, quick=False):
    ref = RefNorth(m)
    j = Jsonl()
    rows = load_workloads()
    # spread across kinds/splits deterministically
    rows = sorted(rows, key=lambda r: (r["split"] != "heldout", r["id"]))
    tasks = [r for i, r in enumerate(rows)][: (2 if quick else ntasks)]
    eos = m.config.eos_token_id
    eos = eos if isinstance(eos, int) else eos[0]
    cfgs = CONFIGS[:2] if quick else CONFIGS
    plain = {}
    for t in tasks:
        ids = build_ids(tok, t)
        plain[t["id"]] = (ids, plain_greedy(ref, ids, n_new, eos_id=eos))
        pg = plain[t["id"]][1]
        # HF greedy cross-check of the reference executor
        with torch.no_grad():
            hf = m.generate(torch.tensor([ids], device="cuda"), max_new_tokens=n_new, do_sample=False)[0, len(ids):].tolist()
        hf = hf[: hf.index(eos) + 1] if eos in hf else hf
        k = next((i for i, (a, b) in enumerate(zip(hf, pg["tokens"])) if a != b), min(len(hf), len(pg["tokens"])))
        j.add(phase="1_refnorth_vs_hf", task=t["id"], split=t["split"], prompt_len=len(ids), n=len(pg["tokens"]), first_div=k,
              identical=bool(hf == pg["tokens"]), margin_at_div=(pg["margins"][k] if k < len(pg["margins"]) else None),
              plain_step_ms_median=1e3 * st.median(pg["step_s"][4:] or pg["step_s"]))
    for name, sk, r, g in cfgs:
        mode = DraftMode(skip_set(sk), r)
        for t in tasks:
            ids, pg = plain[t["id"]]
            res = SelfSpec(ref, mode, g).generate(ids, n_new, eos_id=eos)
            toks = res["tokens"]
            base = pg["tokens"][: len(toks)] if len(toks) > len(pg["tokens"]) else pg["tokens"]
            k = next((i for i, (a, b) in enumerate(zip(toks, pg["tokens"])) if a != b), min(len(toks), len(pg["tokens"])))
            identical = toks == pg["tokens"]
            margin = pg["margins"][k] if k < len(pg["margins"]) and not identical else None
            R = res["rounds"]
            acc = [s.accepted for s in R]
            plain_s = sum(pg["step_s"])
            ntok = len(pg["tokens"])
            per_pos = [sum(a > i for a in acc) / len(acc) for i in range(g)]
            j.add(phase="1_selfspec", config=name, skip_layers=sorted(mode.skip_layers), n_skipped=len(mode.skip_layers), expert_r=r, gamma=g,
                  task=t["id"], kind=t["kind"], split=t["split"], prompt_len=len(ids), n_committed=len(toks), n_plain=ntok,
                  identical=identical, first_div=None if identical else k, div_margin=margin,
                  div_near_tie=None if identical else (margin is not None and margin < NEAR_TIE),
                  rounds=len(R), mean_accepted=st.mean(acc), mean_committed_per_round=len(toks) / len(R),
                  accept_rate_per_pos=per_pos,
                  t_catchup_s=sum(s.t_catchup for s in R), t_draft_s=sum(s.t_draft for s in R),
                  t_verify_s=sum(s.t_verify for s in R), t_rollback_s=sum(s.t_rollback for s in R),
                  spec_gen_s=res["gen_s"], plain_gen_s=plain_s, spec_tok_per_s=len(toks) / res["gen_s"], plain_tok_per_s=ntok / plain_s,
                  speedup_wall=(len(toks) / res["gen_s"]) / (ntok / plain_s),
                  verifier_prefill_s=res["verifier_prefill_s"], draft_prefill_s=res["draft_prefill_s"],
                  draft_overhead_frac=(sum(s.t_catchup + s.t_draft for s in R)) / res["gen_s"])
        print(name, j.lines[-1][:400], flush=True)
    if not quick:  # one long-context (8K) row for the two cheapest configs
        t = next(r for r in rows if r["kind"] == "code")
        ids = build_ids(tok, t, 8192)
        pg = plain_greedy(ref, ids, 48, eos_id=eos)
        for name, sk, r, g in (("r4_g4", 0, 4, 4), ("skip8_g4", 8, None, 4)):
            res = SelfSpec(ref, DraftMode(skip_set(sk), r), g).generate(ids, 48, eos_id=eos)
            R = res["rounds"]
            k = next((i for i, (a, b) in enumerate(zip(res["tokens"], pg["tokens"])) if a != b), None)
            j.add(phase="1_selfspec_8k", config=name, prompt_len=len(ids), identical=res["tokens"] == pg["tokens"], first_div=k,
                  mean_accepted=st.mean(s.accepted for s in R), spec_tok_per_s=len(res["tokens"]) / res["gen_s"],
                  plain_tok_per_s=len(pg["tokens"]) / sum(pg["step_s"]), t_draft_s=sum(s.t_catchup + s.t_draft for s in R), t_verify_s=sum(s.t_verify for s in R))
    return j
