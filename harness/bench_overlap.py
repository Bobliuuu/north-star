"""Phase (3): REAL cross-token overlap. Run native + draft-mode traces on North (eager RefNorth, teacher = North greedy) and
compute, per layer, what fraction of FUTURE draft tokens' experts already lie in the verify block's expert union.
Also dumps the traces so the replay microbench (H100) can be fed real routing."""
import numpy as np
import torch

from common import Jsonl, build_ids, load_workloads
from northstar.refmodel import FULL, DraftMode, RefNorth
from northstar.spec import plain_greedy
from bench_spec import skip_set
from bench_routing import trace
from overlap import analyze

MODES = (("r4", 0, 4), ("skip8", 8, None), ("r2", 0, 2), ("skip8_r4", 8, 4))
GRID = ((2, 3), (4, 5), (8, 9))  # (Td=gamma, K=gamma+1)


@torch.no_grad()
def run(m, tok, n_new=64, splits=("dev", "calib", "heldout")):
    ref = RefNorth(m)
    eos = m.config.eos_token_id
    eos = eos if isinstance(eos, int) else eos[0]
    j = Jsonl()
    dump = {}
    for t in [r for r in load_workloads() if r["split"] in splits]:
        ids = build_ids(tok, t)
        toks = plain_greedy(ref, ids, n_new, eos_id=eos)["tokens"]
        if len(toks) < 20:
            continue
        full = trace(ref, ids, toks, FULL)
        layers = sorted(full)
        arr = np.stack([full[l] for l in layers], 0).astype(np.int16)
        ent = dict(split=t["split"], kind=t["kind"], prompt_len=len(ids), n=len(toks), layers=layers, full=arr.tolist(), modes={})
        for name, sk, r in MODES:
            d = trace(ref, ids, toks, DraftMode(skip_set(sk), r))
            dl = sorted(d)
            ent["modes"][name] = dict(layers=dl, r=(r or 8), arr=np.stack([d[l] for l in dl], 0).astype(np.int16).tolist())
        dump[t["id"]] = ent
        print("traced", t["id"], len(toks), flush=True)
        torch.cuda.empty_cache()
    return j, dump
