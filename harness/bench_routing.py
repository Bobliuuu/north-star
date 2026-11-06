"""Real routing traces: per-layer native top-8 expert ids per token (decode, teacher = North's own greedy output),
plus draft-vs-verify expert overlap for draft modes on the same real tokens."""
import numpy as np
import torch

from common import Jsonl, build_ids, load_workloads
from northstar.refmodel import FULL, DraftMode, RefNorth
from northstar.spec import plain_greedy
from bench_spec import skip_set


@torch.no_grad()
def trace(ref, ids, toks, mode):
    """Forward prompt+gen in one pass per mode; return {layer: [T_gen,8]} for the generated positions."""
    seq = torch.tensor([ids + toks], device="cuda")
    P = len(ids)
    cache = ref.new_cache(1, seq.shape[1] + 2, layers=[i for i in range(ref.cfg.num_hidden_layers) if i not in mode.skip_layers])
    ref.record_experts = []
    ref.prefill(seq[:, : P - 1], cache, mode, logits="none")
    ref.forward(seq[:, P - 1 :], cache, P - 1, mode, logits="none")
    rec, ref.record_experts = ref.record_experts, None
    out = {}
    for layer, idx in rec:  # prefill chunks and the block both append; keep only the last (generation block) per layer
        out[layer] = idx
    return {l: v.cpu().numpy() for l, v in out.items()}


def run(m, tok, n_new=48, splits=("dev", "calib")):
    ref = RefNorth(m)
    j = Jsonl()
    traces = {}
    eos = m.config.eos_token_id
    eos = eos if isinstance(eos, int) else eos[0]
    for t in [r for r in load_workloads() if r["split"] in splits]:
        ids = build_ids(tok, t)
        toks = plain_greedy(ref, ids, n_new, eos_id=eos)["tokens"]
        full = trace(ref, ids, toks, FULL)
        arr = np.stack([full[l] for l in sorted(full)], 0)  # [L_moe, T, 8]
        traces[t["id"]] = dict(split=t["split"], kind=t["kind"], prompt_len=len(ids), tokens=toks,
                               layers=sorted(full), experts=arr.astype(np.int16).tolist())
        T = arr.shape[1]
        # temporal reuse: fraction of token t's top-8 also in token t-1's top-8 (same layer)
        reuse = np.mean([len(set(arr[l, i]) & set(arr[l, i - 1])) / 8 for l in range(arr.shape[0]) for i in range(1, T)])
        # block-union sizes over K consecutive real tokens
        un = {K: float(np.mean([len(set(arr[l, i : i + K].ravel())) for l in range(arr.shape[0]) for i in range(0, T - K + 1, K)])) for K in (1, 2, 4, 8, 16)}
        j.add(phase="route_stats", task=t["id"], split=t["split"], kind=t["kind"], T=T, consecutive_token_overlap=float(reuse), union_experts_by_K=un)
        for name, sk, r in (("r4", 0, 4), ("r2", 0, 2), ("skip8", 8, None), ("skip16", 16, None), ("skip8_r4", 8, 4)):
            mode = DraftMode(skip_set(sk), r)
            d = trace(ref, ids, toks, mode)
            ov, cov, jac = [], [], []
            for l in d:
                a, b = d[l], full[l]
                for i in range(a.shape[0]):
                    sa, sb = set(a[i].tolist()), set(b[i].tolist())
                    ov.append(len(sa & sb) / len(sb))      # fraction of verifier's top-8 loaded by draft
                    jac.append(len(sa & sb) / len(sa | sb))
                # block-level (K=8): fraction of verifier's expert union that draft union already loads
                for i in range(0, a.shape[0] - 7, 8):
                    ua, ub = set(a[i : i + 8].ravel().tolist()), set(b[i : i + 8].ravel().tolist())
                    cov.append(len(ua & ub) / len(ub))
            j.add(phase="draft_vs_verify_overlap", task=t["id"], split=t["split"], draft=name, n_layers_compared=len(d),
                  mean_frac_verifier_top8_covered=float(np.mean(ov)), mean_jaccard=float(np.mean(jac)), mean_block8_union_coverage=float(np.mean(cov)),
                  draft_experts_per_token=(r or 8))
        print(t["id"], j.lines[-1][:200], flush=True)
    return j, traces
