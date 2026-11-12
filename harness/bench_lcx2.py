"""Long-context exact attention sweep for exact v2 (H100): graph step ms at T=1 / T=13 and T-invariance per variant at 32K / 128K.
Variants change only the exact attention split (absolute key chunk, keys per inner step, warps) and the query-row padding (mp_min);
all are position-absolute / T-independent by construction, invariance is re-checked here (rows 0, 5, 12 of a T=13 block == T=1 steps)."""
import json, time, traceback
import torch
import fast_kernels_x2 as f2
from common import Jsonl, build_ids, load_workloads
from northstar.refmodel import RefNorth
from graph_north import time_graph
from fast_north import FastNorth
from longctx_north import LCNorth
from bench_pld import custom_tasks, pick_ctx_subset

VARIANTS = {
    "v1": None,
    "x2": {},
    "x2_mp64": dict(mp_min=64),
    "x2_c512": dict(attn=dict(chunk=512)),
    "x2_c1024": dict(attn=dict(chunk=1024)),
    "x2_c2048": dict(attn=dict(chunk=2048)),
    "x2_c1024_bn128": dict(attn=dict(chunk=1024, bn=128)),
    "x2_mp64_c1024": dict(mp_min=64, attn=dict(chunk=1024)),
    "x2_mp64_c2048": dict(mp_min=64, attn=dict(chunk=2048)),
}


@torch.no_grad()
def run(m, tok, ctxs=(32768, 131072), variants=None):
    j = Jsonl()
    variants = variants or list(VARIANTS)
    ref = RefNorth(m)
    tasks = [r for r in load_workloads() if r["split"] in ("dev", "heldout")] + custom_tasks()
    t = pick_ctx_subset(tasks)[0]
    for ctx in ctxs:
        try:
            Lmax = ctx + 64
            X = FastNorth(ref, Lmax, exact=True); ln = LCNorth(ref, Lmax)
            kv = X.new_kv()
            ids = build_ids(tok, t, ctx); P = len(ids)
            ln.prefill(ids[:-1], kv, chunk=1024 if ctx <= 32768 else 512)
            del ln; torch.cuda.empty_cache()
            blk = ids[-13:]
            for name in variants:
                v = VARIANTS[name]
                try:
                    if v is None:
                        X.x2 = False; X.kcfg_blk = None; X.mp_min = None; X.attn_kw = {}
                    else:
                        X.set_x2(**dict(f2.recommended(), **v))
                    s1 = X.make_step(1, kv, logits="all", argmax=False); s13 = X.make_step(13, kv, logits="all", argmax=False)
                    s1.set(ids[-1:], P - 1); t1 = time_graph(s1, n=20, warm=3)[0]
                    s13.set(blk, P - 1); t13 = time_graph(s13, n=10, warm=2)[0]
                    s13.set(blk, P - 1); s13.run(); L13 = s13.out.clone().reshape(13, -1)
                    inv = True
                    for r in (0, 5, 12):
                        for q in range(r + 1):
                            s1.set([blk[q]], P - 1 + q); s1.run()
                        inv &= bool(torch.equal(s1.out.reshape(-1), L13[r]))
                    j.add(phase="lc", ctx=P, variant=name, ms_T1=t1, ms_T13=t13, invariant=inv, mem_gb=torch.cuda.max_memory_allocated() / 1e9)
                    print("LC", P, name, "T1 %.3f T13 %.3f inv %s" % (t1, t13, inv), flush=True)
                    del s1, s13
                    X._x = {}  # drop this variant's attention workspaces (its graphs are gone); 128K leaves little headroom
                except Exception:
                    j.add(phase="error", ctx=ctx, variant=name, err=traceback.format_exc()[-1000:]); print("ERR", ctx, name, traceback.format_exc()[-800:], flush=True)
                torch.cuda.empty_cache()
            del X, kv; torch.cuda.empty_cache()
        except Exception:
            j.add(phase="fatal", ctx=ctx, err=traceback.format_exc()[-1500:]); print("FATAL", ctx, traceback.format_exc()[-1200:], flush=True)
    return j
