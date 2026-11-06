"""Longer-context draft/verify cost (part A) and batched step cost (part B) on H100, graphed. See longctx_north.py.
emit(row) is called for every row so a crash never loses finished cells."""
import time, traceback
import torch
from common import build_ids, by_split, load_workloads
from northstar.refmodel import RefNorth
from longctx_north import LCNorth, time_graph


def _prompts(tok, ctx, n):
    rows = by_split(load_workloads(), "dev")
    return [build_ids(tok, rows[b % len(rows)], ctx, rotate=b) for b in range(n)]


def _cell(emit, name, ln, B, T, kv, P, reps=30, **kw):
    t0 = time.time()
    try:
        s = ln.make_step(B, T, kv, **kw)
        s.set([ln.tails[b][-T:] for b in range(B)], P - 1)  # real prompt tokens (diverse routing), not a constant token
        med, mn, mx = time_graph(s, n=reps)
        emit(kind="cell", cell=name, B=B, T=T, ctx=P, ms=med, ms_min=mn, ms_max=mx, cap_s=time.time() - t0)
        return s
    except Exception as e:  # noqa
        emit(kind="cell", cell=name, B=B, T=T, ctx=P, err=repr(e)[:300], tb=traceback.format_exc()[-600:])
        torch.cuda.empty_cache()
        return None


@torch.no_grad()
def check_vs_old(m, tok, emit, ctx=2048):
    """Sanity: my prefill+decode (bmm and tri) vs the previously validated GraphNorth path at 2K."""
    from graph_north import GraphNorth
    ref = RefNorth(m)
    ids = _prompts(tok, ctx, 1)[0]
    P = len(ids)
    old = GraphNorth(ref, P + 64, "bmm"); kv0 = old.new_kv(); old.eager_prefill(ids[:-1], kv0)
    so = old.make_step(1, kv0); so.set([ids[-1]], P - 1); so.run(); torch.cuda.synchronize()
    lg_old = so.out.clone() if so.out.dtype != torch.long else None
    tok_old = so.out.clone()
    del so, kv0, old; torch.cuda.empty_cache()
    ln = LCNorth(ref, P + 64); ln.wlo = max(0, P - 1 - 4096 - 16); kv = ln.new_kv(1); ln.prefill(ids[:-1], kv)
    res = {}
    for nm, kw in (("win_bmm", dict(sl="bmm", gl="bmm")), ("padded_bmm", dict(sl="padded", gl="bmm")),
                   ("tri", dict(sl="tri", gl="tri")),
                   ("gather_fixed256", dict(sl="bmm", gl=("gather", "fixed", 256))), ("tri_idx_fixed256", dict(sl="bmm", gl=("tri_idx", "fixed", 256))),
                   ("gather_oracle256", dict(sl="bmm", gl=("gather", "oracle", 256))), ("tri_idx_oracle256", dict(sl="bmm", gl=("tri_idx", "oracle", 256)))):
        try:
            s = ln.make_step(1, 1, kv, **kw); s.set([[ids[-1]]], P - 1); s.run(); torch.cuda.synchronize()
            res[nm] = (s.out[0].clone(), s.out[1].float().clone())
            del s
        except Exception as e:
            emit(kind="check", cell=nm, err=repr(e)[:300], tb=traceback.format_exc()[-500:])
    ref_t, ref_l = res["win_bmm"]
    emit(kind="check", cell="win_bmm_vs_old_graphnorth", same_token=bool((ref_t == tok_old).all().item()), ctx=P)
    for nm, (t_, l_) in res.items():
        emit(kind="check", cell=nm + "_vs_win_bmm", same_token=bool((t_ == ref_t).all().item()),
             max_abs_logit_diff=float((l_ - ref_l).abs().max()), ref_logit_absmax=float(ref_l.abs().max()), ctx=P)
    emit(kind="check", cell="gather_vs_triidx_fixed", max_abs_logit_diff=float((res["gather_fixed256"][1] - res["tri_idx_fixed256"][1]).abs().max())
         if "gather_fixed256" in res and "tri_idx_fixed256" in res else None)
    emit(kind="check", cell="gather_vs_triidx_oracle", max_abs_logit_diff=float((res["gather_oracle256"][1] - res["tri_idx_oracle256"][1]).abs().max())
         if "gather_oracle256" in res and "tri_idx_oracle256" in res else None)
    del kv, ln; torch.cuda.empty_cache()


@torch.no_grad()
def run_long(m, tok, emit, ctxs=(8192, 32768, 131072)):
    ref = RefNorth(m)
    for ctx in ctxs:
        t0 = time.time()
        ids = _prompts(tok, ctx, 1)[0]; P = len(ids)
        ln = LCNorth(ref, P + 64); ln.wlo = max(0, P - 1 - 4096 - 16); ln.tails = [ids[-16:]]
        kv = ln.new_kv(1)
        ln.prefill(ids[:-1], kv, chunk=1024 if ctx <= 32768 else 512)
        emit(kind="prefill", ctx=P, seconds=time.time() - t0, mem_gb=torch.cuda.max_memory_allocated() / 1e9)
        gs = [("pad", dict(sl="padded", gl="bmm")), ("win", dict(sl="bmm", gl="bmm")), ("tri", dict(sl="tri", gl="tri")),
              ("tri_swbmm", dict(sl="bmm", gl="tri"))]
        sp = [("gather_fixed256", dict(sl="bmm", gl=("gather", "fixed", 256))), ("gather_oracle256", dict(sl="bmm", gl=("gather", "oracle", 256))),
              ("triidx_fixed256", dict(sl="bmm", gl=("tri_idx", "fixed", 256))), ("triidx_oracle256", dict(sl="bmm", gl=("tri_idx", "oracle", 256))),
              ("triidx_fixed1024", dict(sl="bmm", gl=("tri_idx", "fixed", 1024))),
              ("triidx_fixed256_swtri", dict(sl="tri", gl=("tri_idx", "fixed", 256)))]
        for r in (None, 4):
            for nm, kw in gs + sp:
                s = _cell(emit, f"T1_r{r}_{nm}", ln, 1, 1, kv, P, r=r, **kw)
                del s
        # no-attention floor (experts/router/lm only) and verify blocks
        for r in (None, 4):
            s = _cell(emit, f"T1_r{r}_noattn", ln, 1, 1, kv, P, r=r, parts=frozenset({"router", "experts", "lm"})); del s
        for T in (3, 5, 9):
            for nm, kw in (("pad", gs[0][1]), ("win", gs[1][1]), ("tri", gs[2][1])):
                s = _cell(emit, f"verify_K{T}_{nm}", ln, 1, T, kv, P, **kw); del s
            s = _cell(emit, f"verify_K{T}_noattn", ln, 1, T, kv, P, parts=frozenset({"router", "experts", "lm"})); del s
        del kv, ln; torch.cuda.empty_cache()


@torch.no_grad()
def run_batch(m, tok, emit, plan=((2048, (1, 2, 4, 8)), (8192, (1, 2, 4, 8)), (32768, (1, 2, 4)))):
    ref = RefNorth(m)
    for ctx, Bs in plan:
        t0 = time.time()
        Bmax = max(Bs)
        idl = _prompts(tok, ctx, Bmax); P = len(idl[0])
        ln = LCNorth(ref, P + 64); ln.wlo = max(0, P - 1 - 4096 - 16)
        kv = ln.new_kv(Bmax); ln.tails = [x[-16:] for x in idl]
        for b in range(Bmax):
            ln.prefill(idl[b][:-1], kv, b=b, chunk=1024)
        emit(kind="prefill", ctx=P, B=Bmax, seconds=time.time() - t0, mem_gb=torch.cuda.max_memory_allocated() / 1e9)
        for B in Bs:
            for an, kw in (("win", dict(sl="bmm", gl="bmm")), ("tri", dict(sl="tri", gl="tri"))):
                for r, tag in ((None, "full"), (4, "draft_r4")):
                    s = _cell(emit, f"T1_{tag}_{an}", ln, B, 1, kv, P, r=r, **kw); del s
                for T in (3, 5, 9):
                    s = _cell(emit, f"verify_K{T}_{an}", ln, B, T, kv, P, **kw); del s
            s = _cell(emit, "T1_noattn", ln, B, 1, kv, P, parts=frozenset({"router", "experts", "lm"})); del s
            s = _cell(emit, "verify_K5_noattn", ln, B, 5, kv, P, parts=frozenset({"router", "experts", "lm"})); del s
        del kv, ln; torch.cuda.empty_cache()
