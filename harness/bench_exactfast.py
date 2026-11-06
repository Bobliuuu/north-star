"""exact-fast: profile + cost + bitwise checks of the exact-mode FastNorth executor (H100). Sections: p profile, c cost, i invariance/bitwise-vs-baseline."""
import collections, traceback
import torch

from common import Jsonl, build_ids, by_split, load_workloads
from northstar.refmodel import RefNorth
from graph_north import GraphNorth, time_graph
from fast_north import FastNorth

TMAX = 13

# name -> FastNorth kwargs (xopts, xcfg). base = original exact path.
PRESETS = {
    "base": dict(xopts=frozenset()),
    "pdl": dict(xopts=frozenset({"pdl"})),
    "rg64": dict(xopts=frozenset({"rg"}), xcfg=dict(attn=dict(BN=64, nw=4))),
    "rg128": dict(xopts=frozenset({"rg"}), xcfg=dict(attn=dict(BN=128, nw=4))),
    "rg128w2": dict(xopts=frozenset({"rg"}), xcfg=dict(attn=dict(BN=128, nw=2))),
    "rg128pdl": dict(xopts=frozenset({"rg", "pdl"}), xcfg=dict(attn=dict(BN=128, nw=4))),
}


def _setup(ref, tok, ctx, **kw):
    code = [r for r in by_split(load_workloads(), "dev") if r["kind"] == "code"]
    ids = build_ids(tok, code[0], ctx); P = len(ids)
    gb = GraphNorth(ref, P + 64, "bmm"); kv = gb.new_kv(); gb.eager_prefill(ids[:-1], kv)
    return ids, P, gb, kv


def profile_step(step, j, tag, iters=3):
    from torch.profiler import profile as tprof, ProfilerActivity
    from torch.autograd import DeviceType
    for _ in range(3):
        step.g.replay()
    torch.cuda.synchronize()
    with tprof(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            step.g.replay()
        torch.cuda.synchronize()
    rows = []
    for ka in prof.key_averages():
        if getattr(ka, "device_type", None) != DeviceType.CUDA:
            continue
        dt = getattr(ka, "self_device_time_total", None)
        if dt is None:
            dt = getattr(ka, "self_cuda_time_total", 0)
        if dt and dt > 0:
            rows.append((ka.key, ka.count / iters, dt / iters / 1e3))
    rows.sort(key=lambda r: -r[2])
    tot = sum(r[2] for r in rows); nk = sum(r[1] for r in rows)
    j.add(phase="profile_summary", tag=tag, kernels=nk, sum_kernel_ms=tot)
    print("PROFILE", tag, "kernels", nk, "sum ms", round(tot, 3), flush=True)
    for name, c, ms in rows[:30]:
        j.add(phase="profile_kernel", tag=tag, kernel=name[:100], count=c, ms=ms, pct=100 * ms / tot)
        print("   %-100s n=%5.0f ms=%.3f (%.1f%%)" % (name[:100], c, ms, 100 * ms / tot), flush=True)


@torch.no_grad()
def profile(ref, tok, j, ctx=2048, xopts=None, xcfg=None):
    ids, P, gb, kv = _setup(ref, tok, ctx)
    kw = {} if xopts is None else dict(xopts=xopts, xcfg=xcfg)
    X = FastNorth(ref, P + 64, exact=True, **kw)
    for T in (1, 13):
        s = X.make_step(T, kv, logits="all") if T > 1 else X.make_step(1, kv)
        s.set(ids[-T:], P - T)
        profile_step(s, j, "exact_T%d_ctx%d" % (T, P)); del s
    O = FastNorth(ref, P + 64)
    s = O.make_step(1, kv); s.set(ids[-1:], P - 1)
    profile_step(s, j, "nonexact_T1_ctx%d" % P)


@torch.no_grad()
def cost(ref, tok, j, ctxs=(2048, 8192), configs=None):
    configs = configs or [(n, PRESETS[n]) for n in ("base",)]
    for ctx in ctxs:
        ids, P, gb, kv = _setup(ref, tok, ctx)
        for name, kw in configs:
            X = FastNorth(ref, P + 64, exact=True, **kw)
            for T in (1, 4, 9, 13):
                try:
                    s = X.make_step(T, kv, logits="all") if T > 1 else X.make_step(1, kv)
                    s.set(ids[-T:], P - T)
                    med, mn, mx = time_graph(s, n=60)
                    j.add(phase="cost", ctx=P, cfg=name, T=T, ms=med, ms_min=mn, ms_max=mx); print("COST", P, name, T, round(med, 3), round(mn, 3), flush=True)
                    del s
                except Exception:
                    j.add(phase="cost_error", ctx=P, cfg=name, T=T, err=traceback.format_exc()[-800:]); print("COSTERR", traceback.format_exc()[-600:], flush=True)
            del X
        O = FastNorth(ref, P + 64)
        s = O.make_step(1, kv); s.set(ids[-1:], P - 1)
        med, mn, mx = time_graph(s, n=60); j.add(phase="cost", ctx=P, cfg="nonexact", T=1, ms=med, ms_min=mn); print("COST", P, "nonexact", 1, round(med, 3), flush=True)
        del O, s, kv; torch.cuda.empty_cache()


@torch.no_grad()
def invariance(m, ref, tok, j, kwnew=None, name=""):
    """new exact executor: (a) T=1 bitwise == rows of T=2/5/13 blocks (eager+graph), (b) logits bitwise == BASELINE exact executor (xopts empty)."""
    rows = by_split(load_workloads(), "dev")
    code = [r for r in rows if r["kind"] == "code"]
    cells = [(r["id"], build_ids(tok, r)) for r in rows] + [("code_ctx2048", build_ids(tok, code[0], 2048)), ("code_ctx8192", build_ids(tok, code[0], 8192))]
    dev = ref.device
    for tid, ids in cells:
        try:
            P = len(ids)
            gb = GraphNorth(ref, P + 64, "bmm"); X = FastNorth(ref, P + 64, exact=True, **(kwnew or {})); B = FastNorth(ref, P + 64, exact=True, xopts=frozenset())
            kv = X.new_kv(); gb.eager_prefill(ids[:-1], kv)
            toks = [ids[-1]]; l1 = []; lb = []
            for i in range(TMAX):
                X.start.fill_(P - 1 + i); B.start.fill_(P - 1 + i)
                lg = X.forward(torch.tensor([[toks[-1]]], device=dev), kv, logits="last")[0, 0]
                lgb = B.forward(torch.tensor([[toks[-1]]], device=dev), kv, logits="last")[0, 0]
                l1.append(lg.clone()); lb.append(lgb.clone()); toks.append(int(lg.argmax()))
            blk = torch.tensor([toks[:TMAX]], device=dev)
            res = {}
            for T in (2, 5, 13):
                X.start.fill_(P - 1); lbk = X.forward(blk[:, :T], kv, logits="all")[0]
                res[T] = all(torch.equal(lbk[r], l1[r]) for r in range(T))
            st = X.make_step(13, kv, logits="all", argmax=False); st.set(toks[:13], P - 1)
            lgr = st.run()[0].clone()
            graph_eq = all(torch.equal(lgr[r], l1[r]) for r in range(13))
            st1 = X.make_step(1, kv, argmax=False); st1.set(toks[:1], P - 1)
            graph1_eq = torch.equal(st1.run()[0, 0], l1[0])
            eq_base = sum(int(torch.equal(l1[r], lb[r])) for r in range(TMAX))
            j.add(phase="invariance", cfg=name, task=tid, P=P, block_T2=res[2], block_T5=res[5], block_T13=res[13], graph13_eq_eager_t1=graph_eq, graph1_eq_eager=graph1_eq, rows_bitwise_eq_baseline=eq_base, rows=TMAX)
            print("INV", name, tid, res, graph_eq, graph1_eq, "eq_baseline %d/%d" % (eq_base, TMAX), flush=True)
            del gb, X, B, kv, st, st1; torch.cuda.empty_cache()
        except Exception:
            j.add(phase="inv_error", task=tid, err=traceback.format_exc()[-900:]); print("INV ERR", tid, traceback.format_exc()[-700:], flush=True)


def run(m, tok, sections="pci", tag="", presets="base", ctxs=(2048, 8192)):
    ref = RefNorth(m); j = Jsonl()
    cfgs = [(n, PRESETS[n]) for n in presets.split(",")]
    for sec, fn in (("p", lambda: profile(ref, tok, j, xopts=cfgs[-1][1].get("xopts"), xcfg=cfgs[-1][1].get("xcfg"))), ("c", lambda: cost(ref, tok, j, ctxs=ctxs, configs=cfgs)),
                    ("i", lambda: [invariance(m, ref, tok, j, kwnew=kw, name=n) for n, kw in cfgs])):
        if sec in sections:
            try:
                fn()
            except Exception:
                j.add(phase="fatal", sec=sec, err=traceback.format_exc()[-900:]); print("FATAL", sec, traceback.format_exc()[-900:], flush=True)
    return j
