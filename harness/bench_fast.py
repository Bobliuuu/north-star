"""fast-exec on H100: (A) per-kernel profile of the graphed-baseline forward, (B) CUDA-graph ms/token for option ablations,
(C) token-identity vs RefNorth (HF-module eager greedy) on dev workloads + host-loop ms/token. Sections are independent (try/except)."""
import collections, statistics as st, time, traceback
import torch

from common import Jsonl, build_ids, by_split, load_workloads
from northstar.refmodel import RefNorth
from northstar.spec import plain_greedy
from graph_north import GraphNorth, time_graph
from fast_north import FastNorth

CONFIGS = {  # name -> opts (None = GraphNorth baseline)
    "base": None,
    "attn": ("attn",),
    "moe": ("moe",),
    "router+moe": ("router", "moe"),
    "router+moe+resid": ("router", "moe", "resid"),
    "attn+resid": ("attn", "resid"),
    "all": ("attn", "moe", "router", "resid"),
}


def profile(gn, kv, P, last_id, j, tag, iters=4):
    from torch.profiler import profile as tprof, ProfilerActivity
    ids = torch.tensor([[last_id]], device=gn.dev)
    gn.start.fill_(P - 1)
    with torch.no_grad():
        for _ in range(2):
            gn.forward(ids, kv)
        torch.cuda.synchronize()
        with tprof(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            for _ in range(iters):
                gn.forward(ids, kv)
            torch.cuda.synchronize()
    rows = []
    from torch.autograd import DeviceType
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
    j.add(phase="profile_summary", tag=tag, kernels_per_forward=nk, sum_kernel_ms=tot)
    for name, c, ms in rows[:45]:
        j.add(phase="profile_kernel", tag=tag, kernel=name[:110], count_per_fwd=c, ms_per_fwd=ms, pct=100 * ms / tot)
    print("profile", tag, "kernels/fwd", nk, "sum kernel ms", round(tot, 3), flush=True)


@torch.no_grad()
def timing(ref, tok, j, ctxs=(2048, 8192), reps=50):
    code = [r for r in by_split(load_workloads(), "dev") if r["kind"] == "code"]
    for ctx in ctxs:
        ids = build_ids(tok, code[0], ctx); P = len(ids)
        base = GraphNorth(ref, P + 64, "bmm"); kv = base.new_kv(); base.eager_prefill(ids[:-1], kv)
        if ctx == ctxs[0]:
            try:
                profile(base, kv, P, ids[-1], j, "base_ctx%d" % P)
            except Exception:
                j.add(phase="profile_error", err=traceback.format_exc()[-600:])
        fast = FastNorth(ref, P + 64, opts=("attn", "moe", "router", "resid"))
        for name, opts in CONFIGS.items():
            try:
                gnx = base if opts is None else fast
                if opts is not None:
                    fast.opts = frozenset(opts)
                s = gnx.make_step(1, kv); s.set([ids[-1]], P - 1)
                med, mn, mx = time_graph(s, n=reps)
                j.add(phase="fast_timing", ctx=P, config=name, graph_ms=med, ms_min=mn, ms_max=mx)
                print("timing", P, name, round(med, 3), flush=True); del s
            except Exception:
                j.add(phase="fast_timing_error", ctx=P, config=name, err=traceback.format_exc()[-800:]); print("ERR", name, traceback.format_exc()[-400:], flush=True)
        # moe kernel tile sweep on 'all'
        fast.opts = frozenset(CONFIGS["all"])
        for kc in (dict(BN1=8, BK1=256, BN2=8, BK2=256), dict(BN1=16, BK1=128, BN2=16, BK2=128), dict(BN1=32, BK1=256, BN2=32, BK2=256, nw1=8, nw2=8),
                   dict(BN1=16, BK1=512, BN2=16, BK2=256, nw1=8, nw2=4), dict(BN1=8, BK1=512, BN2=16, BK2=256, nw1=2, nw2=4)):
            try:
                fast.kcfg = kc
                s = fast.make_step(1, kv); s.set([ids[-1]], P - 1)
                med, mn, mx = time_graph(s, n=reps)
                j.add(phase="fast_tile_sweep", ctx=P, kcfg=str(kc), graph_ms=med); print("tile", P, kc, round(med, 3), flush=True); del s
            except Exception:
                j.add(phase="fast_tile_error", ctx=P, kcfg=str(kc), err=traceback.format_exc()[-400:])
        fast.kcfg = None
        if ctx == ctxs[0]:
            try:
                profile(fast, kv, P, ids[-1], j, "fast_all_ctx%d" % P)
            except Exception:
                j.add(phase="profile_error", err=traceback.format_exc()[-600:])
        del fast, base, kv; torch.cuda.empty_cache()


@torch.no_grad()
def correctness(ref, tok, j, ngen=48, long_ctx=2048):
    rows = by_split(load_workloads(), "dev")
    cells = [(r["id"], build_ids(tok, r)) for r in rows]
    code = [r for r in rows if r["kind"] == "code"]
    cells.append(("code_ctx%d" % long_ctx, build_ids(tok, code[0], long_ctx)))
    names = ["base", "all", "attn", "router+moe+resid"]
    for tid, ids in cells:
        try:
            P = len(ids)
            n = ngen if P < 1500 else 32
            pg = plain_greedy(ref, ids, n)
            rt, margins = pg["tokens"], pg["margins"]
            base = GraphNorth(ref, P + n + 24, "bmm"); kv = base.new_kv(); base.eager_prefill(ids[:-1], kv)
            fast = FastNorth(ref, P + n + 24, opts=("attn", "moe", "router", "resid"))
            outs = {}
            for name in names:
                opts = CONFIGS[name]
                g = base if opts is None else fast
                if opts is not None:
                    fast.opts = frozenset(opts)
                s = g.make_step(1, kv)
                toks, cur = [], ids[-1]
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for i in range(n):
                    s.set([cur], P - 1 + i); cur = int(s.run()[0, 0]); toks.append(cur)
                torch.cuda.synchronize(); dt = time.perf_counter() - t0
                outs[name] = toks
                fd = next((i for i, (a, b) in enumerate(zip(toks, rt)) if a != b), None)
                j.add(phase="fast_correct", task=tid, prompt_len=P, config=name, n=n, identical_to_ref=fd is None, first_div_vs_ref=fd,
                      ref_margin_at_div=None if fd is None else float(margins[fd]), host_loop_ms_per_tok=1e3 * dt / n)
                print("correct", tid, name, "div", fd, "" if fd is None else "margin %.3f" % margins[fd], "%.2f ms/tok" % (1e3 * dt / n), flush=True)
                del s
            fd = next((i for i, (a, b) in enumerate(zip(outs["all"], outs["base"])) if a != b), None)
            j.add(phase="fast_correct_vs_base", task=tid, all_vs_base_first_div=fd)
            del fast, base, kv; torch.cuda.empty_cache()
        except Exception:
            j.add(phase="fast_correct_error", task=tid, err=traceback.format_exc()[-800:]); print("ERR", tid, traceback.format_exc()[-500:], flush=True)


def run(m, tok, sections="tc"):
    ref = RefNorth(m); j = Jsonl()
    if "t" in sections:
        try:
            timing(ref, tok, j)
        except Exception:
            j.add(phase="timing_fatal", err=traceback.format_exc()[-800:])
    if "c" in sections:
        correctness(ref, tok, j)
    return j
