"""Batched exact executor (BatchExact): concurrent attention || MoE streams and tuned linear tiles vs the H20 path, plus a breakdown of the
live plain loop (where does a B=32 plain step spend 15 ms when its graph replays in 8 ms?). H100, same 32-prompt pool as H18-H20.
Rows: eager (logits bitwise vs base), cost (graph ms per B x T x variant), live (plain decode tok/s, per-step host/device split, outputs vs base)."""
import json, time, traceback
import statistics as st
import torch

from common import Jsonl
from northstar.refmodel import RefNorth
from graph_north import GraphNorth, time_graph
from fast_north import FastNorth
import batch_exact as bxm
from batch_exact import BatchExact
import fast_kernels_exact as fx
import bench_batch as bb

TUNED = {(2048, 4096): dict(BN=32, BK=512, nw=4, ns=4), (5120, 2048): dict(BN=64, BK=128, nw=4, ns=6)}


def _variant(name):
    bxm.STREAMS = name in ("streams", "streams_tiles")
    fx.LIN_CFG.clear()
    if name == "streams_tiles":
        fx.LIN_CFG.update(TUNED)


def run_plain_instr(step, grp, n_new, eos):
    """bench_batch.run_plain with timers: set (host -> device inputs), run (graph launch call), sync (.tolist(): waits for the GPU)."""
    B = len(grp)
    cur = [g[2][-1] for g in grp]; pos = [len(g[2]) - 1 for g in grp]
    out = [[] for _ in range(B)]; done = [False] * B
    ts, tr, ty, th = [], [], [], []
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for i in range(n_new):
        a = time.perf_counter()
        step.set([[c] for c in cur], pos, [[not d] for d in done])
        b_ = time.perf_counter()
        r = step.run()
        c = time.perf_counter()
        pred = r.tolist()
        d = time.perf_counter()
        for k in range(B):
            pos[k] += 1
            if done[k]: continue
            tk = pred[k][0]; out[k].append(tk); cur[k] = tk
            if tk == eos or len(out[k]) >= n_new: done[k] = True
        e = time.perf_counter()
        ts.append(b_ - a); tr.append(c - b_); ty.append(d - c); th.append(e - d)
        if all(done): break
    wall = time.perf_counter() - t0
    ms = lambda v: 1e3 * st.median(v)
    return out, wall, dict(set_ms=ms(ts), run_ms=ms(tr), sync_ms=ms(ty), host_ms=ms(th), steps=len(ts))


@torch.no_grad()
def bench(m, tok, j, sizes=(1, 4, 8, 16, 32), n_new=96, variants=("base", "streams", "streams_tiles")):
    ref = RefNorth(m); dev = ref.device
    eos = m.config.eos_token_id; eos = eos if isinstance(eos, int) else eos[0]
    pool = bb.build_pool(tok, 32)
    Lmax = max(len(p[2]) for p in pool) + n_new + 40
    X = FastNorth(ref, Lmax, exact=True); gb = GraphNorth(ref, Lmax, "bmm"); BX = BatchExact(X)
    kvs = []
    for (_, _, ids) in pool:
        kv = X.new_kv(); gb.eager_prefill(ids[:-1], kv); kvs.append(kv)
    del gb; torch.cuda.empty_cache()
    # eager bitwise: logits of each variant vs base (B=8, T=5, ragged starts)
    for B, T in ((8, 5), (32, 1)):
        ids = torch.tensor([p[2][-T:] for p in pool[:B]], device=dev); stt = torch.tensor([len(p[2]) - 1 for p in pool[:B]], device=dev)
        lg = {}
        for v in variants:
            _variant(v); lg[v] = BX.forward(ids, stt, kvs[:B]).clone()
        j.add(phase="eager", B=B, T=T, equal={v: bool(torch.equal(lg[v], lg["base"])) for v in variants})
        print("EAGER B", B, "T", T, json.loads(j.lines[-1])["equal"], flush=True)
    for B in sizes:
        grp = pool[:B]; kv = kvs[:B]
        outs = {}
        for v in variants:
            try:
                _variant(v)
                steps = {T: BX.make_step(B, T, kv) for T in (1, 4, 13)}
                c = {}
                for T in steps:
                    bb._set_rows(steps[T], grp, T, real=True); c[T] = time_graph(steps[T], n=20)[0]
                bb.run_plain(steps[1], grp, 8, eos)  # warm
                o, wall, br = run_plain_instr(steps[1], grp, n_new, eos)
                # replay-only loop (no set / no host read) and replay + sync per step: pure device step time in a loop
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for _ in range(30): steps[1].g.replay()
                torch.cuda.synchronize(); rep_ms = (time.perf_counter() - t0) / 30 * 1e3
                t0 = time.perf_counter()
                for _ in range(30): steps[1].g.replay(); torch.cuda.synchronize()
                rep_sync_ms = (time.perf_counter() - t0) / 30 * 1e3
                outs[v] = o
                nt = sum(len(x) for x in o)
                j.add(phase="live", B=B, variant=v, graph_ms={str(k): x for k, x in c.items()}, agg_tok_s=nt / wall, step_ms=wall / br["steps"] * 1e3,
                      replay_loop_ms=rep_ms, replay_sync_ms=rep_sync_ms, equal_base=(o == outs["base"]) if "base" in outs else None, **br)
                r = json.loads(j.lines[-1])
                print("LIVE B", B, v, "graph", {k: round(x, 2) for k, x in c.items()}, "agg %.0f tok/s step %.2f ms (set %.3f run %.3f sync %.3f host %.3f) replay %.2f / +sync %.2f eq_base %s"
                      % (r["agg_tok_s"], r["step_ms"], r["set_ms"], r["run_ms"], r["sync_ms"], r["host_ms"], rep_ms, rep_sync_ms, r["equal_base"]), flush=True)
                del steps; torch.cuda.empty_cache()
            except Exception:
                j.add(phase="error", B=B, variant=v, err=traceback.format_exc()[-1200:]); print("ERR", B, v, traceback.format_exc()[-900:], flush=True)
    _variant("base")


def run(m, tok, sizes=(1, 4, 8, 16, 32), n_new=96):
    j = Jsonl()
    try:
        bench(m, tok, j, sizes, n_new)
    except Exception:
        j.add(phase="fatal", err=traceback.format_exc()[-1500:]); print("FATAL", traceback.format_exc()[-1500:], flush=True)
    return j
