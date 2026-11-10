"""Per-kernel GPU time of one batched exact step (eager BatchExact.forward, real prompt-tail rows), grouped by kernel name.
Answers: at large B, which kernels dominate the step (attention per sequence, MoE, linears, norms, launch count)."""
import collections, traceback
import torch
from torch.profiler import profile, ProfilerActivity
from common import Jsonl
from northstar.refmodel import RefNorth
from graph_north import GraphNorth, time_graph
from fast_north import FastNorth
from batch_exact import BatchExact
import fast_kernels_batch as fkb
import bench_batch as bb


@torch.no_grad()
def run(m, tok, sizes=(1, 8, 16, 32), Ts=(1, 8), mode=4):
    j = Jsonl()
    try:
        fkb.TILE_GRID = mode
        ref = RefNorth(m); dev = ref.device
        pool = bb.build_pool(tok, 32)
        Lmax = max(len(p[2]) for p in pool) + 232
        X = FastNorth(ref, Lmax, exact=True); gb = GraphNorth(ref, Lmax, "bmm"); BX = BatchExact(X)
        kvs = []
        for (_, _, ids) in pool:
            kv = X.new_kv(); gb.eager_prefill(ids[:-1], kv); kvs.append(kv)
        del gb; torch.cuda.empty_cache()
        for B in sizes:
            grp = pool[:B]; kv = kvs[:B]
            st = torch.tensor([len(g[2]) - 1 for g in grp], device=dev)
            for T in Ts:
                ids = torch.tensor([list(g[2][-T:]) for g in grp], device=dev)
                for _ in range(2): BX.forward(ids, st, kv)
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    BX.forward(ids, st, kv); torch.cuda.synchronize()
                agg = collections.defaultdict(lambda: [0.0, 0])
                for e in prof.events():
                    if e.device_type == torch.autograd.DeviceType.CUDA:
                        agg[e.name[:60]][0] += e.device_time_total / 1e3 if hasattr(e, "device_time_total") else e.cuda_time_total / 1e3
                        agg[e.name[:60]][1] += 1
                tot = sum(v[0] for v in agg.values()); n = sum(v[1] for v in agg.values())
                top = sorted(agg.items(), key=lambda kv_: -kv_[1][0])[:12]
                s = BX.make_step(B, T, kv); bb._set_rows(s, grp, T, real=True); g_ms = time_graph(s, n=20)[0]; del s
                j.add(phase="prof", B=B, T=T, mode=mode, kernel_ms_total=tot, launches=n, graph_ms=g_ms, top={k: dict(ms=v[0], n=v[1]) for k, v in top})
                print("PROF B", B, "T", T, "graph %.2f ms, kernel sum %.2f ms, launches %d" % (g_ms, tot, n), flush=True)
                for k, v in top: print("   %7.2f ms %5d  %s" % (v[0], v[1], k), flush=True)
                torch.cuda.empty_cache()
    except Exception:
        j.add(phase="fatal", err=traceback.format_exc()[-1500:]); print("FATAL", traceback.format_exc()[-1500:], flush=True)
    return j
