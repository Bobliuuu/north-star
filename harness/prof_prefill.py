"""Where does single-prompt prefill time go? GraphNorth.eager_prefill vs LCNorth.prefill at several prompt lengths, plus a CUDA kernel
profile of one GraphNorth prefill. Rows -> Jsonl."""
import collections, time, traceback
import torch
from torch.profiler import profile, ProfilerActivity
from common import Jsonl
from northstar.refmodel import RefNorth
from graph_north import GraphNorth
from fast_north import FastNorth
from longctx_north import LCNorth


@torch.no_grad()
def run(m, tok, lens=(128, 512, 1024, 1700), Lmax=2048):
    j = Jsonl()
    try:
        ref = RefNorth(m)
        X = FastNorth(ref, Lmax, exact=True); kv = X.new_kv()
        gb = GraphNorth(ref, Lmax, "bmm"); ln = LCNorth(ref, Lmax)
        ids_all = list(range(1000, 1000 + max(lens)))
        for L in lens:
            ids = ids_all[:L]
            for name, fn in (("graph_bmm", lambda: gb.eager_prefill(ids, kv)), ("lc_sdpa", lambda: ln.prefill(ids, kv, chunk=1024))):
                fn(); torch.cuda.synchronize()
                ts = []
                for _ in range(3):
                    t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
                j.add(phase="prefill", impl=name, L=L, s=min(ts), tok_s=L / min(ts)); print("PREFILL", name, L, "%.3fs %.0f tok/s" % (min(ts), L / min(ts)), flush=True)
        for name, fn in (("graph_bmm", lambda: gb.eager_prefill(ids_all[:1024], kv)), ("lc_sdpa", lambda: ln.prefill(ids_all[:1024], kv, chunk=1024))):
            fn(); torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
                fn(); torch.cuda.synchronize()
            agg = collections.defaultdict(lambda: [0.0, 0])
            for e in prof.events():
                if e.device_type == torch.autograd.DeviceType.CUDA:
                    agg[e.name[:70]][0] += (e.time_range.end - e.time_range.start) / 1e3; agg[e.name[:70]][1] += 1
            top = sorted(agg.items(), key=lambda x: -x[1][0])[:15]
            tot = sum(v[0] for v in agg.values())
            j.add(phase="prof", impl=name, L=1024, kernel_ms=tot, launches=sum(v[1] for v in agg.values()), top={k: dict(ms=v[0], n=v[1]) for k, v in top})
            print("PROF", name, "kernel ms %.1f launches %d" % (tot, sum(v[1] for v in agg.values())), flush=True)
            for k, v in top: print("   %8.2f ms %5d %s" % (v[0], v[1], k), flush=True)
    except Exception:
        j.add(phase="fatal", err=traceback.format_exc()[-1500:]); print("FATAL", traceback.format_exc()[-1500:], flush=True)
    return j
