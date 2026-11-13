"""Headline re-measurement in ONE launch (H29): the H9 setup (dev code prompt padded to ctx 2048 / 8192, T=1 CUDA-graph replay, median of
50) for four decoders on the same KV and host:
  base     GraphNorth bmm (H2/H5/H9 baseline, 10.73 ms in H9)
  fast     FastNorth non-exact, all four options (the H9 "2.7x" executor, 3.92 ms in H9)
  exact_v1 FastNorth(exact=True) (H14 path)
  exact_v2 FastNorth(exact=True).set_x2(**recommended(Lmax)) (H24)
plus a live greedy decode of 192 tokens from the same prompt per decoder (wall tok/s) and token equality of exact_v1 vs exact_v2."""
import time, traceback
import torch

from common import Jsonl, build_ids, by_split, load_workloads
from northstar.refmodel import RefNorth
from graph_north import GraphNorth, time_graph
from fast_north import FastNorth
import fast_kernels_x2 as f2

N_NEW = 192


def _live(step, ids, P, eos):
    out, cur = [], ids[-1]
    torch.cuda.synchronize(); t = time.perf_counter()
    for i in range(N_NEW):
        step.set([cur], P - 1 + i); cur = int(step.run()[0, 0]); out.append(cur)
        if cur == eos:
            break
    torch.cuda.synchronize()
    return out, len(out) / (time.perf_counter() - t)


@torch.no_grad()
def run(m, tok, ctxs=(2048, 8192), reps=50):
    j = Jsonl()
    eos = m.config.eos_token_id; eos = eos if isinstance(eos, int) else eos[0]
    ref = RefNorth(m)
    code = [r for r in by_split(load_workloads(), "dev") if r["kind"] == "code"]
    for ctx in ctxs:
        try:
            ids = build_ids(tok, code[0], ctx); P = len(ids); Lmax = P + N_NEW + 64
            base = GraphNorth(ref, Lmax, "bmm"); kv = base.new_kv()
            fast = FastNorth(ref, Lmax, opts=("attn", "moe", "router", "resid"))
            x1 = FastNorth(ref, Lmax, exact=True)
            x2 = FastNorth(ref, Lmax, exact=True); x2.set_x2(**f2.recommended(Lmax))
            res, toks = {}, {}
            for name, gn in (("base", base), ("fast", fast), ("exact_v1", x1), ("exact_v2", x2)):
                base.eager_prefill(ids[:-1], kv)      # capture warmups write scratch KV: re-prefill before each decoder
                s = gn.make_step(1, kv)
                base.eager_prefill(ids[:-1], kv)
                s.set([ids[-1]], P - 1)
                med, mn, mx = time_graph(s, n=reps)
                out, tps = _live(s, ids, P, eos)
                res[name] = med; toks[name] = out
                j.add(phase="headline", ctx=P, decoder=name, graph_ms=med, ms_min=mn, ms_max=mx, live_tok_s=tps, n=len(out))
                print("HEADLINE ctx %d %-9s graph %.3f ms (min %.3f) live %.1f tok/s n=%d" % (P, name, med, mn, tps, len(out)), flush=True)
                del s; torch.cuda.empty_cache()
            j.add(phase="summary", ctx=P, speedup_vs_base={k: res["base"] / v for k, v in res.items()},
                  exact_v2_tokens_equal_v1=toks["exact_v2"] == toks["exact_v1"])
            print("SUMMARY ctx %d" % P, {k: round(res["base"] / v, 2) for k, v in res.items()}, "v2==v1 tokens:", toks["exact_v2"] == toks["exact_v1"], flush=True)
            del base, fast, x1, x2, kv; torch.cuda.empty_cache()
        except Exception:
            j.add(phase="error", ctx=ctx, err=traceback.format_exc()[-1500:]); print("ERR", ctx, traceback.format_exc()[-1200:], flush=True)
            torch.cuda.empty_cache()
    return j
