"""A/B: batched MoE row tiles walked serially by the leader (fkb.TILE_GRID=False, H18 bs1) vs one program per 16-row tile (TILE_GRID=True).
Same 32-prompt pool as bench_batch. Per B: (1) eager logits on real prompt-tail rows, bitwise serial vs grid, for T in TS;
(2) graph median ms on the same real rows, both modes; (3) live plain decode n_new tokens, both modes: tok/s and outputs equal."""
import time, traceback
import torch
from common import Jsonl
from northstar.refmodel import RefNorth
from graph_north import GraphNorth, time_graph
from fast_north import FastNorth
from batch_exact import BatchExact
import fast_kernels_batch as fkb
import bench_batch as bb

TS = (1, 2, 4, 8, 13)
MODES = {0: "serial", 1: "tile", 2: "egrid", 3: "gather", 4: "auto"}


@torch.no_grad()
def run(m, tok, sizes=(1, 4, 8, 16, 32), n_new=64):
    j = Jsonl()
    try:
        ref = RefNorth(m); dev = ref.device
        eos = m.config.eos_token_id; eos = eos if isinstance(eos, int) else eos[0]
        pool = bb.build_pool(tok, 32)
        Lmax = max(len(p[2]) for p in pool) + max(n_new, 16) + 40
        X = FastNorth(ref, Lmax, exact=True); gb = GraphNorth(ref, Lmax, "bmm"); BX = BatchExact(X)
        kvs = []
        for (_, _, ids) in pool:
            kv = X.new_kv(); gb.eager_prefill(ids[:-1], kv); kvs.append(kv)
        del gb; torch.cuda.empty_cache()
        for B in sizes:
            idxs = list(range(B)); grp = [pool[i] for i in idxs]; kv = [kvs[i] for i in idxs]
            st = torch.tensor([len(g[2]) - 1 for g in grp], device=dev)
            for T in TS:
                ids = torch.tensor([list(g[2][-T:]) for g in grp], device=dev)
                lg = {}
                for mode in (0, 2, 3, 4):
                    fkb.TILE_GRID = mode; lg[mode] = BX.forward(ids, st, kv).clone()
                eq = bool(all(torch.equal(lg[0], lg[q]) for q in (2, 3, 4)))
                ms = {}
                for mode in (0, 2, 3, 4):
                    fkb.TILE_GRID = mode; s = BX.make_step(B, T, kv); bb._set_rows(s, grp, T, real=True)
                    ms[MODES[mode]] = time_graph(s, n=20)[0]; del s
                j.add(phase="ab", B=B, T=T, NT=B * T, eager_bitwise_equal=eq, ms=ms, speedup={k: ms["serial"] / v for k, v in ms.items()})
                print("AB B", B, "T", T, "eq", eq, {k: round(v, 2) for k, v in ms.items()}, {k: round(ms["serial"] / v, 3) for k, v in ms.items()}, flush=True)
            torch.cuda.empty_cache()
            res = {}
            for mode in (0, 2, 3, 4):
                fkb.TILE_GRID = mode; s = BX.make_step(B, 1, kv)
                bb.run_plain(s, grp, 8, eos)
                out, td, wall, _ = bb.run_plain(s, grp, n_new, eos); del s
                res[mode] = (out, sum(len(o) for o in out) / wall)
            j.add(phase="live_plain", B=B, n_new=n_new, tok_s={MODES[k]: v[1] for k, v in res.items()}, outputs_equal=all(res[k][0] == res[0][0] for k in res))
            print("LIVE B", B, {MODES[k]: round(v[1]) for k, v in res.items()}, "tok/s equal", all(res[k][0] == res[0][0] for k in res), flush=True)
            torch.cuda.empty_cache()
    except Exception:
        j.add(phase="fatal", err=traceback.format_exc()[-1500:]); print("FATAL", traceback.format_exc()[-1500:], flush=True)
    finally:
        fkb.TILE_GRID = 0
    return j
