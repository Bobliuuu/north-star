"""Real-North A/B of the MoE / linear tuning (branch moe-tune). Configs (fkb.TILE_GRID, fkb.LIN_TW):
ref = (0, 16) serial, the exact reference | main = (4, 16) current main | moe5 = (6, 16) | lin64 = (4, 64) | new = (6, 64).
Per B x T cell on real prompt-tail rows: eager logits BITWISE vs ref, graph median ms per config. Per B: live plain decode, outputs equal + tok/s."""
import traceback
import torch
from common import Jsonl
from northstar.refmodel import RefNorth
from graph_north import GraphNorth, time_graph
from fast_north import FastNorth
from batch_exact import BatchExact
import fast_kernels_batch as fkb
import bench_batch as bb

CONFIGS = {"ref": (0, 16), "main": (4, 16), "moe5": (6, 16), "lin64": (4, 64), "new": (6, 64)}


def _use(name):
    fkb.TILE_GRID, fkb.LIN_TW = CONFIGS[name]


@torch.no_grad()
def run(m, tok, sizes=(1, 2, 4, 8, 16, 32), Ts=(1, 2, 4, 8, 13), n_new=64):
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
            grp = pool[:B]; kv = kvs[:B]
            st = torch.tensor([len(g[2]) - 1 for g in grp], device=dev)
            for T in Ts:
                ids = torch.tensor([list(g[2][-T:]) for g in grp], device=dev)
                lg = {}
                for name in CONFIGS:
                    _use(name); lg[name] = BX.forward(ids, st, kv).clone()
                eq = {name: bool(torch.equal(lg[name], lg["ref"])) for name in CONFIGS}
                del lg
                ms = {}
                for name in CONFIGS:
                    _use(name); s = BX.make_step(B, T, kv); bb._set_rows(s, grp, T, real=True)
                    ms[name] = time_graph(s, n=20)[0]; del s
                sp = {name: ms["main"] / v for name, v in ms.items()}
                j.add(phase="ab", B=B, T=T, NT=B * T, bitwise_vs_ref=eq, ms=ms, speedup_vs_main=sp)
                print("AB B", B, "T", T, "bitwise", all(eq.values()), {k: round(v, 2) for k, v in ms.items()}, "new/main x%.3f" % sp["new"], flush=True)
                torch.cuda.empty_cache()
            res = {}
            for name in ("ref", "main", "new"):
                _use(name); s = BX.make_step(B, 1, kv)
                bb.run_plain(s, grp, 8, eos)
                out, td, wall, _ = bb.run_plain(s, grp, n_new, eos); del s
                res[name] = (out, sum(len(o) for o in out) / wall)
            eqo = all(res[k][0] == res["ref"][0] for k in res)
            j.add(phase="live_plain", B=B, n_new=n_new, tok_s={k: v[1] for k, v in res.items()}, outputs_equal=eqo)
            print("LIVE B", B, {k: round(v[1]) for k, v in res.items()}, "tok/s equal", eqo, flush=True)
            torch.cuda.empty_cache()
    except Exception:
        j.add(phase="fatal", err=traceback.format_exc()[-1500:]); print("FATAL", traceback.format_exc()[-1500:], flush=True)
    finally:
        _use("ref")
    return j
