"""A/B: per-sequence attention launches (batch_exact.BATCH_ATTN=False) vs one batched launch per layer (True), same MoE mode (auto gather).
Per (B, T): eager logits bitwise + the K/V bytes written at the new positions bitwise, graph median ms on real rows; per B: live plain decode."""
import traceback
import torch
from common import Jsonl
from northstar.refmodel import RefNorth
from graph_north import GraphNorth, time_graph
from fast_north import FastNorth
import batch_exact as be
import fast_kernels_batch as fkb
import bench_batch as bb


@torch.no_grad()
def run(m, tok, sizes=(1, 4, 16, 32), Ts=(1, 4, 8, 13), n_new=64):
    j = Jsonl()
    try:
        fkb.TILE_GRID = 4
        ref = RefNorth(m); dev = ref.device
        eos = m.config.eos_token_id; eos = eos if isinstance(eos, int) else eos[0]
        pool = bb.build_pool(tok, 32)
        Lmax = max(len(p[2]) for p in pool) + max(n_new, 16) + 40
        X = FastNorth(ref, Lmax, exact=True); gb = GraphNorth(ref, Lmax, "bmm"); BX = be.BatchExact(X)
        kvs = []
        for (_, _, ids) in pool:
            kv = X.new_kv(); gb.eager_prefill(ids[:-1], kv); kvs.append(kv)
        del gb; torch.cuda.empty_cache()
        for B in sizes:
            grp = pool[:B]; kv = kvs[:B]
            P = [len(g[2]) - 1 for g in grp]; st = torch.tensor(P, device=dev)
            for T in Ts:
                ids = torch.tensor([list(g[2][-T:]) for g in grp], device=dev)
                lg, kvw = {}, {}
                for mode in (False, True):
                    be.BATCH_ATTN = mode
                    lg[mode] = BX.forward(ids, st, kv).clone()
                    kvw[mode] = [torch.stack([kv[s][i][w][:, :, P[s]:P[s] + T].clone() for i in (0, len(kv[s]) - 1) for w in (0, 1)]) for s in range(B)]
                eq = bool(torch.equal(lg[False], lg[True])); keq = all(torch.equal(a, b) for a, b in zip(kvw[False], kvw[True]))
                ms = {}
                for mode in (False, True):
                    be.BATCH_ATTN = mode; s = BX.make_step(B, T, kv); bb._set_rows(s, grp, T, real=True)
                    ms["batched" if mode else "perseq"] = time_graph(s, n=20)[0]; del s
                j.add(phase="ab", B=B, T=T, logits_bitwise=eq, kv_bitwise=keq, ms=ms, speedup=ms["perseq"] / ms["batched"])
                print("AB B", B, "T", T, "logits_eq", eq, "kv_eq", keq, {k: round(v, 2) for k, v in ms.items()}, "x%.3f" % (ms["perseq"] / ms["batched"]), flush=True)
                torch.cuda.empty_cache()
            res = {}
            for mode in (False, True):
                be.BATCH_ATTN = mode; s = BX.make_step(B, 1, kv)
                bb.run_plain(s, grp, 8, eos)
                out, td, wall, _ = bb.run_plain(s, grp, n_new, eos); del s
                res[mode] = (out, sum(len(o) for o in out) / wall)
            j.add(phase="live_plain", B=B, n_new=n_new, perseq_tok_s=res[False][1], batched_tok_s=res[True][1], outputs_equal=res[False][0] == res[True][0])
            print("LIVE B", B, "perseq %.0f batched %.0f tok/s" % (res[False][1], res[True][1]), "equal", res[False][0] == res[True][0], flush=True)
            torch.cuda.empty_cache()
    except Exception:
        j.add(phase="fatal", err=traceback.format_exc()[-1500:]); print("FATAL", traceback.format_exc()[-1500:], flush=True)
    finally:
        be.BATCH_ATTN = "auto"
    return j
