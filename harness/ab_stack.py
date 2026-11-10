"""Exactness of the full optimized batched stack vs the original serial path, on the real model.
A = TILE_GRID 0 (serial MoE), per-sequence attention, no row mask, LIN_TW 16 (H18 code path).
B = TILE_GRID 6 (compact gathered MoE), batched attention, random valid-row mask (as short drafts / finished rows produce), LIN_TW 64.
Per (B, T): logits of VALID rows bitwise; graph ms of A and B."""
import random, traceback
import torch
from common import Jsonl
from northstar.refmodel import RefNorth
from graph_north import GraphNorth, time_graph
from fast_north import FastNorth
import batch_exact as be
import fast_kernels_batch as fkb
import bench_batch as bb


def _cfg(opt):
    fkb.TILE_GRID = 6 if opt else 0; be.BATCH_ATTN = "auto" if opt else False; be.MASK_PAD = bool(opt); fkb.LIN_TW = 64 if opt else 16


@torch.no_grad()
def run(m, tok, sizes=(1, 2, 4, 8, 16, 32), Ts=(1, 2, 4, 8, 13)):
    j = Jsonl()
    try:
        ref = RefNorth(m); dev = ref.device
        pool = bb.build_pool(tok, 32)
        Lmax = max(len(p[2]) for p in pool) + 232
        X = FastNorth(ref, Lmax, exact=True); gb = GraphNorth(ref, Lmax, "bmm"); BX = be.BatchExact(X)
        kvs = []
        for (_, _, ids) in pool:
            kv = X.new_kv(); gb.eager_prefill(ids[:-1], kv); kvs.append(kv)
        del gb; torch.cuda.empty_cache()
        rng = random.Random(7)
        for B in sizes:
            grp = pool[:B]; kv = kvs[:B]; st = torch.tensor([len(g[2]) - 1 for g in grp], device=dev)
            for T in Ts:
                ids = torch.tensor([list(g[2][-T:]) for g in grp], device=dev)
                n_ok = [0 if (B > 1 and rng.random() < 0.15) else rng.randint(1, T) for _ in range(B)]   # rows per sequence that are real
                val = torch.tensor([[t < n for t in range(T)] for n in n_ok], device=dev).reshape(-1)
                _cfg(False); la = BX.forward(ids, st, kv).clone()
                _cfg(True); BX._state(B, T)["valid"].copy_(val); lb = BX.forward(ids, st, kv).clone(); BX._state(B, T)["valid"].fill_(True)
                eq = bool(torch.equal(la[val], lb[val]))
                ms = {}
                for opt in (False, True):
                    _cfg(opt); s = BX.make_step(B, T, kv); bb._set_rows(s, grp, T, real=True); ms["opt" if opt else "orig"] = time_graph(s, n=20)[0]; del s
                j.add(phase="stack", B=B, T=T, valid_rows=int(val.sum()), rows=B * T, valid_logits_bitwise=eq, ms=ms, speedup=ms["orig"] / ms["opt"])
                print("STACK B", B, "T", T, "valid %d/%d" % (int(val.sum()), B * T), "bitwise", eq, {k: round(v, 2) for k, v in ms.items()}, "x%.2f" % (ms["orig"] / ms["opt"]), flush=True)
                torch.cuda.empty_cache()
    except Exception:
        j.add(phase="fatal", err=traceback.format_exc()[-1500:]); print("FATAL", traceback.format_exc()[-1500:], flush=True)
    finally:
        _cfg(True)
    return j
