"""diag2: reproduce the bench conditions (pool of 32, groups of B, graphs per group, 192 greedy tokens) and compare every row with the
ORIGINAL single-sequence exact FastNorth T=1 graph (the H14-validated reference). Also step-0 eager logits per row."""
import json, time, traceback
import torch
from common import Jsonl
from northstar.refmodel import RefNorth
from graph_north import GraphNorth
from fast_north import FastNorth
from batch_exact import BatchExact
import bench_batch as bb


def fd(a, b):
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), None if len(a) == len(b) else n)


@torch.no_grad()
def run(m, tok):
    j = Jsonl(); ref = RefNorth(m); dev = ref.device
    eos = m.config.eos_token_id; eos = eos if isinstance(eos, int) else eos[0]
    pool = bb.build_pool(tok, 32)
    Lmax = max(len(p[2]) for p in pool) + 192 + 40
    X = FastNorth(ref, Lmax, exact=True); gb = GraphNorth(ref, Lmax, "bmm"); BX = BatchExact(X)
    kvs = []
    for (tid, kind, ids) in pool:
        kv = X.new_kv(); gb.eager_prefill(ids[:-1], kv); kvs.append(kv)
    # reference: original exact executor, T=1 graph, one sequence at a time
    refs = []
    for i, (tid, kind, ids) in enumerate(pool):
        st = X.make_step(1, kvs[i]); P = len(ids); cur = ids[-1]; out = []
        for s in range(192):
            st.set([cur], P - 1 + s); cur = int(st.run()[0, 0]); out.append(cur)
            if cur == eos: break
        refs.append(out); del st
    j.add(phase="ref", lens=[len(r) for r in refs]); print("ref lens", [len(r) for r in refs], flush=True)
    for B in (1, 2, 4, 8, 32):
        for gi in range(32 // B):
            idxs = list(range(gi * B, (gi + 1) * B))
            steps = BX.make_step(B, 1, [kvs[i] for i in idxs])
            cur = [pool[i][2][-1] for i in idxs]; pos = [len(pool[i][2]) - 1 for i in idxs]
            outs = [[] for _ in idxs]; done = [False] * B
            for s in range(192):
                steps.set([[c] for c in cur], pos); pred = steps.run().tolist()
                for b in range(B):
                    pos[b] += 1
                    if done[b]: continue
                    outs[b].append(pred[b][0]); cur[b] = pred[b][0]
                    if cur[b] == eos: done[b] = True
                if all(done): break
            res = [fd(outs[b], refs[i]) for b, i in enumerate(idxs)]
            j.add(phase="graph_vs_ref", B=B, group=gi, rows=[pool[i][0] for i in idxs], first_mismatch=res, n=[len(o) for o in outs])
            if any(r is not None for r in res): print("B", B, "grp", gi, "first mismatch per row", res, [pool[i][0] for i in idxs], flush=True)
            del steps
            torch.cuda.empty_cache()
        print("done B", B, flush=True)
    return j
