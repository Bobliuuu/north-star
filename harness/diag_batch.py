"""diagnose: (1) why T=1 batched graph is slower than T=2; (2) where plain B>1 diverges from plain B=1 (eager logits, teacher forced, then graph)."""
import time, json, traceback
import torch
from common import Jsonl
from northstar.refmodel import RefNorth
from graph_north import GraphNorth
from fast_north import FastNorth
from batch_exact import BatchExact
import bench_batch as bb


@torch.no_grad()
def run(m, tok):
    j = Jsonl(); ref = RefNorth(m); dev = ref.device
    eos = m.config.eos_token_id; eos = eos if isinstance(eos, int) else eos[0]
    pool = bb.build_pool(tok, 32)
    want = ["pr_hbm", "pr_reason", "go_worker", "humaneval_00_12"]
    idx = [i for i, p in enumerate(pool) if p[0] in want][:4]
    print("diag rows", [pool[i][0] for i in idx], flush=True)
    Lmax = max(len(p[2]) for p in pool) + 192 + 40
    X = FastNorth(ref, Lmax, exact=True); gb = GraphNorth(ref, Lmax, "bmm"); BX = BatchExact(X)
    kvs = {}
    for i in idx:
        kv = X.new_kv(); gb.eager_prefill(pool[i][2][:-1], kv); kvs[i] = kv
    N = 100
    # B=1 eager teacher trace
    toks, logit = {}, {}
    for i in idx:
        cur = pool[i][2][-1]; P = len(pool[i][2]); tk = []; lg = []
        for s in range(N):
            l = BX.forward(torch.tensor([[cur]], device=dev), torch.tensor([P - 1 + s], device=dev), [kvs[i]])
            lg.append(l[0].clone()); cur = int(l[0].argmax()); tk.append(cur)
        toks[i], logit[i] = tk, lg
    # B=len(idx) eager teacher forced with B1 tokens: compare logits bitwise
    for B in (2, 4):
        rows = idx[:B]
        first = {i: None for i in rows}
        for s in range(N):
            ids = torch.tensor([[pool[i][2][-1] if s == 0 else toks[i][s - 1]] for i in rows], device=dev)
            st = torch.tensor([len(pool[i][2]) - 1 + s for i in rows], device=dev)
            l = BX.forward(ids, st, [kvs[i] for i in rows])
            for k, i in enumerate(rows):
                if first[i] is None and not torch.equal(l[k], logit[i][s]): first[i] = (s, float((l[k].float() - logit[i][s].float()).abs().max()))
        j.add(phase="eager_tf", B=B, rows=[pool[i][0] for i in rows], first_mismatch=first); print("eager teacher-forced B", B, first, flush=True)
    # graph T=1 replay with B rows vs B1 tokens
    for B in (2, 4):
        rows = idx[:B]
        steps = BX.make_step(B, 1, [kvs[i] for i in rows])
        first = {i: None for i in rows}
        for s in range(N):
            ids = [[pool[i][2][-1] if s == 0 else toks[i][s - 1]] for i in rows]
            steps.set(ids, [len(pool[i][2]) - 1 + s for i in rows]); out = steps.run().tolist()
            for k, i in enumerate(rows):
                if first[i] is None and out[k][0] != toks[i][s]: first[i] = s
        j.add(phase="graph_tf", B=B, first_token_mismatch=first); print("graph teacher-forced B", B, first, flush=True)
        del steps
    # profile T=1 vs T=4 at B=4
    from torch.profiler import profile, ProfilerActivity
    rows = idx[:4]
    for T in (1, 4):
        ids = torch.zeros(4, T, dtype=torch.long, device=dev); st = torch.tensor([len(pool[i][2]) - 1 for i in rows], device=dev)
        for _ in range(2): BX.forward(ids, st, [kvs[i] for i in rows])
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            BX.forward(ids, st, [kvs[i] for i in rows]); torch.cuda.synchronize()
        tab = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10, max_name_column_width=60)
        print("PROFILE B=4 T=%d\n%s" % (T, tab), flush=True)
        j.add(phase="profile", T=T, table=tab)
    return j
