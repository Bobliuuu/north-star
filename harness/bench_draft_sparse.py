"""Graphed draft-step COST with real sparse attention (gather) on North. Timing only; acceptance is in results/approx_sweep_*.
Cost-faithful: gather+attend work and (oracle) selection scan are real kernels; index values are not content-selected for 'fixed'."""
import torch
from common import Jsonl, build_ids, by_split, load_workloads
from northstar.refmodel import RefNorth
from graph_north import GraphNorth, time_graph


@torch.no_grad()
def run(m, tok, ctxs=(2048, 8192), ks=(256, 1024)):
    ref = RefNorth(m); j = Jsonl()
    code = [r for r in by_split(load_workloads(), "dev") if r["kind"] == "code"]
    for ctx in ctxs:
        ids = build_ids(tok, code[0], ctx); P = len(ids)
        gn = GraphNorth(ref, P + 64, "bmm"); kv = gn.new_kv(); gn.eager_prefill(ids[:-1], kv)
        cells = []
        for r in (None, 4):
            cells.append((f"base_r{r}", None, r))
            cells.append((f"slwin_r{r}", ("slwin", 0), r))
            for K in ks:
                cells += [(f"fixed{K}_r{r}", ("fixed", K), r), (f"oracle{K}_r{r}", ("oracle", K), r)]
        for name, sp, r in cells:
            s = gn.make_step(1, kv, r=r, sparse=sp); s.set([ids[-1]], P - 1)
            med, mn, mx = time_graph(s, n=50)
            j.add(phase="draft_sparse", ctx=P, mode=name, graph_ms=med, ms_min=mn, ms_max=mx)
            print("draft_sparse", P, name, round(med, 3), flush=True); del s
        del kv, gn; torch.cuda.empty_cache()
    return j
