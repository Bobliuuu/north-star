"""Our batched exact executor (exact v2: streams + tuned tiles), plain decode on the same pool groups as bench_vllm_batch (first B prompts),
192 tokens, 2 reps, best. Usage: python run_bx2_plain.py <model_dir> <out.jsonl> [sizes]"""
import json, pathlib, sys, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))


def main():
    model_dir, out = sys.argv[1], pathlib.Path(sys.argv[2])
    sizes = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,2,4,8,16,32").split(",")]
    import torch
    from loader import load_model
    import bench_batch as bb, batch_exact, fast_kernels_batch as fkb, fast_kernels_exact as fx, fast_kernels_x2 as f2
    from northstar.refmodel import RefNorth
    from graph_north import GraphNorth
    from fast_north import FastNorth
    fkb.TILE_GRID = 6; batch_exact.STREAMS = True; fx.LIN_CFG.update(f2.TUNED_LIN_CFG)
    try:
        m, tok = load_model(model_dir, "grouped_mm")
    except TypeError:
        m, tok = load_model(model_dir, None)
    eos = m.config.eos_token_id; eos = eos if isinstance(eos, int) else eos[0]
    pool = bb.build_pool(tok, 32)
    ref = RefNorth(m)
    Lmax = max(len(p[2]) for p in pool) + 232
    X = FastNorth(ref, Lmax, exact=True); gb = GraphNorth(ref, Lmax, "bmm"); BX = batch_exact.BatchExact(X)
    f = out.open("a")
    f.write(json.dumps(dict(phase="meta", torch=torch.__version__, pool=[p[0] for p in pool])) + "\n"); f.flush()
    with torch.no_grad():
        kvs = []
        for (_, _, ids) in pool:
            kv = X.new_kv(); gb.eager_prefill(ids[:-1], kv); kvs.append(kv)
        for B in sizes:
            grp = pool[:B]
            step = BX.make_step(B, 1, kvs[:B])
            bb.run_plain(step, grp, 8, eos)
            best = 0
            for rep in range(2):
                o, td, wall, _ = bb.run_plain(step, grp, 192, eos)
                n = sum(len(x) for x in o)
                f.write(json.dumps(dict(phase="batch", B=B, rep=rep, n=n, wall=wall, agg_tok_s=n / wall, lens=[len(x) for x in o])) + "\n"); f.flush()
                best = max(best, n / wall)
            print("OURS_BATCH B", B, "agg %.0f tok/s" % best, flush=True)
            del step; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
