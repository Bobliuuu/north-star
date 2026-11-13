"""batch>1 exact PLD bench, H100.   modal run --detach modal/batchspec_run.py::main --tag bs1
Result JSONL -> Volume northstar-results:/batchspec_<tag>.jsonl ; fetch: modal volume get northstar-results batchspec_<tag>.jsonl results/"""
import sys, time, json, pathlib
sys.path.insert(0, "modal")
import modal
from nsapp import app, weights_vol, results_vol, WEIGHTS, MODEL_DIR


@app.function(gpu="H100", timeout=3300, volumes={WEIGHTS: weights_vol, "/results": results_vol}, memory=65536, scaledown_window=2, max_containers=1)
def remote(tag: str, n_new: int, reps: int, sizes: str, tilegrid: int = 6, cc: int = 0, streams: int = 0, tuned: int = 0):
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/harness")
    from loader import load_model
    from common import env_meta
    import bench_batch, fast_kernels_batch
    fast_kernels_batch.TILE_GRID = int(tilegrid)
    import batch_exact, fast_kernels_x2
    batch_exact.STREAMS = bool(streams)   # exact v2: attention || MoE streams (branch perf/exact-decode)
    if tuned:
        import fast_kernels_exact as fx
        fx.LIN_CFG.update(fast_kernels_x2.TUNED_LIN_CFG)
    t0 = time.time()
    m, tok = load_model(MODEL_DIR, "grouped_mm")
    meta = env_meta(); meta["load_s"] = time.time() - t0; meta["moe_tile_grid"] = int(tilegrid); meta["streams"] = int(streams); meta["tuned"] = int(tuned)
    print("META", json.dumps(meta), flush=True)
    j = bench_batch.run(m, tok, tag, reps, n_new, tuple(int(x) for x in sizes.split(",")), cc=bool(cc))
    text = j.text(); secs = time.time() - t0
    pathlib.Path(f"/results/batchspec_{tag}.jsonl").write_text(text)
    pathlib.Path(f"/results/batchspec_{tag}.meta.json").write_text(json.dumps(dict(meta, secs=secs)))
    results_vol.commit()
    print("BATCHSPEC_DONE secs", secs, "est usd", secs / 3600 * 3.95, flush=True)
    return dict(secs=secs, rows=len(text.splitlines()))


@app.local_entrypoint()
def main(tag: str = "bs1", n_new: int = 192, reps: int = 2, sizes: str = "1,2,4,8,16,32", tilegrid: int = 6, cc: int = 0, streams: int = 0, tuned: int = 0):
    print(remote.remote(tag, n_new, reps, sizes, tilegrid, cc, streams, tuned))
