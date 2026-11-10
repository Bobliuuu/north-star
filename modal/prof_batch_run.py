"""per-kernel profile of the batched exact step, H100.   modal run --detach modal/prof_batch_run.py::main --tag p1
Result JSONL -> Volume northstar-results:/profbatch_<tag>.jsonl ; fetch: modal volume get northstar-results profbatch_<tag>.jsonl results/"""
import sys, time, json, pathlib
sys.path.insert(0, "modal")
import modal
from nsapp import app, weights_vol, results_vol, WEIGHTS, MODEL_DIR


@app.function(gpu="H100", timeout=3300, volumes={WEIGHTS: weights_vol, "/results": results_vol}, memory=65536, scaledown_window=2, max_containers=1)
def remote(tag: str, n_new: int, reps: int, sizes: str, tilegrid: int = 0):
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/harness")
    from loader import load_model
    from common import env_meta
    import prof_batch
    pass
    t0 = time.time()
    m, tok = load_model(MODEL_DIR, "grouped_mm")
    meta = env_meta(); meta["load_s"] = time.time() - t0
    print("META", json.dumps(meta), flush=True)
    j = prof_batch.run(m, tok, tuple(int(x) for x in sizes.split(",")))
    text = j.text(); secs = time.time() - t0
    pathlib.Path(f"/results/profbatch_{tag}.jsonl").write_text(text)
    pathlib.Path(f"/results/profbatch_{tag}.meta.json").write_text(json.dumps(dict(meta, secs=secs)))
    results_vol.commit()
    print("PROF_DONE secs", secs, "est usd", secs / 3600 * 3.95, flush=True)
    return dict(secs=secs, rows=len(text.splitlines()))


@app.local_entrypoint()
def main(tag: str = "bs1", n_new: int = 64, reps: int = 1, sizes: str = "1,8,16,32", tilegrid: int = 0):
    print(remote.remote(tag, n_new, reps, sizes, tilegrid))
