"""Batched exact executor: streams / tuned tiles vs H20 path + live-loop breakdown, H100 (harness/bench_bx2.py).
    modal run --detach modal/bx2_run.py::main --tag b1
Result JSONL -> Volume northstar-results:/bx2_<tag>.jsonl and results/bx2_<tag>.jsonl (local)."""
import sys, time, json, pathlib
sys.path.insert(0, "modal")
import modal
from nsapp import app, weights_vol, results_vol, WEIGHTS, MODEL_DIR


@app.function(gpu="H100", timeout=3000, volumes={WEIGHTS: weights_vol, "/results": results_vol}, memory=65536, scaledown_window=2, max_containers=1)
def remote(tag: str, sizes: str, n_new: int):
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/harness")
    from loader import load_model
    from common import env_meta
    import bench_bx2
    t0 = time.time()
    m, tok = load_model(MODEL_DIR, "grouped_mm")
    meta = env_meta(); meta["load_s"] = time.time() - t0
    print("META", json.dumps(meta), flush=True)
    j = bench_bx2.run(m, tok, tuple(int(x) for x in sizes.split(",")), n_new)
    text = j.text(); secs = time.time() - t0
    pathlib.Path(f"/results/bx2_{tag}.jsonl").write_text(text)
    pathlib.Path(f"/results/bx2_{tag}.meta.json").write_text(json.dumps(dict(meta, secs=secs)))
    results_vol.commit()
    print("BX2_DONE secs", secs, "est usd", secs / 3600 * 3.95, flush=True)
    return text


@app.local_entrypoint()
def main(tag: str = "b1", sizes: str = "1,4,8,16,32", n_new: int = 96):
    text = remote.remote(tag, sizes, n_new)
    pathlib.Path(f"results/bx2_{tag}.jsonl").write_text(text)
    print("wrote", f"results/bx2_{tag}.jsonl")
