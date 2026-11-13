"""Long-context exact attention sweep (harness/bench_lcx2.py), H100.   modal run --detach modal/lcx2_run.py::main --tag lc1
Rows -> Volume northstar-results:/lcx2_<tag>.jsonl and results/lcx2_<tag>.jsonl."""
import sys, time, json, pathlib
sys.path.insert(0, "modal")
import modal
from nsapp import app, weights_vol, results_vol, WEIGHTS, MODEL_DIR


@app.function(gpu="H100", timeout=3500, volumes={WEIGHTS: weights_vol, "/results": results_vol}, memory=65536, scaledown_window=2, max_containers=1)
def remote(tag: str, ctxs: str):
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/harness")
    from loader import load_model
    from common import env_meta
    import bench_lcx2
    t0 = time.time()
    m, tok = load_model(MODEL_DIR, "grouped_mm")
    meta = env_meta(); meta["load_s"] = time.time() - t0
    print("META", json.dumps(meta), flush=True)
    j = bench_lcx2.run(m, tok, tuple(int(c) for c in ctxs.split(",")))
    text = j.text(); secs = time.time() - t0
    pathlib.Path(f"/results/lcx2_{tag}.jsonl").write_text(text)
    results_vol.commit()
    print("LCX2_DONE secs", secs, "est usd", secs / 3600 * 3.95, flush=True)
    return text


@app.local_entrypoint()
def main(tag: str = "lc1", ctxs: str = "32768,131072"):
    text = remote.remote(tag, ctxs)
    pathlib.Path(f"results/lcx2_{tag}.jsonl").write_text(text)
    print("wrote", f"results/lcx2_{tag}.jsonl")
