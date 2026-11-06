"""real-traffic PLD bench, H100.   modal run --detach modal/realtraffic_run.py::main --tag rt1 --limit 0
Result JSONL -> Volume northstar-results:/realtraffic_<tag>.jsonl ; fetch: modal volume get northstar-results realtraffic_<tag>.jsonl results/"""
import sys, time, json, pathlib
sys.path.insert(0, "modal")
import modal
from nsapp import app, weights_vol, results_vol, WEIGHTS, MODEL_DIR


@app.function(gpu="H100", timeout=3300, volumes={WEIGHTS: weights_vol, "/results": results_vol}, memory=65536, scaledown_window=2, max_containers=1)
def remote(tag: str, n_new: int, limit: int):
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/harness")
    from loader import load_model
    from common import env_meta
    import bench_realtraffic
    t0 = time.time()
    m, tok = load_model(MODEL_DIR, "grouped_mm")
    meta = env_meta(); meta["load_s"] = time.time() - t0
    print("META", json.dumps(meta), flush=True)
    j = bench_realtraffic.run(m, tok, tag, n_new, limit)
    text = j.text(); secs = time.time() - t0
    pathlib.Path(f"/results/realtraffic_{tag}.jsonl").write_text(text)
    pathlib.Path(f"/results/realtraffic_{tag}.meta.json").write_text(json.dumps(dict(meta, secs=secs)))
    results_vol.commit()
    print("REALTRAFFIC_DONE secs", secs, "est usd", secs / 3600 * 3.95, flush=True)
    return dict(secs=secs, rows=len(text.splitlines()))


@app.local_entrypoint()
def main(tag: str = "rt1", n_new: int = 192, limit: int = 0):
    print(remote.remote(tag, n_new, limit))
