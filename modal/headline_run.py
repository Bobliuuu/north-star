"""Same-launch headline re-measurement (harness/bench_headline.py), H100.   modal run --detach modal/headline_run.py::main --tag hl1"""
import sys, time, json, pathlib
sys.path.insert(0, "modal")
import modal
from nsapp import app, weights_vol, results_vol, WEIGHTS, MODEL_DIR


@app.function(gpu="H100", timeout=2400, volumes={WEIGHTS: weights_vol, "/results": results_vol}, memory=65536, scaledown_window=2, max_containers=1)
def remote(tag: str):
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/harness")
    from loader import load_model
    from common import env_meta
    import bench_headline
    t0 = time.time()
    m, tok = load_model(MODEL_DIR, "grouped_mm")
    meta = env_meta(); meta["load_s"] = time.time() - t0
    print("META", json.dumps(meta), flush=True)
    j = bench_headline.run(m, tok)
    text = j.text(); secs = time.time() - t0
    pathlib.Path(f"/results/headline_{tag}.jsonl").write_text(text)
    pathlib.Path(f"/results/headline_{tag}.meta.json").write_text(json.dumps(dict(meta, secs=secs)))
    results_vol.commit()
    print("HL_DONE secs", secs, "est usd", secs / 3600 * 3.95, flush=True)
    return text


@app.local_entrypoint()
def main(tag: str = "hl1"):
    text = remote.remote(tag)
    pathlib.Path(f"results/headline_{tag}.jsonl").write_text(text)
    print("wrote", f"results/headline_{tag}.jsonl")
