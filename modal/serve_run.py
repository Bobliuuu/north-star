"""Exact serving engine validation (harness/bench_serve.py), H100.
    modal run --detach modal/serve_run.py::main --tag sv1 [--sections pool,mix,long] [--limit 0]
Rows -> Volume northstar-results:/serve_<tag>.jsonl and results/serve_<tag>.jsonl (local)."""
import sys, time, json, pathlib
sys.path.insert(0, "modal")
import modal
from nsapp import app, weights_vol, results_vol, WEIGHTS, MODEL_DIR


@app.function(gpu="H100", timeout=5400, volumes={WEIGHTS: weights_vol, "/results": results_vol}, memory=65536, scaledown_window=2, max_containers=1)
def remote(tag: str, sections: str, limit: int):
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/harness")
    from loader import load_model
    from common import env_meta
    import bench_serve
    t0 = time.time()
    m, tok = load_model(MODEL_DIR, "grouped_mm")
    meta = env_meta(); meta["load_s"] = time.time() - t0
    print("META", json.dumps(meta), flush=True)
    j = bench_serve.run(m, tok, tuple(sections.split(",")), limit=limit)
    text = j.text(); secs = time.time() - t0
    pathlib.Path(f"/results/serve_{tag}.jsonl").write_text(text)
    pathlib.Path(f"/results/serve_{tag}.meta.json").write_text(json.dumps(dict(meta, secs=secs)))
    results_vol.commit()
    print("SERVE_DONE secs", secs, "est usd", secs / 3600 * 3.95, flush=True)
    return text


@app.local_entrypoint()
def main(tag: str = "sv1", sections: str = "pool,mix,long", limit: int = 0):
    text = remote.remote(tag, sections, limit)
    pathlib.Path(f"results/serve_{tag}.jsonl").write_text(text)
    print("wrote", f"results/serve_{tag}.jsonl")
