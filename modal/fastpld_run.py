"""PLD on the fast executor, H100.   modal run --detach modal/fastpld_run.py::main --tag fp1 --sections vc
Result JSONL -> Volume northstar-results:/fastpld_<tag>.jsonl ; fetch: modal volume get northstar-results fastpld_<tag>.jsonl results/"""
import sys, time, json, pathlib
sys.path.insert(0, "modal")
import modal
from nsapp import app, weights_vol, results_vol, WEIGHTS, MODEL_DIR


@app.function(gpu="H100", timeout=3300, volumes={WEIGHTS: weights_vol, "/results": results_vol}, memory=65536, scaledown_window=2, max_containers=1)
def remote(tag: str, sections: str, reps: int, hf: bool, live_set: str, n_new: int):
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/harness")
    from loader import load_model
    from common import env_meta
    import bench_fastpld
    t0 = time.time()
    m, tok = load_model(MODEL_DIR, "grouped_mm")
    meta = env_meta(); meta["load_s"] = time.time() - t0
    print("META", json.dumps(meta), flush=True)
    j = bench_fastpld.run(m, tok, sections, tag, reps, hf, live_set, n_new)
    text = j.text(); secs = time.time() - t0
    pathlib.Path(f"/results/fastpld_{tag}.jsonl").write_text(text)
    pathlib.Path(f"/results/fastpld_{tag}.meta.json").write_text(json.dumps(dict(meta, secs=secs)))
    results_vol.commit()
    print("FASTPLD_DONE secs", secs, "est usd", secs / 3600 * 3.95, flush=True)
    return dict(secs=secs, rows=len(text.splitlines()))


@app.local_entrypoint()
def main(tag: str = "fp1", sections: str = "vc", reps: int = 2, hf: bool = True, live_set: str = "", n_new: int = 192):
    print(remote.remote(tag, sections, reps, hf, live_set, n_new))
