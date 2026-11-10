"""Real-North A/B of the MoE / linear tuning, H100.   modal run --detach modal/ab_moetune_run.py::main --tag mt1
Result JSONL -> Volume northstar-results:/abmoetune_<tag>.jsonl ; fetch: modal volume get northstar-results abmoetune_<tag>.jsonl results/"""
import sys, time, json, pathlib
sys.path.insert(0, "modal")
import modal
from nsapp import app, weights_vol, results_vol, WEIGHTS, MODEL_DIR


@app.function(gpu="H100", timeout=3300, volumes={WEIGHTS: weights_vol, "/results": results_vol}, memory=65536, scaledown_window=2, max_containers=1)
def remote(tag: str, sizes: str, n_new: int):
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/harness")
    from loader import load_model
    from common import env_meta
    import ab_moetune
    t0 = time.time()
    m, tok = load_model(MODEL_DIR, "grouped_mm")
    meta = env_meta(); meta["load_s"] = time.time() - t0
    print("META", json.dumps(meta), flush=True)
    j = ab_moetune.run(m, tok, tuple(int(x) for x in sizes.split(",")), n_new=n_new)
    text = j.text(); secs = time.time() - t0
    pathlib.Path(f"/results/abmoetune_{tag}.jsonl").write_text(text)
    pathlib.Path(f"/results/abmoetune_{tag}.meta.json").write_text(json.dumps(dict(meta, secs=secs)))
    results_vol.commit()
    print("ABMOETUNE_DONE secs", secs, "est usd", secs / 3600 * 3.95, flush=True)
    return dict(secs=secs, rows=len(text.splitlines()))


@app.local_entrypoint()
def main(tag: str = "mt1", sizes: str = "1,2,4,8,16,32", n_new: int = 64):
    print(remote.remote(tag, sizes, n_new))
