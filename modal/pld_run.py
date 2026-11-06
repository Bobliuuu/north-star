"""Zero-cost drafting (PLD / suffix / n-gram cache) live on H100 (graph capture needs H100: grouped_mm).
  modal run --detach modal/pld_run.py::main --tag pld1
Result JSONL -> Volume northstar-results:/pld_<tag>.jsonl. Fetch: modal volume get northstar-results pld_<tag>.jsonl results/"""
import sys, time, json, pathlib
sys.path.insert(0, "modal")
import modal
from nsapp import app, weights_vol, results_vol, WEIGHTS, MODEL_DIR


@app.function(gpu="H100", timeout=2700, volumes={WEIGHTS: weights_vol, "/results": results_vol}, memory=65536, scaledown_window=2, max_containers=1)
def pld_remote(tag: str = "pld1", n_new: int = 192, ctxs=(2048, 8192)):
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/harness")
    from loader import load_model
    from common import env_meta
    import bench_pld
    t0 = time.time()
    m, tok = load_model(MODEL_DIR, "grouped_mm")
    meta = env_meta(); meta["load_s"] = time.time() - t0
    print("META", json.dumps(meta), flush=True)
    from common import Jsonl
    j = Jsonl()
    try:
        bench_pld.run(m, tok, n_new=n_new, ctxs=tuple(ctxs), emit=lambda *a: print(*a, flush=True), j=j)
    except Exception:
        import traceback; print("PLD FAILED", traceback.format_exc(), flush=True)
    text = j.text()
    secs = time.time() - t0
    pathlib.Path(f"/results/pld_{tag}.jsonl").write_text(text)
    pathlib.Path(f"/results/pld_{tag}.meta.json").write_text(json.dumps(dict(meta, secs=secs)))
    results_vol.commit()
    print("PLD_DONE secs", secs, "est usd", secs / 3600 * 3.95, flush=True)
    return dict(secs=secs, rows=len(text.splitlines()))


@app.local_entrypoint()
def main(tag: str = "pld1", n_new: int = 192):
    print(pld_remote.remote(tag, n_new))
