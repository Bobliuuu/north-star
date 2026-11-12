"""Prefill profile (harness/prof_prefill.py), H100.   modal run --detach modal/prof_prefill_run.py::main --tag pf1"""
import sys, time, json, pathlib
sys.path.insert(0, "modal")
import modal
from nsapp import app, weights_vol, results_vol, WEIGHTS, MODEL_DIR


@app.function(gpu="H100", timeout=2400, volumes={WEIGHTS: weights_vol, "/results": results_vol}, memory=65536, scaledown_window=2, max_containers=1)
def remote(tag: str, what: str = "prefill"):
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/harness")
    from loader import load_model
    import prof_prefill, probe_inv
    t0 = time.time()
    m, tok = load_model(MODEL_DIR, "grouped_mm")
    j = (probe_inv if what == "probe" else prof_prefill).run(m, tok)
    text = j.text()
    pathlib.Path(f"/results/prefill_{tag}.jsonl").write_text(text); results_vol.commit()
    print("PF_DONE secs", time.time() - t0, flush=True)
    return text


@app.local_entrypoint()
def main(tag: str = "pf1", what: str = "prefill"):
    text = remote.remote(tag, what)
    pathlib.Path(f"results/prefill_{tag}.jsonl").write_text(text)
