"""modal run --detach modal/longctx.py::main --part A|B|C --tag x     (H100-80GB; graphed long-ctx draft/verify cost + batch>1 step cost)
Rows are printed as 'LCROW {json}' and appended to Volume northstar-results:/longctx_<tag>.jsonl after every cell."""
import json, pathlib, sys, time
sys.path.insert(0, "modal")
import modal
from nsapp import app, image, results_vol, weights_vol, WEIGHTS, MODEL_DIR


@app.function(image=image, gpu="H100", volumes={WEIGHTS: weights_vol, "/results": results_vol}, timeout=3300, memory=65536)
def run_remote(tag: str, part: str, ctxs: str):
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/harness")
    import torch
    from loader import load_model
    from common import env_meta
    import bench_longctx as bl
    t = time.time()
    path = pathlib.Path(f"/results/longctx_{tag}.jsonl"); path.parent.mkdir(exist_ok=True)
    f = open(path, "a")

    def emit(**kw):
        kw["t_s"] = round(time.time() - t, 1)
        s = json.dumps(kw, default=float)
        print("LCROW", s, flush=True); f.write(s + "\n"); f.flush(); results_vol.commit()

    emit(kind="meta", meta=env_meta(), part=part)
    if "K" in part:
        bl.run_kernel(emit)
    if not any(c in part for c in "ABT"):
        emit(kind="done", gpu_seconds=time.time() - t)
        return time.time() - t
    m, tok = load_model(MODEL_DIR, "grouped_mm")
    emit(kind="loaded", load_s=time.time() - t)
    if "A" in part:
        bl.check_vs_old(m, tok, emit)
        bl.run_long(m, tok, emit, ctxs=tuple(int(c) for c in ctxs.split(",")))
    if "T" in part:
        bl.run_long_tuned(m, tok, emit)
    if "B" in part:
        bl.run_batch(m, tok, emit)
    emit(kind="done", gpu_seconds=time.time() - t)
    return time.time() - t


@app.local_entrypoint()
def main(tag: str = "lc1", part: str = "A", ctxs: str = "8192,32768,131072"):
    print("GPU_SECONDS", run_remote.remote(tag, part, ctxs))
