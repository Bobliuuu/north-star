"""Exact-mode PLD + fast plain at 8K/32K/128K on H100.   modal run --detach modal/longexact_run.py::main --tag le1
Rows -> Volume northstar-results:/longexact_<tag>.jsonl (committed after every row); fetch: modal volume get northstar-results longexact_<tag>.jsonl results/"""
import sys, time, json, pathlib
sys.path.insert(0, "modal")
import modal
from nsapp import app, weights_vol, results_vol, WEIGHTS, MODEL_DIR


@app.function(gpu="H100", timeout=7200, volumes={WEIGHTS: weights_vol, "/results": results_vol}, memory=65536, scaledown_window=2, max_containers=1)
def remote(tag: str, ctxs: str, reps: int, n_new: int, old: bool):
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/harness")
    from loader import load_model
    from common import env_meta
    import bench_longexact
    t0 = time.time()
    path = pathlib.Path(f"/results/longexact_{tag}.jsonl")
    f = open(path, "a")

    def emit(**kw):
        kw["t_s"] = round(time.time() - t0, 1)
        s = json.dumps(kw, default=float, sort_keys=True)
        print("LEROW", s, flush=True); f.write(s + "\n"); f.flush(); results_vol.commit()

    m, tok = load_model(MODEL_DIR, "grouped_mm")
    meta = env_meta(); meta["load_s"] = time.time() - t0
    emit(kind="meta", meta=meta)
    bench_longexact.run(m, tok, emit, ctxs=tuple(int(c) for c in ctxs.split(",")), n_new=n_new, reps=reps, old=old)
    secs = time.time() - t0
    emit(kind="done", secs=secs)
    print("LONGEXACT_DONE secs", secs, "est usd", secs / 3600 * 3.95, flush=True)
    return secs


@app.local_entrypoint()
def main(tag: str = "le1", ctxs: str = "8192,32768,131072", reps: int = 2, n_new: int = 192, old: bool = True):
    print(remote.remote(tag, ctxs, reps, n_new, old))
