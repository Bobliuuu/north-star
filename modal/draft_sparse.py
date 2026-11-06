"""modal run --detach modal/draft_sparse.py::main --tag ds1   (A100-80GB, graphed draft-step cost with real sparse-attention gather)"""
import json, pathlib, sys, time
sys.path.insert(0, "modal")
import modal
from nsapp import app, image, results_vol, weights_vol, WEIGHTS, MODEL_DIR, GPU


@app.function(image=image, gpu=GPU, volumes={WEIGHTS: weights_vol, "/results": results_vol}, timeout=3000, memory=65536)
def run_remote(tag: str, ctxs: str):
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/harness")
    from loader import load_model
    from common import env_meta
    import bench_draft_sparse
    t = time.time()
    m, tok = load_model(MODEL_DIR, "grouped_mm")
    txt = bench_draft_sparse.run(m, tok, ctxs=tuple(int(c) for c in ctxs.split(","))).text()
    res = dict(meta=env_meta(), gpu_seconds=time.time() - t, jsonl=txt)
    pathlib.Path("/results").mkdir(exist_ok=True)
    pathlib.Path(f"/results/draft_sparse_{tag}.json").write_text(json.dumps(res, default=float))
    results_vol.commit()
    return res


@app.local_entrypoint()
def main(tag: str = "ds1", ctxs: str = "2048,8192"):
    res = run_remote.remote(tag, ctxs)
    pathlib.Path("results").mkdir(exist_ok=True)
    pathlib.Path(f"results/draft_sparse_{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    pathlib.Path(f"results/draft_sparse_{tag}.jsonl").write_text(res["jsonl"])
    sys.path.insert(0, "harness")
    from common import append_spend
    append_spend(pathlib.Path("results"), f"draft_sparse_{tag}", GPU, res["gpu_seconds"])
    print(res["jsonl"])
