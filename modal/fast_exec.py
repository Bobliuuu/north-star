"""modal run --detach modal/fast_exec.py::main --tag fx1 [--sections tc]   (H100: profile + ablation timing + token-identity)"""
import json, pathlib, sys, time
sys.path.insert(0, "modal")
import modal
from nsapp import app, image, results_vol, weights_vol, WEIGHTS, MODEL_DIR


@app.function(image=image, gpu="H100", volumes={WEIGHTS: weights_vol, "/results": results_vol}, timeout=3300, memory=65536)
def run_remote(tag: str, sections: str):
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/harness")
    from loader import load_model
    from common import env_meta
    import bench_fast
    t = time.time()
    m, tok = load_model(MODEL_DIR, "grouped_mm")
    pathlib.Path("/results").mkdir(exist_ok=True)
    j = bench_fast.run(m, tok, sections)
    res = dict(meta=env_meta(), gpu_seconds=time.time() - t, jsonl=j.text())
    pathlib.Path(f"/results/fast_exec_{tag}.json").write_text(json.dumps(res, default=float))
    results_vol.commit()
    return res


@app.local_entrypoint()
def main(tag: str = "fx1", sections: str = "tc"):
    res = run_remote.remote(tag, sections)
    pathlib.Path("results").mkdir(exist_ok=True)
    pathlib.Path(f"results/fast_exec_{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    pathlib.Path(f"results/fast_exec_{tag}.jsonl").write_text(res["jsonl"])
    sys.path.insert(0, "harness")
    from common import append_spend
    append_spend(pathlib.Path("results"), f"fast_exec_{tag}", "H100", res["gpu_seconds"])
    print(res["jsonl"])
