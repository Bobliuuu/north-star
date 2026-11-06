"""Modal entrypoint for the approximation sweep. NOT run by the author (another agent owns GPU runs).

  modal run src/approx/modal_sweep.py                                  # quick preset, groups full/attn/expert/combined
  modal run src/approx/modal_sweep.py --preset full --groups full,attn_only,expert_only,combined,layerskip
  modal run src/approx/modal_sweep.py --smoke true                     # 1 prompt, ctx 2048, 8 configs (sanity)

Reuses modal/nsapp.py (app, A100-80GB, volumes `northstar-weights` at /weights holding the pinned North checkpoint,
`northstar-results`).  Weights volume is read-only here: run `modal run modal/download.py` first if empty.
Results: written to /results/approx_sweep_<tag>.jsonl on the results volume AND returned + saved to
results/approx_sweep_<tag>.jsonl locally."""
import json, pathlib, sys, time

sys.path.insert(0, "modal")
import modal
from nsapp import app, image, results_vol, weights_vol, WEIGHTS, MODEL_DIR, GPU

approx_image = image.add_local_dir("src/approx", "/root/approx", copy=False)


@app.function(image=approx_image, gpu=GPU, volumes={WEIGHTS: weights_vol, "/results": results_vol},
              timeout=3 * 3600, memory=65536)
def sweep_remote(preset: str, groups: str, n_prompts: int, ctxs: str, n_new: int, tag: str, smoke: bool):
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/harness")
    from loader import load_model
    from common import env_meta
    from approx import sweep
    t = time.time()
    m, tok = load_model(MODEL_DIR, "grouped_mm")
    load_s = time.time() - t
    rows = sweep.run(m, tok, tuple(groups.split(",")), preset, n_prompts, tuple(int(c) for c in ctxs.split(",")),
                     n_new, max_configs=8 if smoke else None)
    res = dict(meta=env_meta(), load_s=load_s, gpu_seconds=time.time() - t, rows=rows, summary=sweep.summarize(rows))
    pathlib.Path("/results").mkdir(exist_ok=True)
    pathlib.Path(f"/results/approx_sweep_{tag}.json").write_text(json.dumps(res, default=float))
    results_vol.commit()
    return res


@app.local_entrypoint()
def main(preset: str = "quick", groups: str = "full,attn_only,expert_only,combined", n_prompts: int = 2,
         ctxs: str = "2048,8192", n_new: int = 96, tag: str = "", smoke: bool = False):
    tag = tag or time.strftime("%Y%m%d-%H%M")
    if smoke:
        n_prompts, ctxs, n_new = 1, "2048", 32
    res = sweep_remote.remote(preset, groups, n_prompts, ctxs, n_new, tag, smoke)
    out = pathlib.Path("results"); out.mkdir(exist_ok=True)
    (out / f"approx_sweep_{tag}.json").write_text(json.dumps(res, default=float, indent=1))
    sys.path.insert(0, "harness")
    from common import append_spend
    append_spend(out, f"approx_sweep_{tag}", GPU, res["gpu_seconds"])
    print(json.dumps(res["summary"]["best"], indent=1, default=float))
