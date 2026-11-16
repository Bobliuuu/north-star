"""Race the sass-lane Triton port of moe_fused against north-star's Triton MoE on the decode path (harness/bench_moetriton.py).
  modal run --detach modal/moetriton_run.py::main --tag mt1 --gpu H100        (then B200 / RTX-PRO-6000)
Writes northstar-results:/moetriton_<tag>_<gpu>.jsonl (+ .meta.json)."""
import sys, time, json, pathlib
sys.path.insert(0, "modal")
import modal
from nsapp import image, weights_vol, results_vol, WEIGHTS, MODEL_DIR

app = modal.App("ship-moetriton", image=image)


def _body(tag, gpu, parts, variants=""):
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/harness")
    import os
    if "|" in variants:   # "BASES|VARIANTS": extra exact-v2 baselines, e.g. exact_v2@p0
        b, variants = variants.split("|", 1); os.environ["NS_TRI_BASES"] = b
    if variants.startswith("nostream:"):
        os.environ["NS_TRI_NOSTREAM"] = "1"; variants = variants[len("nostream:"):]
    if variants:
        os.environ["NS_TRI_VARIANTS"] = variants
    from loader import load_model
    from common import env_meta
    import bench_moetriton
    t0 = time.time()
    m, tok = load_model(MODEL_DIR, "grouped_mm")
    meta = env_meta(); meta["load_s"] = time.time() - t0
    j = bench_moetriton.run(m, tok, parts=parts)
    text = j.text(); secs = time.time() - t0
    stem = f"/results/moetriton_{tag}_{gpu}"
    pathlib.Path(stem + ".jsonl").write_text(text)
    pathlib.Path(stem + ".meta.json").write_text(json.dumps(dict(meta, secs=secs)))
    results_vol.commit()
    print("MT_DONE secs", secs, flush=True)
    return text


kw = dict(timeout=3600, volumes={WEIGHTS: weights_vol, "/results": results_vol}, memory=65536, scaledown_window=2, max_containers=1)


@app.function(gpu="H100", **kw)
def h100(tag: str, parts: str, variants: str = ""):
    return _body(tag, "h100", parts, variants)


@app.function(gpu="B200", **kw)
def b200(tag: str, parts: str, variants: str = ""):
    return _body(tag, "b200", parts, variants)


@app.function(gpu="RTX-PRO-6000", **kw)
def rtxpro(tag: str, parts: str, variants: str = ""):
    return _body(tag, "rtxpro6000", parts, variants)


@app.local_entrypoint()
def main(tag: str = "mt1", gpu: str = "H100", parts: str = "AB", variants: str = ""):
    fn = {"H100": h100, "B200": b200, "RTX-PRO-6000": rtxpro}[gpu]
    text = fn.remote(tag, parts, variants)
    pathlib.Path(f"results/moetriton_{tag}_{ {'H100': 'h100', 'B200': 'b200', 'RTX-PRO-6000': 'rtxpro6000'}[gpu]}.jsonl").write_text(text)
