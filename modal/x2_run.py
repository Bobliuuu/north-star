"""Exact-v2 decode bench on the real model, H100 (harness/bench_x2.py).
    modal run --detach modal/x2_run.py::main --tag x1 --phases step,traffic --cfg results/x2_cfg.json
Result JSONL -> Volume northstar-results:/x2_<tag>.jsonl ; fetch: modal volume get northstar-results x2_<tag>.jsonl results/"""
import sys, time, json, pathlib
sys.path.insert(0, "modal")
import modal
from nsapp import app, weights_vol, results_vol, WEIGHTS, MODEL_DIR


@app.function(gpu="H100", timeout=3500, volumes={WEIGHTS: weights_vol, "/results": results_vol}, memory=65536, scaledown_window=2, max_containers=1)
def remote(tag: str, phases: str, cfg_json: str, limit: int, extra_json: str = "{}"):
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/harness")
    from loader import load_model
    from common import env_meta
    import bench_x2
    cfg = json.loads(cfg_json)
    lin_cfg = {tuple(int(v) for v in k.split("x")): c for k, c in cfg.get("lin", {}).items()}
    moe_cfg = cfg.get("moe") or None
    t0 = time.time()
    m, tok = load_model(MODEL_DIR, "grouped_mm")
    meta = env_meta(); meta["load_s"] = time.time() - t0
    print("META", json.dumps(meta), flush=True)
    extra = {}
    for name, v in json.loads(extra_json).items():
        v = dict(v)
        if "lin_cfg" in v:
            v["lin_cfg"] = {tuple(int(q) for q in k.split("x")): c for k, c in v["lin_cfg"].items()}
        extra[name] = v
    cand = cfg.get("cand")
    if cand is not None:
        cand = dict(cand, lin_cfg=lin_cfg, moe_cfg=moe_cfg)
    j = bench_x2.run(m, tok, phases.split(","), lin_cfg, moe_cfg, limit, extra, cand)
    text = j.text(); secs = time.time() - t0
    pathlib.Path(f"/results/x2_{tag}.jsonl").write_text(text)
    pathlib.Path(f"/results/x2_{tag}.meta.json").write_text(json.dumps(dict(meta, secs=secs, cfg=cfg)))
    results_vol.commit()
    print("X2_DONE secs", secs, "est usd", secs / 3600 * 3.95, flush=True)
    return text


@app.local_entrypoint()
def main(tag: str = "x1", phases: str = "step,traffic", cfg: str = "results/x2_cfg.json", limit: int = 0, extra: str = ""):
    text = remote.remote(tag, phases, pathlib.Path(cfg).read_text(), limit, pathlib.Path(extra).read_text() if extra else "{}")
    pathlib.Path(f"results/x2_{tag}.jsonl").write_text(text)
    print("wrote", f"results/x2_{tag}.jsonl")
