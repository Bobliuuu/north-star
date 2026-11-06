"""Modal runners. All GPU functions: A100-80GB, hard timeouts, results returned + saved locally under results/.
  modal run modal/run.py::probe        env + HF/RefNorth sanity (short)
  modal run modal/run.py::baseline     Phase 0.2  HF bf16 greedy sweep  (batch x ctx)
  modal run modal/run.py::ceiling      Phase 0.3  teacher-forced block-verify ceiling + component timing
  modal run modal/run.py::spec         Phase 1    self-speculation on real North
  modal run modal/run.py::tests        CPU: tiny-fixture unit tests
"""
import sys, time, pathlib
sys.path.insert(0, "modal")
import modal
from nsapp import app, weights_vol, results_vol, WEIGHTS, MODEL_DIR, GPU

RES = pathlib.Path("results")
GPUKW = dict(gpu=GPU, volumes={WEIGHTS: weights_vol}, memory=65536)


def _setup():
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/harness")


def _timed(fn):
    t = time.time()
    out = fn()
    out["_gpu_seconds"] = time.time() - t
    return out


@app.function(**GPUKW, timeout=1500)
def probe_remote():
    _setup()
    def go():
        import torch, json, subprocess
        from common import env_meta, build_ids, load_workloads, MODEL_REV
        from loader import load_model
        out = {"meta": env_meta()}
        out["freeze"] = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode()
        t = time.time()
        m, tok = load_model(MODEL_DIR, "grouped_mm")
        out["load_s"] = time.time() - t
        out["experts_impl"] = str(getattr(m.config, "_experts_implementation", None))
        out["attn_impl"] = str(m.config._attn_implementation)
        out["mem_gb_after_load"] = torch.cuda.memory_allocated() / 2**30
        rows = load_workloads()
        ids = build_ids(tok, rows[0])
        out["prompt_len"] = len(ids)
        # HF greedy reference
        x = torch.tensor([ids], device="cuda")
        with torch.no_grad():
            hf = m.generate(x, max_new_tokens=32, do_sample=False)[0, len(ids):].tolist()
        from northstar.refmodel import RefNorth
        from northstar.spec import plain_greedy
        ref = RefNorth(m)
        pg = plain_greedy(ref, ids, 32)
        out["hf_tokens"], out["ref_tokens"] = hf, pg["tokens"]
        out["ref_hf_match_prefix"] = next((i for i, (a, b) in enumerate(zip(hf, pg["tokens"])) if a != b), min(len(hf), len(pg["tokens"])))
        out["ref_step_ms"] = [s * 1000 for s in pg["step_s"]]
        out["text"] = tok.decode(hf)
        # logits agreement on prompt
        cache = ref.new_cache(1, len(ids) + 8)
        lg = ref.prefill(x, cache, logits="last")
        with torch.no_grad():
            hl = m(x, logits_to_keep=1).logits
        out["logit_maxabs_diff"] = (lg.float() - hl.float()).abs().max().item()
        out["logit_argmax_equal"] = bool(lg.argmax() == hl.argmax())
        out["layer_types_full_idx"] = [i for i, t in enumerate(m.config.layer_types) if t == "full_attention"]
        return out
    return _timed(go)


@app.local_entrypoint()
def probe():
    import json
    r = probe_remote.remote()
    RES.mkdir(exist_ok=True)
    (RES / "probe.json").write_text(json.dumps(r, indent=1, default=str))
    sys.path.insert(0, "harness")
    from common import append_spend
    append_spend(RES, "probe", "A100-80GB", r["_gpu_seconds"])
    r.pop("freeze")
    print(json.dumps(r, indent=1, default=str)[:6000])


@app.function(**GPUKW, timeout=2400)
def baseline_remote(quick: bool = False):
    _setup()
    def go():
        from common import env_meta
        from loader import load_model
        import bench_baseline
        m, tok = load_model(MODEL_DIR, "grouped_mm")
        j = bench_baseline.run(m, tok, quick)
        return {"jsonl": j.text(), "meta": env_meta()}
    return _timed(go)


@app.function(**GPUKW, timeout=3000)
def ceiling_remote(quick: bool = False):
    _setup()
    def go():
        from common import env_meta
        from loader import load_model
        import bench_ceiling
        m, tok = load_model(MODEL_DIR, "grouped_mm")
        j = bench_ceiling.run(m, tok, quick)
        return {"jsonl": j.text(), "meta": env_meta()}
    return _timed(go)


@app.function(**GPUKW, timeout=3000)
def spec_remote(quick: bool = False):
    _setup()
    def go():
        from common import env_meta
        from loader import load_model
        import bench_spec
        m, tok = load_model(MODEL_DIR, "grouped_mm")
        j = bench_spec.run(m, tok, quick=quick)
        return {"jsonl": j.text(), "meta": env_meta()}
    return _timed(go)


@app.function(**GPUKW, timeout=1800)
def routing_remote(quick: bool = False):
    _setup()
    def go():
        import json
        from common import env_meta
        from loader import load_model
        import bench_routing
        m, tok = load_model(MODEL_DIR, "grouped_mm")
        j, traces = bench_routing.run(m, tok, n_new=16 if quick else 48)
        return {"jsonl": j.text(), "meta": env_meta(), "traces": json.dumps(traces)}
    return _timed(go)


@app.function(timeout=1200, cpu=4)
def tests_remote():
    import subprocess
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "/root/tests"], capture_output=True, text=True,
                       env={**__import__("os").environ, "PYTHONPATH": "/root:/root/tests:/root/northstar/.."})
    return r.stdout[-3000:] + r.stderr[-1000:]


def _finish(name, r, quick):
    import json
    sys.path.insert(0, "harness")
    from common import append_spend
    RES.mkdir(exist_ok=True)
    tag = name + ("_quick" if quick else "")
    (RES / f"{tag}.jsonl").write_text(r["jsonl"])
    (RES / f"{tag}.meta.json").write_text(json.dumps(r["meta"], indent=1))
    append_spend(RES, tag, "A100-80GB", r["_gpu_seconds"])
    print(f"wrote results/{tag}.jsonl  gpu_s={r['_gpu_seconds']:.0f} est=${r['_gpu_seconds']*0.000694:.2f}")


@app.local_entrypoint()
def baseline(quick: bool = False):
    _finish("baseline", baseline_remote.remote(quick), quick)


@app.local_entrypoint()
def ceiling(quick: bool = False):
    _finish("ceiling", ceiling_remote.remote(quick), quick)


@app.local_entrypoint()
def spec(quick: bool = False):
    _finish("spec", spec_remote.remote(quick), quick)


@app.local_entrypoint()
def routing(quick: bool = False):
    r = routing_remote.remote(quick)
    RES.mkdir(exist_ok=True)
    (RES / "routing_traces.json").write_text(r.pop("traces"))
    _finish("routing", r, quick)


@app.local_entrypoint()
def tests():
    print(tests_remote.remote())
