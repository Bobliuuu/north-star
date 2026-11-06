"""GPU evaluator + laptop-proof launcher for the 'triton' self-improve TRACK (launch parameters of the fast Triton decode kernels). docs/SELFIMPROVE_TRITON.md.

Same Loop / gates / sandbox / store / attribution as modal/selfimprove_eval.py (unchanged code in src/selfimprove); the evaluator runs on an H100
(network blocked), CUDA-graph timing. ALWAYS launch with `modal run --detach`.

  modal run --detach modal/selfimprove_triton.py::smoke_a10g                       # in-process check+quality+bench smoke, A10G (~$0.1), NO gates
  modal run --detach modal/selfimprove_triton.py::calib --tag c1                   # A/A evaluation through the real Evaluator on H100 (noise + baseline errors)
  modal run --detach modal/selfimprove_triton.py::remote --proposer mock --rounds 1 --store selfimprove_triton_dryrun      # dry run, mock proposer
  modal run --detach modal/selfimprove_triton.py::remote --proposer north --rounds 3 --per-round 3 --max-usd 4.5 --confirm-cost --store selfimprove_triton_v1
  modal run modal/selfimprove_triton.py::pull --store selfimprove_triton_v1
"""
import json, os, pathlib, subprocess, sys, time
import modal

ROOT = pathlib.Path(__file__).resolve().parents[1]
GPU = "H100"
USD_PER_HOUR = 3.95
APP_NAME = "northstar-selfimprove-triton"
IGN = ["**/__pycache__", "**/*.pyc", "**/.pytest_cache"]

gpu_image = (modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.10.0", "numpy==2.2.5")
             .env({"PYTHONUNBUFFERED": "1"})
             .add_local_dir(str(ROOT / "src/tritonk"), "/repo/src/tritonk", copy=False, ignore=IGN)
             .add_local_dir(str(ROOT / "src/selfimprove"), "/repo/src/selfimprove", copy=False, ignore=IGN + ["**/tests/**"])
             .add_local_dir(str(ROOT / "harness"), "/repo/harness", copy=False, ignore=IGN))
app = modal.App(APP_NAME, image=gpu_image)
triton_cache = modal.Volume.from_name("northstar-triton-cache", create_if_missing=True)   # persistent ~/.triton: kernels compile once across evaluations


def _cfg(repo, track, live=False):
    import runpy
    sys.path.insert(0, str(pathlib.Path(repo) / "src"))
    return runpy.run_path(str(pathlib.Path(repo) / "harness/selfimprove/config.py"))["make_config"](repo, track, require_measured_calibration=live)


@app.function(gpu=GPU, timeout=3300, cpu=8, memory=32768, block_network=True, max_containers=1, scaledown_window=60, volumes={"/root/.triton": triton_cache})
def evaluate_candidate(payload: dict) -> dict:
    """payload: {track, base_files, new_files}. Runs the Loop's own gated Evaluator in this network-blocked H100 container."""
    repo = "/repo"
    sys.path.insert(0, f"{repo}/src")
    from selfimprove.loop import Evaluator
    cfg = _cfg(repo, payload["track"])
    cfg.tmp_parent = pathlib.Path("/tmp")
    env = {}
    for k, cmd in (("gpu", ["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm,clocks.mem", "--format=csv,noheader"]),):
        try: env[k] = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[-1]
        except Exception as e: env[k] = f"unavailable: {e!r}"[:100]
    try:
        import triton, torch
        env["triton"], env["torch"] = triton.__version__, torch.__version__
    except Exception as e:
        env["triton"] = f"import failed: {e!r}"[:100]
    try:
        out = Evaluator(cfg).evaluate(payload["base_files"], payload["new_files"])
    finally:
        try: triton_cache.commit()
        except Exception: pass
    out["container_env"] = env
    return out


class RemoteEvaluator:
    """Drop-in for loop.Evaluator that runs evaluate() on the Modal GPU function."""
    def __init__(self, track, call=None):
        self.track, self.call = track, call or evaluate_candidate.remote

    def evaluate(self, base_files, new_files):
        t0 = time.time()
        try:
            out = self.call(dict(track=self.track, base_files=base_files, new_files=new_files))
        except Exception as e:
            return dict(verdict="evaluator_error", gates={}, sandbox={"error": repr(e)[:500]}, seeds={}, eval_wall_s=time.time() - t0)
        out.setdefault("eval_wall_s", time.time() - t0)
        return out


def _smoke_body(tag):
    """In-process (no sandbox, no gates): run check, quality and both bench splits with default tuning; print timings. For debugging the suite."""
    sys.path.insert(0, "/repo/src"); sys.path.insert(0, "/repo/harness/selfimprove")
    os.environ["NS_WORKSPACE"] = "/repo"
    import torch, triton_suite as TS
    out = dict(tag=tag, gpu=torch.cuda.get_device_name(), torch=torch.__version__)
    import triton; out["triton"] = triton.__version__
    t = time.time(); cases = TS.run_check(1234); out["check_s"] = time.time() - t
    out["check"] = [(c["name"], round(c["rel_l2"], 6), round(c["max_abs"], 5)) for c in cases]
    print("CHECK", json.dumps(out["check"]), flush=True)
    t = time.time(); items = TS.run_quality(); out["quality_s"] = time.time() - t
    out["quality_items"] = [round(x, 8) for x in items]
    print("QUALITY", json.dumps(out["quality_items"]), flush=True)
    S = TS.State(0)
    for split in ("dev", "heldout"):
        for rep in range(2):
            t = time.time(); r = TS.run_bench(split, S=S); r["wall_s"] = time.time() - t
            out[f"bench_{split}_{rep}"] = dict(time_ms=r["time_ms"], ref_ms=r["ref_ms"], wall_s=r["wall_s"], percase={k: round(v, 4) for k, v in r["percase"].items()})
            print("BENCH", split, rep, json.dumps(out[f"bench_{split}_{rep}"]), flush=True)
    try: triton_cache.commit()
    except Exception: pass
    return out


@app.function(gpu="A10G", timeout=1800, volumes={"/root/.triton": triton_cache})
def smoke_a10g_fn(): return _smoke_body("a10g")


@app.function(gpu=GPU, timeout=1800, volumes={"/root/.triton": triton_cache})
def smoke_h100_fn(): return _smoke_body("h100")


@app.local_entrypoint()
def smoke_a10g():
    out = smoke_a10g_fn.remote(); pathlib.Path(ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/selfimprove_triton_smoke_a10g.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out)[:3000])


@app.local_entrypoint()
def smoke_h100():
    out = smoke_h100_fn.remote()
    (ROOT / "results/selfimprove_triton_smoke_h100.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out)[:3000])


@app.local_entrypoint()
def calib(tag: str = "c1"):
    """A/A: the unmodified baseline against itself through the full Evaluator on the H100 (correctness, quality, ABBA speed on dev+heldout)."""
    sys.path.insert(0, str(ROOT / "src"))
    base = {"src/tritonk/tuning.py": (ROOT / "src/tritonk/tuning.py").read_text()}
    t0 = time.time()
    out = evaluate_candidate.remote(dict(track="triton", base_files=base, new_files=dict(base)))
    out["wall_s"] = time.time() - t0
    p = ROOT / "results" / f"selfimprove_triton_calib_{tag}.json"; p.write_text(json.dumps(out, indent=1, default=str)); print(f"wrote {p}")
    print(json.dumps({k: out.get(k) for k in ("verdict", "wall_s", "container_env")}, default=str))
    sp = (out.get("gates") or {}).get("speed", {}).get("data", {})
    print(json.dumps(sp, default=str)[:2000])


# ---- laptop-death-proof mode (orchestrator on Modal CPU, store on a Volume); identical to modal/selfimprove_eval.py::run_loop_remote -------------------
STORE_VOL = "northstar-selfimprove-store"
store_vol = modal.Volume.from_name(STORE_VOL, create_if_missing=True)
orch_image = (modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.2.5", "modal")
              .add_local_dir(str(ROOT / "src/tritonk"), "/repo/src/tritonk", copy=False, ignore=IGN)
              .add_local_dir(str(ROOT / "src/selfimprove"), "/repo/src/selfimprove", copy=False, ignore=IGN + ["**/tests/**"])
              .add_local_dir(str(ROOT / "harness"), "/repo/harness", copy=False, ignore=IGN))


@app.function(image=orch_image, cpu=1, memory=2048, timeout=24 * 3600, volumes={"/vol": store_vol}, max_containers=1)
def run_loop_remote(p: dict) -> dict:
    repo = "/repo"
    sys.path.insert(0, f"{repo}/src")
    import runpy
    from selfimprove.remote_loop import run_remote_loop
    from selfimprove.proposer import NorthProposer, MockProposer
    from selfimprove.loop import editable_files_in_repo
    H = runpy.run_path(f"{repo}/harness/selfimprove/config.py")
    track, mock = p["track"], p["proposer"] == "mock"
    cfg = H["make_config"](repo, track)
    if mock:
        prop = MockProposer([H["mock_structured"](track)])
    else:
        prop = NorthProposer()
    return run_remote_loop(p, pathlib.Path("/vol") / p["store"], cfg, prop, RemoteEvaluator(track),
                           profile_summary=H["profile_summary"](track), dev_results=H["dev_results"](track), on_log=store_vol.commit)


@app.local_entrypoint()
def remote(track: str = "triton", rounds: int = 1, per_round: int = 1, proposer: str = "mock", max_usd: float = 4.5, store: str = "",
           confirm_cost: bool = False):
    dry = proposer == "mock"
    store = store or ("selfimprove_triton_dryrun" if dry else "selfimprove_triton_v1")
    if dry and "dryrun" not in store: raise SystemExit("mock proposer must write to a store whose name contains 'dryrun'")
    if not dry and not confirm_cost: raise SystemExit("live North run needs --confirm-cost (spends GPU money on the proposer AND the H100 evaluator)")
    print(f"[remote] track={track} store=Volume {STORE_VOL}:/{store} rounds={rounds} x per_round={per_round} hard cap --max-usd {max_usd}")
    out = run_loop_remote.remote(dict(track=track, rounds=rounds, per_round=per_round, proposer=proposer, max_usd=max_usd, store=store, mode="structured"))
    print(json.dumps(out, indent=1, default=str))


@app.local_entrypoint()
def pull(store: str, dest: str = ""):
    sys.path.insert(0, str(ROOT / "src"))
    from selfimprove.remote_loop import pull_tree
    vol = modal.Volume.from_name(STORE_VOL)
    lst = lambda d: [(e.path, e.type == modal.volume.FileEntryType.DIRECTORY) for e in vol.listdir(d)]
    rd = lambda f: b"".join(vol.read_file(f))
    got = pull_tree(lst, rd, store, pathlib.Path(dest or ROOT / "results" / store))
    print(f"pulled {len(got)} files -> {dest or ROOT / 'results' / store}")
