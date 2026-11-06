"""GPU evaluator + launcher for the North self-improvement Loop (Phase 6).  See docs/SELFIMPROVE.md.

  Architecture:  [local `modal run` entrypoint: Loop (propose/parse/audit/store/log)]
                      |-- NorthProposer --> deployed `northstar-selfimprove` NorthProposerService (A100-80GB, North)
                      `-- RemoteEvaluator --> evaluate_candidate (THIS file): A100-80GB, outbound network BLOCKED,
                          sandboxed copy of the repo, nvcc builds baseline and candidate to SEPARATE binaries, gates
                          (correctness -> quality -> speed) run as child processes with hard timeouts.

  Commands (confirm cost first; nothing here runs unless you invoke it):
    modal run modal/selfimprove_eval.py::main --track kernel --proposer mock --rounds 1          # DRY RUN, mock, never North's
    modal run modal/selfimprove_eval.py::main --track kernel --proposer north --rounds 1 --confirm-cost   # LIVE (needs deployed proposer); kernel default = structured edits, --mode diff for raw diffs
  Est. cost: see estimate_cost() (printed before every run; ~USD 0.8-1.2 per round after the first, A100-80GB @ $2.50/h list).
"""
import json, os, pathlib, subprocess, sys, time
import modal

ROOT = pathlib.Path(__file__).resolve().parents[1]
GPU = "A100-80GB"
USD_PER_HOUR = 2.50
APP_NAME = "northstar-selfimprove-eval"
IGN = ["**/__pycache__", "**/*.pyc", "**/.pytest_cache"]

image = (modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
         .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cpu")
         .pip_install("numpy==2.2.5", "ortools==9.15.6755", "pytest==8.3.5")
         .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "PYTHONUNBUFFERED": "1"})
         .add_local_dir(str(ROOT / "src/kvm"), "/repo/src/kvm", copy=False, ignore=IGN)
         .add_local_dir(str(ROOT / "src/opt"), "/repo/src/opt", copy=False, ignore=IGN + ["**/tests/**"])
         .add_local_dir(str(ROOT / "src/selfimprove"), "/repo/src/selfimprove", copy=False, ignore=IGN + ["**/tests/**"])
         .add_local_dir(str(ROOT / "harness/selfimprove"), "/repo/harness/selfimprove", copy=False, ignore=IGN)
         .add_local_dir(str(ROOT / "results"), "/repo/results", copy=False, ignore=IGN + ["selfimprove*/**"]))
app = modal.App(APP_NAME, image=image)


def _cfg(repo, track, live=False):
    import runpy
    sys.path.insert(0, str(pathlib.Path(repo) / "src"))
    return runpy.run_path(str(pathlib.Path(repo) / "harness/selfimprove/config.py"))["make_config"](repo, track, require_measured_calibration=live)


@app.function(gpu=GPU, timeout=3300, cpu=8, memory=32768, block_network=True, max_containers=1, scaledown_window=2)
def evaluate_candidate(payload: dict) -> dict:
    """payload: {track, base_files, new_files}. Runs the Loop's own gated Evaluator inside this network-blocked GPU container.
    Candidate sources are written by the evaluator into throw-away workspaces; the candidate never writes anything itself."""
    repo = "/repo"
    sys.path.insert(0, f"{repo}/src")
    from selfimprove.loop import Evaluator
    cfg = _cfg(repo, payload["track"])
    cfg.tmp_parent = pathlib.Path("/tmp")
    env = {}
    for k, cmd in (("nvcc", ["nvcc", "--version"]), ("gpu", ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])):
        try: env[k] = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[-1]
        except Exception as e: env[k] = f"unavailable: {e!r}"[:100]
    out = Evaluator(cfg).evaluate(payload["base_files"], payload["new_files"])
    out["container_env"] = env
    return out


class RemoteEvaluator:
    """Drop-in for loop.Evaluator that runs evaluate() on the Modal GPU function. `call(payload)->dict` is injectable (tests)."""
    def __init__(self, track, call=None):
        self.track, self.call = track, call or evaluate_candidate.remote

    def evaluate(self, base_files, new_files):
        t0 = time.time()
        try:
            out = self.call(dict(track=self.track, base_files=base_files, new_files=new_files))
        except Exception as e:                      # container crash / timeout: record, never promote
            return dict(verdict="evaluator_error", gates={}, sandbox={"error": repr(e)[:500]}, seeds={}, eval_wall_s=time.time() - t0)
        out.setdefault("eval_wall_s", time.time() - t0)
        return out


# ---- cost model (ESTIMATES from the design, not yet measured; replaced by Store.cost_summary() after round 1) ----------
def estimate_cost(track="kernel", rounds=1, cold_start=True, with_proposer=True):
    """USD at list price, A100-80GB billed per container-minute. Evaluator (sequential after the proposer): kernel = 2 nvcc builds
    (~1.5 min each) + 2 correctness (~1.5 min, CPU torch reference) + 2 quality (~2 min) + 36 bench calls (~8 s) ~ 15 min worst case
    (gates short-circuit, so failing candidates cost less); policy = CPU-bound solver ~8 min. Proposer: North HF-transformers
    generation <= 2048 tokens at ~0.1 s/token ~ 3.5 min, plus (first round) cold start + model load ~ 6 min. ESTIMATES, unmeasured."""
    ev_min = 15 if track == "kernel" else 8
    prop_min = 3.5 if with_proposer else 0.0
    total_min = rounds * (ev_min + prop_min + 1.0) + (6 if (cold_start and with_proposer) else 0) + 1.5   # +1 min/round overhead, +1.5 evaluator cold start
    return dict(minutes=total_min, usd=total_min / 60 * USD_PER_HOUR, eval_min_per_round=ev_min, prop_min_per_round=prop_min)


def _prep_store_guard(store_dir, proposer):
    from selfimprove.store import Store
    if not store_dir.exists(): return
    att = Store(store_dir).attribution()
    if proposer == "north" and att.get("mock_versions"):
        raise SystemExit(f"refusing: store {store_dir} has MOCK versions in its lineage; use a fresh --store for North runs")
    if proposer == "mock" and att.get("north_versions"):
        raise SystemExit(f"refusing: store {store_dir} holds North-authored versions; mock dry runs need their own store")


# ---- laptop-death-proof mode: orchestrator runs ON Modal (CPU); store lives on a Volume --------------------------------------
STORE_VOL = "northstar-selfimprove-store"
store_vol = modal.Volume.from_name(STORE_VOL, create_if_missing=True)
orch_image = (modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.2.5", "modal")
              .add_local_dir(str(ROOT / "src/kvm"), "/repo/src/kvm", copy=False, ignore=IGN)
              .add_local_dir(str(ROOT / "src/opt"), "/repo/src/opt", copy=False, ignore=IGN + ["**/tests/**"])
              .add_local_dir(str(ROOT / "src/selfimprove"), "/repo/src/selfimprove", copy=False, ignore=IGN + ["**/tests/**"])
              .add_local_dir(str(ROOT / "harness/selfimprove"), "/repo/harness/selfimprove", copy=False, ignore=IGN)
              .add_local_dir(str(ROOT / "results"), "/repo/results", copy=False, ignore=IGN + ["selfimprove*/**"]))


@app.function(image=orch_image, cpu=1, memory=2048, timeout=24 * 3600, volumes={"/vol": store_vol}, max_containers=1)
def run_loop_remote(p: dict) -> dict:
    """p: {track, rounds, proposer, mode, max_usd, store}. Runs the unchanged Loop here (no laptop needed). Each proposal record is
    printed as `NS_PROPOSAL_JSON {...}` to app logs and the Volume is committed after every store write. Evaluation runs on the
    network-blocked GPU function `evaluate_candidate`; the proposer is the deployed NorthProposerService (or MockProposer)."""
    repo = "/repo"
    sys.path.insert(0, f"{repo}/src")
    import runpy
    from selfimprove.remote_loop import run_remote_loop
    from selfimprove.proposer import NorthProposer, MockProposer
    from selfimprove.loop import editable_files_in_repo
    H = runpy.run_path(f"{repo}/harness/selfimprove/config.py")
    track, mock = p["track"], p["proposer"] == "mock"
    cfg = H["make_config"](repo, track, require_measured_calibration=not mock)
    if mock:
        base = editable_files_in_repo(pathlib.Path(repo), cfg.scope)
        prop = MockProposer(['{"edits":[{"name":"KC","value":128}],"rationale":"MOCK"}'] if p["mode"] == "structured" else [H["mock_diff"](track, base)])
    else:
        prop = NorthProposer()
    return run_remote_loop(p, pathlib.Path("/vol") / p["store"], cfg, prop, RemoteEvaluator(track),
                           profile_summary=H["profile_summary"](track), dev_results=H["dev_results"](track),
                           on_log=store_vol.commit)


@app.local_entrypoint()
def remote(track: str = "kernel", rounds: int = 1, proposer: str = "mock", max_usd: float = 3.0, store: str = "",
           confirm_cost: bool = False, mode: str = ""):
    """Launch: `modal run --detach modal/selfimprove_eval.py::remote --track kernel --proposer mock --rounds 1`.
    The loop keeps running if this client dies. Pull results: `modal run modal/selfimprove_eval.py::pull --store NAME`."""
    dry = proposer == "mock"
    mode = mode or ("structured" if track == "kernel" else "diff")
    if mode not in ("structured", "diff") or (mode == "structured" and track != "kernel"):
        raise SystemExit("--mode must be 'diff', or 'structured' (kernel track only)")
    store = store or (f"selfimprove_dryrun_{track}" if dry else f"selfimprove_{track}")
    if dry and "dryrun" not in store: raise SystemExit("mock proposer must write to a store whose name contains 'dryrun'")
    if not dry and not confirm_cost: raise SystemExit("live North run needs --confirm-cost (spends GPU money on the proposer AND the evaluator)")
    est = estimate_cost(track, rounds, with_proposer=not dry)
    print(f"[remote] mode={mode} store=Volume {STORE_VOL}:/{store} ~USD {est['usd']:.2f} (hard cap --max-usd {max_usd})")
    out = run_loop_remote.remote(dict(track=track, rounds=rounds, proposer=proposer, max_usd=max_usd, store=store, mode=mode))
    print(json.dumps(out, indent=1, default=str))


@app.local_entrypoint()
def pull(store: str, dest: str = ""):
    """Copy Volume store `store` into results/<store> (or --dest) and verify the hash chain. Never overwrites a diverged local log."""
    sys.path.insert(0, str(ROOT / "src"))
    from selfimprove.remote_loop import pull_tree
    vol = modal.Volume.from_name(STORE_VOL)
    lst = lambda d: [(e.path, e.type == modal.volume.FileEntryType.DIRECTORY) for e in vol.listdir(d)]
    rd = lambda f: b"".join(vol.read_file(f))
    got = pull_tree(lst, rd, store, pathlib.Path(dest or ROOT / "results" / store))
    print(f"pulled {len(got)} files -> {dest or ROOT / 'results' / store}")


@app.local_entrypoint()
def main(track: str = "kernel", rounds: int = 1, proposer: str = "mock", max_usd: float = 3.0, store: str = "",
         confirm_cost: bool = False, mode: str = "", temperature: float = 0.3, max_new_tokens: int = 2048):
    sys.path.insert(0, str(ROOT / "src"))
    from selfimprove.loop import Loop
    from selfimprove.proposer import NorthProposer, MockProposer
    from selfimprove.store import Store
    import runpy
    H = runpy.run_path(str(ROOT / "harness/selfimprove/config.py"))
    dry = proposer == "mock"
    mode = mode or ("structured" if track == "kernel" else "diff")      # structured edits are defined for the kernel tunables track only
    if mode not in ("structured", "diff") or (mode == "structured" and track != "kernel"):
        raise SystemExit("--mode must be 'diff', or 'structured' (kernel track only)")
    store_dir = pathlib.Path(store or ROOT / "results" / (f"selfimprove_dryrun_{track}" if dry else f"selfimprove_{track}"))
    if dry and "dryrun" not in store_dir.name:
        raise SystemExit("mock proposer must write to a store whose directory name contains 'dryrun'")
    est = estimate_cost(track, rounds, with_proposer=not dry)
    print(f"[mode] {mode}")
    print(f"[estimate] ~{est['minutes']:.0f} A100-80GB minutes, ~USD {est['usd']:.2f} for {rounds} round(s) (hard cap --max-usd {max_usd})")
    if dry:
        print("=" * 100 + f"\n{H['MOCK_BANNER']}\n" + "=" * 100)
    elif not confirm_cost:
        raise SystemExit("live North run needs --confirm-cost (spends GPU money on the proposer AND the evaluator)")
    _prep_store_guard(store_dir, proposer)
    cfg = H["make_config"](ROOT, track, require_measured_calibration=not dry)
    st = Store(store_dir)
    if dry:
        from selfimprove.loop import editable_files_in_repo
        base = editable_files_in_repo(ROOT, cfg.scope)
        prop = MockProposer(['{"edits":[{"name":"KC","value":128}],"rationale":"MOCK"}'] if mode == "structured" else [H["mock_diff"](track, base)])
    else:
        prop = NorthProposer()
    lp = Loop(cfg, st, prop, mode=mode)
    lp.ev = RemoteEvaluator(track)
    res = lp.run(rounds, max_total_usd=max_usd, profile_summary=H["profile_summary"](track), dev_results=H["dev_results"](track))
    for r in res:
        print(r["proposal_id"][:8], r["verdict"], "|", r["proposer"]["label"] if "proposer" in r else "no proposer record",
              "| cost", {k: round(v, 3) for k, v in r.get("cost", {}).items() if isinstance(v, float)})
    print(json.dumps(st.attribution(), indent=1)); print(json.dumps(st.cost_summary(), indent=1))
    if dry:
        print(H["MOCK_BANNER"])
