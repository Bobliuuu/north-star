"""Harness diagnostics on a Modal A100 (developer tooling, not North): (1) A/A speed+timer statistics through the REAL protected
bench harness, (2) correctness flakiness across many NS_SEEDs for baseline and the North edit set that failed on some seeds.
  modal run --detach modal/harness_diag.py --tag diag1
Result written to results/harness_diag_<tag>_<ts>.json. Spend estimated and appended to results/SPEND.md.
"""
import datetime, json, os, re, shutil, subprocess, time
import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IGN = ["**/__pycache__", "**/*.pyc", "**/.pytest_cache"]
image = (modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
         .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cpu").pip_install("numpy==2.2.5")
         .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "PYTHONUNBUFFERED": "1"})
         .add_local_dir(os.path.join(ROOT, "src/kvm"), "/repo/src/kvm", copy=False, ignore=IGN)
         .add_local_dir(os.path.join(ROOT, "harness/selfimprove"), "/repo/harness/selfimprove", copy=False, ignore=IGN))
app = modal.App("northstar-harness-diag", image=image)
BASE = dict(KC=256, XPAD=8, HPAD=8, MT_SPLIT=16, MINB_PER_SM=1, UNROLL_KS=16)
VARIANTS = {"base": {}, "northedit": dict(KC=128, XPAD=16, HPAD=16, MT_SPLIT=8, MINB_PER_SM=2), "mt8": dict(MT_SPLIT=8)}


def make_ws(name, over):
    ws = f"/work/{name}"; shutil.rmtree(ws, ignore_errors=True)
    shutil.copytree("/repo", ws)
    p = f"{ws}/src/kvm/moe_fused.cu"; s = open(p).read()
    for k, v in {**BASE, **over}.items():
        s, n = re.subn(rf"(?m)^(constexpr int {k} = )\d+(;)", rf"\g<1>{v}\g<2>", s); assert n == 1, k
    open(p, "w").write(s)
    r = subprocess.run(["python", f"{ws}/harness/selfimprove/kernel_build.py"], env={**os.environ, "NS_NONCE": "x", "NS_WORKSPACE": ws}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    return ws


def run_h(ws, script, seed=0, split="dev"):
    t0 = time.perf_counter()
    r = subprocess.run(["python", f"{ws}/harness/selfimprove/{script}"], cwd=ws, capture_output=True, text=True, timeout=900,
                       env={**os.environ, "NS_NONCE": "n", "NS_WORKSPACE": ws, "NS_SEED": str(seed), "NS_SPLIT": split})
    wall = time.perf_counter() - t0
    try: j = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception: j = {"error": (r.stdout + r.stderr)[-400:]}
    return j, wall


@app.function(gpu=["A100-80GB", "A100-40GB"], timeout=3000, cpu=8, memory=32768, scaledown_window=2, max_containers=1)
def diag(n_seeds: int = 24, n_aa: int = 18):
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time(); os.makedirs("/work", exist_ok=True)
    out = {"gpu": subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip()}
    items = list(VARIANTS.items()) + [("baseB", {})]
    with ThreadPoolExecutor(4) as ex: ws = dict(zip([k for k, _ in items], ex.map(lambda a: make_ws(*a), items)))
    print("built", list(ws), flush=True)
    aa = []   # baseline vs baseline, alternating order, real kernel_bench.py (time_ms = binary's CUDA-event median, ref_ms = harness wall marginal)
    for i in range(n_aa):
        for tag in (("base", "baseB") if i % 2 == 0 else ("baseB", "base")):
            for split in ("dev", "heldout"):
                j, wall = run_h(ws[tag], "kernel_bench.py", split=split)
                aa.append(dict(i=i, tag=tag, split=split, time_ms=j.get("time_ms"), ref_ms=j.get("ref_ms"), harness_wall_ms=j.get("harness_wall_ms"),
                               proc_wall_ms=wall * 1000, err=j.get("error")))
    out["aa"] = aa; print("aa done", time.time() - t0, flush=True)
    jobs = [(nm, s) for s in range(1, n_seeds + 1) for nm in VARIANTS]

    def one(a):
        nm, s = a; seed = s * 7919 + 3; j, _ = run_h(ws[nm], "kernel_check.py", seed=seed)
        cs = j.get("cases", []); bad = [c["name"] for c in cs if not (c["rel_l2"] <= 1e-3)]
        return dict(variant=nm, seed=seed, worst=max([c["rel_l2"] for c in cs] or [None]), n=len(cs), bad=bad, err=j.get("error"))
    with ThreadPoolExecutor(6) as ex: out["corr"] = list(ex.map(one, jobs))
    out["secs"] = time.time() - t0
    print("NS_DIAG_DONE", flush=True)
    return out


@app.local_entrypoint()
def main(tag: str = "diag1", n_seeds: int = 24, n_aa: int = 18):
    r = diag.remote(n_seeds, n_aa)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(ROOT, "results", f"harness_diag_{tag}_{ts}.json"); json.dump(r, open(path, "w"), indent=1)
    cost = r["secs"] / 3600 * 2.5
    open(os.path.join(ROOT, "results", "SPEND.md"), "a").write(f"| {ts} | harness diag {tag} | A100 | {r['secs']:.0f} | {cost:.3f} | True |\n")
    print("est $%.3f ->" % cost, path)
