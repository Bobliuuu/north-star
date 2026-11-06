"""Run the Northstar fused draft/verify MoE microbenchmark on Modal.

  modal run modal/run_modal.py                       # A100-80GB, check + quick sweep
  modal run modal/run_modal.py --full true --seeds 2  # full r/Tv/Td sweep
  modal run modal/run_modal.py --gpu H100 --arch sm_90

Budget: spend is ESTIMATED from wall time x list price and appended to results/SPEND.md. Short timeouts, no idle containers
(scaledown_window=2, min_containers=0).
"""
import datetime, os, subprocess
import modal

RATE = {"A100-80GB": 2.50, "A100-40GB": 2.10, "H100": 3.95}  # USD/hour list price (approx.)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image = (modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
         .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
         .pip_install("numpy<2", "pytest")
         .add_local_dir(os.path.join(ROOT, "src/kvm"), "/work/kvm"))
app = modal.App("northstar-kvm-fused", image=image)


def _sh(cmd, log):
    log.append("$ " + cmd)
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    log.append(p.stdout + p.stderr + f"[rc={p.returncode}]")
    return p.returncode


def _body(arch, full, seeds, reps, skip_check):
    import time
    t0 = time.time(); log = []; out = {}
    log.append(subprocess.run("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv", shell=True, capture_output=True, text=True).stdout)
    rc = _sh(f"cd /work/kvm && nvcc --version | tail -1 && nvcc -O3 -std=c++17 -arch={arch} -o /work/moe_fused moe_fused.cu", log)
    if rc: return dict(log="\n".join(log), jsonl="", ok=False, secs=time.time() - t0)
    ok = True
    if not skip_check:
        rc = _sh("python /work/kvm/tests/check_cuda.py /work/moe_fused /work/chk", log); ok = rc == 0
        rc2 = _sh("cd /work/kvm && python -m pytest -q tests/test_moe_ref.py", log); ok &= rc2 == 0
    jsonl = ""
    if ok:
        rc = _sh(f"/work/moe_fused bench /work/bench.jsonl {'--full' if full else ''} --seeds {seeds} --reps {reps}", log)
        if rc == 0: jsonl = open("/work/bench.jsonl").read()
        ok &= rc == 0
    return dict(log="\n".join(log), jsonl=jsonl, ok=ok, secs=time.time() - t0)


@app.function(gpu="A100-80GB", timeout=1500, scaledown_window=2, max_containers=1, serialized=False)
def run_a100(arch="sm_80", full=False, seeds=1, reps=15, skip_check=False):
    return _body(arch, full, seeds, reps, skip_check)


@app.function(gpu="H100", timeout=1500, scaledown_window=2, max_containers=1)
def run_h100(arch="sm_90", full=False, seeds=1, reps=15, skip_check=False):
    return _body(arch, full, seeds, reps, skip_check)


@app.function(timeout=900)
def compile_check(arch="sm_90"):
    log = []
    rc = _sh(f"cd /work/kvm && nvcc -O3 -std=c++17 -arch={arch} -o /work/moe_fused moe_fused.cu", log)
    return rc, "\n".join(log)


@app.function(gpu="H100", timeout=1200, scaledown_window=2, max_containers=1)
def replay_h100(inp: str, reps: int = 9, arch: str = "sm_90"):
    import time
    t0 = time.time(); log = []
    log.append(subprocess.run("nvidia-smi --query-gpu=name,driver_version --format=csv", shell=True, capture_output=True, text=True).stdout)
    rc = _sh(f"cd /work/kvm && nvcc -O3 -std=c++17 -arch={arch} -o /work/moe_fused moe_fused.cu", log)
    if rc: return dict(log="\n".join(log), jsonl="", ok=False, secs=time.time() - t0)
    open("/work/replay_in.txt", "w").write(inp)
    rc = _sh(f"/work/moe_fused replay /work/replay_in.txt /work/replay_out.jsonl --reps {reps}", log)
    jsonl = open("/work/replay_out.jsonl").read() if rc == 0 else ""
    return dict(log="\n".join(log), jsonl=jsonl, ok=rc == 0, secs=time.time() - t0)


@app.local_entrypoint()
def replay(inpath: str = "src/kvm/replay_input.txt", reps: int = 9, tag: str = "replay", check_only: bool = False):
    if check_only:
        rc, log = compile_check.remote(); print(rc, log[-1500:]); return
    r = replay_h100.remote(open(os.path.join(ROOT, inpath)).read(), reps)
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    base = os.path.join(ROOT, "results", f"kvm_replay_H100_{tag}_{ts}")
    open(base + ".log", "w").write(r["log"])
    if r["jsonl"]: open(base + ".jsonl", "w").write(r["jsonl"])
    cost = r["secs"] / 3600 * RATE["H100"]
    with open(os.path.join(ROOT, "results", "SPEND.md"), "a") as f:
        f.write(f"| {ts} | kvm-fused REAL-routing replay {tag} | H100 | {r['secs']:.0f} | {cost:.3f} | {r['ok']} |\n")
    print(r["log"][-2500:]); print("ok:", r["ok"], "est $%.3f" % cost, "->", base)


@app.local_entrypoint()
def main(gpu: str = "A100-80GB", arch: str = "", full: bool = False, seeds: int = 1, reps: int = 15,
         skip_check: bool = False, tag: str = "run"):
    fn = run_h100 if gpu == "H100" else run_a100
    r = fn.remote(arch or ("sm_90" if gpu == "H100" else "sm_80"), full, seeds, reps, skip_check)
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    base = os.path.join(ROOT, "results", f"kvm_fused_{gpu}_{tag}_{ts}")
    open(base + ".log", "w").write(r["log"])
    if r["jsonl"]: open(base + ".jsonl", "w").write(r["jsonl"])
    cost = r["secs"] / 3600 * RATE.get(gpu, 4.0)
    with open(os.path.join(ROOT, "results", "SPEND.md"), "a") as f:
        if f.tell() == 0: f.write("# Spend log (Modal; estimated = wall seconds x list price, excludes image build)\n\n| utc | task | gpu | secs | est USD | ok |\n|---|---|---|---|---|---|\n")
        f.write(f"| {ts} | kvm-fused {tag} full={full} seeds={seeds} | {gpu} | {r['secs']:.0f} | {cost:.3f} | {r['ok']} |\n")
    print(r["log"][-4000:]); print("ok:", r["ok"], "est $%.3f" % cost, "->", base)
