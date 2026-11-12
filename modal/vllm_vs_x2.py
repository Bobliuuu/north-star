"""Same-host A/B: stock vLLM 0.33 plain vs our exact v1 / exact v2 (bench_x2 traffic phase), one H100 container, run back to back:
ours (99 prompts) -> vLLM plain (21 natural + 99 mix) -> ours again (33-prompt subset, drift check).
    modal run --detach modal/vllm_vs_x2.py::main --tag vx1
Rows -> Volume northstar-results:/vx_<tag>_{ours1,vllm,ours2}.jsonl"""
import sys, time, json, pathlib, subprocess
sys.path.insert(0, "modal"); sys.path.insert(0, "/root")
import modal
from nsapp import weights_vol, results_vol, WEIGHTS, MODEL_DIR

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "build-essential", "ninja-build")
    .run_commands("pip install vllm==0.33.0", "pip install accelerate==1.15.0")
    .env({"HF_HOME": "/tmp/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
    .add_local_dir("harness", "/root/harness", copy=False)
    .add_local_dir("src/northstar", "/root/northstar", copy=False)
    .add_local_file("modal/nsapp.py", "/root/nsapp.py", copy=False)
)
app = modal.App("northstar-vllm-vs-x2", image=image)


@app.function(gpu="H100", timeout=7000, volumes={WEIGHTS: weights_vol, "/results": results_vol}, memory=65536, scaledown_window=2, max_containers=1)
def remote(tag: str, limit2: int):
    t0 = time.time()
    print(subprocess.run("nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem --format=csv; pip list 2>/dev/null | grep -i -E '^(vllm|torch|transformers|triton|accelerate)'",
                         shell=True, capture_output=True, text=True).stdout, flush=True)
    steps = [("ours1", [sys.executable, "/root/harness/run_x2_traffic.py", MODEL_DIR, f"/results/vx_{tag}_ours1.jsonl", "0"]),
             ("vllm", [sys.executable, "/root/harness/bench_vllm.py", MODEL_DIR, "plain", f"/results/vx_{tag}_vllm.jsonl", "0"]),
             ("ours2", [sys.executable, "/root/harness/run_x2_traffic.py", MODEL_DIR, f"/results/vx_{tag}_ours2.jsonl", str(limit2)])]
    log = {}
    for name, cmd in steps:
        s = time.time()
        r = subprocess.run(cmd, cwd="/root")
        log[name] = dict(rc=r.returncode, secs=time.time() - s)
        print("STEP", name, log[name], flush=True)
        results_vol.commit()
    secs = time.time() - t0
    pathlib.Path(f"/results/vx_{tag}.meta.json").write_text(json.dumps(dict(secs=secs, steps=log)))
    results_vol.commit()
    print("VX_DONE secs", secs, "est usd", secs / 3600 * 3.95, flush=True)
    return log


@app.local_entrypoint()
def main(tag: str = "vx1", limit2: int = 33):
    print(remote.remote(tag, limit2))
