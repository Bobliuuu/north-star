"""vLLM baseline for North on H100.  modal run --detach modal/vllm_base.py::main --tag v1 [--cfgs plain,ngram_k8_n3,...] [--limit N]
Each engine config runs in its own subprocess inside one container. Results -> Volume northstar-results:/vllm_<tag>.jsonl (+ .meta.json)."""
import sys, time, json, pathlib, subprocess
sys.path.insert(0, "modal")
import modal
from nsapp import weights_vol, results_vol, WEIGHTS, MODEL_DIR

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "build-essential", "ninja-build")
    .run_commands("pip install vllm==0.33.0")
    .env({"HF_HOME": "/tmp/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
    .add_local_dir("harness", "/root/harness", copy=False)
)
app = modal.App("northstar-vllm", image=image)


@app.function(gpu="H100", timeout=7000, volumes={WEIGHTS: weights_vol, "/results": results_vol}, memory=65536, scaledown_window=2, max_containers=1)
def remote(tag: str, cfgs: str, limit: int):
    t0 = time.time()
    out = pathlib.Path(f"/results/vllm_{tag}.jsonl")
    print(subprocess.run("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; pip list 2>/dev/null | grep -i -E '^(vllm|torch|transformers|flashinfer)'", shell=True, capture_output=True, text=True).stdout, flush=True)
    for c in cfgs.split(","):
        s = time.time()
        r = subprocess.run([sys.executable, "/root/harness/bench_vllm.py", MODEL_DIR, c, str(out), str(limit)])
        print("CFG", c, "rc", r.returncode, "secs", time.time() - s, flush=True)
        results_vol.commit()
    secs = time.time() - t0
    pathlib.Path(f"/results/vllm_{tag}.meta.json").write_text(json.dumps(dict(secs=secs)))
    results_vol.commit()
    print("VLLM_DONE secs", secs, "est usd", secs / 3600 * 3.95, flush=True)
    return dict(secs=secs)


@app.local_entrypoint()
def main(tag: str = "v1", cfgs: str = "plain,ngram_k8_n3,ngram_k12_n2to12,ngram_k5_n2to5", limit: int = 0):
    print(remote.remote(tag, cfgs, limit))
