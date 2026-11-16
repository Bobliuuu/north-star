"""moe_fused.cu on North's real decode path (harness/bench_moecuda.py), H100.
  modal run --detach modal/moecuda_run.py::main --tag mc1 [--pr7-src PATH]
Own image: the CUDA devel base (nvcc, cuobjdump) with nsapp's pins, because the kernel is built here as a shared library.
Writes northstar-results:/moecuda_<tag>.jsonl (+ .meta.json); each phase is committed as it finishes."""
import sys, time, json, pathlib, os
sys.path.insert(0, "modal")
import modal
from nsapp import weights_vol, results_vol, WEIGHTS, MODEL_DIR

PR7 = os.environ.get("NS_PR7_SRC", "")   # optional: the PR #7 moe_fused.cu, to prove the default-mode SASS is unchanged
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch==2.10.0", "transformers==5.22.1", "accelerate==1.15.0", "numpy==2.2.5", "huggingface_hub", "safetensors")
    .env({"HF_HOME": "/tmp/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1", "NS_KVM_SRC": "/root/kvm"})
    .add_local_dir("src/northstar", "/root/northstar", copy=False)
    .add_local_dir("harness", "/root/harness", copy=False)
    .add_local_dir("src/kvm", "/root/kvm", copy=False, ignore=["replay_input.txt", "**/__pycache__/**"])
    .add_local_file("modal/nsapp.py", "/root/nsapp.py", copy=False)
)
if PR7:
    image = image.add_local_file(PR7, "/root/pr7/moe_fused.cu", copy=False)
app = modal.App("ship-moecuda-e2e", image=image)


def _sass_identity():
    """Default-mode (bench/check) build of this branch's moe_fused.cu vs PR #7's: identical SASS means the layout mode
    changed nothing the PR #7 numbers measured."""
    import subprocess
    if not os.path.exists("/root/pr7/moe_fused.cu"):
        return {"skipped": "no PR #7 source"}
    out = {}
    for tag, src in (("branch", "/root/kvm/moe_fused.cu"), ("pr7", "/root/pr7/moe_fused.cu")):
        cub = f"/tmp/{tag}.cubin"
        r = subprocess.run(["nvcc", "-O3", "-std=c++17", "-arch=sm_90", "-cubin", "-I", "/root/kvm", "-I", "/root/kvm/include", src, "-o", cub],
                           capture_output=True, text=True)
        if r.returncode:
            return {"error": tag, "log": r.stderr[-1500:]}
        s = subprocess.run(["cuobjdump", "-sass", cub], capture_output=True, text=True).stdout
        out[tag] = [l.strip() for l in s.splitlines() if l.strip().startswith("/*") and "*/" in l]
    if not out["branch"]:
        return {"error": "no SASS lines parsed"}
    return {"identical": out["branch"] == out["pr7"], "lines": len(out["branch"])}


@app.function(gpu="H100", timeout=3600, volumes={WEIGHTS: weights_vol, "/results": results_vol}, memory=65536, scaledown_window=2, max_containers=1)
def remote(tag: str):
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/harness")
    t0 = time.time()
    meta = {"tag": tag}
    def save(name, text):
        pathlib.Path(f"/results/moecuda_{tag}.{name}").write_text(text); results_vol.commit()
    meta["sass_identity_default_mode_vs_pr7"] = _sass_identity(); save("meta.json", json.dumps(meta))
    print("SASS", meta["sass_identity_default_mode_vs_pr7"], flush=True)
    import moe_cuda
    sos, regs = {}, {}
    for name, defs in (("mc_perarch", ()), ("mc_upstream", ("NS_TUNE_UPSTREAM",))):
        so, rg = moe_cuda.build("/tmp", defines=defs); sos[name] = so; regs[name] = rg
    meta["ptxas"] = regs; save("meta.json", json.dumps(meta)); print("PTXAS", regs, flush=True)
    from loader import load_model
    from common import env_meta
    import bench_moecuda
    m, tok = load_model(MODEL_DIR, "grouped_mm")
    meta.update(env_meta()); meta["load_s"] = time.time() - t0; save("meta.json", json.dumps(meta))
    j = bench_moecuda.run(m, tok, sos)
    text = j.text(); secs = time.time() - t0
    save("jsonl", text); meta["secs"] = secs; save("meta.json", json.dumps(meta))
    print("MC_DONE secs", secs, "est usd", secs / 3600 * 3.95, flush=True)
    return text


@app.local_entrypoint()
def main(tag: str = "mc1"):
    text = remote.remote(tag)
    pathlib.Path(f"results/moecuda_{tag}.jsonl").write_text(text)
    print("wrote", f"results/moecuda_{tag}.jsonl")
