"""cohere-megakernel (pinned 67d0b9ca) on Modal H100: build in the image (CPU only), run plain-decode baseline on North.

  modal run modal/mk.py::check          CPU only: confirm the image built (no GPU spend)
  modal run modal/mk.py::bench          ONE H100 container, hard timeout, appends to results/SPEND.md
"""
import datetime, os, subprocess, time, pathlib
import modal

PIN = "67d0b9ca22ea3652796b715d1d1863459e0e2c3c"
ROOT = pathlib.Path(__file__).resolve().parent.parent
RATE_H100 = 3.95
weights_vol = modal.Volume.from_name("northstar-weights")
jit_vol = modal.Volume.from_name("northstar-mkjit", create_if_missing=True)
CKPT = "/weights/north-mini-code-d11e61a8"

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "cmake", "ninja-build", "libssl-dev", "build-essential")
    .run_commands(
        "pip install torch --index-url https://download.pytorch.org/whl/cu130",
        "pip install transformers safetensors fastapi uvicorn cohere-melody==0.9.0 ninja einops numpy",
        "pip install flash-attn-3 --index-url https://download.pytorch.org/whl/cu130",
        f"git clone --recurse-submodules https://github.com/cohere-ai/cohere-megakernel /mk && cd /mk && git checkout {PIN} && git submodule update --init --recursive",
        "cd /mk && cmake -S . -B build -G Ninja -DPython_EXECUTABLE=$(which python) -DCMAKE_BUILD_TYPE=Release && cmake --build build 2>&1 | tail -40",
        "ls -la /mk/build",
    )
    .env({"LITTLE_JIT_CACHE_DIR": "/jit", "PYTHONUNBUFFERED": "1"})
)
app = modal.App("northstar-mk", image=image)


def sh(cmd, timeout):
    t = time.time()
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd="/mk")
        out, rc = p.stdout + "\n--stderr--\n" + p.stderr[-6000:], p.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = f"TIMEOUT {timeout}s\n{(e.stdout or b'')[-4000:]}\n{(e.stderr or b'')[-4000:]}", -9
    return f"$ {cmd}\n[rc={rc} {time.time()-t:.0f}s]\n{out}\n"


@app.function(timeout=300)
def check():
    print(sh("ls -la build; nvcc --version | tail -2; git rev-parse HEAD; python -c 'import torch;print(torch.__version__, torch.version.cuda)'", 120))


COMMON = ("python src/runner.py --mk --lib-path /mk/build/libmk_release.so --num-sms 132 --frac-vram-utilization 0.85 "
          "--cpp-decode-runtime ")


@app.function(gpu="H100", timeout=1500, scaledown_window=2, max_containers=1,
              volumes={"/weights": weights_vol, "/jit": jit_vol}, memory=65536)
def bench(which: str = "all"):
    t0 = time.time()
    log = [sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv", 60)]
    if which in ("all", "synthetic"):  # no checkpoint load: uniform routing, cheap; also warms the JIT cache
        for L in (2048, 8192):
            log.append(sh(COMMON + f"--fast --batch-size 1 --max-new-tokens 512 --fake-prompt-len {L} 2>&1 | grep -E 'metrics|tok/s|mean/step|rror' | tail -8", 400))
    if which in ("all", "real"):
        for L in (2048, 8192):
            log.append(sh(COMMON + f"--fast --real-weight --checkpoint {CKPT} --batch-size 1 --max-new-tokens 512 --fake-prompt-len {L} 2>&1 | grep -E 'metrics|tok/s|mean/step|rror' | tail -8", 500))
    if which in ("all", "bs"):  # step time vs tokens-in-flight (independent sequences; union of experts grows ~ like a verify block's)
        for B in (1, 2, 4, 8):
            log.append(sh(COMMON + f"--fast --real-weight --checkpoint {CKPT} --batch-size {B} --max-new-tokens 256 --fake-prompt-len 2048 2>&1 | grep -E 'tok/s|mean/step|rror' | tail -4", 500))
    if which in ("all", "prompt"):
        log.append(sh(COMMON + f"--checkpoint {CKPT} --batch-size 1 --msgs 'Write a Python function that merges two sorted lists, with tests.' --max-new-tokens 512 2>&1 | grep -E 'tok/s|mean/step|rror' | tail -8", 500))
    jit_vol.commit()
    return dict(log="\n".join(log), secs=time.time() - t0)


@app.local_entrypoint()
def main(which: str = "all"):
    r = bench.remote(which)
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    d = ROOT / "results"; d.mkdir(exist_ok=True)
    (d / f"megakernel_H100_{which}_{ts}.log").write_text(r["log"])
    cost = r["secs"] / 3600 * RATE_H100
    with open(d / "SPEND.md", "a") as f:
        f.write(f"| {ts} | megakernel {which} | H100 | {r['secs']:.0f} | {cost:.2f} | |\n")
    print(r["log"][-9000:]); print("secs", r["secs"], "est $%.2f" % cost)
