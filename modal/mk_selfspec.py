"""cohere-megakernel (pinned 67d0b9ca) + mk-selfspec patch: draft-variant and verify-block timing INSIDE the megakernel.

Patch (src/mk_selfspec/mk_selfspec.patch): MK_DRAFT_R (top-r experts per token via router_topk_op; slots >=R duplicate slot 0 with
score 0, so the expert tile is loaded once and only R distinct experts stream), MK_UNION_U (emulate a verify block's per-layer expert
union of size U), MK_LAYERS env (layer-skip: truncated layer count). Variants are selected by sed on megakernel.cuh (JIT hash covers it).

  modal run --detach modal/mk_selfspec.py::main --phase A     # reproduce + draft variants
  modal run --detach modal/mk_selfspec.py::main --phase B     # verify-block (BS=K, union-emulated)
Per-config JSON lines go to Volume northstar-results:/mk_selfspec_<phase>.jsonl (pull with `modal volume get`).
"""
import datetime, json, os, re, subprocess, time, pathlib
import modal

PIN = "67d0b9ca22ea3652796b715d1d1863459e0e2c3c"
ROOT = pathlib.Path(__file__).resolve().parent.parent
RATE_H100 = 3.95
weights_vol = modal.Volume.from_name("northstar-weights")
jit_vol = modal.Volume.from_name("northstar-mkjit", create_if_missing=True)
res_vol = modal.Volume.from_name("northstar-results", create_if_missing=True)
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
    )
    .add_local_file(str(ROOT / "src/mk_selfspec/mk_selfspec.patch"), "/mk_selfspec.patch", copy=True)
    .run_commands("cd /mk && git apply /mk_selfspec.patch && git diff --stat")
    .env({"LITTLE_JIT_CACHE_DIR": "/jit", "PYTHONUNBUFFERED": "1"})
)
app = modal.App("northstar-mk-selfspec", image=image)

COMMON = ("python src/runner.py --mk --lib-path /mk/build/libmk_release.so --num-sms 132 --frac-vram-utilization 0.85 "
          "--cpp-decode-runtime --fast ")


def sh(cmd, timeout, env=None):
    t = time.time()
    e = dict(os.environ)
    e.update(env or {})
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd="/mk", env=e)
        return p.stdout + "\n" + p.stderr[-3000:], p.returncode, time.time() - t
    except subprocess.TimeoutExpired:
        return f"TIMEOUT {timeout}", -9, time.time() - t


def set_variant(R, U):
    p = "/mk/src/decode/megakernel.cuh"
    s = open(p).read()
    s = re.sub(r"#define MK_DRAFT_R \d+", f"#define MK_DRAFT_R {R}", s, count=1)
    s = re.sub(r"#define MK_UNION_U \d+", f"#define MK_UNION_U {U}", s, count=1)
    open(p, "w").write(s)


# name, R, U, layers, bs, active rows K, real weights
PHASES = {
    "A": [  # reproduce + draft-step variants, BS=1, ctx 2048
        ("base_R8_real", 8, 0, 49, 1, 1, True),
        ("R4_real", 4, 0, 49, 1, 1, True),
        ("R2_real", 2, 0, 49, 1, 1, True),
        ("R1_real", 1, 0, 49, 1, 1, True),
        ("base_R8_syn", 8, 0, 49, 1, 1, False),
        ("L41_R8_syn", 8, 0, 41, 1, 1, False),
        ("L41_R4_syn", 4, 0, 41, 1, 1, False),
        ("L33_R4_syn", 4, 0, 33, 1, 1, False),
        ("L41_R2_syn", 2, 0, 41, 1, 1, False),
    ],
    "B": [  # verify block of K tokens = BS rows (padded with masked rows); U = emulated union (real traces), U=0 = independent routing
        ("K1_U0_bs1", 8, 0, 49, 1, 1, False),
        ("K2_U12", 8, 12, 49, 2, 2, False), ("K2_U0", 8, 0, 49, 2, 2, False),
        ("K3_U15", 8, 15, 49, 4, 3, False), ("K3_U0", 8, 0, 49, 4, 3, False),
        ("K4_U17", 8, 17, 49, 4, 4, False), ("K4_U0", 8, 0, 49, 4, 4, False),
        ("K5_U19", 8, 19, 49, 8, 5, False), ("K5_U0", 8, 0, 49, 8, 5, False),
        ("K6_U21", 8, 21, 49, 8, 6, False),
        ("K7_U23", 8, 23, 49, 8, 7, False),
        ("K8_U24", 8, 24, 49, 8, 8, False), ("K8_U0", 8, 0, 49, 8, 8, False),
        ("K8_U8_lowerbound", 8, 8, 49, 8, 8, False),
    ],
}


@app.function(gpu="H100", timeout=3300, scaledown_window=2, max_containers=1,
              volumes={"/weights": weights_vol, "/jit": jit_vol, "/results": res_vol}, memory=65536)
def run(phase: str, only: str = ""):
    t0 = time.time()
    outp = f"/results/mk_selfspec_{phase}.jsonl"
    for (name, R, U, L, bs, K, real) in PHASES[phase]:
        if only and name not in only.split(","):
            continue
        set_variant(R, U)
        args = f"--batch-size {bs} --max-new-tokens 256 "
        if K is not None and K < bs:
            args += "--ragged-batch --fake-prompt-len " + " ".join(["2048"] * K + ["-1"] * (bs - K)) + " "
        else:
            args += "--fake-prompt-len 2048 "
        if real:
            args += f"--real-weight --checkpoint {CKPT} "
        out, rc, secs = sh(COMMON + args + "2>&1 | grep -E \"metrics|mean/step|tok/s|rror|Traceback|unsupported\" | tail -12", 900,
                           {"MK_LAYERS": str(L)})
        m = re.search(r"'tpot_ms': ([0-9.]+)", out)
        j = re.search(r"'jit_ms': ([0-9.]+)", out)
        ms = re.search(r"mean/step\s+([0-9.]+) ms", out)
        rec = dict(name=name, R=R, U=U, layers=L, bs=bs, active_rows=K, real=real,
                   tpot_ms=float(m.group(1)) if m else None,
                   mean_step_ms=float(ms.group(1)) if ms else None, jit_ms=float(j.group(1)) if j else None, rc=rc,
                   wall_s=round(secs, 1), tail=out[-700:] if not m else "")
        print(json.dumps(rec), flush=True)
        with open(outp, "a") as f:
            f.write(json.dumps(rec) + "\n")
        res_vol.commit()
        jit_vol.commit()
    return time.time() - t0


@app.local_entrypoint()
def main(phase: str = "A", only: str = ""):
    secs = run.remote(phase, only)
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    with open(ROOT / "results/SPEND.md", "a") as f:
        f.write(f"| {ts} | mk_selfspec phase {phase} | H100 | {secs:.0f} | {secs/3600*RATE_H100:.2f} | |\n")
    print("secs", secs, "est $%.2f" % (secs / 3600 * RATE_H100))
