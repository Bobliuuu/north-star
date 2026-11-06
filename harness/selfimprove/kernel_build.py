"""Build the candidate kernel to a SEPARATE binary build/moe_fused (the only place a build may write)."""
import os, shutil, subprocess, sys
WS = os.path.abspath(os.environ.get("NS_WORKSPACE", os.getcwd()))
ARCH = "sm_80"                      # A100; fixed here (protected), not selectable by a candidate
nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
os.makedirs(os.path.join(WS, "build"), exist_ok=True)
cmd = [nvcc, "-O3", "-std=c++17", f"-arch={ARCH}", "-o", os.path.join(WS, "build", "moe_fused"),
       os.path.join(WS, "src", "kvm", "moe_fused.cu")]
r = subprocess.run(cmd, cwd=WS, capture_output=True, text=True)
sys.stderr.write((r.stdout + r.stderr)[-4000:])
sys.exit(r.returncode)
