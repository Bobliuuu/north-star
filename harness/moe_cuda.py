"""T=1 decode MoE through src/kvm/moe_fused.cu (sass-lane per-arch build), for FastNorth's x2 path (`FastNorth.moe_cuda`).

The kernel reads North's HF expert tensors in place (NS_NORTH_LAYOUT) and accumulates the routed, weighted expert
outputs into one fp32 row with atomicAdd, which `fast_kernels.resid_norm` then adds (its NB == 1). Work items are
(expert, 64-column slice) pairs: TOPK x I/64 = 96 per token. The task table is static except its expert column,
which is refreshed from the router's top-k on the GPU, so the whole step stays inside one CUDA graph.
Numerics differ from the Triton path: gate/up/silu stay fp32 (HF rounds them to bf16) and the fp32 atomics have no
fixed order, so this path is NOT exact; bench_moecuda.py measures token agreement and run-to-run determinism."""
import ctypes
import os
import subprocess

import torch

KVM_NO_VAR = -1
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("NS_KVM_SRC", os.path.join(HERE, "..", "src", "kvm"))


def build(out_dir, arch=None, defines=()):
    """nvcc -> shared library; returns its path. `defines`: extra -D flags (e.g. NS_TUNE_UPSTREAM for the A/B baseline)."""
    if arch is None:
        mj, mn = torch.cuda.get_device_capability()
        arch = f"sm_{mj}{mn}"
    tag = "_".join([arch] + [d.replace("=", "") for d in defines]) or arch
    so = os.path.join(out_dir, f"libnsmoe_{tag}.so")
    cmd = ["nvcc", "-O3", "-std=c++17", f"-arch={arch}", "-shared", "-Xcompiler", "-fPIC", "-Xptxas", "-v",
           "-I", SRC, "-I", os.path.join(SRC, "include")] + [f"-D{d}" for d in defines] + [os.path.join(SRC, "moe_cuda_lib.cu"), "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-3000:])
    return so, [l for l in r.stderr.splitlines() if "registers" in l]


class MoeCuda:
    def __init__(self, so, dm, i_dim, k, dev):
        self.lib = ctypes.CDLL(so)
        h, i, tk, sl = (ctypes.c_int() for _ in range(4))
        self.lib.ns_moe_dims(ctypes.byref(h), ctypes.byref(i), ctypes.byref(tk), ctypes.byref(sl))
        if (h.value, i.value, tk.value) != (dm, i_dim, k):
            raise ValueError(f"kernel built for H={h.value} I={i.value} TOPK={tk.value}, model has {dm}/{i_dim}/{k}")
        self.ns = i.value // sl.value
        self.fn = self.lib.ns_moe_launch
        self.fn.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int, ctypes.c_void_p]
        self.fn.restype = ctypes.c_int
        n = k * self.ns
        t = torch.empty(n, 8, dtype=torch.int32, device=dev)
        j = torch.arange(n, device=dev, dtype=torch.int32)
        t[:, 0] = 0
        t[:, 1] = j % self.ns                 # slice
        t[:, 2] = j // self.ns                # pair_off: the token's top-k slot
        t[:, 3] = 1                           # pair_cnt: one token (T == 1)
        t[:, 4] = KVM_NO_VAR; t[:, 5] = 0; t[:, 6] = KVM_NO_VAR; t[:, 7] = 0
        self.tasks, self.n = t, n
        self.prow = torch.zeros(k, dtype=torch.int32, device=dev)
        self.pw = torch.zeros(k, dtype=torch.float32, device=dev)
        self.out = torch.zeros(1, dm, dtype=torch.float32, device=dev)

    def __call__(self, h, idx, sc, gate_up, down):
        assert h.is_contiguous() and gate_up.is_contiguous() and down.is_contiguous()
        self.tasks[:, 0].copy_(idx.reshape(-1).to(torch.int32).repeat_interleave(self.ns))
        self.pw.copy_(sc.reshape(-1))
        self.out.zero_()
        rc = self.fn(h.data_ptr(), gate_up.data_ptr(), down.data_ptr(), self.tasks.data_ptr(), self.prow.data_ptr(),
                     self.pw.data_ptr(), self.out.data_ptr(), self.n, torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f"moe_kernel launch failed: cudaError {rc}")
        return self.out
