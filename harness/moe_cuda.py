"""T=1 decode MoE through src/kvm/moe_fused.cu (sass-lane per-arch build), for FastNorth's x2 path (`FastNorth.moe_cuda`).

The kernel reads North's HF expert tensors in place (NS_NORTH_LAYOUT) and accumulates the routed, weighted expert
outputs into one fp32 row with atomicAdd, which `fast_kernels.resid_norm` then adds (its NB == 1). At T=1 the work
items are (expert, 64-column slice) pairs: TOPK x I/64 = 96, from a table that is static except its expert column. For a
T-token verify block (non-exact executor) the table is a fixed-size union: one group per distinct expert with all its
routed tokens (built on the GPU by a stable sort), padding groups exit at once. Either way the step stays one CUDA graph.
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
        self.k, self.dm, self._u = k, dm, {}

    def _union(self, T):
        """Fixed-shape union table for a T-token block: one group per DISTINCT expert, its routed tokens contiguous in
        prow/pw (stable sort by expert), NS slice tasks per group; padding groups have pair_cnt 0 and exit at once."""
        if T not in self._u:
            dev, k, n = self.tasks.device, self.k, T * self.k
            t = torch.zeros(n * self.ns, 8, dtype=torch.int32, device=dev)
            j = torch.arange(n * self.ns, device=dev, dtype=torch.int32)
            t[:, 1] = j % self.ns
            t[:, 4] = KVM_NO_VAR; t[:, 6] = KVM_NO_VAR
            self._u[T] = dict(tasks=t, tok=torch.arange(T, device=dev).repeat_interleave(k), pos=torch.arange(n, device=dev),
                              prow=torch.zeros(n, dtype=torch.int32, device=dev), pw=torch.zeros(n, dtype=torch.float32, device=dev),
                              out=torch.zeros(T, self.dm, dtype=torch.float32, device=dev), n=n)
        return self._u[T]

    def __call__(self, h, idx, sc, gate_up, down):
        assert h.is_contiguous() and gate_up.is_contiguous() and down.is_contiguous()
        T = idx.shape[0] if idx.dim() == 2 else 1
        if T == 1:
            tasks, prow, pw, out, n = self.tasks, self.prow, self.pw, self.out, self.n
            tasks[:, 0].copy_(idx.reshape(-1).to(torch.int32).repeat_interleave(self.ns))
            pw.copy_(sc.reshape(-1))
        else:
            u = self._union(T)
            tasks, prow, pw, out = u["tasks"], u["prow"], u["pw"], u["out"]
            e = idx.reshape(-1).long()
            es, order = torch.sort(e, stable=True)
            prow.copy_(u["tok"][order])
            pw.copy_(sc.reshape(-1).float()[order])
            first = torch.ones_like(es, dtype=torch.bool)
            first[1:] = es[1:] != es[:-1]
            gid = torch.cumsum(first.long(), 0) - 1
            m_ = u["n"]
            cnt = torch.zeros(m_, dtype=torch.long, device=e.device).scatter_add_(0, gid, torch.ones_like(gid))
            start = torch.full((m_,), m_, dtype=torch.long, device=e.device).scatter_reduce_(0, gid, u["pos"], "amin")
            gexp = torch.zeros(m_, dtype=torch.long, device=e.device).scatter_(0, gid, es)
            tasks[:, 0].copy_(gexp.repeat_interleave(self.ns))
            tasks[:, 2].copy_(start.repeat_interleave(self.ns))
            tasks[:, 3].copy_(cnt.repeat_interleave(self.ns))
            n = tasks.shape[0]
        out.zero_()
        rc = self.fn(h.data_ptr(), gate_up.data_ptr(), down.data_ptr(), tasks.data_ptr(), prow.data_ptr(),
                     pw.data_ptr(), out.data_ptr(), n, torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f"moe_kernel launch failed: cudaError {rc}")
        return out
