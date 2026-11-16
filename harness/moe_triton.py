"""The sass-lane Triton port of moe_fused (magni-sass PR #4, triton/ns_triton.py at b61905b) on North's decode path.

Same algorithm and work list as src/kvm/moe_fused.cu (harness/moe_cuda.WorkTables): one program per (expert, 64-wide slice
of I) streams that slice's gate/up rows once for every token routed to the expert, then does the slice's partial down
projection. Adapted here to read North's HF expert tensors in place: gate_up [E, 2I, H] (up rows after the gate rows),
down [E, H, I] (the [SL, BN] tile is read with stride I). Two ways to combine the 12 slices x k experts per token:
  det=False  fp32 atomic_add into out (the lane's kernel): fastest, order nondeterministic -> non-exact executor only
  det=True   each program stores its weighted partial to part[pair, slice, :]; a second kernel sums a token's k x 12
             partials in a FIXED order (slot j, then slice s). Deterministic, and a row's value does not depend on the
             other rows of its tile (tl.dot rows are independent), so T=1 and verify-block rows agree: exact-path safe.
Tile config per chip from the lane's sweep (traces/sass-lane/<chip>/tri-w1..w3-20261116): BM 16, BK 128, BN 128,
4 warps, 4 stages (3 on sm_120, 99 KB of shared memory)."""
import torch
import triton
import triton.language as tl

from moe_cuda import WorkTables

SL = 64


@triton.jit
def _moe_port(x_ptr, gu_ptr, dn_ptr, task_ptr, prow_ptr, pw_ptr, out_ptr, part_ptr, idx_ptr, sc_ptr, pidx_ptr,
              H: tl.constexpr, I: tl.constexpr, SL: tl.constexpr, NS: tl.constexpr, BM: tl.constexpr,
              BK: tl.constexpr, BN: tl.constexpr, DET: tl.constexpr, T1: tl.constexpr = 0, FUSED: tl.constexpr = 0):
    pid = tl.program_id(0)
    if T1:   # T == 1: program (slot j, slice s) reads its expert and weight from the router output, no task table
        e = tl.load(idx_ptr + pid // NS)
        s = pid % NS
        off = pid // NS
        cnt = 1
    else:
        e = tl.load(task_ptr + pid * 8 + 0)
        s = tl.load(task_ptr + pid * 8 + 1)
        off = tl.load(task_ptr + pid * 8 + 2)
        cnt = tl.load(task_ptr + pid * 8 + 3)
    rm = tl.arange(0, BM)
    rk = tl.arange(0, BK)
    rj = tl.arange(0, SL)
    rn = tl.arange(0, BN)
    gbase = e.to(tl.int64) * (2 * I * H) + (s * SL).to(tl.int64) * H
    gp = gu_ptr + gbase + rj[None, :] * H + rk[:, None]                 # [BK, SL] view of the slice's gate rows
    up = gp + I * H                                                    # up rows follow the gate rows
    dp = dn_ptr + e.to(tl.int64) * (H * I) + rn[None, :] * I + (s * SL + rj)[:, None]   # [SL, BN] of down [E, H, I]
    for m0 in range(0, cnt, BM):
        mmask = (m0 + rm) < cnt
        if T1:
            rows = tl.zeros((BM,), dtype=tl.int32)
            wts = tl.where(rm == 0, tl.load(sc_ptr + off).to(tl.float32), 0.0)
        else:
            rows = tl.load(prow_ptr + off + m0 + rm, mask=mmask, other=0)
            wts = tl.load(pw_ptr + off + m0 + rm, mask=mmask, other=0.0)
        acc_g = tl.zeros((BM, SL), dtype=tl.float32)
        acc_u = tl.zeros((BM, SL), dtype=tl.float32)
        xp = x_ptr + rows.to(tl.int64)[:, None] * H + rk[None, :]
        for k in range(0, H, BK):
            xa = tl.load(xp + k, mask=mmask[:, None], other=0.0)
            acc_g = tl.dot(xa, tl.load(gp + k), acc_g)
            acc_u = tl.dot(xa, tl.load(up + k), acc_u)
        h = (acc_g / (1.0 + tl.exp(-acc_g)) * acc_u).to(tl.bfloat16)
        for n in range(0, H, BN):
            acc = tl.dot(h, tl.load(dp + n * I)) * wts[:, None]
            if DET:
                if FUSED:   # flat (token*k + slot) index: resid_norm then sums a token's k*NS rows in fixed order
                    if T1:
                        prw = (off + rm * 0).to(tl.int64)
                    else:
                        prw = tl.load(pidx_ptr + off + m0 + rm, mask=mmask, other=0).to(tl.int64)
                else:
                    prw = (off + m0 + rm).to(tl.int64)
                pp = part_ptr + (prw[:, None] * NS + s) * H + n + rn[None, :]
                tl.store(pp, acc, mask=mmask[:, None])
            else:
                op = out_ptr + rows.to(tl.int64)[:, None] * H + n + rn[None, :]
                tl.atomic_add(op, acc, mask=mmask[:, None], sem="relaxed")


@triton.jit
def _union_table(idx_ptr, sc_ptr, task_ptr, prow_ptr, pw_ptr, pidx_ptr, N, K: tl.constexpr, NS: tl.constexpr, NP: tl.constexpr,
                 NSP: tl.constexpr):
    """One program builds the whole T>1 work list (N = T*k <= NP routed pairs): stable rank by expert from an NP x NP
    compare; the first pair of each expert (its leader) owns the group: NS tasks (expert, slice, start, count); every
    other pair's NS tasks get count 0 and exit at once. Grid of the port kernel: N * NS."""
    i = tl.arange(0, NP)
    valid = i < N
    e = tl.load(idx_ptr + i, mask=valid, other=2147483647)
    w = tl.load(sc_ptr + i, mask=valid, other=0.0).to(tl.float32)
    ei = e[:, None]
    ej = e[None, :]
    lt = ej < ei
    eq = ej == ei
    before = i[None, :] < i[:, None]
    rank = tl.sum((lt | (eq & before)).to(tl.int32), axis=1)          # stable sorted position of pair i
    start = tl.sum(lt.to(tl.int32), axis=1)                            # first position of its expert's group
    cnt = tl.sum(eq.to(tl.int32), axis=1)
    lead = tl.sum((eq & before).to(tl.int32), axis=1) == 0
    tl.store(prow_ptr + rank, (i // K).to(tl.int32), mask=valid)
    tl.store(pw_ptr + rank, w, mask=valid)
    tl.store(pidx_ptr + rank, i.to(tl.int32), mask=valid)
    sl = tl.arange(0, NSP)                                             # NS (12) padded to a power of two
    base = task_ptr + (i[:, None] * NS + sl[None, :]) * 8
    m2 = valid[:, None] & (sl[None, :] < NS)
    tl.store(base + 0, tl.broadcast_to(e[:, None], (NP, NSP)), mask=m2)
    tl.store(base + 1, tl.broadcast_to(sl[None, :], (NP, NSP)), mask=m2)
    tl.store(base + 2, tl.broadcast_to(start[:, None], (NP, NSP)), mask=m2)
    tl.store(base + 3, tl.broadcast_to(tl.where(lead, cnt, 0)[:, None], (NP, NSP)), mask=m2)


@triton.jit
def _moe_reduce(part_ptr, inv_ptr, out_ptr, H: tl.constexpr, K: tl.constexpr, NS: tl.constexpr, BN: tl.constexpr):
    t = tl.program_id(0)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for j in tl.static_range(K):
        p = tl.load(inv_ptr + t * K + j).to(tl.int64)
        for s in tl.static_range(NS):
            acc += tl.load(part_ptr + (p * NS + s) * H + cols)
    tl.store(out_ptr + t * H + cols, acc)


def default_cfg():
    mj, _ = torch.cuda.get_device_capability()
    return dict(BM=16, BK=128, BN=128, num_warps=4, num_stages=3 if mj >= 12 else 4)


class MoeTriton(WorkTables):
    """Callable like harness/moe_cuda.MoeCuda: (h [T, H], idx [T, k], sc [T, k], gate_up, down) -> fp32 MoE output.
    fused=True (with det): no reduce kernel, no zeroing, no torch work-list ops: returns the partials [T*k*NS, H] in flat
    (token, slot, slice) order, which fast_kernels.resid_norm sums per token in fixed order (its NB = k*NS). T == 1 reads
    the router output directly (one launch); T > 1 builds its work list with one _union_table launch."""

    def __init__(self, dm, i_dim, k, dev, det=False, cfg=None, tmax=16, fused=False, reduce="resid", sl=SL):
        super().__init__(dm, i_dim // sl, k, dev)
        self.sl = sl                                       # columns of I per program: 64 (the lane's) or 32 (shorter CTAs)
        self.i_dim, self.det = i_dim, det
        self.reduce = "atomic" if not det else reduce     # fused: "resid" (resid_norm sums the partials) | "kernel" (_moe_reduce) | "atomic"
        self.fused = fused
        self.cfg = dict(default_cfg(), **(cfg or {}))
        self.part = torch.zeros(tmax * k * self.ns, dm, dtype=torch.float32, device=dev) if det else torch.zeros(1, device=dev)
        self.RB = 256
        n = tmax * k
        self.ftasks = torch.zeros(n * self.ns, 8, dtype=torch.int32, device=dev)
        self.fprow = torch.zeros(n, dtype=torch.int32, device=dev)
        self.fpw = torch.zeros(n, dtype=torch.float32, device=dev)
        self.fpidx = torch.zeros(n, dtype=torch.int32, device=dev)
        self.NP = triton.next_power_of_2(n)
        self.iota = torch.arange(n, dtype=torch.int32, device=dev)          # flat partial layout: pair p sits at row p
        self.fout = torch.zeros(tmax, dm, dtype=torch.float32, device=dev)

    def _launch(self, grid, h, gate_up, down, tasks, prow, pw, out, idx, sc, pidx, t1, fused):
        c = self.cfg
        _moe_port[grid](h, gate_up, down, tasks, prow, pw, out, self.part, idx, sc, pidx, H=self.dm, I=self.i_dim, SL=self.sl, NS=self.ns,
                        BM=c["BM"], BK=c["BK"], BN=c["BN"], DET=self.det, T1=t1, FUSED=fused, num_warps=c["num_warps"], num_stages=c["num_stages"])

    def __call__(self, h, idx, sc, gate_up, down):
        assert h.is_contiguous() and gate_up.is_contiguous() and down.is_contiguous()
        if self.fused:
            T = idx.shape[0] if idx.dim() == 2 else 1
            n = T * self.k
            out = self.fout[:T]
            if self.reduce == "atomic":
                out.zero_()
            if T > 1:
                _union_table[(1,)](idx, sc, self.ftasks, self.fprow, self.fpw, self.fpidx, n, K=self.k, NS=self.ns, NP=self.NP,
                                   NSP=triton.next_power_of_2(self.ns), num_warps=4)
            self._launch((n * self.ns,), h, gate_up, down, self.ftasks, self.fprow, self.fpw, out if self.reduce == "atomic" else self.part,
                         idx, sc, self.fpidx, int(T == 1), 1)
            if self.reduce == "resid":
                return self.part[: n * self.ns]
            if self.reduce == "kernel":
                _moe_reduce[(T, self.dm // self.RB)](self.part, self.iota, out, H=self.dm, K=self.k, NS=self.ns, BN=self.RB, num_warps=4)
            return out
        tasks, prow, pw, out, n, inv, T = self.tables(idx, sc)
        if not self.det:
            out.zero_()
        self._launch((n,), h, gate_up, down, tasks, prow, pw, out, idx, sc, self.fpidx, 0, 0)
        if self.det:
            _moe_reduce[(T, self.dm // self.RB)](self.part, inv, out, H=self.dm, K=self.k, NS=self.ns, BN=self.RB, num_warps=4)
        return out


def recommended(dm, i_dim, k, dev, Lmax, exact):
    """The measured per-chip choice (RESULTS H32, H100 / B200 / RTX PRO 6000), or None where north-star's own Triton MoE is as
    fast or faster. exact=True: deterministic only (fixed-order reduce). Opt-in: FastNorth.moe_cuda = recommended(...)."""
    mj, _ = torch.cuda.get_device_capability()
    if exact:
        if mj == 10:   # B200: 1.028x at ctx 2048 (SL 64), 1.037x at ctx 8192 (SL 32, shorter CTAs leave room for the attention stream)
            return MoeTriton(dm, i_dim, k, dev, det=True, fused=True, reduce="kernel", sl=64 if Lmax <= 4096 else 32)
        return None    # H100: tie at 2048, 0.95x at 8192; RTX PRO 6000: exact v2 itself does not fit (lin_x2 shared memory)
    if mj in (9, 10, 12):  # non-exact executor: fp32 atomics, in-kernel work lists
        return MoeTriton(dm, i_dim, k, dev, det=False, fused=True, sl=32 if (mj == 10 and Lmax > 4096) else 64)
    return None
