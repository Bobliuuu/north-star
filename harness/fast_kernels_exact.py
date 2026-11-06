"""BATCH-INVARIANT kernels for the exact-speculation path of FastNorth (branch exact-spec).

Goal: the logits of the row for token position p are BITWISE identical whether p is computed alone (T=1 plain step) or as any row of a
T=2..16 verify block. Rules used everywhere:
  * every output element is reduced by ONE fixed program shape (tile sizes depend on the weight shape only, never on T / start);
  * tensor-core tl.dot with the query/token axis padded to a FIXED size (16 tokens; MPF attention rows) so the instruction and the
    K-reduction order never change with T (MMA outputs do not depend on other rows);
  * attention splits at ABSOLUTE key positions (chunk C), not at start-relative ones, merged by a fixed-shape power-of-2 combine;
    keys beyond a row's position are masked (p == 0 exactly, alpha == 1 exactly), so extra/skipped blocks cannot change a row;
  * argmax picks the lowest index on exact bf16 ties, with a row-independent 2-stage reduction.
cuBLAS / torch reductions (qkv, o_proj, router, dense MLP, lm_head, first RMSNorm, argmax) are replaced by these kernels."""
import torch
import triton
import triton.language as tl
import pdl_flag as pdl

TP = 16        # token axis padding of the dot (T <= 16)
CHUNK = 256    # absolute key chunk of one attention split program


# ---------------------------------------------------------------- batch-invariant linear: out[t,n] = sum_k x[t,k] w[n,k]  (bf16 out, fp32 acc)
@triton.jit
def _lin_kernel(x_ptr, w_ptr, o_ptr, T: tl.constexpr, N: tl.constexpr, K: tl.constexpr, TPAD: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    pn = tl.program_id(0)
    rn = pn * BN + tl.arange(0, BN)
    rt = tl.arange(0, TPAD)
    acc = tl.zeros([BN, TPAD], dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        xt = tl.load(x_ptr + rt[None, :] * K + rk[:, None], mask=(rt < T)[None, :], other=0.0)
        w = tl.load(w_ptr + rn[:, None] * K + rk[None, :], mask=(rn < N)[:, None], other=0.0)
        acc = tl.dot(w, xt, acc)
    tl.store(o_ptr + rt[None, :] * N + rn[:, None], acc.to(tl.bfloat16), mask=(rn < N)[:, None] & (rt < T)[None, :])


LIN_CFG = {}   # (N,K) -> dict(BN,BK,nw,ns); shape-keyed only


def lin_cfg(N, K):
    if (N, K) in LIN_CFG:
        return LIN_CFG[(N, K)]
    return dict(BN=16 if N <= 4096 else 64, BK=256 if K <= 2048 else 128, nw=4, ns=3)


def lin(x, w, out=None):
    """x [T,K] bf16 contiguous, w [N,K] bf16 -> [T,N] bf16."""
    T, K = x.shape
    N = w.shape[0]
    if out is None:
        out = torch.empty(T, N, device=x.device, dtype=torch.bfloat16)
    c = lin_cfg(N, K)
    _lin_kernel[(triton.cdiv(N, c["BN"]),)](x, w, out, T=T, N=N, K=K, TPAD=TP, BN=c["BN"], BK=c["BK"], num_warps=c["nw"], num_stages=c["ns"], **pdl.kw())
    return out


# ---------------------------------------------------------------- row argmax, lowest index on ties
@triton.jit
def _amax1(l_ptr, pm_ptr, pi_ptr, V: tl.constexpr, CH: tl.constexpr, NCH: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0)
    c = tl.program_id(1)
    r = c * CH + tl.arange(0, CH)
    v = tl.load(l_ptr + row * V + r, mask=r < V, other=float("-inf")).to(tl.float32)
    m = tl.max(v, axis=0)
    i = tl.min(tl.where(v == m, r, 2147483647), axis=0)
    tl.store(pm_ptr + row * NCH + c, m)
    tl.store(pi_ptr + row * NCH + c, i)


@triton.jit
def _amax2(pm_ptr, pi_ptr, o_ptr, NCH: tl.constexpr, NP: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0)
    r = tl.arange(0, NP)
    m = tl.load(pm_ptr + row * NCH + r, mask=r < NCH, other=float("-inf"))
    i = tl.load(pi_ptr + row * NCH + r, mask=r < NCH, other=2147483647)
    M = tl.max(m, axis=0)
    tl.store(o_ptr + row, tl.min(tl.where(m == M, i, 2147483647), axis=0).to(tl.int64))


def argmax_rows(logits):
    """logits [T,V] -> [T] int64 (lowest index among exact ties)."""
    T, V = logits.shape
    CH = 4096
    NCH = triton.cdiv(V, CH)
    pm = torch.empty(T, NCH, device=logits.device, dtype=torch.float32)
    pi = torch.empty(T, NCH, device=logits.device, dtype=torch.int32)
    o = torch.empty(T, device=logits.device, dtype=torch.int64)
    _amax1[(T, NCH)](logits, pm, pi, V=V, CH=CH, NCH=NCH, num_warps=4, **pdl.kw())
    _amax2[(T,)](pm, pi, o, NCH=NCH, NP=triton.next_power_of_2(NCH), num_warps=1, **pdl.kw())
    return o


# ---------------------------------------------------------------- attention: absolute-chunk split-K, fixed MP rows
@triton.jit
def _attn_split_x(q_ptr, kc_ptr, vc_ptr, start_ptr, wacc_ptr, wm_ptr, wl_ptr, Lmax, scale,
                  T: tl.constexpr, G: tl.constexpr, D: tl.constexpr, MP: tl.constexpr, NS: tl.constexpr, BN: tl.constexpr, C: tl.constexpr,
                  SLIDING: tl.constexpr, W: tl.constexpr):
    kvh = tl.program_id(0)
    sp = tl.program_id(1)
    start = tl.load(start_ptr)
    rm = tl.arange(0, MP)
    rd = tl.arange(0, D)
    o = (kvh * NS + sp) * MP
    lo = sp * C
    hi = tl.minimum(lo + C, start + T)
    dead = lo >= start + T
    if SLIDING:
        dead = dead | (lo + C <= start - W + 1)
    rowok = rm < T * G
    if dead:
        tl.store(wm_ptr + o + rm, tl.zeros([MP], dtype=tl.float32) - 1e30, mask=rowok)
        tl.store(wl_ptr + o + rm, tl.zeros([MP], dtype=tl.float32), mask=rowok)
    else:
        q = tl.load(q_ptr + (kvh * MP + rm)[:, None] * D + rd[None, :])
        qpos = start + rm // G
        m_i = tl.zeros([MP], dtype=tl.float32) - 1e30
        l_i = tl.zeros([MP], dtype=tl.float32)
        acc = tl.zeros([MP, D], dtype=tl.float32)
        kbase = kc_ptr + kvh * Lmax * D
        vbase = vc_ptr + kvh * Lmax * D
        for s in range(lo, hi, BN):
            rk = s + tl.arange(0, BN)
            mk = rk < hi
            k = tl.load(kbase + rk[:, None] * D + rd[None, :], mask=mk[:, None], other=0.0)
            sc = tl.dot(q, tl.trans(k)) * scale
            ok = mk[None, :] & (rk[None, :] <= qpos[:, None])
            if SLIDING:
                ok = ok & (rk[None, :] > qpos[:, None] - W)
            sc = tl.where(ok, sc, -1e30)
            m_new = tl.maximum(m_i, tl.max(sc, axis=1))
            p = tl.exp(sc - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v = tl.load(vbase + rk[:, None] * D + rd[None, :], mask=mk[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
            m_i = m_new
        tl.store(wm_ptr + o + rm, m_i, mask=rowok)
        tl.store(wl_ptr + o + rm, l_i, mask=rowok)
        tl.store(wacc_ptr + (o + rm)[:, None] * D + rd[None, :], acc, mask=rowok[:, None])


# ---- row-group variant: each program handles a FIXED 16 query rows (grid dim 2 = row group), so T=1 launches 1 group instead of a padded 128-row tile.
# Every row is computed by the same 16-row program shape for any T, so row results do not depend on T (bitwise test: modal/exact_unit.py).
@triton.jit
def _attn_split_x2(q_ptr, kc_ptr, vc_ptr, start_ptr, wacc_ptr, wm_ptr, wl_ptr, Lmax, scale,
                   T: tl.constexpr, G: tl.constexpr, D: tl.constexpr, MPT: tl.constexpr, NS: tl.constexpr, BN: tl.constexpr, C: tl.constexpr,
                   SLIDING: tl.constexpr, W: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    kvh = tl.program_id(0)
    sp = tl.program_id(1)
    rg = tl.program_id(2)
    start = tl.load(start_ptr)
    rm = rg * 16 + tl.arange(0, 16)
    rd = tl.arange(0, D)
    o = (kvh * NS + sp) * MPT
    lo = sp * C
    hi = tl.minimum(lo + C, start + T)
    dead = lo >= start + T
    if SLIDING:
        dead = dead | (lo + C <= start - W + 1)
    rowok = rm < T * G
    if dead:
        tl.store(wm_ptr + o + rm, tl.zeros([16], dtype=tl.float32) - 1e30, mask=rowok)
        tl.store(wl_ptr + o + rm, tl.zeros([16], dtype=tl.float32), mask=rowok)
    else:
        q = tl.load(q_ptr + (kvh * MPT + rm)[:, None] * D + rd[None, :])
        qpos = start + rm // G
        m_i = tl.zeros([16], dtype=tl.float32) - 1e30
        l_i = tl.zeros([16], dtype=tl.float32)
        acc = tl.zeros([16, D], dtype=tl.float32)
        kbase = kc_ptr + kvh * Lmax * D
        vbase = vc_ptr + kvh * Lmax * D
        for s in range(lo, hi, BN):
            rk = s + tl.arange(0, BN)
            mk = rk < hi
            k = tl.load(kbase + rk[:, None] * D + rd[None, :], mask=mk[:, None], other=0.0)
            sc = tl.dot(q, tl.trans(k)) * scale
            ok = mk[None, :] & (rk[None, :] <= qpos[:, None])
            if SLIDING:
                ok = ok & (rk[None, :] > qpos[:, None] - W)
            sc = tl.where(ok, sc, -1e30)
            m_new = tl.maximum(m_i, tl.max(sc, axis=1))
            p = tl.exp(sc - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v = tl.load(vbase + rk[:, None] * D + rd[None, :], mask=mk[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
            m_i = m_new
        tl.store(wm_ptr + o + rm, m_i, mask=rowok)
        tl.store(wl_ptr + o + rm, l_i, mask=rowok)
        tl.store(wacc_ptr + (o + rm)[:, None] * D + rd[None, :], acc, mask=rowok[:, None])


@triton.jit
def _attn_combine_x(wacc_ptr, wm_ptr, wl_ptr, out_ptr, T: tl.constexpr, G: tl.constexpr, Hkv: tl.constexpr, D: tl.constexpr,
                    MP: tl.constexpr, NS: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(0)
    kvh = pid // (T * G)
    r = pid % (T * G)
    t = r // G
    g = r % G
    rs = tl.arange(0, NS)
    rd = tl.arange(0, D)
    row = (kvh * NS + rs) * MP + r
    m = tl.load(wm_ptr + row)
    live = m > -1e29
    l = tl.load(wl_ptr + row, mask=live, other=0.0)
    M = tl.max(m, axis=0)
    w = tl.where(live, tl.exp(m - M), 0.0)
    L = tl.sum(w * l, axis=0)
    acc = tl.load(wacc_ptr + row[:, None] * D + rd[None, :], mask=live[:, None], other=0.0)
    o = tl.sum(acc * w[:, None], axis=0) / L
    tl.store(out_ptr + t * (Hkv * G * D) + (kvh * G + g) * D + rd, o.to(tl.bfloat16))


class AttnWSX:
    def __init__(self, Hkv, G, D, Lmax, T, dev, tblk=16):
        self.Hkv, self.G, self.D, self.T = Hkv, G, D, T
        self.MP = max(16, triton.next_power_of_2(tblk * G))
        self.NS = max(2, triton.next_power_of_2(triton.cdiv(Lmax, CHUNK)))
        n = Hkv * self.NS * self.MP
        self.acc = torch.zeros(n, D, device=dev, dtype=torch.float32)
        self.m = torch.full((n,), -1e30, device=dev, dtype=torch.float32)
        self.l = torch.zeros(n, device=dev, dtype=torch.float32)
        self.q = torch.zeros(Hkv * self.MP * D, device=dev, dtype=torch.bfloat16)  # padded query rows stay zero
        self.o = torch.empty(T, Hkv * G * D, device=dev, dtype=torch.bfloat16)


def exact_attention(ws: AttnWSX, qkv, cos, sin, start, kc, vc, Lmax, rope, sliding, window, rope_write, BN=64, rowgroup=False, nw=8, cnw=4):
    Hkv, G, D, T, MP, NS = ws.Hkv, ws.G, ws.D, ws.T, ws.MP, ws.NS
    rope_write[(Hkv, T)](qkv, cos, sin, start, kc, vc, ws.q, Lmax, G=G, D=D, QD=Hkv * G * D, KD=Hkv * D, MP=MP, ROPE=rope, num_warps=1, **pdl.kw())
    if rowgroup:
        _attn_split_x2[(Hkv, NS, triton.cdiv(T * G, 16))](ws.q, kc, vc, start, ws.acc, ws.m, ws.l, Lmax, D ** -0.5, T=T, G=G, D=D, MPT=MP, NS=NS, BN=BN, C=CHUNK,
                                                          SLIDING=sliding, W=window, num_warps=nw, **pdl.kw())
    else:
        _attn_split_x[(Hkv, NS)](ws.q, kc, vc, start, ws.acc, ws.m, ws.l, Lmax, D ** -0.5, T=T, G=G, D=D, MP=MP, NS=NS, BN=BN, C=CHUNK,
                                 SLIDING=sliding, W=window, num_warps=8)
    _attn_combine_x[(Hkv * T * G,)](ws.acc, ws.m, ws.l, ws.o, T=T, G=G, Hkv=Hkv, D=D, MP=MP, NS=NS, num_warps=cnw, **pdl.kw())
    return ws.o
