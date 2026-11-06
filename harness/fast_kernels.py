"""Triton decode (T=1) kernels for the fast North executor. All are CUDA-graph safe (no host sync; the position is read
from a device scalar; split ranges are computed on device, so ONE captured graph serves every position and attention
reads only the LIVE keys).  Numerics mirror the HF/GraphNorth bf16 rounding points (see comments)."""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------- experts: decode-specialised GEMV for top-k at T=1
@triton.jit
def _gateup_kernel(x_ptr, idx_ptr, w_ptr, out_ptr, K: tl.constexpr, I: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pe = tl.program_id(0)
    pn = tl.program_id(1)
    e = tl.load(idx_ptr + pe).to(tl.int64)
    rn = pn * BN + tl.arange(0, BN)
    base = w_ptr + e * (2 * I * K)
    ag = tl.zeros([BN], dtype=tl.float32)
    au = tl.zeros([BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + rk).to(tl.float32)
        wg = tl.load(base + rn[:, None] * K + rk[None, :]).to(tl.float32)
        wu = tl.load(base + (rn[:, None] + I) * K + rk[None, :]).to(tl.float32)
        ag += tl.sum(wg * x[None, :], axis=1)
        au += tl.sum(wu * x[None, :], axis=1)
    g = ag.to(tl.bfloat16).to(tl.float32)   # HF: bf16 matmul output
    u = au.to(tl.bfloat16).to(tl.float32)
    s = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)  # bf16 silu
    tl.store(out_ptr + pe * I + rn, (s * u).to(tl.bfloat16))


@triton.jit
def _down_kernel(h_ptr, idx_ptr, sc_ptr, w_ptr, out_ptr, K: tl.constexpr, N: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pe = tl.program_id(0)
    pn = tl.program_id(1)
    e = tl.load(idx_ptr + pe).to(tl.int64)
    sc = tl.load(sc_ptr + pe).to(tl.float32)
    rn = pn * BN + tl.arange(0, BN)
    base = w_ptr + e * (N * K)
    acc = tl.zeros([BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        h = tl.load(h_ptr + pe * K + rk).to(tl.float32)
        w = tl.load(base + rn[:, None] * K + rk[None, :]).to(tl.float32)
        acc += tl.sum(w * h[None, :], axis=1)
    o = acc.to(tl.bfloat16).to(tl.float32)  # bf16 proj_out, then * bf16 weight -> bf16 (HF order)
    tl.store(out_ptr + pe * N + rn, (o * sc).to(tl.bfloat16))


def moe_gemv(x, idx, sc, gate_up, down, hbuf, obuf, cfg=None):
    """x [Dm] bf16, idx [k] int, sc [k] bf16, gate_up [E,2I,Dm], down [E,Dm,I] -> obuf [k,Dm] bf16 (weighted per-expert outs)."""
    c = dict(BN1=16, BK1=256, BN2=16, BK2=256, nw1=4, nw2=4)
    if cfg:
        c.update(cfg)
    k = idx.shape[-1]
    Dm, I = x.shape[-1], down.shape[-1]
    _gateup_kernel[(k, I // c["BN1"])](x, idx, gate_up, hbuf, K=Dm, I=I, BN=c["BN1"], BK=c["BK1"], num_warps=c["nw1"])
    _down_kernel[(k, Dm // c["BN2"])](hbuf, idx, sc, down, obuf, K=I, N=Dm, BN=c["BN2"], BK=c["BK2"], num_warps=c["nw2"])
    return obuf


# ---------------------------------------------------------------- router top-k (+sigmoid) on bf16 logits
@triton.jit
def _topk_kernel(l_ptr, idx_ptr, sc_ptr, N: tl.constexpr, KK: tl.constexpr):
    r = tl.arange(0, N)
    v = tl.load(l_ptr + r).to(tl.float32)
    for j in tl.static_range(KK):
        m = tl.max(v, axis=0)
        i = tl.min(tl.where(v == m, r, N), axis=0)
        tl.store(idx_ptr + j, i)
        tl.store(sc_ptr + j, tl.sigmoid(m).to(tl.bfloat16))
        v = tl.where(r == i, float("-inf"), v)


def router_topk(logits, idx, sc, k):
    _topk_kernel[(1,)](logits, idx, sc, N=logits.shape[-1], KK=k, num_warps=1)


# ---------------------------------------------------------------- fused residual (x + a + sum(m rows)) + next RMSNorm
@triton.jit
def _resid_norm_kernel(x_ptr, a_ptr, m_ptr, w_ptr, xo_ptr, ho_ptr, eps, D: tl.constexpr, NB: tl.constexpr):
    r = tl.arange(0, D)
    x = tl.load(x_ptr + r).to(tl.float32)
    a = tl.load(a_ptr + r).to(tl.float32)
    t = (x + a).to(tl.bfloat16).to(tl.float32)            # (x + a) rounded to bf16, like torch
    ms = tl.zeros([D], dtype=tl.float32)
    for b in tl.static_range(NB):
        ms += tl.load(m_ptr + b * D + r).to(tl.float32)
    ms = ms.to(tl.bfloat16).to(tl.float32)                # bf16 sum over experts
    xn = (t + ms).to(tl.bfloat16)
    tl.store(xo_ptr + r, xn)
    xf = xn.to(tl.float32)
    var = tl.sum(xf * xf, axis=0) / D
    hn = (xf * tl.rsqrt(var + eps)).to(tl.bfloat16).to(tl.float32)   # HF: hidden.to(fp32)*rsqrt -> bf16
    w = tl.load(w_ptr + r).to(tl.float32)
    tl.store(ho_ptr + r, (w * hn).to(tl.bfloat16))


def resid_norm(x, a, m, w, xo, ho, eps):
    D = x.shape[-1]
    _resid_norm_kernel[(1,)](x, a, m, w, xo, ho, eps, D=D, NB=m.numel() // D, num_warps=8)


# ---------------------------------------------------------------- attention: rope + KV write
@triton.jit
def _rope_write_kernel(qkv_ptr, cos_ptr, sin_ptr, start_ptr, kc_ptr, vc_ptr, qo_ptr, Lmax,
                       G: tl.constexpr, D: tl.constexpr, QD: tl.constexpr, KD: tl.constexpr, ROPE: tl.constexpr):
    h = tl.program_id(0)
    start = tl.load(start_ptr)
    d = tl.arange(0, D)
    dp = d ^ 1
    sgn = tl.where((d % 2) == 0, -1.0, 1.0)
    cs = tl.load(cos_ptr + d).to(tl.float32)
    sn = tl.load(sin_ptr + d).to(tl.float32)
    for g in tl.static_range(G):
        off = (h * G + g) * D
        q = tl.load(qkv_ptr + off + d).to(tl.float32)
        if ROPE:
            qp = tl.load(qkv_ptr + off + dp).to(tl.float32)
            q = q * cs + sgn * qp * sn
        tl.store(qo_ptr + off + d, q.to(tl.bfloat16))
    ko = QD + h * D
    k = tl.load(qkv_ptr + ko + d).to(tl.float32)
    if ROPE:
        kp = tl.load(qkv_ptr + ko + dp).to(tl.float32)
        k = k * cs + sgn * kp * sn
    v = tl.load(qkv_ptr + QD + KD + h * D + d)
    dst = (h * Lmax + start) * D + d
    tl.store(kc_ptr + dst, k.to(tl.bfloat16))
    tl.store(vc_ptr + dst, v)


# ---------------------------------------------------------------- attention: split-K flash-decode over LIVE keys only
@triton.jit
def _attn_split_kernel(q_ptr, kc_ptr, vc_ptr, start_ptr, wacc_ptr, wm_ptr, wl_ptr, Lmax, scale,
                       G: tl.constexpr, D: tl.constexpr, NSPLIT: tl.constexpr, BN: tl.constexpr, SLIDING: tl.constexpr, W: tl.constexpr):
    kvh = tl.program_id(0)
    sp = tl.program_id(1)
    start = tl.load(start_ptr)
    hi_all = start + 1
    if SLIDING:
        lo_all = tl.maximum(0, start - W + 1)
    else:
        lo_all = start * 0
    n = hi_all - lo_all
    chunk = ((n + NSPLIT - 1) // NSPLIT + BN - 1) // BN * BN
    lo = lo_all + sp * chunk
    hi = tl.minimum(lo + chunk, hi_all)
    rg = tl.arange(0, 16)
    rd = tl.arange(0, D)
    q = tl.load(q_ptr + (kvh * G + rg)[:, None] * D + rd[None, :], mask=(rg < G)[:, None], other=0.0)
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
        sc = tl.where(mk[None, :], sc, -1e30)
        m_new = tl.maximum(m_i, tl.max(sc, axis=1))
        p = tl.exp(sc - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(vbase + rk[:, None] * D + rd[None, :], mask=mk[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    o = (kvh * NSPLIT + sp) * 16
    tl.store(wm_ptr + o + rg, m_i)
    tl.store(wl_ptr + o + rg, l_i)
    tl.store(wacc_ptr + (o + rg)[:, None] * D + rd[None, :], acc)


@triton.jit
def _attn_combine_kernel(wacc_ptr, wm_ptr, wl_ptr, out_ptr, G: tl.constexpr, D: tl.constexpr, NSPLIT: tl.constexpr):
    hq = tl.program_id(0)
    kvh = hq // G
    g = hq % G
    rs = tl.arange(0, NSPLIT)
    rd = tl.arange(0, D)
    row = (kvh * NSPLIT + rs) * 16 + g
    m = tl.load(wm_ptr + row)
    l = tl.load(wl_ptr + row)
    M = tl.max(m, axis=0)
    w = tl.exp(m - M)
    L = tl.sum(w * l, axis=0)
    acc = tl.load(wacc_ptr + row[:, None] * D + rd[None, :])
    o = tl.sum(acc * w[:, None], axis=0) / L
    tl.store(out_ptr + hq * D + rd, o.to(tl.bfloat16))


class AttnWS:
    def __init__(self, Hkv, G, D, nsplit, dev):
        self.Hkv, self.G, self.D, self.nsplit = Hkv, G, D, nsplit
        self.acc = torch.empty(Hkv * nsplit * 16, D, device=dev, dtype=torch.float32)
        self.m = torch.empty(Hkv * nsplit * 16, device=dev, dtype=torch.float32)
        self.l = torch.empty(Hkv * nsplit * 16, device=dev, dtype=torch.float32)
        self.q = torch.empty(Hkv * G * D, device=dev, dtype=torch.bfloat16)
        self.o = torch.empty(Hkv * G * D, device=dev, dtype=torch.bfloat16)


def decode_attention(ws: AttnWS, qkv, cos, sin, start, kc, vc, Lmax, rope, sliding, window, BN=64):
    Hkv, G, D, ns = ws.Hkv, ws.G, ws.D, ws.nsplit
    _rope_write_kernel[(Hkv,)](qkv, cos, sin, start, kc, vc, ws.q, Lmax, G=G, D=D, QD=Hkv * G * D, KD=Hkv * D, ROPE=rope, num_warps=1)
    _attn_split_kernel[(Hkv, ns)](ws.q, kc, vc, start, ws.acc, ws.m, ws.l, Lmax, D ** -0.5,
                                   G=G, D=D, NSPLIT=ns, BN=BN, SLIDING=sliding, W=window, num_warps=4)
    _attn_combine_kernel[(Hkv * G,)](ws.acc, ws.m, ws.l, ws.o, G=G, D=D, NSPLIT=ns, num_warps=4)
    return ws.o
