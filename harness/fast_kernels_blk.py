"""Multi-query VERIFY-block kernels (T = 2..16 query rows at positions start..start+T-1) for the fast North executor.
Same bf16 rounding points as the T=1 kernels in fast_kernels.py (which are untouched); CUDA-graph safe (position from a device scalar).
  * experts: union-of-experts grouped GEMV. One "leader" program per DISTINCT expert (first (token,slot) pair that names it) reads that
    expert's weight tile ONCE and multiplies it against every token that routed to it (tl.dot, tokens padded to 16); other programs exit.
  * attention: rope + KV write for T rows, split-K flash over the LIVE keys only, per-query-row causal (+sliding) mask, combine."""
import torch
import triton
import triton.language as tl


@triton.jit
def _gateup_blk(x_ptr, idx_ptr, w_ptr, out_ptr, K: tl.constexpr, I: tl.constexpr, T: tl.constexpr, KK: tl.constexpr,
                TP: tl.constexpr, TKP: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    p = tl.program_id(0)
    pn = tl.program_id(1)
    e32 = tl.load(idx_ptr + p)
    rp = tl.arange(0, TKP)
    allidx = tl.load(idx_ptr + rp, mask=rp < T * KK, other=-1)
    dup = tl.sum(((allidx == e32) & (rp < p)).to(tl.int32), axis=0)
    if dup == 0:
        e = e32.to(tl.int64)
        rt = tl.arange(0, TP)
        rj = tl.arange(0, KK)
        id2 = tl.load(idx_ptr + rt[:, None] * KK + rj[None, :], mask=(rt < T)[:, None], other=-1)
        match = id2 == e32
        has = tl.sum(match.to(tl.int32), axis=1) > 0
        slot = tl.sum(tl.where(match, rj[None, :], 0), axis=1)
        rn = pn * BN + tl.arange(0, BN)
        base = w_ptr + e * (2 * I * K)
        ag = tl.zeros([BN, TP], dtype=tl.float32)
        au = tl.zeros([BN, TP], dtype=tl.float32)
        for k0 in range(0, K, BK):
            rk = k0 + tl.arange(0, BK)
            xt = tl.load(x_ptr + rt[None, :] * K + rk[:, None], mask=(rt < T)[None, :], other=0.0)
            wg = tl.load(base + rn[:, None] * K + rk[None, :])
            wu = tl.load(base + (rn[:, None] + I) * K + rk[None, :])
            ag = tl.dot(wg, xt, ag)
            au = tl.dot(wu, xt, au)
        g = ag.to(tl.bfloat16).to(tl.float32)
        u = au.to(tl.bfloat16).to(tl.float32)
        s = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
        tl.store(out_ptr + (rt * KK + slot)[None, :] * I + rn[:, None], (s * u).to(tl.bfloat16), mask=has[None, :])


@triton.jit
def _down_blk(h_ptr, idx_ptr, sc_ptr, w_ptr, out_ptr, K: tl.constexpr, N: tl.constexpr, T: tl.constexpr, KK: tl.constexpr,
              TP: tl.constexpr, TKP: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    p = tl.program_id(0)
    pn = tl.program_id(1)
    e32 = tl.load(idx_ptr + p)
    rp = tl.arange(0, TKP)
    allidx = tl.load(idx_ptr + rp, mask=rp < T * KK, other=-1)
    dup = tl.sum(((allidx == e32) & (rp < p)).to(tl.int32), axis=0)
    if dup == 0:
        e = e32.to(tl.int64)
        rt = tl.arange(0, TP)
        rj = tl.arange(0, KK)
        id2 = tl.load(idx_ptr + rt[:, None] * KK + rj[None, :], mask=(rt < T)[:, None], other=-1)
        match = id2 == e32
        has = tl.sum(match.to(tl.int32), axis=1) > 0
        slot = tl.sum(tl.where(match, rj[None, :], 0), axis=1)
        row = rt * KK + slot
        sc = tl.load(sc_ptr + row, mask=has, other=0.0).to(tl.float32)
        rn = pn * BN + tl.arange(0, BN)
        base = w_ptr + e * (N * K)
        acc = tl.zeros([BN, TP], dtype=tl.float32)
        for k0 in range(0, K, BK):
            rk = k0 + tl.arange(0, BK)
            ht = tl.load(h_ptr + row[None, :] * K + rk[:, None], mask=has[None, :], other=0.0)
            w = tl.load(base + rn[:, None] * K + rk[None, :])
            acc = tl.dot(w, ht, acc)
        o = acc.to(tl.bfloat16).to(tl.float32)
        tl.store(out_ptr + row[None, :] * N + rn[:, None], (o * sc[None, :]).to(tl.bfloat16), mask=has[None, :])


def moe_gemv_blk(x, idx, sc, gate_up, down, hbuf, obuf, cfg=None):
    """x [T,Dm] bf16, idx [T,k] int32, sc [T,k] bf16 -> obuf [T*k, Dm] (row t*k+j = weighted output of token t's j-th expert)."""
    c = dict(BN1=16, BK1=128, BN2=16, BK2=128, nw1=4, nw2=4, ns1=None, ns2=None)
    if cfg:
        c.update(cfg)
    T, k = idx.shape
    Dm, I = x.shape[-1], down.shape[-1]
    TP = max(16, triton.next_power_of_2(T))
    TKP = triton.next_power_of_2(T * k)
    _gateup_blk[(T * k, I // c["BN1"])](x, idx, gate_up, hbuf, K=Dm, I=I, T=T, KK=k, TP=TP, TKP=TKP, BN=c["BN1"], BK=c["BK1"], num_warps=c["nw1"], **({"num_stages": c["ns1"]} if c["ns1"] else {}))
    _down_blk[(T * k, Dm // c["BN2"])](hbuf, idx, sc, down, obuf, K=I, N=Dm, T=T, KK=k, TP=TP, TKP=TKP, BN=c["BN2"], BK=c["BK2"], num_warps=c["nw2"], **({"num_stages": c["ns2"]} if c["ns2"] else {}))
    return obuf


@triton.jit
def _rope_write_blk(qkv_ptr, cos_ptr, sin_ptr, start_ptr, kc_ptr, vc_ptr, qo_ptr, Lmax,
                    G: tl.constexpr, D: tl.constexpr, QD: tl.constexpr, KD: tl.constexpr, MP: tl.constexpr, ROPE: tl.constexpr):
    h = tl.program_id(0)
    t = tl.program_id(1)
    start = tl.load(start_ptr)
    d = tl.arange(0, D)
    dp = d ^ 1
    sgn = tl.where((d % 2) == 0, -1.0, 1.0)
    cs = tl.load(cos_ptr + t * D + d).to(tl.float32)
    sn = tl.load(sin_ptr + t * D + d).to(tl.float32)
    row = qkv_ptr + t * (QD + 2 * KD)
    for g in tl.static_range(G):
        off = (h * G + g) * D
        q = tl.load(row + off + d).to(tl.float32)
        if ROPE:
            qp = tl.load(row + off + dp).to(tl.float32)
            q = q * cs + sgn * qp * sn
        tl.store(qo_ptr + (h * MP + t * G + g) * D + d, q.to(tl.bfloat16))
    ko = QD + h * D
    k = tl.load(row + ko + d).to(tl.float32)
    if ROPE:
        kp = tl.load(row + ko + dp).to(tl.float32)
        k = k * cs + sgn * kp * sn
    v = tl.load(row + QD + KD + h * D + d)
    dst = (h * Lmax + start + t) * D + d
    tl.store(kc_ptr + dst, k.to(tl.bfloat16))
    tl.store(vc_ptr + dst, v)


@triton.jit
def _attn_split_blk(q_ptr, kc_ptr, vc_ptr, start_ptr, wacc_ptr, wm_ptr, wl_ptr, Lmax, scale,
                    T: tl.constexpr, G: tl.constexpr, D: tl.constexpr, MP: tl.constexpr, NSPLIT: tl.constexpr, BN: tl.constexpr,
                    SLIDING: tl.constexpr, W: tl.constexpr):
    kvh = tl.program_id(0)
    sp = tl.program_id(1)
    start = tl.load(start_ptr)
    hi_all = start + T
    if SLIDING:
        lo_all = tl.maximum(0, start - W + 1)
    else:
        lo_all = start * 0
    n = hi_all - lo_all
    chunk = ((n + NSPLIT - 1) // NSPLIT + BN - 1) // BN * BN
    lo = lo_all + sp * chunk
    hi = tl.minimum(lo + chunk, hi_all)
    rm = tl.arange(0, MP)
    rd = tl.arange(0, D)
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
    o = (kvh * NSPLIT + sp) * MP
    tl.store(wm_ptr + o + rm, m_i)
    tl.store(wl_ptr + o + rm, l_i)
    tl.store(wacc_ptr + (o + rm)[:, None] * D + rd[None, :], acc)


@triton.jit
def _attn_combine_blk(wacc_ptr, wm_ptr, wl_ptr, out_ptr, T: tl.constexpr, G: tl.constexpr, Hkv: tl.constexpr, D: tl.constexpr,
                      MP: tl.constexpr, NSPLIT: tl.constexpr):
    pid = tl.program_id(0)
    kvh = pid // (T * G)
    r = pid % (T * G)           # row inside the kv-head block: t*G + g
    t = r // G
    g = r % G
    rs = tl.arange(0, NSPLIT)
    rd = tl.arange(0, D)
    row = (kvh * NSPLIT + rs) * MP + r
    m = tl.load(wm_ptr + row)
    l = tl.load(wl_ptr + row)
    M = tl.max(m, axis=0)
    w = tl.exp(m - M)
    L = tl.sum(w * l, axis=0)
    acc = tl.load(wacc_ptr + row[:, None] * D + rd[None, :])
    o = tl.sum(acc * w[:, None], axis=0) / L
    tl.store(out_ptr + t * (Hkv * G * D) + (kvh * G + g) * D + rd, o.to(tl.bfloat16))


class AttnWSBlk:
    def __init__(self, Hkv, G, D, nsplit, T, dev):
        self.Hkv, self.G, self.D, self.nsplit, self.T = Hkv, G, D, nsplit, T
        self.MP = max(16, triton.next_power_of_2(T * G))
        n = Hkv * nsplit * self.MP
        self.acc = torch.empty(n, D, device=dev, dtype=torch.float32)
        self.m = torch.empty(n, device=dev, dtype=torch.float32)
        self.l = torch.empty(n, device=dev, dtype=torch.float32)
        self.q = torch.zeros(Hkv * self.MP * D, device=dev, dtype=torch.bfloat16)  # padded query rows stay zero
        self.o = torch.empty(T, Hkv * G * D, device=dev, dtype=torch.bfloat16)


def verify_attention(ws: AttnWSBlk, qkv, cos, sin, start, kc, vc, Lmax, rope, sliding, window, BN=64):
    Hkv, G, D, ns, T, MP = ws.Hkv, ws.G, ws.D, ws.nsplit, ws.T, ws.MP
    _rope_write_blk[(Hkv, T)](qkv, cos, sin, start, kc, vc, ws.q, Lmax, G=G, D=D, QD=Hkv * G * D, KD=Hkv * D, MP=MP, ROPE=rope, num_warps=1)
    _attn_split_blk[(Hkv, ns)](ws.q, kc, vc, start, ws.acc, ws.m, ws.l, Lmax, D ** -0.5, T=T, G=G, D=D, MP=MP, NSPLIT=ns, BN=BN,
                               SLIDING=sliding, W=window, num_warps=4 if MP <= 32 else 8)
    _attn_combine_blk[(Hkv * T * G,)](ws.acc, ws.m, ws.l, ws.o, T=T, G=G, Hkv=Hkv, D=D, MP=MP, NSPLIT=ns, num_warps=4)
    return ws.o
