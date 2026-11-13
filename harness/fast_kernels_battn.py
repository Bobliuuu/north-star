"""BATCHED exact attention: ONE launch per layer for all B sequences (grid axis 2 = sequence) instead of B launches of
fast_kernels_blk._rope_write_blk / fast_kernels_exact._attn_split_x / _attn_combine_x. Kernel bodies are copies of those three with only
the per-sequence base offsets added (KV via a device pointer table, q/workspace/out rows offset by b), so every row's math, block shapes,
chunking (absolute CHUNK splits) and reduction order are unchanged: per-row results are bitwise equal to the per-sequence path.
Profile p1 (H100, B=32 T=1): per-sequence attention was 22.6 of 30.6 ms per step (4704 launches)."""
import torch
import triton
import triton.language as tl

import fast_kernels_exact as fx


@triton.jit
def _rope_write_b(qkv_ptr, cos_ptr, sin_ptr, start_ptr, kp_ptr, vp_ptr, qo_ptr, Lmax,
                  T: tl.constexpr, G: tl.constexpr, D: tl.constexpr, QD: tl.constexpr, KD: tl.constexpr, MP: tl.constexpr, HKV: tl.constexpr,
                  ROPE: tl.constexpr):
    h = tl.program_id(0)
    t = tl.program_id(1)
    b = tl.program_id(2)
    kc_ptr = tl.load(kp_ptr + b).to(tl.pointer_type(tl.bfloat16))
    vc_ptr = tl.load(vp_ptr + b).to(tl.pointer_type(tl.bfloat16))
    start = tl.load(start_ptr + b)
    rt = b * T + t
    qo_ptr = qo_ptr + b * (HKV * MP * D)
    d = tl.arange(0, D)
    dp = d ^ 1
    sgn = tl.where((d % 2) == 0, -1.0, 1.0)
    cs = tl.load(cos_ptr + rt * D + d).to(tl.float32)
    sn = tl.load(sin_ptr + rt * D + d).to(tl.float32)
    row = qkv_ptr + rt * (QD + 2 * KD)
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
def _attn_split_b(q_ptr, kp_ptr, vp_ptr, start_ptr, wacc_ptr, wm_ptr, wl_ptr, Lmax, scale,
                  T: tl.constexpr, G: tl.constexpr, D: tl.constexpr, MP: tl.constexpr, NS: tl.constexpr, BN: tl.constexpr, C: tl.constexpr,
                  HKV: tl.constexpr, SLIDING: tl.constexpr, W: tl.constexpr):
    kvh = tl.program_id(0)
    sp = tl.program_id(1)
    b = tl.program_id(2)
    kc_ptr = tl.load(kp_ptr + b).to(tl.pointer_type(tl.bfloat16))
    vc_ptr = tl.load(vp_ptr + b).to(tl.pointer_type(tl.bfloat16))
    q_ptr = q_ptr + b * (HKV * MP * D)
    wb = b * (HKV * NS * MP)
    start = tl.load(start_ptr + b)
    rm = tl.arange(0, MP)
    rd = tl.arange(0, D)
    o = wb + (kvh * NS + sp) * MP
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


@triton.jit
def _attn_combine_b(wacc_ptr, wm_ptr, wl_ptr, out_ptr, T: tl.constexpr, G: tl.constexpr, Hkv: tl.constexpr, D: tl.constexpr,
                    MP: tl.constexpr, NS: tl.constexpr):
    pid = tl.program_id(0)
    b = tl.program_id(1)
    kvh = pid // (T * G)
    r = pid % (T * G)
    t = r // G
    g = r % G
    rs = tl.arange(0, NS)
    rd = tl.arange(0, D)
    row = b * (Hkv * NS * MP) + (kvh * NS + rs) * MP + r
    m = tl.load(wm_ptr + row)
    live = m > -1e29
    l = tl.load(wl_ptr + row, mask=live, other=0.0)
    M = tl.max(m, axis=0)
    w = tl.where(live, tl.exp(m - M), 0.0)
    L = tl.sum(w * l, axis=0)
    acc = tl.load(wacc_ptr + row[:, None] * D + rd[None, :], mask=live[:, None], other=0.0)
    o = tl.sum(acc * w[:, None], axis=0) / L
    tl.store(out_ptr + (b * T + t) * (Hkv * G * D) + (kvh * G + g) * D + rd, o.to(tl.bfloat16))


class AttnWSB:
    """Per-sequence copies of fx.AttnWSX buffers, B of them back to back (same MP / NS as the per-sequence workspace)."""
    def __init__(self, ws1, B, dev):
        self.Hkv, self.G, self.D, self.T, self.MP, self.NS, self.B = ws1.Hkv, ws1.G, ws1.D, ws1.T, ws1.MP, ws1.NS, B
        self.C = getattr(ws1, "C", fx.CHUNK)  # absolute key chunk: must match NS (exact v2 scales it with Lmax)
        n = B * self.Hkv * self.NS * self.MP
        self.acc = torch.zeros(n, self.D, device=dev, dtype=torch.float32)
        self.m = torch.full((n,), -1e30, device=dev, dtype=torch.float32)
        self.l = torch.zeros(n, device=dev, dtype=torch.float32)
        self.q = torch.zeros(B * self.Hkv * self.MP * self.D, device=dev, dtype=torch.bfloat16)


def attn_batched(ws, qkv, cos, sin, starts, kp, vp, Lmax, rope, sliding, window, out, BN=64):
    """qkv/cos/sin/out: [B*T, ...] rows (sequence-major); starts [B] long; kp/vp [B] int64 data pointers of this layer's K/V caches."""
    Hkv, G, D, T, MP, NS, B = ws.Hkv, ws.G, ws.D, ws.T, ws.MP, ws.NS, ws.B
    _rope_write_b[(Hkv, T, B)](qkv, cos, sin, starts, kp, vp, ws.q, Lmax, T=T, G=G, D=D, QD=Hkv * G * D, KD=Hkv * D, MP=MP, HKV=Hkv,
                                ROPE=rope, num_warps=1)
    _attn_split_b[(Hkv, NS, B)](ws.q, kp, vp, starts, ws.acc, ws.m, ws.l, Lmax, D ** -0.5, T=T, G=G, D=D, MP=MP, NS=NS, BN=BN, C=getattr(ws, "C", fx.CHUNK),
                                 HKV=Hkv, SLIDING=sliding, W=window, num_warps=8)
    _attn_combine_b[(Hkv * T * G, B)](ws.acc, ws.m, ws.l, out, T=T, G=G, Hkv=Hkv, D=D, MP=MP, NS=NS, num_warps=4)
