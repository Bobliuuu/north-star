"""BATCHED batch-invariant kernels (B sequences x T rows, ragged per-sequence positions) for the exact-mode FastNorth (branch batch-spec).
Row semantics are IDENTICAL to fast_kernels_exact.py / fast_kernels_blk.py: every row is reduced by the same fixed program shape (token axis padded to 16,
fixed K tiles); rows beyond 16 are handled as additional 16-row TILES of the same program (never as a wider MMA), so the logits of a row do not depend on
how many other rows (sequences, draft rows, padding rows) share the launch. Number of rows NT is a RUNTIME arg (no per-shape recompiles)."""
import torch
import triton
import triton.language as tl

import fast_kernels_exact as fx
import fast_kernels_blk as fb

TP = 16


@triton.jit
def _lin_bt(x_ptr, w_ptr, o_ptr, NT, N: tl.constexpr, K: tl.constexpr, TPAD: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pt = tl.program_id(0)   # token tile on the FAST axis: tiles sharing a weight block run concurrently (L2 hits)
    pn = tl.program_id(1)
    rn = pn * BN + tl.arange(0, BN)
    rt = pt * TPAD + tl.arange(0, TPAD)
    acc = tl.zeros([BN, TPAD], dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        xt = tl.load(x_ptr + rt[None, :] * K + rk[:, None], mask=(rt < NT)[None, :], other=0.0)
        w = tl.load(w_ptr + rn[:, None] * K + rk[None, :], mask=(rn < N)[:, None], other=0.0)
        acc = tl.dot(w, xt, acc)
    tl.store(o_ptr + rt[None, :] * N + rn[:, None], acc.to(tl.bfloat16), mask=(rn < N)[:, None] & (rt < NT)[None, :])


def lin_bt(x, w):
    """x [NT,K] bf16 contiguous, w [N,K] -> [NT,N] bf16; same math per row as fx.lin."""
    NT, K = x.shape
    N = w.shape[0]
    out = torch.empty(NT, N, device=x.device, dtype=torch.bfloat16)
    c = fx.lin_cfg(N, K)
    _lin_bt[(triton.cdiv(NT, TP), triton.cdiv(N, c["BN"]))](x, w, out, NT, N=N, K=K, TPAD=TP, BN=c["BN"], BK=c["BK"], num_warps=c["nw"], num_stages=c["ns"])
    return out


@triton.jit
def _leader(idx_ptr, flag_ptr, NK, TKP: tl.constexpr):
    p = tl.program_id(0)
    e = tl.load(idx_ptr + p)
    rp = tl.arange(0, TKP)
    a = tl.load(idx_ptr + rp, mask=rp < NK, other=-1)
    dup = tl.sum(((a == e) & (rp < p)).to(tl.int32), axis=0)
    tl.store(flag_ptr + p, (dup == 0).to(tl.int32))


@triton.jit
def _gateup_bt(x_ptr, idx_ptr, flag_ptr, w_ptr, out_ptr, NT, K: tl.constexpr, I: tl.constexpr, KK: tl.constexpr, TPAD: tl.constexpr,
               BN: tl.constexpr, BK: tl.constexpr):
    p = tl.program_id(0)
    pn = tl.program_id(1)
    if tl.load(flag_ptr + p) != 0:
        e32 = tl.load(idx_ptr + p)
        e = e32.to(tl.int64)
        rj = tl.arange(0, KK)
        rn = pn * BN + tl.arange(0, BN)
        base = w_ptr + e * (2 * I * K)
        for tt in range(0, NT, TPAD):
            rt = tt + tl.arange(0, TPAD)
            id2 = tl.load(idx_ptr + rt[:, None] * KK + rj[None, :], mask=(rt < NT)[:, None], other=-1)
            match = id2 == e32
            has = tl.sum(match.to(tl.int32), axis=1) > 0
            if tl.sum(has.to(tl.int32), axis=0) > 0:
                slot = tl.sum(tl.where(match, rj[None, :], 0), axis=1)
                ag = tl.zeros([BN, TPAD], dtype=tl.float32)
                au = tl.zeros([BN, TPAD], dtype=tl.float32)
                for k0 in range(0, K, BK):
                    rk = k0 + tl.arange(0, BK)
                    xt = tl.load(x_ptr + rt[None, :] * K + rk[:, None], mask=(rt < NT)[None, :], other=0.0)
                    wg = tl.load(base + rn[:, None] * K + rk[None, :])
                    wu = tl.load(base + (rn[:, None] + I) * K + rk[None, :])
                    ag = tl.dot(wg, xt, ag)
                    au = tl.dot(wu, xt, au)
                g = ag.to(tl.bfloat16).to(tl.float32)
                u = au.to(tl.bfloat16).to(tl.float32)
                s = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
                tl.store(out_ptr + (rt * KK + slot)[None, :] * I + rn[:, None], (s * u).to(tl.bfloat16), mask=has[None, :])


@triton.jit
def _down_bt(h_ptr, idx_ptr, flag_ptr, sc_ptr, w_ptr, out_ptr, NT, K: tl.constexpr, N: tl.constexpr, KK: tl.constexpr, TPAD: tl.constexpr,
             BN: tl.constexpr, BK: tl.constexpr):
    p = tl.program_id(0)
    pn = tl.program_id(1)
    if tl.load(flag_ptr + p) != 0:
        e32 = tl.load(idx_ptr + p)
        e = e32.to(tl.int64)
        rj = tl.arange(0, KK)
        rn = pn * BN + tl.arange(0, BN)
        base = w_ptr + e * (N * K)
        for tt in range(0, NT, TPAD):
            rt = tt + tl.arange(0, TPAD)
            id2 = tl.load(idx_ptr + rt[:, None] * KK + rj[None, :], mask=(rt < NT)[:, None], other=-1)
            match = id2 == e32
            has = tl.sum(match.to(tl.int32), axis=1) > 0
            if tl.sum(has.to(tl.int32), axis=0) > 0:
                slot = tl.sum(tl.where(match, rj[None, :], 0), axis=1)
                row = rt * KK + slot
                sc = tl.load(sc_ptr + row, mask=has, other=0.0).to(tl.float32)
                acc = tl.zeros([BN, TPAD], dtype=tl.float32)
                for k0 in range(0, K, BK):
                    rk = k0 + tl.arange(0, BK)
                    ht = tl.load(h_ptr + row[None, :] * K + rk[:, None], mask=has[None, :], other=0.0)
                    w = tl.load(base + rn[:, None] * K + rk[None, :])
                    acc = tl.dot(w, ht, acc)
                o = acc.to(tl.bfloat16).to(tl.float32)
                tl.store(out_ptr + row[None, :] * N + rn[:, None], (o * sc[None, :]).to(tl.bfloat16), mask=has[None, :])


def moe_bt(x, idx, sc, gate_up, down, hbuf, obuf, flag, cfg=None):
    """x [NT,Dm], idx [NT,k] int32, sc [NT,k] bf16 -> obuf [NT*k, Dm]. One leader program per distinct expert across ALL rows; the leader
    walks the rows in 16-row tiles (weights re-read from L2, same per-row math as fb.moe_gemv_blk)."""
    c = dict(BN1=16, BK1=128, BN2=16, BK2=128, nw1=4, nw2=4)
    if cfg:
        c.update(cfg)
    NT, k = idx.shape
    Dm, I = x.shape[-1], down.shape[-1]
    NK = NT * k
    _leader[(NK,)](idx, flag, NK, TKP=triton.next_power_of_2(NK), num_warps=4)
    _gateup_bt[(NK, I // c["BN1"])](x, idx, flag, gate_up, hbuf, NT, K=Dm, I=I, KK=k, TPAD=TP, BN=c["BN1"], BK=c["BK1"], num_warps=c["nw1"])
    _down_bt[(NK, Dm // c["BN2"])](hbuf, idx, flag, sc, down, obuf, NT, K=I, N=Dm, KK=k, TPAD=TP, BN=c["BN2"], BK=c["BK2"], num_warps=c["nw2"])
    return obuf


def attn_seq(ws, qkv, cos, sin, start, kc, vc, Lmax, rope, sliding, window, out, BN=64):
    """Exact attention of ONE sequence's T rows (same kernels as fx.exact_attention) writing into the `out` row slice."""
    Hkv, G, D, T, MP, NS = ws.Hkv, ws.G, ws.D, ws.T, ws.MP, ws.NS
    fb._rope_write_blk[(Hkv, T)](qkv, cos, sin, start, kc, vc, ws.q, Lmax, G=G, D=D, QD=Hkv * G * D, KD=Hkv * D, MP=MP, ROPE=rope, num_warps=1)
    fx._attn_split_x[(Hkv, NS)](ws.q, kc, vc, start, ws.acc, ws.m, ws.l, Lmax, D ** -0.5, T=T, G=G, D=D, MP=MP, NS=NS, BN=BN, C=fx.CHUNK,
                                SLIDING=sliding, W=window, num_warps=8)
    fx._attn_combine_x[(Hkv * T * G,)](ws.acc, ws.m, ws.l, out, T=T, G=G, Hkv=Hkv, D=D, MP=MP, NS=NS, num_warps=4)
