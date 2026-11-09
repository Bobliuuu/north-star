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
# MoE row-tile scheduling: 0 = leader program walks the 16-row tiles serially (H18 bs1); 1/2 = one program per tile (grid axis 2).
# Read at launch/capture time, so set it before make_step. Env NS_MOE_TILE_GRID=1 sets the default.
import os
TILE_GRID = int(os.environ.get("NS_MOE_TILE_GRID", "0"))   # 0 serial | 1 leader x tile | 2 expert x tile | 3 expert-gathered rows | 4 auto (3 if rows >= GATHER_MIN_NT else 0)
GATHER_MIN_NT = 48   # real North A/B (abtile_ab1, H100): gather 0.97x at NT=32, 1.02-1.35x at 52-64, 1.6-2.2x at NT>=104; plain T=1 never gains below B=32


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
               BN: tl.constexpr, BK: tl.constexpr, GRID_T: tl.constexpr, EGRID: tl.constexpr):
    p = tl.program_id(0)
    pn = tl.program_id(1)
    # GRID_T=1: one 16-row tile per program (grid axis 2); GRID_T=0: the leader walks every tile (axis 2 has size 1). Same per-row math.
    t_lo = tl.program_id(2) * TPAD
    t_hi = NT * (1 - GRID_T) + (t_lo + TPAD) * GRID_T
    # EGRID=1: axis 0 is the expert id itself (grid size E, no leader flags); tiles with no row routed to it are skipped below.
    if EGRID:
        go = p >= 0
        e32 = p
    else:
        go = tl.load(flag_ptr + p) != 0
        e32 = tl.load(idx_ptr + p)
    if go:
        e = e32.to(tl.int64)
        rj = tl.arange(0, KK)
        rn = pn * BN + tl.arange(0, BN)
        base = w_ptr + e * (2 * I * K)
        for tt in range(t_lo, t_hi, TPAD):
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
             BN: tl.constexpr, BK: tl.constexpr, GRID_T: tl.constexpr, EGRID: tl.constexpr):
    p = tl.program_id(0)
    pn = tl.program_id(1)
    # GRID_T=1: one 16-row tile per program (grid axis 2); GRID_T=0: the leader walks every tile (axis 2 has size 1). Same per-row math.
    t_lo = tl.program_id(2) * TPAD
    t_hi = NT * (1 - GRID_T) + (t_lo + TPAD) * GRID_T
    # EGRID=1: axis 0 is the expert id itself (grid size E, no leader flags); tiles with no row routed to it are skipped below.
    if EGRID:
        go = p >= 0
        e32 = p
    else:
        go = tl.load(flag_ptr + p) != 0
        e32 = tl.load(idx_ptr + p)
    if go:
        e = e32.to(tl.int64)
        rj = tl.arange(0, KK)
        rn = pn * BN + tl.arange(0, BN)
        base = w_ptr + e * (N * K)
        for tt in range(t_lo, t_hi, TPAD):
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
    mode = int(TILE_GRID)            # 0 serial tiles (bs1), 1 leader x tile grid, 2 expert x tile grid
    if mode == 3 or (mode == 4 and idx.shape[0] >= GATHER_MIN_NT):
        return moe_eg(x, idx, sc, gate_up, down, hbuf, obuf, c)
    if mode == 4:
        mode = 0
    gt, eg = int(mode >= 1), int(mode == 2)
    nt = triton.cdiv(NT, TP) if gt else 1
    g0 = gate_up.shape[0] if eg else NK
    if not eg:
        _leader[(NK,)](idx, flag, NK, TKP=triton.next_power_of_2(NK), num_warps=4)
    _gateup_bt[(g0, I // c["BN1"], nt)](x, idx, flag, gate_up, hbuf, NT, K=Dm, I=I, KK=k, TPAD=TP, BN=c["BN1"], BK=c["BK1"], GRID_T=gt, EGRID=eg, num_warps=c["nw1"])
    _down_bt[(g0, Dm // c["BN2"], nt)](hbuf, idx, flag, sc, down, obuf, NT, K=I, N=Dm, KK=k, TPAD=TP, BN=c["BN2"], BK=c["BK2"], GRID_T=gt, EGRID=eg, num_warps=c["nw2"])
    return obuf


def attn_seq(ws, qkv, cos, sin, start, kc, vc, Lmax, rope, sliding, window, out, BN=64):
    """Exact attention of ONE sequence's T rows (same kernels as fx.exact_attention) writing into the `out` row slice."""
    Hkv, G, D, T, MP, NS = ws.Hkv, ws.G, ws.D, ws.T, ws.MP, ws.NS
    fb._rope_write_blk[(Hkv, T)](qkv, cos, sin, start, kc, vc, ws.q, Lmax, G=G, D=D, QD=Hkv * G * D, KD=Hkv * D, MP=MP, ROPE=rope, num_warps=1)
    fx._attn_split_x[(Hkv, NS)](ws.q, kc, vc, start, ws.acc, ws.m, ws.l, Lmax, D ** -0.5, T=T, G=G, D=D, MP=MP, NS=NS, BN=BN, C=fx.CHUNK,
                                SLIDING=sliding, W=window, num_warps=8)
    fx._attn_combine_x[(Hkv * T * G,)](ws.acc, ws.m, ws.l, out, T=T, G=G, Hkv=Hkv, D=D, MP=MP, NS=NS, num_warps=4)


# ---- mode 3: expert-GATHERED rows. Slots (row*k + j) are sorted by expert; program (e, n-block, t) handles slots [off_e + 16t, off_e + 16t + 16)
# of expert e. Each expert's weights are streamed once per 16 of ITS OWN rows (vs once per 16-row tile that touches it in modes 0-2).
# Per-slot math is unchanged: same 16-wide tile, same fixed K loop and bf16 rounding points, masked columns contribute zeros.
@triton.jit
def _gateup_eg(x_ptr, ord_ptr, off_ptr, w_ptr, out_ptr, K: tl.constexpr, I: tl.constexpr, KK: tl.constexpr, TPAD: tl.constexpr,
               BN: tl.constexpr, BK: tl.constexpr):
    e = tl.program_id(0)
    pn = tl.program_id(1)
    t = tl.program_id(2)
    lo = tl.load(off_ptr + e)
    hi = tl.load(off_ptr + e + 1)
    if lo + t * TPAD < hi:
        rs = lo + t * TPAD + tl.arange(0, TPAD)
        has = rs < hi
        slot = tl.load(ord_ptr + rs, mask=has, other=0)
        row = slot // KK
        rn = pn * BN + tl.arange(0, BN)
        base = w_ptr + e.to(tl.int64) * (2 * I * K)
        ag = tl.zeros([BN, TPAD], dtype=tl.float32)
        au = tl.zeros([BN, TPAD], dtype=tl.float32)
        for k0 in range(0, K, BK):
            rk = k0 + tl.arange(0, BK)
            xt = tl.load(x_ptr + row[None, :] * K + rk[:, None], mask=has[None, :], other=0.0)
            wg = tl.load(base + rn[:, None] * K + rk[None, :])
            wu = tl.load(base + (rn[:, None] + I) * K + rk[None, :])
            ag = tl.dot(wg, xt, ag)
            au = tl.dot(wu, xt, au)
        g = ag.to(tl.bfloat16).to(tl.float32)
        u = au.to(tl.bfloat16).to(tl.float32)
        s = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
        tl.store(out_ptr + slot[None, :] * I + rn[:, None], (s * u).to(tl.bfloat16), mask=has[None, :])


@triton.jit
def _down_eg(h_ptr, ord_ptr, off_ptr, sc_ptr, w_ptr, out_ptr, K: tl.constexpr, N: tl.constexpr, TPAD: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    e = tl.program_id(0)
    pn = tl.program_id(1)
    t = tl.program_id(2)
    lo = tl.load(off_ptr + e)
    hi = tl.load(off_ptr + e + 1)
    if lo + t * TPAD < hi:
        rs = lo + t * TPAD + tl.arange(0, TPAD)
        has = rs < hi
        slot = tl.load(ord_ptr + rs, mask=has, other=0)
        sc = tl.load(sc_ptr + slot, mask=has, other=0.0).to(tl.float32)
        rn = pn * BN + tl.arange(0, BN)
        base = w_ptr + e.to(tl.int64) * (N * K)
        acc = tl.zeros([BN, TPAD], dtype=tl.float32)
        for k0 in range(0, K, BK):
            rk = k0 + tl.arange(0, BK)
            ht = tl.load(h_ptr + slot[None, :] * K + rk[:, None], mask=has[None, :], other=0.0)
            w = tl.load(base + rn[:, None] * K + rk[None, :])
            acc = tl.dot(w, ht, acc)
        o = acc.to(tl.bfloat16).to(tl.float32)
        tl.store(out_ptr + slot[None, :] * N + rn[:, None], (o * sc[None, :]).to(tl.bfloat16), mask=has[None, :])


def moe_eg(x, idx, sc, gate_up, down, hbuf, obuf, c):
    """Mode 3 of moe_bt. Graph-capturable: fixed-shape sort / scatter_add / cumsum on device, no host sync."""
    NT, k = idx.shape
    E, Dm, I = gate_up.shape[0], x.shape[-1], down.shape[-1]
    flat = idx.reshape(-1).long()
    order = torch.argsort(flat, stable=True).to(torch.int32)
    cnt = torch.zeros(E, dtype=torch.int32, device=x.device).scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
    off = torch.zeros(E + 1, dtype=torch.int32, device=x.device)
    off[1:] = torch.cumsum(cnt, 0)
    nt = triton.cdiv(NT, TP)   # an expert gets at most one slot per row
    _gateup_eg[(E, I // c["BN1"], nt)](x, order, off, gate_up, hbuf, K=Dm, I=I, KK=k, TPAD=TP, BN=c["BN1"], BK=c["BK1"], num_warps=c["nw1"])
    _down_eg[(E, Dm // c["BN2"], nt)](hbuf, order, off, sc, down, obuf, K=I, N=Dm, TPAD=TP, BN=c["BN2"], BK=c["BK2"], num_warps=c["nw2"])
    return obuf
