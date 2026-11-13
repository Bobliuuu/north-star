"""BATCHED batch-invariant kernels (B sequences x T rows, ragged per-sequence positions) for the exact-mode FastNorth (branch batch-spec).
Row semantics are IDENTICAL to fast_kernels_exact.py / fast_kernels_blk.py: every row is reduced by the same fixed program shape (token axis padded to 16,
fixed K tiles); rows beyond 16 are handled as additional 16-row TILES of the same program (never as a wider MMA), so the logits of a row do not depend on
how many other rows (sequences, draft rows, padding rows) share the launch. Number of rows NT is a RUNTIME arg (no per-shape recompiles).
Exception (branch moe-tune): lin_bt uses 32/64-row tiles at large NT (LIN_TW). It was verified bitwise equal to 16-row tiles on real North weights
(results/abmoetune_mt1.jsonl, all 30 B x T cells); MoE mode 5 keeps 16-row tiles (a 64-row tile with BN 128 was NOT bitwise, moe_tune_kernel2.json)."""
import os
import torch
import triton
import triton.language as tl

import fast_kernels_exact as fx
import fast_kernels_blk as fb

TP = 16
LIN_TW = int(os.environ.get("NS_LIN_TW", "64"))   # lin_bt max row-tile width (dot N dim): 16 = original; 64 = 32-row tiles from NT>=32, 64-row tiles from NT>=64.
                   # Real North A/B (results/abmoetune_mt1.jsonl): logits bitwise equal to 16 in all 30 B x T cells, so 64 is the default.
                   # Kernel test (results/moe_tune_kernel2.json): bitwise equal to 16 for qkv/o_proj/router/lm_head at NT 16..416.
# MoE row-tile scheduling: 0 = leader program walks the 16-row tiles serially (H18 bs1); 1/2 = one program per tile (grid axis 2).
# Read at launch/capture time, so set it before make_step. Env NS_MOE_TILE_GRID=1 sets the default.
TILE_GRID = int(os.environ.get("NS_MOE_TILE_GRID", "6"))   # default 6 since H20 (bitwise = 0 in abstack_st1, all 30 cells)   # 0 serial | 1 leader x tile | 2 expert x tile | 3 expert-gathered rows | 4 auto (3 if rows >= GATHER_MIN_NT else 0)
# | 5 mode-3 math, Triton counting sort + compact (expert, tile) grid | 6 auto (5 if rows >= GATHER5_MIN_NT else 0)
GATHER5_MIN_NT = 16  # mode 6 = auto(5): kernel test mode 5 bn64 beats serial from NT=16; real A/B mt1 neutral at 16 rows, 1.17-1.56x from 32
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


def lin_bt(x, w, out=None):
    """x [NT,K] bf16 contiguous, w [N,K] -> [NT,N] bf16; same math per row as fx.lin. out: optional static [NT,N] buffer."""
    NT, K = x.shape
    N = w.shape[0]
    if out is None:
        out = torch.empty(NT, N, device=x.device, dtype=torch.bfloat16)
    c = dict(fx.lin_cfg(N, K))
    tw = TP if LIN_TW <= TP else min(LIN_TW, 64 if NT >= 64 else 32 if NT >= 32 else TP)
    # wider row tiles: shrink pipeline stages, then BK, to fit shared memory (neither changes a row's k16 MMA accumulation order)
    while (c["BN"] + tw) * c["BK"] * 2 * c["ns"] > 220_000:
        if c["ns"] > 2:
            c["ns"] -= 1
        else:
            c["BK"] //= 2
    _lin_bt[(triton.cdiv(NT, tw), triton.cdiv(N, c["BN"]))](x, w, out, NT, N=N, K=K, TPAD=tw, BN=c["BN"], BK=c["BK"], num_warps=c["nw"], num_stages=c["ns"])
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


def moe_bt(x, idx, sc, gate_up, down, hbuf, obuf, flag, cfg=None, valid=None):
    """x [NT,Dm], idx [NT,k] int32, sc [NT,k] bf16 -> obuf [NT*k, Dm]. One leader program per distinct expert across ALL rows; the leader
    walks the rows in 16-row tiles (weights re-read from L2, same per-row math as fb.moe_gemv_blk)."""
    c = dict(BN1=16, BK1=128, BN2=16, BK2=128, nw1=4, nw2=4)
    if cfg:
        c.update(cfg)
        c["_bn_explicit"] = "BN1" in cfg or "BN2" in cfg
    NT, k = idx.shape
    Dm, I = x.shape[-1], down.shape[-1]
    NK = NT * k
    mode = int(TILE_GRID)            # 0 serial tiles (bs1), 1 leader x tile grid, 2 expert x tile grid
    if mode == 3 or (mode == 4 and idx.shape[0] >= GATHER_MIN_NT):
        return moe_eg(x, idx, sc, gate_up, down, hbuf, obuf, c, valid)
    if mode == 5 or (mode == 6 and idx.shape[0] >= GATHER5_MIN_NT):
        return moe_eg5(x, idx, sc, gate_up, down, hbuf, obuf, c, valid)
    if mode == 6:
        mode = 0
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


def moe_eg(x, idx, sc, gate_up, down, hbuf, obuf, c, valid=None):
    """Mode 3 of moe_bt. Graph-capturable: fixed-shape sort / scatter_add / cumsum on device, no host sync.
    valid [NT] bool (optional): rows with False (padding of a short draft, finished sequences) are routed to a sentinel expert E that no
    program reads, so they cost no expert work; their obuf rows keep stale (finite) values and must be ignored by the caller. Other rows
    are unaffected (row-invariant kernels)."""
    NT, k = idx.shape
    E, Dm, I = gate_up.shape[0], x.shape[-1], down.shape[-1]
    flat = idx.reshape(-1).long()
    if valid is not None:
        flat = torch.where(valid[:, None].expand(NT, k).reshape(-1), flat, E)
    order = torch.argsort(flat, stable=True).to(torch.int32)
    cnt = torch.zeros(E + 1, dtype=torch.int32, device=x.device).scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
    off = torch.zeros(E + 1, dtype=torch.int32, device=x.device)
    off[1:] = torch.cumsum(cnt[:E], 0)
    nt = triton.cdiv(NT, TP)   # an expert gets at most one slot per row
    _gateup_eg[(E, I // c["BN1"], nt)](x, order, off, gate_up, hbuf, K=Dm, I=I, KK=k, TPAD=TP, BN=c["BN1"], BK=c["BK1"], num_warps=c["nw1"])
    _down_eg[(E, Dm // c["BN2"], nt)](hbuf, order, off, sc, down, obuf, K=I, N=Dm, TPAD=TP, BN=c["BN2"], BK=c["BK2"], num_warps=c["nw2"])
    return obuf


# ---- mode 5: mode-3 math (expert-gathered rows, identical per-slot tile/K loop/rounding) with compact scheduling:
# a deterministic Triton counting sort (counts, offsets, per-expert 16-row tile prefix, stable rank) replaces torch argsort/scatter/cumsum,
# and the GEMM grid is one program per REAL (expert, tile) work item instead of E x cdiv(NT,16) mostly-empty programs.
@triton.jit
def _eg_count(idx_ptr, off_ptr, tp_ptr, NK, E: tl.constexpr, CH: tl.constexpr, TPAD: tl.constexpr):
    re = tl.arange(0, E)
    cnt = tl.zeros([E], dtype=tl.int32)
    for c0 in range(0, NK, CH):
        rc = c0 + tl.arange(0, CH)
        v = tl.load(idx_ptr + rc, mask=rc < NK, other=-1)
        cnt += tl.sum((v[None, :] == re[:, None]).to(tl.int32), axis=1)
    inc = tl.cumsum(cnt, axis=0)
    tl.store(off_ptr + re, inc - cnt)
    tl.store(off_ptr + E + re * 0, tl.sum(cnt, axis=0), mask=re == 0)
    nt = (cnt + TPAD - 1) // TPAD
    tinc = tl.cumsum(nt, axis=0)
    tl.store(tp_ptr + re, tinc - nt)
    tl.store(tp_ptr + E + re * 0, tl.sum(nt, axis=0), mask=re == 0)


@triton.jit
def _eg_rank(idx_ptr, off_ptr, ord_ptr, NK, BP: tl.constexpr, CH: tl.constexpr):
    """order[off[e] + rank] = p, rank = #earlier slots with the same expert (== stable argsort)."""
    rp = tl.program_id(0) * BP + tl.arange(0, BP)
    okp = rp < NK
    ep = tl.load(idx_ptr + rp, mask=okp, other=-1)
    rank = tl.zeros([BP], dtype=tl.int32)
    hi = tl.minimum(tl.program_id(0) * BP + BP, NK)
    for c0 in range(0, hi, CH):
        rq = c0 + tl.arange(0, CH)
        eq = tl.load(idx_ptr + rq, mask=rq < NK, other=-2)
        rank += tl.sum(((eq[None, :] == ep[:, None]) & (rq[None, :] < rp[:, None])).to(tl.int32), axis=1)
    base = tl.load(off_ptr + ep, mask=okp, other=0)
    tl.store(ord_ptr + base + rank, rp.to(tl.int32), mask=okp)


@triton.jit
def _eg_item(tp_ptr, off_ptr, w, E: tl.constexpr, TPAD: tl.constexpr):
    re = tl.arange(0, E)
    tp = tl.load(tp_ptr + 1 + re)            # inclusive tile prefix of experts 0..E-1
    e = tl.sum((tp <= w).to(tl.int32), axis=0)
    ec = tl.minimum(e, E - 1)
    t = w - tl.load(tp_ptr + ec)
    lo = tl.load(off_ptr + ec)
    hi = tl.load(off_ptr + ec + 1)
    return e, ec, lo + t * TPAD, hi


@triton.jit
def _gateup_eg5(x_ptr, ord_ptr, off_ptr, tp_ptr, w_ptr, out_ptr, K: tl.constexpr, I: tl.constexpr, KK: tl.constexpr, TPAD: tl.constexpr,
                BN: tl.constexpr, BK: tl.constexpr, E: tl.constexpr):
    e, ec, s0, hi = _eg_item(tp_ptr, off_ptr, tl.program_id(0), E, TPAD)
    pn = tl.program_id(1)
    if e < E:
        rs = s0 + tl.arange(0, TPAD)
        has = rs < hi
        slot = tl.load(ord_ptr + rs, mask=has, other=0)
        row = slot // KK
        rn = pn * BN + tl.arange(0, BN)
        base = w_ptr + ec.to(tl.int64) * (2 * I * K)
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
def _down_eg5(h_ptr, ord_ptr, off_ptr, tp_ptr, sc_ptr, w_ptr, out_ptr, K: tl.constexpr, N: tl.constexpr, TPAD: tl.constexpr, BN: tl.constexpr,
              BK: tl.constexpr, E: tl.constexpr):
    e, ec, s0, hi = _eg_item(tp_ptr, off_ptr, tl.program_id(0), E, TPAD)
    pn = tl.program_id(1)
    if e < E:
        rs = s0 + tl.arange(0, TPAD)
        has = rs < hi
        slot = tl.load(ord_ptr + rs, mask=has, other=0)
        sc = tl.load(sc_ptr + slot, mask=has, other=0.0).to(tl.float32)
        rn = pn * BN + tl.arange(0, BN)
        base = w_ptr + ec.to(tl.int64) * (N * K)
        acc = tl.zeros([BN, TPAD], dtype=tl.float32)
        for k0 in range(0, K, BK):
            rk = k0 + tl.arange(0, BK)
            ht = tl.load(h_ptr + slot[None, :] * K + rk[:, None], mask=has[None, :], other=0.0)
            w = tl.load(base + rn[:, None] * K + rk[None, :])
            acc = tl.dot(w, ht, acc)
        o = acc.to(tl.bfloat16).to(tl.float32)
        tl.store(out_ptr + slot[None, :] * N + rn[:, None], (o * sc[None, :]).to(tl.bfloat16), mask=has[None, :])


EG5_CFG = dict(nw1=4, ns1=3, nw2=4, ns2=3, tw=16, bn1=64, bn2=64)   # bn64 tw16 nw4: best bitwise in moe_tune_kernel2   # launch-only knobs (no change to per-slot math); tuned by modal/moe_tune_test.py
_eg5_bufs = {}


def cfg_explicit_bn(c):
    return bool(c.get("_bn_explicit"))


def moe_eg5(x, idx, sc, gate_up, down, hbuf, obuf, c, valid=None):
    """valid [NT] bool (optional): False rows get expert id E (not counted by _eg_count, ranked after all real slots), so no work
    item covers them; their obuf rows keep stale values (callers ignore them). Same contract as moe_eg."""
    NT, k = idx.shape
    E, Dm, I = gate_up.shape[0], x.shape[-1], down.shape[-1]
    NK = NT * k
    key = (x.device, NK, E)
    if key not in _eg5_bufs:   # allocated on first (eager warmup) call, reused by the captured graph
        _eg5_bufs[key] = (torch.empty(E + 1, dtype=torch.int32, device=x.device), torch.empty(E + 1, dtype=torch.int32, device=x.device),
                          torch.empty(NK, dtype=torch.int32, device=x.device))
    off, tp, order = _eg5_bufs[key]
    flat = idx.reshape(-1)
    if valid is not None:
        flat = torch.where(valid[:, None].expand(NT, k).reshape(-1), flat, torch.full_like(flat, E))
    g = EG5_CFG
    tw = g.get("tw", TP)
    _eg_count[(1,)](flat, off, tp, NK, E=E, CH=64, TPAD=tw, num_warps=4)
    BP = 64
    _eg_rank[(triton.cdiv(NK, BP),)](flat, off, order, NK, BP=BP, CH=64, num_warps=4)
    wmax = min(E, NK) + triton.cdiv(NK, tw)   # >= sum_e cdiv(cnt_e, tw)
    bn1, bn2 = g.get("bn1", c["BN1"]), g.get("bn2", c["BN2"])
    if cfg_explicit_bn(c):
        bn1, bn2 = c["BN1"], c["BN2"]
    _gateup_eg5[(wmax, I // bn1)](x, order, off, tp, gate_up, hbuf, K=Dm, I=I, KK=k, TPAD=tw, BN=bn1, BK=c["BK1"], E=E,
                                       num_warps=g["nw1"], num_stages=g["ns1"])
    _down_eg5[(wmax, Dm // bn2)](hbuf, order, off, tp, sc, down, obuf, K=I, N=Dm, TPAD=tw, BN=bn2, BK=c["BK2"], E=E,
                                      num_warps=g["nw2"], num_stages=g["ns2"])
    return obuf
