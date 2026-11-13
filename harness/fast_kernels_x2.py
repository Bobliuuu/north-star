"""Exact-v2 kernels (branch perf/exact-decode): H100-tuned, still BATCH-INVARIANT linear for the exact FastNorth path.

`lin_x2` computes out[t,n] = sum_k x[t,k] w[n,k] (bf16 out, fp32 acc) like fast_kernels_exact.lin, but with
  * a per-shape tile table (BN, BK, warps, stages) tuned on H100 (modal/x2_tune.py), and
  * optional fixed split-K (SK programs along K, each a fixed-shape tl.dot loop, fp32 partials summed in a fixed order by a
    second kernel), which raises the CTA count for the small-N shapes (o_proj 2048 x 4096, router 128 x 2048).
Every choice depends on the weight shape only, never on T or the position, and the token axis is padded to the same 16 rows,
so a row of a T<=16 block is bitwise equal to the same row computed alone (rule of fast_kernels_exact.py; tested in
modal/x2_tune.py and on the real model in harness/bench_x2.py)."""
import torch
import triton
import triton.language as tl

import fast_kernels_exact as fx

TP = fx.TP


@triton.jit
def _lin_sk(x_ptr, w_ptr, p_ptr, T: tl.constexpr, N: tl.constexpr, K: tl.constexpr, TPAD: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
            KS: tl.constexpr):
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    rn = pn * BN + tl.arange(0, BN)
    rt = tl.arange(0, TPAD)
    acc = tl.zeros([BN, TPAD], dtype=tl.float32)
    for k0 in range(ps * KS, ps * KS + KS, BK):
        rk = k0 + tl.arange(0, BK)
        xt = tl.load(x_ptr + rt[None, :] * K + rk[:, None], mask=(rt < T)[None, :], other=0.0)
        w = tl.load(w_ptr + rn[:, None] * K + rk[None, :], mask=(rn < N)[:, None], other=0.0)
        acc = tl.dot(w, xt, acc)
    tl.store(p_ptr + (ps * TPAD + rt[None, :]) * N + rn[:, None], acc, mask=(rn < N)[:, None] & (rt < T)[None, :])


@triton.jit
def _lin_red(p_ptr, o_ptr, TN: tl.constexpr, S: tl.constexpr, SN: tl.constexpr, BR: tl.constexpr):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    m = r < TN
    acc = tl.load(p_ptr + r, mask=m, other=0.0)
    for s in tl.static_range(1, S):
        acc += tl.load(p_ptr + s * SN + r, mask=m, other=0.0)
    tl.store(o_ptr + r, acc.to(tl.bfloat16), mask=m)


# H100 sweep winners (results/x2_tune_t1lin.jsonl; best invariant T=1 config per shape). Split-K gave nothing on the real model (x1prof),
# and without it these tiles keep every row's k16 MMA accumulation order, so outputs are bitwise equal to the default tiles (x3: 99/99 tokens).
TUNED_LIN_CFG = {(2048, 4096): dict(BN=32, BK=512, nw=4, ns=4), (5120, 2048): dict(BN=64, BK=128, nw=4, ns=6),
                 (262144, 2048): dict(BN=64, BK=256, nw=4, ns=4)}
TUNED_MOE_CFG = dict(BN1=64, BK1=128, BN2=64, BK2=128, nw1=4, nw2=4, ns1=4, ns2=3)

# (N, K) -> tile config used by lin_x2 (set by FastNorth.set_x2); shapes not listed use fx.lin_cfg defaults, SK=1.
CFG = {}


def cfg_for(N, K):
    c = dict(fx.lin_cfg(N, K), SK=1)
    c.update(CFG.get((N, K), {}))
    return c


class _WS:
    bufs = {}


def lin_x2(x, w, out=None, cfg=None):
    """x [T,K] bf16 contiguous (T <= 16), w [N,K] bf16 -> [T,N] bf16."""
    T, K = x.shape
    N = w.shape[0]
    c = cfg or cfg_for(N, K)
    if out is None:
        out = torch.empty(T, N, device=x.device, dtype=torch.bfloat16)
    sk = c.get("SK", 1)
    if sk == 1:
        fx._lin_kernel[(triton.cdiv(N, c["BN"]),)](x, w, out, T=T, N=N, K=K, TPAD=TP, BN=c["BN"], BK=c["BK"], num_warps=c["nw"], num_stages=c["ns"])
        return out
    assert K % (sk * c["BK"]) == 0, (K, sk, c)
    key = (x.device, sk, N, K)  # one partial buffer per weight shape (streams never run two lins of one shape concurrently)
    p = _WS.bufs.get(key)
    if p is None:
        p = _WS.bufs[key] = torch.empty(sk * TP * N, device=x.device, dtype=torch.float32)
    _lin_sk[(triton.cdiv(N, c["BN"]), sk)](x, w, p, T=T, N=N, K=K, TPAD=TP, BN=c["BN"], BK=c["BK"], KS=K // sk, num_warps=c["nw"], num_stages=c["ns"])
    BR = 1024
    _lin_red[(triton.cdiv(T * N, BR),)](p, out, TN=T * N, S=sk, SN=TP * N, BR=BR, num_warps=4)
    return out


def attn_chunk_for(Lmax):
    """Absolute key chunk of one exact-attention split program, by KV capacity (results/lcx2_lc1.jsonl, H100): 256 keys gives 512 splits
    per head at 128K and the combine dominates (T=1 10.3 ms); 1024 at 128K: 5.3 ms, T=13 25.8 -> 8.3 ms; 512 is best at 32K.
    Depends on Lmax only (fixed per executor), never on T or the position, so plain and verify rows stay bitwise equal."""
    if Lmax is None or Lmax <= 16384:
        return 256   # == fast_kernels_exact.CHUNK: outputs identical to the v1 exact path
    return 512 if Lmax <= 65536 else 1024


def recommended(Lmax=None):
    """FastNorth.set_x2(**recommended(X.Lmax)): exact v2 = tuned tiles + attention || MoE streams, attention stream at high priority,
    attention split chunk by context capacity. Measured (results/x2_x3.jsonl, H100, 99 real prompts): plain 237 -> 387 tok/s, tokens
    identical to the v1 exact path on 99/99 (Lmax <= 16K keeps v1's chunk)."""
    return dict(on=True, streams=True, lin_cfg=dict(TUNED_LIN_CFG), moe_cfg=dict(TUNED_MOE_CFG), side_prio=-1,
                attn=dict(chunk=attn_chunk_for(Lmax)))


# ---------------------------------------------------------------- fused MoE (gate_up + SiLU*up + down slice) + reduce in resid_norm
# Program (p, j): leader program of expert idx[p] (first (row, slot) naming it, as in fast_kernels_blk) computes the intermediate slice
# j*BJ..(j+1)*BJ for every row routed to that expert (same tl.dot shapes / rounding as _gateup_blk), then multiplies it into the down
# projection columns of that slice and stores fp32 partials part[(row*KK+slot)*J + j, :]. _resid_norm_moe sums the J partials in a fixed
# order, rounds to bf16 (= down output), applies the routing weight, and does the residual + RMSNorm of fast_kernels.resid_norm.
# Every shape depends on (I, Dm, BJ, BN2, BK) only, never on T -> rows stay T-invariant. vs v1: the down reduction over I is split into J
# fp32 partials, so bf16 outputs can differ from v1 in the last bit (exactness of spec vs plain is unaffected).
@triton.jit
def _moe_fused(x_ptr, idx_ptr, gu_ptr, dn_ptr, part_ptr, K: tl.constexpr, I: tl.constexpr, N: tl.constexpr, T: tl.constexpr, KK: tl.constexpr,
               TP: tl.constexpr, TKP: tl.constexpr, BJ: tl.constexpr, BK: tl.constexpr, BN2: tl.constexpr, J: tl.constexpr):
    p = tl.program_id(0)
    j = tl.program_id(1)
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
        ri = j * BJ + tl.arange(0, BJ)
        base = gu_ptr + e * (2 * I * K)
        ag = tl.zeros([BJ, TP], dtype=tl.float32)
        au = tl.zeros([BJ, TP], dtype=tl.float32)
        for k0 in range(0, K, BK):
            rk = k0 + tl.arange(0, BK)
            xt = tl.load(x_ptr + rt[None, :] * K + rk[:, None], mask=(rt < T)[None, :], other=0.0)
            wg = tl.load(base + ri[:, None] * K + rk[None, :])
            wu = tl.load(base + (ri[:, None] + I) * K + rk[None, :])
            ag = tl.dot(wg, xt, ag)
            au = tl.dot(wu, xt, au)
        g = ag.to(tl.bfloat16).to(tl.float32)
        u = au.to(tl.bfloat16).to(tl.float32)
        s = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
        h = (s * u).to(tl.bfloat16)                       # [BJ, TP] == the v1 hbuf slice (bitwise)
        h = tl.where(has[None, :], h, 0.0).to(tl.bfloat16)
        dbase = dn_ptr + e * (N * I)
        prow = ((rt * KK + slot) * J + j).to(tl.int64)
        for n0 in range(0, N, BN2):
            rn = n0 + tl.arange(0, BN2)
            w = tl.load(dbase + rn[:, None] * I + ri[None, :])
            acc = tl.dot(w, h)                            # [BN2, TP] fp32 partial over this slice
            tl.store(part_ptr + prow[None, :] * N + rn[:, None], acc, mask=has[None, :])


@triton.jit
def _resid_norm_moe(x_ptr, a_ptr, part_ptr, sc_ptr, w_ptr, xo_ptr, ho_ptr, eps, D: tl.constexpr, KK: tl.constexpr, J: tl.constexpr):
    row = tl.program_id(0)
    r = tl.arange(0, D)
    x = tl.load(x_ptr + row * D + r).to(tl.float32)
    a = tl.load(a_ptr + row * D + r).to(tl.float32)
    t = (x + a).to(tl.bfloat16).to(tl.float32)
    ms = tl.zeros([D], dtype=tl.float32)
    for b in tl.static_range(KK):
        o = tl.zeros([D], dtype=tl.float32)
        for jj in tl.static_range(J):
            o += tl.load(part_ptr + ((row * KK + b) * J + jj) * D + r)
        sc = tl.load(sc_ptr + row * KK + b).to(tl.float32)
        ms += (o.to(tl.bfloat16).to(tl.float32) * sc).to(tl.bfloat16).to(tl.float32)   # same rounding points as _down_blk + resid_norm
    ms = ms.to(tl.bfloat16).to(tl.float32)
    xn = (t + ms).to(tl.bfloat16)
    tl.store(xo_ptr + row * D + r, xn)
    xf = xn.to(tl.float32)
    var = tl.sum(xf * xf, axis=0) / D
    hn = (xf * tl.rsqrt(var + eps)).to(tl.bfloat16).to(tl.float32)
    w = tl.load(w_ptr + r).to(tl.float32)
    tl.store(ho_ptr + row * D + r, (w * hn).to(tl.bfloat16))


FUSED_MOE_CFG = dict(BJ=64, BK=128, BN2=128, nw=4, ns=3)


def moe_fused(x, idx, gate_up, down, part, cfg=None):
    """x [T,Dm], idx [T,k] int32, gate_up [E,2I,Dm], down [E,Dm,I] -> part [T*k*J, Dm] fp32 partials (consumed by resid_norm_moe)."""
    c = dict(FUSED_MOE_CFG); c.update(cfg or {})
    T, k = idx.shape
    Dm, I = x.shape[-1], down.shape[-1]
    J = I // c["BJ"]
    TP = max(16, triton.next_power_of_2(T))
    TKP = triton.next_power_of_2(T * k)
    _moe_fused[(T * k, J)](x, idx, gate_up, down, part, K=Dm, I=I, N=Dm, T=T, KK=k, TP=TP, TKP=TKP, BJ=c["BJ"], BK=c["BK"], BN2=c["BN2"], J=J,
                           num_warps=c["nw"], num_stages=c["ns"])
    return part, J


def resid_norm_moe(x, a, part, sc, J, w, xo, ho, eps):
    D = x.shape[-1]
    T, k = sc.shape
    _resid_norm_moe[(T,)](x, a, part, sc, w, xo, ho, eps, D=D, KK=k, J=J, num_warps=8)


# ---------------------------------------------------------------- router GEMV + top-k + sigmoid in ONE launch
# Same router-logit math as fx._lin_kernel with the router's tile config (bitwise equal logits), then the LAST program to finish (atomic
# ticket, acq_rel) runs fast_kernels._topk_kernel's selection over the full 128 logits of every row and resets the ticket. The ticket only
# orders work; the result does not depend on which program is last, so it is deterministic and T-invariant.
@triton.jit
def _router_topk_fused(x_ptr, w_ptr, lg_ptr, idx_ptr, sc_ptr, cnt_ptr, T: tl.constexpr, N: tl.constexpr, K: tl.constexpr, TPAD: tl.constexpr,
                       BN: tl.constexpr, BK: tl.constexpr, KK: tl.constexpr, NPROG: tl.constexpr):
    pn = tl.program_id(0)
    rn = pn * BN + tl.arange(0, BN)
    rt = tl.arange(0, TPAD)
    acc = tl.zeros([BN, TPAD], dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        xt = tl.load(x_ptr + rt[None, :] * K + rk[:, None], mask=(rt < T)[None, :], other=0.0)
        w = tl.load(w_ptr + rn[:, None] * K + rk[None, :], mask=(rn < N)[:, None], other=0.0)
        acc = tl.dot(w, xt, acc)
    tl.store(lg_ptr + rt[None, :] * N + rn[:, None], acc.to(tl.bfloat16), mask=(rn < N)[:, None] & (rt < T)[None, :])
    tl.debug_barrier()
    ticket = tl.atomic_add(cnt_ptr, 1, sem="acq_rel")
    if ticket == NPROG - 1:
        # all rows at once: per row the same max / lowest-index tie-break / -inf masking sequence as _topk_kernel (bitwise equal)
        r = tl.arange(0, N)
        ok = rt < T
        v = tl.load(lg_ptr + rt[:, None] * N + r[None, :], mask=ok[:, None], other=float("-inf"), volatile=True).to(tl.float32)
        for j in tl.static_range(KK):
            m = tl.max(v, axis=1)
            i = tl.min(tl.where(v == m[:, None], r[None, :], N), axis=1)
            tl.store(idx_ptr + rt * KK + j, i, mask=ok)
            tl.store(sc_ptr + rt * KK + j, tl.sigmoid(m).to(tl.bfloat16), mask=ok)
            v = tl.where(r[None, :] == i[:, None], float("-inf"), v)
        tl.atomic_xchg(cnt_ptr, 0)


def router_topk_fused(x, w, lg, idx, sc, cnt):
    """x [T,K] bf16, w [N,K] router weight, lg [T,N] bf16 logits buffer, idx [T,k] int32, sc [T,k] bf16, cnt int32[1] zero ticket."""
    T, K = x.shape
    N = w.shape[0]
    c = cfg_for(N, K)
    nprog = triton.cdiv(N, c["BN"])
    _router_topk_fused[(nprog,)](x, w, lg, idx, sc, cnt, T=T, N=N, K=K, TPAD=TP, BN=c["BN"], BK=c["BK"], KK=idx.shape[-1], NPROG=nprog,
                                 num_warps=c["nw"], num_stages=c["ns"])
