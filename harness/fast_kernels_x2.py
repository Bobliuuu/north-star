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
