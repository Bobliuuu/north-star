"""PyTorch CPU reference for the fused draft/verify MoE expert GEMM (one North MoE layer).

Semantics (match src/kvm/moe_fused.cu):
  * expert e: h = bf16( silu(x @ Wg[e].T) * (x @ Wu[e].T) )  (fp32 accumulate, h rounded to bf16)
  * y = h @ Wd_t[e]  (Wd_t is the down projection stored transposed, [I][H]);  fp32 accumulate
  * out[t] = sum over routed experts k of w[t,k] * y_{t,e_k}; w are RAW sigmoid scores (norm_topk_prob=false)
  * verify tokens use all TOPK=8 routed experts; draft tokens use only the first r (highest score) of THEIR routed set
    with the same raw weights by default; renorm=True divides by the sum of the kept top-r weights (documented approximation).
"""
import torch

H, I, TOPK = 2048, 768, 8


def make_inputs(E, Tv, Td, seed=0, scale=0.03):
    g = torch.Generator().manual_seed(seed)
    T = Tv + Td
    x = (torch.rand(T, H, generator=g) * 2 - 1).bfloat16()
    wg = ((torch.rand(E, I, H, generator=g) * 2 - 1) * scale).bfloat16()
    wu = ((torch.rand(E, I, H, generator=g) * 2 - 1) * scale).bfloat16()
    wd = ((torch.rand(E, I, H, generator=g) * 2 - 1) * scale).bfloat16()  # transposed down: [E][I][H]
    logits = torch.randn(T, E, generator=g)
    top = torch.topk(logits, TOPK, dim=1)           # sorted descending
    idx = top.indices.to(torch.int32)
    w = torch.sigmoid(top.values).float()
    return x, wg, wu, wd, idx, w


def expert_forward(xs, wg, wu, wd):
    """xs [n,H] bf16 -> [n,H] fp32 (before routing weight)."""
    g = xs.float() @ wg.float().T
    u = xs.float() @ wu.float().T
    h = (torch.nn.functional.silu(g) * u).bfloat16().float()
    return h @ wd.float()


def _w(w, t, k_use, renorm):
    return w[t, :k_use] / w[t, :k_use].sum() if (renorm and k_use < TOPK) else w[t, :k_use]


def reference(x, wg, wu, wd, idx, w, Tv, r, renorm=False):
    T = x.shape[0]
    out = torch.zeros(T, H)
    for t in range(T):
        k_use = TOPK if t < Tv else r
        for k in range(k_use):
            e = int(idx[t, k])
            out[t] += _w(w, t, k_use, renorm)[k] * expert_forward(x[t:t + 1], wg[e], wu[e], wd[e])[0]
    return out


def reference_grouped(x, wg, wu, wd, idx, w, Tv, r, renorm=False):
    """Grouped-by-expert formulation (what the kernel does): one pass per expert over all its tokens."""
    T = x.shape[0]
    out = torch.zeros(T, H)
    E = wg.shape[0]
    for e in range(E):
        rows, ws = [], []
        for t in range(T):
            k_use = TOPK if t < Tv else r
            for k in range(k_use):
                if int(idx[t, k]) == e:
                    rows.append(t); ws.append(float(_w(w, t, k_use, renorm)[k]))
        if rows:
            y = expert_forward(x[rows], wg[e], wu[e], wd[e])
            for j, t in enumerate(rows):
                out[t] += ws[j] * y[j]
    return out


def weight_bytes(idx, Tv, r, E, fused):
    """Expert weight bytes streamed once per distinct (pass, expert)."""
    V = set(idx[:Tv].flatten().tolist())
    D = set(idx[Tv:, :r].flatten().tolist())
    n = len(V | D) if fused else len(V) + len(D)
    return n * 3 * I * H * 2
