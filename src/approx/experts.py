"""Expert-budget policies for draft mode (axis A).

North router facts (modeling_cohere2_moe.py, tiny-config-verified in tests):
  logits = h W^T over 128 experts; top-8 on RAW logits; weights = sigmoid(selected logits); NOT renormalised
  (norm_topk_prob=False).  Everything below consumes (router_logits, native_weights[N,k] sorted desc, native_idx[N,k]).

Policies
  topr : keep the first r of the native top-k (a subset of the draft's own top-k), weight convention:
         raw_sigmoid  - keep native sigmoid weights (dropped mass is simply lost)
         renorm       - divide kept weights by their sum (weights sum to 1; NOT North's native convention)
         mass_preserve- renorm then rescale to the native top-k weight sum (keeps total gate mass)
  eat  : Expert Account Tracker (arXiv 2609.33614) - see class EATState / eat_select and docs/APPROX.md for
         exactly what is reproduced.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .config import EATParams


def apply_weight_convention(native_w: torch.Tensor, kept_w: torch.Tensor, conv: str) -> torch.Tensor:
    """native_w [N,k] all native weights; kept_w [N,k'] (zeros allowed for dropped slots)."""
    kf = kept_w.float()
    if conv == "raw_sigmoid":
        out = kf
    elif conv == "renorm":
        out = kf / kf.sum(-1, keepdim=True).clamp_min(1e-8)
    elif conv == "mass_preserve":
        out = kf / kf.sum(-1, keepdim=True).clamp_min(1e-8) * native_w.float().sum(-1, keepdim=True)
    else:
        raise ValueError(conv)
    return out.to(kept_w.dtype)


def topr(native_w, native_idx, r, conv):
    r = min(r, native_idx.shape[-1])
    return apply_weight_convention(native_w, native_w[:, :r], conv), native_idx[:, :r]


# ---------------------------------------------------------------------------------- EAT
@dataclass
class EATState:
    """Per-layer expert history (paper section 'Expert Account Tracker').
      N[i]    activation count of expert i
      Ptot[i] cumulative router gating weight assigned to expert i over its activations
      Cs[i]   EMA of expert output L2 norm:  C <- (1-alpha) C + alpha ||o||_2
    """
    num_experts: int
    N: torch.Tensor = None
    Ptot: torch.Tensor = None
    Cs: torch.Tensor = None

    def __post_init__(self):
        z = lambda: torch.zeros(self.num_experts, dtype=torch.float64)
        self.N, self.Ptot, self.Cs = z(), z(), z()

    def to(self, device):
        self.N, self.Ptot, self.Cs = self.N.to(device), self.Ptot.to(device), self.Cs.to(device)
        return self

    @torch.no_grad()
    def update(self, idx, w, out_norm, alpha):
        """idx,w,out_norm: [N,k] (token order = EMA order). Exact sequential EMA via closed form per expert."""
        dev = self.N.device
        idx, w, out_norm = idx.to(dev), w.double().to(dev), out_norm.double().to(dev)
        flat_i, flat_w, flat_o = idx.reshape(-1), w.reshape(-1), out_norm.reshape(-1)
        for e in torch.unique(flat_i).tolist():
            m = flat_i == e  # row-major flatten keeps token order
            o = flat_o[m]
            n_prev = self.N[e].item()
            c = self.Cs[e]
            if n_prev == 0:
                c, o = o[0], o[1:]
            n = o.numel()
            if n:
                pw = (1 - alpha) ** torch.arange(n - 1, -1, -1, device=dev, dtype=torch.float64)
                c = (1 - alpha) ** n * c + alpha * (pw * o).sum()
            self.Cs[e] = c
            self.N[e] += m.sum()
            self.Ptot[e] += flat_w[m].sum()

    def history_score(self, p: EATParams):
        """S_h,i = w1 N_i/sum N + w2 Ptot_i/N_i + w3 Cs_i   (as printed; no per-term normalisation)."""
        tot = self.N.sum().clamp_min(1)
        mean_p = torch.where(self.N > 0, self.Ptot / self.N.clamp_min(1), torch.zeros_like(self.Ptot))
        w1, w2, w3 = p.w
        return (w1 * self.N / tot + w2 * mean_p + w3 * self.Cs).float()


@torch.no_grad()
def slot_output_norms(experts, hidden, idx):
    """||expert_{idx[n,s]}(h_n)||_2 for every (token, slot), unweighted. Mirrors Cohere2MoeExperts math.
    This is an EXTRA computation (needed for EAT's C_s history): only used in calibration/verifier passes."""
    N, k = idx.shape
    out = torch.zeros(N, k, device=hidden.device, dtype=torch.float32)
    for e in torch.unique(idx).tolist():
        tok, slot = torch.where(idx == e)
        g, u = F.linear(hidden[tok], experts.gate_up_proj[e]).chunk(2, dim=-1)
        y = F.linear(experts.act_fn(g) * u, experts.down_proj[e])
        out[tok, slot] = y.float().norm(dim=-1)
    return out


@torch.no_grad()
def eat_select(logits, native_w, native_idx, state: EATState, p: EATParams, conv: str, selection_fn="sigmoid",
               r_cap=None):
    """Returns (weights[N,K], idx[N,K], active[N,K] bool). Dropped slots have weight 0 (functionally dropped).
    Implements: I = (1-beta) p + beta S_h ; tau = max(a1 mu, mu - a2 sigma) over raw gating scores ;
    E = {I > tau}; |E|=0 -> argmax I; |E|>K -> top-K by I; renormalise (paper) -> via `conv`."""
    N, E = logits.shape
    K = native_idx.shape[-1]
    g = torch.sigmoid(logits.float()) if selection_fn == "sigmoid" else torch.softmax(logits.float(), -1)
    Sh = state.history_score(p).to(logits.device)
    I = (1 - p.beta) * g + p.beta * Sh[None]
    mu, sd = g.mean(-1, keepdim=True), g.std(-1, unbiased=False, keepdim=True)
    tau = torch.maximum(p.a1 * mu, mu - p.a2 * sd) * p.tau_scale
    if p.tau_bounds is not None:
        lo = torch.quantile(g, p.tau_bounds[0], dim=-1, keepdim=True)
        hi = torch.quantile(g, p.tau_bounds[1], dim=-1, keepdim=True)
        tau = torch.minimum(torch.maximum(tau, lo), hi)
    allowed = torch.ones_like(g, dtype=torch.bool)
    if p.candidate_scope == "native_topk":
        allowed = torch.zeros_like(g, dtype=torch.bool).scatter_(1, native_idx, True)
    Im = I.masked_fill(~allowed, float("-inf"))
    cand = (Im > tau) & allowed
    cap = K if r_cap is None else min(K, r_cap)
    topv, topi = Im.topk(cap, dim=-1)
    keep = torch.zeros_like(cand).scatter_(1, topi, True) & cand
    empty = ~keep.any(-1)
    keep[empty, Im[empty].argmax(-1)] = True
    # slot form: order slots by I desc among kept
    key = Im.masked_fill(~keep, float("-inf"))
    v, idx = key.topk(cap, dim=-1)
    active = torch.isfinite(v)
    gw = g.gather(1, idx)  # raw gating score g_i of the selected experts
    w = apply_weight_convention(native_w, (gw * active).to(native_w.dtype), conv)
    return w, idx, active
