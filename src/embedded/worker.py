"""Internal stateful workers injected into a frozen decoder layer's residual stream.
Modes (all share the same plumbing so ablations isolate ONE factor):
  multi    : K private slot states, learned router (token -> slot attention), persistent GRU state
  oracle   : like multi but router replaced by ground-truth register id (upper bound for separation)
  uniform  : like multi but every token updates every slot equally (private states, NO routing)
  single   : ONE persistent state (width matched to multi's param count via --match_params)
  multi_rr : multi + routed READ (return weighted by token-conditioned attention over slots)
  adapter  : stateless per-token MLP (no persistence at all)
Slots share weights; slot identity = random per-sequence vector, so K can be raised at test time.
Return = sum_k phi(LN(s_k)) added to the residual BEFORE the remaining layers."""
import torch, torch.nn as nn, torch.nn.functional as F

class Workers(nn.Module):
    def __init__(s, d_model, K=8, d_s=64, mode="multi", d_k=64, single_width=None, occ_gate=True, ret_norm=False):
        super().__init__()
        s.mode, s.K, s.d_s, s.occ_gate, s.ret_norm = mode, K, d_s, occ_gate, ret_norm
        s.ret_scale = 1.0
        if mode == "single": s.K_eff, s.ds_eff = 1, (single_width or d_s * K)
        else: s.K_eff, s.ds_eff = K, d_s
        ds = s.ds_eff
        s.inp = nn.Linear(d_model, ds)
        s.q = nn.Linear(d_model, d_k); s.k = nn.Linear(ds, d_k)
        s.wg = nn.Linear(d_model, 1)
        s.cell = nn.GRUCell(ds, ds)
        s.ln = nn.LayerNorm(ds)
        s.phi = nn.Sequential(nn.Linear(ds, 4 * ds), nn.GELU(), nn.Linear(4 * ds, d_model))
        nn.init.zeros_(s.phi[2].weight); nn.init.zeros_(s.phi[2].bias)
        if mode == "adapter":
            w = single_width or ds
            s.ad = nn.Sequential(nn.Linear(d_model, 4 * w), nn.GELU(), nn.Linear(4 * w, d_model))
            nn.init.zeros_(s.ad[2].weight); nn.init.zeros_(s.ad[2].bias)
        if mode == "multi_rr":   # routed READ: return weights over slots from the current token (own query/key)
            s.q2 = nn.Linear(d_model, d_k); s.k2 = nn.Linear(ds, d_k)
        s.ablate = None      # None|'zero'|'shuffle'|'drop:<k>'
        s.reg = None         # [B,T] long ground-truth register ids (-1 none), set by caller, used by oracle
        s.stats = None

    def forward(s, h, mask):
        B, T, D = h.shape
        if s.mode == "adapter":
            d = s.ad(h.float()) * mask[..., None]
            return d * 0 if s.ablate == "zero" else (d.roll(1, 0) if s.ablate == "shuffle" else d)
        hf = h.float(); K, ds = s.K_eff, s.ds_eff
        st = torch.randn(B, K, ds, device=h.device) * 0.5
        x = s.inp(hf); q = s.q(hf); wg = torch.sigmoid(s.wg(hf)).squeeze(-1)
        outs, assigns = [], []
        occ = torch.zeros(B, K, device=h.device)          # cumulative write mass per slot (0 = never used)
        for t in range(T):
            m = mask[:, t]
            if s.mode == "oracle":
                r = s.reg[:, t]
                a = F.one_hot(r.clamp(min=0) % K, K).float() * (r >= 0).float()[:, None]
            elif s.mode == "uniform" and K > 1:
                a = torch.full((B, K), 1.0 / K, device=h.device)
            elif K > 1:
                key = s.k(st); logit = (key * q[:, t, None]).sum(-1) / key.shape[-1] ** 0.5
                a = F.softmax(logit * 3.0, dim=-1)
            else:
                a = torch.ones(B, 1, device=h.device)
            if s.ablate and s.ablate.startswith("drop:"):
                a = a.clone(); a[:, int(s.ablate[5:]) % K] = 0
            g = (a * wg[:, t, None] * m[:, None]).unsqueeze(-1)
            if s.mode == "uniform": g = g * K    # keep total write mass comparable
            g = g.clamp(max=1.0)
            new = s.cell(x[:, t, None].expand(B, K, ds).reshape(B * K, ds), st.reshape(B * K, ds)).view(B, K, ds)
            st = (1 - g) * st + g * new
            occ = (occ + g.squeeze(-1)).clamp(max=1.0)
            pk = s.phi(s.ln(st))                                     # [B,K,D]
            if s.occ_gate: pk = pk * occ.unsqueeze(-1)               # unused slots (pure identity noise) return nothing
            if s.mode == "multi_rr" and K > 1:
                rl = (s.k2(st) * s.q2(hf[:, t])[:, None]).sum(-1) / 8.0
                pk = pk * F.softmax(rl, dim=-1).unsqueeze(-1) * K     # routed read (scaled to keep magnitude)
            if s.ret_norm and K > 1:                                 # mean over occupied slots: return magnitude no longer grows with #active registers
                outs.append(pk.sum(1) / occ.sum(1, keepdim=True).clamp(min=1.0))
            else:
                outs.append(pk.sum(1))
            assigns.append(a.detach())
        d = torch.stack(outs, 1) * mask[..., None] * s.ret_scale
        s.stats = torch.stack(assigns, 1)
        if s.ablate == "zero": d = d * 0
        if s.ablate == "shuffle": d = d.roll(1, 0)
        return d

def nparams(m): return sum(p.numel() for p in m.parameters())

def matched_single_width(d_model, K, d_s):
    """width of single state so Workers('single') has ~ same #params as Workers('multi')"""
    target = nparams(Workers(d_model, K, d_s, "multi"))
    lo, hi = d_s, d_s * K * 4
    while lo < hi:
        mid = (lo + hi) // 2
        if nparams(Workers(d_model, K, d_s, "single", single_width=mid)) < target: lo = mid + 1
        else: hi = mid
    return lo

def matched_adapter_width(d_model, K, d_s):
    target = nparams(Workers(d_model, K, d_s, "multi"))
    lo, hi = 8, 4096
    while lo < hi:
        mid = (lo + hi) // 2
        if nparams(Workers(d_model, K, d_s, "adapter", single_width=mid)) < target: lo = mid + 1
        else: hi = mid
    return lo


class LoRABank(nn.Module):
    """Hand-rolled LoRA on q/k/v/o of selected decoder layers. Params fp32; delta = scale * (x A^T) B^T added to the frozen linear's output."""
    def __init__(s, specs, scale):
        super().__init__()
        s.scale = scale; s.A = nn.ParameterList(); s.B = nn.ParameterList()
        for (din, dout, r) in specs:
            A = torch.empty(r, din); nn.init.kaiming_uniform_(A, a=5 ** 0.5)
            s.A.append(nn.Parameter(A)); s.B.append(nn.Parameter(torch.zeros(dout, r)))

def attach_lora(lm, layer, place, target, alpha, dev):
    """Rank matching: per-layer uniform base rank r (largest with params <= target), then r+1 on as many
    layers as keep total closest to target. Returns the LoRABank (hooks registered on the frozen linears)."""
    n = len(lm.model.layers)
    idx = list(range(0, layer + 1)) if place == "pre" else list(range(layer + 1, n))
    def mods(i):
        at = lm.model.layers[i].self_attn
        return [at.q_proj, at.k_proj, at.v_proj, at.o_proj]
    per = [sum(m.in_features + m.out_features for m in mods(i)) for i in idx]   # params per unit rank, per layer
    r0 = max(1, int(target // sum(per))); ranks = [r0] * len(idx)
    best = (abs(sum(per[j] * ranks[j] for j in range(len(idx))) - target), list(ranks))
    for j in range(len(idx)):
        ranks[j] += 1
        tot = sum(per[k] * ranks[k] for k in range(len(idx)))
        if abs(tot - target) < best[0]: best = (abs(tot - target), list(ranks))
    ranks = best[1]
    specs = [(m.in_features, m.out_features, ranks[j]) for j, i in enumerate(idx) for m in mods(i)]
    bank = LoRABank(specs, alpha).to(dev)
    c = 0
    for j, i in enumerate(idx):
        for m in mods(i):
            def hook(mod, inp, out, c=c):
                x = inp[0].float()
                return out + (bank.scale * (x @ bank.A[c].t() @ bank.B[c].t())).to(out.dtype)
            m.register_forward_hook(hook); c += 1
    print("lora place", place, "layers", idx[0], "-", idx[-1], "ranks", sorted(set(ranks)), "params", sum(p.numel() for p in bank.parameters()), "target", target, flush=True)
    return bank
