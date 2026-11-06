"""Draft-only router adapters. The edit is applied ONLY inside `EditedDraft.draft()`; verifier mode is untouched.

Draft MoE step (matches approx.DraftConfig(expert_r=r, weight_convention='raw_sigmoid') at identity init):
    logits' = logits + delta_l(h)          delta = (h A) B  | b
    idx     = topr(logits')                weights = gain_l * sigmoid(logits'[idx])
Identity init: B = 0, b = 0, gain = 1  =>  step 0 == the unedited top-r draft."""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn

from torch.func import functional_call

from .quant import fake_quant
from .recipe import Recipe, moe_layers


class RouterAdapters(nn.Module):
    def __init__(self, mc, recipe: Recipe):
        super().__init__()
        self.kind, self.rank = recipe.kind, recipe.rank
        self.layer_ids = recipe.layer_ids(mc)
        H, E = mc.hidden_size, mc.num_experts
        g = torch.Generator().manual_seed(recipe.seed)
        self.A, self.B, self.b, self.g = nn.ParameterDict(), nn.ParameterDict(), nn.ParameterDict(), nn.ParameterDict()
        for i in self.layer_ids:
            k = str(i)
            if recipe.kind == "router_lowrank":
                self.A[k] = nn.Parameter(torch.randn(H, recipe.rank, generator=g) * 0.02)
                self.B[k] = nn.Parameter(torch.zeros(recipe.rank, E))
            else:
                self.b[k] = nn.Parameter(torch.zeros(E))
            if recipe.gain:
                self.g[k] = nn.Parameter(torch.ones(()))

    def has(self, i):
        return str(i) in self.A or str(i) in self.b or str(i) in self.g

    def delta(self, i, h):
        k = str(i)
        if k in self.A:
            return (h.float() @ self.A[k].to(h.device)) @ self.B[k].to(h.device)
        if k in self.b:
            return self.b[k].to(h.device)[None, :].expand(h.shape[0], -1)
        return 0.0

    def gain(self, i, device):
        k = str(i)
        return self.g[k].to(device) if k in self.g else 1.0

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def flops_per_token_layer(self, E):
        if self.kind == "router_lowrank":
            H = next(iter(self.A.values())).shape[0]
            return 2 * H * self.rank + 2 * self.rank * E
        return E


class EditedDraft:
    """Context manager: patches MoE blocks (instance-level forward, same pattern as approx.DraftPatch).
    Layers without an adapter still run plain top-r (gain 1, no delta) in draft mode, so the draft is consistent."""

    def __init__(self, model, adapters: RouterAdapters, expert_r: int, expert_quant: str = "none"):
        self.model, self.adapters, self.r = model, adapters, expert_r
        self.quant = expert_quant
        self.mc = model.config
        assert getattr(self.mc, "num_shared_experts", 0) == 0
        assert self.mc.expert_selection_fn == "sigmoid"
        self.active = False
        self._patched = []
        self.slots_active = 0
        self.tokens = 0

    def install(self):
        assert not self._patched
        for i in moe_layers(self.mc):
            mlp = self.model.model.layers[i].mlp
            assert "forward" not in mlp.__dict__
            mlp.forward = self._moe_fwd(i, mlp, mlp.forward)
            self._patched.append(mlp)
        return self

    def remove(self):
        for m in self._patched:
            m.__dict__.pop("forward", None)
        self._patched, self.active = [], False

    def __enter__(self):
        return self.install()

    def __exit__(self, *a):
        self.remove()

    @contextlib.contextmanager
    def draft(self):
        prev, self.active = self.active, True
        try:
            yield self
        finally:
            self.active = prev

    @contextlib.contextmanager
    def verifier(self):
        prev, self.active = self.active, False
        try:
            yield self
        finally:
            self.active = prev

    def _moe_fwd(self, i, mlp, orig):
        def fwd(hidden_states):
            if not self.active:
                return orig(hidden_states)
            B, T, Dm = hidden_states.shape
            flat = hidden_states.reshape(-1, Dm)
            logits, w_native, _ = mlp.gate(flat)
            lg = logits.float() + self.adapters.delta(i, flat)
            top = lg.topk(self.r, dim=-1)
            w = torch.sigmoid(top.values) * self.adapters.gain(i, flat.device)
            self.tokens += flat.shape[0]
            self.slots_active += top.indices.numel()
            if self.quant != "none":  # draft-only 4-bit copy: dequantised on the fly, original params untouched
                ex = mlp.experts
                qp = {n: fake_quant(getattr(ex, n).detach(), self.quant) for n in ("gate_up_proj", "down_proj")}
                out = functional_call(ex, qp, (flat, top.indices, w.to(w_native.dtype)))
            else:
                out = mlp.experts(flat, top.indices, w.to(w_native.dtype))
            return out.reshape(B, T, Dm)
        return fwd
