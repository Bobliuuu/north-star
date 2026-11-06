"""Fake int4 (RTN, symmetric, group g, bf16 scale) for routed-expert weights, with an exact backup/restore editor.
Weights stay in their original dtype after rounding, so quality is measured on the real int4 grid; no speed is measured."""
from __future__ import annotations

import hashlib

import torch

from weightedit.recipe import moe_layers


def fake_quant_int4(w: torch.Tensor, group: int) -> torch.Tensor:
    """Round w (last dim divisible by group) to int4 [-8,7] with per-group absmax/7 scale rounded to bf16."""
    n = w.shape[-1]
    assert n % group == 0, f"last dim {n} not divisible by group {group}"
    out = torch.empty_like(w)
    flat_in, flat_out = w.reshape(-1, n), out.view(-1, n)
    step = max(1, (1 << 24) // n)  # bound fp32 temp memory
    for s in range(0, flat_in.shape[0], step):
        x = flat_in[s:s + step].float().reshape(-1, n // group, group)
        scale = (x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7).to(torch.bfloat16).float()
        q = torch.clamp(torch.round(x / scale), -8, 7)
        flat_out[s:s + step] = (q * scale).reshape(-1, n).to(w.dtype)
    return out


def bits_per_weight(group: int) -> float:
    return 4 + 16 / group


def tensor_hash(t: torch.Tensor, sampled=True) -> str:
    f = t.detach().reshape(-1)
    if sampled and f.numel() > 8192:
        f = torch.cat([f[:4096], f[-4096:]])
    return hashlib.sha256(f.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


class ExpertQuantEditor:
    """Holds a backup of every routed-expert tensor (host memory by default) and applies/restores int4 edits.
    apply() always starts from the ORIGINAL (restores first), so recipes never compound."""

    def __init__(self, m, backup_device="cpu"):
        self.m = m
        self.layers = moe_layers(m.config)
        self.backup = {}
        for i in self.layers:
            e = m.model.layers[i].mlp.experts
            self.backup[i] = (e.gate_up_proj.detach().to(backup_device, copy=True),
                              e.down_proj.detach().to(backup_device, copy=True))
        self.applied = []
        self.orig_hash = {i: self.layer_hash(i) for i in self.layers}

    def _experts(self, i):
        return self.m.model.layers[i].mlp.experts

    def layer_hash(self, i):
        e = self._experts(i)
        return tensor_hash(e.gate_up_proj) + tensor_hash(e.down_proj)

    @torch.no_grad()
    def restore_all(self):
        for i in self.applied:
            e = self._experts(i)
            e.gate_up_proj.copy_(self.backup[i][0].to(e.gate_up_proj.device))
            e.down_proj.copy_(self.backup[i][1].to(e.down_proj.device))
        self.applied = []

    @torch.no_grad()
    def apply(self, layers, group):
        self.restore_all()
        for i in layers:
            e = self._experts(i)
            gu = self.backup[i][0].to(e.gate_up_proj.device)
            dn = self.backup[i][1].to(e.down_proj.device)
            e.gate_up_proj.copy_(fake_quant_int4(gu, group))
            e.down_proj.copy_(fake_quant_int4(dn, group))
        self.applied = list(layers)

    def unlisted_identical(self, layers):
        s = set(layers)
        return all(self.layer_hash(i) == self.orig_hash[i] for i in self.layers if i not in s)

    def listed_changed(self, layers):
        return all(self.layer_hash(i) != self.orig_hash[i] for i in layers)
