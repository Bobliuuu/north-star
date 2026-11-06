"""Experiment B: recover quantization damage with an always-on router adapter (verifier path), trained by
self-distillation toward the ORIGINAL model's cached logits. Edit changes the served model => full quality gates."""
from __future__ import annotations

import contextlib
import time

import numpy as np
import torch

from weightedit.adapter import RouterAdapters
from weightedit.distill import frozen, grad_checkpointed, kl_loss, logits_of
from weightedit.recipe import Recipe, moe_layers

from .metrics import boot_reduction_lower, eval_vs_teacher, summarize


class AlwaysOnRouterEdit:
    """Instance-level MoE forward wrapper: logits' = logits + delta; top-K (native K) on logits'; w = gain*sigmoid."""

    def __init__(self, model, adapters: RouterAdapters):
        self.model, self.adapters, self.mc = model, adapters, model.config
        assert self.mc.expert_selection_fn == "sigmoid" and not self.mc.norm_topk_prob
        self.enabled, self._patched = True, []

    def __enter__(self):
        for i in moe_layers(self.mc):
            mlp = self.model.model.layers[i].mlp
            assert "forward" not in mlp.__dict__
            mlp.forward = self._fwd(i, mlp, mlp.forward)
            self._patched.append(mlp)
        return self

    def __exit__(self, *a):
        for m in self._patched:
            m.__dict__.pop("forward", None)
        self._patched = []

    @contextlib.contextmanager
    def off(self):
        prev, self.enabled = self.enabled, False
        try:
            yield
        finally:
            self.enabled = prev

    def _fwd(self, i, mlp, orig):
        def fwd(h):
            if not self.enabled:
                return orig(h)
            B, T, D = h.shape
            flat = h.reshape(-1, D)
            logits, w_nat, idx = mlp.gate(flat)
            lg = logits.float() + self.adapters.delta(i, flat)
            top = lg.topk(idx.shape[-1], dim=-1)
            w = torch.sigmoid(top.values) * self.adapters.gain(i, flat.device)
            return mlp.experts(flat, top.indices, w.to(w_nat.dtype)).reshape(B, T, D)
        return fwd


def train_recover(m, patch, adapters, seqs, lo, cache, steps, lr, batch, seed=0, log=print, ckpt=True):
    dev = next(m.parameters()).device
    adapters.to(dev)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(adapters.parameters(), lr=lr, weight_decay=0.0)
    losses, t0 = [], time.time()
    with frozen(m), (grad_checkpointed(m) if ckpt else contextlib.nullcontext()):
        for step in range(steps):
            ix = torch.randint(len(seqs), (batch,), generator=g).tolist()
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            for j in ix:  # one sequence at a time (variable teacher slices), grads accumulate
                x = torch.tensor([seqs[j]], device=dev)
                t = cache[j].to(dev)[None]
                d = logits_of(m, x)[:, lo:-1]
                loss = kl_loss(t, d) / batch
                loss.backward()
                tot += loss.item()
            opt.step()
            losses.append(tot)
            if step % max(1, steps // 10) == 0:
                log(f"  recover step {step} kl={tot:.4f}")
    return dict(loss=losses, first=losses[0], last=losses[-1], seconds=time.time() - t0)


def run_experiment_B(m, editor, recipe, data, train, cfg, log=print, ckpt=lambda n, o: None):
    """data: dev_* (gate split) ; train: tr_seqs, tr_lo, tr_cache. cfg: steps, lr, batch, rank, min_rel_reduction=0.15."""
    mc = m.config
    editor.apply(recipe.layers(), recipe.group)
    k0, a0 = eval_vs_teacher(m, data["dev_seqs"], data["dev_lo"], data["dev_cache"])
    kt0, at0 = eval_vs_teacher(m, train["tr_seqs"], train["tr_lo"], train["tr_cache"])
    r = Recipe(kind="router_lowrank", rank=cfg.get("rank", 4), gain=True, seed=cfg.get("seed", 0))
    ad = RouterAdapters(mc, r)
    with AlwaysOnRouterEdit(m, ad) as patch:
        tr = train_recover(m, patch, ad, train["tr_seqs"], train["tr_lo"], train["tr_cache"], cfg["steps"], cfg["lr"],
                           cfg.get("batch", 2), cfg.get("seed", 0), log, cfg.get("ckpt", True))
        with torch.no_grad():
            k1, a1 = eval_vs_teacher(m, data["dev_seqs"], data["dev_lo"], data["dev_cache"])
            kt1, at1 = eval_vs_teacher(m, train["tr_seqs"], train["tr_lo"], train["tr_cache"])
            with patch.off():  # sanity: adapter off == edited-only baseline
                k_off, _ = eval_vs_teacher(m, data["dev_seqs"], data["dev_lo"], data["dev_cache"])
    editor.restore_all()
    s0, s1 = summarize(k0, a0), summarize(k1, a1)
    rel = (s0["kl"] - s1["kl"]) / max(s0["kl"], 1e-12)
    lo5 = boot_reduction_lower(k0, k1)
    H, I = mc.hidden_size, mc.intermediate_size
    flops_frac = ad.flops_per_token_layer(mc.num_experts) / (mc.num_experts_per_tok * 2 * 3 * H * I)
    gates = dict(recover=bool(rel >= cfg.get("min_rel_reduction", 0.15) and lo5 > 0 and s1["agree"] >= s0["agree"]),
                 cost=bool(flops_frac <= 0.05))
    out = dict(recipe=recipe.to_dict(), kl_before=s0["kl"], kl_after=s1["kl"], rel_reduction=rel, boot_lower_rel=lo5,
               kl_train_before=summarize(kt0, at0)["kl"], kl_train_after=summarize(kt1, at1)["kl"],
               agree_before=s0["agree"], agree_after=s1["agree"], kl_adapter_off=summarize(k_off, a0)["kl"],
               adapter_params=ad.n_params(), adapter_flops_frac=flops_frac, train=dict(first=tr["first"], last=tr["last"],
               seconds=tr["seconds"], steps=cfg["steps"]), gates=gates, passed=all(gates.values()))
    ckpt("exp_B", out)
    return out
