"""Draft-mode patching of a loaded Cohere2Moe HF model, WITHOUT altering verifier mode.

Mechanism: for each patched module we set an *instance attribute* `forward` (closure over the original bound
method).  The closure first checks `patch.active`; when False it calls the original forward untouched.  `remove()`
deletes the instance attributes so the module is back to its class forward.  No parameter, buffer, config or
class is ever modified.  Verifier guarantees (tests/test_hooks.py):
  * params/buffers bit-identical (hash) before, during, after;
  * with the patch installed but inactive (or inside `patch.verifier()`), outputs are bit-identical to unpatched;
  * after `remove()`, no module has a `forward` entry in `__dict__`.
Covered paths: HF native forward (full-sequence / teacher-forced, and HF cache decode for attention).  RefNorth
(src/northstar/refmodel.py) bypasses module forwards; use `to_ref_draft_mode` for its skip/top-r subset."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import torch

from .attention import AttnCost, make_selector, sparse_attention
from .config import DraftConfig, attn_layer_ids
from .experts import EATState, eat_select, slot_output_norms, topr


@dataclass
class DraftStats:
    attn: AttnCost = field(default_factory=AttnCost)
    expert_tokens: int = 0
    expert_slots_native: int = 0
    expert_slots_active: int = 0
    layers_skipped: int = 0

    def summary(self):
        d = self.attn.summary()
        d.update(expert_tokens=self.expert_tokens,
                 experts_per_token=self.expert_slots_active / max(self.expert_tokens, 1),
                 experts_native_per_token=self.expert_slots_native / max(self.expert_tokens, 1))
        return d


class DraftPatch:
    def __init__(self, model, cfg: DraftConfig, eat_states: dict | None = None):
        self.model, self.cfg = model, cfg
        self.mc = model.config
        self.eat_states = eat_states or {}
        self.active = False
        self.stats = DraftStats()
        self._patched = []
        self.selector = make_selector(cfg) if cfg.attn_on else None
        assert getattr(self.mc, "num_shared_experts", 0) == 0
        if cfg.expert_policy == "eat":
            for i, t in enumerate(self.mc.mlp_layer_types):
                if t != "dense" and i not in self.eat_states:
                    self.eat_states[i] = EATState(self.mc.num_experts)  # empty history unless calibrated

    # ---- lifecycle -----------------------------------------------------------------------
    def install(self):
        assert not self._patched, "already installed"
        cfg, mc = self.cfg, self.mc
        attn_ids = set(attn_layer_ids(mc, cfg.attn_layers)) if cfg.attn_on else set()
        for i, layer in enumerate(self.model.model.layers):
            if i in cfg.skip_layers:
                self._wrap(layer, self._skip_fwd(layer.forward))
            if i in attn_ids:
                assert mc.layer_types[i] == "full_attention"
                self._wrap(layer.self_attn, self._attn_fwd(i, layer.self_attn, layer.self_attn.forward))
            if cfg.expert_on and mc.mlp_layer_types[i] != "dense":
                self._wrap(layer.mlp, self._moe_fwd(i, layer.mlp, layer.mlp.forward))
        return self

    def remove(self):
        for m in self._patched:
            m.__dict__.pop("forward", None)
        self._patched = []
        self.active = False

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

    def _wrap(self, module, fn):
        assert "forward" not in module.__dict__
        module.forward = fn
        self._patched.append(module)

    # ---- wrappers ----------------------------------------------------------------------------
    def _skip_fwd(self, orig):
        def fwd(hidden_states, *a, **k):
            if not self.active:
                return orig(hidden_states, *a, **k)
            self.stats.layers_skipped += 1
            return hidden_states
        return fwd

    def _attn_fwd(self, i, mod, orig):
        cfg = self.cfg

        def fwd(hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kw):
            if not self.active:
                return orig(hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask,
                            past_key_values=past_key_values, **kw)
            from transformers.models.cohere2_moe.modeling_cohere2_moe import apply_rotary_pos_emb
            ishape = hidden_states.shape[:-1]
            hshape = (*ishape, -1, mod.head_dim)
            q = mod.q_proj(hidden_states).view(hshape).transpose(1, 2)
            k = mod.k_proj(hidden_states).view(hshape).transpose(1, 2)
            v = mod.v_proj(hidden_states).view(hshape).transpose(1, 2)
            if mod.sliding_window is not None or mod.force_rope:  # NoPE on layers 4..48
                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
            if past_key_values is not None:
                k, v = past_key_values.update(k, v, mod.layer_idx)
            T, S = q.shape[2], k.shape[2]
            qpos = torch.arange(S - T, S, device=q.device)
            o = sparse_attention(q, k, v, qpos, self.selector, cfg, mod.scaling, self.stats.attn)
            o = o.transpose(1, 2).reshape(*ishape, -1).contiguous()
            return mod.o_proj(o), None
        return fwd

    def _moe_fwd(self, i, mlp, orig):
        cfg = self.cfg

        def fwd(hidden_states):
            if not self.active:
                return orig(hidden_states)
            B, T, Dm = hidden_states.shape
            flat = hidden_states.view(-1, Dm)
            logits, w, idx = mlp.gate(flat)
            if cfg.expert_policy == "eat":
                st = self.eat_states[i].to(flat.device)
                w2, idx2, act = eat_select(logits, w, idx, st, cfg.eat, cfg.weight_convention,
                                           mlp.gate.expert_selection_fn, cfg.expert_r)
                n_act = int(act.sum())
            else:
                w2, idx2 = topr(w, idx, cfg.expert_r, cfg.weight_convention)
                n_act = idx2.numel()
            self.stats.expert_tokens += flat.shape[0]
            self.stats.expert_slots_native += idx.numel()
            self.stats.expert_slots_active += n_act
            return mlp.experts(flat, idx2, w2).reshape(B, T, Dm)
        return fwd


class EATCalibrator:
    """Runs alongside VERIFIER-mode forwards (outputs unchanged) and accumulates EAT history (N, Ptot, Cs) per MoE
    layer from the native top-k.  Costs an extra pass of expert math (output norms) - calibration only."""

    def __init__(self, model, alpha=0.95):
        self.model, self.alpha = model, alpha
        self.states = {i: EATState(model.config.num_experts) for i, t in enumerate(model.config.mlp_layer_types)
                       if t != "dense"}
        self._patched = []

    def __enter__(self):
        for i, layer in enumerate(self.model.model.layers):
            if i in self.states:
                mlp, orig = layer.mlp, layer.mlp.forward

                def fwd(hidden_states, mlp=mlp, orig=orig, i=i):
                    flat = hidden_states.view(-1, hidden_states.shape[-1])
                    with torch.no_grad():
                        _, w, idx = mlp.gate(flat)
                        on = slot_output_norms(mlp.experts, flat, idx)
                        self.states[i].to(flat.device).update(idx, w, on, self.alpha)
                    return orig(hidden_states)
                mlp.forward = fwd
                self._patched.append(mlp)
        return self

    def __exit__(self, *a):
        for m in self._patched:
            m.__dict__.pop("forward", None)
        self._patched = []


def to_ref_draft_mode(cfg: DraftConfig):
    """Map the subset RefNorth supports (skip layers, expert top-r with raw/renorm). Attention budget and EAT are
    NOT representable there (RefNorth has no sparse-attention hook)."""
    from northstar.refmodel import DraftMode
    assert cfg.expert_policy == "topr" and cfg.weight_convention in ("raw_sigmoid", "renorm") and not cfg.attn_on
    return DraftMode(skip_layers=cfg.skip_layers, expert_r=cfg.expert_r, renorm_topr=cfg.weight_convention == "renorm")
