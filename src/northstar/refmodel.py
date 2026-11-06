"""RefNorth: a small, explicit re-implementation of the Cohere2Moe forward pass that
reuses the *HF module weights* (no copies) but owns the KV cache, so we can:
  * roll back by moving a length pointer (verification rejection),
  * keep a separate draft KV cache,
  * run a draft mode: fixed layer skip and/or expert top-r of the native top-k,
  * time components (attn / router / experts / lm_head / ...) with CUDA events.

Architecture facts honoured (from transformers modeling_cohere2_moe.py + config):
  * parallel block: x = x + attn(ln(x)) + mlp(ln(x)) with ONE input_layernorm.
  * layer 0 is a dense MLP (first_k_dense_replace=1, intermediate 3072); layers 1.. are sparse MoE.
  * layer_types: full_attention every 4th layer (0,4,...,48), others sliding (window 4096,
    key visible iff q_pos - k_pos < window).  RoPE (interleaved/rotate_half-of-pairs) is applied
    on sliding layers and on the dense layer 0 (force_rope); other full-attention layers are NoPE.
  * router: logits = x W^T; top-k (sorted desc) of the logits; weights = sigmoid(topk logits)
    (expert_selection_fn="sigmoid"), NOT renormalised in the released config (norm_topk_prob=False).
    Draft top-r therefore keeps the first r of the sorted top-k with their native sigmoid weights,
    unrenormalised unless renorm_topr=True (an explicit, labelled variant).
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers.models.cohere2_moe.modeling_cohere2_moe import apply_rotary_pos_emb


class KVCache:
    """Per-layer KV. Full layers: linear buffer of max_len. Sliding layers: ring buffer of
    window+chunk slots (slot = pos % cap). Rollback = lowering `.len`; correct as long as every
    write has T <= cap - window + 1 (see read())."""

    def __init__(self, cfg, batch, max_len, layers, device, dtype, chunk=1024):
        self.cfg, self.B, self.max_len, self.chunk = cfg, batch, max_len, chunk
        self.W = cfg.sliding_window
        self.k, self.v, self.cap = {}, {}, {}
        hkv, d = cfg.num_key_value_heads, cfg.head_dim
        for i in layers:
            if cfg.layer_types[i] == "sliding_attention":
                cap = min(max_len, self.W + chunk)
            else:
                cap = max_len
            self.cap[i] = cap
            self.k[i] = torch.zeros(batch, hkv, cap, d, device=device, dtype=dtype)
            self.v[i] = torch.zeros(batch, hkv, cap, d, device=device, dtype=dtype)
        self.len = 0  # number of positions whose KV is valid

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in list(self.k.values()) + list(self.v.values()))

    def write(self, i, start, k, v):
        T, cap = k.shape[2], self.cap[i]
        s = start % cap
        if s + T <= cap:
            self.k[i][:, :, s : s + T] = k
            self.v[i][:, :, s : s + T] = v
        else:
            slots = torch.arange(start, start + T, device=k.device) % cap
            self.k[i].index_copy_(2, slots, k)
            self.v[i].index_copy_(2, slots, v)

    def read(self, i, lo, hi):
        cap = self.cap[i]
        s = lo % cap
        n = hi - lo
        assert n <= cap, "ring too small for requested span"
        if s + n <= cap:
            return self.k[i][:, :, s : s + n], self.v[i][:, :, s : s + n]
        slots = torch.arange(lo, hi, device=self.k[i].device) % cap
        return self.k[i].index_select(2, slots), self.v[i].index_select(2, slots)


class SectionTimer:
    """CUDA-event section timer. Usage: with timer.sec('attn'): ... ; timer.collect() -> {name: ms}."""

    def __init__(self, use_profiler=False):
        self.pending = []
        self.use_profiler = use_profiler  # annotate with record_function instead of events

    @contextlib.contextmanager
    def sec(self, name):
        if self.use_profiler:
            with torch.profiler.record_function("sec::" + name):
                yield
            return
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        yield
        b.record()
        self.pending.append((name, a, b))

    def collect(self):
        torch.cuda.synchronize()
        out = {}
        for n, a, b in self.pending:
            out[n] = out.get(n, 0.0) + a.elapsed_time(b)
        self.pending = []
        return out


@dataclass
class DraftMode:
    skip_layers: frozenset = frozenset()
    expert_r: int | None = None  # None => native top-k
    renorm_topr: bool = False

    def label(self):
        return f"skip{len(self.skip_layers)}_r{self.expert_r or 'native'}"


FULL = DraftMode()


class RefNorth:
    def __init__(self, hf_model, timer: SectionTimer | None = None):
        self.m = hf_model
        self.cfg = hf_model.config
        self.model = hf_model.model
        self.lm_head = hf_model.lm_head
        self.timer = timer
        self.record_experts = None  # if a list: append (layer, idx) for every MoE call
        assert self.cfg.num_shared_experts == 0
        assert getattr(self.cfg, "logit_scale", 1.0) is not None
        self.device = next(hf_model.parameters()).device
        self.dtype = next(hf_model.parameters()).dtype

    def new_cache(self, batch, max_len, layers=None, chunk=1024):
        layers = range(self.cfg.num_hidden_layers) if layers is None else layers
        return KVCache(self.cfg, batch, max_len, list(layers), self.device, self.dtype, chunk)

    def _sec(self, name):
        return self.timer.sec(name) if self.timer is not None else contextlib.nullcontext()

    # ---- attention -------------------------------------------------------------------
    def _attn(self, i, layer, h, cos, sin, cache, start):
        cfg = self.cfg
        a = layer.self_attn
        B, T, _ = h.shape
        D, H, Hkv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
        G = H // Hkv
        sliding = cfg.layer_types[i] == "sliding_attention"
        q = a.q_proj(h).view(B, T, H, D).transpose(1, 2)
        k = a.k_proj(h).view(B, T, Hkv, D).transpose(1, 2)
        v = a.v_proj(h).view(B, T, Hkv, D).transpose(1, 2)
        if sliding or a.force_rope:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        cache.write(i, start, k, v)
        S = start + T
        W = cfg.sliding_window
        lo = max(0, start - W + 1) if sliding else 0
        K, V = cache.read(i, lo, S)
        L = S - lo
        if T == 1:
            o = F.scaled_dot_product_attention(q.reshape(B, Hkv, G, D), K, V)  # GQA folded into rows
            o = o.reshape(B, 1, H * D)
        else:
            qpos = start + torch.arange(T, device=h.device)
            kpos = lo + torch.arange(L, device=h.device)
            if T <= 64:  # verify block: fold GQA groups into the row dim (no KV repeat)
                m = kpos[None, :] <= qpos[:, None]
                if sliding:
                    m &= kpos[None, :] > qpos[:, None] - W
                qg = q.reshape(B, Hkv, G, T, D).reshape(B, Hkv, G * T, D)
                o = F.scaled_dot_product_attention(qg, K, V, attn_mask=m.repeat(G, 1))
                o = o.reshape(B, Hkv, G, T, D).reshape(B, H, T, D).transpose(1, 2).reshape(B, T, H * D)
            else:  # prefill chunk
                Kf, Vf = K.repeat_interleave(G, dim=1), V.repeat_interleave(G, dim=1)
                if start == 0 and (not sliding or T <= W):
                    o = F.scaled_dot_product_attention(q, Kf, Vf, is_causal=True)
                else:
                    m = kpos[None, :] <= qpos[:, None]
                    if sliding:
                        m &= kpos[None, :] > qpos[:, None] - W
                    o = F.scaled_dot_product_attention(q, Kf, Vf, attn_mask=m[None, None])
                o = o.transpose(1, 2).reshape(B, T, H * D)
        return a.o_proj(o)

    # ---- mlp / moe ---------------------------------------------------------------------
    def _mlp(self, i, mlp, h, mode: DraftMode):
        if not hasattr(mlp, "gate"):  # dense prefix layer
            return mlp(h)
        B, T, Dm = h.shape
        hf = h.reshape(-1, Dm)
        with self._sec("router"):
            _, scores, idx = mlp.gate(hf)
            if mode.expert_r is not None:
                scores, idx = scores[:, : mode.expert_r], idx[:, : mode.expert_r]
                if mode.renorm_topr:
                    scores = scores / scores.sum(-1, keepdim=True)
        if self.record_experts is not None:
            self.record_experts.append((i, idx.detach().clone()))
        with self._sec("experts"):
            out = mlp.experts(hf, idx, scores)
        return out.reshape(B, T, Dm)

    # ---- forward -------------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, ids, cache: KVCache, start: int, mode: DraftMode = FULL, logits: str = "all"):
        """ids [B,T] at absolute positions start..start+T-1. Writes KV; does NOT touch cache.len.
        logits: 'all' | 'last' | 'none'."""
        cfg = self.cfg
        B, T = ids.shape
        assert 0 not in mode.skip_layers
        with self._sec("embed"):
            x = self.model.embed_tokens(ids)
            pos = (start + torch.arange(T, device=ids.device))[None]
            cos, sin = self.model.rotary_emb(x, pos)
        for i, layer in enumerate(self.model.layers):
            if i in mode.skip_layers:
                continue
            with self._sec("norm"):
                h = layer.input_layernorm(x)
            with self._sec("attn"):
                a = self._attn(i, layer, h, cos, sin, cache, start)
            with self._sec("mlp_dense" if not hasattr(layer.mlp, "gate") else "moe"):
                m = self._mlp(i, layer.mlp, h, mode)
            with self._sec("resid"):
                x = x + a + m
        if logits == "none":
            return None
        with self._sec("norm"):
            if logits == "last":
                x = x[:, -1:]
            x = self.model.norm(x)
        with self._sec("lm_head"):
            out = self.lm_head(x) * cfg.logit_scale
        return out

    @torch.no_grad()
    def prefill(self, ids, cache: KVCache, mode: DraftMode = FULL, chunk: int = 1024, logits: str = "last"):
        """Feed ids [B,P] chunked starting at cache.len; advances cache.len. Returns logits for last token."""
        P = ids.shape[1]
        out = None
        for s in range(0, P, chunk):
            e = min(P, s + chunk)
            last = e == P and logits != "none"
            out = self.forward(ids[:, s:e], cache, cache.len, mode, logits=logits if last else "none")
            cache.len += e - s
        return out
