"""CUDA-graph-friendly forward for North (reuses the HF module weights through RefNorth, no copies).

Static shapes: per-layer KV is a LINEAR buffer of Lmax slots for every layer (sliding layers included); the position is a
0-d device tensor `start`; causal/sliding masks are built on-device from it; KV writes use index_copy_ with device indices.
So one captured graph serves every position. Rollback = just reuse earlier `start` (stale slots are masked by kpos<=qpos).
NOTE (honest cost caveat): attention always reads all Lmax slots (padded), so for contexts > sliding_window (4096) the
sliding layers are over-charged; at ctx<=4096 padded cost ~= true cost when Lmax ~= ctx.
"""
import time, statistics as st
import torch
import torch.nn.functional as F
from transformers.models.cohere2_moe.modeling_cohere2_moe import apply_rotary_pos_emb

ALL = frozenset({"attn", "router", "experts", "lm"})


class GraphNorth:
    def __init__(self, ref, Lmax, attn_impl="bmm"):
        self.attn_impl = attn_impl
        self.ref, self.cfg, self.model, self.lm_head = ref, ref.cfg, ref.model, ref.lm_head
        self.Lmax, self.dev, self.dtype = Lmax, ref.device, ref.dtype
        self.pool = torch.cuda.graph_pool_handle()
        self.start = torch.zeros((), dtype=torch.int64, device=self.dev)
        self.ar = torch.arange(Lmax, device=self.dev)

    def new_kv(self, layers=None):
        c = self.cfg
        layers = range(c.num_hidden_layers) if layers is None else layers
        return {i: (torch.zeros(1, c.num_key_value_heads, self.Lmax, c.head_dim, device=self.dev, dtype=self.dtype),
                    torch.zeros(1, c.num_key_value_heads, self.Lmax, c.head_dim, device=self.dev, dtype=self.dtype)) for i in layers}

    def forward(self, ids, kv, skip=frozenset(), r=None, parts=ALL, logits="last", sparse=None):
        """sparse=None | (mode, K): T==1 draft-step only. mode in slwin (sliding layers gather last W slots),
        fixed (+global layers gather K causal-valid slots, index chosen WITHOUT scoring: pure gather+attend cost),
        oracle (+global layers score q.K over all slots, topk K, gather: cost of the oracle selector)."""
        cfg, T = self.cfg, ids.shape[1]
        D, H, Hkv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
        G, W = H // Hkv, cfg.sliding_window
        qpos = self.start + torch.arange(T, device=self.dev)
        x = self.model.embed_tokens(ids)
        cos, sin = self.model.rotary_emb(x, qpos[None])
        mfull = (self.ar[None, :] <= qpos[:, None])
        msl = mfull & (self.ar[None, :] > qpos[:, None] - W)
        mfull, msl = mfull.repeat(G, 1), msl.repeat(G, 1)
        for i, layer in enumerate(self.model.layers):
            if i in skip:
                continue
            h = layer.input_layernorm(x)
            a = 0
            if "attn" in parts:
                at = layer.self_attn
                sliding = cfg.layer_types[i] == "sliding_attention"
                q = at.q_proj(h).view(1, T, H, D).transpose(1, 2)
                k = at.k_proj(h).view(1, T, Hkv, D).transpose(1, 2)
                v = at.v_proj(h).view(1, T, Hkv, D).transpose(1, 2)
                if sliding or at.force_rope:
                    q, k = apply_rotary_pos_emb(q, k, cos, sin)
                kc, vc = kv[i]
                kc.index_copy_(2, qpos, k)
                vc.index_copy_(2, qpos, v)
                qg = q.reshape(1, Hkv, G, T, D).reshape(1, Hkv, G * T, D)
                mk = msl if sliding else mfull
                if sparse is not None and T == 1 and (sliding or sparse[0] != "slwin"):
                    smode, KK = sparse
                    Wn = min(cfg.sliding_window, self.Lmax)
                    if sliding:
                        idx = (self.start - Wn + 1 + self.ar[:Wn]).clamp(min=0)[None, None, :].expand(1, Hkv, Wn)
                    else:
                        wi = (self.start - 127 + self.ar[:128]).clamp(min=0)
                        if smode == "fixed":
                            core = (self.ar[:KK].float() * (self.start.float() / KK)).long()
                            idx = torch.cat([self.ar[:4], core, wi])[None, None, :].expand(1, Hkv, -1)
                        else:
                            qm = qg.reshape(1, Hkv, G, D).float().mean(2)
                            scs = torch.bmm(qm, kc[0].transpose(1, 2).float()).masked_fill(~(self.ar[None, :] <= self.start), float("-inf"))
                            top = scs.topk(KK, dim=-1).indices[None]
                            idx = torch.cat([self.ar[:4][None, None].expand(1, Hkv, 4), top, wi[None, None].expand(1, Hkv, 128)], -1)
                    ix = idx[..., None].expand(1, Hkv, idx.shape[-1], D)
                    kg, vg = torch.gather(kc, 2, ix), torch.gather(vc, 2, ix)
                    sc = torch.bmm(qg[0], kg[0].transpose(1, 2), out_dtype=torch.float32) * (D ** -0.5)
                    p_ = torch.softmax(sc, dim=-1).to(vg.dtype)
                    o = torch.bmm(p_, vg[0])[None]
                elif self.attn_impl == "sdpa_mask":
                    o = F.scaled_dot_product_attention(qg, kc, vc, attn_mask=mk)
                else:  # bmm: scores in fp32 (out_dtype), fp32 softmax, bf16 P@V  -- memory-bound, no mask-SDPA inefficiency
                    sc = torch.bmm(qg[0], kc[0].transpose(1, 2), out_dtype=torch.float32) * (D ** -0.5)
                    sc = sc.masked_fill(~mk[None], float("-inf"))
                    p_ = torch.softmax(sc, dim=-1).to(vc.dtype)
                    o = torch.bmm(p_, vc[0])[None]
                o = o.reshape(1, Hkv, G, T, D).reshape(1, H, T, D).transpose(1, 2).reshape(1, T, H * D)
                a = at.o_proj(o)
            m = 0
            mlp = layer.mlp
            if not hasattr(mlp, "gate"):
                if "experts" in parts:
                    m = mlp(h)
            else:
                if "router" in parts:
                    hf = h.reshape(-1, h.shape[-1])
                    _, sc, ix = mlp.gate(hf)
                    if r is not None:
                        sc, ix = sc[:, :r], ix[:, :r]
                    if "experts" in parts:
                        m = mlp.experts(hf, ix, sc).reshape(1, T, -1)
                    else:
                        m = sc.sum() * 0  # keep router work live
            x = x + a + m
        if logits == "none" or "lm" not in parts:
            return x
        if logits == "last":
            x = x[:, -1:]
        return self.lm_head(self.model.norm(x)) * cfg.logit_scale

    # ---- graph helpers -------------------------------------------------------------------------
    def capture(self, fn, warmup=3):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn()
        return g, out

    def make_step(self, T, kv, skip=frozenset(), r=None, parts=ALL, logits="last", argmax=True, sparse=None):
        """Returns a Step with .ids (static input), .run() = replay, .out (tensor), .tok (argmax ids)."""
        ids = torch.zeros(1, T, dtype=torch.long, device=self.dev)

        def fn():
            lg = self.forward(ids, kv, skip, r, parts, logits, sparse)
            return lg.argmax(-1) if argmax and lg.shape[-1] == self.cfg.vocab_size else lg

        g, out = self.capture(fn)
        return Step(self, g, ids, out, fn)

    def eager_prefill(self, ids_list, kv, skip=frozenset(), r=None, chunk=1024, begin=0):
        for s in range(0, len(ids_list), chunk):
            c = torch.tensor([ids_list[s:s + chunk]], device=self.dev)
            self.start.fill_(begin + s)
            with torch.no_grad():
                self.forward(c, kv, skip, r, ALL, "none")
        torch.cuda.synchronize()


class Step:
    def __init__(self, gn, g, ids, out, fn):
        self.gn, self.g, self.ids, self.out, self.fn = gn, g, ids, out, fn

    def set(self, toks, start):
        self.ids.copy_(torch.tensor([toks], device=self.gn.dev))
        self.gn.start.fill_(start)

    def run(self):
        self.g.replay()
        return self.out


def time_graph(step, n=30, warm=5):
    for _ in range(warm):
        step.g.replay()
    torch.cuda.synchronize()
    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n)]
    for a, b in evs:
        a.record(); step.g.replay(); b.record()
    torch.cuda.synchronize()
    ts = [a.elapsed_time(b) for a, b in evs]
    return st.median(ts), min(ts), max(ts)


def time_eager(fn, n=8, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1e3)
    return st.median(ts)
