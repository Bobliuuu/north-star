"""FastNorth: decode-specialised (T=1) executor for North, built ON TOP of GraphNorth (graph_north.py is untouched).
Options (each independently switchable so wins can be ablated):
  attn   : merged qkv GEMV + fused rope/KV-write kernel + split-K flash-decode over LIVE keys only (no padded Lmax,
           sliding layers read <= window keys) + combine kernel + o_proj           (~25 launches/layer -> 5)
  moe    : top-k expert GEMV (Triton, reads only the 8 selected experts, no sort/histc/gather/grouped_mm)
  router : fused top-k + sigmoid kernel on the bf16 router logits (1 launch instead of topk/sigmoid/cast)
  resid  : fused (x + a + sum(expert outs)) + next-layer RMSNorm (one launch)
T == 1: decode path. 2 <= T <= TBLK with logits='all' and all four options: multi-query VERIFY block (_forward_block: T-row cuBLAS
qkv/o_proj/router GEMMs, T-row split-K causal attention, union-of-experts grouped GEMV, T-row fused resid+norm).
skip / r / parts / sparse / prefill chunks (logits != 'all') fall back to GraphNorth.forward.
Weights are the HF module weights (no change); only a merged qkv copy (0.5 GB) is added when attn is on."""
import torch
import torch.nn.functional as F

from graph_north import GraphNorth, ALL
import fast_kernels as fk
import fast_kernels_blk as fb
import fast_kernels_exact as fx
import pdl_flag as pdl

OPTS = ("attn", "moe", "router", "resid")
XOPTS = frozenset()  # filled as exact-fast optimisations land
TBLK = 16  # largest verify block handled by the fast block path


class FastNorth(GraphNorth):
    def __init__(self, ref, Lmax, opts=OPTS, nsplit=32, kcfg=None, exact=False, xopts=None, xcfg=None):
        super().__init__(ref, Lmax, "bmm")
        self.xopts = XOPTS if xopts is None else frozenset(xopts)  # exact-fast optimisations (empty = original exact path)
        self.xcfg = dict(xcfg or {})  # per-option tile configs for the exact-fast options: attn / lin / moe
        self.exact = exact  # batch-invariant path: T=1 and every row of a T<=16 block use the SAME kernels (see fast_kernels_exact.py)
        self._x = {}
        self.opts = frozenset(opts)
        self.kcfg = kcfg
        c = self.cfg
        D, H, Hkv = c.head_dim, c.num_attention_heads, c.num_key_value_heads
        self.H, self.Hkv, self.D, self.G = H, Hkv, D, H // Hkv
        self.Dm = c.hidden_size
        self.ws = fk.AttnWS(Hkv, self.G, D, nsplit, self.dev)
        self.bufs = dict(
            idx=torch.zeros(c.num_experts_per_tok, dtype=torch.int32, device=self.dev),
            sc=torch.zeros(c.num_experts_per_tok, dtype=torch.bfloat16, device=self.dev),
            hb=torch.zeros(c.num_experts_per_tok, c.intermediate_size, dtype=torch.bfloat16, device=self.dev),
            ob=torch.zeros(c.num_experts_per_tok, self.Dm, dtype=torch.bfloat16, device=self.dev),
            x2=torch.zeros(self.Dm, dtype=torch.bfloat16, device=self.dev),
            h2=torch.zeros(self.Dm, dtype=torch.bfloat16, device=self.dev),
            zero=torch.zeros(1, self.Dm, dtype=torch.bfloat16, device=self.dev),
        )
        self.wqkv = {}
        if "attn" in self.opts:
            for i, layer in enumerate(self.model.layers):
                at = layer.self_attn
                self.wqkv[i] = torch.cat([at.q_proj.weight, at.k_proj.weight, at.v_proj.weight], 0).contiguous()
        self.nl = c.num_hidden_layers
        self._blk = {}
        self.kcfg_blk = None

    def _blk_state(self, T):
        if T not in self._blk:
            c = self.cfg
            k = c.num_experts_per_tok
            z = lambda *s: torch.zeros(*s, dtype=torch.bfloat16, device=self.dev)
            self._blk[T] = dict(ws=fb.AttnWSBlk(self.Hkv, self.G, self.D, self.ws.nsplit, T, self.dev),
                                idx=torch.zeros(T, k, dtype=torch.int32, device=self.dev), sc=z(T, k),
                                hb=z(T * k, c.intermediate_size), ob=z(T * k, self.Dm), x2=z(T, self.Dm), h2=z(T, self.Dm))
        return self._blk[T]

    def _forward_block(self, ids, kv):
        cfg, T = self.cfg, ids.shape[1]
        W = cfg.sliding_window
        qpos = self.start + torch.arange(T, device=self.dev)
        x = self.model.embed_tokens(ids).reshape(T, self.Dm)
        cos, sin = self.model.rotary_emb(x.view(1, T, -1), qpos[None])
        cos, sin = cos.reshape(T, -1).contiguous(), sin.reshape(T, -1).contiguous()
        layers = self.model.layers
        h = layers[0].input_layernorm(x).contiguous()
        b = self._blk_state(T)
        k = cfg.num_experts_per_tok
        for i, layer in enumerate(layers):
            sliding = cfg.layer_types[i] == "sliding_attention"
            at = layer.self_attn
            kc, vc = kv[i]
            qkv = F.linear(h, self.wqkv[i])
            ao = fb.verify_attention(b["ws"], qkv, cos, sin, self.start, kc, vc, self.Lmax, bool(sliding or at.force_rope), bool(sliding), W)
            a = F.linear(ao, at.o_proj.weight)
            mlp = layer.mlp
            if not hasattr(mlp, "gate"):
                m = mlp(h).reshape(T, self.Dm).contiguous()
            else:
                lg = F.linear(h, mlp.gate.weight)
                fk.router_topk(lg, b["idx"], b["sc"], k)
                m = fb.moe_gemv_blk(h, b["idx"], b["sc"], mlp.experts.gate_up_proj, mlp.experts.down_proj, b["hb"], b["ob"], self.kcfg_blk)
            wn = layers[i + 1].input_layernorm.weight if i + 1 < self.nl else self.model.norm.weight
            fk.resid_norm(x, a, m, wn, b["x2"], b["h2"], layer.input_layernorm.variance_epsilon)
            x, h = b["x2"], b["h2"]
        return self.lm_head(h.view(1, T, -1)) * cfg.logit_scale

    def _x_state(self, T):
        if T not in self._x:
            c = self.cfg
            k = c.num_experts_per_tok
            z = lambda *s: torch.zeros(*s, dtype=torch.bfloat16, device=self.dev)
            self._x[T] = dict(ws=fx.AttnWSX(self.Hkv, self.G, self.D, self.Lmax, T, self.dev), idx=torch.zeros(T, k, dtype=torch.int32, device=self.dev),
                              sc=z(T, k), hb=z(T * k, c.intermediate_size), ob=z(T * k, self.Dm), x2=z(T, self.Dm), h2=z(T, self.Dm), zero=z(T, self.Dm))
        return self._x[T]

    def _forward_exact(self, ids, kv):
        """Row-invariant forward for 1 <= T <= TBLK: returns bf16 logits [1,T,V]. Same code for plain (T=1) and verify blocks."""
        cfg, T = self.cfg, ids.shape[1]
        W = cfg.sliding_window
        xo = self.xopts
        pdl.ON = "pdl" in xo
        fx.LIN_CFG.clear()
        if "lin" in xo:
            fx.LIN_CFG.update(self.xcfg.get("lin", {}))
        ak = dict(rowgroup=True, **self.xcfg.get("attn", {})) if "rg" in xo else {}
        kmoe = self.xcfg.get("moe") if "moe" in xo else self.kcfg_blk
        qpos = self.start + torch.arange(T, device=self.dev)
        x = self.model.embed_tokens(ids).reshape(T, self.Dm).contiguous()
        cos, sin = self.model.rotary_emb(x.view(1, T, -1), qpos[None])
        cos, sin = cos.reshape(T, -1).contiguous(), sin.reshape(T, -1).contiguous()
        layers = self.model.layers
        b = self._x_state(T)
        k = cfg.num_experts_per_tok
        h = b["h2"]
        fk.resid_norm(x, b["zero"], b["zero"], layers[0].input_layernorm.weight, b["x2"], h, layers[0].input_layernorm.variance_epsilon)
        x, h = b["x2"], b["h2"]
        h = h.clone(); x = x.clone()  # inputs of layer 0 (the buffers are rewritten by the per-layer fused norm)
        for i, layer in enumerate(layers):
            sliding = cfg.layer_types[i] == "sliding_attention"
            at = layer.self_attn
            kc, vc = kv[i]
            qkv = fx.lin(h, self.wqkv[i])
            ao = fx.exact_attention(b["ws"], qkv, cos, sin, self.start, kc, vc, self.Lmax, bool(sliding or at.force_rope), bool(sliding), W, fb._rope_write_blk, **ak)
            a = fx.lin(ao, at.o_proj.weight)
            mlp = layer.mlp
            if not hasattr(mlp, "gate"):
                g_ = fx.lin(h, mlp.gate_proj.weight)
                u_ = fx.lin(h, mlp.up_proj.weight)
                m = fx.lin(mlp.act_fn(g_) * u_, mlp.down_proj.weight)
            else:
                lg = fx.lin(h, mlp.gate.weight)
                fk.router_topk(lg, b["idx"], b["sc"], k)
                m = fb.moe_gemv_blk(h, b["idx"], b["sc"], mlp.experts.gate_up_proj, mlp.experts.down_proj, b["hb"], b["ob"], kmoe)
            wn = layers[i + 1].input_layernorm.weight if i + 1 < self.nl else self.model.norm.weight
            fk.resid_norm(x, a, m, wn, b["x2"], b["h2"], layer.input_layernorm.variance_epsilon)
            x, h = b["x2"], b["h2"]
        lg = fx.lin(h, self.lm_head.weight).view(1, T, -1)
        return lg if cfg.logit_scale == 1.0 else lg * cfg.logit_scale

    def make_step(self, T, kv, skip=frozenset(), r=None, parts=ALL, logits="last", argmax=True, sparse=None):
        if not (self.exact and 1 <= T <= TBLK and not skip and r is None and parts == ALL and sparse is None and logits in ("all", "last")):
            return super().make_step(T, kv, skip, r, parts, logits, argmax, sparse)
        from graph_north import Step
        ids = torch.zeros(1, T, dtype=torch.long, device=self.dev)

        def fn():
            lg = self._forward_exact(ids, kv)
            return fx.argmax_rows(lg[0]).view(1, T) if argmax else lg

        g, out = self.capture(fn)
        return Step(self, g, ids, out, fn)

    def forward(self, ids, kv, skip=frozenset(), r=None, parts=ALL, logits="last", sparse=None):
        if self.exact and 1 <= ids.shape[1] <= TBLK and logits in ("all", "last") and not skip and r is None and parts == ALL and sparse is None:
            return self._forward_exact(ids, kv)
        if 1 < ids.shape[1] <= TBLK and logits == "all" and not skip and r is None and parts == ALL and sparse is None and self.opts >= frozenset(OPTS):
            return self._forward_block(ids, kv)
        if ids.shape[1] != 1 or skip or r is not None or parts != ALL or logits != "last" or sparse is not None:
            return super().forward(ids, kv, skip, r, parts, logits, sparse)
        cfg, o = self.cfg, self.opts
        W = cfg.sliding_window
        qpos = self.start + torch.arange(1, device=self.dev)
        x = self.model.embed_tokens(ids).reshape(self.Dm)
        cos, sin = self.model.rotary_emb(x.view(1, 1, -1), qpos[None])
        cos, sin = cos.reshape(-1), sin.reshape(-1)
        layers = self.model.layers
        h = layers[0].input_layernorm(x.view(1, -1)).reshape(self.Dm)
        b = self.bufs
        for i, layer in enumerate(layers):
            sliding = cfg.layer_types[i] == "sliding_attention"
            at = layer.self_attn
            kc, vc = kv[i]
            if "attn" in o:
                qkv = F.linear(h, self.wqkv[i])
                rope = bool(sliding or at.force_rope)
                ao = fk.decode_attention(self.ws, qkv, cos, sin, self.start, kc, vc, self.Lmax, rope, bool(sliding), W)
                a = F.linear(ao, at.o_proj.weight)
            else:
                a = self._attn_torch(i, layer, h.view(1, 1, -1), cos, sin, kv, qpos).reshape(self.Dm)
            mlp = layer.mlp
            if not hasattr(mlp, "gate"):
                m = mlp(h.view(1, -1)).reshape(1, self.Dm)
            else:
                lg = F.linear(h.view(1, -1), mlp.gate.weight)
                if "router" in o:
                    fk.router_topk(lg, b["idx"], b["sc"], cfg.num_experts_per_tok)
                    idx, sc = b["idx"], b["sc"]
                else:
                    sc_, ix = torch.topk(lg, cfg.num_experts_per_tok, dim=-1)
                    sc_ = torch.sigmoid(sc_)
                    idx, sc = ix.reshape(-1), sc_.reshape(-1).to(h.dtype)
                if "moe" in o:
                    m = fk.moe_gemv(h, idx, sc, mlp.experts.gate_up_proj, mlp.experts.down_proj, b["hb"], b["ob"], self.kcfg)
                    if "resid" not in o:
                        m = m.sum(0, keepdim=True)
                else:
                    m = mlp.experts(h.view(1, -1), idx.view(1, -1), sc.view(1, -1)).reshape(1, self.Dm)
            wn = layers[i + 1].input_layernorm.weight if i + 1 < self.nl else self.model.norm.weight
            if "resid" in o:
                fk.resid_norm(x, a, m.contiguous(), wn, b["x2"], b["h2"], layer.input_layernorm.variance_epsilon)
                # x/h live in static buffers; the kernel is elementwise-in-place safe for x (each lane loads x[r] before storing x2[r])
                x, h = b["x2"], b["h2"]
            else:
                x = x + a + m.reshape(self.Dm)
                nrm = layers[i + 1].input_layernorm if i + 1 < self.nl else self.model.norm
                h = nrm(x.view(1, -1)).reshape(self.Dm)
        return self.lm_head(h.view(1, 1, -1)) * cfg.logit_scale

    def _attn_torch(self, i, layer, h, cos, sin, kv, qpos):
        """Original GraphNorth attention for T=1 (used when 'attn' option is off)."""
        from transformers.models.cohere2_moe.modeling_cohere2_moe import apply_rotary_pos_emb
        cfg = self.cfg
        D, H, Hkv, G = self.D, self.H, self.Hkv, self.G
        sliding = cfg.layer_types[i] == "sliding_attention"
        at = layer.self_attn
        q = at.q_proj(h).view(1, 1, H, D).transpose(1, 2)
        k = at.k_proj(h).view(1, 1, Hkv, D).transpose(1, 2)
        v = at.v_proj(h).view(1, 1, Hkv, D).transpose(1, 2)
        if sliding or at.force_rope:
            q, k = apply_rotary_pos_emb(q, k, cos.view(1, 1, -1), sin.view(1, 1, -1))
        kc, vc = kv[i]
        kc.index_copy_(2, qpos, k)
        vc.index_copy_(2, qpos, v)
        qg = q.reshape(1, Hkv, G, 1, D).reshape(1, Hkv, G, D)
        mfull = (self.ar[None, :] <= qpos[:, None])
        msl = mfull & (self.ar[None, :] > qpos[:, None] - cfg.sliding_window)
        mk = (msl if sliding else mfull).repeat(G, 1)
        sc = torch.bmm(qg[0], kc[0].transpose(1, 2), out_dtype=torch.float32) * (D ** -0.5)
        sc = sc.masked_fill(~mk[None], float("-inf"))
        p_ = torch.softmax(sc, dim=-1).to(vc.dtype)
        o = torch.bmm(p_, vc[0])[None]
        o = o.reshape(1, H * D)
        return at.o_proj(o)
