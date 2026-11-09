"""Batched exact-mode executor: B sequences (own KV cache, own position), T rows each, one CUDA graph per (B, T).
Linears / MoE / norms / lm_head run on the flattened [B*T] rows (weights streamed once for all rows); attention runs per sequence
(its own KV, own start) with the same exact kernels as FastNorth(exact=True). Built on a FastNorth(exact=True) instance (weights, merged qkv)."""
import torch
import fast_kernels as fk
import fast_kernels_exact as fx
import fast_kernels_batch as fkb


class Stepb:
    def __init__(self, ex, g, ids, starts, out, B, T):
        self.ex, self.g, self.ids, self.starts, self.out, self.B, self.T = ex, g, ids, starts, out, B, T

    def set(self, toks, starts):
        dev = self.ex.X.dev
        self.ids.copy_(torch.tensor(toks, device=dev)); self.starts.copy_(torch.tensor(starts, device=dev))

    def run(self):
        self.g.replay()
        return self.out


class BatchExact:
    def __init__(self, X, kvs_all=None):
        assert X.exact
        self.X = X
        self._st = {}

    def _state(self, B, T):
        if (B, T) not in self._st:
            X, c = self.X, self.X.cfg
            k, N, dev = c.num_experts_per_tok, B * T, X.dev
            z = lambda *s: torch.zeros(*s, dtype=torch.bfloat16, device=dev)
            self._st[(B, T)] = dict(ws=X._x_state(T)["ws"], idx=torch.zeros(N, k, dtype=torch.int32, device=dev), sc=z(N, k),
                                    flag=torch.zeros(N * k, dtype=torch.int32, device=dev), hb=z(N * k, c.intermediate_size), ob=z(N * k, X.Dm),
                                    x2=z(N, X.Dm), h2=z(N, X.Dm), zero=z(N, X.Dm), ao=z(N, X.H * X.D))
        return self._st[(B, T)]

    @torch.no_grad()
    def forward(self, ids, starts, kvs):
        """ids [B,T] long, starts [B] long (device), kvs: list of B kv dicts. Returns bf16 logits [B*T, V]."""
        X = self.X
        cfg = X.cfg
        B, T = ids.shape
        N = B * T
        W, Dm = cfg.sliding_window, X.Dm
        qpos = (starts[:, None] + torch.arange(T, device=X.dev)[None, :]).reshape(1, N)
        x = X.model.embed_tokens(ids).reshape(N, Dm).contiguous()
        cos, sin = X.model.rotary_emb(x.view(1, N, -1), qpos)
        cos, sin = cos.reshape(N, -1).contiguous(), sin.reshape(N, -1).contiguous()
        layers = X.model.layers
        b = self._state(B, T)
        k = cfg.num_experts_per_tok
        fk.resid_norm(x, b["zero"], b["zero"], layers[0].input_layernorm.weight, b["x2"], b["h2"], layers[0].input_layernorm.variance_epsilon)
        x, h = b["x2"].clone(), b["h2"].clone()
        D = X.D
        for i, layer in enumerate(layers):
            sliding = cfg.layer_types[i] == "sliding_attention"
            at = layer.self_attn
            qkv = fkb.lin_bt(h, X.wqkv[i])
            rope = bool(sliding or at.force_rope)
            for s in range(B):
                kc, vc = kvs[s][i]
                fkb.attn_seq(b["ws"], qkv[s * T:(s + 1) * T], cos[s * T:(s + 1) * T], sin[s * T:(s + 1) * T], starts[s:s + 1], kc, vc, X.Lmax,
                             rope, bool(sliding), W, b["ao"][s * T:(s + 1) * T])
            a = fkb.lin_bt(b["ao"], at.o_proj.weight)
            mlp = layer.mlp
            if not hasattr(mlp, "gate"):
                g_ = fkb.lin_bt(h, mlp.gate_proj.weight)
                u_ = fkb.lin_bt(h, mlp.up_proj.weight)
                m = fkb.lin_bt(mlp.act_fn(g_) * u_, mlp.down_proj.weight)
            else:
                lg = fkb.lin_bt(h, mlp.gate.weight)
                fk.router_topk(lg, b["idx"], b["sc"], k)
                m = fkb.moe_bt(h, b["idx"], b["sc"], mlp.experts.gate_up_proj, mlp.experts.down_proj, b["hb"], b["ob"], b["flag"])
            wn = layers[i + 1].input_layernorm.weight if i + 1 < X.nl else X.model.norm.weight
            fk.resid_norm(x, a, m, wn, b["x2"], b["h2"], layer.input_layernorm.variance_epsilon)
            x, h = b["x2"], b["h2"]
        lg = fkb.lin_bt(h, X.lm_head.weight)
        return lg if cfg.logit_scale == 1.0 else lg * cfg.logit_scale

    def make_step(self, B, T, kvs):
        X = self.X
        ids = torch.zeros(B, T, dtype=torch.long, device=X.dev)
        starts = torch.zeros(B, dtype=torch.long, device=X.dev)

        def fn():
            return fx.argmax_rows(self.forward(ids, starts, kvs)).view(B, T)

        # warmup/trace write junk KV at a scratch position in the unused tail. starts STAYS there until .set(): replaying a graph whose inputs were
        # never set used to write junk KV at positions 0..T-1 (the prompt prefix) of every row; that is what broke plain B>1 == B=1 in bs1 group 0 (H18).
        starts.fill_(max(0, X.Lmax - 16))
        g, out = X._capture(fn)
        return Stepb(self, g, ids, starts, out, B, T)
