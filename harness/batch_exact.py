"""Batched exact-mode executor: B sequences (own KV cache, own position), T rows each, one CUDA graph per (B, T).
Linears / MoE / norms / lm_head run on the flattened [B*T] rows (weights streamed once for all rows); attention runs per sequence
(its own KV, own start) with the same exact kernels as FastNorth(exact=True). Built on a FastNorth(exact=True) instance (weights, merged qkv)."""
import torch
import fast_kernels as fk
import fast_kernels_exact as fx
import fast_kernels_batch as fkb
import fast_kernels_battn as fba

# True: one batched attention launch per layer for all sequences (fast_kernels_battn); False: one launch per sequence (H18/H19);
# "auto": batched iff B >= 2 (abattn_ba1: 0.96x at B=1, 1.34x / 2.46x / 3.10x at B=4/16/32, T=1). Bitwise-identical per row.
BATCH_ATTN = "auto"


# True: attention branch (qkv, attention, o_proj) on a side stream, concurrent with router + MoE (North's parallel block); joins at the
# residual. Same kernels, same values (branch perf/exact-decode); outputs go to static buffers so the graph has no cross-stream allocator reuse.
STREAMS = False
STREAM_PRIO = -1   # CUDA priority of the attention stream (lower = higher). Scheduling only; x4prio: verify blocks 5-7% faster at -1


# True: rows flagged invalid by Stepb.set(valid=...) skip expert work in the gathered MoE (mode 3 / auto above GATHER_MIN_NT).
MASK_PAD = True


def _battn(B):
    return BATCH_ATTN is True or (BATCH_ATTN == "auto" and B >= 2)


class Stepb:
    def __init__(self, ex, g, ids, starts, out, B, T):
        self.ex, self.g, self.ids, self.starts, self.out, self.B, self.T = ex, g, ids, starts, out, B, T
        self.valid = ex._state(B, T)["valid"]

    def set(self, toks, starts, valid=None):
        """valid: optional [B][T] bools; False rows (padding, finished sequences) produce ignored outputs and may skip expert work."""
        dev = self.ex.X.dev
        self.ids.copy_(torch.tensor(toks, device=dev)); self.starts.copy_(torch.tensor(starts, device=dev))
        if valid is None:
            self.valid.fill_(True)
        else:
            self.valid.copy_(torch.tensor(valid, device=dev, dtype=torch.bool).reshape(-1))

    def run(self):
        self.g.replay()
        return self.out


class BatchExact:
    def __init__(self, X, kvs_all=None):
        assert X.exact
        self.X = X
        self._st = {}
        self._pt = {}

    def _state(self, B, T):
        if (B, T) not in self._st:
            X, c = self.X, self.X.cfg
            k, N, dev = c.num_experts_per_tok, B * T, X.dev
            z = lambda *s: torch.zeros(*s, dtype=torch.bfloat16, device=dev)
            self._st[(B, T)] = dict(ws=X._x_state(T)["ws"], idx=torch.zeros(N, k, dtype=torch.int32, device=dev), sc=z(N, k),
                                    flag=torch.zeros(N * k, dtype=torch.int32, device=dev), hb=z(N * k, c.intermediate_size), ob=z(N * k, X.Dm),
                                    x2=z(N, X.Dm), h2=z(N, X.Dm), zero=z(N, X.Dm), ao=z(N, X.H * X.D),
                                    valid=torch.ones(N, dtype=torch.bool, device=dev), qkv=z(N, X.wqkv[0].shape[0]), a=z(N, X.Dm))
            self._st[(B, T)]["wsb"] = fba.AttnWSB(self._st[(B, T)]["ws"], B, dev)
        return self._st[(B, T)]

    def ptrs(self, kvs):
        """[nl, 2, B] int64 device table of K/V cache data pointers for this list of sequences (built eagerly, cached; never inside a capture)."""
        key = tuple(kv[0][0].data_ptr() for kv in kvs)
        if key not in self._pt:
            nl = len(kvs[0])
            self._pt[key] = torch.tensor([[[kvs[s][i][w].data_ptr() for s in range(len(kvs))] for w in (0, 1)] for i in range(nl)],
                                         dtype=torch.int64, device=self.X.dev)
        return self._pt[key]

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
        ba = _battn(B)
        pt = self.ptrs(kvs) if ba else None
        main = torch.cuda.current_stream()
        side = None
        if STREAMS:
            if not hasattr(X, "_side_stream"):
                X._side_stream = torch.cuda.Stream(device=X.dev, priority=STREAM_PRIO)
            side = X._side_stream
        import contextlib
        for i, layer in enumerate(layers):
            sliding = cfg.layer_types[i] == "sliding_attention"
            at = layer.self_attn
            if side is not None:
                side.wait_stream(main)
            ctx = torch.cuda.stream(side) if side is not None else contextlib.nullcontext()
            ctx.__enter__()
            qkv = fkb.lin_bt(h, X.wqkv[i], b["qkv"] if side is not None else None)
            rope = bool(sliding or at.force_rope)
            if ba:
                fba.attn_batched(b["wsb"], qkv, cos, sin, starts, pt[i, 0], pt[i, 1], X.Lmax, rope, bool(sliding), W, b["ao"])
            else:
                for s in range(B):
                    kc, vc = kvs[s][i]
                    fkb.attn_seq(b["ws"], qkv[s * T:(s + 1) * T], cos[s * T:(s + 1) * T], sin[s * T:(s + 1) * T], starts[s:s + 1], kc, vc, X.Lmax,
                                 rope, bool(sliding), W, b["ao"][s * T:(s + 1) * T])
            a = fkb.lin_bt(b["ao"], at.o_proj.weight, b["a"] if side is not None else None)
            ctx.__exit__(None, None, None)
            mlp = layer.mlp
            if not hasattr(mlp, "gate"):
                g_ = fkb.lin_bt(h, mlp.gate_proj.weight)
                u_ = fkb.lin_bt(h, mlp.up_proj.weight)
                m = fkb.lin_bt(mlp.act_fn(g_) * u_, mlp.down_proj.weight)
            else:
                lg = fkb.lin_bt(h, mlp.gate.weight)
                fk.router_topk(lg, b["idx"], b["sc"], k)
                m = fkb.moe_bt(h, b["idx"], b["sc"], mlp.experts.gate_up_proj, mlp.experts.down_proj, b["hb"], b["ob"], b["flag"], valid=b["valid"] if MASK_PAD else None)
            if side is not None:
                main.wait_stream(side)
            wn = layers[i + 1].input_layernorm.weight if i + 1 < X.nl else X.model.norm.weight
            fk.resid_norm(x, a, m, wn, b["x2"], b["h2"], layer.input_layernorm.variance_epsilon)
            x, h = b["x2"], b["h2"]
        lg = fkb.lin_bt(h, X.lm_head.weight)
        return lg if cfg.logit_scale == 1.0 else lg * cfg.logit_scale

    def make_step(self, B, T, kvs):
        X = self.X
        ids = torch.zeros(B, T, dtype=torch.long, device=X.dev)
        starts = torch.zeros(B, dtype=torch.long, device=X.dev)

        if _battn(B):
            self.ptrs(kvs); self._state(B, T)   # pointer table + workspaces allocated before capture
        def fn():
            return fx.argmax_rows(self.forward(ids, starts, kvs)).view(B, T)

        # warmup/trace write junk KV at a scratch position in the unused tail. starts STAYS there until .set(): replaying a graph whose inputs were
        # never set used to write junk KV at positions 0..T-1 (the prompt prefix) of every row; that is what broke plain B>1 == B=1 in bs1 group 0 (H18).
        starts.fill_(max(0, X.Lmax - 16))
        g, out = X._capture(fn)
        return Stepb(self, g, ids, starts, out, B, T)
