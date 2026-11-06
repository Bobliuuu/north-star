"""Long-context / batched graphed North forward for the `longctx-batch` study (timing-oriented, reuses RefNorth/HF weights).

Differences from graph_north.GraphNorth (which is left untouched):
  * batch B (lock-step: all sequences share the same position `start`, each has its own KV and its own prompt);
  * sliding layers read only a window slice of the linear KV buffer (static python offset `wlo`), NOT all Lmax slots
    (sl='bmm' / 'tri'); sl='padded' reproduces the old over-reading executor for comparison;
  * global layers: gl='bmm' (padded dense, as before) | 'tri' (fused Triton split-K flash-decode, reads only valid slots) |
    ('gather', kind, K) (torch.gather copy + bmm, as H8) | ('tri_idx', kind, K) (Triton split-K over a gathered index, NO K/V copy);
    kind 'fixed' = index built without scoring (cost of attending a precomputed/cached selection),
    kind 'oracle' = q.K scan over all slots + topk (cost of the selector included). Sparse modes: B==1, T==1 only.
  * eager SDPA chunked prefill with python-int positions (memory-safe up to 128K).
All numbers produced with this module are TIMING studies; correctness is checked by tri-vs-bmm logit agreement in the runner.
"""
import statistics as st
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers.models.cohere2_moe.modeling_cohere2_moe import apply_rotary_pos_emb

ALL = frozenset({"attn", "router", "experts", "lm"})
NEG = -1.0e30


@triton.jit
def _partial_kernel(Q, K, V, IDX, START, PO, PM, PL, skb, skh, skn, svb, svh, svn, N, scale,
                    HKV: tl.constexpr, G: tl.constexpr, T: tl.constexpr, RP: tl.constexpr, D: tl.constexpr,
                    BN: tl.constexpr, ITERS: tl.constexpr, WIN: tl.constexpr, GATHER: tl.constexpr, NS: tl.constexpr):
    s = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    start = tl.load(START)
    rows = tl.arange(0, RP)
    d = tl.arange(0, D)
    qpos = tl.where(rows < G * T, start + rows % T, -1)
    q = tl.load(Q + ((b * HKV + h) * RP + rows[:, None]) * D + d[None, :])
    if WIN > 0:
        base = tl.maximum(start - WIN + 1, 0)
    else:
        base = 0
    m = tl.full([RP], NEG_C, tl.float32)
    l = tl.zeros([RP], tl.float32)
    acc = tl.zeros([RP, D], tl.float32)
    for it in tl.static_range(ITERS):
        blk = (s * ITERS + it) * BN + tl.arange(0, BN)
        if GATHER:
            vb = blk < N
            pos = tl.load(IDX + (b * HKV + h) * N + blk, mask=vb, other=0).to(tl.int32)
        else:
            pos = base + blk
            vb = pos <= start + T - 1
        mask = vb[None, :] & (pos[None, :] <= qpos[:, None]) & (qpos[:, None] >= 0)
        if WIN > 0:
            mask = mask & (pos[None, :] > qpos[:, None] - WIN)
        k = tl.load(K + b * skb + h * skh + pos[:, None] * skn + d[None, :], mask=vb[:, None], other=0.0)
        sc = tl.dot(q, tl.trans(k)) * scale
        sc = tl.where(mask, sc, NEG_C)
        m_new = tl.maximum(m, tl.max(sc, 1))
        p = tl.where(mask, tl.exp(sc - m_new[:, None]), 0.0)
        alpha = tl.exp(m - m_new)
        l = l * alpha + tl.sum(p, 1)
        v = tl.load(V + b * svb + h * svh + pos[:, None] * svn + d[None, :], mask=vb[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new
    o = ((b * HKV + h) * NS + s)
    tl.store(PO + (o * RP + rows[:, None]) * D + d[None, :], acc)
    tl.store(PM + o * RP + rows, m)
    tl.store(PL + o * RP + rows, l)


@triton.jit
def _combine_kernel(PO, PM, PL, O, HKV: tl.constexpr, RP: tl.constexpr, D: tl.constexpr, NS: tl.constexpr):
    h = tl.program_id(0)
    b = tl.program_id(1)
    rows = tl.arange(0, RP)
    d = tl.arange(0, D)
    m = tl.full([RP], NEG_C, tl.float32)
    l = tl.zeros([RP], tl.float32)
    acc = tl.zeros([RP, D], tl.float32)
    for s in range(NS):
        o = ((b * HKV + h) * NS + s)
        ms = tl.load(PM + o * RP + rows)
        ls = tl.load(PL + o * RP + rows)
        a = tl.load(PO + (o * RP + rows[:, None]) * D + d[None, :])
        m_new = tl.maximum(m, ms)
        al = tl.exp(m - m_new)
        be = tl.exp(ms - m_new)
        acc = acc * al[:, None] + a * be[:, None]
        l = l * al + ls * be
        m = m_new
    out = acc / tl.maximum(l, 1e-20)[:, None]
    tl.store(O + ((b * HKV + h) * RP + rows[:, None]) * D + d[None, :], out.to(O.dtype.element_ty))


NEG_C = tl.constexpr(NEG)


def tri_attn(qg, kc, vc, start, G, T, win, idx=None, bn=128, iters=None):
    """qg [B,Hkv,G*T,D] bf16 -> out [B,Hkv,G*T,D]. Valid keys: pos<=start+t (causal) and pos>start+t-win if win>0.
    idx: None (contiguous: whole [0,Lmax) if win==0 else the window) or int tensor [B,Hkv,N] of slot indices (gather)."""
    B, Hkv, R, D = qg.shape
    RP = max(16, triton.next_power_of_2(R))
    q = torch.zeros(B, Hkv, RP, D, dtype=qg.dtype, device=qg.device)
    q[:, :, :R] = qg
    gather = idx is not None
    if gather:
        N = idx.shape[-1]
        span = N
        idx = idx.contiguous()
        it = 1 if iters is None else iters
    else:
        N = 0
        span = (win + T - 1) if win > 0 else kc.shape[2]
        it = (2 if win > 0 else 4) if iters is None else iters
    NS = triton.cdiv(span, bn * it)
    po = torch.empty(B, Hkv, NS, RP, D, dtype=torch.float32, device=qg.device)
    pm = torch.empty(B, Hkv, NS, RP, dtype=torch.float32, device=qg.device)
    pl = torch.empty_like(pm)
    _partial_kernel[(NS, Hkv, B)](q, kc, vc, idx if gather else q, start, po, pm, pl,
                                  kc.stride(0), kc.stride(1), kc.stride(2), vc.stride(0), vc.stride(1), vc.stride(2),
                                  N, D ** -0.5, HKV=Hkv, G=G, T=T, RP=RP, D=D, BN=bn, ITERS=it, WIN=win, GATHER=gather, NS=NS,
                                  num_warps=4, num_stages=2)
    out = torch.empty(B, Hkv, RP, D, dtype=qg.dtype, device=qg.device)
    _combine_kernel[(Hkv, B)](po, pm, pl, out, HKV=Hkv, RP=RP, D=D, NS=NS, num_warps=4)
    return out[:, :, :R]


class LCNorth:
    def __init__(self, ref, Lmax):
        self.ref, self.cfg, self.model, self.lm_head = ref, ref.cfg, ref.model, ref.lm_head
        self.Lmax, self.dev, self.dtype = Lmax, ref.device, ref.dtype
        self.start = torch.zeros((), dtype=torch.int64, device=self.dev)
        self.ar = torch.arange(Lmax, device=self.dev)
        self.wlo = 0  # static python offset of the sliding-window slice (set to ctx - W - 16 by the caller)

    def new_kv(self, B):
        c = self.cfg
        mk = lambda: torch.zeros(B, c.num_key_value_heads, self.Lmax, c.head_dim, device=self.dev, dtype=self.dtype)
        return {i: (mk(), mk()) for i in range(c.num_hidden_layers)}

    # ---------------------------------------------------------------- decode / verify forward (graph friendly)
    def forward(self, ids, kv, sl="bmm", gl="bmm", r=None, parts=ALL, logits="last"):
        cfg = self.cfg
        B, T = ids.shape
        D, H, Hkv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
        G, W = H // Hkv, cfg.sliding_window
        qpos = self.start + torch.arange(T, device=self.dev)
        x = self.model.embed_tokens(ids)
        cos, sin = self.model.rotary_emb(x, qpos[None])
        mfull = (self.ar[None, :] <= qpos[:, None])
        Wr = min(W + 32, self.Lmax - self.wlo)
        kw_ = self.wlo + self.ar[:Wr]
        msw = (kw_[None, :] <= qpos[:, None]) & (kw_[None, :] > qpos[:, None] - W)
        mpad_sl = mfull & (self.ar[None, :] > qpos[:, None] - W)
        mfull, msw, mpad_sl = mfull.repeat(G, 1), msw.repeat(G, 1), mpad_sl.repeat(G, 1)
        for i, layer in enumerate(self.model.layers):
            h = layer.input_layernorm(x)
            a = 0
            if "attn" in parts:
                at = layer.self_attn
                sliding = cfg.layer_types[i] == "sliding_attention"
                q = at.q_proj(h).view(B, T, H, D).transpose(1, 2)
                k = at.k_proj(h).view(B, T, Hkv, D).transpose(1, 2)
                v = at.v_proj(h).view(B, T, Hkv, D).transpose(1, 2)
                if sliding or at.force_rope:
                    q, k = apply_rotary_pos_emb(q, k, cos, sin)
                kc, vc = kv[i]
                kc.index_copy_(2, qpos, k)
                vc.index_copy_(2, qpos, v)
                qg = q.reshape(B, Hkv, G, T, D).reshape(B, Hkv, G * T, D)
                scale = D ** -0.5
                mode = sl if sliding else gl
                if mode == "tri":
                    o = tri_attn(qg, kc, vc, self.start, G, T, W if sliding else 0)
                elif isinstance(mode, tuple) and mode[0] in ("gather", "tri_idx") and not sliding:
                    assert B == 1 and T == 1
                    impl, kind, KK = mode
                    wi = (self.start - 127 + self.ar[:128]).clamp(min=0)
                    if kind == "fixed":
                        core = (self.ar[:KK].float() * (self.start.float() / KK)).long()
                        idx = torch.cat([self.ar[:4], core, wi])[None, None, :].expand(1, Hkv, -1)
                    else:
                        qm = qg.reshape(1, Hkv, G, D).float().mean(2)
                        scs = torch.bmm(qm[0][:, None, :], kc[0].transpose(1, 2).float())[:, 0].masked_fill(
                            ~(self.ar[None, :] <= self.start), float("-inf"))
                        top = scs.topk(KK, dim=-1).indices[None]
                        idx = torch.cat([self.ar[:4][None, None].expand(1, Hkv, 4), top, wi[None, None].expand(1, Hkv, 128)], -1)
                    if impl == "tri_idx":
                        o = tri_attn(qg, kc, vc, self.start, G, T, 0, idx=idx)
                    else:
                        ix = idx[..., None].expand(1, Hkv, idx.shape[-1], D)
                        kg, vg = torch.gather(kc, 2, ix), torch.gather(vc, 2, ix)
                        sc = torch.bmm(qg[0], kg[0].transpose(1, 2), out_dtype=torch.float32) * scale
                        o = torch.bmm(torch.softmax(sc, -1).to(vg.dtype), vg[0])[None]
                else:  # bmm: window slice for sliding ('bmm'), whole Lmax for global / sliding 'padded'
                    if sliding and mode == "bmm":
                        ks, vs, mk = kc[:, :, self.wlo:self.wlo + Wr], vc[:, :, self.wlo:self.wlo + Wr], msw
                    else:
                        ks, vs, mk = kc, vc, (mpad_sl if sliding else mfull)
                    L_ = ks.shape[2]
                    sc = torch.bmm(qg.reshape(B * Hkv, G * T, D), ks.reshape(B * Hkv, L_, D).transpose(1, 2),
                                   out_dtype=torch.float32) * scale
                    sc = sc.masked_fill(~mk[None], float("-inf"))
                    p_ = torch.softmax(sc, dim=-1).to(vs.dtype)
                    o = torch.bmm(p_, vs.reshape(B * Hkv, L_, D)).view(B, Hkv, G * T, D)
                o = o.reshape(B, Hkv, G, T, D).reshape(B, H, T, D).transpose(1, 2).reshape(B, T, H * D)
                a = at.o_proj(o)
            m = 0
            mlp = layer.mlp
            if not hasattr(mlp, "gate"):
                if "experts" in parts:
                    m = mlp(h)
            elif "router" in parts:
                hf = h.reshape(-1, h.shape[-1])
                _, sc_, ix_ = mlp.gate(hf)
                if r is not None:
                    sc_, ix_ = sc_[:, :r], ix_[:, :r]
                if "experts" in parts:
                    m = mlp.experts(hf, ix_, sc_).reshape(B, T, -1)
                else:
                    m = sc_.sum() * 0
            x = x + a + m
        if logits == "none" or "lm" not in parts:
            return x
        if logits == "last":
            x = x[:, -1:]
        return self.lm_head(self.model.norm(x)) * cfg.logit_scale

    # ---------------------------------------------------------------- eager prefill (python-int positions, SDPA)
    @torch.no_grad()
    def prefill(self, ids_list, kv, b=0, chunk=1024):
        cfg = self.cfg
        D, H, Hkv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
        G, W = H // Hkv, cfg.sliding_window
        n = len(ids_list)
        for s0 in range(0, n, chunk):
            c = torch.tensor([ids_list[s0:s0 + chunk]], device=self.dev)
            T = c.shape[1]
            qpos = torch.arange(s0, s0 + T, device=self.dev)
            x = self.model.embed_tokens(c)
            cos, sin = self.model.rotary_emb(x, qpos[None])
            hi = min(((s0 + T + 7) // 8) * 8, self.Lmax)
            for i, layer in enumerate(self.model.layers):
                h = layer.input_layernorm(x)
                at = layer.self_attn
                sliding = cfg.layer_types[i] == "sliding_attention"
                q = at.q_proj(h).view(1, T, H, D).transpose(1, 2)
                k = at.k_proj(h).view(1, T, Hkv, D).transpose(1, 2)
                v = at.v_proj(h).view(1, T, Hkv, D).transpose(1, 2)
                if sliding or at.force_rope:
                    q, k = apply_rotary_pos_emb(q, k, cos, sin)
                kc, vc = kv[i][0][b:b + 1], kv[i][1][b:b + 1]
                kc.index_copy_(2, qpos, k)
                vc.index_copy_(2, qpos, v)
                lo = max(0, s0 - W) if sliding else 0
                lo = (lo // 8) * 8
                kp = torch.arange(lo, hi, device=self.dev)
                mk = (kp[None, :] <= qpos[:, None])
                if sliding:
                    mk = mk & (kp[None, :] > qpos[:, None] - W)
                mk = mk.repeat(G, 1)
                qg = q.reshape(1, Hkv, G, T, D).reshape(1, Hkv, G * T, D)
                o = F.scaled_dot_product_attention(qg, kc[:, :, lo:hi], vc[:, :, lo:hi], attn_mask=mk)
                o = o.reshape(1, Hkv, G, T, D).reshape(1, H, T, D).transpose(1, 2).reshape(1, T, H * D)
                a = at.o_proj(o)
                mlp = layer.mlp
                if not hasattr(mlp, "gate"):
                    m = mlp(h)
                else:
                    hf = h.reshape(-1, h.shape[-1])
                    _, sc_, ix_ = mlp.gate(hf)
                    m = mlp.experts(hf, ix_, sc_).reshape(1, T, -1)
                x = x + a + m
        torch.cuda.synchronize()

    # ---------------------------------------------------------------- graph helpers
    def make_step(self, B, T, kv, **kw):
        ids = torch.zeros(B, T, dtype=torch.long, device=self.dev)
        kvv = {i: (a[:B], b_[:B]) for i, (a, b_) in kv.items()}  # contiguous leading-dim views

        def fn():
            lg = self.forward(ids, kvv, **kw)
            return lg.argmax(-1), lg

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn()
        return Step(self, g, ids, out)


class Step:
    def __init__(self, ln, g, ids, out):
        self.ln, self.g, self.ids, self.out = ln, g, ids, out

    def set(self, toks, start):  # toks [B][T]
        self.ids.copy_(torch.tensor(toks, device=self.ln.dev))
        self.ln.start.fill_(start)

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
