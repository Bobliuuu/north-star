"""Evaluation suite of the 'triton' self-improve track: launch parameters (src/tritonk/tuning.py) of the fast Triton decode kernels
(harness/fast_kernels.py T=1 path, fast_kernels_blk.py / fast_kernels_exact.py exact + verify-block path). PROTECTED (hash-checked in
every sandbox). The kernel bodies are the repo's real kernels, unmodified; this module only LAUNCHES them with the tuning values.

Three entry points (thin command scripts triton_check.py / triton_quality.py / triton_bench.py call them):
  run_check(seed, c)   correctness cases vs torch references on fresh random inputs (+ BITWISE T=1-vs-block-row invariance of the exact kernels)
  run_quality(c)       24 fixed-seed items: 1 - rel_l2(full synthetic exact step with c, same step with the FROZEN default launch config)
  run_bench(split, c)  CUDA-graph replay timing (CUDA events + independent host wall) of per-kernel microbenchmarks and full synthetic
                       T=1 (fast), T=1 (exact) and T=B (exact verify block) steps; split 'dev' vs disjoint 'heldout' shapes (ctx lengths, block sizes, linear shapes)
Synthetic step = 4 layers (3 sliding + 1 full attention) with distinct random weights of the real North-Mini-Code shapes (128 experts, top-8,
Dm 2048, I 768, 4 kv heads x 8 groups x 128, vocab 262144); weights are random because the kernels are data independent."""
import math, os, statistics, sys, time

_T_IMPORT = time.perf_counter()
_ROOT = os.environ.get("NS_WORKSPACE") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "harness")):
    if _p not in sys.path: sys.path.insert(0, _p)
import torch
import torch.nn.functional as F
import triton
import fast_kernels as fk
import fast_kernels_blk as fb
import fast_kernels_exact as fx
import tritonk.tuning as TU

DEV = "cuda"
BF = torch.bfloat16
E, DM, I, KT, HKV, G, D, WIN, V = 128, 2048, 768, 8, 4, 8, 128, 4096, 262144
H = HKV * G
QD, KD = H * D, HKV * D
EPS = 1e-6

DEFAULTS = dict(GU_BN=8, GU_BK=256, GU_NW=4, GU_NS=3, DN_BN=8, DN_BK=256, DN_NW=4, DN_NS=3, ATT_NSPLIT=32, ATT_BN=64, ATT_NW=4, ATT_NS=3,
                CMB_NW=4, ROPE_NW=1, RN_NW=8, TK_NW=1, XA_CHUNK=256, XA_BN=64, XA_NW=8, XA_NS=3, XCMB_NW=4, ROPEB_NW=1, LIN_BN_S=16,
                LIN_BN_L=64, LIN_BK_A=256, LIN_BK_B=128, LIN_NW_S=4, LIN_NS_S=3, LIN_NW_L=4, LIN_NS_L=3, MB_BN1=16, MB_BK1=128,
                MB_NW1=4, MB_NS1=3, MB_BN2=16, MB_BK2=128, MB_NW2=4, MB_NS2=3, AM_CH=4096, AM_NW=4)   # FROZEN copy: the quality reference


def cfg(over=None):
    """Tuning values from src/tritonk/tuning.py (+ optional overrides, used by the human-free sweep script only)."""
    c = {k: getattr(TU, k) for k in dir(TU) if k.isupper()}
    if set(c) != set(DEFAULTS) or not all(type(v) is int for v in c.values()):
        raise RuntimeError(f"tuning.py must define exactly the known int knobs; diff={sorted(set(c) ^ set(DEFAULTS))}")
    c.update(over or {})
    return c


# ------------------------------------------------------------------------------------------------ launchers (real kernels, tunable launch)
def moe_t1(c, x, idx, sc, gu, dn, hb, ob):
    k, dm, i = idx.shape[-1], x.shape[-1], dn.shape[-1]
    fk._gateup_kernel[(k, i // c["GU_BN"])](x, idx, gu, hb, K=dm, I=i, BN=c["GU_BN"], BK=c["GU_BK"], num_warps=c["GU_NW"], num_stages=c["GU_NS"])
    fk._down_kernel[(k, dm // c["DN_BN"])](hb, idx, sc, dn, ob, K=i, N=dm, BN=c["DN_BN"], BK=c["DN_BK"], num_warps=c["DN_NW"], num_stages=c["DN_NS"])
    return ob


def moe_blk(c, x, idx, sc, gu, dn, hb, ob):
    T, k = idx.shape
    dm, i = x.shape[-1], dn.shape[-1]
    tp, tkp = max(16, triton.next_power_of_2(T)), triton.next_power_of_2(T * k)
    fb._gateup_blk[(T * k, i // c["MB_BN1"])](x, idx, gu, hb, K=dm, I=i, T=T, KK=k, TP=tp, TKP=tkp, BN=c["MB_BN1"], BK=c["MB_BK1"],
                                              num_warps=c["MB_NW1"], num_stages=c["MB_NS1"])
    fb._down_blk[(T * k, dm // c["MB_BN2"])](hb, idx, sc, dn, ob, K=i, N=dm, T=T, KK=k, TP=tp, TKP=tkp, BN=c["MB_BN2"], BK=c["MB_BK2"],
                                             num_warps=c["MB_NW2"], num_stages=c["MB_NS2"])
    return ob


class WS1:
    """AttnWS of the T=1 path (fast_kernels.AttnWS) for the tuned split count."""
    def __init__(self, c): self.ws = fk.AttnWS(HKV, G, D, c["ATT_NSPLIT"], DEV)


def attn_t1(c, w, qkv, cos, sin, start, kc, vc, lmax, rope, sliding):
    ws, ns = w.ws, c["ATT_NSPLIT"]
    fk._rope_write_kernel[(HKV,)](qkv, cos, sin, start, kc, vc, ws.q, lmax, G=G, D=D, QD=QD, KD=KD, ROPE=rope, num_warps=c["ROPE_NW"])
    fk._attn_split_kernel[(HKV, ns)](ws.q, kc, vc, start, ws.acc, ws.m, ws.l, lmax, D ** -0.5, G=G, D=D, NSPLIT=ns, BN=c["ATT_BN"],
                                     SLIDING=sliding, W=WIN, num_warps=c["ATT_NW"], num_stages=c["ATT_NS"])
    fk._attn_combine_kernel[(HKV * G,)](ws.acc, ws.m, ws.l, ws.o, G=G, D=D, NSPLIT=ns, num_warps=c["CMB_NW"])
    return ws.o


def make_wsx(c, lmax, T):
    fx.CHUNK = c["XA_CHUNK"]                     # AttnWSX sizes its split count from this module global
    return fx.AttnWSX(HKV, G, D, lmax, T, DEV)


def attn_x(c, ws, qkv, cos, sin, start, kc, vc, lmax, rope, sliding):
    T, mp, ns = ws.T, ws.MP, ws.NS
    fb._rope_write_blk[(HKV, T)](qkv, cos, sin, start, kc, vc, ws.q, lmax, G=G, D=D, QD=QD, KD=KD, MP=mp, ROPE=rope, num_warps=c["ROPEB_NW"])
    fx._attn_split_x[(HKV, ns)](ws.q, kc, vc, start, ws.acc, ws.m, ws.l, lmax, D ** -0.5, T=T, G=G, D=D, MP=mp, NS=ns, BN=c["XA_BN"],
                                C=c["XA_CHUNK"], SLIDING=sliding, W=WIN, num_warps=c["XA_NW"], num_stages=c["XA_NS"])
    fx._attn_combine_x[(HKV * T * G,)](ws.acc, ws.m, ws.l, ws.o, T=T, G=G, Hkv=HKV, D=D, MP=mp, NS=ns, num_warps=c["XCMB_NW"])
    return ws.o


def lin_x(c, x, w, out=None):
    T, K = x.shape
    N = w.shape[0]
    cl = "S" if N <= 4096 else "L"
    bn, bk = c["LIN_BN_" + cl], (c["LIN_BK_A"] if K <= 2048 else c["LIN_BK_B"])
    if out is None: out = torch.empty(T, N, device=x.device, dtype=BF)
    fx._lin_kernel[(triton.cdiv(N, bn),)](x, w, out, T=T, N=N, K=K, TPAD=fx.TP, BN=bn, BK=bk, num_warps=c["LIN_NW_" + cl], num_stages=c["LIN_NS_" + cl])
    return out


def argmax_x(c, lg):
    T, v = lg.shape
    ch = c["AM_CH"]; nch = triton.cdiv(v, ch)
    pm = torch.empty(T, nch, device=lg.device, dtype=torch.float32); pi = torch.empty(T, nch, device=lg.device, dtype=torch.int32)
    o = torch.empty(T, device=lg.device, dtype=torch.int64)
    fx._amax1[(T, nch)](lg, pm, pi, V=v, CH=ch, NCH=nch, num_warps=c["AM_NW"])
    fx._amax2[(T,)](pm, pi, o, NCH=nch, NP=triton.next_power_of_2(nch), num_warps=1)
    return o


def topk(c, lg, idx, sc):
    fk._topk_kernel[(lg.shape[0],)](lg, idx, sc, N=lg.shape[-1], KK=KT, num_warps=c["TK_NW"])


def resid(c, x, a, m, w, xo, ho):
    fk._resid_norm_kernel[(x.numel() // DM,)](x, a, m, w, xo, ho, EPS, D=DM, NB=m.numel() // x.numel(), num_warps=c["RN_NW"])


# ------------------------------------------------------------------------------------------------ synthetic model state
def rn(g, *s, scale=0.02):
    return (torch.randn(*s, device=DEV, dtype=BF, generator=g) * scale)


class State:
    def __init__(self, seed=0, nl=4, with_lm=True):
        g = torch.Generator(device=DEV); g.manual_seed(seed); self.g = g
        self.layers = []
        for _ in range(nl):
            self.layers.append(dict(wqkv=rn(g, QD + 2 * KD, DM), wo=rn(g, DM, QD), wg=rn(g, E, DM, scale=0.1), gu=rn(g, E, 2 * I, DM), dn=rn(g, E, DM, I),
                                    ln=(torch.rand(DM, device=DEV, generator=g) + 0.5).to(BF)))
        self.lm = rn(g, V, DM) if with_lm else None
        self.extra = {}

    def wset(self, N, K, copies):
        k = (N, K, copies)
        if k not in self.extra: self.extra[k] = [rn(self.g, N, K) for _ in range(copies)]
        return self.extra[k]


def rope_tables(g, start, T):
    inv = 1.0 / (50000 ** (torch.arange(0, D, 2, device=DEV).float() / D))
    pos = (start + torch.arange(T, device=DEV)).float()[:, None] * inv
    fr = pos.repeat_interleave(2, dim=-1)
    return fr.cos().to(BF).contiguous(), fr.sin().to(BF).contiguous()


def kv_caches(g, nl, lmax):
    return [(rn(g, 1, HKV, lmax, D, scale=1.0), rn(g, 1, HKV, lmax, D, scale=1.0)) for _ in range(nl)]


def sliding_of(i): return (i % 4) != 3


def routed(g, T):
    idx = torch.stack([torch.randperm(E, device=DEV, generator=g)[:KT] for _ in range(T)]).to(torch.int32).contiguous()
    return idx, torch.rand(T, KT, device=DEV, generator=g).to(BF).contiguous()


# ------------------------------------------------------------------------------------------------ full synthetic steps
def build_step(c, S, ctx, T, mode, g=None):
    """mode 'fast' (T=1 kernels of fast_kernels.py, cuBLAS linears) or 'exact' (batch-invariant kernels, any 1 <= T <= 16). Returns fn()->final hidden."""
    g = g or S.g
    lmax = ctx + 64
    kv = kv_caches(g, len(S.layers), lmax)
    start = torch.tensor(ctx, device=DEV, dtype=torch.int64)
    cos, sin = rope_tables(g, ctx, T)
    x0, h0 = rn(g, T, DM, scale=1.0), rn(g, T, DM, scale=1.0)
    x2, h2 = torch.zeros(T, DM, device=DEV, dtype=BF), torch.zeros(T, DM, device=DEV, dtype=BF)
    idx, sc = torch.zeros(T, KT, device=DEV, dtype=torch.int32), torch.zeros(T, KT, device=DEV, dtype=BF)
    hb, ob = torch.zeros(T * KT, I, device=DEV, dtype=BF), torch.zeros(T * KT, DM, device=DEV, dtype=BF)
    if mode == "fast":
        assert T == 1
        w1 = WS1(c); cos1, sin1 = cos.reshape(-1), sin.reshape(-1)
    else:
        wx = make_wsx(c, lmax, T)

    def fn():
        x, h = x0, h0
        for i, L in enumerate(S.layers):
            kc, vc = kv[i]
            nxt = S.layers[i + 1]["ln"] if i + 1 < len(S.layers) else S.layers[0]["ln"]
            if mode == "fast":
                hv = h.view(-1)
                qkv = F.linear(hv, L["wqkv"])
                ao = attn_t1(c, w1, qkv, cos1, sin1, start, kc, vc, lmax, True, sliding_of(i))
                a = F.linear(ao, L["wo"])
                lg = F.linear(h, L["wg"])
                topk(c, lg, idx, sc)
                moe_t1(c, hv, idx.view(-1), sc.view(-1), L["gu"], L["dn"], hb, ob)
                resid(c, x.view(-1), a, ob, nxt, x2.view(-1), h2.view(-1))
            else:
                qkv = lin_x(c, h, L["wqkv"])
                ao = attn_x(c, wx, qkv, cos, sin, start, kc, vc, lmax, True, sliding_of(i))
                a = lin_x(c, ao, L["wo"])
                lg = lin_x(c, h, L["wg"])
                topk(c, lg, idx, sc)
                moe_blk(c, h, idx, sc, L["gu"], L["dn"], hb, ob)
                resid(c, x, a, ob, nxt, x2, h2)
            x, h = x2, h2
        if S.lm is not None:
            if mode == "fast": F.linear(h, S.lm).argmax(-1)
            else: argmax_x(c, lin_x(c, h, S.lm))
        return h2
    return fn


# ------------------------------------------------------------------------------------------------ graph timing
def capture(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr): out = fn()
    return gr, out


def time_graph(gr, target_ms=15.0, batches=5):
    """(event_ms, host_wall_ms) per replay, medians over batches. Replays per batch sized so a batch lasts >= target_ms."""
    for _ in range(3): gr.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); gr.replay(); b.record(); b.synchronize()
    reps = max(5, min(400, int(target_ms / max(a.elapsed_time(b), 0.01))))
    ev, host = [], []
    for _ in range(batches):
        torch.cuda.synchronize()
        t0 = time.perf_counter(); a.record()
        for _ in range(reps): gr.replay()
        b.record(); b.synchronize()
        host.append((time.perf_counter() - t0) * 1000.0 / reps); ev.append(a.elapsed_time(b) / reps)
    return statistics.median(ev), statistics.median(host), reps * batches


SPLITS = {
    "dev": dict(ctx=[2048, 8192], TB=13, lin=[(5120, 2048), (262144, 2048)], lin_T=[1, 13], step_ctx=[2048, 8192], small_T=[1, 13]),
    "heldout": dict(ctx=[1500, 5000, 12000], TB=11, lin=[(3072, 2048), (6144, 2048), (131072, 2048), (2560, 4096)], lin_T=[1, 5], step_ctx=[3000, 10000], small_T=[1, 7]),
}


def case_builders(split, c, S):
    """[(name, build() -> fn)] for the split. Micro cases cycle the 4 distinct layer weights/caches so L2 cannot hold the working set."""
    sp = SPLITS[split]; g = S.g; out = []
    nl = len(S.layers)

    def moe1():
        idxs = [routed(g, 1) for _ in range(nl)]; x = rn(g, DM, scale=1.0); hb = torch.zeros(KT, I, device=DEV, dtype=BF); ob = torch.zeros(KT, DM, device=DEV, dtype=BF)
        def fn():
            for L, (ix, s) in zip(S.layers, idxs): moe_t1(c, x, ix.view(-1), s.view(-1), L["gu"], L["dn"], hb, ob)
            return ob
        return fn
    out.append(("moe_t1", moe1))

    def moeb(T):
        def b():
            idxs = [routed(g, T) for _ in range(nl)]; x = rn(g, T, DM, scale=1.0); hb = torch.zeros(T * KT, I, device=DEV, dtype=BF); ob = torch.zeros(T * KT, DM, device=DEV, dtype=BF)
            def fn():
                for L, (ix, s) in zip(S.layers, idxs): moe_blk(c, x, ix, s, L["gu"], L["dn"], hb, ob)
                return ob
            return fn
        return b
    out.append((f"moe_blk_T{sp['TB']}", moeb(sp["TB"])))

    def attn(ctx, T):
        def b():
            lmax = ctx + 64; kv = kv_caches(g, nl, lmax); start = torch.tensor(ctx, device=DEV, dtype=torch.int64); cos, sin = rope_tables(g, ctx, T)
            qkv = rn(g, T, QD + 2 * KD, scale=1.0)
            if T == 1:
                w1 = WS1(c); cos1, sin1 = cos.reshape(-1), sin.reshape(-1); q1 = qkv.view(-1)
                def fn():
                    for i, (kc, vc) in enumerate(kv): attn_t1(c, w1, q1, cos1, sin1, start, kc, vc, lmax, True, sliding_of(i))
                    return w1.ws.o
            else:
                wx = make_wsx(c, lmax, T)
                def fn():
                    for i, (kc, vc) in enumerate(kv): attn_x(c, wx, qkv, cos, sin, start, kc, vc, lmax, True, sliding_of(i))
                    return wx.o
            return fn
        return b
    for ctx in sp["ctx"]:
        out.append((f"attn_t1_c{ctx}", attn(ctx, 1)))
        out.append((f"attn_x_T{sp['TB']}_c{ctx}", attn(ctx, sp["TB"])))

    def linc(N, K, T):
        def b():
            ws = S.wset(N, K, 1 if N * K > 3e8 else 4); x = rn(g, T, K, scale=1.0)
            def fn():
                for w in ws: lin_x(c, x, w)
                return x
            return fn
        return b
    for (N, K) in sp["lin"]:
        for T in sp["lin_T"]: out.append((f"lin_N{N}_K{K}_T{T}", linc(N, K, T)))

    def small(T):
        def b():
            lg = rn(g, T, E, scale=1.0); idx = torch.zeros(T, KT, device=DEV, dtype=torch.int32); sc = torch.zeros(T, KT, device=DEV, dtype=BF)
            x, a = rn(g, T, DM, scale=1.0), rn(g, T, DM, scale=1.0); m = rn(g, T * KT, DM, scale=1.0); w = S.layers[0]["ln"]
            xo, ho = torch.zeros(T, DM, device=DEV, dtype=BF), torch.zeros(T, DM, device=DEV, dtype=BF); big = rn(g, T, V, scale=2.0)
            def fn():
                for _ in range(8): topk(c, lg, idx, sc); resid(c, x, a, m, w, xo, ho)
                argmax_x(c, big)
                return xo
            return fn
        return b
    for T in sp["small_T"]: out.append((f"small_T{T}", small(T)))

    for ctx in sp["step_ctx"]:
        out.append((f"step_fast_T1_c{ctx}", lambda ctx=ctx: build_step(c, S, ctx, 1, "fast")))
        out.append((f"step_exact_T1_c{ctx}", lambda ctx=ctx: build_step(c, S, ctx, 1, "exact")))
        out.append((f"step_exact_T{sp['TB']}_c{ctx}", lambda ctx=ctx: build_step(c, S, ctx, sp["TB"], "exact")))
    return out


def run_bench(split="dev", c=None, S=None, only=None, quiet=True):
    """Returns dict(time_ms=sum of event ms, ref_ms=sum of host-wall ms, n_iters, percase={name: ms}, harness_wall_ms)."""
    c = c or cfg(); S = S or State(0)
    per, host, n_it = {}, {}, 10 ** 9
    for name, build in case_builders(split, c, S):
        if only and not any(name.startswith(o) for o in only): continue
        gr, _ = capture(build())
        ev, hw, n = time_graph(gr)
        per[name], host[name], n_it = ev, hw, min(n_it, n)
        del gr
    return dict(time_ms=sum(per.values()), ref_ms=sum(host.values()), n_iters=n_it, percase=per, harness_wall_ms=(time.perf_counter() - _T_IMPORT) * 1000.0, split=split)


# ------------------------------------------------------------------------------------------------ references
def bf(x): return x.to(BF).float()


def ref_moe(x, idx, sc, gu, dn):
    """x [T,Dm] bf16, idx [T,k], sc [T,k] -> [T*k, Dm] fp32-accumulated reference with the kernels' bf16 rounding points."""
    rows = []
    for t in range(x.shape[0]):
        for j in range(idx.shape[1]):
            e = int(idx[t, j]); gw = gu[e].float() @ x[t].float()
            gg, uu = bf(gw[:I]), bf(gw[I:]); s = bf(gg * torch.sigmoid(gg)); h = bf(s * uu)
            rows.append(bf(bf(dn[e].float() @ h) * sc[t, j].float()))
    return torch.stack(rows)


def ref_attn(qkv, kc, vc, start, T, sliding):
    """qkv [T, QD+2KD] (rope OFF, i.e. cos=1 sin=0), kc/vc [1,HKV,L,D] BEFORE the write. Returns [T, H*D] fp32-rounded-to-bf16."""
    kc2, vc2 = kc.clone(), vc.clone()
    q = qkv[:, :QD].view(T, HKV, G, D).float(); kk = qkv[:, QD:QD + KD].view(T, HKV, D); vv = qkv[:, QD + KD:].view(T, HKV, D)
    for t in range(T): kc2[0, :, start + t] = kk[t]; vc2[0, :, start + t] = vv[t]
    L = kc2.shape[2]; ar = torch.arange(L, device=DEV); out = []
    for t in range(T):
        s = torch.einsum("hgd,hld->hgl", q[t], kc2[0].float()) * D ** -0.5
        ok = ar <= start + t
        if sliding: ok = ok & (ar > start + t - WIN)
        s = s.masked_fill(~ok[None, None], float("-inf")); m = s.max(-1, keepdim=True).values
        p = torch.exp(s - m); l = p.sum(-1, keepdim=True)
        o = torch.einsum("hgl,hld->hgd", bf(p), vc2[0].float()) / l
        out.append(bf(o).reshape(-1))
    return torch.stack(out)


def ref_resid(x, a, m, w):
    T = x.shape[0]
    t = bf(x.float() + a.float()); ms = bf(m.float().view(T, -1, DM).sum(1)); xn = bf(t + ms)
    hn = bf(xn * torch.rsqrt((xn * xn).mean(-1, keepdim=True) + EPS)); return xn, bf(w.float() * hn)


def err(o, r):
    o, r = o.float(), r.float()
    fin = bool(torch.isfinite(o).all())
    return dict(rel_l2=float((o - r).norm() / (r.norm() + 1e-12)) if fin else 1e9, max_abs=float((o - r).abs().max()) if fin else 1e9, finite=fin)


def exact_eq(a, b):
    ok = bool(torch.equal(a, b)); d = float((a.float() - b.float()).abs().max()) if not ok and torch.isfinite(a.float()).all() else 0.0
    return dict(rel_l2=0.0 if ok else 1.0, max_abs=d, finite=True)


# ------------------------------------------------------------------------------------------------ correctness cases
def run_check(seed, c=None):
    import random
    c = c or cfg(); rng = random.Random(seed * 7919 + 17); torch.manual_seed(seed % (2 ** 31)); S = State(seed % 1000 + 1, nl=1, with_lm=False)
    g = S.g; L0 = S.layers[0]; cases = []

    def add(name, d): cases.append(dict(name=name, **d))

    for k in range(2):                                                         # T=1 expert GEMV vs torch
        x = rn(g, 1, DM, scale=1.0); ix, s = routed(g, 1); hb = torch.zeros(KT, I, device=DEV, dtype=BF); ob = torch.zeros(KT, DM, device=DEV, dtype=BF)
        moe_t1(c, x.view(-1), ix.view(-1), s.view(-1), L0["gu"], L0["dn"], hb, ob); add(f"moe_t1_{k}", err(ob, ref_moe(x, ix, s, L0["gu"], L0["dn"])))
    for T in (2, 5, 13):                                                       # block experts: vs torch AND bitwise vs T=1 calls
        x = rn(g, T, DM, scale=1.0); ix, s = routed(g, T); hb = torch.zeros(T * KT, I, device=DEV, dtype=BF); ob = torch.zeros(T * KT, DM, device=DEV, dtype=BF)
        moe_blk(c, x, ix, s, L0["gu"], L0["dn"], hb, ob); add(f"moe_blk_T{T}", err(ob, ref_moe(x, ix, s, L0["gu"], L0["dn"])))
        eq = True
        for t in range(T):
            h1 = torch.zeros(KT, I, device=DEV, dtype=BF); o1 = torch.zeros(KT, DM, device=DEV, dtype=BF)
            moe_blk(c, x[t:t + 1].contiguous(), ix[t:t + 1].contiguous(), s[t:t + 1].contiguous(), L0["gu"], L0["dn"], h1, o1)
            eq &= bool(torch.equal(ob[t * KT:(t + 1) * KT], o1))
        add(f"moe_blk_rowinv_T{T}", dict(rel_l2=0.0 if eq else 1.0, max_abs=0.0, finite=True))
    cs1, sn1 = torch.ones(1, D, device=DEV, dtype=BF), torch.zeros(1, D, device=DEV, dtype=BF)
    for sliding in (False, True):                                              # T=1 split-K attention vs torch (rope off so the reference is exact)
        ctx = rng.randint(300, 13000); lmax = ctx + 64; kc, vc = kv_caches(g, 1, lmax)[0]; qkv = rn(g, 1, QD + 2 * KD, scale=1.0)
        ref = ref_attn(qkv, kc, vc, ctx, 1, sliding); st = torch.tensor(ctx, device=DEV, dtype=torch.int64); w1 = WS1(c)
        o = attn_t1(c, w1, qkv.view(-1), cs1.view(-1), sn1.view(-1), st, kc, vc, lmax, False, sliding).view(1, -1); add(f"attn_t1_{'sl' if sliding else 'full'}", err(o, ref))
    for sliding in (False, True):                                              # exact attention T=13 vs torch, and sequential-vs-block bitwise (rope ON)
        ctx = rng.randint(300, 13000); lmax = ctx + 64; T = 13; kc, vc = kv_caches(g, 1, lmax)[0]; qkv = rn(g, T, QD + 2 * KD, scale=1.0)
        csT, snT = torch.ones(T, D, device=DEV, dtype=BF), torch.zeros(T, D, device=DEV, dtype=BF); st = torch.tensor(ctx, device=DEV, dtype=torch.int64)
        wx = make_wsx(c, lmax, T); o = attn_x(c, wx, qkv, csT, snT, st, kc.clone(), vc.clone(), lmax, False, sliding)
        add(f"attn_x_ref_{'sl' if sliding else 'full'}", err(o, ref_attn(qkv, kc, vc, ctx, T, sliding)))
        start = rng.choice([0, 3, 255, 256, 257, rng.randint(500, 12000)]); T = rng.choice([5, 13]); lm = start + 64
        kc0, vc0 = kv_caches(g, 1, lm)[0]; qkv = rn(g, T, QD + 2 * KD, scale=1.0); cs, sn = rope_tables(g, start, T)
        stt = torch.tensor(start, device=DEV, dtype=torch.int64); wxT = make_wsx(c, lm, T); kcA, vcA = kc0.clone(), vc0.clone()
        ob = attn_x(c, wxT, qkv, cs, sn, stt, kcA, vcA, lm, True, sliding).clone()
        w1x = make_wsx(c, lm, 1); kcB, vcB = kc0.clone(), vc0.clone(); rows = []
        for t in range(T):
            stt.fill_(start + t)
            rows.append(attn_x(c, w1x, qkv[t:t + 1].contiguous(), cs[t:t + 1].contiguous(), sn[t:t + 1].contiguous(), stt, kcB, vcB, lm, True, sliding).clone())
        eq = bool(torch.equal(ob, torch.cat(rows))) and bool(torch.equal(kcA, kcB)) and bool(torch.equal(vcA, vcB))
        add(f"attn_x_rowinv_{'sl' if sliding else 'full'}", dict(rel_l2=0.0 if eq else 1.0, max_abs=0.0, finite=True))
    for (N, K) in ((5120, 2048), (2048, 4096), (128, 2048)):                   # exact linears vs torch + bitwise row invariance (T=13 / T=16 block vs T=1)
        w = rn(g, N, K); x = rn(g, 16, K, scale=1.0); full = lin_x(c, x, w); b13 = lin_x(c, x[:13].contiguous(), w)
        add(f"lin_ref_N{N}", err(b13, bf(x[:13].float() @ w.float().T)))
        eq = all(torch.equal(lin_x(c, x[r:r + 1].contiguous(), w)[0], full[r]) for r in range(16)) and all(torch.equal(b13[r], full[r]) for r in range(13))
        add(f"lin_rowinv_N{N}", dict(rel_l2=0.0 if eq else 1.0, max_abs=0.0, finite=True))
    lg = rn(g, 13, V, scale=2.0); lg[3, 5000] = 40; lg[3, 9] = 40; lg[7, V - 1] = 50; lg[7, V - 100] = 50; lg[0, 4096] = 60; lg[0, 4095] = 60
    a = argmax_x(c, lg); ref = torch.stack([(lg[r] == lg[r].max()).nonzero()[0, 0] for r in range(13)])
    add("argmax_lowest_tie", dict(rel_l2=0.0 if torch.equal(a, ref) else 1.0, max_abs=0.0, finite=True))
    add("argmax_rowinv", dict(rel_l2=0.0 if torch.equal(a, torch.cat([argmax_x(c, lg[r:r + 1].contiguous()) for r in range(13)])) else 1.0, max_abs=0.0, finite=True))
    for T in (1, 13):                                                          # residual+norm and router top-k
        x, aa, m = rn(g, T, DM, scale=1.0), rn(g, T, DM, scale=1.0), rn(g, T * KT, DM, scale=1.0); w = L0["ln"]
        xo, ho = torch.zeros(T, DM, device=DEV, dtype=BF), torch.zeros(T, DM, device=DEV, dtype=BF); resid(c, x, aa, m, w, xo, ho)
        rx, rh = ref_resid(x, aa, m, w); add(f"resid_T{T}", err(torch.cat([xo, ho]), torch.cat([rx, rh])))
        lgt = torch.stack([torch.randperm(E, device=DEV, generator=g).float() / 16 for _ in range(T)]).to(BF).contiguous()
        ix, s = torch.zeros(T, KT, device=DEV, dtype=torch.int32), torch.zeros(T, KT, device=DEV, dtype=BF); topk(c, lgt, ix, s)
        tv, ti = torch.topk(lgt.float(), KT, -1); okk = all(set(ix[t].tolist()) == set(ti[t].tolist()) for t in range(T)) and bool(torch.equal(s.float().sort(-1).values, torch.sigmoid(tv).to(BF).float().sort(-1).values))
        add(f"topk_T{T}", dict(rel_l2=0.0 if okk else 1.0, max_abs=0.0, finite=True))
    torch.cuda.synchronize()
    return cases


# ------------------------------------------------------------------------------------------------ quality items
def run_quality(c=None):
    """24 paired items, fixed inputs (seed 0): 1 - rel_l2 of the final hidden of a full synthetic exact step vs the same step with the frozen default launch config."""
    c = c or cfg(); S = State(0, nl=4, with_lm=False); items = []
    ctxs = [700, 1500, 2300, 3100, 4500, 6000, 9000, 13000]
    for T in (1, 5, 13):
        for ci, ctx in enumerate(ctxs):
            seed = 1000 + T * 100 + ci
            outs = []
            for conf in (c, DEFAULTS):
                g = torch.Generator(device=DEV); g.manual_seed(seed)
                fn = build_step(conf, S, ctx, T, "exact", g=g); outs.append(fn().clone())
            torch.cuda.synchronize()
            items.append(1.0 - float((outs[0].float() - outs[1].float()).norm() / (outs[1].float().norm() + 1e-12)) if torch.isfinite(outs[0].float()).all() else -1e9)
    return items
