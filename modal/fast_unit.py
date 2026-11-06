"""modal run modal/fast_unit.py   -- A10G ($~1.1/h) unit tests of harness/fast_kernels.py vs torch references (no weights)."""
import sys, time, json, pathlib
import modal

image = (modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.10.0", "numpy==2.2.5")
         .add_local_dir("harness", "/root/harness", copy=False))
app = modal.App("northstar-fast-unit", image=image)


@app.function(gpu="A10G", timeout=900)
def run():
    sys.path.insert(0, "/root/harness")
    import torch, torch.nn.functional as F
    import fast_kernels as fk
    dev, bf = "cuda", torch.bfloat16
    out = []
    torch.manual_seed(0)

    def rel(a, b):
        a, b = a.float(), b.float()
        return ((a - b).abs().max() / (b.abs().max() + 1e-9)).item()

    # ---- moe
    E, Dm, I, k = 16, 2048, 768, 8
    gu = (torch.randn(E, 2 * I, Dm, device=dev) * 0.02).to(bf)
    dn = (torch.randn(E, Dm, I, device=dev) * 0.02).to(bf)
    x = torch.randn(Dm, device=dev).to(bf)
    idx = torch.randperm(E, device=dev)[:k].to(torch.int32)
    sc = torch.rand(k, device=dev).to(bf)
    hb = torch.zeros(k, I, device=dev, dtype=bf); ob = torch.zeros(k, Dm, device=dev, dtype=bf)
    fk.moe_gemv(x, idx, sc, gu, dn, hb, ob)
    ref = []
    for j in range(k):
        e = int(idx[j]); g, u = F.linear(x[None], gu[e]).chunk(2, -1)
        h = F.silu(g) * u
        ref.append(F.linear(h, dn[e])[0] * sc[j])
    ref = torch.stack(ref)
    out.append(dict(test="moe_gemv", rel_err=rel(ob, ref)))
    for cfg in (dict(BN1=8, BK1=512, BN2=8, BK2=256), dict(BN1=32, BK1=128, BN2=32, BK2=128, nw1=8, nw2=8)):
        ob.zero_(); fk.moe_gemv(x, idx, sc, gu, dn, hb, ob, cfg); out.append(dict(test="moe_gemv_cfg", cfg=cfg, rel_err=rel(ob, ref)))

    # ---- router
    lg = torch.randn(1, 128, device=dev).to(bf)
    i32 = torch.zeros(8, dtype=torch.int32, device=dev); s16 = torch.zeros(8, dtype=bf, device=dev)
    fk.router_topk(lg, i32, s16, 8)
    tv, ti = torch.topk(lg, 8, -1)
    # bf16 logits can tie: compare the score vectors and index sets
    out.append(dict(test="router", scores_eq=bool((torch.sigmoid(tv)[0] == s16).all()), idx_set_eq=set(ti[0].tolist()) == set(i32.tolist()), idx_order_eq=bool((ti[0].int() == i32).all())))

    # ---- resid_norm
    xx = torch.randn(Dm, device=dev).to(bf); aa = torch.randn(Dm, device=dev).to(bf); mm = torch.randn(8, Dm, device=dev).to(bf)
    w = (torch.rand(Dm, device=dev) + 0.5).to(bf)
    xo = torch.zeros(Dm, device=dev, dtype=bf); ho = torch.zeros(Dm, device=dev, dtype=bf)
    fk.resid_norm(xx, aa, mm, w, xo, ho, 1e-6)
    xr = xx + aa + mm.sum(0)
    v = xr.float().pow(2).mean(-1, keepdim=True); hr = w * (xr.float() * torch.rsqrt(v + 1e-6)).to(bf)
    out.append(dict(test="resid_norm", x_rel=rel(xo, xr), h_rel=rel(ho, hr), x_exact=bool((xo == xr).all()), h_exact=bool((ho == hr).all())))

    # ---- attention
    Hkv, G, D, Lmax, Wn = 4, 8, 128, 6000, 4096
    H = Hkv * G
    inv = 1.0 / (50000 ** (torch.arange(0, D, 2, device=dev).float() / D))

    def rot(t, c, s):
        t = t.float(); x1, x2 = t[..., ::2], t[..., 1::2]
        rh = torch.stack([-x2, x1], -1).flatten(-2)
        return (t * c + rh * s).to(bf)

    for sliding in (False, True):
        for rope in (False, True):
            for start in (0, 5, 63, 64, 1000, 5000):
                kc = (torch.randn(1, Hkv, Lmax, D, device=dev)).to(bf); vc = torch.randn(1, Hkv, Lmax, D, device=dev).to(bf)
                kc2, vc2 = kc.clone(), vc.clone()
                qkv = torch.randn(H * D + 2 * Hkv * D, device=dev).to(bf)
                fr = (start * inv).repeat_interleave(2); cs, sn = fr.cos().to(bf), fr.sin().to(bf)
                st = torch.tensor(start, device=dev, dtype=torch.int64)
                ws = fk.AttnWS(Hkv, G, D, 32, dev)
                o = fk.decode_attention(ws, qkv, cs, sn, st, kc, vc, Lmax, rope, sliding, Wn).clone()
                q = qkv[:H * D].view(H, D); kk = qkv[H * D:H * D + Hkv * D].view(Hkv, D); vv = qkv[H * D + Hkv * D:].view(Hkv, D)
                if rope:
                    q, kk = rot(q, cs.float(), sn.float()), rot(kk, cs.float(), sn.float())
                kc2[0, :, start] = kk; vc2[0, :, start] = vv
                ar = torch.arange(Lmax, device=dev)
                m = ar <= start
                if sliding: m &= ar > start - Wn
                qg = q.view(Hkv, G, D)
                s_ = torch.bmm(qg, kc2[0].transpose(1, 2), out_dtype=torch.float32) * D ** -0.5
                s_ = s_.masked_fill(~m[None, None], float("-inf"))
                p = torch.softmax(s_, -1).to(bf)
                ref = torch.bmm(p, vc2[0]).reshape(-1)
                out.append(dict(test="attn", sliding=sliding, rope=rope, start=start, rel_err=rel(o, ref), kv_written=bool((kc[0, :, start] == kc2[0, :, start]).all() and (vc[0, :, start] == vc2[0, :, start]).all())))

    # ---- graph capture with changing start
    st = torch.tensor(10, device=dev, dtype=torch.int64)
    kc = torch.randn(1, Hkv, Lmax, D, device=dev).to(bf); vc = torch.randn(1, Hkv, Lmax, D, device=dev).to(bf)
    qkv = torch.randn(H * D + 2 * Hkv * D, device=dev).to(bf)
    cs = torch.ones(D, device=dev).to(bf); sn = torch.zeros(D, device=dev).to(bf)
    ws = fk.AttnWS(Hkv, G, D, 32, dev)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fk.decode_attention(ws, qkv, cs, sn, st, kc, vc, Lmax, True, False, Wn)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        og = fk.decode_attention(ws, qkv, cs, sn, st, kc, vc, Lmax, True, False, Wn)
    res = []
    for start in (10, 3000):
        st.fill_(start); g.replay(); torch.cuda.synchronize(); a1 = og.clone()
        eager = fk.decode_attention(ws, qkv, cs, sn, st, kc, vc, Lmax, True, False, Wn).clone()
        res.append(rel(a1, eager))
    out.append(dict(test="graph_replay_vs_eager", rel_err=res))
    for r in out: print(json.dumps(r), flush=True)
    return out


@app.local_entrypoint()
def main():
    res = run.remote()
    pathlib.Path("results").mkdir(exist_ok=True)
    pathlib.Path("results/fast_unit_A10G.jsonl").write_text("\n".join(json.dumps(r) for r in res) + "\n")
