"""modal run --detach modal/exact_unit.py   -- BITWISE batch-invariance unit tests of the exact-spec kernels (no weights), A10G or H100.
For every kernel: the result for token row r computed as row r of a T=2..13 block equals (torch.equal) the result of the T=1 call on that token."""
import sys, time, json, pathlib
import modal

image = (modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.10.0", "numpy==2.2.5")
         .add_local_dir("harness", "/root/harness", copy=False))
app = modal.App("northstar-exact-unit", image=image)


def body(tag):
    sys.path.insert(0, "/root/harness")
    import torch, triton
    import fast_kernels as fk, fast_kernels_blk as fb, fast_kernels_exact as fx
    dev, bf = "cuda", torch.bfloat16
    out = []
    torch.manual_seed(0)
    rec = lambda **kw: (out.append(dict(gpu=torch.cuda.get_device_name(), **kw)), print(json.dumps(out[-1]), flush=True))

    def rel(a, b):
        a, b = a.float(), b.float()
        return ((a - b).abs().max() / (b.abs().max() + 1e-9)).item()

    # ---- lin
    for (N, K) in ((5120, 2048), (2048, 4096), (128, 2048), (768, 2048), (2048, 768), (32768, 2048)):
        w = (torch.randn(N, K, device=dev) * 0.05).to(bf)
        x = torch.randn(16, K, device=dev).to(bf)
        ref = torch.nn.functional.linear(x, w)
        full = {T: fx.lin(x[:T].contiguous(), w) for T in (1, 2, 5, 13, 16)}
        ok = all(torch.equal(full[T][r], full[1 if False else 16][r]) for T in (2, 5, 13) for r in range(T))
        one = all(torch.equal(fx.lin(x[r:r + 1].contiguous(), w)[0], full[16][r]) for r in range(16))
        rec(test="lin", N=N, K=K, block_rows_equal_T16=ok, T1_equals_block_row=one, rel_vs_cublas=rel(full[16], ref))

    # ---- argmax
    V = 262144
    lg = (torch.randn(13, V, device=dev) * 2).to(bf)
    lg[3, 5000] = 40; lg[3, 9] = 40; lg[7, 262143] = 50; lg[7, 262000] = 50; lg[0, 4096] = 60; lg[0, 4095] = 60
    a = fx.argmax_rows(lg); b1 = torch.cat([fx.argmax_rows(lg[r:r + 1]) for r in range(13)])
    ref = torch.stack([(lg[r] == lg[r].max()).nonzero()[0, 0] for r in range(13)])
    rec(test="argmax", equal_rows=bool(torch.equal(a, b1)), lowest_index_ties=bool(torch.equal(a, ref)))

    Hkv, G, D, Lmax, Wn = 4, 8, 128, 6000, 4096
    H = Hkv * G
    inv = 1.0 / (50000 ** (torch.arange(0, D, 2, device=dev).float() / D))
    ws = {T: fx.AttnWSX(Hkv, G, D, Lmax, T, dev) for T in (1, 2, 5, 13)}
    VARIANTS = [{}, dict(rowgroup=True, BN=64, nw=4), dict(rowgroup=True, BN=128, nw=4), dict(rowgroup=True, BN=128, nw=8), dict(rowgroup=True, BN=64, nw=2), dict(rowgroup=True, BN=128, nw=2)]
    def attn_tests(ak):
        # ---- attention: sequential T=1 steps vs one block of T tokens, bitwise (output rows AND kv cache)
        allok = True
        for sliding in (False, True):
            for rope in (False, True):
                for start in (0, 3, 255, 256, 257, 1000, 4090, 5000):
                    for T in (2, 5, 13):
                        kc0 = torch.randn(1, Hkv, Lmax, D, device=dev).to(bf); vc0 = torch.randn(1, Hkv, Lmax, D, device=dev).to(bf)
                        qkv = torch.randn(T, H * D + 2 * Hkv * D, device=dev).to(bf)
                        pos = (start + torch.arange(T, device=dev)).float()[:, None] * inv
                        fr = pos.repeat_interleave(2, dim=-1)
                        cs, sn = fr.cos().to(bf).contiguous(), fr.sin().to(bf).contiguous()
                        st = torch.tensor(start, device=dev, dtype=torch.int64)
                        kcA, vcA = kc0.clone(), vc0.clone()
                        ob = fx.exact_attention(ws[T], qkv, cs, sn, st, kcA, vcA, Lmax, rope, sliding, Wn, fb._rope_write_blk, **ak).clone()
                        kcB, vcB = kc0.clone(), vc0.clone()
                        rows = []
                        for t in range(T):
                            st.fill_(start + t)
                            rows.append(fx.exact_attention(ws[1], qkv[t:t + 1].contiguous(), cs[t:t + 1].contiguous(), sn[t:t + 1].contiguous(), st, kcB, vcB, Lmax, rope, sliding, Wn, fb._rope_write_blk, **ak).clone())
                        o1 = torch.cat(rows)
                        eq = bool(torch.equal(ob, o1)) and bool(torch.equal(kcA, kcB)) and bool(torch.equal(vcA, vcB))
                        allok &= eq
                        if not eq or (start in (255, 5000) and T == 13):
                            rec(test="attn_seq_vs_block", variant=str(ak), sliding=sliding, rope=rope, start=start, T=T, bitwise_equal=eq, maxabs=float((ob.float() - o1.float()).abs().max()))
        rec(test="attn_all_bitwise", variant=str(ak), ok=allok)
    for ak in VARIANTS:
        attn_tests(ak)

    # numeric sanity vs torch (block T=5, start 1000)
    for sliding in (False, True):
        T, start = 5, 1000
        kc = torch.randn(1, Hkv, Lmax, D, device=dev).to(bf); vc = torch.randn(1, Hkv, Lmax, D, device=dev).to(bf)
        qkv = torch.randn(T, H * D + 2 * Hkv * D, device=dev).to(bf)
        cs = torch.ones(T, D, device=dev).to(bf); sn = torch.zeros(T, D, device=dev).to(bf)
        st = torch.tensor(start, device=dev, dtype=torch.int64)
        kc2, vc2 = kc.clone(), vc.clone()
        o = fx.exact_attention(ws[T], qkv, cs, sn, st, kc, vc, Lmax, False, sliding, Wn, fb._rope_write_blk).clone()
        q = qkv[:, :H * D].view(T, H, D); kk = qkv[:, H * D:H * D + Hkv * D].view(T, Hkv, D); vv = qkv[:, H * D + Hkv * D:].view(T, Hkv, D)
        for t in range(T):
            kc2[0, :, start + t] = kk[t]; vc2[0, :, start + t] = vv[t]
        ar = torch.arange(Lmax, device=dev)
        refs = []
        for t in range(T):
            m = ar <= start + t
            if sliding: m &= ar > start + t - Wn
            s_ = torch.bmm(q[t].view(Hkv, G, D), kc2[0].transpose(1, 2), out_dtype=torch.float32) * D ** -0.5
            s_ = s_.masked_fill(~m[None, None], float("-inf"))
            refs.append(torch.bmm(torch.softmax(s_, -1).to(bf), vc2[0]).reshape(-1))
        rec(test="attn_numeric", sliding=sliding, rel_err=rel(o, torch.stack(refs)))

    # ---- moe block rows vs single-token calls
    E, Dm, I, k = 128, 2048, 768, 8
    gu = (torch.randn(E, 2 * I, Dm, device=dev) * 0.02).to(bf); dn = (torch.randn(E, Dm, I, device=dev) * 0.02).to(bf)
    okm = True
    for T in (2, 5, 13):
        x = torch.randn(T, Dm, device=dev).to(bf)
        idx = torch.stack([torch.randperm(E, device=dev)[:k] for _ in range(T)]).to(torch.int32)
        if T > 2:
            pool = torch.tensor([e for e in range(E) if e not in idx[0, :4].tolist()], device=dev)
            idx[1] = torch.cat([idx[0, :4].long(), pool[torch.randperm(len(pool), device=dev)[:4]]]).to(torch.int32)  # shared experts, no duplicate slot
        sc = torch.rand(T, k, device=dev).to(bf)
        hb = torch.zeros(T * k, I, device=dev, dtype=bf); ob = torch.zeros(T * k, Dm, device=dev, dtype=bf)
        fb.moe_gemv_blk(x, idx, sc, gu, dn, hb, ob)
        for t in range(T):
            hb1 = torch.zeros(k, I, device=dev, dtype=bf); ob1 = torch.zeros(k, Dm, device=dev, dtype=bf)
            fb.moe_gemv_blk(x[t:t + 1].contiguous(), idx[t:t + 1].contiguous(), sc[t:t + 1].contiguous(), gu, dn, hb1, ob1)
            okm &= bool(torch.equal(ob[t * k:(t + 1) * k], ob1))
    rec(test="moe_blk_rows_vs_T1", ok=okm)

    # ---- resid_norm rows
    Tn = 13
    x = torch.randn(Tn, Dm, device=dev).to(bf); a = torch.randn(Tn, Dm, device=dev).to(bf); m = torch.randn(Tn * k, Dm, device=dev).to(bf); w = (torch.rand(Dm, device=dev) + .5).to(bf)
    xo = torch.zeros(Tn, Dm, device=dev, dtype=bf); ho = torch.zeros_like(xo)
    fk.resid_norm(x, a, m, w, xo, ho, 1e-6)
    okr = True
    for t in range(Tn):
        xo1 = torch.zeros(1, Dm, device=dev, dtype=bf); ho1 = torch.zeros_like(xo1)
        fk.resid_norm(x[t:t + 1].contiguous(), a[t:t + 1].contiguous(), m[t * k:(t + 1) * k].contiguous(), w, xo1, ho1, 1e-6)
        okr &= bool(torch.equal(xo[t], xo1[0])) and bool(torch.equal(ho[t], ho1[0]))
    rec(test="resid_norm_rows", ok=okr)

    # ---- timing of lin kernels vs cuBLAS (memory-bound GEMV class), ms
    def tm(f, n=50):
        for _ in range(5): f()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(n): f()
        torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3
    for (N, K) in ((5120, 2048), (2048, 4096), (262144, 2048)):
        if N * K * 2 > 4e9: continue
        w = (torch.randn(N, K, device=dev) * 0.05).to(bf); x = torch.randn(13, K, device=dev).to(bf)
        x1 = x[:1].contiguous(); x13 = x.contiguous()
        res = dict(test="lin_time_ms", N=N, K=K, cublas_T1=tm(lambda: torch.nn.functional.linear(x1, w)), cublas_T13=tm(lambda: torch.nn.functional.linear(x13, w)))
        for cfg in (dict(BN=16, BK=256, nw=4, ns=3), dict(BN=32, BK=256, nw=4, ns=3), dict(BN=64, BK=128, nw=4, ns=3), dict(BN=64, BK=128, nw=8, ns=4), dict(BN=128, BK=128, nw=8, ns=3), dict(BN=16, BK=128, nw=2, ns=4), dict(BN=32, BK=128, nw=4, ns=4)):
            fx.LIN_CFG[(N, K)] = cfg
            res[json.dumps(cfg)] = (tm(lambda: fx.lin(x1, w)), tm(lambda: fx.lin(x13, w)))
        fx.LIN_CFG.pop((N, K), None)
        rec(**res)
    return out


@app.function(gpu="A10G", timeout=1500)
def run_a10g():
    return body("A10G")


@app.function(gpu="H100!", timeout=1500)
def run_h100():
    return body("H100")


@app.local_entrypoint()
def main(gpu: str = "A10G"):
    res = (run_a10g if gpu == "A10G" else run_h100).remote()
    pathlib.Path("results").mkdir(exist_ok=True)
    pathlib.Path(f"results/exact_unit_{gpu}.jsonl").write_text("\n".join(json.dumps(r) for r in res) + "\n")
