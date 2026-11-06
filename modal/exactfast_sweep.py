"""exact-fast kernel sweep on H100 (no weights): lin + moe_blk tile configs, time (per layer, 48 rotating weight copies) and bitwise equality vs default cfg.
modal run --detach modal/exactfast_sweep.py::main --tag sw1"""
import sys, json, pathlib, itertools
import modal

image = (modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.10.0", "numpy==2.2.5").add_local_dir("harness", "/root/harness", copy=False))
app = modal.App("northstar-exactfast-sweep", image=image)


@app.function(gpu="H100", timeout=3000, image=image)
def body(which: str):
    sys.path.insert(0, "/root/harness")
    import torch, triton, time
    import fast_kernels as fk, fast_kernels_blk as fb, fast_kernels_exact as fx
    dev, bf = "cuda", torch.bfloat16
    out = []
    rec = lambda **kw: (out.append(kw), print(json.dumps(kw), flush=True))
    rec(triton=triton.__version__, torch=torch.__version__, gpu=torch.cuda.get_device_name())
    torch.manual_seed(0)

    def tm(f, n=48, rep=3):
        best = 1e9
        for r in range(rep):
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for i in range(n):
                    f(i)
            g.replay(); torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
            best = min(best, e0.elapsed_time(e1) / n)
        return best * 1e3  # us per call

    if "l" in which:
        for name, N, K, nw_ in (("qkv", 5120, 2048, 48), ("o", 2048, 4096, 48), ("router", 128, 2048, 48), ("lm_head", 262144, 2048, 1)):
            ws = [(torch.randn(N, K, device=dev) * 0.05).to(bf) for _ in range(nw_)]
            x = torch.randn(16, K, device=dev).to(bf)
            x1, x13 = x[:1].contiguous(), x[:13].contiguous()
            base = fx.lin(x13, ws[0]); base1 = fx.lin(x1, ws[0])
            res = []
            for BN, BK, nw, ns in itertools.product((16, 32, 64, 128), (64, 128, 256, 512), (2, 4, 8), (2, 3, 4, 5)):
                if BN * BK * 2 * ns > 200000 or (N < BN):
                    continue
                fx.LIN_CFG[(N, K)] = dict(BN=BN, BK=BK, nw=nw, ns=ns)
                try:
                    o1 = fx.lin(x1, ws[0]); o13 = fx.lin(x13, ws[0])
                    eq = bool(torch.equal(o1[0], o13[0])) and bool(torch.equal(o13, base)) and bool(torch.equal(o1, base1))
                    t1 = tm(lambda i: fx.lin(x1, ws[i % nw_]))
                    t13 = tm(lambda i: fx.lin(x13, ws[i % nw_])) if name != "lm_head" else 0
                    res.append((t1, dict(name=name, BN=BN, BK=BK, nw=nw, ns=ns, us_T1=t1, us_T13=t13, bitwise_same_as_default=eq)))
                except Exception as e:
                    pass
            fx.LIN_CFG.pop((N, K), None)
            d = fx.lin_cfg(N, K)
            dt = tm(lambda i: fx.lin(x1, ws[i % nw_]))
            rec(kind="lin_default", name=name, cfg=d, us_T1=dt)
            res.sort(key=lambda r: r[0])
            for t, r in res[:12]:
                rec(kind="lin", **r)
            del ws
    if "m" in which:
        E, Dm, I, k = 128, 2048, 768, 8
        gu = (torch.randn(E, 2 * I, Dm, device=dev) * 0.02).to(bf); dn = (torch.randn(E, Dm, I, device=dev) * 0.02).to(bf)
        for T in (1, 13):
            x = torch.randn(T, Dm, device=dev).to(bf)
            idxs = [torch.stack([torch.randperm(E, device=dev)[:k] for _ in range(T)]).to(torch.int32) for _ in range(48)]
            sc = torch.rand(T, k, device=dev).to(bf)
            hb = torch.zeros(T * k, I, device=dev, dtype=bf); ob = torch.zeros(T * k, Dm, device=dev, dtype=bf)
            fb.moe_gemv_blk(x, idxs[0], sc, gu, dn, hb, ob); base = ob.clone(); baseh = hb.clone()
            res = []
            for BN1, BK1, nw1, ns1, BN2, BK2, nw2, ns2 in itertools.product((16, 32, 64), (64, 128, 256), (2, 4, 8), (2, 3, 4), (16, 32, 64), (64, 128, 256), (2, 4, 8), (2, 3, 4)):
                pass
            # sweep gateup and down independently (they are separate kernels): time each via full call with the other at default
            for stage, grid in (("gu", itertools.product((16, 32, 64), (64, 128, 256), (2, 4, 8), (2, 3, 4, 5))), ("dn", itertools.product((16, 32, 64), (64, 128, 256), (2, 4, 8), (2, 3, 4, 5)))):
                rr = []
                for BN, BK, nw, ns in grid:
                    cfg = dict(BN1=BN, BK1=BK, nw1=nw, ns1=ns) if stage == "gu" else dict(BN2=BN, BK2=BK, nw2=nw, ns2=ns)
                    try:
                        fb.moe_gemv_blk(x, idxs[0], sc, gu, dn, hb, ob, cfg)
                        eq = bool(torch.equal(ob, base)) and bool(torch.equal(hb, baseh))
                        # time only this stage: run whole call, subtract nothing (stage cfg dominates differences)
                        t = tm(lambda i: fb.moe_gemv_blk(x, idxs[i], sc, gu, dn, hb, ob, cfg), rep=2)
                        rr.append((t, dict(kind="moe", T=T, stage=stage, BN=BN, BK=BK, nw=nw, ns=ns, us_total=t, bitwise_same_as_default=eq)))
                    except Exception as e:
                        pass
                rr.sort(key=lambda r: r[0])
                dflt = tm(lambda i: fb.moe_gemv_blk(x, idxs[i], sc, gu, dn, hb, ob), rep=2)
                rec(kind="moe_default", T=T, stage=stage, us_total=dflt)
                for t, r in rr[:10]:
                    rec(**r)
    return out


@app.local_entrypoint()
def main(tag: str = "sw1", which: str = "lm"):
    r = body.remote(which)
    pathlib.Path("results").mkdir(exist_ok=True)
    pathlib.Path(f"results/exactfast_sweep_{tag}.jsonl").write_text("\n".join(json.dumps(x) for x in r) + "\n")
