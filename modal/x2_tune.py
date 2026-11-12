"""H100 tile sweep for the exact-v2 kernels (no model weights: random tensors at North's real dims).
    modal run modal/x2_tune.py --part lin|moe|all --tag t1
Each config is checked for (a) row invariance: every row of a T=13 call bitwise equals the same row computed alone (T=1), and
(b) closeness to an fp32 torch reference. Timing: one CUDA graph of R sequential calls cycling over distinct weight buffers
(> L2), so it is HBM-bound like the real decode chain. Rows -> results/x2_tune_<tag>.jsonl (written locally by the entrypoint)."""
import sys, time, json, pathlib, itertools, traceback
import modal

image = (modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.10.0", "numpy==2.2.5")
         .env({"PYTHONUNBUFFERED": "1"})
         .add_local_dir("harness", "/root/harness", copy=False))
app = modal.App("northstar-x2-tune", image=image)

LIN_SHAPES = [(5120, 2048), (5248, 2048), (2048, 4096), (128, 2048), (262144, 2048)]


def _graph_ms(fn, R, reps=10):
    import torch
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            for i in range(R): fn(i)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(R): fn(i)
    g.replay(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / (reps * R) * 1e3


def lin_part(rec):
    import torch
    import fast_kernels_exact as fx, fast_kernels_x2 as f2
    dev, bf = "cuda", torch.bfloat16
    torch.manual_seed(0)
    for (N, K) in LIN_SHAPES:
        nb = max(1, int(3e8 // (N * K * 2)))
        ws = [(torch.randn(N, K, device=dev) * 0.03).to(bf) for _ in range(min(nb, 16))]
        R = 48 if N * K < 1e8 else 4
        x13 = torch.randn(13, K, device=dev).to(bf)
        x1 = x13[:1].contiguous(); x7 = x13[7:8].contiguous()
        ref = (x13.float() @ ws[0].float().t())
        outs = {T: torch.empty(T, N, device=dev, dtype=bf) for T in (1, 13)}
        base = dict(fx.lin_cfg(N, K), SK=1)
        cands = [base]
        for BN, BK, nw, ns in itertools.product((16, 32, 64, 128), (64, 128, 256, 512), (4, 8), (2, 3, 4, 6)):
            if BN * BK * 2 * ns > 200_000 or BK > K:
                continue
            cands.append(dict(BN=BN, BK=BK, nw=nw, ns=ns, SK=1))
        res1 = []
        for c in cands:
            try:
                t1 = _graph_ms(lambda i: f2.lin_x2(x1, ws[i % len(ws)], outs[1], c), R)
                res1.append((t1, c))
            except Exception as e:
                rec(part="lin", N=N, K=K, cfg=c, err=str(e)[:200])
        res1.sort(key=lambda r: r[0])
        top = [c for _, c in res1[:6]]
        for c in list(top):
            for sk in (2, 4, 8):
                if K % (sk * c["BK"]) == 0:
                    top.append(dict(c, SK=sk))
        for c in [base] + top:
            try:
                t1 = _graph_ms(lambda i: f2.lin_x2(x1, ws[i % len(ws)], outs[1], c), R)
                t13 = _graph_ms(lambda i: f2.lin_x2(x13, ws[i % len(ws)], outs[13], c), R)
                o13 = f2.lin_x2(x13, ws[0], None, c); o1 = f2.lin_x2(x1, ws[0], None, c); o7 = f2.lin_x2(x7, ws[0], None, c)
                inv = bool(torch.equal(o13[0], o1[0]) and torch.equal(o13[7], o7[0]))
                err = ((o13.float() - ref).abs().max() / ref.abs().max()).item()
                gbs = N * K * 2 / (t1 * 1e-3) / 1e9
                rec(part="lin", N=N, K=K, cfg=c, base=c is base, ms_T1=t1, ms_T13=t13, GBps_T1=gbs, invariant=inv, rel_err=err)
                print(f"lin {N}x{K} {c} T1 {t1*1e3:.1f}us T13 {t13*1e3:.1f}us {gbs:.0f}GB/s inv={inv} err={err:.1e}", flush=True)
            except Exception as e:
                rec(part="lin", N=N, K=K, cfg=c, err=str(e)[:200])
        del ws
        torch.cuda.empty_cache()


def moe_part(rec):
    import torch
    import fast_kernels_blk as fb
    dev, bf = "cuda", torch.bfloat16
    torch.manual_seed(0)
    E, Dm, I, k = 128, 2048, 768, 8
    NL = 4
    gus = [(torch.randn(E, 2 * I, Dm, device=dev) * 0.02).to(bf) for _ in range(NL)]
    dns = [(torch.randn(E, Dm, I, device=dev) * 0.02).to(bf) for _ in range(NL)]
    R = 48

    def mk_idx(T):
        # realistic block routing: consecutive tokens share ~half their experts
        rows = []
        prev = torch.randperm(E)[:k]
        for t in range(T):
            keep = prev[:4]
            pool = torch.tensor([e for e in range(E) if e not in keep.tolist()])
            cur = torch.cat([keep, pool[torch.randperm(len(pool))[:4]]])
            rows.append(cur[torch.randperm(k)]); prev = cur
        return torch.stack(rows).to(torch.int32).to(dev)

    for T in (1, 13):
        idxs = [mk_idx(T) for _ in range(R)]
        x = torch.randn(T, Dm, device=dev).to(bf)
        sc = torch.rand(T, k, device=dev).to(bf)
        hb = torch.zeros(T * k, I, device=dev, dtype=bf); ob = torch.zeros(T * k, Dm, device=dev, dtype=bf)
        base = dict(BN1=16, BK1=128, BN2=16, BK2=128, nw1=4, nw2=4)
        # stage 1: gate_up tiles with the default down; stage 2: down tiles with the best gate_up
        g_c = [dict(base, BN1=bn, BK1=bk, nw1=nw, ns1=ns) for bn, bk, nw, ns in itertools.product((16, 32, 64), (64, 128, 256, 512), (2, 4, 8), (2, 3, 4))
               if bn * bk * 2 * 2 * ns <= 200_000]
        res = []
        for c in [base] + g_c:
            try:
                ms = _graph_ms(lambda i: fb.moe_gemv_blk(x, idxs[i], sc, gus[i % NL], dns[i % NL], hb, ob, c), R)
                res.append((ms, c))
            except Exception as e:
                rec(part="moe", T=T, cfg=c, err=str(e)[:200])
        res.sort(key=lambda r: r[0])
        best_g = res[0][1]
        for ms, c in res[:8]:
            rec(part="moe", stage="gateup", T=T, cfg=c, ms=ms); print("moe gu T", T, c, f"{ms*1e3:.1f}us", flush=True)
        d_c = [dict(best_g, BN2=bn, BK2=bk, nw2=nw, ns2=ns) for bn, bk, nw, ns in itertools.product((16, 32, 64, 128), (64, 128, 256), (2, 4, 8), (2, 3, 4))
               if bn * bk * 2 * ns <= 200_000 and bk <= I]
        res2 = []
        for c in d_c:
            try:
                res2.append((_graph_ms(lambda i: fb.moe_gemv_blk(x, idxs[i], sc, gus[i % NL], dns[i % NL], hb, ob, c), R), c))
            except Exception as e:
                rec(part="moe", T=T, cfg=c, err=str(e)[:200])
        res2.sort(key=lambda r: r[0])
        base_ms = _graph_ms(lambda i: fb.moe_gemv_blk(x, idxs[i], sc, gus[i % NL], dns[i % NL], hb, ob, base), R)
        for ms, c in res2[:8]:
            # invariance of the chosen config: rows of the T block == the same token alone
            inv = None
            if T > 1:
                fb.moe_gemv_blk(x, idxs[0], sc, gus[0], dns[0], hb, ob, c)
                inv = True
                for t in (0, 5, 12):
                    hb1 = torch.zeros(k, I, device=dev, dtype=bf); ob1 = torch.zeros(k, Dm, device=dev, dtype=bf)
                    fb.moe_gemv_blk(x[t:t + 1].contiguous(), idxs[0][t:t + 1].contiguous(), sc[t:t + 1].contiguous(), gus[0], dns[0], hb1, ob1, c)
                    inv &= bool(torch.equal(ob[t * k:(t + 1) * k], ob1))
            rec(part="moe", stage="down", T=T, cfg=c, ms=ms, base_ms=base_ms, invariant=inv)
            print("moe dn T", T, c, f"{ms*1e3:.1f}us (base {base_ms*1e3:.1f}us) inv={inv}", flush=True)


@app.function(gpu="H100", timeout=3000)
def run(part: str):
    sys.path.insert(0, "/root/harness")
    import torch
    out = []

    def rec(**kw):
        out.append(kw)
    meta = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__)
    import triton
    meta["triton"] = triton.__version__
    rec(part="meta", **meta)
    t0 = time.time()
    try:
        if part in ("lin", "all"):
            lin_part(rec)
        if part in ("moe", "all"):
            moe_part(rec)
    except Exception:
        rec(part="fatal", err=traceback.format_exc()[-1500:])
        print(traceback.format_exc(), flush=True)
    rec(part="done", secs=time.time() - t0)
    return out


@app.local_entrypoint()
def main(part: str = "all", tag: str = "t1"):
    res = run.remote(part)
    p = pathlib.Path(f"results/x2_tune_{tag}.jsonl")
    p.write_text("\n".join(json.dumps(r) for r in res) + "\n")
    print("wrote", p, len(res), "rows")
