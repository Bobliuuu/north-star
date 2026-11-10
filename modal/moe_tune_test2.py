"""Sweep 2 (kernel-only, real North dims, random weights): mode-5 MoE tile width tw x BN x warps, and lin_bt row-tile width,
each BITWISE-checked against the original (mode 0 MoE / lin_bt TW=16). modal run modal/moe_tune_test2.py"""
import sys
sys.path.insert(0, "modal")
from nsapp import app


@app.function(gpu="H100", timeout=1800, scaledown_window=2, max_containers=1)
def remote():
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/harness")
    import torch, itertools
    import fast_kernels_batch as fkb
    torch.manual_seed(0); dev = "cuda"
    E, k, Dm, I = 128, 8, 2048, 768
    gu = (torch.randn(E, 2 * I, Dm, device=dev) * 0.02).bfloat16(); dn = (torch.randn(E, Dm, I, device=dev) * 0.02).bfloat16()
    pz = 1.0 / torch.arange(1, E + 1, device=dev).float() ** 0.8; pz = pz[torch.randperm(E, device=dev)]
    route = lambda NT: torch.stack([torch.multinomial(pz, k, replacement=False) for _ in range(NT)]).int()

    def tm(fn, n=30):
        for _ in range(3): fn()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(n): fn()
        b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / n

    rows = []
    var = []
    for tw, bn, nw in itertools.product((16, 32, 64), (32, 64, 128), (2, 4, 8)):
        if bn == 128 and Dm % 128: continue
        var.append(("tw%d_bn%d_w%d" % (tw, bn, nw), tw, bn, nw))
    for NT in (16, 32, 64, 128, 256, 416):
        x = torch.randn(NT, Dm, device=dev).bfloat16(); idx = route(NT); sc = torch.rand(NT, k, device=dev).bfloat16()
        hb = torch.zeros(NT * k, I, device=dev, dtype=torch.bfloat16); fl = torch.zeros(NT * k, device=dev, dtype=torch.int32)
        ob = torch.zeros(NT * k, Dm, device=dev, dtype=torch.bfloat16)
        fkb.TILE_GRID = 0; fkb.moe_bt(x, idx, sc, gu, dn, hb, ob, fl); torch.cuda.synchronize(); base = ob.clone()
        m0 = tm(lambda: fkb.moe_bt(x, idx, sc, gu, dn, hb, ob, fl))
        res = {}
        for name, tw, bn, nw in var:
            try:
                fkb.TILE_GRID = 5; fkb.EG5_CFG.update(nw1=nw, ns1=3, nw2=nw, ns2=3, tw=tw); cfg = dict(BN1=bn, BN2=bn)
                ob.zero_(); fkb.moe_bt(x, idx, sc, gu, dn, hb, ob, fl, cfg); torch.cuda.synchronize()
                res[name] = dict(bitwise=bool(torch.equal(ob, base)), ms=tm(lambda: fkb.moe_bt(x, idx, sc, gu, dn, hb, ob, fl, cfg)))
            except Exception as ex:
                res[name] = dict(bitwise=False, ms=None, err=str(ex)[:160])
        fkb.EG5_CFG.update(nw1=4, ns1=3, nw2=4, ns2=3, tw=16)
        ok = {n: r for n, r in res.items() if r["bitwise"] and r["ms"]}
        best = sorted(ok, key=lambda n: ok[n]["ms"])[:4]
        print("MOE NT", NT, "m0 %.3f" % m0, "best", [(n, round(ok[n]["ms"], 3)) for n in best], "non-bitwise", [n for n, r in res.items() if not r["bitwise"]], flush=True)
        rows.append(dict(kind="moe", NT=NT, m0_ms=m0, res=res))
    # lin_bt row-tile width
    shapes = {"qkv": (5120, 2048), "o_proj": (2048, 4096), "router": (128, 2048), "lm_head": (262144, 2048)}
    W = {n: (torch.randn(N, K, device=dev) * 0.02).bfloat16() for n, (N, K) in shapes.items()}
    for NT in (16, 32, 64, 128, 256, 416):
        for n, (N, K) in shapes.items():
            x = torch.randn(NT, K, device=dev).bfloat16()
            fkb.LIN_TW, fkb.LIN_TW_MIN_NT = 16, 1; base = fkb.lin_bt(x, W[n]).clone(); t16 = tm(lambda: fkb.lin_bt(x, W[n]), n=10 if n == "lm_head" else 30)
            r = dict(kind="lin", NT=NT, name=n, ms16=t16)
            for tw in (32, 64, 128):
                try:
                    fkb.LIN_TW = tw; o = fkb.lin_bt(x, W[n]); torch.cuda.synchronize()
                    r["bitwise%d" % tw] = bool(torch.equal(o, base)); r["ms%d" % tw] = tm(lambda: fkb.lin_bt(x, W[n]), n=10 if n == "lm_head" else 30)
                except Exception as ex:
                    r["bitwise%d" % tw] = False; r["err%d" % tw] = str(ex)[:160]
            fkb.LIN_TW, fkb.LIN_TW_MIN_NT = 16, 64
            print("LIN NT", NT, n, {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in r.items() if kk not in ("kind", "NT", "name")}, flush=True)
            rows.append(r)
    return rows


@app.local_entrypoint()
def main():
    import json, pathlib
    rows = remote.remote()
    pathlib.Path("results/moe_tune_kernel2.json").write_text(json.dumps(rows, indent=1))
    print("WROTE results/moe_tune_kernel2.json")
