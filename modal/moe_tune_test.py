"""Kernel-only MoE tuning at real North dims (E=128 k=8 Dm=2048 I=768), random weights, Zipf-skewed routing.
Every variant must be BITWISE equal to mode 0 (serial). modal run modal/moe_tune_test.py"""
import sys
sys.path.insert(0, "modal")
from nsapp import app


@app.function(gpu="H100", timeout=1500, scaledown_window=2, max_containers=1)
def remote():
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/harness")
    import torch, itertools
    import fast_kernels_batch as fkb
    torch.manual_seed(0); dev = "cuda"
    E, k, Dm, I = 128, 8, 2048, 768
    gu = (torch.randn(E, 2 * I, Dm, device=dev) * 0.02).bfloat16(); dn = (torch.randn(E, Dm, I, device=dev) * 0.02).bfloat16()
    pz = 1.0 / torch.arange(1, E + 1, device=dev).float() ** 0.8; pz = pz[torch.randperm(E, device=dev)]

    def route(NT):
        return torch.stack([torch.multinomial(pz, k, replacement=False) for _ in range(NT)]).int()

    def tm(fn, n=30):
        for _ in range(3): fn()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(n): fn()
        b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / n

    rows = []
    variants = [("m0", 0, None, None), ("m3", 3, None, None), ("m5", 5, None, dict(nw1=4, ns1=3, nw2=4, ns2=3))]
    for nw1, ns1, nw2, ns2 in itertools.product((1, 2, 4), (2, 3, 4), (1, 2, 4), (2, 3, 4)):
        if (nw1, ns1) == (nw2, ns2) and not (nw1 == 4 and ns1 == 3):
            variants.append(("m5_w%ds%d_w%ds%d" % (nw1, ns1, nw2, ns2), 5, None, dict(nw1=nw1, ns1=ns1, nw2=nw2, ns2=ns2)))
    for bn1, bn2 in ((32, 16), (16, 32), (32, 32), (64, 64)):
        variants.append(("m5_bn%d_%d" % (bn1, bn2), 5, dict(BN1=bn1, BN2=bn2), dict(nw1=4, ns1=3, nw2=4, ns2=3)))
    for NT in (16, 32, 64, 104, 128, 256, 416):
        x = torch.randn(NT, Dm, device=dev).bfloat16(); idx = route(NT); sc = torch.rand(NT, k, device=dev).bfloat16()
        hb = torch.zeros(NT * k, I, device=dev, dtype=torch.bfloat16); fl = torch.zeros(NT * k, device=dev, dtype=torch.int32)
        res = {}
        for name, mode, cfg, g in variants:
            try:
                fkb.TILE_GRID = mode
                if g: fkb.EG5_CFG.update(g)
                ob = torch.zeros(NT * k, Dm, device=dev, dtype=torch.bfloat16)
                f = lambda: fkb.moe_bt(x, idx, sc, gu, dn, hb, ob, fl, cfg)
                f(); torch.cuda.synchronize(); out = ob.clone()
                res[name] = dict(ms=tm(f), bitwise=None, out=out)
            except Exception as ex:
                res[name] = dict(ms=None, bitwise=False, err=str(ex)[:200], out=None)
        base = res["m0"]["out"]
        for name, r in res.items():
            if r["out"] is not None: r["bitwise"] = bool(torch.equal(r["out"], base))
            r.pop("out")
        ne = int(torch.unique(idx).numel())
        row = dict(NT=NT, distinct_experts=ne, res=res); rows.append(row)
        ok = {n: r for n, r in res.items() if r["bitwise"] and r["ms"]}
        best = min(ok, key=lambda n: ok[n]["ms"])
        print("NT", NT, "experts", ne, "m0 %.3f m3 %.3f m5 %.3f best %s %.3f" % (res["m0"]["ms"], res["m3"]["ms"], res["m5"]["ms"] or -1, best, ok[best]["ms"]),
              "non-bitwise:", [(n, r.get("err", "")[:80]) for n, r in res.items() if not r["bitwise"]], flush=True)
    # graph capture of mode 5 with new routing at replay
    fkb.TILE_GRID = 5; fkb.EG5_CFG.update(nw1=4, ns1=3, nw2=4, ns2=3); NT = 128
    x = torch.randn(NT, Dm, device=dev).bfloat16(); idx = route(NT); sc = torch.rand(NT, k, device=dev).bfloat16()
    hb = torch.zeros(NT * k, I, device=dev, dtype=torch.bfloat16); ob = torch.zeros(NT * k, Dm, device=dev, dtype=torch.bfloat16); fl = torch.zeros(NT * k, device=dev, dtype=torch.int32)
    s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_): fkb.moe_bt(x, idx, sc, gu, dn, hb, ob, fl)
    torch.cuda.current_stream().wait_stream(s_)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr): fkb.moe_bt(x, idx, sc, gu, dn, hb, ob, fl)
    idx.copy_(route(NT)); ob.zero_(); gr.replay(); torch.cuda.synchronize(); got = ob.clone()
    fkb.TILE_GRID = 0; fkb.moe_bt(x, idx, sc, gu, dn, hb, ob, fl); torch.cuda.synchronize()
    cap = dict(graph_replay_new_routing_bitwise=bool(torch.equal(got, ob))); print(cap, flush=True); rows.append(cap)
    return rows


@app.local_entrypoint()
def main():
    import json, pathlib
    rows = remote.remote()
    pathlib.Path("results/moe_tune_kernel.json").write_text(json.dumps(rows, indent=1))
    print("WROTE results/moe_tune_kernel.json")
