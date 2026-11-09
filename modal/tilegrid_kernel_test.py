"""Kernel-only GPU test (random weights, no model): fkb.moe_bt serial vs per-tile grid, bitwise, plus a loose fp32 torch reference.
modal run modal/tilegrid_kernel_test.py"""
import sys
sys.path.insert(0, "modal")
from nsapp import app


@app.function(gpu="H100", timeout=900, scaledown_window=2, max_containers=1)
def remote():
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/harness")
    import torch, time
    import fast_kernels_batch as fkb
    torch.manual_seed(0); dev = "cuda"
    E, k, Dm, I = int(__import__('os').environ.get('KT_E', '128')), 8, 2048, 512
    gu = (torch.randn(E, 2 * I, Dm, device=dev) * 0.05).bfloat16(); dn = (torch.randn(E, Dm, I, device=dev) * 0.05).bfloat16()
    out = []
    for NT in (1, 16, 17, 32, 64, 128, 208, 416):
        x = torch.randn(NT, Dm, device=dev).bfloat16()
        idx = torch.stack([torch.randperm(E, device=dev)[:k] for _ in range(NT)]).int(); sc = torch.rand(NT, k, device=dev).bfloat16()
        r = {}
        for mode in (0, 1, 2, 3):
            fkb.TILE_GRID = mode
            hb = torch.zeros(NT * k, I, device=dev, dtype=torch.bfloat16); ob = torch.zeros(NT * k, Dm, device=dev, dtype=torch.bfloat16)
            fl = torch.zeros(NT * k, device=dev, dtype=torch.int32)
            fkb.moe_bt(x, idx, sc, gu, dn, hb, ob, fl); torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(20): fkb.moe_bt(x, idx, sc, gu, dn, hb, ob, fl)
            torch.cuda.synchronize(); r[mode] = (ob.clone(), (time.perf_counter() - t) / 20 * 1e3)
        # reference
        xf = x.float(); ref = torch.zeros(NT * k, Dm, device=dev)
        for t_ in range(NT):
            for j in range(k):
                e = int(idx[t_, j]); w = gu[e].float(); g = w[:I] @ xf[t_]; u = w[I:] @ xf[t_]
                ref[t_ * k + j] = (dn[e].float() @ (torch.nn.functional.silu(g) * u)) * float(sc[t_, j])
        err = float((r[3][0].float() - ref).abs().max()); scale = float(ref.abs().max())
        row = dict(NT=NT, bitwise=bool(all(torch.equal(r[0][0], r[q][0]) for q in (1, 2, 3))), ms_serial=round(r[0][1], 4), ms_tile=round(r[1][1], 4), ms_egrid=round(r[2][1], 4), ms_gather=round(r[3][1], 4), err=round(err, 4), ref_max=round(scale, 3))
        print(row, flush=True); out.append(row)
    # mode 3 must be CUDA-graph capturable (no host sync)
    fkb.TILE_GRID = 3; NT = 64
    x = torch.randn(NT, Dm, device=dev).bfloat16(); idx = torch.stack([torch.randperm(E, device=dev)[:k] for _ in range(NT)]).int(); sc = torch.rand(NT, k, device=dev).bfloat16()
    hb = torch.zeros(NT * k, I, device=dev, dtype=torch.bfloat16); ob = torch.zeros(NT * k, Dm, device=dev, dtype=torch.bfloat16); fl = torch.zeros(NT * k, device=dev, dtype=torch.int32)
    s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_):
        fkb.moe_bt(x, idx, sc, gu, dn, hb, ob, fl)
    torch.cuda.current_stream().wait_stream(s_); want = ob.clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fkb.moe_bt(x, idx, sc, gu, dn, hb, ob, fl)
    ob.zero_(); g.replay(); torch.cuda.synchronize()
    idx.copy_(torch.stack([torch.randperm(E, device=dev)[:k] for _ in range(NT)]).int()); g.replay(); torch.cuda.synchronize(); got2 = ob.clone()
    fkb.TILE_GRID = 0; fkb.moe_bt(x, idx, sc, gu, dn, hb, ob, fl); torch.cuda.synchronize()
    cap = dict(graph_capture_ok=True, graph_replay_new_routing_bitwise=bool(torch.equal(got2, ob)))
    print(cap, flush=True); out.append(cap)
    return out


@app.local_entrypoint()
def main():
    rows = remote.remote()
    ab = [r for r in rows if "NT" in r]; ok = all(r["bitwise"] for r in ab) and all(r["err"] < 0.05 * r["ref_max"] + 1e-2 for r in ab) and rows[-1]["graph_replay_new_routing_bitwise"]
    print("KERNEL_TEST", "PASS" if ok else "FAIL")
