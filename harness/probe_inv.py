"""Row-invariance probe for batched prefill components on real North weights (layer 1): is row r of f(X_batch) bitwise equal to
row r of f(X_alone) when X_alone (one 300-row prompt) sits inside an 8192-row batch? Also times each f at 8192 rows."""
import time, traceback
import torch
import torch.nn.functional as F
from common import Jsonl
import fast_kernels as fk
import fast_kernels_batch as fkb


def _t(fn, n=5):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


@torch.no_grad()
def run(m, tok, N=8192, L=300, off=4100):
    j = Jsonl()
    try:
        layer = m.model.layers[1]; at = layer.self_attn; mlp = layer.mlp
        torch.manual_seed(0)
        ids = torch.randint(1000, 200000, (1, N), device="cuda")
        X = layer.input_layernorm(m.model.embed_tokens(ids))[0].contiguous()      # [N, D] realistic-scale rows
        A = X[off:off + L].contiguous()
        wqkv = torch.cat([at.q_proj.weight, at.k_proj.weight, at.v_proj.weight], 0).contiguous()
        k = m.config.num_experts_per_tok

        def rec(name, fb, fa, fn_time):
            ob, oa = fb(), fa()
            eq = bool(torch.equal(ob[off:off + L], oa))
            mx = float((ob[off:off + L].float() - oa.float()).abs().max())
            ms = _t(fn_time)
            j.add(phase="probe", comp=name, invariant=eq, maxabs=mx, ms_at_N=ms, N=N)
            print("PROBE %-28s invariant=%s maxabs=%.3g  %.2f ms @ %d rows" % (name, eq, mx, ms, N), flush=True)

        rec("cublas F.linear qkv", lambda: F.linear(X, wqkv), lambda: F.linear(A, wqkv), lambda: F.linear(X, wqkv))
        rec("triton lin_bt qkv", lambda: fkb.lin_bt(X, wqkv), lambda: fkb.lin_bt(A, wqkv), lambda: fkb.lin_bt(X, wqkv))
        rec("cublas o_proj", lambda: F.linear(torch.cat([X, X], 1), at.o_proj.weight), lambda: F.linear(torch.cat([A, A], 1), at.o_proj.weight),
            lambda: F.linear(torch.cat([X, X], 1), at.o_proj.weight))
        rec("cublas router", lambda: F.linear(X, mlp.gate.weight), lambda: F.linear(A, mlp.gate.weight), lambda: F.linear(X, mlp.gate.weight))

        def hf_moe(Z):
            _, sc_, ix_ = mlp.gate(Z)
            return mlp.experts(Z, ix_, sc_)
        rec("HF gate+grouped_mm experts", lambda: hf_moe(X), lambda: hf_moe(A), lambda: hf_moe(X))

        def our_moe(Z, tw):
            n = Z.shape[0]
            lg = fkb.lin_bt(Z, mlp.gate.weight)
            idx = torch.zeros(n, k, dtype=torch.int32, device="cuda"); sc = torch.zeros(n, k, dtype=torch.bfloat16, device="cuda")
            fk.router_topk(lg, idx, sc, k)
            hb = torch.zeros(n * k, mlp.experts.down_proj.shape[-1], dtype=torch.bfloat16, device="cuda")
            ob = torch.zeros(n * k, Z.shape[1], dtype=torch.bfloat16, device="cuda")
            old = dict(fkb.EG5_CFG); fkb.EG5_CFG["tw"] = tw
            try:
                fkb.moe_eg5(Z, idx, sc, mlp.experts.gate_up_proj, mlp.experts.down_proj, hb, ob, dict(BN1=64, BK1=128, BN2=64, BK2=128))
            finally:
                fkb.EG5_CFG.clear(); fkb.EG5_CFG.update(old)
            return ob.view(n, k, -1).float().sum(1)   # sum over slots (fixed order) just for comparison
        for tw in (16, 32, 64):
            rec("triton eg5 MoE tw=%d" % tw, lambda tw=tw: our_moe(X, tw), lambda tw=tw: our_moe(A, tw), lambda tw=tw: our_moe(X, tw))
        a16, a64 = our_moe(A, 16), our_moe(A, 64)
        j.add(phase="probe", comp="eg5 tw64 vs tw16 (alone)", invariant=bool(torch.equal(a16, a64)))
        print("PROBE eg5 tw64 == tw16:", bool(torch.equal(a16, a64)), flush=True)
    except Exception:
        j.add(phase="fatal", err=traceback.format_exc()[-1500:]); print("FATAL", traceback.format_exc()[-1500:], flush=True)
    return j
