"""Phase 0.2: HF transformers bf16 greedy baseline. Manual prefill/decode loop (not generate) so every
phase is timed; CUDA-synchronised per step. Raw rows -> JSONL."""
import statistics as st
import time

import torch

from common import Jsonl, build_ids, by_split, load_workloads


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


@torch.no_grad()
def hf_cell(m, tok, id_rows, n_dec=48, chunk=2048, warm=4):
    from transformers import DynamicCache
    B, L = len(id_rows), len(id_rows[0])
    x = torch.tensor(id_rows, device="cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()
    cache = DynamicCache(config=m.config)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = None
    for s in range(0, L, chunk):
        out = m(input_ids=x[:, s : s + chunk], past_key_values=cache, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    steps, toks = [], [nxt[:, 0].tolist()]
    for i in range(n_dec):
        t1 = time.perf_counter()
        out = m(input_ids=nxt, past_key_values=cache, use_cache=True, logits_to_keep=1)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        steps.append(time.perf_counter() - t1)
        toks.append(nxt[:, 0].tolist())
    s_ = steps[warm:]
    return dict(batch=B, ctx=L, prefill_s=prefill_s, prefill_tok_per_s=B * L / prefill_s,
                decode_ms_median=1e3 * st.median(s_), decode_ms_mean=1e3 * st.mean(s_),
                decode_ms_p10=1e3 * _pct(s_, .1), decode_ms_p90=1e3 * _pct(s_, .9),
                decode_tok_per_s=B / st.median(s_), n_dec=n_dec,
                peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30, mem_over_weights_gb=(torch.cuda.max_memory_allocated() - base_mem) / 2**30,
                first_tokens=[t[0] for t in toks[:8]])


def run(m, tok, quick=False):
    rows = by_split(load_workloads(), "dev")
    code = [r for r in rows if r["kind"] == "code"]
    j = Jsonl()
    impl = str(m.config._experts_implementation)
    # warmup (kernels / allocator)
    hf_cell(m, tok, [build_ids(tok, code[0], 512)], n_dec=8)
    ctxs = [2048] if quick else [2048, 8192, 32768]
    for ctx in ctxs:
        for B in (1, 2, 4, 8):
            ids = [build_ids(tok, code[b % len(code)], ctx, rotate=b) for b in range(B)]
            try:
                r = hf_cell(m, tok, ids)
                j.add(phase="0.2_baseline", engine="hf_transformers", experts_impl=impl, attn="sdpa", dtype="bf16", workload="code+repo_filler", status="ok", **r)
            except torch.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                j.add(phase="0.2_baseline", engine="hf_transformers", experts_impl=impl, batch=B, ctx=ctx, status="OOM", err=str(e)[:200])
            print(j.lines[-1][:300], flush=True)
    # natural-length prompts per workload kind, batch 1, dev split
    for r in rows:
        ids = [build_ids(tok, r)]
        c = hf_cell(m, tok, ids, n_dec=64)
        j.add(phase="0.2_baseline_natural", engine="hf_transformers", experts_impl=impl, task=r["id"], kind=r["kind"], status="ok", **c)
    # experts implementation comparison (batch 1, 2K and batch 8, 2K)
    other = "eager" if impl != "eager" else "grouped_mm"
    m.config._experts_implementation = other
    try:
        for B in (1, 8):
            ids = [build_ids(tok, code[b % len(code)], 2048, rotate=b) for b in range(B)]
            c = hf_cell(m, tok, ids, n_dec=24)
            j.add(phase="0.2_experts_impl_compare", engine="hf_transformers", experts_impl=other, status="ok", **c)
            print(j.lines[-1][:300], flush=True)
    finally:
        m.config._experts_implementation = impl
    return j
