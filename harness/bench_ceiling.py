"""Phase 0.3: teacher-forced block-verification ceiling + per-component timing.
Teacher tokens = North's own greedy continuation (real tokens). For block size K we time
  (a) ONE forward over K real tokens (what a verifier does) vs (b) K sequential single-token forwards,
at the same cache state; cache.len is reset after each measurement (rollback)."""
import statistics as st
import time

import torch

from common import Jsonl, build_ids, by_split, load_workloads
from northstar.refmodel import FULL, DraftMode, RefNorth, SectionTimer


def _t(fn, reps, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    return st.median(ts), min(ts), max(ts)


@torch.no_grad()
def greedy_batch(ref, ids, n, max_extra=0):
    """Batched plain greedy; returns (tokens [B,n], cache with len = P-1+n... ) not needed: tokens only."""
    B, P = ids.shape
    cache = ref.new_cache(B, P + n + 2)
    ref.prefill(ids[:, : P - 1], cache, logits="none")
    cur, out = ids[:, -1:], []
    for _ in range(n):
        lg = ref.forward(cur, cache, cache.len, FULL, logits="last")
        cache.len += 1
        cur = lg[:, -1].argmax(-1, keepdim=True)
        out.append(cur)
    return torch.cat(out, 1)


def expert_bytes(m):
    cfg = m.config
    per_expert = 3 * cfg.hidden_size * cfg.intermediate_size * 2
    n_moe = sum(1 for t in cfg.mlp_layer_types if t == "sparse")
    tot = sum(p.numel() * p.element_size() for p in m.parameters())
    expert_total = n_moe * cfg.num_experts * per_expert
    return dict(per_expert=per_expert, n_moe=n_moe, total_params_bytes=tot, expert_total_bytes=expert_total,
                nonexpert_bytes=tot - expert_total)


@torch.no_grad()
def run(m, tok, quick=False):
    ref = RefNorth(m)
    j = Jsonl()
    eb = expert_bytes(m)
    j.add(phase="0.3_model_bytes", **eb, hbm_gbps_a100_80_peak=2039)
    code = [r for r in by_split(load_workloads(), "dev") if r["kind"] == "code"]
    cells = [(1, 2048), (4, 2048)] if quick else [(1, 2048), (1, 8192), (4, 2048)]
    KS = list(range(1, 17))
    NGEN = 40
    for B, ctx in cells:
        ids = torch.tensor([build_ids(tok, code[b % len(code)], ctx, rotate=b) for b in range(B)], device="cuda")
        P = ids.shape[1]
        gen = greedy_batch(ref, ids, NGEN)  # real tokens
        tf = torch.cat([ids, gen], 1)       # teacher-forced full sequence
        cache = ref.new_cache(B, P + NGEN + 4)
        ref.prefill(tf[:, : P - 1], cache, logits="none")
        base = cache.len  # P-1
        for off in (0, 12):
            # advance cache to base+off using real tokens
            if off:
                ref.forward(tf[:, base : base + off], cache, base, logits="none")
                cache.len = base + off
            s0 = cache.len
            res = {}
            for K in KS:
                blk = tf[:, s0 : s0 + K]

                def block():
                    ref.forward(blk, cache, s0, FULL, logits="all")

                def single():
                    for i in range(K):
                        ref.forward(tf[:, s0 + i : s0 + i + 1], cache, s0 + i, FULL, logits="last")

                bm, bmin, bmax = _t(block, 7)
                sm, smin, smax = _t(single, 3 if K > 1 else 7)
                res[K] = (bm, sm)
                # union of experts across the K real tokens (layer-mean), native top-8 and top-r
                ref.record_experts = []
                ref.forward(blk, cache, s0, FULL, logits="none")
                rec, ref.record_experts = ref.record_experts, None
                un = {}
                for r in (8, 4, 2):
                    un[f"union_r{r}"] = st.mean(len(torch.unique(idx[:, :r])) for _, idx in rec) if rec else 0
                j.add(phase="0.3_block_verify", batch=B, ctx=P, offset=off, K=K, block_ms=bm, block_ms_min=bmin, block_ms_max=bmax,
                      seq_ms=sm, seq_ms_min=smin, seq_ms_max=smax, verify_only_speedup=sm / bm, **un)
            print("ceiling", B, ctx, off, {k: round(v[0], 1) for k, v in res.items() if k in (1, 4, 8, 16)}, flush=True)
            cache.len = s0
        if B != 1 and ctx != 2048:
            continue
        # component timing (event timeline) + profiler (device-kernel busy time) at K in (1, 8, 16)
        s0 = base
        for K in (1, 8, 16):
            blk = tf[:, s0 : s0 + K]
            ref.timer = SectionTimer()
            ref.forward(blk, cache, s0, FULL, logits="last")
            ref.timer.collect()
            torch.cuda.synchronize()
            t = time.perf_counter()
            ref.forward(blk, cache, s0, FULL, logits="last")
            ref.timer.pending
            sec = ref.timer.collect()
            wall = (time.perf_counter() - t) * 1e3
            ref.timer = None
            j.add(phase="0.3_components_events", batch=B, ctx=P, K=K, wall_ms=wall, sections_ms=sec)
            # profiler
            from torch.profiler import ProfilerActivity, profile
            ref.timer = SectionTimer(use_profiler=True)
            for _ in range(2):
                ref.forward(blk, cache, s0, FULL, logits="last")
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                ref.forward(blk, cache, s0, FULL, logits="last")
                torch.cuda.synchronize()
            ref.timer = None
            ka = prof.key_averages()
            def dev(e):
                return getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
            def devtot(e):
                return getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)
            kern = sorted(((dev(e) / 1e3, e.key, e.count) for e in ka if dev(e) > 0), reverse=True)
            busy = sum(k[0] for k in kern)
            secs = {e.key[5:]: devtot(e) / 1e3 for e in ka if e.key.startswith("sec::")}
            j.add(phase="0.3_components_profiler", batch=B, ctx=P, K=K, gpu_busy_ms=busy, n_kernel_launch=sum(k[2] for k in kern),
                  sections_device_ms=secs, top_kernels=[(round(a, 3), b[:90], c) for a, b, c in kern[:14]])
            print("prof", B, P, K, round(busy, 2), "ms busy;", {k: round(v, 2) for k, v in secs.items()}, flush=True)
        cache.len = base
    return j
