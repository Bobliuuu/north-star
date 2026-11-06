"""Longer-context + batch>1 plain decode, and plain-vs-self-spec step cost at long context, real North, A100-80GB bf16.
  modal run --detach modal/ctx_batch.py::main          (always detached)
Result JSON is written to Volume northstar-results:/ctx_batch_<ts>.json and printed in logs as 'CTXBATCH_JSON {...}'."""
import sys, time, json, pathlib
sys.path.insert(0, "modal")
import modal
from nsapp import app, weights_vol, results_vol, WEIGHTS, MODEL_DIR, GPU

GPUKW = dict(gpu=GPU, volumes={WEIGHTS: weights_vol, "/results": results_vol}, memory=65536)


@app.function(**GPUKW, timeout=2400)
def ctx_batch_remote(ctxs=(8192, 32768), batches=(1, 2, 4), spec_ctxs=(8192, 32768), n_new=32):
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/harness")
    import statistics as st, torch
    from common import env_meta, build_ids, load_workloads, by_split
    from loader import load_model
    from bench_baseline import hf_cell
    from northstar.refmodel import RefNorth, DraftMode
    from northstar.spec import SelfSpec, plain_greedy
    T0 = time.time()
    m, tok = load_model(MODEL_DIR, "grouped_mm")
    out = dict(meta=env_meta(), load_s=time.time() - T0, rows=[], notes=dict(
        engine="HF transformers eager, grouped_mm experts, sdpa attn, bf16, A100-80GB, greedy",
        prompts="first dev 'code' task chat-templated + CPython stdlib filler to EXACT ctx; batch rows rotate filler/task per sequence",
        hf_cell="DynamicCache, prefill chunk 2048 timed separately (prefill_s), decode_ms = median of 44 post-warmup steps, n_dec=48, logits_to_keep=1",
        spec="RefNorth KV cache; plain_greedy vs SelfSpec(top-4 experts draft, gamma=4), batch 1, n_new tokens, prefill excluded from gen times"))
    code = [r for r in by_split(load_workloads(), "dev") if r["kind"] == "code"]
    hf_cell(m, tok, [build_ids(tok, code[0], 512)], n_dec=8)  # warmup
    for ctx in ctxs:
        for B in batches:
            try:
                ids = [build_ids(tok, code[b % len(code)], ctx, rotate=b) for b in range(B)]
                r = hf_cell(m, tok, ids)
                out["rows"].append(dict(kind="plain_decode_hf", status="ok", **r))
            except Exception as e:
                torch.cuda.empty_cache()
                out["rows"].append(dict(kind="plain_decode_hf", status="ERR", batch=B, ctx=ctx, err=repr(e)[:200]))
            print(json.dumps(out["rows"][-1])[:300], flush=True)
    ref = RefNorth(m)
    eos = m.config.eos_token_id; eos = eos if isinstance(eos, int) else eos[0]
    mode = DraftMode(frozenset(), 4)
    for ctx in spec_ctxs:
        try:
            ids = build_ids(tok, code[0], ctx)
            pg = plain_greedy(ref, ids, n_new, eos_id=eos)
            sp = SelfSpec(ref, mode, 4).generate(ids, n_new, eos_id=eos)
            R = sp["rounds"]; toks = sp["tokens"]
            pstep = st.median(pg["step_s"][4:] or pg["step_s"])
            row = dict(kind="selfspec_vs_plain_refnorth", status="ok", ctx=ctx, batch=1, draft="top4_experts", gamma=4,
                       n_plain=len(pg["tokens"]), n_spec=len(toks), identical=toks == pg["tokens"], rounds=len(R),
                       mean_accepted=st.mean(s.accepted for s in R), committed_per_round=len(toks) / len(R),
                       plain_step_ms_median=1e3 * pstep, plain_tok_per_s=1 / pstep,
                       plain_prefill_s=pg["prefill_s"], spec_verifier_prefill_s=sp["verifier_prefill_s"], spec_draft_prefill_s=sp["draft_prefill_s"],
                       round_ms_median=1e3 * st.median(s.t_catchup + s.t_draft + s.t_verify + s.t_rollback for s in R),
                       catchup_ms_median=1e3 * st.median(s.t_catchup for s in R), draft_ms_median=1e3 * st.median(s.t_draft for s in R),
                       verify_ms_median=1e3 * st.median(s.t_verify for s in R),
                       spec_tok_per_s=len(toks) / sp["gen_s"], speedup_wall_decode_only=(len(toks) / sp["gen_s"]) * pstep,
                       prefill_included_in_tok_per_s=False)
        except Exception as e:
            row = dict(kind="selfspec_vs_plain_refnorth", status="ERR", ctx=ctx, err=repr(e)[:300])
        out["rows"].append(row); print(json.dumps(row)[:400], flush=True)
        torch.cuda.empty_cache()
    out["_gpu_seconds"] = time.time() - T0
    name = f"ctx_batch_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    pathlib.Path("/results", name).write_text(json.dumps(out, default=str))
    results_vol.commit()
    print("CTXBATCH_JSON " + json.dumps(out, default=str), flush=True)
    return out


@app.local_entrypoint()
def main():
    print(ctx_batch_remote.remote())


# ---------------- run2: 128K, batched self-spec, mixed lengths, 3-prompt self-spec ----------------
def _margins(lg):  # lg [B, V] -> top1-top2 margins
    import torch
    t = torch.topk(lg.float(), 2, dim=-1).values
    return (t[:, 0] - t[:, 1]).tolist()


def plain_batched(ref, prompts, n_new, chunk=1024):
    import torch
    from northstar.refmodel import FULL
    dev, B, P = ref.device, len(prompts), len(prompts[0])
    cache = ref.new_cache(B, P + n_new + 2, chunk=chunk)
    ids = torch.tensor(prompts, device=dev)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    ref.prefill(ids[:, : P - 1], cache, FULL, chunk, logits="none")
    torch.cuda.synchronize(); pre = time.perf_counter() - t0
    cur = ids[:, -1:]; toks = [[] for _ in range(B)]; marg = [[] for _ in range(B)]; steps = []
    for _ in range(n_new):
        t1 = time.perf_counter()
        lg = ref.forward(cur, cache, cache.len, FULL, logits="last"); cache.len += 1
        nxt = lg[:, 0].argmax(-1); nl = nxt.tolist()
        steps.append(time.perf_counter() - t1)
        for b, mg in enumerate(_margins(lg[:, 0])): marg[b].append(mg)
        for b in range(B): toks[b].append(nl[b])
        cur = nxt[:, None]
    del cache; torch.cuda.empty_cache()
    return dict(tokens=toks, margins=marg, step_s=steps, prefill_s=pre)


def selfspec_batched(ref, prompts, n_new, g=4, chunk=1024):
    """Lock-step batched greedy self-spec: every sequence commits min_b(a_b)+1 tokens per round (all verified
    by the full model, so output is exactly greedy-verified). Per-sequence accepted a_b is recorded."""
    import torch
    from northstar.refmodel import DraftMode, FULL
    mode = DraftMode(frozenset(), 4)
    dev, B, P = ref.device, len(prompts), len(prompts[0])
    max_len = P + n_new + g + 2
    vc = ref.new_cache(B, max_len, chunk=chunk); dc = ref.new_cache(B, max_len, chunk=chunk)
    ids = torch.tensor(prompts, device=dev)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    ref.prefill(ids[:, : P - 1], vc, FULL, chunk, logits="none")
    torch.cuda.synchronize(); vpre = time.perf_counter() - t0
    t0 = time.perf_counter()
    ref.prefill(ids[:, : P - 1], dc, mode, chunk, logits="none")
    torch.cuda.synchronize(); dpre = time.perf_counter() - t0
    t = [list(p) for p in prompts]; out = [[] for _ in range(B)]; rounds = []
    while len(out[0]) < n_new:
        n = len(t[0])
        t1 = time.perf_counter()
        feed = torch.tensor([row[dc.len:n] for row in t], device=dev)
        lg = ref.prefill(feed, dc, mode, chunk, logits="last")
        drafts = [lg[:, 0].argmax(-1)]
        torch.cuda.synchronize(); t_c = time.perf_counter() - t1
        t1 = time.perf_counter()
        for _ in range(1, g):
            lg = ref.forward(drafts[-1][:, None], dc, dc.len, mode, logits="last"); dc.len += 1
            drafts.append(lg[:, 0].argmax(-1))
        torch.cuda.synchronize(); t_d = time.perf_counter() - t1
        t1 = time.perf_counter()
        D = torch.stack(drafts, 1)
        vin = torch.cat([torch.tensor([[row[n - 1]] for row in t], device=dev), D], 1)
        pred = ref.forward(vin, vc, n - 1, FULL, logits="all").argmax(-1)
        Dl, Pl = D.tolist(), pred.tolist()
        torch.cuda.synchronize(); t_v = time.perf_counter() - t1
        a = []
        for b in range(B):
            k = 0
            while k < g and Dl[b][k] == Pl[b][k]: k += 1
            a.append(k)
        am = min(a)
        for b in range(B):
            new = Dl[b][:am] + [Pl[b][am]]; t[b].extend(new); out[b].extend(new)
        vc.len = n + am; dc.len = n + min(am, g - 1)
        rounds.append(dict(a=a, a_min=am, committed=am + 1, catch_ms=1e3 * t_c, draft_ms=1e3 * t_d, verify_ms=1e3 * t_v))
    del vc, dc; torch.cuda.empty_cache()
    return dict(tokens=[o[:n_new] for o in out], rounds=rounds, vpre=vpre, dpre=dpre,
                gen_s=sum((r["catch_ms"] + r["draft_ms"] + r["verify_ms"]) / 1e3 for r in rounds))


def _first_mismatch(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def spec_cell(ref, tok, rows, ctx, B, n_new, label):
    import statistics as st
    from common import build_ids
    prompts = [build_ids(tok, rows[b % len(rows)], ctx, rotate=b) for b in range(B)]
    pg = plain_batched(ref, prompts, n_new)
    sp = selfspec_batched(ref, prompts, n_new)
    R = sp["rounds"]; pstep = st.median(pg["step_s"][4:])
    per = []
    for b in range(B):
        mm = _first_mismatch(sp["tokens"][b], pg["tokens"][b]); mg = sorted(pg["margins"][b])
        per.append(dict(seq=b, mean_accepted_own=st.mean(r["a"][b] for r in R), identical=mm is None, first_mismatch=mm,
                        plain_margin_at_mismatch=None if mm is None else pg["margins"][b][mm],
                        plain_margin_min=mg[0], plain_margin_p10=mg[len(mg) // 10]))
    return dict(kind="selfspec_batched_vs_plain", status="ok", label=label, ctx=ctx, batch=B, draft="top4_experts", gamma=4, n_new=n_new,
                rounds=len(R), a_min_mean=st.mean(r["a_min"] for r in R), committed_per_round=st.mean(r["committed"] for r in R),
                plain_step_ms_median=1e3 * pstep, plain_tok_per_s_total=B / pstep, plain_prefill_s=pg["prefill_s"], spec_vprefill_s=sp["vpre"], spec_dprefill_s=sp["dpre"],
                round_ms_median=st.median(r["catch_ms"] + r["draft_ms"] + r["verify_ms"] for r in R),
                catchup_ms_median=st.median(r["catch_ms"] for r in R), draft_ms_median=st.median(r["draft_ms"] for r in R), verify_ms_median=st.median(r["verify_ms"] for r in R),
                spec_tok_per_s_total=B * len(sp["tokens"][0]) / sp["gen_s"], per_seq=per, prefill_included_in_tok_per_s=False)


def hf_mixed(m, tok, id_rows, n_dec=48, chunk=512, warm=4):  # run2 OOM'd at chunk=2048 (masked SDPA materializes scores)
    """Left-padded batch of different true lengths, attention_mask + explicit position_ids, DynamicCache."""
    import torch, statistics as st
    from transformers import DynamicCache
    B, Lmax = len(id_rows), max(len(r) for r in id_rows)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    x = torch.tensor([[pad] * (Lmax - len(r)) + r for r in id_rows], device="cuda")
    am = torch.tensor([[0] * (Lmax - len(r)) + [1] * len(r) for r in id_rows], device="cuda")
    pos = (am.cumsum(-1) - 1).clamp(min=0)
    lens = am.sum(-1, keepdim=True)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    cache = DynamicCache(config=m.config)
    torch.cuda.synchronize(); t0 = time.perf_counter(); out = None
    for s in range(0, Lmax, chunk):
        out = m(input_ids=x[:, s:s + chunk], attention_mask=am[:, :s + chunk], position_ids=pos[:, s:s + chunk], past_key_values=cache, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize(); prefill_s = time.perf_counter() - t0
    nxt = out.logits[:, -1].argmax(-1, keepdim=True); steps = []; first = [nxt[:, 0].tolist()]
    for i in range(n_dec):
        am = torch.cat([am, torch.ones(B, 1, dtype=am.dtype, device="cuda")], 1)
        t1 = time.perf_counter()
        out = m(input_ids=nxt, attention_mask=am, position_ids=lens + i, past_key_values=cache, use_cache=True, logits_to_keep=1)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        torch.cuda.synchronize(); steps.append(time.perf_counter() - t1); first.append(nxt[:, 0].tolist())
    s_ = sorted(steps[warm:])
    return dict(batch=B, lens=[len(r) for r in id_rows], padded_len=Lmax, prefill_s=prefill_s, decode_ms_median=1e3 * st.median(s_),
                decode_ms_p10=1e3 * s_[int(.1 * len(s_))], decode_ms_p90=1e3 * s_[int(.9 * len(s_))], n_dec=n_dec,
                decode_tok_per_s=B / st.median(s_), peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30,
                first_tokens_per_seq=[[f[b] for f in first[:8]] for b in range(B)])


@app.function(**GPUKW, timeout=3600)
def ctx_batch2_remote(parts=("pspec8k", "pbatch", "pmixed", "p128k"), n_new=32):
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/harness")
    import torch
    from common import env_meta, build_ids, load_workloads, by_split
    from loader import load_model
    from bench_baseline import hf_cell
    from northstar.refmodel import RefNorth
    T0 = time.time()
    m, tok = load_model(MODEL_DIR, "grouped_mm")
    out = dict(meta=env_meta(), load_s=time.time() - T0, rows=[], parts=list(parts))
    code = [r for r in by_split(load_workloads(), "dev") if r["kind"] == "code"]
    out["n_dev_code"] = len(code)
    ts = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())

    def emit(row):
        out["rows"].append(row); print("ROW " + json.dumps(row, default=str)[:1500], flush=True)
        out["_gpu_seconds"] = time.time() - T0
        pathlib.Path("/results", f"ctx_batch2_{ts}.json").write_text(json.dumps(out, default=str)); results_vol.commit()

    def guard(fn, **kw):
        try:
            emit(fn())
        except Exception as e:
            torch.cuda.empty_cache(); emit(dict(status="ERR", err=repr(e)[:400], **kw))
    hf_cell(m, tok, [build_ids(tok, code[0], 512)], n_dec=8)
    ref = RefNorth(m)
    if "pspec8k" in parts:
        for i in range(3):
            guard(lambda i=i: spec_cell(ref, tok, code[i:] + code[:i], 8192, 1, n_new, f"prompt{i}"), kind="spec", ctx=8192, batch=1, prompt=i)
    if "pbatch" in parts:
        for B in (2, 4):
            guard(lambda B=B: spec_cell(ref, tok, code, 8192, B, n_new, f"batch{B}"), kind="spec", ctx=8192, batch=B)
    if "pmixed" in parts:
        def mixed(lens):
            ids = [build_ids(tok, code[b % len(code)], L, rotate=b) for b, L in enumerate(lens)]
            return dict(kind="plain_decode_hf_mixed_leftpad", status="ok", **hf_mixed(m, tok, ids))
        guard(lambda: mixed((2048, 8192, 32768)), kind="mixed")
    if "p128k" in parts:
        guard(lambda: dict(kind="plain_decode_hf", status="ok", **hf_cell(m, tok, [build_ids(tok, code[0], 131072)])), kind="plain128k", ctx=131072, batch=1)
        guard(lambda: spec_cell(ref, tok, code, 131072, 1, 24, "128k"), kind="spec128k", ctx=131072, batch=1)
    print("CTXBATCH2_JSON " + json.dumps(out, default=str), flush=True)
    return out


@app.local_entrypoint()
def main2(parts: str = "pspec8k,pbatch,pmixed,p128k"):
    print(ctx_batch2_remote.remote(parts=tuple(parts.split(","))))
