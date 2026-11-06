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
