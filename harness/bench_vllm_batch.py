"""vLLM batched plain decode on the H18-H20 32-prompt pool (same groups as ours: the first B prompts). One engine, max_num_seqs=32.
Per B: t_full = generate(B prompts, max_tokens=192), t_one = generate(B prompts, max_tokens=1); decode agg tok/s = (sum n - B) / (t_full - t_one).
Usage: python bench_vllm_batch.py <model_dir> <out.jsonl> [sizes]"""
import json, pathlib, sys, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))


def main():
    model_dir, out = sys.argv[1], pathlib.Path(sys.argv[2])
    sizes = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,2,4,8,16,32").split(",")]
    from transformers import AutoTokenizer
    import bench_batch as bb
    tok = AutoTokenizer.from_pretrained(model_dir)
    pool = bb.build_pool(tok, 32)
    import vllm
    from vllm import LLM, SamplingParams
    kw = dict(model=model_dir, dtype="bfloat16", max_model_len=8192, max_num_seqs=max(sizes), enable_prefix_caching=False, gpu_memory_utilization=0.88)
    t0 = time.time(); llm = LLM(**kw)
    f = out.open("a")
    f.write(json.dumps(dict(phase="meta", vllm=vllm.__version__, kw={k: v for k, v in kw.items() if k != "model"}, load_s=time.time() - t0,
                            pool=[p[0] for p in pool])) + "\n"); f.flush()
    sp, sp1 = SamplingParams(temperature=0, max_tokens=192), SamplingParams(temperature=0, max_tokens=1)

    def gen(B, p):
        reqs = [{"prompt_token_ids": list(g[2])} for g in pool[:B]]
        t = time.perf_counter(); o = llm.generate(reqs, p, use_tqdm=False)
        return time.perf_counter() - t, [list(x.outputs[0].token_ids) for x in o]
    gen(4, sp)  # warm
    for B in sizes:
        best = None
        for rep in range(2):
            tf, toks = gen(B, sp); t1, _ = gen(B, sp1)
            n = sum(len(t) for t in toks)
            r = dict(phase="batch", B=B, rep=rep, t_full=tf, t_one=t1, n=n, decode_agg_tok_s=(n - B) / (tf - t1), e2e_agg_tok_s=n / tf, lens=[len(t) for t in toks])
            f.write(json.dumps(r) + "\n"); f.flush()
            best = max(best or 0, r["decode_agg_tok_s"])
        print("VLLM_BATCH B", B, "decode agg %.0f tok/s" % best, flush=True)


if __name__ == "__main__":
    main()
