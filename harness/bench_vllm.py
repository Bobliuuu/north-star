"""vLLM baseline worker for North (one engine config per process). Usage: python bench_vllm.py <model_dir> <cfgname> <out.jsonl> [limit]
Prompts = exactly the H13 natural-21 tasks (harness/workloads dev+heldout + bench_pld.custom_tasks) and the H15 99-prompt mix, chat-templated
with the model tokenizer, fed as token ids. Greedy, <=192 new tokens, batch 1 (max_num_seqs=1), prefix caching OFF.
Per task, per rep: t_full = wall of generate(max_tokens=192); t_one = wall of generate(max_tokens=1) (prefill + first token).
decode_tok_s = (n-1)/(t_full-t_one) (comparable to H13/H15 decode-only loops); e2e_tok_s = n/t_full. Best of reps (H13 convention)."""
import ast, json, pathlib, sys, textwrap, time, traceback
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import _ids, build_ids, load_workloads

CFGS = {
    "plain": None,
    "ngram_k8_n3": dict(method="ngram", num_speculative_tokens=8, prompt_lookup_max=3, prompt_lookup_min=3),    # ~ our pld3 g8
    "ngram_k12_n2to12": dict(method="ngram", num_speculative_tokens=12, prompt_lookup_max=12, prompt_lookup_min=2),  # ~ our sam2 g12
    "ngram_k5_n2to5": dict(method="ngram", num_speculative_tokens=5, prompt_lookup_max=5, prompt_lookup_min=2),     # vLLM docs-style
}


def custom_tasks():
    src = (HERE / "bench_pld.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "custom_tasks")
    ns = dict(ast=ast, textwrap=textwrap)
    exec(compile(ast.Module([fn], []), "custom_tasks", "exec"), ns)
    return ns["custom_tasks"]()


def main():
    model_dir, cfg, out, limit = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 0
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    nat = [r for r in load_workloads() if r["split"] in ("dev", "heldout")] + custom_tasks()
    tasks = [("natural", t["id"], build_ids(tok, t)) for t in nat]
    mix = [json.loads(l) for l in (HERE / "workloads_real" / "realtraffic.jsonl").read_text().splitlines() if l.strip()]
    tasks += [("mix99", t["id"], _ids(tok, [dict(x) for x in t["messages"]])) for t in mix]
    if limit:
        tasks = tasks[:limit] + tasks[len(nat):len(nat) + limit]
    print("tasks", len(tasks), "natural", len(nat), flush=True)
    import vllm
    from vllm import LLM, SamplingParams
    kw = dict(model=model_dir, dtype="bfloat16", max_model_len=8192, max_num_seqs=1, enable_prefix_caching=False, gpu_memory_utilization=0.88)
    if CFGS[cfg]:
        kw["speculative_config"] = CFGS[cfg]
    t0 = time.time()
    llm = LLM(**kw)
    f = out.open("a")
    def add(**r):
        f.write(json.dumps(r) + "\n"); f.flush()
    add(phase="meta", cfg=cfg, vllm=vllm.__version__, kw={k: v for k, v in kw.items() if k != "model"}, load_s=time.time() - t0)
    sp = SamplingParams(temperature=0, max_tokens=192)
    sp1 = SamplingParams(temperature=0, max_tokens=1)
    def gen(ids, p):
        t = time.perf_counter(); o = llm.generate([{"prompt_token_ids": ids}], p, use_tqdm=False)[0]
        return time.perf_counter() - t, list(o.outputs[0].token_ids)
    for _ in range(2):
        gen(tasks[0][2], sp)
    for cell, tid, ids in tasks:
        try:
            best = None
            for rep in range(2):
                tf, toks = gen(ids, sp); t1, _ = gen(ids, sp1)
                n = len(toks); dec = (n - 1) / max(tf - t1, 1e-6) if n > 1 else float("nan")
                add(phase="rep", cfg=cfg, cell=cell, task=tid, rep=rep, prompt_len=len(ids), n=n, t_full=tf, t_one=t1, decode_tok_s=dec, e2e_tok_s=n / tf, gen=toks)
            print(cfg, cell, tid, "P", len(ids), "n", n, "dec %.1f e2e %.1f" % (dec, n / tf), flush=True)
        except Exception:
            add(phase="task_error", cfg=cfg, task=tid, err=traceback.format_exc()[-800:]); print("TASK ERR", tid, traceback.format_exc()[-500:], flush=True)
    add(phase="done", cfg=cfg, secs=time.time() - t0)


if __name__ == "__main__":
    main()
