"""vLLM 0.33 under the SAME arrival schedules as bench_cbatch.py (LLMEngine.add_request at arrival time + step loop), H100.
Usage: python bench_vllm_cb.py <model_dir> <out.jsonl> <sched.json>
Engine: bf16, max_num_seqs=32, max_model_len 8192, prefix caching off, other settings vLLM defaults (incl. its chunked prefill)."""
import json, pathlib, sys, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))


def main():
    model_dir, out, schedf = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    S = json.loads(schedf.read_text())
    from transformers import AutoTokenizer
    import bench_batch as bb
    from bench_cbatch import summarize
    tok = AutoTokenizer.from_pretrained(model_dir)
    pool = bb.build_pool(tok, 32)
    assert [p[0] for p in pool] == S["pool"], "prompt pool differs from ours"
    prompts = [p[2] for p in pool]
    import vllm
    from vllm import EngineArgs, LLMEngine, SamplingParams
    ea = EngineArgs(model=model_dir, dtype="bfloat16", max_model_len=8192, max_num_seqs=32, enable_prefix_caching=False, gpu_memory_utilization=0.88)
    t0 = time.time(); eng = LLMEngine.from_engine_args(ea)
    f = out.open("a")
    add = lambda **r: (f.write(json.dumps(r) + "\n"), f.flush())
    add(phase="meta", vllm=vllm.__version__, load_s=time.time() - t0)
    sp = SamplingParams(temperature=0, max_tokens=S["max_new"])

    def run(sc):
        pend = sorted(sc, key=lambda s: s["arrival"]); live = {}; done = {}
        c0 = time.perf_counter()
        while pend or live:
            now = time.perf_counter() - c0
            while pend and pend[0]["arrival"] <= now:
                s = pend.pop(0)
                eng.add_request(str(s["rid"]), {"prompt_token_ids": list(prompts[s["prompt"]])}, sp)
                live[str(s["rid"])] = dict(arrival=s["arrival"], t_first=None)
            if not live:
                time.sleep(max(0.0, min(0.002, pend[0]["arrival"] - (time.perf_counter() - c0)))); continue
            outs = eng.step()
            tn = time.perf_counter() - c0
            for o in outs:
                d = live.get(o.request_id)
                if d is None:
                    continue
                if d["t_first"] is None and len(o.outputs[0].token_ids) > 0:
                    d["t_first"] = tn
                if o.finished:
                    d["t_done"] = tn; d["n"] = len(o.outputs[0].token_ids); d["toks"] = list(o.outputs[0].token_ids)
                    done[o.request_id] = live.pop(o.request_id)
        return done

    run([dict(rid=10000 + i, prompt=i % len(prompts), arrival=0.0) for i in range(32)])  # warm
    for rate, sc in S["scheds"].items():
        done = run(sc)
        reqs = list(done.values())
        sm = summarize(reqs, max(d["t_done"] for d in reqs) - min(d["arrival"] for d in reqs))
        add(phase="load", system="vllm", rate=rate, **sm)
        print("LOAD vllm rate", rate, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in sm.items()}), flush=True)


if __name__ == "__main__":
    main()
