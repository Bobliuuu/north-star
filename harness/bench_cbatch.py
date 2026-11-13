"""Continuous batching validation + load test (ours), H100. Also writes the request schedules so bench_vllm_cb.py replays the SAME load.
Usage: python bench_cbatch.py <model_dir> <out.jsonl> <sched.json> [n_req] [rates]
  exact : N requests (pool prompts in a seeded random order) all at t=0 on 32 slots; every output must equal the same prompt decoded alone
  load  : Poisson arrivals at each rate (req/s; 'inf' = all at t=0), N requests, 192 max new tokens, EOS stops.
Metrics per rate: throughput (generated tokens / makespan), request latency (done - arrival), TTFT (first token - arrival), TPOT."""
import json, pathlib, random, sys, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

MAX_NEW = 192
import os
PREFILL = os.environ.get("NS_CB_PREFILL", "batched")


def schedule(n, rate, seed, npool):
    rng = random.Random(seed)
    order = [rng.randrange(npool) for _ in range(n)]
    t, arr = 0.0, []
    for _ in range(n):
        if rate != "inf":
            t += rng.expovariate(float(rate))
        arr.append(t)
    return [dict(rid=i, prompt=order[i], arrival=arr[i]) for i in range(n)]


def summarize(reqs_done, makespan):
    """reqs_done: list of dict(arrival, t_first, t_done, n)."""
    def pct(v, q):
        v = sorted(v); return v[min(len(v) - 1, int(q * len(v)))]
    lat = [r["t_done"] - r["arrival"] for r in reqs_done]
    ttft = [r["t_first"] - r["arrival"] for r in reqs_done]
    tpot = [(r["t_done"] - r["t_first"]) / max(1, r["n"] - 1) for r in reqs_done]
    n = sum(r["n"] for r in reqs_done)
    return dict(requests=len(reqs_done), tokens=n, makespan_s=makespan, throughput_tok_s=n / makespan,
                lat_mean_s=sum(lat) / len(lat), lat_p50_s=pct(lat, .5), lat_p90_s=pct(lat, .9),
                ttft_mean_s=sum(ttft) / len(ttft), ttft_p50_s=pct(ttft, .5), ttft_p90_s=pct(ttft, .9),
                tpot_mean_ms=1e3 * sum(tpot) / len(tpot), tpot_p90_ms=1e3 * pct(tpot, .9))


def main():
    model_dir, out, schedf = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    n_req = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    rates = (sys.argv[5] if len(sys.argv) > 5 else "2,4,8,16,inf").split(",")
    import torch
    from loader import load_model
    import bench_batch as bb
    from cbatch import ContinuousEngine, Request
    try:
        m, tok = load_model(model_dir, "grouped_mm")
    except TypeError:
        m, tok = load_model(model_dir, None)
    pool = bb.build_pool(tok, 32)
    prompts = [p[2] for p in pool]
    f = out.open("a")
    add = lambda **r: (f.write(json.dumps(r) + "\n"), f.flush())
    t0 = time.time()
    eng = ContinuousEngine(m, max(len(p) for p in prompts) + MAX_NEW + 40, slots=32, prefill=PREFILL)
    add(phase="meta", prefill=PREFILL, torch=torch.__version__, pool=[p[0] for p in pool], buckets=eng.buckets, init_s=time.time() - t0)
    scheds = {r: schedule(n_req, r, 20261112, len(prompts)) for r in rates}
    schedf.write_text(json.dumps(dict(max_new=MAX_NEW, pool=[p[0] for p in pool], scheds=scheds)))
    # warm every bucket once
    eng.run([Request(-1 - b, prompts[b % len(prompts)], 4) for b in range(32)])
    # ---- reference: every distinct prompt decoded alone (B=1 bucket)
    ref = {}
    for i, p in enumerate(prompts):
        r = Request(i, p, MAX_NEW); eng.run([r]); ref[i] = r.out
    # ---- exactness under contention: all at t=0
    ex = scheds["inf"] if "inf" in scheds else schedule(n_req, "inf", 20261112, len(prompts))
    reqs = [Request(s["rid"], prompts[s["prompt"]], MAX_NEW, 0.0) for s in ex]
    st = eng.run(reqs)
    eq = [r.out == ref[s["prompt"]] for r, s in zip(reqs, ex)]
    add(phase="exact", n=len(eq), equal=sum(eq), mismatched=[s["rid"] for s, e in zip(ex, eq) if not e], **st)
    print("EXACT", sum(eq), "/", len(eq), st, flush=True)
    # ---- load
    for rate in rates:
        sc = scheds[rate]
        reqs = [Request(s["rid"], prompts[s["prompt"]], MAX_NEW, s["arrival"]) for s in sc]
        st = eng.run(reqs)
        done = [dict(arrival=r.arrival, t_first=r.t_first, t_done=r.t_done, n=len(r.out)) for r in reqs]
        sm = summarize(done, max(d["t_done"] for d in done) - min(d["arrival"] for d in done))
        same = sum(r.out == ref[s["prompt"]] for r, s in zip(reqs, sc))
        add(phase="load", system="ours", rate=rate, exact_vs_alone=same, prefill_s_total=sum(r.prefill_s for r in reqs), **sm,
            **{"eng_" + k: v for k, v in st.items()})
        print("LOAD ours rate", rate, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in sm.items()}), "exact", same, flush=True)


if __name__ == "__main__":
    main()
