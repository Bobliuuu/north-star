"""Live North demo on Modal: streams REAL generation events (schema: demo/README.md).

LAUNCH (paid, ~$0.25-0.45 for the defaults; one A100-80GB, model load dominates). Always detached, never plain `modal run`:
  modal run --detach modal/demo.py --task <workload id> --mode both --gamma 4 --skip-k 8 --n-new 64
The local entrypoint prints the live table and writes demo/traces/<utc>.jsonl; the remote also saves a copy to the
`northstar-results` volume (/results/demo/) so the trace survives a dropped client.  Replay later (free, CPU):
  python demo/view.py --replay demo/traces/<utc>.jsonl
"""
import sys, time, json, pathlib
sys.path.insert(0, "modal")
sys.path.insert(0, "demo")
sys.path.insert(0, "harness")
import modal
from nsapp import app, image, weights_vol, results_vol, WEIGHTS, MODEL_DIR, MODEL_ID, MODEL_REV, GPU

USD_PER_S = 0.000694  # A100-80GB list price used by harness/common.py and results/SPEND.md
demo_image = image.add_local_dir("demo", "/root/demo", copy=False)


@app.function(image=demo_image, gpu=GPU, volumes={WEIGHTS: weights_vol, "/results": results_vol}, memory=65536, timeout=1800)
def demo_remote(task: str, mode: str, gamma: int, skip_k: int, expert_r: int, n_new: int, spend_prior: float, spend_rows: int):
    """Generator: yields event dicts live. Every number comes from the live run (timers, torch.cuda, token ids)."""
    import queue, threading
    t_start = time.time()
    sys.path[:0] = ["/root", "/root/harness", "/root/demo"]
    import torch
    import events as E
    from common import build_ids, load_workloads
    from loader import load_model
    from bench_spec import skip_set
    from northstar.refmodel import RefNorth, DraftMode
    from northstar.spec import SelfSpec, plain_greedy

    out_path = pathlib.Path("/results/demo") / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out_path, "w")
    q = queue.Queue()

    def emit(ev):
        E.write_event(fh, ev)
        q.put(ev)

    def cost():
        run = (time.time() - t_start) * USD_PER_S
        return run, spend_prior + run

    def mem():
        return torch.cuda.memory_allocated() / 2**30, torch.cuda.max_memory_allocated() / 2**30

    def work():
        try:
            m, tok = load_model(MODEL_DIR, "grouped_mm")
            rows = {r["id"]: r for r in load_workloads()}
            ids = build_ids(tok, rows[task])
            eos = m.config.eos_token_id
            eos = eos if isinstance(eos, int) else eos[0]
            ref = RefNorth(m)
            dmode = DraftMode(skip_set(skip_k), expert_r or None)
            draft_label = f"skip{skip_k}" + (f"_r{expert_r}" if expert_r else "")
            emit(dict(type="run", schema=E.SCHEMA, task=task, prompt_len=len(ids), n_new=n_new, mode_request=mode,
                      spend_prior_usd=spend_prior, spend_rows=spend_rows,
                      provenance=dict(source=E.REAL_SOURCE, model_id=MODEL_ID, model_rev=MODEL_REV, gpu=GPU,
                                      created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                      torch=torch.__version__, executor="northstar.refmodel.RefNorth",
                                      load_s=round(time.time() - t_start, 1))))
            for seg in (["plain", "selfspec"] if mode == "both" else [mode]):
                st = dict(i=0, total=0, gen_s=0.0, acc=0, drafted=0, last=None)
                if mode == "both":
                    emit(dict(type="mode_switch", mode=seg))

                def push(new, dt, accepted, drafted, seg=seg, st=st):
                    st["i"] += 1; st["total"] += len(new); st["gen_s"] += dt
                    st["acc"] += accepted; st["drafted"] += drafted
                    g, p = mem(); rc, tc = cost()
                    ev = dict(type="step", i=st["i"], tokens=list(new), text=tok.decode(new), mode=seg,
                              gamma=(gamma if seg != "plain" else 0), draft=(draft_label if seg != "plain" else None),
                              accepted=accepted, drafted=drafted, committed_total=st["total"], gen_s=st["gen_s"],
                              inst_tok_s=len(new) / dt if dt > 0 else 0.0, cum_tok_s=st["total"] / st["gen_s"],
                              accept_cum=(st["acc"] / st["drafted"] if st["drafted"] else None),
                              gpu_mem_gb=g, gpu_peak_gb=p, run_cost_usd=rc, cum_cost_usd=tc)
                    st["last"] = ev
                    emit(ev)

                if seg == "plain":
                    plain_greedy(ref, ids, n_new, eos_id=eos, on_step=lambda t, dt: push([t], dt, 0, 0))
                else:
                    SelfSpec(ref, dmode, gamma).generate(
                        ids, n_new, eos_id=eos,
                        on_round=lambda new, s: push(new, s.t_catchup + s.t_draft + s.t_verify + s.t_rollback, s.accepted, gamma))
                last = st["last"]
                emit(dict(type="end", mode=seg, committed_total=st["total"], gen_s=st["gen_s"], cum_tok_s=last["cum_tok_s"],
                          gpu_peak_gb=last["gpu_peak_gb"], run_cost_usd=last["run_cost_usd"], cum_cost_usd=last["cum_cost_usd"]))
        except BaseException as e:  # surface the failure to the client; never fabricate output
            q.put(dict(type="error", error=repr(e)))
        finally:
            q.put(None)

    th = threading.Thread(target=work, daemon=True)
    th.start()
    while True:
        ev = q.get()
        if ev is None:
            break
        yield ev
    fh.close()
    results_vol.commit()


@app.local_entrypoint()
def main(task: str, mode: str = "both", gamma: int = 4, skip_k: int = 8, expert_r: int = 0, n_new: int = 64):
    assert mode in ("plain", "selfspec", "both")
    import render as R
    from spend import load_spend
    from common import append_spend
    prior, rows, _ = load_spend("results/SPEND.md")
    outp = pathlib.Path("demo/traces")
    outp.mkdir(parents=True, exist_ok=True)
    path = outp / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + ".jsonl")
    t0 = time.time()
    with open(path, "w") as f:
        for ev in demo_remote.remote_gen(task, mode, gamma, skip_k, expert_r, n_new, prior, rows):
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            f.flush()
            for line in R.render_event(ev):
                print(line, flush=True)
            if ev.get("type") == "error":
                print("REMOTE ERROR:", ev["error"], file=sys.stderr)
    append_spend("results", f"demo {task} {mode}", GPU, time.time() - t0, "wall incl. queue; trace in demo/traces")
    print(f"trace saved: {path}")
