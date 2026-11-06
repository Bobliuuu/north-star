"""Real-traffic PLD validation: FastNorth(exact=True) plain T=1 graph vs speculative configs (draft = CPU, verify = exact block graph), H100.
Tasks: harness/workloads_real/realtraffic.jsonl (see build_realtraffic.py). Every task is run; nothing is dropped except prompts that do not fit the
largest bucket (logged). Tasks with rep2=True get 2 reps of plain and of every config, the rest 1 rep. Exactness = spec output == plain output of the
SAME executor (token ids, <=n_new). Rows: plain / live / skip / task_error."""
import json, pathlib, time, traceback
import torch

from common import Jsonl, _ids
from northstar.refmodel import RefNorth
from graph_north import GraphNorth
from fast_north import FastNorth
import pld

TMAX = 13
BUCKETS = (1280, 2560, 5120, 10240)  # max prompt tokens per cell
CONFIGS = [("sam2", 12, "fixed"), ("pld3", 8, "fixed"), ("sam2", 12, "bo"), ("sam2", 12, "off"), ("sam2", 12, "bo+off")]
WL = pathlib.Path(__file__).resolve().parent / "workloads_real" / "realtraffic.jsonl"


def _fd(a, b):
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), None if len(a) == len(b) else n)


@torch.no_grad()
def bench(m, tok, j, n_new=192, reps_sub=2, tag="", limit=None, configs=None):
    configs = configs or CONFIGS
    ref = RefNorth(m)
    eos = m.config.eos_token_id
    eos = eos if isinstance(eos, int) else eos[0]
    tasks = [json.loads(l) for l in WL.read_text().splitlines() if l.strip()]
    if limit:
        tasks = tasks[::max(1, len(tasks) // limit)][:limit]
    items = []
    for t in tasks:
        ids = _ids(tok, [dict(x) for x in t["messages"]])
        items.append((t, ids))
    print("tokenized", len(items), "prompt len min/max", min(len(i) for _, i in items), max(len(i) for _, i in items), flush=True)
    for bi, edge in enumerate(BUCKETS):
        lo = BUCKETS[bi - 1] if bi else 0
        cell = [(t, ids) for t, ids in items if lo < len(ids) <= edge]
        if bi == len(BUCKETS) - 1:
            for t, ids in items:
                if len(ids) > edge:
                    j.add(phase="skip", task=t["id"], prompt_len=len(ids), why="longer than largest bucket"); print("SKIP", t["id"], len(ids), flush=True)
        if not cell:
            continue
        Lmax = edge + n_new + 40
        X = FastNorth(ref, Lmax, exact=True); gb = GraphNorth(ref, Lmax, "bmm")
        kv = X.new_kv()
        steps = {1: X.make_step(1, kv)}
        for T in range(2, TMAX + 1):
            steps[T] = X.make_step(T, kv, logits="all")
        print("bucket", edge, "captured; tasks", len(cell), flush=True)
        for t, ids in cell:
            try:
                P = len(ids)
                reps = reps_sub if t["rep2"] else 1
                gb.eager_prefill(ids[:-1], kv)

                def one_plain():
                    tk, cur = [], ids[-1]
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    for i in range(n_new):
                        steps[1].set([cur], P - 1 + i); cur = int(steps[1].run()[0, 0]); tk.append(cur)
                        if cur == eos: break
                    return tk, time.perf_counter() - t0

                tp, _ = one_plain()  # warm + reference trace
                ptoks = tp
                for rep in range(reps):
                    tk, dt = one_plain()
                    j.add(phase="plain", task=t["id"], kind=t["kind"], rep=rep, tag=tag, prompt_len=P, n=len(tk), tok_s=len(tk) / dt, plain_rep_identical=tk == ptoks, bucket=edge)
                n = len(ptoks)
                for name, g, mode in configs:
                    for rep in range(reps):
                        def verify(tcur, drafts):
                            T = 1 + len(drafts)
                            stp = steps[T]; stp.set([tcur[-1]] + drafts, len(tcur) - 1)
                            return stp.run()[0].tolist()
                        pol = pld.Policy(mode, g)
                        torch.cuda.synchronize(); t0 = time.perf_counter()
                        out, rounds = pld.spec_loop(pld.make(name), ids, n, g, verify, eos, policy=pol)
                        torch.cuda.synchronize(); dt = time.perf_counter() - t0
                        nd = [r for r in rounds if r[0] > 0]
                        j.add(phase="live", task=t["id"], kind=t["kind"], rep=rep, tag=tag, prompt_len=P, bucket=edge, proposer=name, gamma=g, mode=mode,
                              n=len(out), n_plain=n, identical=out == ptoks, fd=_fd(out, ptoks), spec_tok_s=len(out) / dt, rounds=len(rounds), draft_rounds=len(nd),
                              drafted=sum(d for d, _ in nd), accepted=sum(a for _, a in nd), tok_in_draft_rounds=sum(a + 1 for _, a in nd))
                last = [json.loads(l) for l in j.lines if '"phase": "live"' in l][-len(configs) * reps:]
                pl = [json.loads(l) for l in j.lines if '"phase": "plain"' in l][-reps:]
                ptps = sum(r["tok_s"] for r in pl) / len(pl)
                print(t["id"], "P", P, "n", n, "plain %.0f;" % ptps, " ".join("%s%d%s=%.2fx%s" % (r["proposer"], r["gamma"], r["mode"], r["spec_tok_s"] / ptps, "" if r["identical"] else "!") for r in last[::reps]), flush=True)
            except Exception:
                j.add(phase="task_error", task=t["id"], err=traceback.format_exc()[-900:]); print("TASK ERR", t["id"], traceback.format_exc()[-700:], flush=True)
        del X, gb, kv, steps
        torch.cuda.empty_cache()


def run(m, tok, tag="", n_new=192, limit=0):
    j = Jsonl()
    try:
        bench(m, tok, j, n_new=n_new, tag=tag, limit=limit or None)
    except Exception:
        j.add(phase="fatal", err=traceback.format_exc()[-900:]); print("FATAL", traceback.format_exc()[-900:], flush=True)
    return j
