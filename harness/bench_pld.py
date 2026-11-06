"""Zero-cost drafting (prompt lookup / suffix match / n-gram cache) on the CUDA-graphed North, H100.
Per task: (1) plain graphed greedy decode (the baseline AND the acceptance trace), (2) offline simulation of every proposer x gamma against that
trace with the in-run measured block-verify cost table, (3) LIVE wall-clock for a short list of configs: draft = CPU only, verify = one graphed
block step of T=1+n_drafts (T=1, i.e. the plain step, when the proposer has no draft)."""
import ast, statistics as st, sys, textwrap, time
import torch

from common import Jsonl, build_ids, load_workloads
from northstar.refmodel import RefNorth
from graph_north import GraphNorth, time_graph
import pld

TMAX = 13  # largest verify block = gamma 12
SIM_G = (2, 3, 4, 6, 8, 12)
LIVE = [("pld3", 4), ("pld3", 8), ("sam", 4), ("sam", 8), ("samw", 8), ("ngc", 4), ("pld2", 4)]


def custom_tasks():
    """Context-heavy tasks built from CPython stdlib text at run time (edit / extract / summarize)."""
    import sysconfig, pathlib
    lib = pathlib.Path(sysconfig.get_paths()["stdlib"])
    def funcs(mod, lo=18, hi=45):
        src = (lib / (mod + ".py")).read_text()
        lines = src.splitlines()
        fs = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and lo <= n.end_lineno - n.lineno + 1 <= hi]
        fs.sort(key=lambda n: n.lineno)
        return ["\n".join(lines[n.lineno - 1:n.end_lineno]) for n in fs]
    def chunk(mod, a, b):
        return "\n".join((lib / (mod + ".py")).read_text().splitlines()[a:b])
    out = []
    for i, (mod, k) in enumerate((("shutil", 3), ("difflib", 2))):
        f = textwrap.dedent(funcs(mod)[k])
        out.append(dict(id=f"edit_{mod}", kind="edit", split="custom", messages=[dict(role="user", content=
            "Rewrite this Python function with full type hints and a complete Google-style docstring. Keep the body identical. Output only code.\n\n```python\n" + f + "\n```")]))
    for mod, a, b in (("textwrap", 60, 150), ("heapq", 60, 160)):
        out.append(dict(id=f"extract_{mod}", kind="extract", split="custom", messages=[dict(role="user", content=
            "List every function defined in the code below as a JSON array of objects with keys \"name\" and \"signature\". Output only JSON.\n\n```python\n" + chunk(mod, a, b) + "\n```")]))
    for mod, a, b in (("argparse", 100, 220), ("shutil", 200, 300)):
        out.append(dict(id=f"summ_{mod}", kind="summarize", split="custom", messages=[dict(role="user", content=
            "Summarize what each function in this code does, one bullet per function, quoting the function name and its parameters.\n\n```python\n" + chunk(mod, a, b) + "\n```")]))
    return out


def pick_ctx_subset(tasks):
    want = [("code", "dev"), ("code", "heldout"), ("prose", "heldout"), ("terminal", "heldout"), ("edit", "custom"), ("extract", "custom"), ("summarize", "custom")]
    sel = []
    for k, s in want:
        c = [t for t in tasks if t["kind"] == k and t["split"] == s]
        if c: sel.append(c[0])
    return sel


@torch.no_grad()
def run(m, tok, n_new=192, ctxs=(2048, 8192), emit=print, j=None, live=None):
    live = live or LIVE
    ref = RefNorth(m)
    j = j or Jsonl()
    eos = m.config.eos_token_id
    eos = eos if isinstance(eos, int) else eos[0]
    base = [r for r in load_workloads() if r["split"] in ("dev", "heldout")]
    tasks = base + custom_tasks()
    cells = [("natural", tasks, None, 2048 + n_new + 40)] + [("ctx%d" % c, pick_ctx_subset(tasks), c, c + n_new + 40) for c in ctxs]
    for cname, ts, tl, Lmax in cells:
        gn = GraphNorth(ref, Lmax, "bmm")
        kv = gn.new_kv()
        steps = {1: gn.make_step(1, kv)}
        for T in range(2, TMAX + 1):
            steps[T] = gn.make_step(T, kv, logits="all")
        # cost table (device graph time, ms) for the simulator; set on a real position first
        ids0 = build_ids(tok, ts[0], tl) if tl else build_ids(tok, ts[0])
        gn.eager_prefill(ids0[:-1], kv)
        cost = {}
        for T, s in steps.items():
            s.set(ids0[-T:], len(ids0) - T); cost[T] = time_graph(s)[0]  # REAL tokens: repeated tokens would route to few experts and under-cost
        j.add(phase="cost_table", cell=cname, Lmax=Lmax, cost_ms={str(k): v for k, v in cost.items()})
        emit(cname, "graphs captured; cost T=1/4/9/13:", [round(cost[t], 2) for t in (1, 4, 9, 13)])
        for t in ts:
            ids = build_ids(tok, t, tl) if tl else build_ids(tok, t)
            P = len(ids)
            if P + n_new + 2 > Lmax:
                emit("skip too long", t["id"], P); continue
            gn.eager_prefill(ids[:-1], kv)
            s1 = steps[1]
            best, toks = None, None
            for rep in range(2):
                tk, cur = [], ids[-1]
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for i in range(n_new):
                    s1.set([cur], P - 1 + i); cur = int(s1.run()[0, 0]); tk.append(cur)
                    if cur == eos: break
                dt = time.perf_counter() - t0
                if best is None or dt < best: best = dt
                toks = tk if toks is None else toks
            n = len(toks)
            plain_tok_s = n / best
            j.add(phase="trace", cell=cname, task=t["id"], kind=t["kind"], split=t["split"], prompt_len=P, n=n, plain_tok_s=plain_tok_s, plain_ms_per_tok=1e3 * best / n,
                  gen=toks, prompt_ids=ids if not tl else None)
            # ---- offline simulation: every proposer x gamma vs the plain trace ----
            for name in pld.PROPOSERS:
                for g in SIM_G:
                    r = pld.simulate(name, g, ids, toks, cost, eos)
                    j.add(phase="sim", cell=cname, task=t["id"], kind=t["kind"], split=t["split"], prompt_len=P, proposer=name, gamma=g, **r)
            # ---- live wall clock ----
            for name, g in live:
                def verify(tcur, drafts):
                    T = 1 + len(drafts)
                    stp = steps[T]; stp.set([tcur[-1]] + drafts, len(tcur) - 1)
                    return stp.run()[0].tolist()
                torch.cuda.synchronize(); t0 = time.perf_counter()
                out, rounds = pld.spec_loop(pld.make(name), ids, n, g, verify, eos)
                torch.cuda.synchronize(); dt = time.perf_counter() - t0
                fd = next((i for i, (a, b) in enumerate(zip(out, toks)) if a != b), None) if out != toks else None
                sim = pld.simulate(name, g, ids, toks, cost, eos)
                j.add(phase="live", cell=cname, task=t["id"], kind=t["kind"], split=t["split"], prompt_len=P, proposer=name, gamma=g, n=len(out), n_plain=n,
                      identical=out == toks, first_div=fd, spec_tok_s=len(out) / dt, plain_tok_s=plain_tok_s, speedup_wall=(len(out) / dt) / plain_tok_s,
                      sim_speedup=sim["speedup"], sim_tok_per_round=sim["tok_per_round"], **pld.summarize_rounds(rounds, len(out)))
            emit(cname, t["id"], "plain", round(plain_tok_s, 1), "tok/s;", " ".join("%s_g%d=%.2fx" % (r["proposer"], r["gamma"], r["speedup_wall"])
                 for r in map(__import__("json").loads, j.lines[-len(live):])))
        del gn, kv, steps
        torch.cuda.empty_cache()
    return j
