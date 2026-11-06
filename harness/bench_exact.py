"""exact-spec: batch-invariant FastNorth (exact=True) vs the previous FastNorth (exact=False), PLD sam2 g12, H100.
Sections:  e = bitwise row-invariance on the real model (T=1 sequential vs a T=13 block, graph replay vs eager) + logits vs HF
           c = graph cost table T=1..13 for both executors
           b = live bench: plain exact, plain old-fast, spec exact, spec old-fast; identity of spec vs SAME-executor plain; identity vs HF."""
import time, traceback
import torch

from common import Jsonl, build_ids, by_split, load_workloads
from northstar.refmodel import RefNorth
from graph_north import GraphNorth, time_graph
from fast_north import FastNorth
from bench_pld import custom_tasks, pick_ctx_subset
from bench_fastpld import _fd, _top2
import pld

TMAX = 13


@torch.no_grad()
def invariance(m, ref, tok, j):
    rows = by_split(load_workloads(), "dev")
    code = [r for r in rows if r["kind"] == "code"]
    cells = [(r["id"], build_ids(tok, r)) for r in rows] + [("code_ctx2048", build_ids(tok, code[0], 2048)), ("code_ctx8192", build_ids(tok, code[0], 8192))]
    dev = ref.device
    for tid, ids in cells:
        try:
            P = len(ids)
            gb = GraphNorth(ref, P + 64, "bmm"); X = FastNorth(ref, P + 64, exact=True)
            kv = X.new_kv(); gb.eager_prefill(ids[:-1], kv)
            # greedy continuation with the exact T=1 path (eager) gives realistic tokens and the reference per-position logits
            toks = [ids[-1]]; l1 = []
            for i in range(TMAX):
                X.start.fill_(P - 1 + i)
                lg = X.forward(torch.tensor([[toks[-1]]], device=dev), kv, logits="last")[0, 0]
                l1.append(lg.clone()); toks.append(int(lg.argmax()))
            blk = torch.tensor([toks[:TMAX]], device=dev)
            res = {}
            for T in (2, 5, 13):
                X.start.fill_(P - 1); lb = X.forward(blk[:, :T], kv, logits="all")[0]
                res[T] = all(torch.equal(lb[r], l1[r]) for r in range(T))
            # graph replay block vs eager
            st = X.make_step(13, kv, logits="all", argmax=False); st.set(toks[:13], P - 1)
            lgr = st.run()[0].clone()
            graph_eq = all(torch.equal(lgr[r], l1[r]) for r in range(13))
            st1 = X.make_step(1, kv, argmax=False); st1.set(toks[:1], P - 1)
            graph1_eq = torch.equal(st1.run()[0, 0], l1[0])
            # numeric sanity vs HF full forward (rows 0..12, teacher forced)
            hf = m(torch.tensor([ids[:-1] + toks[:13]], device=dev)).logits[0, P - 1:P - 1 + 13].float()
            mx = max(float((l1[r].float() - hf[r]).abs().max()) for r in range(13))
            agree = sum(int(l1[r].argmax()) == int(hf[r].argmax()) for r in range(13))
            j.add(phase="invariance", task=tid, P=P, block_T2_bitwise=res[2], block_T5_bitwise=res[5], block_T13_bitwise=res[13], graph13_eq_eager_t1=graph_eq,
                  graph1_eq_eager=graph1_eq, maxabs_vs_hf=mx, argmax_agree_hf=agree, rows=13)
            print("inv", tid, res, graph_eq, graph1_eq, "maxabs_hf %.2f agree %d/13" % (mx, agree), flush=True)
            del gb, X, kv, st, st1; torch.cuda.empty_cache()
        except Exception:
            j.add(phase="inv_error", task=tid, err=traceback.format_exc()[-900:]); print("INV ERR", tid, traceback.format_exc()[-700:], flush=True)


@torch.no_grad()
def cost_table(ref, tok, j, ctxs=(2048, 8192)):
    code = [r for r in by_split(load_workloads(), "dev") if r["kind"] == "code"]
    for ctx in ctxs:
        ids = build_ids(tok, code[0], ctx); P = len(ids)
        gb = GraphNorth(ref, P + 64, "bmm"); X = FastNorth(ref, P + 64, exact=True); O = FastNorth(ref, P + 64)
        kv = X.new_kv(); gb.eager_prefill(ids[:-1], kv)
        for name, ex in (("exact", X), ("old_fast", O)):
            for T in (1, 2, 4, 6, 9, 13):
                try:
                    s = ex.make_step(T, kv, logits="all") if T > 1 else ex.make_step(1, kv)
                    s.set(ids[-T:], P - T)
                    med, mn, mx = time_graph(s, n=40)
                    j.add(phase="cost", ctx=P, exec=name, T=T, ms=med, ms_min=mn, ms_max=mx); print("cost", P, name, T, round(med, 3), flush=True)
                    del s
                except Exception:
                    j.add(phase="cost_error", ctx=P, exec=name, T=T, err=traceback.format_exc()[-800:]); print("COST ERR", traceback.format_exc()[-600:], flush=True)
        del gb, X, O, kv; torch.cuda.empty_cache()


@torch.no_grad()
def _margin_x(X, kv, ids, prefix, dev):
    seq = ids + prefix
    X.eager_prefill(seq[:-1], kv); X.start.fill_(len(seq) - 1)
    lg = X.forward(torch.tensor([[seq[-1]]], device=dev), kv, logits="last")[0, 0].float()
    mg, top = _top2(lg)
    return dict(top2_margin=mg, top2=top)


@torch.no_grad()
def bench(m, tok, j, n_new=192, ctxs=(2048, 8192), live=(("sam2", 12),), reps=2, hf=True, tag=""):
    ref = RefNorth(m)
    dev = ref.device
    eos = m.config.eos_token_id
    eos = eos if isinstance(eos, int) else eos[0]
    tasks = [r for r in load_workloads() if r["split"] in ("dev", "heldout")] + custom_tasks()
    cells = [("natural", tasks, None, 2048 + n_new + 40)] + [("ctx%d" % c, pick_ctx_subset(tasks), c, c + n_new + 40) for c in ctxs]
    for cname, ts, tl, Lmax in cells:
        X = FastNorth(ref, Lmax, exact=True); O = FastNorth(ref, Lmax); gb = GraphNorth(ref, Lmax, "bmm")
        kv = X.new_kv()
        stepsX = {1: X.make_step(1, kv)}; stepsO = {1: O.make_step(1, kv)}
        for T in range(2, TMAX + 1):
            stepsX[T] = X.make_step(T, kv, logits="all"); stepsO[T] = O.make_step(T, kv, logits="all")
        print(cname, "captured", flush=True)
        for t in ts:
            try:
                ids = build_ids(tok, t, tl) if tl else build_ids(tok, t)
                P = len(ids)
                if P + n_new + 2 > Lmax:
                    print("skip too long", t["id"], P); continue
                gb.eager_prefill(ids[:-1], kv)

                def plain(step):
                    best, toks = None, None
                    for rep in range(reps):
                        tk, cur = [], ids[-1]
                        torch.cuda.synchronize(); t0 = time.perf_counter()
                        for i in range(n_new):
                            step.set([cur], P - 1 + i); cur = int(step.run()[0, 0]); tk.append(cur)
                            if cur == eos: break
                        dt = time.perf_counter() - t0
                        if best is None or dt < best: best = dt
                        toks = tk if toks is None else toks
                        if tk != toks: toks = toks  # plain is deterministic per executor; recorded below
                    return toks, len(toks) / best

                tX, tpsX = plain(stepsX[1]); tO, tpsO = plain(stepsO[1])
                th = None
                if hf and cname != "ctx8192":
                    th = m.generate(torch.tensor([ids], device=dev), max_new_tokens=n_new, do_sample=False)[0, P:].tolist()
                    if eos in th: th = th[:th.index(eos) + 1]
                n = len(tX)
                j.add(phase="trace", cell=cname, task=t["id"], kind=t["kind"], tag=tag, prompt_len=P, n=n, plain_exact_tok_s=tpsX, plain_old_tok_s=tpsO,
                      fd_exact_vs_old=_fd(tX, tO), fd_exact_vs_hf=_fd(tX, th) if th is not None else "na", fd_old_vs_hf=_fd(tO, th) if th is not None else "na",
                      exact_eq_hf=(tX == th) if th is not None else "na", old_eq_hf=(tO == th) if th is not None else "na")
                for name, g in live:
                    for which, steps, tplain, tps_plain in (("exact", stepsX, tX, tpsX), ("old", stepsO, tO, tpsO)):
                        for rep in range(reps):
                            def verify(tcur, drafts, steps=steps):
                                T = 1 + len(drafts)
                                stp = steps[T]; stp.set([tcur[-1]] + drafts, len(tcur) - 1)
                                return stp.run()[0].tolist()
                            torch.cuda.synchronize(); t0 = time.perf_counter()
                            out, rounds = pld.spec_loop(pld.make(name), ids, len(tplain), g, verify, eos)
                            torch.cuda.synchronize(); dt = time.perf_counter() - t0
                            fd = _fd(out, tplain)
                            mg = None
                            if fd is not None and fd < min(len(out), len(tplain)) and rep == 0 and which == "exact":
                                try: mg = _margin_x(X, kv, ids, tplain[:fd], dev)
                                except Exception: mg = "err"
                            j.add(phase="live", cell=cname, task=t["id"], kind=t["kind"], tag=tag, prompt_len=P, exec=which, proposer=name, gamma=g, rep=rep, n=len(out),
                                  n_plain=len(tplain), identical_same_exec=out == tplain, fd_vs_same_exec=fd, identical_hf=(out == th) if th is not None else "na",
                                  fd_vs_hf=_fd(out, th) if th is not None else "na", spec_tok_s=len(out) / dt, plain_same_exec_tok_s=tps_plain, plain_old_tok_s=tpsO,
                                  plain_exact_tok_s=tpsX, margin_at_div=mg, **pld.summarize_rounds(rounds, len(out)))
                            gb.eager_prefill(ids[:-1], kv) if False else None
                last = [__import__("json").loads(l) for l in j.lines if '"phase": "live"' in l][-2 * reps * len(live):]
                print(cname, t["id"], "exact plain %.1f old plain %.1f;" % (tpsX, tpsO), " ".join("%s:%s%.0f/%s" % (r["exec"], "=" if r["identical_same_exec"] else "!", r["spec_tok_s"], r["rep"]) for r in last), flush=True)
            except Exception:
                j.add(phase="task_error", cell=cname, task=t["id"], err=traceback.format_exc()[-900:]); print("TASK ERR", t["id"], traceback.format_exc()[-700:], flush=True)
        del X, O, gb, kv, stepsX, stepsO
        torch.cuda.empty_cache()


def run(m, tok, sections="ecb", tag="", reps=2, hf=True, n_new=192):
    ref = RefNorth(m); j = Jsonl()
    for sec, fn in (("e", lambda: invariance(m, ref, tok, j)), ("c", lambda: cost_table(ref, tok, j)),
                    ("b", lambda: bench(m, tok, j, n_new=n_new, reps=reps, hf=hf, tag=tag))):
        if sec in sections:
            try:
                fn()
            except Exception:
                j.add(phase="fatal", sec=sec, err=traceback.format_exc()[-900:]); print("FATAL", sec, traceback.format_exc()[-900:], flush=True)
    return j
