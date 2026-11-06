"""North-fast-pld: zero-cost PLD/suffix speculation on the FAST executor (FastNorth T=1 decode + FastNorth multi-query verify blocks), H100.
Sections:  v = verify-block validation vs HF full forward / GraphNorth block (logit diffs, argmax agreement, MARGINS at disagreements)
           c = block cost table (graph ms, T=1..13) fast vs GraphNorth
           b = live bench (natural + ctx2048 + ctx8192 cells): plain graphed PyTorch, plain fast, PLD configs on fast verify blocks, >=2 reps,
               token identity vs plain-fast / plain-graphed / HF generate, margins at every divergence."""
import statistics as st, time, traceback
import torch

from common import Jsonl, build_ids, by_split, load_workloads
from northstar.refmodel import RefNorth
from graph_north import GraphNorth, time_graph
from fast_north import FastNorth
from bench_pld import custom_tasks, pick_ctx_subset
import pld

TMAX = 13
LIVE = [("sam2", 12), ("pld3", 8)]


def _top2(lg):
    t = lg.float().topk(2)
    return float(t.values[0] - t.values[1]), [int(i) for i in t.indices]


@torch.no_grad()
def validate(m, ref, tok, j, NC=16, Ts=(2, 3, 5, 8, 13)):
    rows = by_split(load_workloads(), "dev")
    code = [r for r in rows if r["kind"] == "code"]
    cells = [(r["id"], build_ids(tok, r)) for r in rows] + [("code_ctx2048", build_ids(tok, code[0], 2048)), ("code_ctx8192", build_ids(tok, code[0], 8192))]
    for tid, ids in cells:
        try:
            P = len(ids)
            dev = ref.device
            hf = m.generate(torch.tensor([ids], device=dev), max_new_tokens=NC, do_sample=False)[0, P:].tolist()
            nc = len(hf)
            truth = m(torch.tensor([ids + hf], device=dev)).logits[0, P - 1:P - 1 + nc].float()  # row i predicts hf[i]
            base = GraphNorth(ref, P + 64, "bmm"); fast = FastNorth(ref, P + 64)
            kv = fast.new_kv(); base.eager_prefill(ids[:-1], kv)
            for T in Ts:
                if T > nc:
                    continue
                blk = torch.tensor([[ids[-1]] + hf[:T - 1]], device=dev)
                base.start.fill_(P - 1); lb = base.forward(blk, kv, logits="all")[0].float()
                fast.start.fill_(P - 1); lf = fast.forward(blk, kv, logits="all")[0].float()
                tr = truth[:T]
                dis = [i for i in range(T) if int(lf[i].argmax()) != int(tr[i].argmax())]
                disb = [i for i in range(T) if int(lb[i].argmax()) != int(tr[i].argmax())]
                marg = []
                for i in dis:
                    mg, top = _top2(tr[i])
                    marg.append(dict(pos=i, hf_top2_margin=mg, hf_gap_truth_minus_fast_tok=float(tr[i][int(tr[i].argmax())] - tr[i][int(lf[i].argmax())])))
                j.add(phase="val_block", task=tid, prompt_len=P, T=T, maxabs_fast_vs_hf=float((lf - tr).abs().max()), maxabs_base_vs_hf=float((lb - tr).abs().max()),
                      maxabs_fast_vs_base=float((lf - lb).abs().max()), argmax_agree_fast_hf=T - len(dis), argmax_agree_base_hf=T - len(disb), T_rows=T,
                      fast_disagree=marg, base_disagree_pos=disb, finite=bool(torch.isfinite(lf).all()))
                print("val", tid, T, "fast-hf %.3f base-hf %.3f fast-base %.3f agree %d/%d (base %d)" % (float((lf - tr).abs().max()), float((lb - tr).abs().max()),
                      float((lf - lb).abs().max()), T - len(dis), T, T - len(disb)), marg, flush=True)
            # T=1 fast path at the same position, for reference
            fast.start.fill_(P - 1)
            l1 = fast.forward(torch.tensor([[ids[-1]]], device=dev), kv)[0, 0].float()
            j.add(phase="val_t1", task=tid, maxabs_fast_t1_vs_hf=float((l1 - truth[0]).abs().max()), agree=int(l1.argmax()) == int(truth[0].argmax()))
            del base, fast, kv; torch.cuda.empty_cache()
        except Exception:
            j.add(phase="val_error", task=tid, err=traceback.format_exc()[-1000:]); print("VAL ERR", tid, traceback.format_exc()[-700:], flush=True)


@torch.no_grad()
def cost_table(ref, tok, j, ctxs=(2048, 8192)):
    code = [r for r in by_split(load_workloads(), "dev") if r["kind"] == "code"]
    for ctx in ctxs:
        ids = build_ids(tok, code[0], ctx); P = len(ids)
        Lmax = P + 64
        base = GraphNorth(ref, Lmax, "bmm"); fast = FastNorth(ref, Lmax)
        kv = fast.new_kv(); base.eager_prefill(ids[:-1], kv)
        sb = base.make_step(1, kv); sb.set(ids[-1:], P - 1)
        j.add(phase="cost", ctx=P, exec="graphed_pytorch", T=1, ms=time_graph(sb, n=40)[0])
        for T in range(1, TMAX + 1):
            try:
                s = fast.make_step(T, kv, logits="all") if T > 1 else fast.make_step(1, kv)
                s.set(ids[-T:], P - T)
                med, mn, mx = time_graph(s, n=40)
                j.add(phase="cost", ctx=P, exec="fast", T=T, ms=med, ms_min=mn, ms_max=mx)
                print("cost", P, T, round(med, 3), flush=True)
                if T in (4, 9, 13):
                    sb2 = base.make_step(T, kv, logits="all"); sb2.set(ids[-T:], P - T)
                    j.add(phase="cost", ctx=P, exec="graphed_pytorch", T=T, ms=time_graph(sb2, n=20)[0]); del sb2
                del s
            except Exception:
                j.add(phase="cost_error", ctx=P, T=T, err=traceback.format_exc()[-800:]); print("COST ERR", T, traceback.format_exc()[-600:], flush=True)
        del base, fast, kv; torch.cuda.empty_cache()


def _margin(m, fast, kv, ids, prefix, a, b, use_hf, dev):
    """Teacher-forced on prompt+prefix: logit margins between tokens a and b (a = reference executor's token, b = other)."""
    seq = ids + prefix
    out = {}
    if use_hf:
        lg = m(torch.tensor([seq], device=dev)).logits[0, -1].float()
        mg, top = _top2(lg)
        out["hf"] = dict(top2_margin=mg, top2=top, logit_a_minus_b=float(lg[a] - lg[b]))
    fast.eager_prefill(seq[:-1], kv)
    fast.start.fill_(len(seq) - 1)
    lg = fast.forward(torch.tensor([[seq[-1]]], device=dev), kv)[0, 0].float()
    mg, top = _top2(lg)
    out["fast_t1"] = dict(top2_margin=mg, top2=top, logit_a_minus_b=float(lg[a] - lg[b]))
    return out


def _fd(a, b):
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), None if len(a) == len(b) else n)


@torch.no_grad()
def bench(m, tok, j, n_new=192, ctxs=(2048, 8192), live=None, reps=2, hf=True, tag=""):
    live = live or LIVE
    ref = RefNorth(m)
    dev = ref.device
    eos = m.config.eos_token_id
    eos = eos if isinstance(eos, int) else eos[0]
    base_tasks = [r for r in load_workloads() if r["split"] in ("dev", "heldout")]
    tasks = base_tasks + custom_tasks()
    cells = [("natural", tasks, None, 2048 + n_new + 40)] + [("ctx%d" % c, pick_ctx_subset(tasks), c, c + n_new + 40) for c in ctxs]
    for cname, ts, tl, Lmax in cells:
        fast = FastNorth(ref, Lmax); gb = GraphNorth(ref, Lmax, "bmm")
        kv = fast.new_kv()
        s1 = fast.make_step(1, kv); sb = gb.make_step(1, kv)
        steps = {1: s1}
        for T in range(2, TMAX + 1):
            steps[T] = fast.make_step(T, kv, logits="all")
        ids0 = build_ids(tok, ts[0], tl) if tl else build_ids(tok, ts[0])
        gb.eager_prefill(ids0[:-1], kv)
        cost = {}
        for T, s in steps.items():
            s.set(ids0[-T:], len(ids0) - T); cost[T] = time_graph(s)[0]
        sb.set(ids0[-1:], len(ids0) - 1)
        j.add(phase="cost_table", cell=cname, tag=tag, Lmax=Lmax, cost_ms={str(k): v for k, v in cost.items()}, base_t1_ms=time_graph(sb)[0])
        print(cname, "captured; fast cost T=1/4/9/13", [round(cost[t], 2) for t in (1, 4, 9, 13)], flush=True)
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
                        j.add(phase="plain_rep", cell=cname, task=t["id"], tag=tag, exec="base" if step is sb else "fast", rep=rep, n=len(tk), tok_s=len(tk) / dt)
                        if best is None or dt < best: best = dt
                        toks = tk if toks is None else toks
                    return toks, len(toks) / best

                tb, tps_b = plain(sb)
                tf, tps_f = plain(s1)
                th = None
                if hf and cname != "ctx8192":
                    th = m.generate(torch.tensor([ids], device=dev), max_new_tokens=n_new, do_sample=False)[0, P:].tolist()
                    if eos in th: th = th[:th.index(eos) + 1]
                j.add(phase="trace", cell=cname, task=t["id"], kind=t["kind"], split=t["split"], tag=tag, prompt_len=P, n=len(tf), plain_base_tok_s=tps_b, plain_fast_tok_s=tps_f,
                      fd_fast_vs_base=_fd(tf, tb), fd_fast_vs_hf=_fd(tf, th) if th is not None else "na", fd_base_vs_hf=_fd(tb, th) if th is not None else "na", gen=tf)
                divs = {}  # fd -> (ref_tok, other_tok, what)
                if _fd(tf, tb) is not None: divs.setdefault(("fast_vs_base", _fd(tf, tb)), None)
                if th is not None and _fd(tf, th) is not None: divs.setdefault(("fast_vs_hf", _fd(tf, th)), None)
                n = len(tf)
                spec_divs = []
                for name, g in live:
                    for rep in range(reps):
                        def verify(tcur, drafts):
                            T = 1 + len(drafts)
                            stp = steps[T]; stp.set([tcur[-1]] + drafts, len(tcur) - 1)
                            return stp.run()[0].tolist()
                        torch.cuda.synchronize(); t0 = time.perf_counter()
                        out, rounds = pld.spec_loop(pld.make(name), ids, n, g, verify, eos)
                        torch.cuda.synchronize(); dt = time.perf_counter() - t0
                        fdf, fdb = _fd(out, tf), _fd(out, tb)
                        fdh = _fd(out, th) if th is not None else "na"
                        if fdf is not None and rep == 0 and fdf < len(out) and fdf < len(tf): spec_divs.append((name, g, fdf, out[fdf]))
                        j.add(phase="live", cell=cname, task=t["id"], kind=t["kind"], split=t["split"], tag=tag, prompt_len=P, proposer=name, gamma=g, rep=rep, n=len(out), n_plain=n,
                              identical_fast=out == tf, fd_vs_fast=fdf, identical_base=out == tb, fd_vs_base=fdb, identical_hf=(out == th) if th is not None else "na", fd_vs_hf=fdh,
                              spec_tok_s=len(out) / dt, plain_fast_tok_s=tps_f, plain_base_tok_s=tps_b, speedup_vs_fast=(len(out) / dt) / tps_f, speedup_vs_base=(len(out) / dt) / tps_b,
                              **pld.summarize_rounds(rounds, len(out)))
                # margins at every distinct divergence (teacher-forced on the common prefix; a = reference token, b = the diverging token)
                for (what, fd) in sorted(divs):
                    if fd >= n: continue
                    if what == "fast_vs_base": a, b = tb[fd], tf[fd]
                    elif what == "fast_vs_hf": a, b = th[fd], tf[fd]
                    try:
                        mg = _margin(m, fast, kv, ids, tf[:fd], a, b, th is not None, dev)
                        j.add(phase="margin", cell=cname, task=t["id"], tag=tag, what=what, fd=fd, tok_a=a, tok_b=b, **mg)
                    except Exception:
                        j.add(phase="margin_error", task=t["id"], err=traceback.format_exc()[-400:])
                for (name, g, fdf, stok) in spec_divs:
                    if fdf < min(n, len(tf)):
                        try:
                            mg = _margin(m, fast, kv, ids, tf[:fdf], tf[fdf], stok, th is not None, dev)
                            j.add(phase="margin", cell=cname, task=t["id"], tag=tag, what="spec_%s_g%d_vs_fast" % (name, g), fd=fdf, tok_a=tf[fdf], tok_b=stok, **mg)
                        except Exception:
                            j.add(phase="margin_error", task=t["id"], err=traceback.format_exc()[-400:])
                print(cname, t["id"], "base %.1f fast %.1f tok/s;" % (tps_b, tps_f), " ".join("%s_g%d=%.2fx" % (r["proposer"], r["gamma"], r["speedup_vs_fast"])
                      for r in map(__import__("json").loads, [l for l in j.lines if '"phase": "live"' in l][-len(live) * reps:])), flush=True)
            except Exception:
                j.add(phase="task_error", cell=cname, task=t["id"], err=traceback.format_exc()[-900:]); print("TASK ERR", t["id"], traceback.format_exc()[-700:], flush=True)
        del fast, gb, kv, steps, s1, sb
        torch.cuda.empty_cache()


def run(m, tok, sections="vcb", tag="", reps=2, hf=True, live_set="", n_new=192):
    ref = RefNorth(m); j = Jsonl()
    live = [(a.split(":")[0], int(a.split(":")[1])) for a in live_set.split(",")] if live_set else None
    for sec, fn in (("v", lambda: validate(m, ref, tok, j)), ("c", lambda: cost_table(ref, tok, j)),
                    ("b", lambda: bench(m, tok, j, n_new=n_new, live=live, reps=reps, hf=hf, tag=tag))):
        if sec in sections:
            try:
                fn()
            except Exception:
                j.add(phase="fatal", sec=sec, err=traceback.format_exc()[-900:]); print("FATAL", sec, traceback.format_exc()[-900:], flush=True)
    return j
