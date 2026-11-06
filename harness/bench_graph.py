"""H100 phase (2)+(5): CUDA-graphed North decode / block-verify (K=1..16) / component split / self-spec wall speedup.
All timings are CUDA-graph replays (no launch gaps) of HF-module weights via GraphNorth."""
import statistics as st, time
import numpy as np
import torch

from common import Jsonl, build_ids, by_split, load_workloads
from northstar.refmodel import RefNorth
from graph_north import GraphNorth, ALL, time_graph, time_eager
from bench_spec import skip_set


def mem():
    return torch.cuda.memory_allocated() / 2**30


@torch.no_grad()
def run_blocks(m, tok, ctxs=(2048, 8192), KS=range(1, 17), NGEN=40, check_eager=True, attn_impl="bmm"):
    ref = RefNorth(m)
    j = Jsonl()
    code = [r for r in by_split(load_workloads(), "dev") if r["kind"] == "code"]
    for ctx in ctxs:
        ids = build_ids(tok, code[0], ctx)
        P = len(ids)
        gn = GraphNorth(ref, P + NGEN + 24, attn_impl)
        kv = gn.new_kv()
        gn.eager_prefill(ids[:-1], kv)
        # greedy generation with the graphed T=1 step (real tokens)
        s1 = gn.make_step(1, kv)
        toks, cur = [], ids[-1]
        for i in range(NGEN):
            s1.set([cur], P - 1 + i); cur = int(s1.run()[0, 0]); toks.append(cur)
        tf = [ids[-1]] + toks  # teacher-forced inputs at positions P-1 ...
        sb = gn.make_step(8, kv, logits="all"); sb.set(tf[:8], P - 1)
        pb = sb.run()[0].tolist()
        j.add(phase="g_block_consistency", attn_impl=attn_impl, ctx=P, block8_argmax_equals_sequential=int(sum(a == b for a, b in zip(pb, toks[:8]))), of=8)
        del sb
        if check_eager and ctx == ctxs[0]:
            from northstar.spec import plain_greedy
            pg = plain_greedy(ref, ids, 16)["tokens"]
            first = next((i for i, (a, b) in enumerate(zip(pg, toks)) if a != b), 16)
            j.add(phase="g_correctness", attn_impl=attn_impl, ctx=P, graph_vs_refnorth_first_div_in_16=first, same16=bool(pg == toks[:16]))
            print("graph vs ref first div:", first, flush=True)
            del pg
        # ---- block verify timing, graphed ----
        base_ms = None
        for K in KS:
            step = gn.make_step(K, kv, logits="all")
            step.set(tf[:K], P - 1)
            med, mn, mx = time_graph(step)
            if K == 1: base_ms = med
            eager = None
            if K in (1, 4, 8, 16):
                eager = time_eager(lambda: (gn.start.fill_(P - 1), gn.forward(step.ids, kv, logits="all")))
            j.add(phase="g_block", attn_impl=attn_impl, ctx=P, K=K, graph_ms=med, graph_ms_min=mn, graph_ms_max=mx, eager_same_fwd_ms=eager,
                  seq_K_decodes_ms=K * base_ms, verify_only_speedup=K * base_ms / med, cost_vs_one_token=med / base_ms)
            print("blk", P, K, round(med, 3), "ms", flush=True)
            del step
        # ---- component split via ablated graphs (differences) ----
        for K in (1, 5, 9, 16):
            def t(parts, logits="last" if K == 1 else "all"):
                s = gn.make_step(K, kv, parts=parts, logits=logits, argmax=False); s.set(tf[:K], P - 1)
                v = time_graph(s)[0]; del s; return v
            full = t(ALL)
            no_exp = t(frozenset({"attn", "router", "lm"}))
            no_attn = t(frozenset({"router", "experts", "lm"}))
            no_lm = t(frozenset({"attn", "router", "experts"}))
            router_only = t(frozenset({"router"}), "none")
            j.add(phase="g_components", attn_impl=attn_impl, ctx=P, K=K, full_ms=full, experts_ms=full - no_exp, attn_ms=full - no_attn, lm_head_ms=full - no_lm,
                  router_norm_embed_resid_ms=router_only, rest_ms=full - (full - no_exp) - (full - no_attn) - (full - no_lm) - router_only,
                  noexperts_ms=no_exp, noattn_ms=no_attn, nolm_ms=no_lm)
            print("comp", P, K, round(full, 3), "exp", round(full - no_exp, 3), "attn", round(full - no_attn, 3), "lm", round(full - no_lm, 3), flush=True)
        # draft-mode single-step costs (graphed): for the cycle model
        for name, sk, r in (("full", 0, None), ("r4", 0, 4), ("r2", 0, 2), ("skip8", 8, None), ("skip8_r4", 8, 4)):
            skip = skip_set(sk)
            for T in (1, 2):
                s = gn.make_step(T, kv, skip=skip, r=r); s.set(tf[:T], P - 1)
                med, mn, mx = time_graph(s)
                j.add(phase="g_draft_step", attn_impl=attn_impl, ctx=P, mode=name, T=T, graph_ms=med)
                print("draft", name, T, round(med, 3), flush=True); del s
        del kv, gn, s1
        torch.cuda.empty_cache()
    return j


@torch.no_grad()
def run_spec(m, tok, ntasks=4, n_new=64, long_ctx=2048, attn_impl="bmm"):
    ref = RefNorth(m)
    j = Jsonl()
    eos = m.config.eos_token_id
    eos = eos if isinstance(eos, int) else eos[0]
    rows = sorted(load_workloads(), key=lambda r: (r["split"] != "heldout", r["id"]))
    cells = [("natural", rows[:ntasks], None, 512)]
    code = [r for r in rows if r["kind"] == "code"]
    if long_ctx:
        cells.append(("ctx%d" % long_ctx, code[:2], long_ctx, long_ctx + n_new + 40))
    for cname, tasks, tl, Lmax in cells:
        gn = GraphNorth(ref, Lmax, attn_impl)
        vkv = gn.new_kv()
        modes = [("r4_g4", 0, 4, 4), ("skip8_g4", 8, None, 4), ("r4_g2", 0, 4, 2), ("r4_g8", 0, 4, 8)]
        s_plain = gn.make_step(1, vkv)
        dk = {}
        for name, sk, r, g in modes:
            skip = skip_set(sk)
            key = (sk, r)
            if key not in dk:
                dkv = gn.new_kv([i for i in range(ref.cfg.num_hidden_layers) if i not in skip])
                dk[key] = (dkv, gn.make_step(1, dkv, skip=skip, r=r), gn.make_step(2, dkv, skip=skip, r=r))
        vsteps = {g: gn.make_step(g + 1, vkv, logits="all") for g in sorted({g for *_, g in modes})}
        print("spec cell", cname, "graphs captured, mem", round(mem(), 1), flush=True)
        for t in tasks:
            ids = build_ids(tok, t, tl) if tl else build_ids(tok, t)
            P = len(ids)
            gn.eager_prefill(ids[:-1], vkv)
            # plain graphed decode
            toks, cur = [], ids[-1]
            ts = []
            for i in range(n_new):
                t0 = time.perf_counter()
                s_plain.set([cur], P - 1 + i); cur = int(s_plain.run()[0, 0]); ts.append(time.perf_counter() - t0)
                toks.append(cur)
                if cur == eos: break
            plain_s = sum(ts); nplain = len(toks)
            j.add(phase="g_plain", cell=cname, task=t["id"], prompt_len=P, n=nplain, ms_per_tok=1e3 * plain_s / nplain, tok_s=nplain / plain_s)
            for name, sk, r, g in modes:
                skip = skip_set(sk)
                dkv, d1, d2 = dk[(sk, r)]
                gn.eager_prefill(ids[:-1], dkv, skip=skip, r=r)
                v = vsteps[g]
                tcur = list(ids); out = []; dlen = P - 1
                acc_l, t_dr, t_ver = [], 0.0, 0.0
                torch.cuda.synchronize(); t0 = time.perf_counter()
                done = False
                while len(out) < nplain and not done:
                    n = len(tcur)
                    feed = tcur[dlen:n]
                    t1 = time.perf_counter()
                    stp = d1 if len(feed) == 1 else d2
                    stp.set(feed, dlen); drafts = [int(stp.run()[0, -1])]
                    dl = dlen + len(feed)
                    for i in range(1, g):
                        d1.set([drafts[-1]], dl); drafts.append(int(d1.run()[0, 0])); dl += 1
                    t2 = time.perf_counter()
                    v.set([tcur[n - 1]] + drafts, n - 1); pred = v.run()[0].tolist()
                    t3 = time.perf_counter()
                    a = 0
                    while a < g and drafts[a] == pred[a]: a += 1
                    new = drafts[:a] + [pred[a]]
                    dlen = n + min(a, g - 1)
                    if eos in new: new = new[: new.index(eos) + 1]; done = True
                    tcur += new; out += new; acc_l.append(a); t_dr += t2 - t1; t_ver += t3 - t2
                spec_s = time.perf_counter() - t0
                out = out[:nplain]
                k = next((i for i, (x, y) in enumerate(zip(out, toks)) if x != y), min(len(out), len(toks)))
                j.add(phase="g_spec", cell=cname, config=name, task=t["id"], prompt_len=P, gamma=g, n_committed=len(out), n_plain=nplain,
                      identical=bool(out == toks), first_div=None if out == toks else k, rounds=len(acc_l), mean_accepted=st.mean(acc_l),
                      committed_per_round=len(out) / len(acc_l), spec_tok_s=len(out) / spec_s, plain_tok_s=nplain / plain_s,
                      speedup_wall=(len(out) / spec_s) / (nplain / plain_s), t_draft_s=t_dr, t_verify_s=t_ver,
                      ms_per_round=1e3 * spec_s / len(acc_l), ms_draft_per_round=1e3 * t_dr / len(acc_l), ms_verify_per_round=1e3 * t_ver / len(acc_l))
                print(cname, name, j.lines[-1][:330], flush=True)
        del gn, vkv, dk, vsteps
        torch.cuda.empty_cache()
    return j
