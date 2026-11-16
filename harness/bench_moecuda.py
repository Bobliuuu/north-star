"""moe_fused.cu on North's real decode path (exact v2 x2 executor, T=1), H29's setup: dev code prompt padded to ctx 2048 /
8192, T=1 CUDA-graph replay (median of 50) and a 192-token live greedy decode, all decoders on the same KV and host.
  exact_v2      FastNorth(exact=True).set_x2(**recommended(Lmax))       the shipped headline (H24/H29), Triton MoE
  mc_perarch    exact_v2 with FastNorth.moe_cuda = moe_fused.cu (per-arch codegen defaults, sass-lane sweep)
  mc_upstream   the same with -DNS_TUNE_UPSTREAM (the kernel before the sweep)
Exactness evidence per moe_cuda decoder: 192 greedy tokens vs exact_v2 (first divergence), the same live decode twice
(run-to-run determinism: fp32 atomics have no fixed order), and an op-level diff of the MoE output vs the Triton path
on real hidden states of every layer (one forward of the prompt's last token)."""
import time, traceback
import torch

from common import Jsonl, build_ids, by_split, load_workloads
from northstar.refmodel import RefNorth
from graph_north import GraphNorth, time_graph
from fast_north import FastNorth
import fast_kernels as fk
import fast_kernels_blk as fb
import fast_kernels_x2 as f2
import moe_cuda

N_NEW = 192


def _live(step, ids, P, eos):
    out, cur = [], ids[-1]
    torch.cuda.synchronize(); t = time.perf_counter()
    for i in range(N_NEW):
        step.set([cur], P - 1 + i); cur = int(step.run()[0, 0]); out.append(cur)
        if cur == eos:
            break
    torch.cuda.synchronize()
    return out, len(out) / (time.perf_counter() - t)


def _first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


@torch.no_grad()
def op_check(m, mc, ids, dev):
    """MoE output of every layer on real hidden states: moe_cuda vs the Triton blk path (what exact v2 uses at T=1)."""
    cfg = m.config
    k, Dm, I = cfg.num_experts_per_tok, cfg.hidden_size, cfg.intermediate_size
    emb = m.model.embed_tokens(torch.tensor([[ids[-1]]], device=dev))
    h = emb[0].to(torch.bfloat16).contiguous()
    rows = []
    for li, layer in enumerate(m.model.layers):
        mlp = layer.mlp
        if not hasattr(mlp, "gate"):
            continue
        hn = layer.input_layernorm(h).contiguous()
        lg = (hn @ mlp.gate.weight.t()).contiguous()
        idx = torch.empty(1, k, dtype=torch.int32, device=dev); sc = torch.empty(1, k, dtype=torch.bfloat16, device=dev)
        fk.router_topk(lg, idx, sc, k)
        hb = torch.empty(k, I, dtype=torch.bfloat16, device=dev); ob = torch.empty(k, Dm, dtype=torch.bfloat16, device=dev)
        ref = fb.moe_gemv_blk(hn, idx, sc, mlp.experts.gate_up_proj, mlp.experts.down_proj, hb, ob, None).float().sum(0)
        got = mc(hn, idx, sc, mlp.experts.gate_up_proj, mlp.experts.down_proj)[0]
        got2 = mc(hn, idx, sc, mlp.experts.gate_up_proj, mlp.experts.down_proj)[0].clone()
        got = mc(hn, idx, sc, mlp.experts.gate_up_proj, mlp.experts.down_proj)[0].clone()
        rel = float((got - ref).norm() / ref.norm().clamp_min(1e-30))
        bf_eq = float((got.to(torch.bfloat16) == ref.to(torch.bfloat16)).float().mean())
        rows.append(dict(layer=li, rel_l2=rel, bf16_equal_frac=bf_eq, self_maxabs=float((got - got2).abs().max())))
        h = (h + ref.to(torch.bfloat16)).contiguous()  # walk forward on the reference path (attention omitted: op check only)
    return rows


@torch.no_grad()
def run(m, tok, sos, ctxs=(2048, 8192), reps=50):
    """sos: {"mc_perarch": path, "mc_upstream": path} shared libraries from moe_cuda.build."""
    j = Jsonl()
    dev = next(m.parameters()).device
    eos = m.config.eos_token_id; eos = eos if isinstance(eos, int) else eos[0]
    ref = RefNorth(m)
    cfg = m.config
    code = [r for r in by_split(load_workloads(), "dev") if r["kind"] == "code"]
    mcs = {name: moe_cuda.MoeCuda(so, cfg.hidden_size, cfg.intermediate_size, cfg.num_experts_per_tok, dev) for name, so in sos.items()}
    try:
        ids0 = build_ids(tok, code[0], 2048)
        for name, mc in mcs.items():
            rows = op_check(m, mc, ids0, dev)
            j.add(phase="op_check", decoder=name, layers=rows,
                  rel_l2_max=max(r["rel_l2"] for r in rows), bf16_equal_frac_min=min(r["bf16_equal_frac"] for r in rows),
                  self_maxabs_max=max(r["self_maxabs"] for r in rows))
            print("OPCHECK %s rel_l2 max %.2e  bf16-equal frac min %.4f  run-to-run maxabs %.2e" % (
                name, max(r["rel_l2"] for r in rows), min(r["bf16_equal_frac"] for r in rows), max(r["self_maxabs"] for r in rows)), flush=True)
    except Exception:
        j.add(phase="error", where="op_check", err=traceback.format_exc()[-1500:]); print("ERR op_check", traceback.format_exc()[-1500:], flush=True)
    for ctx in ctxs:
        try:
            ids = build_ids(tok, code[0], ctx); P = len(ids); Lmax = P + N_NEW + 64
            base = GraphNorth(ref, Lmax, "bmm"); kv = base.new_kv()
            decs = {}
            x2 = FastNorth(ref, Lmax, exact=True); x2.set_x2(**f2.recommended(Lmax)); decs["exact_v2"] = x2
            for name, mc in mcs.items():
                d = FastNorth(ref, Lmax, exact=True); d.set_x2(**f2.recommended(Lmax)); d.moe_cuda = mc; decs[name] = d
            res, toks = {}, {}
            for name, gn in decs.items():
                base.eager_prefill(ids[:-1], kv)
                s = gn.make_step(1, kv)
                base.eager_prefill(ids[:-1], kv)
                s.set([ids[-1]], P - 1)
                med, mn, mx = time_graph(s, n=reps)
                base.eager_prefill(ids[:-1], kv)
                out, tps = _live(s, ids, P, eos)
                row = dict(phase="decode", ctx=P, decoder=name, graph_ms=med, ms_min=mn, ms_max=mx, live_tok_s=tps, n=len(out))
                if name != "exact_v2":
                    base.eager_prefill(ids[:-1], kv)
                    out2, tps2 = _live(s, ids, P, eos)
                    row.update(live_tok_s_2=tps2, tokens_equal_exact_v2=out == toks["exact_v2"], first_diff_vs_exact_v2=_first_diff(out, toks["exact_v2"]),
                               run_to_run_equal=out == out2, run_to_run_first_diff=_first_diff(out, out2))
                res[name] = med; toks[name] = out
                j.add(**row)
                print("DECODE ctx %d %-12s graph %.3f ms (min %.3f) live %.1f tok/s n=%d %s" % (
                    P, name, med, mn, tps, len(out), {k: v for k, v in row.items() if k.startswith(("tokens", "first", "run"))}), flush=True)
                del s; torch.cuda.empty_cache()
            j.add(phase="summary", ctx=P, speedup_vs_exact_v2={k: res["exact_v2"] / v for k, v in res.items()})
            print("SUMMARY ctx %d" % P, {k: round(res["exact_v2"] / v, 4) for k, v in res.items()}, flush=True)
            del base, decs, kv; torch.cuda.empty_cache()
        except Exception:
            j.add(phase="error", ctx=ctx, err=traceback.format_exc()[-1500:]); print("ERR", ctx, traceback.format_exc()[-1500:], flush=True)
            torch.cuda.empty_cache()
    return j


# ---------------------------------------------------------------- non-exact executor: T=1 and verify blocks (H13 setup)
@torch.no_grad()
def op_check_block(m, mc, ids, dev, T=13):
    """Per MoE layer, a T-token block of real hidden states: moe_cuda's union path vs the Triton union-of-experts GEMV."""
    cfg = m.config
    k, Dm, I = cfg.num_experts_per_tok, cfg.hidden_size, cfg.intermediate_size
    h = m.model.embed_tokens(torch.tensor([ids[-T:]], device=dev))[0].to(torch.bfloat16).contiguous()
    rows = []
    for li, layer in enumerate(m.model.layers):
        mlp = layer.mlp
        if not hasattr(mlp, "gate"):
            continue
        hn = layer.input_layernorm(h).contiguous()
        lg = (hn @ mlp.gate.weight.t()).contiguous()
        idx = torch.empty(T, k, dtype=torch.int32, device=dev); sc = torch.empty(T, k, dtype=torch.bfloat16, device=dev)
        fk.router_topk(lg, idx, sc, k)
        hb = torch.empty(T * k, I, dtype=torch.bfloat16, device=dev); ob = torch.empty(T * k, Dm, dtype=torch.bfloat16, device=dev)
        ref = fb.moe_gemv_blk(hn, idx, sc, mlp.experts.gate_up_proj, mlp.experts.down_proj, hb, ob, None).float().view(T, k, Dm).sum(1)
        got = mc(hn, idx, sc, mlp.experts.gate_up_proj, mlp.experts.down_proj).clone()
        rows.append(dict(layer=li, rel_l2=float((got - ref).norm() / ref.norm().clamp_min(1e-30)),
                         bf16_equal_frac=float((got.to(torch.bfloat16) == ref.to(torch.bfloat16)).float().mean())))
        h = (h + ref.to(torch.bfloat16)).contiguous()
    return rows


@torch.no_grad()
def run_fast(m, tok, so, n_new=192, n_tasks=4, TMAX=13, proposer=("sam2", 12), reps=2):
    """FastNorth non-exact (H9/H13 executor) with and without moe_cuda: verify-block cost per T, then plain and PLD
    speculative decoding (H13's spec loop and best proposer) on natural dev/heldout tasks."""
    import pld
    j = Jsonl()
    dev = next(m.parameters()).device
    eos = m.config.eos_token_id; eos = eos if isinstance(eos, int) else eos[0]
    cfg = m.config
    ref = RefNorth(m)
    mc = moe_cuda.MoeCuda(so, cfg.hidden_size, cfg.intermediate_size, cfg.num_experts_per_tok, dev)
    tasks = [r for r in load_workloads() if r["split"] in ("dev", "heldout")][:n_tasks]
    try:
        rows = op_check_block(m, mc, build_ids(tok, tasks[0]), dev)
        j.add(phase="op_check_block", T=13, layers=rows, rel_l2_max=max(r["rel_l2"] for r in rows), bf16_equal_frac_min=min(r["bf16_equal_frac"] for r in rows))
        print("OPCHECK_BLOCK T=13 rel_l2 max %.2e bf16-equal min %.4f" % (max(r["rel_l2"] for r in rows), min(r["bf16_equal_frac"] for r in rows)), flush=True)
    except Exception:
        j.add(phase="error", where="op_check_block", err=traceback.format_exc()[-1500:]); print("ERR opblock", traceback.format_exc()[-1500:], flush=True)
    Lmax = 2048 + n_new + 40
    fast = FastNorth(ref, Lmax)
    fmc = FastNorth(ref, Lmax); fmc.moe_cuda = mc
    gb = GraphNorth(ref, Lmax, "bmm"); kv = fast.new_kv()
    execs = {}
    for name, ex in (("fast", fast), ("fast_mc", fmc)):
        st = {1: ex.make_step(1, kv)}
        for T in range(2, TMAX + 1):
            st[T] = ex.make_step(T, kv, logits="all")
        execs[name] = st
    ids0 = build_ids(tok, tasks[0])
    cost = {}
    for name, st in execs.items():
        gb.eager_prefill(ids0[:-1], kv)
        cost[name] = {}
        for T, s in st.items():
            s.set(ids0[-T:], len(ids0) - T); cost[name][T] = time_graph(s, n=50)[0]
    j.add(phase="cost_table", cost_ms={n: {str(t): v for t, v in c.items()} for n, c in cost.items()},
          ratio_mc_over_fast={str(t): cost["fast_mc"][t] / cost["fast"][t] for t in cost["fast"]})
    print("COST T: fast / fast_mc ms", {t: (round(cost["fast"][t], 3), round(cost["fast_mc"][t], 3)) for t in (1, 2, 4, 9, 13)}, flush=True)
    pname, g = proposer
    for t in tasks:
        try:
            ids = build_ids(tok, t); P = len(ids)
            gb.eager_prefill(ids[:-1], kv)
            res = {}
            for name, st in execs.items():
                best, toks = None, None
                for rep in range(reps):
                    tk, cur = [], ids[-1]
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    for i in range(n_new):
                        st[1].set([cur], P - 1 + i); cur = int(st[1].run()[0, 0]); tk.append(cur)
                        if cur == eos:
                            break
                    dt = time.perf_counter() - t0
                    best = dt if best is None or dt < best else best
                    toks = tk if toks is None else toks
                plain_tps = len(toks) / best
                sbest, sout = None, None
                for rep in range(reps):
                    def verify(tcur, drafts, st=st):
                        T = 1 + len(drafts)
                        s = st[T]; s.set([tcur[-1]] + drafts, len(tcur) - 1)
                        return s.run()[0].tolist()
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    out, rounds = pld.spec_loop(pld.make(pname), ids, len(toks), g, verify, eos)
                    torch.cuda.synchronize(); dt = time.perf_counter() - t0
                    sbest = dt if sbest is None or dt < sbest else sbest
                    sout = out if sout is None else sout
                res[name] = dict(plain_tok_s=plain_tps, spec_tok_s=len(sout) / sbest, plain=toks, spec=sout, **pld.summarize_rounds(rounds, len(sout)))
            f, c = res["fast"], res["fast_mc"]
            row = dict(phase="live", task=t["id"], prompt_len=P, proposer=pname, gamma=g,
                       fast_plain_tok_s=f["plain_tok_s"], fast_spec_tok_s=f["spec_tok_s"], mc_plain_tok_s=c["plain_tok_s"], mc_spec_tok_s=c["spec_tok_s"],
                       mc_plain_vs_fast_plain=c["plain_tok_s"] / f["plain_tok_s"], mc_spec_vs_fast_spec=c["spec_tok_s"] / f["spec_tok_s"],
                       mc_spec_vs_fast_plain=c["spec_tok_s"] / f["plain_tok_s"],
                       mc_plain_first_diff_vs_fast=_first_diff(c["plain"], f["plain"]), mc_spec_first_diff_vs_own_plain=_first_diff(c["spec"], c["plain"]),
                       fast_spec_first_diff_vs_own_plain=_first_diff(f["spec"], f["plain"]),
                       fast_tok_per_round=f.get("tok_per_round"), mc_tok_per_round=c.get("tok_per_round"),
                       fast_mean_accepted=f.get("mean_accepted_per_verify"), mc_mean_accepted=c.get("mean_accepted_per_verify"))
            j.add(**row)
            print("LIVE", t["id"], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k not in ("phase",)}, flush=True)
        except Exception:
            j.add(phase="task_error", task=t["id"], err=traceback.format_exc()[-1200:]); print("TASK ERR", t["id"], traceback.format_exc()[-1200:], flush=True)
    return j
