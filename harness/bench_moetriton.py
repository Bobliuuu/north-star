"""Race: the sass-lane Triton port of moe_fused (harness/moe_triton.py) vs north-star's own Triton MoE on the real decode path.
Part A, exact v2 (H29 setup, ctx 2048 / 8192): exact_v2 | tri_fused (deterministic, partials summed by resid_norm, in-kernel
work lists) | tri_det (deterministic, separate reduce kernel) | tri_atomic (fp32 atomics);
graph ms, live tok/s, tokens vs exact_v2, the same decode twice, and exact speculation (sam2 g12) == own plain decode.
Part B, non-exact executor (H13 setup): fast | fast_tri_fused | fast_tri_atomic | fast_tri_det; verify-block cost T=1..13, live plain + sam2 g12.
Plus op checks (T=1 and a 13-token block) against the Triton blk MoE on real hidden states."""
import statistics, time, traceback
import torch

from common import Jsonl, build_ids, by_split, load_workloads
from northstar.refmodel import RefNorth
from graph_north import GraphNorth, time_graph
from fast_north import FastNorth
import fast_kernels_x2 as f2
import moe_triton
import pld
from bench_moecuda import _first_diff, _live, op_check, op_check_block

N_NEW = 192
VARIANTS = tuple(__import__("os").environ.get("NS_TRI_VARIANTS", "tri_fast,tri_fast_atomic,tri_fused,tri_det,tri_atomic").split(","))


class _Comb:
    """For op checks only: sum MoeTriton(fused=True)'s per-(slot, slice) partials to one row per token."""
    def __init__(self, mi):
        self.mi = mi

    def __call__(self, h, idx, sc, gu, dn):
        r = self.mi(h, idx, sc, gu, dn)
        T = idx.shape[0] if idx.dim() == 2 else 1
        return r if r.shape[0] == T else r.view(T, -1, r.shape[-1]).sum(1)


def _spec(st, ids, n, eos, name="sam2", g=12):
    def verify(tcur, drafts):
        s = st[1 + len(drafts)]; s.set([tcur[-1]] + drafts, len(tcur) - 1)
        return s.run()[0].tolist()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out, rounds = pld.spec_loop(pld.make(name), ids, n, g, verify, eos)
    torch.cuda.synchronize()
    return out, len(out) / (time.perf_counter() - t0), pld.summarize_rounds(rounds, len(out))


def _compiled(fn):
    """(n_regs, n_spills, shared bytes, num_warps) of every compiled variant of a Triton kernel (Triton 3.x caches)."""
    out = []
    try:
        for cache in fn.device_caches.values():
            for k in cache[0].values():
                out.append(dict(regs=getattr(k, "n_regs", None), spills=getattr(k, "n_spills", None),
                                shared=getattr(getattr(k, "metadata", None), "shared", None), warps=getattr(getattr(k, "metadata", None), "num_warps", None)))
    except Exception as e:  # noqa: BLE001 - diagnostics only
        out.append(dict(err=repr(e)[:200]))
    return out


@torch.no_grad()
def op_profile(m, impls, ids, dev, Ts=(1, 13), reps=200):
    """The MoE op alone, CUDA-graph captured, on a middle layer's real routing: time, kernel launches, bytes (distinct
    experts x 3*I*H*2) and achieved bandwidth; plus registers / shared memory of the Triton kernels involved."""
    import fast_kernels_blk as fb
    cfg = m.config
    k, Dm, I = cfg.num_experts_per_tok, cfg.hidden_size, cfg.intermediate_size
    layers = [l for l in m.model.layers if hasattr(l.mlp, "gate")]
    layer = layers[len(layers) // 2]; mlp = layer.mlp
    rows = []
    for T in Ts:
        h = layer.input_layernorm(m.model.embed_tokens(torch.tensor([ids[-T:]], device=dev))[0].to(torch.bfloat16)).contiguous()
        lg = (h @ mlp.gate.weight.t()).contiguous()
        idx = torch.empty(T, k, dtype=torch.int32, device=dev); sc = torch.empty(T, k, dtype=torch.bfloat16, device=dev)
        __import__("fast_kernels").router_topk(lg, idx, sc, k)
        nexp = int(torch.unique(idx).numel())
        hb = torch.empty(T * k, I, dtype=torch.bfloat16, device=dev); ob = torch.empty(T * k, Dm, dtype=torch.bfloat16, device=dev)
        cands = {"north_blk": lambda: fb.moe_gemv_blk(h, idx, sc, mlp.experts.gate_up_proj, mlp.experts.down_proj, hb, ob, None)}
        for name, mi in impls.items():
            cands[name] = (lambda mi=mi: mi(h, idx, sc, mlp.experts.gate_up_proj, mlp.experts.down_proj))
        for name, f in cands.items():
            for _ in range(3):
                f()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as pr:
                f(); torch.cuda.synchronize()
            kern = [e.name for e in pr.events() if e.device_type == torch.autograd.DeviceType.CUDA]
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                f()
            for _ in range(10):
                g.replay()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(reps):
                g.replay()
            b.record(); torch.cuda.synchronize()
            us = a.elapsed_time(b) * 1000 / reps
            byt = nexp * 3 * I * Dm * 2
            rows.append(dict(T=T, impl=name, us=us, launches=len(kern), kernels=sorted(set(kern))[:8], distinct_experts=nexp,
                             weight_MB=byt / 1e6, GBs=byt / (us * 1e-6) / 1e9))
            print("PROF T=%d %-16s %8.1f us  launches %d  experts %d  %.0f GB/s" % (T, name, us, len(kern), nexp, byt / (us * 1e-6) / 1e9), flush=True)
    occ = {"moe_port": _compiled(moe_triton._moe_port), "union_table": _compiled(moe_triton._union_table), "moe_reduce": _compiled(moe_triton._moe_reduce),
           "north_gateup_blk": _compiled(fb._gateup_blk), "north_down_blk": _compiled(fb._down_blk)}
    print("OCC", occ, flush=True)
    return rows, occ


@torch.no_grad()
def run(m, tok, parts="AB", ctxs=(2048, 8192), n_tasks=4, TMAX=13, cfg=None):
    j = Jsonl()
    dev = next(m.parameters()).device
    eos = m.config.eos_token_id; eos = eos if isinstance(eos, int) else eos[0]
    c = m.config
    ref = RefNorth(m)
    mk = lambda det, fused=False, reduce="resid": moe_triton.MoeTriton(c.hidden_size, c.intermediate_size, c.num_experts_per_tok, dev,
                                                                       det=det, cfg=cfg, fused=fused, reduce=reduce)
    mks = lambda st: moe_triton.MoeTriton(c.hidden_size, c.intermediate_size, c.num_experts_per_tok, dev, det=True, fused=True, reduce="kernel",
                                          cfg=dict(cfg or {}, num_stages=st))
    mksl = lambda det: moe_triton.MoeTriton(c.hidden_size, c.intermediate_size, c.num_experts_per_tok, dev, det=det, fused=True,
                                            reduce="kernel" if det else "resid", cfg=cfg, sl=32)
    impls = {"tri_fast": mk(True, True, "kernel"), "tri_fast_sl32": mksl(True), "tri_fast_atomic_sl32": mksl(False), "tri_fast_s3": mks(3), "tri_fast_s2": mks(2), "tri_fast_atomic": mk(False, True),
             "tri_fused": mk(True, True), "tri_det": mk(True), "tri_atomic": mk(False)}
    impls = {k: v for k, v in impls.items() if k in VARIANTS}
    j.add(phase="config", cfg={k: dict(v.cfg, det=v.det, fused=v.fused, reduce=v.reduce) for k, v in impls.items()}, triton=__import__("triton").__version__)
    code = [r for r in by_split(load_workloads(), "dev") if r["kind"] == "code"]
    ids0 = build_ids(tok, code[0], 2048)
    for name, mi in impls.items():
        try:
            cm = _Comb(mi)
            r1 = op_check(m, cm, ids0, dev); rb = op_check_block(m, cm, ids0, dev)
            j.add(phase="op_check", impl=name, t1_rel_max=max(r["rel_l2"] for r in r1), t1_bf16eq_min=min(r["bf16_equal_frac"] for r in r1),
                  t1_self_maxabs=max(r["self_maxabs"] for r in r1), t13_rel_max=max(r["rel_l2"] for r in rb), t13_bf16eq_min=min(r["bf16_equal_frac"] for r in rb))
            print("OPCHECK %s T=1 rel max %.2e self %.1e | T=13 rel max %.2e" % (name, max(r["rel_l2"] for r in r1), max(r["self_maxabs"] for r in r1), max(r["rel_l2"] for r in rb)), flush=True)
        except Exception:
            j.add(phase="error", where="op_check " + name, err=traceback.format_exc()[-1500:]); print("ERR op", name, traceback.format_exc()[-1500:], flush=True)
    if "P" in parts:
        try:
            rows, occ = op_profile(m, impls, ids0, dev)
            j.add(phase="op_profile", rows=rows, occupancy=occ)
        except Exception:
            j.add(phase="error", where="op_profile", err=traceback.format_exc()[-1500:]); print("ERR prof", traceback.format_exc()[-1500:], flush=True)
    if "R" in parts:   # interleaved repeats: every decoder's T=1 graph timed `rounds` times in rotating order, same KV
        for ctx in ctxs:
            try:
                ids = build_ids(tok, code[0], ctx); P = len(ids); Lmax = P + N_NEW + 64
                base = GraphNorth(ref, Lmax, "bmm"); kv = base.new_kv()
                stp = {}
                nost = __import__("os").environ.get("NS_TRI_NOSTREAM") == "1"   # diagnostic: also every decoder with streams off
                for name in ["exact_v2"] + list(impls):
                    for streams in ((True, False) if nost else (True,)):
                        d = FastNorth(ref, Lmax, exact=True); d.set_x2(**dict(f2.recommended(Lmax), streams=streams))
                        if name != "exact_v2":
                            d.moe_cuda = impls[name]
                        base.eager_prefill(ids[:-1], kv)
                        stp[name + ("" if streams else "_nostream")] = d.make_step(1, kv)
                base.eager_prefill(ids[:-1], kv)
                names = list(stp)
                ms = {n: [] for n in names}
                for r in range(5):
                    for n in names[r % len(names):] + names[:r % len(names)]:
                        stp[n].set([ids[-1]], P - 1); ms[n].append(time_graph(stp[n], n=50)[0])
                ratio = {n: [a / b for a, b in zip(ms["exact_v2"], ms[n])] for n in names}
                row = dict(phase="exact_repeats", ctx=P, ms=ms, ratio_vs_exact_v2=ratio,
                           median=dict((n, statistics.median(v)) for n, v in ratio.items()), spread=dict((n, max(v) - min(v)) for n, v in ratio.items()))
                j.add(**row)
                print("REPEATS ctx %d" % P, {n: (round(row["median"][n], 4), round(row["spread"][n], 4)) for n in names}, flush=True)
                del stp, base, kv; torch.cuda.empty_cache()
            except Exception:
                j.add(phase="error", where="repeats %d" % ctx, err=traceback.format_exc()[-1500:]); print("ERR rep", ctx, traceback.format_exc()[-1500:], flush=True)
                torch.cuda.empty_cache()
    if "A" in parts:
        for ctx in ctxs:
            try:
                ids = build_ids(tok, code[0], ctx); P = len(ids); Lmax = P + N_NEW + 64
                base = GraphNorth(ref, Lmax, "bmm"); kv = base.new_kv()
                decs = {}
                for name in ["exact_v2"] + list(impls):
                    d = FastNorth(ref, Lmax, exact=True); d.set_x2(**f2.recommended(Lmax))
                    if name != "exact_v2":
                        d.moe_cuda = impls[name]
                    decs[name] = d
                res, toks = {}, {}
                for name, gn in decs.items():
                    base.eager_prefill(ids[:-1], kv)
                    s = gn.make_step(1, kv)
                    base.eager_prefill(ids[:-1], kv)
                    s.set([ids[-1]], P - 1)
                    med, mn, mx = time_graph(s, n=50)
                    base.eager_prefill(ids[:-1], kv)
                    out, tps = _live(s, ids, P, eos)
                    base.eager_prefill(ids[:-1], kv)
                    out2, tps2 = _live(s, ids, P, eos)
                    row = dict(phase="exact_decode", ctx=P, decoder=name, graph_ms=med, ms_min=mn, live_tok_s=tps, live_tok_s_2=tps2, n=len(out),
                               run_to_run_equal=out == out2, run_to_run_first_diff=_first_diff(out, out2))
                    if name != "exact_v2":
                        row.update(tokens_equal_exact_v2=out == toks["exact_v2"], first_diff_vs_exact_v2=_first_diff(out, toks["exact_v2"]))
                    if impls.get(name) is None and ctx == ctxs[0] or impls.get(name) is not None and impls[name].det:   # exact speculation: verify blocks T<=13 must reproduce the own plain decode
                        st = {1: s}
                        for T in range(2, TMAX + 1):
                            st[T] = gn.make_step(T, kv, logits="all")
                        base.eager_prefill(ids[:-1], kv)
                        so, stps, summ = _spec(st, ids, len(out), eos)
                        row.update(spec_tok_s=stps, spec_equal_own_plain=so == out, spec_first_diff_vs_own_plain=_first_diff(so, out), spec_tok_per_round=summ["tok_per_round"])
                        del st
                    res[name] = med; toks[name] = out
                    j.add(**row)
                    print("EXACT ctx %d %-10s graph %.3f ms live %.1f tok/s %s" % (P, name, med, tps, {k: v for k, v in row.items() if k.startswith(("tokens", "first", "run", "spec"))}), flush=True)
                    del s; torch.cuda.empty_cache()
                j.add(phase="exact_summary", ctx=P, speedup_vs_exact_v2={k: res["exact_v2"] / v for k, v in res.items()})
                print("SUMMARY exact ctx %d" % P, {k: round(res["exact_v2"] / v, 4) for k, v in res.items()}, flush=True)
                del base, decs, kv; torch.cuda.empty_cache()
            except Exception:
                j.add(phase="error", where="exact ctx %d" % ctx, err=traceback.format_exc()[-1500:]); print("ERR exact", ctx, traceback.format_exc()[-1500:], flush=True)
                torch.cuda.empty_cache()
    if "B" in parts:
        try:
            tasks = [r for r in load_workloads() if r["split"] in ("dev", "heldout")][:n_tasks]
            Lmax = 2048 + N_NEW + 40
            ex = {}
            for name in ["fast"] + ["fast_" + k for k in impls]:
                f = FastNorth(ref, Lmax)
                if name != "fast":
                    f.moe_cuda = impls[name[len("fast_"):]]
                ex[name] = f
            gb = GraphNorth(ref, Lmax, "bmm"); kv = ex["fast"].new_kv()
            steps = {}
            for name, f in ex.items():
                st = {1: f.make_step(1, kv)}
                for T in range(2, TMAX + 1):
                    st[T] = f.make_step(T, kv, logits="all")
                steps[name] = st
            idsb = build_ids(tok, tasks[0])
            cost = {}
            for name, st in steps.items():
                gb.eager_prefill(idsb[:-1], kv)
                cost[name] = {}
                for T, s in st.items():
                    s.set(idsb[-T:], len(idsb) - T); cost[name][T] = time_graph(s, n=50)[0]
            j.add(phase="cost_table", cost_ms={n: {str(t): v for t, v in c_.items()} for n, c_ in cost.items()},
                  ratio_vs_fast={n: {str(t): cost[n][t] / cost["fast"][t] for t in cost["fast"]} for n in cost})
            print("COST ms T=1/2/4/9/13", {n: [round(cost[n][t], 3) for t in (1, 2, 4, 9, 13)] for n in cost}, flush=True)
            for t in tasks:
                ids = build_ids(tok, t); P = len(ids)
                gb.eager_prefill(ids[:-1], kv)
                row = dict(phase="fast_live", task=t["id"])
                plain = {}
                for name, st in steps.items():
                    best = None
                    for rep in range(2):
                        tk, tps = _live(st[1], ids, P, eos)
                        best = tps if best is None or tps > best else best
                        plain.setdefault(name, tk)
                    so, stps, summ = _spec(st, ids, len(plain[name]), eos)
                    row.update({f"{name}_plain_tok_s": best, f"{name}_spec_tok_s": stps, f"{name}_spec_first_diff_vs_own_plain": _first_diff(so, plain[name]),
                                f"{name}_tok_per_round": summ["tok_per_round"]})
                for name in ex:
                    if name != "fast":
                        row[f"{name}_plain_first_diff_vs_fast"] = _first_diff(plain[name], plain["fast"])
                j.add(**row)
                print("FASTLIVE", t["id"], {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items() if k != "phase"}, flush=True)
            del ex, steps, kv, gb; torch.cuda.empty_cache()
        except Exception:
            j.add(phase="error", where="fast", err=traceback.format_exc()[-1500:]); print("ERR fast", traceback.format_exc()[-1500:], flush=True)
    return j
