"""Exact-v2 decode on the real model (H100): v1 exact path vs v2 variants IN THE SAME LAUNCH.
Variants (all FastNorth(exact=True)):
  v1      : _forward_exact (H15/H17 path)
  s       : v2 code, v1 tiles, concurrent attention || MoE streams  -> must be BITWISE equal to v1 (same kernels, same order per value)
  t       : v2 tuned tiles, no streams
  ts      : v2 tuned tiles + streams (the candidate)
Phases:
  step    : graph step ms at T=1,2,4,8,13 at two real contexts, per variant; v1-vs-s logits bitwise; per-variant T-invariance
            (row j of the T=13 block logits == the T=1 logits computed at that position, bitwise)
  traffic : the 99-prompt real-traffic mix (H15): per task plain tok/s for v1 and ts, ts spec (sam2 g12 fixed) tok/s, exactness of ts
            spec vs ts plain, and v1 plain vs s plain token equality."""
import json, pathlib, time, traceback
import statistics as st
import torch

from common import Jsonl, _ids
from northstar.refmodel import RefNorth
from graph_north import GraphNorth, time_graph
from fast_north import FastNorth
import pld

WL = pathlib.Path(__file__).resolve().parent / "workloads_real" / "realtraffic.jsonl"
BUCKETS = (1280, 2560, 5120, 10240)
TMAX = 13


def variants(lin_cfg, moe_cfg):
    return {"v1": None, "s": dict(on=True, streams=True, lin_cfg={}, moe_cfg=None),
            "t": dict(on=True, streams=False, lin_cfg=lin_cfg, moe_cfg=moe_cfg), "ts": dict(on=True, streams=True, lin_cfg=lin_cfg, moe_cfg=moe_cfg)}


def _apply(X, v):
    if v is None:
        X.x2 = False
        X.kcfg_blk = None
        X.mp_min = None
        X.attn_kw = {}
    else:
        X.set_x2(**v)


def _cap(X, kv, v, Ts, logits="all", argmax=True):
    _apply(X, v)
    return {T: X.make_step(T, kv, logits=logits, argmax=argmax) for T in Ts}


@torch.no_grad()
def phase_step(m, tok, j, lin_cfg, moe_cfg, extra=None):
    ref = RefNorth(m)
    tasks = [json.loads(l) for l in WL.read_text().splitlines() if l.strip()]
    items = sorted(((t, _ids(tok, [dict(x) for x in t["messages"]])) for t in tasks), key=lambda r: len(r[1]))
    picks = [items[len(items) // 4], items[int(len(items) * 0.9)]]
    Lmax = max(len(i) for _, i in picks) + 64
    X = FastNorth(ref, Lmax, exact=True); gb = GraphNorth(ref, Lmax, "bmm")
    kv = X.new_kv()
    V = dict(variants(lin_cfg, moe_cfg))
    V.update(extra or {})
    Ts = (1, 2, 4, 8, 13)
    steps = {name: _cap(X, kv, v, Ts, argmax=False) for name, v in V.items()}
    for t, ids in picks:
        P = len(ids)
        gb.eager_prefill(ids[:-1], kv)
        toks = ids[-TMAX:]
        lg = {}
        for name in V:
            for T in Ts:
                s = steps[name][T]
                s.set(ids[-1:] + toks[:T - 1] if T > 1 else ids[-1:], P - 1)  # write positions P-1.. (overwrites only scratch beyond P-1)
                med, mn, mx = time_graph(s, n=30, warm=5)
                j.add(phase="step", task=t["id"], ctx=P, variant=name, T=T, ms_med=med, ms_min=mn, ms_max=mx)
            # bitwise logits at T=1 and the T=13 block (rows) for this variant
            s1, s13 = steps[name][1], steps[name][13]
            blk = [ids[-1]] + toks[:12]
            s13.set(blk, P - 1); s13.run(); L13 = s13.out.clone()
            inv = True
            for r in (0, 3, 12):
                gb.eager_prefill(ids[:-1], kv)  # restore the prefix KV (block wrote positions P-1..P+11)
                # replay rows 0..r-1 as single steps so the KV for row r is written by T=1 steps, then compare row r
                for q in range(r):
                    s1.set([blk[q]], P - 1 + q); s1.run()
                s1.set([blk[r]], P - 1 + r); s1.run()
                inv &= bool(torch.equal(s1.out.reshape(-1), L13.reshape(13, -1)[r]))
                if r == 0:
                    lg[name] = s1.out.clone()
            gb.eager_prefill(ids[:-1], kv)
            j.add(phase="invariance", task=t["id"], ctx=P, variant=name, rows_equal=inv)
            print("STEP", t["id"], "P", P, name, " ".join("T%d=%.3f" % (r["T"], r["ms_med"]) for r in [json.loads(l) for l in j.lines[-len(Ts) - 1:-1]]),
                  "inv", inv, flush=True)
        j.add(phase="v1_vs_s", task=t["id"], ctx=P, logits_equal=bool(torch.equal(lg["v1"], lg["s"])),
              ts_vs_v1_argmax_equal=bool(lg["v1"].float().argmax() == lg["ts"].float().argmax()),
              ts_vs_v1_maxabs=float((lg["v1"].float() - lg["ts"].float()).abs().max()))
        print("V1_VS_S", json.loads(j.lines[-1]), flush=True)
    del X, gb, kv, steps
    torch.cuda.empty_cache()


@torch.no_grad()
def phase_traffic(m, tok, j, lin_cfg, moe_cfg, n_new=192, limit=None, cand=None):
    ref = RefNorth(m)
    eos = m.config.eos_token_id
    eos = eos if isinstance(eos, int) else eos[0]
    tasks = [json.loads(l) for l in WL.read_text().splitlines() if l.strip()]
    if limit:
        tasks = tasks[::max(1, len(tasks) // limit)][:limit]
    items = [(t, _ids(tok, [dict(x) for x in t["messages"]])) for t in tasks]
    V = variants(lin_cfg, moe_cfg)
    for bi, edge in enumerate(BUCKETS):
        lo = BUCKETS[bi - 1] if bi else 0
        cell = [(t, ids) for t, ids in items if lo < len(ids) <= edge]
        if not cell:
            continue
        Lmax = edge + n_new + 40
        X = FastNorth(ref, Lmax, exact=True); gb = GraphNorth(ref, Lmax, "bmm")
        kv = X.new_kv()
        p_v1 = _cap(X, kv, V["v1"], (1,), logits="last")[1]
        p_s = _cap(X, kv, V["s"], (1,), logits="last")[1]
        sts = _cap(X, kv, cand or V["ts"], range(1, TMAX + 1))
        print("bucket", edge, "captured; tasks", len(cell), flush=True)
        for t, ids in cell:
            try:
                P = len(ids)
                gb.eager_prefill(ids[:-1], kv)

                def plain(step):
                    tk, cur = [], ids[-1]
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    for i in range(n_new):
                        step.set([cur], P - 1 + i); cur = int(step.run()[0, 0]); tk.append(cur)
                        if cur == eos:
                            break
                    return tk, time.perf_counter() - t0

                plain(p_v1)  # warm
                res = {}
                for name, stp in (("v1", p_v1), ("s", p_s), ("ts", sts[1])):
                    tk, dt = plain(stp)
                    res[name] = (tk, dt)
                tk, dt = plain(p_v1)  # second v1 rep after the others (drift check)
                v1b = len(tk) / dt
                ptoks = res["ts"][0]
                n = len(ptoks)

                def verify(tcur, drafts):
                    T = 1 + len(drafts)
                    stp = sts[T]; stp.set([tcur[-1]] + drafts, len(tcur) - 1)
                    return stp.run()[0].tolist()
                torch.cuda.synchronize(); t0 = time.perf_counter()
                out, rounds = pld.spec_loop(pld.make("sam2"), ids, n, 12, verify, eos, policy=pld.Policy("fixed", 12))
                torch.cuda.synchronize(); sdt = time.perf_counter() - t0
                row = dict(phase="traffic", task=t["id"], kind=t["kind"], prompt_len=P, bucket=edge, n_v1=len(res["v1"][0]), n_ts=n,
                           v1_tok_s=len(res["v1"][0]) / res["v1"][1], v1b_tok_s=v1b, s_tok_s=len(res["s"][0]) / res["s"][1], ts_tok_s=n / res["ts"][1],
                           ts_spec_tok_s=len(out) / sdt, spec_identical=out == ptoks, s_equals_v1=res["s"][0] == res["v1"][0],
                           ts_equals_v1=res["ts"][0] == res["v1"][0], rounds=len(rounds))
                j.add(**row)
                print("TR", t["id"], "P", P, "v1 %.0f/%.0f s %.0f ts %.0f spec %.0f" % (row["v1_tok_s"], v1b, row["s_tok_s"], row["ts_tok_s"], row["ts_spec_tok_s"]),
                      "exact" if row["spec_identical"] else "MISMATCH", "s==v1" if row["s_equals_v1"] else "s!=v1", flush=True)
            except Exception:
                j.add(phase="task_error", task=t["id"], err=traceback.format_exc()[-900:]); print("TASK ERR", t["id"], traceback.format_exc()[-700:], flush=True)
        del X, gb, kv, p_v1, p_s, sts
        torch.cuda.empty_cache()


@torch.no_grad()
def phase_prof(m, tok, j, lin_cfg, moe_cfg, extra=None):
    """Kernel timeline of one graph replay per variant (CUPTI sees kernels inside graphs): wall, busy union, per-kernel sums, and
    ablation variants given in `extra` (name -> set_x2 kwargs)."""
    import collections
    from torch.profiler import profile, ProfilerActivity
    ref = RefNorth(m)
    tasks = [json.loads(l) for l in WL.read_text().splitlines() if l.strip()]
    items = sorted(((t, _ids(tok, [dict(x) for x in t["messages"]])) for t in tasks), key=lambda r: len(r[1]))
    t, ids = items[len(items) // 2]
    P = len(ids)
    X = FastNorth(ref, P + 64, exact=True); gb = GraphNorth(ref, P + 64, "bmm")
    kv = X.new_kv(); gb.eager_prefill(ids[:-1], kv)
    V = dict(variants(lin_cfg, moe_cfg))
    V.update(extra or {})
    for name, v in V.items():
        for T in (1, 13):
            s = _cap(X, kv, v, (T,))[T]
            s.set(ids[-T:] if T > 1 else ids[-1:], P - 1)
            med = time_graph(s, n=30, warm=5)[0]
            for _ in range(3): s.run()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                s.run(); torch.cuda.synchronize()
            ev = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
            iv = sorted((e.time_range.start, e.time_range.end, e.name) for e in ev)
            agg = collections.defaultdict(lambda: [0.0, 0])
            busy, cur_s, cur_e = 0.0, None, None
            for a, b_, nm in iv:
                agg[nm[:48]][0] += (b_ - a) / 1e3; agg[nm[:48]][1] += 1
                if cur_e is None or a > cur_e:
                    if cur_e is not None: busy += cur_e - cur_s
                    cur_s, cur_e = a, b_
                else:
                    cur_e = max(cur_e, b_)
            if cur_e is not None: busy += cur_e - cur_s
            span = (iv[-1][1] - iv[0][0]) / 1e3 if iv else 0
            top = sorted(agg.items(), key=lambda kv_: -kv_[1][0])[:14]
            j.add(phase="prof", variant=name, T=T, ctx=P, graph_ms=med, span_ms=span, busy_ms=busy / 1e3, kernel_sum_ms=sum(v_[0] for v_ in agg.values()),
                  launches=len(iv), top={k: dict(ms=v_[0], n=v_[1]) for k, v_ in top})
            print("PROF", name, "T", T, "graph %.3f span %.3f busy %.3f ksum %.3f launches %d" % (med, span, busy / 1e3, sum(v_[0] for v_ in agg.values()), len(iv)), flush=True)
            for k_, v_ in top: print("   %7.3f ms %4d  %s" % (v_[0], v_[1], k_), flush=True)
            del s
    del X, gb, kv
    torch.cuda.empty_cache()


def run(m, tok, phases, lin_cfg, moe_cfg, limit=0, extra=None, cand=None):
    j = Jsonl()
    j.add(phase="config", lin_cfg={f"{k[0]}x{k[1]}": v for k, v in lin_cfg.items()}, moe_cfg=moe_cfg)
    for ph in phases:
        try:
            if ph == "prof":
                phase_prof(m, tok, j, lin_cfg, moe_cfg, extra)
                continue
            if ph == "step":
                phase_step(m, tok, j, lin_cfg, moe_cfg, extra)
            else:
                phase_traffic(m, tok, j, lin_cfg, moe_cfg, limit=limit or None, cand=cand)
        except Exception:
            j.add(phase="fatal", where=ph, err=traceback.format_exc()[-1500:]); print("FATAL", ph, traceback.format_exc()[-1500:], flush=True)
    return j

