"""batch>1 exact-mode PLD (sam2 g12) vs plain on the SAME batched exact executor, H100.
Sections: e = bitwise row-invariance of the batched executor vs the single-sequence exact FastNorth (ragged starts, B=3 / 16, T=1/5/13)
          c = graph cost table (B x T)
          b = live bench: per batch size B, disjoint groups of B prompts from one fixed pool; per group per rep: plain run and spec run interleaved.
Uneven progress: every round takes T = 1 + max_b(drafts_b); rows with fewer drafts are padded with their own last token (pred of padded rows is ignored,
their stale KV slots are masked by pos<=qpos and overwritten later); rows that reached n_new/eos keep riding along as dummies (position frozen)."""
import json, pathlib, random, time, traceback
import torch

from common import Jsonl, build_ids, load_workloads, _ids
from northstar.refmodel import RefNorth
from graph_north import GraphNorth, time_graph
from fast_north import FastNorth
from batch_exact import BatchExact
from bench_pld import custom_tasks
import pld

TMAX = 13
WL = pathlib.Path(__file__).resolve().parent / "workloads_real" / "realtraffic.jsonl"


def build_pool(tok, n=32, maxlen=1700):
    nat = [r for r in load_workloads() if r["split"] in ("dev", "heldout")] + custom_tasks()
    items = []
    for t in nat:
        ids = build_ids(tok, t)
        if len(ids) <= maxlen: items.append((t["id"], t["kind"], ids))
    rt = [json.loads(l) for l in WL.read_text().splitlines() if l.strip()]
    rt = rt[::3] + rt[1::3] + rt[2::3]
    for t in rt:
        if len(items) >= n: break
        ids = _ids(tok, [dict(x) for x in t["messages"]])
        if len(ids) <= maxlen: items.append((t["id"], t["kind"], ids))
    random.Random(20261108).shuffle(items)
    return items[:n]


def _set_rows(step, grp, T, real=True):
    """Graph inputs for cost timing. real: each row = its prompt's last T tokens (distinct, real routing); else every slot = the prompt's last
    token (what a mostly-padded spec round looks like). Writes KV only at positions >= P-1, which every later run overwrites (same as the warmup)."""
    ids = [(list(g[2][-T:]) if real else [g[2][-1]] * T) for g in grp]
    step.set(ids, [len(g[2]) - 1 for g in grp])


def _fd(a, b):
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), None if len(a) == len(b) else n)


@torch.no_grad()
def invariance(X, BX, pool, kvs, j):
    dev = X.dev
    for B in (3, 16):
        if B > len(pool): continue
        for T in (1, 5, 13):
            try:
                rows = list(range(B))
                ids = torch.zeros(B, T, dtype=torch.long, device=dev); starts = []
                for s, r in enumerate(rows):
                    P = len(pool[r][2]); starts.append(P - 1)
                    blk = pool[r][2][-T:]
                    ids[s] = torch.tensor(blk, device=dev)
                st = torch.tensor(starts, device=dev)
                lb = BX.forward(ids, st, [kvs[r] for r in rows]).view(B, T, -1).clone()
                ok = []
                for s, r in enumerate(rows):
                    X.start.fill_(starts[s]); l1 = X._forward_exact(ids[s:s + 1], kvs[r])[0]
                    ok.append(bool(torch.equal(lb[s], l1)))
                j.add(phase="invariance", B=B, T=T, rows_equal=ok, all_equal=all(ok))
                print("inv B", B, "T", T, "all_equal", all(ok), ok if not all(ok) else "", flush=True)
            except Exception:
                j.add(phase="inv_error", B=B, T=T, err=traceback.format_exc()[-900:]); print("INV ERR", B, T, traceback.format_exc()[-800:], flush=True)


def run_plain(step, grp, n_new, eos):
    B = len(grp)
    cur = [g[2][-1] for g in grp]; pos = [len(g[2]) - 1 for g in grp]
    out = [[] for _ in range(B)]; done = [False] * B; td = [None] * B
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for i in range(n_new):
        step.set([[c] for c in cur], pos, [[not d] for d in done])
        pred = step.run().tolist(); now = time.perf_counter() - t0
        for b in range(B):
            pos[b] += 1
            if done[b]: continue
            tk = pred[b][0]; out[b].append(tk); cur[b] = tk
            if tk == eos or len(out[b]) >= n_new: done[b] = True; td[b] = now
        if all(done): break
    return out, td, time.perf_counter() - t0, None


def _pick_T(dr, q, cost, done):
    """Cost-aware round width: T maximising expected new tokens / measured graph ms. Row b yields 1 + sum_{i<=min(d_b,T-1)} q_b^i tokens
    (geometric prefix acceptance, q_b = running per-row continuation rate). T=1 (plain step) is always a candidate."""
    tmax = 1 + max(len(d) for d in dr)
    best, bT = -1.0, 1
    for T in range(1, tmax + 1):
        tok = 0.0
        for b, d in enumerate(dr):
            if done[b]: continue
            m = min(len(d), T - 1); qb = q[b]
            tok += 1.0 + (qb * (1 - qb ** m) / (1 - qb) if qb < 1 else m)
        v = tok / cost[T]
        if v > best: best, bT = v, T
    return bT


def run_spec(steps, grp, n_new, eos, name, gamma, cost=None):
    """cost: {T: ms} enables the cost-aware cap (_pick_T); drafts are truncated to T-1 tokens. None = plain PLD (T = 1 + longest draft)."""
    B = len(grp)
    qa = [1.0] * B; qr = [1.0] * B   # per-row accepted / rejected-continuation counts (prior q = 0.5)
    props = [pld.make(name) for _ in grp]
    for p, g in zip(props, grp): p.extend(g[2])
    tcur = [list(g[2]) for g in grp]; out = [[] for _ in range(B)]; done = [False] * B; td = [None] * B
    rounds = 0; slots = 0; used = 0; nT = []; acc = 0; drafted = 0; roundsT = {}; t_dev = 0.0
    torch.cuda.synchronize(); t0 = time.perf_counter()
    while not all(done):
        dr = [(props[b].propose(gamma)[0] if not done[b] else []) for b in range(B)]
        if cost is not None:
            T = _pick_T(dr, [qa[b] / (qa[b] + qr[b]) for b in range(B)], cost, done)
            dr = [d[:T - 1] for d in dr]
        T = 1 + max(len(d) for d in dr)
        ids = []; pos = []; val = []
        for b in range(B):
            row = [tcur[b][-1]] + dr[b]
            ids.append(row + [row[-1]] * (T - len(row))); pos.append(len(tcur[b]) - 1)
            val.append([(not done[b]) and t < len(row) for t in range(T)])
        steps[T].set(ids, pos, val); tg = time.perf_counter()
        pred = steps[T].run().tolist(); now = time.perf_counter() - t0; t_dev += time.perf_counter() - tg
        rounds += 1; roundsT[T] = roundsT.get(T, 0) + 1; slots += B * T
        for b in range(B):
            if done[b]: continue
            d = dr[b]; a = 0
            while a < len(d) and d[a] == pred[b][a]: a += 1
            new = d[:a] + [pred[b][a]]
            if eos in new: new = new[:new.index(eos) + 1]; done[b] = True
            used += 1 + len(d); acc += a; drafted += len(d)
            qa[b] += a; qr[b] += int(a < len(d))
            tcur[b] += new; out[b] += new; props[b].extend(new)
            if len(out[b]) >= n_new: done[b] = True
            if done[b]: td[b] = now
    wall = time.perf_counter() - t0
    out = [o[:n_new] for o in out]
    return out, td, wall, dict(rounds=rounds, mean_T=sum(k * v for k, v in roundsT.items()) / rounds, T_hist=roundsT, slot_util=used / max(1, slots), accepted=acc, drafted=drafted,
                            dev_wall=t_dev, host_wall=wall - t_dev)


@torch.no_grad()
def bench(m, tok, j, sizes=(1, 2, 4, 8, 16, 32), n_new=192, reps=2, name="sam2", gamma=12, pool_n=32, tag="", cc=False):
    ref = RefNorth(m); dev = ref.device
    eos = m.config.eos_token_id; eos = eos if isinstance(eos, int) else eos[0]
    pool = build_pool(tok, pool_n)
    print("pool", len(pool), [len(p[2]) for p in pool], flush=True)
    Lmax = max(len(p[2]) for p in pool) + n_new + 40
    X = FastNorth(ref, Lmax, exact=True); gb = GraphNorth(ref, Lmax, "bmm"); BX = BatchExact(X)
    kvs = []
    for (tid, kind, ids) in pool:
        kv = X.new_kv(); gb.eager_prefill(ids[:-1], kv); kvs.append(kv)
    del gb
    torch.cuda.empty_cache()
    print("prefilled", len(pool), "mem GB", torch.cuda.memory_allocated() / 1e9, flush=True)
    j.add(phase="pool", ids=[p[0] for p in pool], kinds=[p[1] for p in pool], lens=[len(p[2]) for p in pool], Lmax=Lmax, n_new=n_new, eos=eos)
    invariance(X, BX, pool, kvs, j)
    ref1 = {}   # pool index -> plain B=1 output (the single-sequence reference)
    for B in sizes:
        if B > len(pool): continue
        ngrp = len(pool) // B
        # graphs bind kv tensors, so capture per group (cheap relative to the run) with all T
        for gi in range(ngrp):
            idxs = list(range(gi * B, (gi + 1) * B)); grp = [pool[i] for i in idxs]
            try:
                t_c = time.time()
                steps = {T: BX.make_step(B, T, [kvs[i] for i in idxs]) for T in range(1, TMAX + 1)}
                cap_s = time.time() - t_c
                if gi == 0:
                    run_plain(steps[1], grp, 16, eos); run_spec(steps, grp, 32, eos, name, gamma)   # warmup
                    c, cpad = {}, {}
                    for T in (1, 2, 4, 7, 13):
                        # time on REAL distinct tokens (prompt tail), not on whatever the warmup left in the graph inputs: the MoE skips
                        # experts no row routes to, so padded rows (copies of one token) under-state the cost (bs1: live rounds 1.5-2.0x the old table)
                        _set_rows(steps[T], grp, T, real=True); c[T] = time_graph(steps[T], n=20)[0]
                        _set_rows(steps[T], grp, T, real=False); cpad[T] = time_graph(steps[T], n=20)[0]   # control: old-style padded rows
                    j.add(phase="cost", B=B, inputs="prompt_tail", ms={str(k): v for k, v in c.items()}, ms_padded={str(k): v for k, v in cpad.items()})
                    print("cost B", B, c, "padded", cpad, flush=True)
                    if cc:
                        cfull = {}
                        for T in range(1, TMAX + 1):
                            _set_rows(steps[T], grp, T, real=True); cfull[T] = time_graph(steps[T], n=10)[0]
                        j.add(phase="cost_full", B=B, ms={str(k): v for k, v in cfull.items()}); print("cost_full B", B, {k: round(v, 2) for k, v in cfull.items()}, flush=True)
                for rep in range(reps):
                    order = ("plain", "spec") if rep % 2 == 0 else ("spec", "plain")
                    if cc: order = order + ("speccc",) if rep % 2 == 0 else ("speccc",) + order
                    res = {}
                    for w in order:
                        res[w] = (run_plain(steps[1], grp, n_new, eos) if w == "plain" else
                                  run_spec(steps, grp, n_new, eos, name, gamma, cost=cfull if w == "speccc" else None))
                    po, ptd, pw, _ = res["plain"]; so, std, sw, sinfo = res["spec"]
                    if B == 1 and rep == 0: ref1[idxs[0]] = po[0]
                    ntp = [len(o) for o in po]; nts = [len(o) for o in so]
                    j.add(phase="group", B=B, group=gi, rep=rep, tag=tag, tasks=[g[0] for g in grp], kinds=[g[1] for g in grp], prompt_lens=[len(g[2]) for g in grp],
                          n_plain=ntp, n_spec=nts, plain_wall=pw, spec_wall=sw, plain_agg_tok_s=sum(ntp) / pw, spec_agg_tok_s=sum(nts) / sw,
                          plain_seq_tok_s=[n / t for n, t in zip(ntp, ptd)], spec_seq_tok_s=[n / t for n, t in zip(nts, std)],
                          spec_eq_plain_same_B=[a == b for a, b in zip(so, po)],
                          fd_spec_vs_plain=[_fd(a, b) for a, b in zip(so, po)],
                          plain_eq_B1=[(po[k] == ref1[i]) if i in ref1 else "na" for k, i in enumerate(idxs)],
                          spec_eq_B1=[(so[k] == ref1[i]) if i in ref1 else "na" for k, i in enumerate(idxs)], capture_s=cap_s, **sinfo)
                    if cc:
                        co, ctd, cw, cinfo = res["speccc"]; ntc = [len(o) for o in co]
                        j.add(phase="group_cc", B=B, group=gi, rep=rep, tag=tag, cc_wall=cw, cc_agg_tok_s=sum(ntc) / cw, plain_agg_tok_s=sum(ntp) / pw, spec_agg_tok_s=sum(nts) / sw,
                              cc_eq_plain=[a == b for a, b in zip(co, po)], n_cc=ntc, **{"cc_" + k: v for k, v in cinfo.items()})
                        print("  cc B", B, "grp", gi, "rep", rep, "cc %.0f (plain %.0f spec %.0f)" % (sum(ntc) / cw, sum(ntp) / pw, sum(nts) / sw),
                              "cc==plain", sum(a == b for a, b in zip(co, po)), "/", B, "meanT %.2f" % cinfo["mean_T"], flush=True)
                    print("B", B, "grp", gi, "rep", rep, "plain %.0f spec %.0f agg tok/s ratio %.3f" % (sum(ntp) / pw, sum(nts) / sw, (sum(nts) / sw) / (sum(ntp) / pw)),
                          "spec==plain", sum(a == b for a, b in zip(so, po)), "/", B, "meanT %.2f util %.2f" % (sinfo["mean_T"], sinfo["slot_util"]), flush=True)
                del steps
                torch.cuda.empty_cache()
            except Exception:
                j.add(phase="group_error", B=B, group=gi, err=traceback.format_exc()[-1200:]); print("GROUP ERR", B, gi, traceback.format_exc()[-900:], flush=True)
                steps = {}; torch.cuda.empty_cache()


def run(m, tok, tag="", reps=2, n_new=192, sizes=(1, 2, 4, 8, 16, 32), cc=False):
    j = Jsonl()
    try:
        bench(m, tok, j, sizes=sizes, n_new=n_new, reps=reps, tag=tag, cc=cc)
    except Exception:
        j.add(phase="fatal", err=traceback.format_exc()[-1200:]); print("FATAL", traceback.format_exc()[-1200:], flush=True)
    return j
