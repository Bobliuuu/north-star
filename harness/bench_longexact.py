"""Long-context (8K/32K/128K) exact-mode PLD vs plain, FastNorth (exact=True) and old fast (exact=False), H100, greedy.
Same methodology as bench_exact.bench (H14): per task plain exact/old (best of reps), then spec (sam2 g12) reps, same executor, identity vs same-exec plain.
Differences (needed for long ctx): prefill = LCNorth.prefill (chunked SDPA, memory-safe to 128K) into the FastNorth KV (same layout);
exact attention workspaces shared across T (saves ~3 GB at 128K); prompt-index build time of the proposer timed separately (spec tok/s reported with and without it).
emit(row) is called after every row."""
import time, traceback, json
import torch
import fast_kernels_exact as fx
from common import build_ids, load_workloads
from northstar.refmodel import RefNorth
from fast_north import FastNorth
from graph_north import time_graph
from longctx_north import LCNorth
from bench_pld import custom_tasks, pick_ctx_subset
from bench_fastpld import _fd
import pld

TMAX = 13


class TimedProp:
    def __init__(self, p): self.p, self.index_s, self._first = p, 0.0, True
    def extend(self, toks):
        if self._first:
            t = time.perf_counter(); self.p.extend(toks); self.index_s = time.perf_counter() - t; self._first = False
        else:
            self.p.extend(toks)
    def propose(self, g): return self.p.propose(g)


def share_ws(X):
    sh = {}
    orig = X._x_state
    def f(T):
        new = T not in X._x
        b = orig(T)
        if new:
            ws = b["ws"]
            key = tuple(ws.acc.shape)  # share only between workspaces of the same split layout (MP, NS)
            if key not in sh: sh[key] = (ws.acc, ws.m, ws.l)
            else: ws.acc, ws.m, ws.l = sh[key]
        return b
    X._x_state = f


def mem(): return dict(alloc_gb=torch.cuda.memory_allocated() / 1e9, peak_gb=torch.cuda.max_memory_allocated() / 1e9, reserved_gb=torch.cuda.memory_reserved() / 1e9)


@torch.no_grad()
def run(m, tok, emit, ctxs=(8192, 32768, 131072), n_new=192, reps=2, name="sam2", g=12, old=True, x2=False, overlap_index=False):
    """x2=True: the second arm ("old" slots in the rows, arm2="x2") is exact v2 (FastNorth.set_x2(**fast_kernels_x2.recommended())) on the
    SAME FastNorth instance and KV, instead of the old non-exact executor."""
    if x2:
        old = True
    arm2 = "x2" if x2 else "old"
    ref = RefNorth(m); dev = ref.device
    eos = m.config.eos_token_id; eos = eos if isinstance(eos, int) else eos[0]
    cfg = m.config
    emit(kind="model", layers=cfg.num_hidden_layers, max_position_embeddings=getattr(cfg, "max_position_embeddings", None), sliding_window=cfg.sliding_window,
         layer_types_global=sum(t != "sliding_attention" for t in cfg.layer_types), **mem())
    tasks = [r for r in load_workloads() if r["split"] in ("dev", "heldout")] + custom_tasks()
    ts = pick_ctx_subset(tasks)
    for ctx in ctxs:
        cname = "ctx%d" % ctx
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            Lmax = ctx + n_new + 40
            kvgb = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * Lmax * cfg.head_dim * 2 / 1e9
            t0 = time.time()
            X = FastNorth(ref, Lmax, exact=True); share_ws(X)
            O = FastNorth(ref, Lmax) if (old and not x2) else None
            ln = LCNorth(ref, Lmax)
            kv = X.new_kv()
            emit(kind="alloc", cell=cname, Lmax=Lmax, kv_gb=kvgb, s=time.time() - t0, **mem())
            ids0 = build_ids(tok, ts[0], ctx)
            ln.prefill(ids0[:-1], kv, chunk=1024 if ctx <= 32768 else 512)
            torch.cuda.empty_cache()
            emit(kind="prefill_probe", cell=cname, s=time.time() - t0, **mem())
            costO = None
            stepsX = {1: X.make_step(1, kv)}; stepsO = {1: O.make_step(1, kv)} if (old and not x2) else {}
            for T in range(2, TMAX + 1):
                stepsX[T] = X.make_step(T, kv, logits="all")
                if old and not x2: stepsO[T] = O.make_step(T, kv, logits="all")
            if x2:
                import fast_kernels_x2 as f2
                X.set_x2(**f2.recommended(Lmax))
                stepsO = {1: X.make_step(1, kv)}
                for T in range(2, TMAX + 1):
                    stepsO[T] = X.make_step(T, kv, logits="all")
                X.x2 = False
            torch.cuda.synchronize()
            emit(kind="captured", cell=cname, s=time.time() - t0, **mem())
        except Exception:
            emit(kind="alloc_error", cell=cname, err=traceback.format_exc()[-1500:], **mem())
            print("ALLOC/CAPTURE ERR", cname, traceback.format_exc()[-1000:], flush=True)
            X = O = ln = kv = stepsX = stepsO = None; torch.cuda.empty_cache(); continue
        for ti, t in enumerate(ts):
            try:
                ids = build_ids(tok, t, ctx); P = len(ids)
                tp = time.time()
                th, pool = (pld.prebuild(name, ids, 3 * reps) if overlap_index else (None, None))  # index during prefill (H25)
                if ti > 0: ln.prefill(ids[:-1], kv, chunk=1024 if ctx <= 32768 else 512)
                else: ln.prefill(ids[:-1], kv, chunk=1024 if ctx <= 32768 else 512)
                pre_s = time.time() - tp
                if th is not None:
                    th.join()
                idx_wait_s = time.time() - tp - pre_s
                # plain exact, plain old (interleaved per task: X, O, then spec X, spec O reps)
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
                        if toks is None: toks = tk
                        elif tk != toks: emit(kind="plain_nondeterministic", cell=cname, task=t["id"])
                    return toks, len(toks) / best
                tX, tpsX = plain(stepsX[1])
                tO, tpsO = plain(stepsO[1]) if old else ([], float("nan"))
                emit(kind="trace", cell=cname, task=t["id"], tkind=t["kind"], prompt_len=P, n=len(tX), prefill_s=pre_s, plain_exact_tok_s=tpsX, plain_old_tok_s=tpsO, arm2=arm2, overlap_index=bool(overlap_index), index_wait_after_prefill_s=idx_wait_s,
                     fd_exact_vs_old=_fd(tX, tO) if old else "na", **mem())
                arms = [("exact", stepsX, tX, tpsX), ("old", stepsO, tO, tpsO)]
                if x2:
                    if costO is None:  # v2 cost table at this context, timed on this prompt's tail; the replays only write KV at >= P-1
                        costO = {}
                        for T in range(1, TMAX + 1):
                            stepsO[T].set(ids[-T:] if T > 1 else ids[-1:], P - 1)
                            costO[T] = time_graph(stepsO[T], n=8, warm=2)[0]
                        emit(kind="cost", cell=cname, ms=costO)
                    arms.append(("x2cost", stepsO, tO, tpsO))
                for which, steps, tplain, tps_plain in arms:
                    if which == "old" and not old: continue
                    for rep in range(reps):
                        def verify(tcur, drafts, steps=steps):
                            stp = steps[1 + len(drafts)]; stp.set([tcur[-1]] + drafts, len(tcur) - 1)
                            return stp.run()[0].tolist()
                        prop = TimedProp(pool.pop() if pool else pld.make(name))
                        torch.cuda.synchronize(); t0 = time.perf_counter()
                        pol = pld.CostPolicy(g, costO) if which == "x2cost" else None
                        out, rounds = pld.spec_loop(prop, ids, len(tplain), g, verify, eos, policy=pol, prebuilt=bool(overlap_index))
                        torch.cuda.synchronize(); dt = time.perf_counter() - t0
                        fd = _fd(out, tplain)
                        emit(kind="live", cell=cname, task=t["id"], tkind=t["kind"], prompt_len=P, exec=which if which in ("exact", "x2cost") else arm2, proposer=name, gamma=g, rep=rep, n=len(out), n_plain=len(tplain),
                             identical_same_exec=out == tplain, fd_vs_same_exec=fd, spec_tok_s=len(out) / dt, spec_tok_s_excl_index=len(out) / max(1e-9, dt - prop.index_s),
                             index_s=prop.index_s, plain_same_exec_tok_s=tps_plain, **pld.summarize_rounds(rounds, len(out)))
                print(cname, t["id"], "plain exact %.1f old %.1f" % (tpsX, tpsO), flush=True)
            except Exception:
                emit(kind="task_error", cell=cname, task=t["id"], err=traceback.format_exc()[-1200:], **mem())
                print("TASK ERR", cname, t["id"], traceback.format_exc()[-900:], flush=True)
                torch.cuda.empty_cache()
        del X, O, ln, kv, stepsX, stepsO
        torch.cuda.empty_cache()
