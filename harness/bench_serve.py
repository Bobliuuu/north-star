"""Validation of harness/serve.py Engine on H100: for every path, generate() == generate_plain() (exactness) and decode tok/s of both.
Sections: pool (H18-H20 32-prompt pool, B = 1 x 32 prompts, then B = 2/4/8/16/32 on the first B prompts; batched rows also compared
with the B=1 outputs), mix (99-prompt H15 mix at B=1, one Engine per length bucket), long (padded natural prompts at 32K and 128K, B=1)."""
import json, pathlib, time, traceback
import torch

from common import Jsonl, _ids, build_ids, load_workloads
import bench_batch as bb
from bench_pld import custom_tasks, pick_ctx_subset
import serve

WL = pathlib.Path(__file__).resolve().parent / "workloads_real" / "realtraffic.jsonl"
BUCKETS = (1280, 2560, 5120, 10240)


def _cmp(j, sec, key, eng, prompts, n_new, extra=None):
    out, st = eng.generate(prompts, n_new)
    ref, rs = eng.generate_plain(prompts, n_new)
    row = dict(phase=sec, key=key, B=len(prompts), path=st["path"], exact=out == ref, rows_exact=[a == b for a, b in zip(out, ref)],
               tok_s=st["decode_tok_s"], plain_tok_s=rs["decode_tok_s"], speedup=st["decode_tok_s"] / rs["decode_tok_s"],
               prefill_s=st["prefill_s"], index_wait_s=st.get("index_wait_s"), mean_T=st.get("mean_T"), tokens=st["tokens"], **(extra or {}))
    j.add(**row)
    print(sec.upper(), key, "B", row["B"], row["path"], "%.0f vs plain %.0f tok/s (%.3fx)" % (row["tok_s"], row["plain_tok_s"], row["speedup"]),
          "EXACT" if row["exact"] else "MISMATCH", "prefill %.2fs wait %.3fs" % (row["prefill_s"], row["index_wait_s"] or 0), flush=True)
    return out


@torch.no_grad()
def sec_pool(m, tok, j, n_new):
    pool = bb.build_pool(tok, 32)
    eng = serve.Engine(m, max(len(p[2]) for p in pool) + n_new + 40, bmax=32, spec_bmax=2)  # sv1 ran with B=2 speculation on
    eng.warmup((1,))
    single = {}
    for i, p in enumerate(pool):
        single[i] = _cmp(j, "pool", p[0], eng, [p[2]], n_new)[0]
    for B in (2, 4, 8, 16, 32):
        outs = _cmp(j, "pool", "first%d" % B, eng, [p[2] for p in pool[:B]], n_new)
        eq = [outs[b] == single[b] for b in range(B)]
        j.add(phase="pool_vs_single", B=B, rows_equal_B1=eq)
        print("  batched rows == B=1 outputs:", sum(eq), "/", B, flush=True)
    del eng; torch.cuda.empty_cache()


@torch.no_grad()
def sec_mix(m, tok, j, n_new, limit=0):
    tasks = [json.loads(l) for l in WL.read_text().splitlines() if l.strip()]
    if limit:
        tasks = tasks[::max(1, len(tasks) // limit)][:limit]
    items = [(t, _ids(tok, [dict(x) for x in t["messages"]])) for t in tasks]
    for bi, edge in enumerate(BUCKETS):
        lo = BUCKETS[bi - 1] if bi else 0
        cell = [(t, ids) for t, ids in items if lo < len(ids) <= edge]
        if not cell:
            continue
        eng = serve.Engine(m, edge + n_new + 40, bmax=1)
        eng.warmup((1,))
        j.add(phase="mix_cost", bucket=edge, cost=eng.cost1)
        for t, ids in cell:
            try:
                _cmp(j, "mix", t["id"], eng, [ids], n_new, dict(kind=t["kind"], prompt_len=len(ids)))
            except Exception:
                j.add(phase="error", key=t["id"], err=traceback.format_exc()[-800:]); print("ERR", t["id"], traceback.format_exc()[-600:], flush=True)
        del eng; torch.cuda.empty_cache()


@torch.no_grad()
def sec_long(m, tok, j, n_new, ctxs=(32768, 131072), per=2):
    tasks = pick_ctx_subset([r for r in load_workloads() if r["split"] in ("dev", "heldout")] + custom_tasks())
    for ctx in ctxs:
        try:
            eng = serve.Engine(m, ctx + n_new + 64, bmax=1)
            eng.warmup((1,))
            for t in tasks[:per if ctx <= 32768 else 1]:
                ids = build_ids(tok, t, ctx)
                _cmp(j, "long", "%s@%d" % (t["id"], ctx), eng, [ids], n_new, dict(prompt_len=len(ids)))
            del eng; torch.cuda.empty_cache()
        except Exception:
            j.add(phase="error", key="long%d" % ctx, err=traceback.format_exc()[-1200:]); print("ERR long", ctx, traceback.format_exc()[-900:], flush=True)
            torch.cuda.empty_cache()


@torch.no_grad()
def sec_b2(m, tok, j, n_new):
    """B=2 policy check over all 16 disjoint pairs of the pool: cost-capped batched PLD vs batched plain."""
    pool = bb.build_pool(tok, 32)
    eng = serve.Engine(m, max(len(p[2]) for p in pool) + n_new + 40, bmax=2, spec_bmax=2)
    for g in range(16):
        _cmp(j, "b2", "pair%d" % g, eng, [pool[2 * g][2], pool[2 * g + 1][2]], n_new)
    del eng; torch.cuda.empty_cache()


def run(m, tok, sections=("pool", "mix", "long"), n_new=192, limit=0):
    j = Jsonl()
    for s in sections:
        try:
            if s == "pool":
                sec_pool(m, tok, j, n_new)
            elif s == "mix":
                sec_mix(m, tok, j, n_new, limit)
            elif s == "long":
                sec_long(m, tok, j, n_new)
            elif s == "b2":
                sec_b2(m, tok, j, n_new)
        except Exception:
            j.add(phase="fatal", where=s, err=traceback.format_exc()[-1500:]); print("FATAL", s, traceback.format_exc()[-1500:], flush=True)
    return j
