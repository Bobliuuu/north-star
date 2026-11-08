"""PLD block verify INSIDE the cohere-megakernel (pinned 67d0b9ca + the H11 patch, whose defaults are no-ops).

Design finding (docs/PLD_MEGAKERNEL.md S1): no kernel change is needed. A T-row block of ONE sequence is BS=B rows (B in 1,2,4,8, rows
T..B-1 masked) with cache_seqlens p..p+T-1 (per-row RoPE position + causal length) and page-table rows aliasing one sequence's pages;
QKV writes all rows' K/V before the attention barrier. This script drives the engine's own Python decode loop (decode/schedule.py)
with per-step teacher-forced tokens, per-row positions and the page-table aliasing (harness/mk_block.adopt_blocks).

  modal run --detach modal/mk_block.py::main --part s2     # block correctness vs sequential BS=1 (real weights, rollback exercised)
  modal run --detach modal/mk_block.py::main --part s3     # block cost curve T=1..8 at ctx 2K (real prefill) / 8K (synthetic KV)
Results: results/mk_block_<part>_<tag>.json (written by the client from the function's return value) + Volume northstar-results.
"""
import datetime, json, os, re, subprocess, time, pathlib
import modal

PIN = "67d0b9ca22ea3652796b715d1d1863459e0e2c3c"
ROOT = pathlib.Path(__file__).resolve().parent.parent
RATE_H100 = 3.95
weights_vol = modal.Volume.from_name("northstar-weights")
jit_vol = modal.Volume.from_name("northstar-mkjit", create_if_missing=True)
res_vol = modal.Volume.from_name("northstar-results", create_if_missing=True)
CKPT = "/weights/north-mini-code-d11e61a8"

image = (  # identical layers to modal/mk_selfspec.py so the Modal build cache is reused
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "cmake", "ninja-build", "libssl-dev", "build-essential")
    .run_commands(
        "pip install torch --index-url https://download.pytorch.org/whl/cu130",
        "pip install transformers safetensors fastapi uvicorn cohere-melody==0.9.0 ninja einops numpy",
        "pip install flash-attn-3 --index-url https://download.pytorch.org/whl/cu130",
        f"git clone --recurse-submodules https://github.com/cohere-ai/cohere-megakernel /mk && cd /mk && git checkout {PIN} && git submodule update --init --recursive",
        "cd /mk && cmake -S . -B build -G Ninja -DPython_EXECUTABLE=$(which python) -DCMAKE_BUILD_TYPE=Release && cmake --build build 2>&1 | tail -40",
    )
    .add_local_file(str(ROOT / "src/mk_selfspec/mk_selfspec.patch"), "/mk_selfspec.patch", copy=True)
    .run_commands("cd /mk && git apply /mk_selfspec.patch && git diff --stat")
    .env({"LITTLE_JIT_CACHE_DIR": "/jit", "PYTHONUNBUFFERED": "1"})
    .add_local_file(str(ROOT / "harness/mk_block.py"), "/mk/src/mk_block_h.py")
)
app = modal.App("northstar-mk-block", image=image)

SRC = '''
import os, sys, json, time, statistics
sys.path.insert(0, "/mk/src")
import torch
import native, runner
from mk_block_h import adopt_blocks
import decode.schedule as ds
import prefill.torch_prefill as tp

NUM_SMS = 132
PAGE_BLOCK = runner.DEFAULT_PAGE_BLOCK
SNIP = """import heapq, collections, itertools

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = collections.OrderedDict()

    def get(self, key):
        if key not in self.data:
            return None
        self.data.move_to_end(key)
        return self.data[key]

    def put(self, key, value):
        self.data[key] = value
        self.data.move_to_end(key)
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)


def dijkstra(graph, src):
    dist = {node: float("inf") for node in graph}
    dist[src] = 0
    heap = [(0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist
"""


class World:
    """weights + tokenizer + cfg, loaded once."""

    def __init__(self, ckpt):
        from transformers import AutoTokenizer
        native.bind(library_path="/mk/build/libmk_release.so", debug=False)
        self.dev = torch.device("cuda"); self.dt = torch.bfloat16
        self.cfg = tp.NmcConfig.from_checkpoint(ckpt)
        t = time.time()
        self.w = tp.load_weights(checkpoint=ckpt, cfg=self.cfg, device=self.dev, dtype=self.dt,
                                 upgate_chunk=runner.DEFAULT_UPGATE_LAYOUT_CHUNK, moe_upgate_chunk=runner.DEFAULT_MOE_UPGATE_LAYOUT_CHUNK)
        torch.cuda.synchronize()
        self.load_s = time.time() - t
        self.tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=False, local_files_only=True)

    def prompt_text(self, n_tokens):
        ids = self.tok(SNIP * (n_tokens // 150 + 2), add_special_tokens=True)["input_ids"][:n_tokens]
        return self.tok.decode(ids)

    def prefill(self, text, B, max_new=64):
        return tp.run_reference_prefill(prompt=text, batch=B, weights=self.w, tokenizer=self.tok, cfg=self.cfg, device=self.dev,
                                        dtype=self.dt, raw_prompt=True, page_block=PAGE_BLOCK,
                                        prefill_chunk_size=runner.DEFAULT_PREFILL_CHUNK_SIZE, frac_vram_utilization=0.5,
                                        max_new_tokens=max_new, temperature=0.0, top_p=1.0)

    def synth(self, ctx, B, max_new=64):
        return tp.make_synthetic_kv_prefill(cfg=self.cfg, device=self.dev, dtype=self.dt, batch=B, fake_prompt_len=ctx,
                                            max_new_tokens=max_new, page_block=PAGE_BLOCK, frac_vram_utilization=0.5, row_active=None)


class BlockExec:
    """BS=B megakernel decode state over one prefill; step() = one block (T active rows) of ONE sequence via page-table aliasing."""

    def __init__(self, world, pre, steps, alias=True):
        self.w, self.pre, self.alias = world, pre, alias
        self.B = int(pre.input_ids.shape[0]); self.lib = native.ext()
        self.tiling = ds.default_nmc_tiling_for_bs(batch=self.B, moe_combine_atomic_tma=False)
        p = int(pre.prompt_len)
        self.reg = ds.build_nmc_rr_decode_variant_registry(
            cfg=world.cfg, batch=self.B, num_sms=NUM_SMS, device=world.dev, tiling=self.tiling, num_layers=int(world.cfg.num_hidden_layers),
            launch_mode="all", page_block=int(pre.kv_cache.page_block), max_attn_splits=ds.DEFAULT_MAX_ATTN_SPLITS,
            min_attn_chunk=ds.DEFAULT_MIN_ATTN_CHUNK, min_context_len=p + 1, max_context_len=p + steps + 8,
            attn_drain=True, attn_drain_sms=None, moe_drain_sms=None, rr_dependency_affinity=False, ablation=False)
        self.state = ds.make_decode_state(library=self.lib, prefill=pre, weights=world.w, cfg=world.cfg, num_sms=NUM_SMS,
                                          tiling=self.tiling, decode_registry=self.reg)
        t = time.time(); self.reg.ensure_jit_handles(library=self.lib); self.jit_s = time.time() - t
        self.V = pre.kv_cache.page_table[:, 0, :].clone()  # history pages of row 0 (all rows prefilled the same prompt)
        self.first = pre.next_token.reshape(-1).tolist()

    def step(self, tokens, positions, T):
        """tokens/positions: length T (rows 0..T-1 active). Returns (logits[T,V] float32, cuda_ms)."""
        B, kv, st = self.B, self.pre.kv_cache, self.state
        active = [1] * T + [0] * (B - T)
        pos = list(positions) + [positions[0]] * (B - T)
        tok = list(tokens) + [tokens[0]] * (B - T)
        variant, _ = ds._activate_nmc_decode_variant(st, context_len=max(positions) + 1)
        handle = self.reg.jit_handle_for(variant)
        kv.step_decode_positions(pos, active)
        if self.alias:
            pt = kv.page_table; pb = int(kv.page_block)
            self.V = adopt_blocks(self.V, pt[:, T - 1, :], min(positions) // pb, max(positions) // pb, xp=torch)
            pt[:, :, :] = self.V[:, None, :]
        ds._refresh_attn_drain_queue(st, cache_seqlens=pos)
        ds._reset(st)
        ds._seed(st, self.w.w, self.w.cfg, torch.tensor(tok, device=self.w.dev, dtype=torch.long))
        _, cuda_ms = ds._launch_nmc_decode(handle=handle, state=st, device=self.w.dev)
        return st.tensors["lm_logits"][:T].float().clone(), cuda_ms

    def close(self):
        self.reg.close(); self.pre.close()


def margin(lg):
    t = torch.topk(lg, 2); return float(t.values[0] - t.values[1])
'''

def write_lib():
    open("/mk/src/mk_block_lib.py", "w").write(SRC)


@app.function(gpu="H100", timeout=3300, scaledown_window=2, max_containers=1,
              volumes={"/weights": weights_vol, "/jit": jit_vol, "/results": res_vol}, memory=65536)
def run(part: str, tag: str, ctxs: str = "2048,8192", n_ref: int = 24, rounds: int = 14):
    import sys, random, statistics
    t0 = time.time()
    sys.path.insert(0, "/mk/src"); sys.path.insert(0, "/")
    write_lib()
    import torch
    _im = torch.inference_mode(); _im.__enter__()  # engine tensors (page table) are inference tensors; in-place aliasing needs inference mode
    import mk_block_lib as L
    world = L.World(CKPT)
    out = dict(part=part, tag=tag, load_s=world.load_s, rows=[])
    outp = f"/results/mk_block_{part}_{tag}.jsonl"

    def emit(rec, commit=True):
        rec["t"] = round(time.time() - t0, 1)
        print(json.dumps(rec), flush=True)
        out["rows"].append(rec)
        with open(outp, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if commit:
            res_vol.commit(); jit_vol.commit()

    text = world.prompt_text(300 if part == "s2" else 2000)
    # ---- reference: sequential BS=1 greedy, record logits
    pre = world.prefill(text, 1)
    ex = L.BlockExec(world, pre, n_ref)
    p = int(pre.prompt_len); seq = [ex.first[0]]; ref = []
    t = time.time()
    for i in range(n_ref):
        lg, ms = ex.step([seq[i]], [p + i], 1)
        ref.append(lg[0].cpu()); seq.append(int(lg[0].argmax()))
    emit(dict(kind="ref", prompt_len=p, seq=seq, jit_s=round(ex.jit_s, 1), wall_s=round(time.time() - t, 1)))
    ex.close()
    del ex, pre

    if part == "s2":
        rng = random.Random(0)
        for T in (2, 3, 4, 5, 8):
            B = 2 if T == 2 else (4 if T <= 4 else 8)
            pre = world.prefill(text, B)
            ex = L.BlockExec(world, pre, n_ref)
            assert int(pre.prompt_len) == p
            c, rnd, prev_rej = 0, 0, False
            stat = dict(rows=0, argmax_mismatch=0, mismatch_margin_gt_025=0, max_abs_dlogit=0.0, rounds=0, accept_hist=[])
            while c + T <= n_ref:
                a = rng.randint(0, T - 1)  # simulated accepted drafts; rows > a get WRONG tokens, their KV must be harmless after rollback
                toks = [seq[c + j] if j <= a else rng.randrange(1000, 20000) for j in range(T)]
                lg, ms = ex.step(toks, [p + c + j for j in range(T)], T)
                for j in range(a + 1):
                    r = ref[c + j]; stat["rows"] += 1
                    b = lg[j].cpu(); e = (b - r).abs()
                    d = float(e.max()); stat["max_abs_dlogit"] = max(stat["max_abs_dlogit"], d)
                    ra = int(r.argmax()); top = float(r[ra]); tk = torch.topk(r, 2)
                    # per-row diagnostics (jsonl kind=s2row): margin, scale, error at the ref top-1 and at the worst vocab slot,
                    # rank of the REFERENCE token under block logits (0 == argmax agrees), padding and post-reject flags
                    emit(dict(kind="s2row", T=T, B=B, round=rnd, row=j, pos=c + j, acc=a, padded=bool(B > T), prev_rejected=bool(prev_rej),
                              ref_margin=round(float(tk.values[0] - tk.values[1]), 4), ref_top=round(top, 3),
                              ref_rms=round(float(r.float().pow(2).mean().sqrt()), 3),
                              err_max=round(d, 4), err_argmax_tok=int(e.argmax()), err_at_top=round(float(e[ra]), 4),
                              err_rms=round(float(e.pow(2).mean().sqrt()), 4), rel_err_max=round(d / max(abs(top), 1e-6), 4),
                              ref_tok_rank_in_block=int((b > b[ra]).sum()), blk_margin=round(L.margin(b), 4)), commit=False)
                    if int(b.argmax()) != ra:
                        stat["argmax_mismatch"] += 1
                        if L.margin(r) > 0.25:
                            stat["mismatch_margin_gt_025"] += 1
                prev_rej = a < T - 1; rnd += 1
                stat["accept_hist"].append(a); stat["rounds"] += 1
                c += a + 1
            emit(dict(kind="s2", T=T, B=B, jit_s=round(ex.jit_s, 1), **stat))
            ex.close(); del ex, pre
    else:
        for ctx in [int(x) for x in ctxs.split(",")]:
            for T in (1, 2, 3, 4, 5, 6, 7, 8):
                B = 1 if T == 1 else (2 if T == 2 else (4 if T <= 4 else 8))
                pre = world.prefill(text, B) if ctx <= 2048 else world.synth(ctx, B)
                ex = L.BlockExec(world, pre, rounds + 10)
                pp = int(pre.prompt_len); ms_l = []
                for r in range(rounds + 2):
                    c = r % (n_ref - T)  # windows of real tokens; positions advance like a committed stream
                    lg, ms = ex.step(seq[c:c + T], [pp + r + j for j in range(T)], T)
                    if r >= 2:
                        ms_l.append(ms)
                emit(dict(kind="s3", ctx=ctx, T=T, B=B, prompt_len=pp, ms_median=round(statistics.median(ms_l), 4),
                          ms_min=round(min(ms_l), 4), ms_all=[round(x, 3) for x in ms_l], jit_s=round(ex.jit_s, 1)))
                ex.close(); del ex, pre
    out["secs"] = time.time() - t0
    return out


@app.local_entrypoint()
def main(part: str = "s2", tag: str = "m1", ctxs: str = "2048,8192"):
    r = run.remote(part, tag, ctxs)
    secs = r["secs"]
    (ROOT / f"results/mk_block_{part}_{tag}.json").write_text(json.dumps(r, indent=1))
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    with open(ROOT / "results/SPEND.md", "a") as f:
        f.write(f"| {ts} | mk_block {part} {tag} | H100 | {secs:.0f} | {secs/3600*RATE_H100:.2f} | |\n")
    print("secs", secs, "est $%.2f" % (secs / 3600 * RATE_H100))
