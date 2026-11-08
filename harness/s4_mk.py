"""S4 megakernel side: plain greedy decode vs PLD block verify on the cohere-megakernel (pinned 67d0b9ca + mk_selfspec.patch).
Run inside the megakernel venv:  python s4_mk.py <prompts.json> <out.jsonl> <pass_idx> [gammas=3,7]
One Multi per run (fresh prefill: sliding-layer page eviction makes a KV rerun unsafe). A Multi holds four decode states (BS=1,2,4,8) that
share ONE KV pool: a B=8 master prefill owns allocation (step_decode_positions), V = its row-0 page table is aliased into every row of the
state's own page table (adopt_blocks), T rows (1,2,3-4,5-8) pick BS 1,2,4,8 with padded rows masked (the S2/S3 contract)."""
import sys, types, json, time, dataclasses, pathlib, statistics, os

sys.path.insert(0, "/work/harness"); sys.path.insert(0, "/work"); sys.path.insert(0, "/work/modal")


class _Any:
    def __init__(s, *a, **k): pass
    def __getattr__(s, n): return _Any()
    def __call__(s, *a, **k):
        if len(a) == 1 and callable(a[0]) and not k: return a[0]
        return _Any()
class _Vol(_Any):
    def commit(s): pass
class _App(_Any):
    def function(s, *a, **k): return lambda f: f
    def local_entrypoint(s, *a, **k): return lambda f: f
class _Img(_Any):
    def __getattr__(s, n): return lambda *a, **k: s
_m = types.ModuleType("modal"); _m.App = _App
_m.Volume = types.SimpleNamespace(from_name=lambda *a, **k: _Vol())
_m.Image = types.SimpleNamespace(from_registry=lambda *a, **k: _Img())
sys.modules["modal"] = _m

import torch
_im = torch.inference_mode(); _im.__enter__()
import mk_block                      # modal/mk_block.py (stubbed modal) -> write_lib()
mk_block.write_lib()
sys.path.insert(0, "/mk/src")
import mk_block_lib as L
import native, runner
import decode.schedule as ds
import prefill.torch_prefill as tp
from mk_block_h import adopt_blocks
import pld

CKPT = mk_block.CKPT
BSEL = {1: 1, 2: 2, 3: 4, 4: 4, 5: 8, 6: 8, 7: 8, 8: 8}


class Multi:
    def __init__(self, world, ids, n_new):
        self.w = world; dev = world.dev
        P = len(ids); self.P = P
        steps = n_new + 16
        inp = torch.tensor([ids] * 8, device=dev, dtype=torch.long)
        self.pre = pre = tp._nmc_prefill_forward_from_ids(
            input_ids=inp, weights=world.w, cfg=world.cfg, device=dev, dtype=world.dt, page_block=L.PAGE_BLOCK,
            prefill_chunk_size=runner.DEFAULT_PREFILL_CHUNK_SIZE, frac_vram_utilization=0.5, max_seq=P + steps,
            temperature=0.0, top_p=1.0, trace=None)
        kvm = pre.kv_cache; self.kvm = kvm
        self.pb = int(kvm.page_block)
        self.V = kvm.page_table[:, 0, :].clone()
        self.first = int(pre.next_token.reshape(-1)[0])
        self.states, self.regs, self.kvs, self.hdl = {}, {}, {}, {}
        lib = native.ext(); jit_s = 0.0
        for B in (1, 2, 4, 8):
            if B == 8:
                kvB, preB = kvm, pre
            else:
                Lr, Pp = int(kvm.num_layers), int(kvm.max_pages)
                pt = self.V[:, None, :].expand(Lr, B, Pp).contiguous()
                cs = torch.full((B,), P, device=dev, dtype=torch.int32)
                ra = torch.ones((B,), device=dev, dtype=torch.int32)
                kvB = dataclasses.replace(kvm, page_table=pt, cache_seqlens=cs, row_active=ra, handle=None, batch_size=B)
                preB = types.SimpleNamespace(input_ids=pre.input_ids[:B], hidden=pre.hidden[:B], next_token=pre.next_token.reshape(-1)[:B],
                                             cos=pre.cos, sin=pre.sin, kv_cache=kvB, prompt_len=pre.prompt_len, max_seq=pre.max_seq)
            tiling = ds.default_nmc_tiling_for_bs(batch=B, moe_combine_atomic_tma=False)
            reg = ds.build_nmc_rr_decode_variant_registry(
                cfg=world.cfg, batch=B, num_sms=L.NUM_SMS, device=dev, tiling=tiling, num_layers=int(world.cfg.num_hidden_layers),
                launch_mode="all", page_block=self.pb, max_attn_splits=ds.DEFAULT_MAX_ATTN_SPLITS,
                min_attn_chunk=ds.DEFAULT_MIN_ATTN_CHUNK, min_context_len=P + 1, max_context_len=P + steps + 8,
                attn_drain=True, attn_drain_sms=None, moe_drain_sms=None, rr_dependency_affinity=False, ablation=False)
            st = ds.make_decode_state(library=lib, prefill=preB, weights=world.w, cfg=world.cfg, num_sms=L.NUM_SMS, tiling=tiling, decode_registry=reg)
            t = time.time(); reg.ensure_jit_handles(library=lib); jit_s += time.time() - t
            self.states[B], self.regs[B], self.kvs[B] = st, reg, kvB
        self.jit_s = jit_s
        self.cuda_ms = 0.0

    def step(self, tokens, positions, T):
        """-> (argmax ids[T], top2 margins[T]); rows 0..T-1 active in the BS=BSEL[T] state."""
        B = BSEL[T]; st, kvB, km = self.states[B], self.kvs[B], self.kvm
        pos8 = list(positions) + [positions[0]] * (8 - T); act8 = [1] * T + [0] * (8 - T)
        posB = pos8[:B]
        variant, _ = ds._activate_nmc_decode_variant(st, context_len=max(positions) + 1)
        handle = self.regs[B].jit_handle_for(variant)
        km.step_decode_positions(pos8, act8)
        ptm = km.page_table
        self.V = adopt_blocks(self.V, ptm[:, T - 1, :], min(positions) // self.pb, max(positions) // self.pb, xp=torch)
        if B == 8:
            ptm[:, :, :] = self.V[:, None, :]
        else:
            kvB.page_table[:, :, :] = self.V[:, None, :]
            meta = torch.tensor(posB + act8[:B], dtype=torch.int32, device=self.w.dev)
            kvB.cache_seqlens.copy_(meta[:B]); kvB.row_active.copy_(meta[B:])
        ds._refresh_attn_drain_queue(st, cache_seqlens=posB)
        ds._reset(st)
        tok = list(tokens) + [tokens[0]] * (B - T)
        ds._seed(st, self.w.w, self.w.cfg, torch.tensor(tok, device=self.w.dev, dtype=torch.long))
        _, cuda_ms = ds._launch_nmc_decode(handle=handle, state=st, device=self.w.dev)
        self.cuda_ms += cuda_ms
        lg = st.tensors["lm_logits"][:T].float()
        tk = torch.topk(lg, 2, dim=-1)
        ids = tk.indices[:, 0].tolist()
        mg = (tk.values[:, 0] - tk.values[:, 1]).tolist()
        return ids, mg

    def close(self):
        for B in self.regs:
            self.regs[B].close()
        self.pre.close()
        self.states.clear(); self.kvs.clear(); self.regs.clear()


def per_token_margins(rounds, mrows, n):
    out = []
    for (nd, a), m in zip(rounds, mrows):
        out += m[:a + 1]
    return out[:n]


def fd(a, b):
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), None if len(a) == len(b) else n)


def run_one(world, ids, n_new, mode, g, eos):
    m = Multi(world, ids, n_new)
    P = m.P; first = m.first
    gen, marg = [first], [None]
    steps_n = 0
    torch.cuda.synchronize(); t0 = time.perf_counter()
    if mode == "plain":
        cur = first
        for i in range(n_new - 1):
            if cur == eos: break
            idl, mg = m.step([cur], [P + i], 1)
            cur = idl[0]; gen.append(cur); marg.append(mg[0]); steps_n += 1
        rinfo = dict(rounds=steps_n)
    else:
        mrows = []

        def verify(tcur, drafts):
            T = 1 + len(drafts); pos0 = len(tcur) - 1
            idl, mg = m.step([tcur[-1]] + drafts, [pos0 + j for j in range(T)], T)
            mrows.append(mg)
            return idl
        out, rounds = pld.spec_loop(pld.make("sam2"), ids + [first], n_new - 1, g, verify, eos)
        gen += out; marg += per_token_margins(rounds, mrows, len(out))
        rinfo = pld.summarize_rounds(rounds, len(out)); rinfo["T_hist"] = {str(k): sum(1 for d, _ in rounds if d + 1 == k) for k in range(1, 9)}
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    nd = len(gen) - 1  # tokens produced by decode (first token came from prefill)
    res = dict(n=len(gen), wall_s=dt, tok_s=nd / dt, cuda_ms=m.cuda_ms, tok_s_cuda=nd / (m.cuda_ms / 1000), jit_s=m.jit_s, gen=gen, margins=marg, **rinfo)
    m.close(); del m
    return res


def main():
    prompts_p, out_p, pass_idx = sys.argv[1], sys.argv[2], int(sys.argv[3])
    gammas = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "3,7").split(",")]
    only = os.environ.get("S4_ONLY")  # optional "cell" filter for smoke tests
    n_new = 192
    rows = json.loads(pathlib.Path(prompts_p).read_text())
    world = L.World(CKPT)
    eos = world.cfg.eos_token_id; eos = eos if isinstance(eos, int) else eos[0]
    outf = open(out_p, "a")

    def emit(rec):
        rec["pass"] = pass_idx
        outf.write(json.dumps(rec) + "\n"); outf.flush()
        print(json.dumps({k: v for k, v in rec.items() if k not in ("gen", "margins")}), flush=True)

    emit(dict(kind="meta", exec="mk", load_s=world.load_s, torch=torch.__version__, n_prompts=len(rows)))
    # warm: one throwaway plain+PLD on the first prompt (JIT, allocator)
    for r in rows:
        if only and r["cell"] != only:
            continue
        ids = r["ids"]
        plain = run_one(world, ids, n_new, "plain", 0, eos)
        emit(dict(kind="run", exec="mk", mode="plain", cell=r["cell"], task=r["task"], prompt_len=len(ids), **plain))
        for g in gammas:
            res = run_one(world, ids, n_new, "pld", g, eos)
            d = fd(res["gen"], plain["gen"])
            ex = dict(identical_to_mk_plain=res["gen"] == plain["gen"], fd_vs_mk_plain=d)
            if d is not None and d < min(len(res["gen"]), len(plain["gen"])):
                ex.update(margin_plain_at_fd=plain["margins"][d], margin_pld_at_fd=res["margins"][d])
            emit(dict(kind="run", exec="mk", mode="pld", gamma=g, cell=r["cell"], task=r["task"], prompt_len=len(ids),
                      plain_tok_s_same_pass=plain["tok_s"], speedup_vs_mk_plain=res["tok_s"] / plain["tok_s"], **ex, **res))
    emit(dict(kind="done"))


if __name__ == "__main__":
    main()
