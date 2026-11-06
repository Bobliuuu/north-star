"""H100 phases on real North (HF weights from the northstar-weights volume). ONE H100 container at a time, short timeouts.
  modal run modal/h100.py::tiny       A100-40GB tiny-model CUDA-graph correctness test (cents)
  modal run modal/h100.py::blocks     H100: graphed block-verify K=1..16 + components + draft-step costs + routing traces for overlap
  modal run modal/h100.py::spec       H100: graphed self-spec (draft+verify) wall speedup
"""
import sys, time, json, pathlib, datetime
sys.path.insert(0, "modal")
import modal
from nsapp import app, weights_vol, WEIGHTS, MODEL_DIR

RES = pathlib.Path("results")
RATE = {"H100": 3.95, "A100-40GB": 2.10}
KW = dict(volumes={WEIGHTS: weights_vol}, memory=65536, scaledown_window=2, max_containers=1)


def _setup():
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/harness")


def _phase(out, name, fn):
    t = time.time()
    try:
        out[name] = fn()
        out[name + "_secs"] = time.time() - t
    except Exception as e:
        import traceback
        out[name + "_error"] = traceback.format_exc()[-3000:]
        print("PHASE FAILED", name, e, flush=True)


@app.function(gpu="H100", timeout=600, scaledown_window=2, max_containers=1)
def tiny_remote():
    _setup(); sys.path.insert(0, "/root/tests")
    t0 = time.time(); log = []
    import torch
    from transformers.models.cohere2_moe.modeling_cohere2_moe import Cohere2MoeForCausalLM
    from fixtures import tiny_cfg
    from northstar.refmodel import RefNorth, FULL, DraftMode
    from graph_north import GraphNorth
    torch.manual_seed(0)
    cfg = tiny_cfg(hidden_size=128, intermediate_size=64, num_experts=16, head_dim=32, sliding_window=16, num_hidden_layers=9, prefix_dense_intermediate_size=128)
    cfg._experts_implementation = "grouped_mm"
    m = Cohere2MoeForCausalLM(cfg).cuda().to(torch.bfloat16).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.ndim >= 2: p.normal_(0, 0.1 if "embed" not in n else 1.0)
    log.append("experts impl: %s" % getattr(m.config, "_experts_implementation", None))
    ref = RefNorth(m)
    ids = torch.randint(0, cfg.vocab_size, (1, 40), device="cuda")
    cache = ref.new_cache(1, 80)
    ref.prefill(ids[:, :39], cache)
    lg_ref = ref.forward(ids[:, 39:], cache, 39, FULL, logits="last").float()
    gn = GraphNorth(ref, 80)
    kv = gn.new_kv()
    gn.eager_prefill(ids[0, :39].tolist(), kv)
    # bisect capture failures
    ex_ = gn.model.layers[1].mlp.experts
    S = 4
    eid = torch.randint(0, 16, (S,), device="cuda"); w4 = torch.rand(S, device="cuda")
    xs = torch.randn(S, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    offs = torch.tensor([0,0,1,1,2,2,3,3,4,4,4,4,4,4,4,4], device="cuda", dtype=torch.int32)
    gg = torch.randn(S, 2 * cfg.intermediate_size, device="cuda", dtype=torch.bfloat16)
    def tryc(name, fn):
        try:
            gn.capture(fn, warmup=2); log.append("capture OK: " + name)
        except Exception as e:
            log.append("capture FAIL: %s: %s" % (name, str(e)[:150]))
            torch.cuda.synchronize() if False else None
    x1 = torch.zeros(1, 1, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    ids1 = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    tryc("embed+rotary", lambda: gn.model.rotary_emb(gn.model.embed_tokens(ids1), (gn.start + torch.arange(1, device="cuda"))[None]))
    lay = gn.model.layers
    tryc("dense mlp", lambda: lay[0].mlp(x1))
    tryc("router", lambda: lay[1].mlp.gate(x1.reshape(-1, cfg.hidden_size)))
    def ex():
        _, sc, ix = lay[1].mlp.gate(x1.reshape(-1, cfg.hidden_size)); return lay[1].mlp.experts(x1.reshape(-1, cfg.hidden_size), ix, sc)
    tryc("router+experts", ex)
    tryc("sort", lambda: torch.sort(eid))
    tryc("histc", lambda: torch.histc(eid.int(), bins=16, min=0, max=15))
    tryc("cumsum", lambda: torch.cumsum(torch.histc(eid.int(), bins=16, min=0, max=15), 0, dtype=torch.int32))
    tryc("grouped_mm", lambda: torch.nn.functional.grouped_mm(xs, ex_.gate_up_proj.transpose(-2, -1), offs=offs))
    def ip():
        perm = torch.sort(eid)[1]; inv = torch.empty_like(perm); inv[perm] = torch.arange(S, device="cuda"); return inv
    tryc("inv_perm index_put", ip)
    tryc("gather idx", lambda: xs[torch.sort(eid)[1] // 4])
    tryc("act_fn gate", lambda: ex_._apply_gate(gg))
    log.append("has _is_expert_parallel: %s ; impl fn: %s" % (getattr(ex_, "_is_expert_parallel", None), ex_.forward.__name__))
    tryc("forward parts=attn", lambda: gn.forward(ids1, kv, parts=frozenset({"attn"}), logits="none"))
    tryc("forward parts=router+experts", lambda: gn.forward(ids1, kv, parts=frozenset({"router", "experts"}), logits="none"))
    tryc("forward all", lambda: gn.forward(ids1, kv))
    try:
        st = gn.make_step(1, kv, argmax=False)
    except Exception as e:
        return dict(log="\n".join(log) + "\nmake_step FAILED " + str(e)[:300], secs=time.time() - t0)
    st.set([int(ids[0, 39])], 39)
    lg_g = st.run().float()
    d = (lg_ref - lg_g).abs().max().item()
    log.append(f"graph vs ref logits max abs diff {d:.4f} (ref max {lg_ref.abs().max().item():.3f}) argmax equal {bool(lg_ref.argmax()==lg_g.argmax())}")
    # replay at another position, check against ref
    ref.forward(ids[:, 39:], cache, 39, FULL, logits="none"); cache.len = 40
    nxt = int(lg_ref.argmax())
    lg_ref2 = ref.forward(torch.tensor([[nxt]], device="cuda"), cache, 40, FULL, logits="last").float()
    st.set([nxt], 40); lg_g2 = st.run().float()
    log.append(f"replay pos40 diff {(lg_ref2-lg_g2).abs().max().item():.4f} argmax equal {bool(lg_ref2.argmax()==lg_g2.argmax())}")
    # block graph
    blk = torch.randint(0, cfg.vocab_size, (1, 5), device="cuda")
    s5 = gn.make_step(5, kv, logits="all", argmax=False); s5.set(blk[0].tolist(), 41)
    lgb = s5.run().float()
    lgr = ref.forward(blk, cache, 41, FULL, logits="all").float()
    log.append(f"block K=5 diff {(lgb-lgr).abs().max().item():.4f}")
    # draft mode
    s_d = gn.make_step(1, gn.new_kv([i for i in range(9) if i not in (3, 5)]), skip=frozenset({3, 5}), r=2)
    s_d.set([1], 3); log.append("draft graph ok " + str(s_d.run().tolist()))
    # components + spec smoke
    import bench_graph
    class Tok: pass
    return dict(log="\n".join(log), secs=time.time() - t0)


@app.function(gpu="H100", timeout=2700, **KW)
def blocks_remote(do_blocks: bool = True, do_overlap: bool = True, ctxs=(2048, 8192)):
    _setup()
    t0 = time.time(); out = {}
    from loader import load_model
    from common import env_meta
    import bench_graph, bench_overlap
    m, tok = load_model(MODEL_DIR, "grouped_mm")
    out["load_secs"] = time.time() - t0
    out["meta"] = env_meta()
    if do_blocks:
        _phase(out, "blocks", lambda: bench_graph.run_blocks(m, tok, ctxs=tuple(ctxs)).text())
    if do_overlap:
        def ov():
            j, dump = bench_overlap.run(m, tok)
            return json.dumps(dump)
        _phase(out, "overlap", ov)
    out["secs"] = time.time() - t0
    return out


@app.function(gpu="H100", timeout=2400, **KW)
def spec_remote(long_ctx: int = 2048, ntasks: int = 4):
    _setup()
    t0 = time.time(); out = {}
    from loader import load_model
    from common import env_meta
    import bench_graph
    m, tok = load_model(MODEL_DIR, "grouped_mm")
    out["load_secs"] = time.time() - t0
    out["meta"] = env_meta()
    _phase(out, "spec", lambda: bench_graph.run_spec(m, tok, ntasks=ntasks, long_ctx=long_ctx).text())
    out["secs"] = time.time() - t0
    return out


def _spend(task, gpu, secs, ok=True):
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    cost = secs / 3600 * RATE[gpu]
    with open(RES / "SPEND.md", "a") as f:
        f.write(f"| {ts} | {task} | {gpu} | {secs:.0f} | {cost:.2f} | {ok} |\n")
    print(f"SPEND {task} {gpu} {secs:.0f}s est ${cost:.2f}")


@app.local_entrypoint()
def tiny():
    r = tiny_remote.remote()
    print(r["log"]); _spend("tiny graph test (debug, tiny model)", "H100", r["secs"])


def _save(r, tag):
    RES.mkdir(exist_ok=True)
    (RES / f"{tag}.meta.json").write_text(json.dumps(r["meta"], indent=1))
    for k in ("blocks", "spec"):
        if k in r: (RES / f"h100_{k}.jsonl").write_text(r[k])
    if "overlap" in r: (RES / "overlap_traces.json").write_text(r["overlap"])
    errs = {k: v for k, v in r.items() if k.endswith("_error")}
    if errs: (RES / f"{tag}.errors.txt").write_text(json.dumps(errs, indent=1))
    print({k: v for k, v in r.items() if k.endswith("_secs") or k.endswith("_error") or k == "secs"})
    _spend(tag, "H100", r["secs"], ok=not errs)


@app.local_entrypoint()
def blocks(do_blocks: bool = True, do_overlap: bool = True, ctxs: str = "2048,8192"):
    r = blocks_remote.remote(do_blocks, do_overlap, [int(c) for c in ctxs.split(",")])
    _save(r, "h100_blocks_overlap")


@app.local_entrypoint()
def spec(long_ctx: int = 2048, ntasks: int = 4):
    _save(spec_remote.remote(long_ctx, ntasks), "h100_spec")
