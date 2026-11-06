"""Modal entrypoint for track 2 (docs/SELF_EDIT.md): per-layer int4 experts chosen by North + recovery adapter.
  modal run --detach modal/se_run.py::main --smoke --tag se_smoke1
  modal run --detach modal/se_run.py::main --tag se_r1
Results survive a dead client: every artifact is written to Volume northstar-results:/selfedit_<tag>/ and printed as
`SE_JSON <name> <json>` lines in the app logs. Pull with `modal volume get northstar-results selfedit_<tag> results/`."""
import json, pathlib, sys, time

sys.path.insert(0, "modal")
import modal
from nsapp import app, image, results_vol, weights_vol, WEIGHTS, MODEL_DIR, GPU

se_image = (image.add_local_dir("src/approx", "/root/approx", copy=False)
            .add_local_dir("src/weightedit", "/root/weightedit", copy=False)
            .add_local_dir("src/selfedit", "/root/selfedit", copy=False))


def _build(tok, tasks, k, L, rotate):
    """Exact-length ids for the k-th task that fits in L tokens (skip too-long tasks); fall back to truncation."""
    from common import build_ids
    for j in range(len(tasks)):
        try:
            return build_ids(tok, tasks[(k + j) % len(tasks)], L, rotate=rotate)
        except ValueError:
            continue
    ids = build_ids(tok, tasks[k % len(tasks)])
    assert len(ids) >= L
    return ids[:L]


@app.function(image=se_image, gpu=GPU, volumes={WEIGHTS: weights_vol, "/results": results_vol},
              timeout=3 * 3600, memory=131072)
def run_remote(tag: str, smoke: bool, phase: str, K: int, group: int, rounds: int, n_random: int, n_eval: int,
               n_cal: int, n_tr: int, seq_len: int, tr_len: int, n_new: int, steps: int, lr: float, sens_n: int):
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/harness")
    import torch
    from loader import load_model
    from common import env_meta, by_split, load_workloads
    from selfedit.experiment import run_experiment_A
    from selfedit.metrics import teacher_cache
    from selfedit.quant import ExpertQuantEditor
    from selfedit.recipe import QRecipe
    from selfedit.recover import run_experiment_B
    from weightedit.recipe import moe_layers
    t0 = time.time()
    out_dir = pathlib.Path(f"/results/selfedit_{tag}"); out_dir.mkdir(parents=True, exist_ok=True)

    def ckpt(name, obj):
        (out_dir / f"{name}.json").write_text(json.dumps(obj, default=float))
        print(f"SE_JSON {name} {json.dumps(obj, default=float)}", flush=True)
        results_vol.commit()

    log = lambda s: print(s, flush=True)
    m, tok = load_model(MODEL_DIR, "eager")  # eager: backward-capable, one load for eval + training
    log(f"loaded {time.time()-t0:.0f}s; moe layers {len(moe_layers(m.config))} H={m.config.hidden_size} I={m.config.intermediate_size} V={m.config.vocab_size}")
    editor = ExpertQuantEditor(m)
    log(f"backup done {time.time()-t0:.0f}s")
    rows = load_workloads()
    dev_t, cal_t = by_split(rows, "dev"), by_split(rows, "calib")
    lo = 128
    dev_seqs = []
    for k in range(n_eval):
        ids = _build(tok, dev_t, k, seq_len, 100 + k)
        with torch.no_grad():
            g = m.generate(torch.tensor([ids], device=m.device), max_new_tokens=n_new, do_sample=False)
        dev_seqs.append(g[0].tolist())
    cal_seqs = [_build(tok, cal_t, k, seq_len, k) for k in range(n_cal)]
    tr_seqs = [_build(tok, cal_t, k, tr_len, 50 + k) for k in range(n_tr)]
    data = dict(dev_seqs=dev_seqs, dev_lo=lo, dev_cache=teacher_cache(m, dev_seqs, lo),
                cal_seqs=cal_seqs, cal_lo=lo, cal_cache=teacher_cache(m, cal_seqs, lo))
    train = dict(tr_seqs=tr_seqs, tr_lo=16, tr_cache=teacher_cache(m, tr_seqs, 16))
    log(f"teacher caches done {time.time()-t0:.0f}s")

    def gen(prompt):
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True,
                                      reasoning_effort="none")["input_ids"].to(m.device)
        with torch.no_grad():
            o = m.generate(ids, max_new_tokens=1024, do_sample=False)
        return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)

    res = dict(meta=env_meta(), tag=tag, smoke=smoke)
    cfgA = dict(K=K, group=group, rounds=rounds, n_random=n_random, frontier_Ks=[] if smoke else [8, 16, 32, 48],
                sens_layers=moe_layers(m.config)[:sens_n] if sens_n else None, uniform_all=False)
    A = run_experiment_A(m, editor, data, cfgA, gen_fn=gen, log=log, ckpt=ckpt)
    res["A"] = dict(summary=A["summary"], table=A["table"], arms=[{k: v for k, v in a.items() if k != "kl_by_seq"} for a in A["arms"]])
    if phase in ("all", "B"):
        pool = [a for a in A["arms"] if a["K"] == K]
        best = min(pool, key=lambda a: a["kl"])
        log(f"B recipe: {best['name']} kl={best['kl']:.5f}")
        res["B"] = run_experiment_B(m, editor, QRecipe(tuple(best["layers"]), group).validate(m.config), data, train,
                                    dict(steps=steps, lr=lr, batch=2, rank=4, ckpt=True), log=log, ckpt=ckpt)
        res["B"]["base_recipe_name"] = best["name"]
    res["gpu_seconds"] = time.time() - t0
    ckpt("result", res)
    return res


@app.local_entrypoint()
def main(tag: str = "", smoke: bool = False, phase: str = "all", k: int = 24, group: int = 32, rounds: int = 3,
         n_random: int = 5, n_eval: int = 6, n_cal: int = 4, n_tr: int = 16, seq_len: int = 1024, tr_len: int = 256,
         n_new: int = 48, steps: int = 80, lr: float = 3e-3, sens_n: int = 0):
    tag = tag or time.strftime("%Y%m%d-%H%M")
    if smoke:
        k, rounds, n_random, n_eval, n_cal, n_tr, seq_len, tr_len, n_new, steps, sens_n = 2, 1, 2, 2, 2, 4, 512, 128, 16, 3, 3
    res = run_remote.remote(tag, smoke, phase, k, group, rounds, n_random, n_eval, n_cal, n_tr, seq_len, tr_len, n_new,
                            steps, lr, sens_n)
    out = pathlib.Path("results"); out.mkdir(exist_ok=True)
    (out / f"selfedit_{tag}.json").write_text(json.dumps(res, default=float, indent=1))
