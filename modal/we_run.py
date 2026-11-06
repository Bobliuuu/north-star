"""Modal entrypoint for the weight-edit loop (docs/WEIGHT_EDIT.md). NOT launched yet: needs GPU-budget authorization.

  modal run --detach modal/we_run.py::main --smoke                 # 1 recipe, 5 steps, ctx 1024
  modal run --detach modal/we_run.py::main --proposer north --rounds 6  # North proposes its own recipes

Results survive a dead client: the remote function writes /results/weightedit_<tag>/ (store + result.json) to the
Volume `northstar-results` AND prints every round as JSON to the app logs.  Pull with
`modal volume get northstar-results weightedit_<tag> results/` and `modal app logs <id>`; log the launch in RUNS.md."""
import json, pathlib, sys, time

sys.path.insert(0, "modal")
import modal
from nsapp import app, image, results_vol, weights_vol, WEIGHTS, MODEL_DIR, GPU

we_image = (image.add_local_dir("src/approx", "/root/approx", copy=False)
            .add_local_dir("src/weightedit", "/root/weightedit", copy=False))


def _seqs(m, tok, tasks, n, ctx, n_new, rotate0):
    import torch
    from common import build_ids
    seqs, lo = [], []
    for k in range(n):
        ids = build_ids(tok, tasks[k % len(tasks)], ctx, rotate=rotate0 + k)
        with torch.no_grad():
            out = m.generate(torch.tensor([ids], device=m.device), max_new_tokens=n_new, do_sample=False)
        seqs.append(out[0].tolist()); lo.append(len(ids) - 1)
    return seqs, lo


@app.function(image=we_image, gpu=GPU, volumes={WEIGHTS: weights_vol, "/results": results_vol},
              timeout=4 * 3600, memory=65536)
def run_remote(proposer: str, rounds: int, tag: str, smoke: bool, seq_len: int, n_new: int, n_train: int, n_gate: int,
               experts_impl: str, steps: int, window: int):
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/harness")
    import torch
    from loader import load_model
    from common import env_meta, by_split, load_workloads
    from weightedit.loop import GridProposer, LLMProposer, run_round
    from weightedit.store import EditStore
    t0 = time.time()
    m, tok = load_model(MODEL_DIR, experts_impl)
    rows = load_workloads()
    tr, tr_lo = _seqs(m, tok, by_split(rows, "calib"), n_train, seq_len, n_new, 0)
    ga, ga_lo = _seqs(m, tok, by_split(rows, "dev"), n_gate, seq_len, n_new, 100)
    store = EditStore(f"/results/weightedit_{tag}")
    grid = GridProposer(rs=(2, 3), ranks=(4, 8), lrs=(1e-3, 3e-3), steps=steps, window=window, batch=1)

    def gen(prompt):  # North in VERIFIER mode (no patch installed during proposal)
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)["input_ids"].to(m.device)
        with torch.no_grad():
            out = m.generate(ids, max_new_tokens=300, do_sample=False)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    prop = LLMProposer(gen, m.config, grid) if proposer == "north" else grid
    history, cache = [], {}
    for r in range(1 if smoke else rounds):
        e = run_round(m, prop, store, tr, tr_lo, ga, ga_lo, history, cache=cache, fp_mode="sampled",
                      ckpt=True, log=lambda s: print(s, flush=True))
        print("ROUND_JSON " + json.dumps(e, default=float), flush=True)
        results_vol.commit()
    res = dict(meta=env_meta(), gpu_seconds=time.time() - t0, proposer=proposer, history=history,
               ledger=store.ledger(), current=store.current())
    pathlib.Path(f"/results/weightedit_{tag}/result.json").write_text(json.dumps(res, default=float))
    results_vol.commit()
    print("RESULT_JSON " + json.dumps(res, default=float), flush=True)
    return res


@app.local_entrypoint()
def main(proposer: str = "grid", rounds: int = 6, tag: str = "", smoke: bool = False, seq_len: int = 2048,
         n_new: int = 96, n_train: int = 8, n_gate: int = 8, experts_impl: str = "eager", steps: int = 100,
         window: int = 256):
    tag = tag or time.strftime("%Y%m%d-%H%M")
    if smoke:
        seq_len, n_train, n_gate, steps, window, n_new = 1024, 2, 3, 5, 128, 32
    res = run_remote.remote(proposer, rounds, tag, smoke, seq_len, n_new, n_train, n_gate, experts_impl, steps, window)
    out = pathlib.Path("results"); out.mkdir(exist_ok=True)
    (out / f"weightedit_{tag}.json").write_text(json.dumps(res, default=float, indent=1))
