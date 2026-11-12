"""EMBEDDED North-in-North (qwenagent-style): trainable stateful workers inside ONE frozen North forward pass.

  modal run --detach modal/embedded_north_run.py --batch smoke|main

Workers (src/embedded/worker.py, copied from qwenagent) tap the residual stream after layer 12 of frozen North (49 layers, hidden 2048),
keep private recurrent slot states with a learned router, and add a return signal back into the residual stream before layers 13-48.
No prompts, no extra model calls; only the workers (~hundreds of k params) are trained. Task: q12 'ledger' (entities with balances updated
in natural-language sentences, answer = digit 0-9), trained on P=2,3 entities and tested at P=2..8. Modes: none (frozen North, no training),
adapter (stateless MLP), single (one shared state), multi (private slots + learned router = the embedded subagents).
Results/logs/checkpoints land in Volume northstar-results:/embedded_<batch>/ and are printed as EMB_JSON.
"""
import json, os, subprocess, sys, time
import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS, MODEL_DIR = "/weights", "/weights/north-mini-code-d11e61a8"
weights_vol = modal.Volume.from_name("northstar-weights")
results_vol = modal.Volume.from_name("northstar-results", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.10.0", "transformers==5.22.1", "accelerate==1.15.0", "numpy==2.2.5", "safetensors")
         .add_local_dir(os.path.join(ROOT, "src/embedded"), "/app"))
app = modal.App("northstar-embedded", image=image)


@app.function(gpu="A100-80GB", volumes={WEIGHTS: weights_vol, "/res": results_vol}, timeout=5 * 3600, memory=65536, max_containers=8)
def job(batch: str, mode: str, seed: int, steps: int, bs: int, n_eval: int, layer: int = 12):
    out_dir = f"/res/embedded_{batch}"; os.makedirs(out_dir, exist_ok=True)
    tag = f"{mode}_s{seed}"; out = f"{out_dir}/{tag}.json"
    cmd = ["python", "/app/train.py", "--north", "1", "--model", MODEL_DIR, "--mode", mode, "--seed", str(seed), "--layer", str(layer),
           "--K", "4", "--occ", "1", "--ret_norm", "1", "--task", "ledger", "--train_P", "2,3", "--test_P", "2,3,4,6,8",
           "--opr", "2,4", "--steps", str(steps), "--bs", str(bs), "--n_eval", str(n_eval), "--out", out, "--save", f"{out_dir}/{tag}.pt"]
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(f"{out_dir}/{tag}.log", "w") as lf:
        for line in p.stdout:
            lf.write(line); lf.flush(); print(f"[{tag}] {line.rstrip()[:300]}", flush=True)
    p.wait(); results_vol.commit()
    ok = os.path.exists(out)
    res = json.load(open(out)) if ok else None
    row = dict(tag=tag, mode=mode, seed=seed, rc=p.returncode, secs=time.time() - t0, final=res["final"] if res else None, params=res["params"] if res else None,
               log=res["log"] if res else None)
    print("EMB_JSON " + json.dumps(row), flush=True)
    return row


BATCHES = {
    "smoke": [("smoke", "multi", 0, 30, 8, 8)],
    "main": [("main", m, s, 1500, 16, 100) for m in ("adapter", "single", "multi") for s in (0, 1)] + [("main", "none", 0, 0, 16, 100)],
}


@app.local_entrypoint()
def main(batch: str = "smoke"):
    rows = list(job.starmap(BATCHES[batch]))
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = os.path.join(ROOT, "results", f"embedded_{batch}_{ts}.json")
    json.dump(rows, open(path, "w"), indent=1)
    secs = sum(r["secs"] for r in rows)
    with open(os.path.join(ROOT, "results", "SPEND.md"), "a") as f:
        f.write(f"| {ts} | embedded North-in-North {batch} ({len(rows)} runs) | A100-80GB | {secs:.0f} | {secs / 3600 * 2.5:.3f} | True |\n")
    print("->", path)
