"""Dump North's greedy outputs (token ids) for the harness workloads, for offline drafter evaluation
(src/suffixspec). Detached-safe: writes /results/north_greedy_dump_<tag>.jsonl to Volume northstar-results AND prints
each record as 'REC_JSON ...' to the app logs.
  modal run --detach modal/gen_dump.py::main --tag d1"""
import json, pathlib, sys, time

sys.path.insert(0, "modal")
import modal
from nsapp import app, image, results_vol, weights_vol, WEIGHTS, MODEL_DIR, GPU


@app.function(image=image, gpu=GPU, volumes={WEIGHTS: weights_vol, "/results": results_vol}, timeout=3600, memory=65536)
def dump_remote(tag: str, n_new: int, batch: int):
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/harness")
    import torch
    from loader import load_model
    from common import build_ids, load_workloads, env_meta
    t0 = time.time()
    m, tok = load_model(MODEL_DIR, "grouped_mm")
    rows = load_workloads()
    recs = [dict(id=r["id"], kind=r["kind"], split=r["split"], prompt_ids=build_ids(tok, r)) for r in rows]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos = set(x for x in [tok.eos_token_id, getattr(m.generation_config, "eos_token_id", None)] if isinstance(x, int))
    if isinstance(getattr(m.generation_config, "eos_token_id", None), list):
        eos |= set(m.generation_config.eos_token_id)
    out_path = pathlib.Path(f"/results/north_greedy_dump_{tag}.jsonl")
    with out_path.open("w") as f:
        f.write("# " + json.dumps(env_meta()) + "\n")
        for b in range(0, len(recs), batch):
            grp = recs[b:b + batch]
            L = max(len(r["prompt_ids"]) for r in grp)
            ids = torch.tensor([[pad] * (L - len(r["prompt_ids"])) + r["prompt_ids"] for r in grp], device=m.device)
            att = torch.tensor([[0] * (L - len(r["prompt_ids"])) + [1] * len(r["prompt_ids"]) for r in grp], device=m.device)
            with torch.no_grad():
                o = m.generate(ids, attention_mask=att, max_new_tokens=n_new, do_sample=False, pad_token_id=pad)
            for r, row in zip(grp, o[:, L:].tolist()):
                cut = next((i for i, t in enumerate(row) if t in eos), len(row))
                r["out_ids"] = row[:cut]
                f.write(json.dumps(r) + "\n"); f.flush()
                print("REC_JSON " + json.dumps(r), flush=True)
            results_vol.commit()
    secs = time.time() - t0
    print(f"DONE records={len(recs)} gpu_seconds={secs:.0f}", flush=True)
    results_vol.commit()
    return dict(records=len(recs), gpu_seconds=secs)


@app.local_entrypoint()
def main(tag: str = "d1", n_new: int = 512, batch: int = 11):
    res = dump_remote.remote(tag, n_new, batch)
    print(res)
