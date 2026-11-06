"""W1 (OpenHands coding-agent panel) sampled generation on North, for src/suffixspec (docs/SUFFIX_W1.md).
Prompts: results/w1_prompts.npz (token ids, chat template reasoning_effort=none), uploaded to Volume northstar-results:/w1_prompts.npz.
Sampling: T=0.7, top_p=0.95, seed 42+index, max_new 4096, stop at EOS.  Chunked prefill (512).  One request at a time.
Streams every finished request as 'W1_REC {json}' to app logs and appends to Volume northstar-results:/w1_gen_<tag>.jsonl.
  modal run --detach modal/w1_gen.py::main --tag g1 --n-sessions 12 --max-seconds 3000"""
import json, sys, time

sys.path.insert(0, "modal")
import modal
from nsapp import app, image, results_vol, weights_vol, WEIGHTS, MODEL_DIR, GPU


@app.function(image=image, gpu=[GPU, "H100"], volumes={WEIGHTS: weights_vol, "/results": results_vol},
              timeout=3 * 3600, memory=65536)
def gen_remote(tag: str, n_sessions: int, max_seconds: int, max_new: int, chunk: int, smoke: bool):
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/harness")
    import numpy as np, torch
    from transformers import DynamicCache
    from loader import load_model
    from common import env_meta
    t0 = time.time()
    m, tok = load_model(MODEL_DIR, "grouped_mm")
    dev = m.device
    man = json.load(open("/results/w1_prompts_manifest.json"))
    npz = np.load("/results/w1_prompts.npz")
    sessions = []
    for r in man:
        if r["session"] not in sessions:
            sessions.append(r["session"])
    keep = set(sessions[:n_sessions])
    reqs = [r for r in man if r["session"] in keep]
    if smoke:
        reqs = reqs[:1]; max_new = 64
    eos = m.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    out = open(f"/results/w1_gen_{tag}.jsonl", "a")
    print("META_JSON " + json.dumps(env_meta()), flush=True)
    done = 0
    for i, r in enumerate(reqs):
        if time.time() - t0 > max_seconds:
            print(f"W1_STOP time budget after {done} requests", flush=True); break
        ids = torch.tensor(npz[r["id"]].astype("int64"), device=dev)[None]
        P = ids.shape[1]
        g = torch.Generator(device=dev).manual_seed(42 + i)
        t1 = time.time()
        cache = DynamicCache()
        with torch.no_grad():
            for s in range(0, P - 1, chunk):
                e = min(s + chunk, P - 1)
                m(input_ids=ids[:, s:e], past_key_values=cache, use_cache=True, logits_to_keep=1)
            torch.cuda.synchronize(); t_pre = time.time() - t1
            cur, outs = ids[:, P - 1:], []
            for k in range(max_new):
                lg = m(input_ids=cur, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[0, -1].float()
                p = torch.softmax(lg / 0.7, -1)
                sp, si = p.sort(descending=True)
                cs = sp.cumsum(0)
                sp[(cs - sp) > 0.95] = 0
                nxt = si[torch.multinomial(sp / sp.sum(), 1, generator=g)]
                t = int(nxt)
                outs.append(t)
                if t in eos:
                    break
                cur = nxt[None]
        rec = dict(id=r["id"], session=r["session"], turn=r["turn"], P=P, n_out=len(outs), capped=len(outs) >= max_new and outs[-1] not in eos,
                   prefill_s=t_pre, total_s=time.time() - t1, out_ids=outs)
        out.write(json.dumps(rec) + "\n"); out.flush(); results_vol.commit()
        print("W1_REC " + json.dumps(rec), flush=True)
        done += 1
        del cache
        torch.cuda.empty_cache()
    secs = time.time() - t0
    print(f"W1_DONE requests={done} gpu_seconds={secs:.0f}", flush=True)
    return dict(requests=done, gpu_seconds=secs)


@app.local_entrypoint()
def main(tag: str = "g1", n_sessions: int = 12, max_seconds: int = 3000, max_new: int = 4096, chunk: int = 512,
         smoke: bool = False):
    print(gen_remote.remote(tag, n_sessions, max_seconds, max_new, chunk, smoke))
