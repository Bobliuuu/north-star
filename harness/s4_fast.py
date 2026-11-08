"""S4 fast-executor side: plain fast decode vs PLD (sam2) on the same prompt ids as s4_mk.py. Runs in the fast env (torch 2.10, transformers 5.22.1).
    python s4_fast.py <prompts.json> <out.jsonl> <pass_idx> [gammas=3,7,12]
Same loops as bench_fastpld.bench (CUDA-graph T=1 step; multi-query verify blocks, pld.spec_loop). Non-exact FastNorth (H9/H13 default)."""
import sys, json, time, pathlib, os
sys.path.insert(0, "/work/harness"); sys.path.insert(0, "/work")
import torch
from loader import load_model
from northstar.refmodel import RefNorth
from fast_north import FastNorth
import pld

MODEL_DIR = "/weights/north-mini-code-d11e61a8"
TMAX = 13


def fd(a, b):
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), None if len(a) == len(b) else n)


@torch.no_grad()
def main():
    prompts_p, out_p, pass_idx = sys.argv[1], sys.argv[2], int(sys.argv[3])
    gammas = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "3,7,12").split(",")]
    n_new = 192
    rows = json.loads(pathlib.Path(prompts_p).read_text())
    t0 = time.time()
    m, tok = load_model(MODEL_DIR, "grouped_mm")
    ref = RefNorth(m); dev = ref.device
    eos = m.config.eos_token_id; eos = eos if isinstance(eos, int) else eos[0]
    outf = open(out_p, "a")

    def emit(rec):
        rec["pass"] = pass_idx
        outf.write(json.dumps(rec) + "\n"); outf.flush()
        print(json.dumps({k: v for k, v in rec.items() if k != "gen"}), flush=True)

    emit(dict(kind="meta", exec="fast", load_s=time.time() - t0, torch=torch.__version__, n_prompts=len(rows)))
    cells = []
    for r in rows:
        if r["cell"] not in cells:
            cells.append(r["cell"])
    for cname in cells:
        ts = [r for r in rows if r["cell"] == cname]
        Lmax = max(len(r["ids"]) for r in ts) + n_new + 40
        fast = FastNorth(ref, Lmax); kv = fast.new_kv()
        steps = {1: fast.make_step(1, kv)}
        for T in range(2, TMAX + 1):
            steps[T] = fast.make_step(T, kv, logits="all")
        for r in ts:
            ids = r["ids"]; P = len(ids)
            fast.eager_prefill(ids[:-1], kv)
            s1 = steps[1]
            tk, cur = [], ids[-1]
            torch.cuda.synchronize(); t1 = time.perf_counter()
            for i in range(n_new):
                s1.set([cur], P - 1 + i); cur = int(s1.run()[0, 0]); tk.append(cur)
                if cur == eos: break
            dt = time.perf_counter() - t1
            plain = dict(n=len(tk), tok_s=len(tk) / dt, gen=tk)
            emit(dict(kind="run", exec="fast", mode="plain", cell=cname, task=r["task"], prompt_len=P, **plain))
            for g in gammas:
                def verify(tcur, drafts):
                    T = 1 + len(drafts)
                    stp = steps[T]; stp.set([tcur[-1]] + drafts, len(tcur) - 1)
                    return stp.run()[0].tolist()
                torch.cuda.synchronize(); t1 = time.perf_counter()
                out, rounds = pld.spec_loop(pld.make("sam2"), ids, len(tk), g, verify, eos)
                torch.cuda.synchronize(); dt = time.perf_counter() - t1
                d = fd(out, tk)
                emit(dict(kind="run", exec="fast", mode="pld", gamma=g, cell=cname, task=r["task"], prompt_len=P, n=len(out), tok_s=len(out) / dt,
                          plain_tok_s_same_pass=plain["tok_s"], speedup_vs_fast_plain=(len(out) / dt) / plain["tok_s"],
                          identical_to_fast_plain=out == tk, fd_vs_fast_plain=d, gen=out, **pld.summarize_rounds(rounds, len(out))))
        del steps, fast, kv; torch.cuda.empty_cache()
    emit(dict(kind="done"))


if __name__ == "__main__":
    main()
