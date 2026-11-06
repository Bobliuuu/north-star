"""Factorial draft-approximation sweep: full / attn-only / expert-only / combined (+ layerskip group).

Protocol (teacher-forced, real tokens, no wall-clock claims):
  1. prompt = harness workload (exact-length repository-context filler when ctx given), calib split for EAT
     history / skip sensitivity, DEV split for evaluation (heldout untouched).
  2. teacher = the VERIFIER's own greedy continuation (n_new tokens, HF generate, unpatched).
  3. verifier targets: argmax of one unpatched full-sequence forward over prompt+teacher at the n_new positions
     that predict the teacher tokens ('vpred'; match with the teacher is reported as a determinism sanity check).
  4. each config: one DRAFT forward of the same sequence under DraftPatch.draft() (draft KV is produced by the draft
     computation itself, exactly as in self-speculation); per-position draft argmax vs vpred.
  5. accepted length for block size gamma starting at position s = length of the agreement run from s, capped at
     gamma.  This is exact for greedy verification because while drafts agree the draft's input equals the teacher.
Metrics: top1 agreement, verifier-top1-in-draft-top5, accepted-length histogram per gamma, mean committed tokens
per cycle (accepted+1), and implementation-independent cost proxies (experts/token, attended-key fraction, index
scan/build/bytes).  Execution cost is NOT measured here (another track owns GPU timing)."""
from __future__ import annotations

import itertools
import json
import time

import numpy as np
import torch

from .config import DraftConfig, EATParams, LSAParams
from .hooks import DraftPatch, EATCalibrator
from .layerskip import candidate_skip_sets

# --------------------------------------------------------------------------------------------- pure metrics

def accepted_lengths(agree, gamma):
    """agree: 1-D bool array over consecutive positions. Returns int array a[s] = min(gamma, run of True from s)."""
    agree = np.asarray(agree, dtype=bool)
    n = len(agree)
    run = np.zeros(n + 1, dtype=np.int64)
    for s in range(n - 1, -1, -1):
        run[s] = run[s + 1] + 1 if agree[s] else 0
    return np.minimum(run[:n], gamma)


def accepted_hist(agree, gamma):
    a = accepted_lengths(agree, gamma)
    h = np.bincount(a, minlength=gamma + 1)
    return dict(hist=h.tolist(), mean_accepted=float(a.mean()), mean_committed=float(a.mean() + 1))


# ------------------------------------------------------------------------------------------------- grids

def build_grid(preset="quick", skip_sets=True):
    """Returns {group: [(name, DraftConfig)]}. 'combined' = product of a SUBSET of attn-only x expert-only configs
    (identical component settings, so interaction = combined vs best single is read off directly)."""
    if preset == "quick":
        budgets, sels, rs, convs, eats, c_b, c_r = [256, 1024], ["oracle_topk", "lsa_tree"], [2, 4, 6], \
            ["raw_sigmoid", "renorm"], [1.0], [256, 1024], [4]
    else:
        budgets, sels, rs, convs, eats, c_b, c_r = [128, 256, 512, 1024, 2048], \
            ["oracle_topk", "lsa_flat", "lsa_tree", "subq_style"], [1, 2, 3, 4, 5, 6, 7], \
            ["raw_sigmoid", "renorm", "mass_preserve"], [0.5, 1.0, 1.5], [256, 1024], [2, 4, 6]
    g = {"full": [("full", DraftConfig())], "attn_only": [], "expert_only": [], "combined": [], "layerskip": []}
    for s, b in itertools.product(sels, budgets):
        g["attn_only"].append((f"attn:{s}:k{b}", DraftConfig(attn_budget=b, attn_selector=s)))
    for r, c in itertools.product(rs, convs):
        g["expert_only"].append((f"topr:r{r}:{c}", DraftConfig(expert_r=r, weight_convention=c)))
    for t in eats:
        g["expert_only"].append((f"eat:tau{t}:renorm", DraftConfig(expert_policy="eat", weight_convention="renorm",
                                                                    eat=EATParams(tau_scale=t))))
    for s, b, r, c in itertools.product(sels, c_b, c_r, ["raw_sigmoid", "renorm"]):
        g["combined"].append((f"attn:{s}:k{b}+topr:r{r}:{c}",
                              DraftConfig(attn_budget=b, attn_selector=s, expert_r=r, weight_convention=c)))
    if skip_sets:
        for n, s in candidate_skip_sets(49).items():
            g["layerskip"].append((f"skip:{n}", DraftConfig(skip_layers=s)))
    return g


# ------------------------------------------------------------------------------------------------ runner

@torch.no_grad()
def _logits_at(m, tf, lo, n):
    h = m.model(tf).last_hidden_state[:, lo : lo + n]
    return m.lm_head(h) * m.config.logit_scale


@torch.no_grad()
def _teacher(m, ids, n_new):
    x = torch.tensor([ids], device=m.device)
    out = m.generate(x, max_new_tokens=n_new, do_sample=False)
    return out[0, len(ids):].tolist()


def run(m, tok, groups=("full", "attn_only", "expert_only", "combined"), preset="quick", n_prompts=2,
        ctxs=(2048, 8192), n_new=96, gammas=(2, 4, 8), n_calib=2, calib_ctx=2048, log=print, max_configs=None):
    import sys, pathlib
    for up in (1, 2):  # /root/harness in the Modal image, <repo>/harness locally
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[up] / "harness"))
    from common import build_ids, by_split, load_workloads
    rows = load_workloads()
    dev = [r for r in by_split(rows, "dev")]
    calib = [r for r in by_split(rows, "calib")]
    grid = build_grid(preset)
    need_eat = any(c.expert_policy == "eat" for gname in groups for _, c in grid[gname])
    eat_states = None
    t0 = time.time()
    if need_eat:
        with EATCalibrator(m) as cal:
            for k in range(n_calib):
                ids = build_ids(tok, calib[k % len(calib)], calib_ctx, rotate=k)
                _logits_at(m, torch.tensor([ids], device=m.device), len(ids) - 1, 1)
        eat_states = cal.states
        log(f"EAT calibration done ({n_calib} calib prompts, ctx {calib_ctx}) {time.time()-t0:.0f}s")
    out = []
    for ctx in ctxs:
        for pi in range(n_prompts):
            task = dev[pi % len(dev)]
            ids = build_ids(tok, task, ctx, rotate=pi + 5)
            P = len(ids)
            teach = _teacher(m, ids, n_new)
            tf = torch.tensor([ids + teach], device=m.device)
            vl = _logits_at(m, tf, P - 1, n_new)
            vpred = vl.argmax(-1)[0]
            tmatch = float((vpred.cpu() == torch.tensor(teach)).float().mean())
            log(f"ctx={ctx} prompt={task['id']} P={P} verifier-vs-teacher match={tmatch:.3f}")
            done = 0
            for gname in groups:
                for name, cfg in grid[gname]:
                    if max_configs is not None and done >= max_configs:
                        break
                    done += 1
                    t1 = time.time()
                    if cfg.is_identity:
                        dl = _logits_at(m, tf, P - 1, n_new)
                        stats = {}
                    else:
                        with DraftPatch(m, cfg, eat_states) as p, p.draft():
                            dl = _logits_at(m, tf, P - 1, n_new)
                        stats = p.stats.summary()
                        stats["layers_skipped"] = p.stats.layers_skipped // 1
                    dpred = dl.argmax(-1)[0]
                    agree = (dpred == vpred).cpu().numpy()
                    top5 = (dl.topk(5, -1).indices[0] == vpred[:, None]).any(-1).float().mean().item()
                    r = dict(group=gname, name=name, cfg=cfg.to_dict(), ctx=ctx, prompt=task["id"], kind=task["kind"],
                             P=P, n_new=n_new, verifier_teacher_match=tmatch, top1_agree=float(agree.mean()),
                             verifier_top1_in_draft_top5=top5, stats=stats, seconds=time.time() - t1,
                             accepted={str(g): accepted_hist(agree, g) for g in gammas})
                    out.append(r)
                    log(f"  [{gname}] {name}: agree={r['top1_agree']:.3f} "
                        + " ".join(f"g{g}:{r['accepted'][str(g)]['mean_committed']:.2f}" for g in gammas))
    return out


def summarize(rows, gamma="4"):
    """Mean over (ctx,prompt) per config name: agreement and committed tokens/cycle; plus the exit-gate comparison
    (best combined vs best individual by mean committed at the same gamma)."""
    by = {}
    for r in rows:
        by.setdefault((r["group"], r["name"]), []).append(r)
    tab = []
    for (g, n), rs in by.items():
        tab.append(dict(group=g, name=n, n=len(rs), agree=float(np.mean([x["top1_agree"] for x in rs])),
                        committed=float(np.mean([x["accepted"][gamma]["mean_committed"] for x in rs])),
                        experts_per_token=float(np.mean([x["stats"].get("experts_per_token", np.nan) for x in rs])),
                        attended_frac=float(np.mean([x["stats"].get("attended_frac", np.nan) for x in rs]))))
    best = {g: max((t for t in tab if t["group"] == g), key=lambda t: t["committed"], default=None)
            for g in ("attn_only", "expert_only", "combined", "layerskip")}
    return dict(table=sorted(tab, key=lambda t: (t["group"], -t["committed"])), best=best, gamma=gamma)


def _main():
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--preset", default="quick", choices=["quick", "full"])
    ap.add_argument("--groups", default="full,attn_only,expert_only,combined")
    ap.add_argument("--n-prompts", type=int, default=2)
    ap.add_argument("--ctxs", default="2048,8192")
    ap.add_argument("--n-new", type=int, default=96)
    ap.add_argument("--out", default="results/approx_sweep.jsonl")
    a = ap.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "harness"))
    from loader import load_model
    m, tok = load_model(a.model_dir, "grouped_mm")
    rows = run(m, tok, tuple(a.groups.split(",")), a.preset, a.n_prompts, tuple(int(x) for x in a.ctxs.split(",")), a.n_new)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(json.dumps(r, default=float) for r in rows) + "\n")
    print(json.dumps(summarize(rows), indent=1, default=float)[:4000])


if __name__ == "__main__":
    _main()
