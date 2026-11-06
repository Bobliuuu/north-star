"""Build the dependency DAG of one steady-state speculative cycle.

Cycle = [commit(prev) -> index update] -> gamma' draft steps -> checkpoint -> verify(gamma+1 tokens) -> accept.
Each draft step / the verify pass is a chain of layer groups. Inside a group the attention branch, the MoE branch
(and the dense branch in group 0) are INDEPENDENT (North's parallel block) and join at the residual add, i.e. the
next group depends on all branches of the previous one. Draft skips layers in `cfg.skip`; verify never does.

Scenarios (cross-cycle speculation, modes spec_two_pass / fused): the verify pass also carries ONE speculative
next-cycle draft token through the non-skipped layers. With probability h = P(A>=gamma) * carry_hit_given_full
the next cycle's first draft step is already done (scenario 'A': gamma-1 draft steps); otherwise (scenario 'B')
all gamma steps run and the speculative work is wasted. mode two_pass has no speculation (scenario 'B' only, h=0).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config as C


@dataclass
class Task:
    id: str
    kind: str                    # attn | moe | dense | lm_head | idx | ckpt | commit | accept
    phase: str                   # prep | draft | verify | tail
    variants: list
    deps: list = field(default_factory=list)
    group: int = -1
    step: int = 0


@dataclass
class Buffer:
    name: str
    nbytes: int
    start_task: str
    end_tasks: list


@dataclass
class Dag:
    tasks: list                  # topologically ordered
    buffers: list
    meta: dict

    def by_id(self):
        return {t.id: t for t in self.tasks}


def build_cycle(cm, bucket, cfg, scenario="B", n_groups=6) -> Dag:
    B = bucket.batch; ctx = bucket.ctx
    gamma = cfg.gamma
    Tv = B * (gamma + 1)
    spec = cfg.mode != "two_pass"
    nds = gamma - (1 if (scenario == "A" and spec) else 0)
    groups = C.layer_groups(n_groups)
    skip = set(cfg.skip)
    if 0 in skip:
        raise ValueError("layer 0 cannot be skipped")
    tasks = []; add = lambda t: (tasks.append(t), t)[1]

    commit = add(Task("commit", "commit", "prep", cm.commit()))
    ckpt = add(Task("ckpt", "ckpt", "prep", cm.ckpt_save(Tv), []))      # needs no earlier work; schedulable freely
    idx = {}
    if cfg.attn_budget:
        for g, ls in enumerate(groups):
            ng = sum(1 for l in ls if C.is_global(l) and l not in skip)
            if ng:
                idx[g] = add(Task(f"idx.g{g}", "idx", "prep", cm.index_update(ng, Tv), [commit.id], g))

    prev = [commit.id]            # join set feeding the next group
    first_draft = None
    for s in range(1, nds + 1):
        for g, ls in enumerate(groups):
            act = [l for l in ls if l not in skip]
            if not act:
                continue
            branches = []
            av = cm.attn_group(act, B, ctx, cfg.attn_budget, draft=True)
            deps = list(prev) + ([idx[g].id] if g in idx else [])
            t = add(Task(f"d{s}.attn.g{g}", "attn", "draft", av, deps, g, s)); branches.append(t.id)
            first_draft = first_draft or t.id
            if 0 in act:
                branches.append(add(Task(f"d{s}.dense.g{g}", "dense", "draft", cm.dense_group(B), list(prev), g, s)).id)
            nm = sum(1 for l in act if l != 0)
            if nm:
                branches.append(add(Task(f"d{s}.moe.g{g}", "moe", "draft",
                                         cm.moe_group(nm, B, cfg.r, "two_pass", "draft"), list(prev), g, s)).id)
            prev = branches
        lm = add(Task(f"d{s}.lm", "lm_head", "draft", cm.lm_head(B), list(prev), -1, s))
        prev = [lm.id]
    last_draft_join = list(prev)

    vprev = last_draft_join + [ckpt.id]
    for g, ls in enumerate(groups):
        branches = []
        av = cm.attn_group(ls, Tv, ctx, 0, draft=False)
        if spec:
            act = [l for l in ls if l not in skip]
            if act:     # speculative draft token's attention rides along in the verify pass
                extra = cm.attn_group(act, B, ctx, cfg.attn_budget, draft=True)
                av = [_addv(v, e) for v, e in zip(av, extra)]
        branches.append(add(Task(f"v.attn.g{g}", "attn", "verify", av, list(vprev), g)).id)
        if 0 in ls:
            branches.append(add(Task(f"v.dense.g{g}", "dense", "verify", cm.dense_group(Tv), list(vprev), g)).id)
        nm = sum(1 for l in ls if l != 0)
        if nm:
            nspec = sum(1 for l in ls if l != 0 and l not in skip) if spec else 0
            mv = cm.moe_group(nm, Tv, cfg.r, cfg.mode, "verify", n_spec_layers=nspec, spec_tokens=B)
            branches.append(add(Task(f"v.moe.g{g}", "moe", "verify", mv, list(vprev), g)).id)
        vprev = branches
    lmv = add(Task("v.lm", "lm_head", "verify", cm.lm_head(Tv), list(vprev)))
    acc = add(Task("accept", "accept", "tail", cm.accept(), [lmv.id]))

    bufs = []
    n_act = C.N_LAYERS - len(skip)
    per_step = cm.draft_scratch_bytes(1, n_act, B, 0, ctx)
    idx_bytes = cm.draft_scratch_bytes(0, n_act, B, cfg.attn_budget, ctx)
    d_first = {}
    for t in tasks:
        if t.phase == "draft" and t.step not in d_first:
            d_first[t.step] = t.id
    for s, tid in d_first.items():
        bufs.append(Buffer(f"draft_kv.s{s}", per_step, tid, [acc.id]))
    if idx_bytes and idx:
        bufs.append(Buffer("sparse_index", idx_bytes, min(idx.values(), key=lambda t: t.group).id, [acc.id]))
    bufs.append(Buffer("ckpt", cm.ckpt_bytes(Tv), ckpt.id, [acc.id]))
    return Dag(tasks, bufs, {"scenario": scenario, "gamma": gamma, "n_draft_steps": nds, "groups": n_groups,
                             "mode": cfg.mode})


def _addv(a, b):
    from .costmodel import Variant
    src = dict(a.src_us)
    for k, x in b.src_us:
        src[k] = src.get(k, 0.0) + x
    return Variant(a.name, a.us + b.us, a.bw, a.sm, tuple(sorted(src.items())))


def scenarios(cm, bucket, cfg, h, n_groups=6):
    """[(probability, Dag)] for the cycle."""
    if cfg.mode == "two_pass" or h <= 0.0:
        return [(1.0, build_cycle(cm, bucket, cfg, "B", n_groups))]
    return [(h, build_cycle(cm, bucket, cfg, "A", n_groups)),
            (1.0 - h, build_cycle(cm, bucket, cfg, "B", n_groups))]
