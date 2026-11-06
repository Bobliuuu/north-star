"""Independent gates for a trained edit. The trainer never sees the gate split."""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from approx import DraftConfig, DraftPatch
from approx.sweep import accepted_hist, accepted_lengths

from .adapter import EditedDraft, RouterAdapters
from .distill import logits_of
from .recipe import Recipe, moe_layers

DEFAULTS = dict(margin=0.02, max_overfit=0.10, max_adapter_flops_frac=0.05, gamma=4, boot=1000)


def fingerprint(m, mode="full"):
    """sha256 over parameters+buffers. mode='sampled' hashes shape + first/last 4096 elems (big models)."""
    h = hashlib.sha256()
    for n, t in list(m.named_parameters()) + list(m.named_buffers()):
        h.update(n.encode())
        t = t.detach()
        if mode == "sampled" and t.numel() > 8192:
            f = t.reshape(-1)
            t = torch.cat([f[:4096], f[-4096:]])
            h.update(str(tuple(t.shape)).encode())
        h.update(t.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def _vpred_and_logits(m, seq):
    x = torch.tensor([seq], device=next(m.parameters()).device)
    return logits_of(m, x)[0]


@torch.no_grad()
def agreement(m, seqs, eval_from, draft_ctx):
    """Per-sequence bool arrays: draft top-1 == verifier top-1 at positions eval_from..L-2 (teacher-forced on seq).
    draft_ctx: context manager factory yielding a draft-mode scope (or None -> verifier==draft, sanity)."""
    out = []
    for seq, lo in zip(seqs, eval_from):
        v = _vpred_and_logits(m, seq)[lo:-1].argmax(-1)
        with draft_ctx():
            d = _vpred_and_logits(m, seq)[lo:-1].argmax(-1)
        out.append((d == v).cpu().numpy())
    return out


def committed(agree_list, gamma):
    return float(np.mean([accepted_lengths(a, gamma).mean() + 1 for a in agree_list]))


def boot_lower(a, b, gamma, n=1000, seed=0, q=5):
    """Paired cluster bootstrap over sequences: q-th percentile of (mean_b - mean_a) agreement and committed."""
    rng = np.random.default_rng(seed)
    k = len(a)
    ag_a = np.array([x.mean() for x in a]); ag_b = np.array([x.mean() for x in b])
    cm_a = np.array([accepted_lengths(x, gamma).mean() + 1 for x in a])
    cm_b = np.array([accepted_lengths(x, gamma).mean() + 1 for x in b])
    da, dc = [], []
    for _ in range(n):
        ix = rng.integers(0, k, k)
        da.append((ag_b[ix] - ag_a[ix]).mean()); dc.append((cm_b[ix] - cm_a[ix]).mean())
    return float(np.percentile(da, q)), float(np.percentile(dc, q))


def baseline_ctx(m, r):
    cfg = DraftConfig(expert_r=r, weight_convention="raw_sigmoid")
    def f():
        p = DraftPatch(m, cfg).install()
        class C:
            def __enter__(s): p.active = True; return p
            def __exit__(s, *a): p.remove()
        return C()
    return f


def edited_ctx(patch: EditedDraft):
    return patch.draft


def run_gates(m, patch: EditedDraft, adapters: RouterAdapters, recipe: Recipe, train_seqs, train_from, gate_seqs,
              gate_from, fp_before, fp_mode="full", cache=None, **kw):
    """Returns report dict with per-gate booleans, numbers, and `passed`. `cache` memoises baseline agreement."""
    c = {**DEFAULTS, **kw}
    rep = dict(recipe=recipe.to_dict(), gates={}, numbers={})
    # G-lossless: base unchanged, verifier logits bit-identical with patch installed but inactive, and under verifier()
    plain = [_vpred_and_logits(m, s) for s in gate_seqs[:2]]
    patch.remove()
    ref = [_vpred_and_logits(m, s) for s in gate_seqs[:2]]
    key = ("base", recipe.expert_r, len(gate_seqs))  # baseline needs the module forwards free (DraftPatch asserts)
    if cache is not None and key in cache:
        base_g = cache[key]
    else:
        base_g = agreement(m, gate_seqs, gate_from, baseline_ctx(m, recipe.expert_r))
        if cache is not None:
            cache[key] = base_g
    patch.install()
    ident = all(torch.equal(a, b) for a, b in zip(plain, ref))
    with patch.draft(), patch.verifier():
        in_v = [_vpred_and_logits(m, s) for s in gate_seqs[:2]]
    ident = ident and all(torch.equal(a, b) for a, b in zip(ref, in_v))
    same_params = fingerprint(m, fp_mode) == fp_before
    rep["gates"]["lossless"] = bool(ident and same_params)
    rep["numbers"].update(verifier_bit_identical=bool(ident), base_params_unchanged=bool(same_params), fp_mode=fp_mode)
    # G-gain
    edit_g = agreement(m, gate_seqs, gate_from, edited_ctx(patch))
    edit_t = agreement(m, train_seqs, train_from, edited_ctx(patch))
    g = c["gamma"]
    ag_b, ag_e = float(np.mean(np.concatenate(base_g))), float(np.mean(np.concatenate(edit_g)))
    ag_t = float(np.mean(np.concatenate(edit_t)))
    lo_ag, lo_cm = boot_lower(base_g, edit_g, g, c["boot"])
    rep["numbers"].update(agree_base=ag_b, agree_edit=ag_e, agree_train=ag_t, gain=ag_e - ag_b,
                          committed_base=committed(base_g, g), committed_edit=committed(edit_g, g),
                          gain_ci_lower_agree=lo_ag, gain_ci_lower_committed=lo_cm, gamma=g, n_gate_seqs=len(gate_seqs))
    rep["gates"]["gain"] = bool(ag_e - ag_b >= c["margin"] and lo_ag > 0)
    # G-overfit
    rep["gates"]["overfit"] = bool(ag_t - ag_e <= c["max_overfit"])
    # G-cost
    mc = m.config
    exp_flops = recipe.expert_r * 2 * 3 * mc.hidden_size * mc.intermediate_size
    ad_flops = adapters.flops_per_token_layer(mc.num_experts)
    exact_r = patch.tokens == 0 or patch.slots_active == patch.tokens * recipe.expert_r
    rep["numbers"].update(adapter_params=adapters.n_params(), adapter_flops_frac=ad_flops / exp_flops,
                          active_experts_exact=bool(exact_r))
    rep["gates"]["cost"] = bool(exact_r and ad_flops / exp_flops <= c["max_adapter_flops_frac"])
    rep["passed"] = all(rep["gates"].values())
    return rep
