"""Experiment A driver: sensitivity table -> arms (greedy/random/uniform/north_blind/north_table) -> gates -> claim."""
from __future__ import annotations

import numpy as np
import torch

from weightedit.recipe import moe_layers

from .gates import REG, apply_gates, claim_check, nonexpert_fingerprint
from .metrics import eval_vs_teacher, summarize
from .proposers import LayerProposer, greedy_from_table, random_layers
from .recipe import QRecipe


def sensitivity(m, editor, cal_seqs, cal_lo, cal_cache, group, layers=None, log=print):
    table = {}
    for i in (layers if layers is not None else editor.layers):
        editor.apply([i], group)
        kls, ags = eval_vs_teacher(m, cal_seqs, cal_lo, cal_cache)
        table[i] = summarize(kls, ags)["kl"]
        log(f"  sens layer {i}: kl={table[i]:.5f}")
    editor.restore_all()
    return table


def evaluate_recipe(m, editor, recipe: QRecipe, data, fp0, source, name, cfg=None):
    mc = m.config
    editor.apply(recipe.layers(), recipe.group)
    kls, ags = eval_vs_teacher(m, data["dev_seqs"], data["dev_lo"], data["dev_cache"])
    num = summarize(kls, ags)
    bs = recipe.bytes_saved(mc)
    gates = apply_gates(num, bs, nonexpert_fingerprint(m) == fp0, editor.unlisted_identical(recipe.layers()),
                        editor.listed_changed(recipe.layers()) if recipe.layers() else True, source, cfg)
    editor.restore_all()
    return dict(name=name, source=source, layers=recipe.layers(), group=recipe.group, K=len(recipe.layers()),
                kl=num["kl"], agree=num["agree"], n_pos=num["n_pos"], kl_by_seq=num["kl_by_seq"],
                bytes_saved_inferred=bs, gates=gates, passed=all(gates.values()))


def run_experiment_A(m, editor, data, cfg, gen_fn=None, log=print, ckpt=lambda name, obj: None):
    """cfg: K, group, rounds, n_random, frontier_Ks, sens_layers(None=all), gates(optional override)."""
    mc, K, g = m.config, cfg["K"], cfg["group"]
    fp0 = nonexpert_fingerprint(m)
    table = sensitivity(m, editor, data["cal_seqs"], data["cal_lo"], data["cal_cache"], g, cfg.get("sens_layers"), log)
    ckpt("sensitivity", table)
    res = []

    def add(r):
        res.append(r)
        log(f"{r['name']}: K={r['K']} src={r['source']} kl={r['kl']:.5f} agree={r['agree']:.4f} passed={r['passed']} "
            f"gates={r['gates']}")
        ckpt("arm_" + r["name"], r)
        return r

    ev = lambda layers, src, name: evaluate_recipe(m, editor, QRecipe(tuple(layers), g).validate(mc), data, fp0, src, name,
                                                   cfg.get("gates"))
    for k in cfg.get("frontier_Ks", []):
        add(ev(greedy_from_table(table, k), "harness", f"greedy_table_K{k}"))
    greedy_main = add(ev(greedy_from_table(table, K), "harness", f"greedy_table_K{K}")) \
        if K not in cfg.get("frontier_Ks", []) else next(r for r in res if r["name"] == f"greedy_table_K{K}")
    rnd = [add(ev(random_layers(mc, K, s), "harness", f"random_s{s}_K{K}")) for s in range(cfg.get("n_random", 5))]
    if cfg.get("uniform_all", True):
        add(ev(moe_layers(mc), "harness", "uniform_all"))
    norths = {}
    if gen_fn is not None:
        for arm, tab in (("north_blind", None), ("north_table", table)):
            prop, hist, norths[arm] = LayerProposer(gen_fn, mc, K, tab, g), [], []
            for r in range(cfg.get("rounds", 3)):
                d, meta = prop.propose(hist)
                rec = QRecipe.from_dict(d).validate(mc, exact_k=K)
                out = add(ev(rec.layers(), meta["source"], f"{arm}_r{r+1}"))
                out["raw_reply"] = meta.get("raw"); out["proposer_error"] = meta.get("error")
                hist.append(out); norths[arm].append(out)
                ckpt("arm_" + out["name"], out)
    rkl = [r["kl"] for r in rnd]
    claims = {arm: [claim_check(o, rkl, greedy_main["kl"]) | dict(name=o["name"]) for o in outs] for arm, outs in norths.items()}
    summary = dict(K=K, group=g, registered_gates={**REG, **(cfg.get("gates") or {})}, random_kls=rkl,
                   greedy_main_kl=greedy_main["kl"], claims=claims,
                   any_self_improvement_claim=any(c["claim"] for cs in claims.values() for c in cs))
    ckpt("summary_A", summary)
    return dict(table=table, arms=res, summary=summary)
