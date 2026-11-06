import numpy as np
import pytest
import torch

from tiny import seqs, tiny_model
from selfedit.experiment import run_experiment_A
from selfedit.gates import claim_check, nonexpert_fingerprint
from selfedit.metrics import eval_vs_teacher, summarize, teacher_cache
from selfedit.proposers import LayerProposer, greedy_from_table, random_layers
from selfedit.quant import ExpertQuantEditor, fake_quant_int4
from selfedit.recipe import QRecipe
from selfedit.recover import AlwaysOnRouterEdit, run_experiment_B
from weightedit.recipe import RecipeError


def data_for(m, n=4, L=48, lo=8):
    dev, cal = seqs(n, L, 11), seqs(n, L, 12)
    return dict(dev_seqs=dev, dev_lo=lo, dev_cache=teacher_cache(m, dev, lo),
                cal_seqs=cal, cal_lo=lo, cal_cache=teacher_cache(m, cal, lo))


def test_fake_quant_grid_and_error():
    w = torch.randn(4, 8, 64)
    q = fake_quant_int4(w, 32)
    g = q.reshape(4, 8, 2, 32)
    # at most 16 distinct values per group (int4), error bounded by half a step (+bf16 scale rounding)
    assert all(len(torch.unique(g[a, b, c])) <= 16 for a in range(4) for b in range(8) for c in range(2))
    amax = w.reshape(4, 8, 2, 32).abs().amax(-1, keepdim=True)
    assert ((w - q).reshape(4, 8, 2, 32).abs() <= amax / 7 * 0.5 * 1.02 + 1e-6).all()
    assert fake_quant_int4(q, 32).sub(q).abs().max() < 1e-2  # (near) idempotent up to bf16 scale
    assert fake_quant_int4(w.bfloat16(), 32).dtype == torch.bfloat16
    with pytest.raises(AssertionError):
        fake_quant_int4(torch.randn(4, 30), 32)


def test_editor_apply_restore_exact():
    m = tiny_model()
    fp0 = nonexpert_fingerprint(m)
    ed = ExpertQuantEditor(m)
    before = {i: (m.model.layers[i].mlp.experts.gate_up_proj.clone()) for i in ed.layers}
    ed.apply([1, 3], 32)
    assert ed.listed_changed([1, 3]) and ed.unlisted_identical([1, 3])
    assert not torch.equal(m.model.layers[1].mlp.experts.gate_up_proj, before[1])
    assert nonexpert_fingerprint(m) == fp0
    ed.apply([2], 32)  # does not compound: layers 1,3 back to original
    assert torch.equal(m.model.layers[1].mlp.experts.gate_up_proj, before[1])
    ed.restore_all()
    assert all(torch.equal(m.model.layers[i].mlp.experts.gate_up_proj, before[i]) for i in ed.layers)


def test_kl_zero_unedited_and_grows_with_layers():
    m = tiny_model()
    d = data_for(m)
    ed = ExpertQuantEditor(m)
    k, a = eval_vs_teacher(m, d["dev_seqs"], d["dev_lo"], d["dev_cache"])
    assert summarize(k, a)["kl"] < 1e-9 and summarize(k, a)["agree"] == 1.0
    kls = []
    for layers in ([1], [1, 2, 3], ed.layers):
        ed.apply(layers, 32)
        k, a = eval_vs_teacher(m, d["dev_seqs"], d["dev_lo"], d["dev_cache"])
        kls.append(summarize(k, a)["kl"])
    ed.restore_all()
    assert kls[0] > 0 and kls[2] > kls[0]


def test_recipe_validation():
    m = tiny_model()
    c = m.config
    QRecipe((1, 2), 32).validate(c, exact_k=2)
    for bad in (QRecipe((0,), 32), QRecipe((99,), 32), QRecipe((1, 1), 32), QRecipe((1,), 7), QRecipe((1,), 64)):
        with pytest.raises(RecipeError):
            bad.validate(c)
    with pytest.raises(RecipeError):
        QRecipe((1, 2), 32).validate(c, exact_k=3)
    with pytest.raises(RecipeError):
        QRecipe.from_dict({"int4_layers": [1], "bits": 3})
    assert 0 < QRecipe((1, 2), 32).bytes_saved(c) < 1


def test_proposer_valid_and_fallback_and_prompt():
    m = tiny_model()
    ok = LayerProposer(lambda p: 'x {"int4_layers":[1,2],"group":32}', m.config, 2, table={1: .1, 2: .2, 3: .3, 4: .4})
    d, meta = ok.propose([])
    assert meta["source"] == "llm" and d["int4_layers"] == [1, 2]
    assert "1:0.1000" in ok.prompt([]) and "EXACTLY 2" in ok.prompt([])
    for reply in ('{"int4_layers":[1]}', "no json", '{"int4_layers":[0,1]}'):
        d, meta = LayerProposer(lambda p, r=reply: r, m.config, 2).propose([])
        assert meta["source"] == "fallback" and len(d["int4_layers"]) == 2 and meta["raw"] == reply
    assert greedy_from_table({1: .3, 2: .1, 3: .2}, 2) == [2, 3]
    assert random_layers(m.config, 2, 0) == random_layers(m.config, 2, 0)


def test_claim_logic():
    mk = lambda kl, **g: dict(kl=kl, gates=dict(quality=True, bytes=True, nonexpert=True, selfauthored=True) | g)
    assert claim_check(mk(.01), [.02, .03, .04, .005, .006])["claim"]            # beats 3 of 5
    assert not claim_check(mk(.01), [.02, .03, .005, .004, .006])["claim"]       # beats 2 of 5
    assert not claim_check(mk(.01, selfauthored=False), [.02, .03, .04, .05, .06])["claim"]
    assert not claim_check(mk(.01, quality=False), [.02, .03, .04, .05, .06])["claim"]


def test_experiment_A_end_to_end():
    m = tiny_model()
    d = data_for(m)
    ed = ExpertQuantEditor(m)
    fp0 = nonexpert_fingerprint(m)
    replies = iter(['{"int4_layers":[1,2],"group":32}', "garbage"] * 4)
    out = {}
    res = run_experiment_A(m, ed, d, dict(K=2, group=32, rounds=2, n_random=3, frontier_Ks=[1], gates=dict(kl_max=1e9, agree_min=0.0, bytes_min=0.0)),
                           gen_fn=lambda p: next(replies), log=lambda *_: None, ckpt=lambda n, o: out.update({n: o}))
    names = [a["name"] for a in res["arms"]]
    assert "greedy_table_K1" in names and "greedy_table_K2" in names and "uniform_all" in names
    assert sum(n.startswith("random_") for n in names) == 3
    assert {"north_blind_r1", "north_blind_r2", "north_table_r1", "north_table_r2"} <= set(names)
    srcs = {a["name"]: a["source"] for a in res["arms"]}
    assert srcs["north_blind_r1"] == "llm" and srcs["north_blind_r2"] == "fallback"
    assert all(a["gates"]["nonexpert"] for a in res["arms"])
    assert nonexpert_fingerprint(m) == fp0 and ed.unlisted_identical([]) and not ed.applied
    assert "summary_A" in out and "sensitivity" in out and set(res["table"]) == set(ed.layers)
    # model fully restored: unedited KL is exactly 0
    k, a = eval_vs_teacher(m, d["dev_seqs"], d["dev_lo"], d["dev_cache"])
    assert summarize(k, a)["kl"] < 1e-9


def test_always_on_identity_and_recovery():
    m = tiny_model()
    d = data_for(m)
    ed = ExpertQuantEditor(m)
    tr = seqs(8, 32, 21)
    train = dict(tr_seqs=tr, tr_lo=4, tr_cache=teacher_cache(m, tr, 4))
    # identity adapter on the unedited model == original logits
    from weightedit.adapter import RouterAdapters
    from weightedit.recipe import Recipe
    ad = RouterAdapters(m.config, Recipe(rank=4))
    x = torch.tensor(seqs(1, 40, 3))
    with torch.no_grad():
        base = m(x).logits
        with AlwaysOnRouterEdit(m, ad):
            assert torch.allclose(m(x).logits, base, atol=1e-5)
    out = run_experiment_B(m, ed, QRecipe(tuple(ed.layers), 32).validate(m.config), d, train,
                           dict(steps=120, lr=3e-3, batch=2, rank=8, ckpt=False), log=lambda *_: None)
    assert out["train"]["last"] < out["train"]["first"]
    assert out["kl_train_after"] < out["kl_train_before"], out      # fits the train split
    rule = out["rel_reduction"] >= 0.15 and out["boot_lower_rel"] > 0 and out["agree_after"] >= out["agree_before"]
    assert out["gates"]["recover"] == bool(rule) and out["passed"] == (rule and out["gates"]["cost"])  # gate follows its rule
    assert abs(out["kl_adapter_off"] - out["kl_before"]) < 1e-6  # adapter-off == edited-only baseline
    assert not ed.applied and not any("forward" in mod.__dict__ for mod in m.modules())
