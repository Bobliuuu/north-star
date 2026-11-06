import json

import pytest
import torch

from approx import DraftConfig, DraftPatch
from tiny import seqs, tiny_model
from weightedit import EditedDraft, Recipe, RecipeError, RouterAdapters
from weightedit.distill import grad_checkpointed, kl_loss, logits_of, train
from weightedit.gates import fingerprint
from weightedit.loop import GridProposer, LLMProposer, extract_json, run_loop, run_round
from weightedit.store import EditStore


def L(m, x):
    with torch.no_grad():
        return m(x).logits


def test_identity_init_equals_unedited_topr_draft():
    m = tiny_model()
    x = torch.tensor(seqs(1, 40, 1))
    r = Recipe(expert_r=2).validate(m.config)
    ad = RouterAdapters(m.config, r)
    with DraftPatch(m, DraftConfig(expert_r=2, weight_convention="raw_sigmoid")) as p, p.draft():
        base = L(m, x)
    with EditedDraft(m, ad, 2) as e, e.draft():
        got = L(m, x)
    assert torch.allclose(base, got, atol=1e-5)
    assert not torch.equal(base, L(m, x)), "draft must differ from verifier"


def test_verifier_untouched_and_params_frozen():
    m = tiny_model()
    x = torch.tensor(seqs(1, 40, 1))
    h0, base = fingerprint(m), L(m, x)
    r = Recipe(expert_r=2, steps=5, window=16, batch=2).validate(m.config)
    ad = RouterAdapters(m.config, r)
    with EditedDraft(m, ad, 2) as e:
        assert torch.equal(L(m, x), base)               # installed, inactive
        train(m, ad, e, seqs(4, 40, 2), r, log=lambda *_: None)
        assert torch.equal(L(m, x), base)               # after training, still bit-identical
        with e.draft(), e.verifier():
            assert torch.equal(L(m, x), base)
    assert fingerprint(m) == h0
    assert all("forward" not in mod.__dict__ for mod in m.modules())
    assert all(p.grad is None for p in m.parameters())


def test_training_reduces_kl():
    m = tiny_model()
    r = Recipe(expert_r=2, steps=150, lr=3e-3, window=32, batch=4, rank=8).validate(m.config)
    ad = RouterAdapters(m.config, r)
    with EditedDraft(m, ad, 2) as e:
        out = train(m, ad, e, seqs(8, 64, 3), r, log=lambda *_: None)
    assert out["last"] < 0.9 * out["first"], out


def test_grad_checkpoint_matches():
    m = tiny_model()
    r = Recipe(expert_r=2, steps=1, window=32, batch=2).validate(m.config)
    x = torch.tensor(seqs(2, 32, 4))
    grads = []
    for ck in (False, True):
        ad = RouterAdapters(m.config, r)
        with EditedDraft(m, ad, 2) as e:
            for p in ad.parameters():
                p.data += 0.01 * torch.randn_like(p) if ad.kind else 0
            torch.manual_seed(0)
            for p in ad.parameters():
                p.data.copy_(torch.randn_like(p) * 0.02)
            import contextlib
            from weightedit.distill import frozen
            with frozen(m), (grad_checkpointed(m) if ck else contextlib.nullcontext()):
                with torch.no_grad(), e.verifier():
                    tl = logits_of(m, x)
                with e.draft():
                    loss = kl_loss(tl, logits_of(m, x))
                    loss.backward()
        grads.append([p.grad.clone() for p in ad.parameters()])
    for a, b in zip(*grads):
        assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("bad", [dict(expert_r=0), dict(expert_r=4), dict(rank=17), dict(steps=0), dict(lr=1.0),
                                 dict(kind="full_weights"), dict(layers=[0]), dict(layers=[]), dict(window=4),
                                 dict(gain="yes")])
def test_recipe_rejects(bad):
    m = tiny_model()
    with pytest.raises(RecipeError):
        Recipe(**bad).validate(m.config)


def test_recipe_unknown_field_and_json():
    with pytest.raises(RecipeError):
        Recipe.from_dict({"expert_r": 2, "lr_scale": 3})
    with pytest.raises(RecipeError):
        Recipe.from_json("[1]")
    assert Recipe.from_json('{"expert_r":3}').expert_r == 3


def test_extract_json():
    assert extract_json('sure {"a": {"b": 1}} ok') == {"a": {"b": 1}}
    assert extract_json("none") is None
    assert extract_json('{bad} {"x":2}') == {"x": 2}
    assert extract_json('ex {"a":1} final {"b":2}') == {"b": 2}
    assert extract_json('ex {"a":1} final {"b":2}', last=False) == {"a": 1}


def test_llm_proposer_valid_and_fallback():
    m = tiny_model()
    good = LLMProposer(lambda p: 'Here: {"expert_r": 3, "rank": 2}', m.config)
    d, meta = good.propose([])
    assert meta["source"] == "llm" and d["expert_r"] == 3
    bad = LLMProposer(lambda p: '{"expert_r": 99}', m.config)
    d, meta = bad.propose([])
    assert meta["source"] == "fallback" and "RecipeError" in meta["error"]
    boom = LLMProposer(lambda p: (_ for _ in ()).throw(RuntimeError("x")), m.config)
    assert boom.propose([])[1]["source"] == "fallback"
    assert "Past rounds" in good.prompt([{"recipe": {"expert_r": 2}, "passed": False, "gates": {"gain": False}}])


def test_store_commit_rollback(tmp_path):
    m = tiny_model()
    r = Recipe(expert_r=2).validate(m.config)
    ad = RouterAdapters(m.config, r)
    st = EditStore(tmp_path)
    v1 = st.commit(ad, r, dict(numbers={}), "grid")
    v2 = st.commit(ad, r, dict(numbers={}), "llm")
    assert (v1, v2, st.current()) == ("v001", "v002", "v002")
    st.rollback(v1)
    assert st.current() == "v001"
    st.rollback("")
    assert st.current() == "" and st.load_state() is None
    st.reject(r, {}, "grid", "x")
    ev = [e["event"] for e in st.ledger()]
    assert ev == ["commit", "commit", "rollback", "rollback", "reject"]
    assert st.versions() == ["v001", "v002"]
    sd = st.load_state("v001")
    ad2 = RouterAdapters(m.config, st.load_recipe("v001"))
    ad2.load_state_dict(sd)


def test_loop_end_to_end(tmp_path):
    m = tiny_model()
    tr, ga = seqs(8, 64, 5), seqs(8, 64, 6)
    st = EditStore(tmp_path)
    prop = GridProposer(rs=(2,), ranks=(8,), lrs=(3e-3, 2e-3), steps=150, window=32, batch=4)
    hist = run_loop(m, prop, st, tr, [0] * 8, ga, [0] * 8, rounds=2, log=lambda *_: None, margin=0.0, boot=200)
    assert len(hist) == 2 and len(st.ledger()) == 2
    for h in hist:
        assert h["gates"]["lossless"] and h["gates"]["cost"]
        assert h["numbers"]["verifier_bit_identical"] and h["numbers"]["base_params_unchanged"]
    # invalid proposal -> recorded reject, nothing trained
    class Bad:
        def propose(self, h): return dict(expert_r=99), dict(source="llm")
    e = run_round(m, Bad(), st, tr, [0] * 8, ga, [0] * 8, hist, log=lambda *_: None)
    assert not e["passed"] and "invalid recipe" in e["error"]
    assert st.ledger()[-1]["event"] == "reject"


def test_train_ckpt_same_as_plain():
    outs = []
    for ck in (False, True):
        m = tiny_model()
        r = Recipe(expert_r=2, steps=4, window=32, batch=2, rank=4).validate(m.config)
        ad = RouterAdapters(m.config, r)
        with EditedDraft(m, ad, 2) as e:
            outs.append(train(m, ad, e, seqs(4, 40, 2), r, log=lambda *_: None, ckpt=ck)["loss"])
    assert all(abs(a - b) < 1e-4 for a, b in zip(*outs)), outs


def test_grad_checkpoint_with_layer_skip_restores_forward():
    """Regression: r4s crashed (AssertionError in grad_checkpointed) because skipped layers already had an instance forward."""
    m = tiny_model()
    r = Recipe(expert_r=2, steps=2, window=32, batch=2, rank=4).validate(m.config)
    ad = RouterAdapters(m.config, r)
    with EditedDraft(m, ad, 2, skip_layers={1}) as e:
        before = {i: "forward" in L.__dict__ for i, L in enumerate(m.model.layers)}
        assert before[1]
        train(m, ad, e, seqs(4, 40, 2), r, log=lambda *_: None, ckpt=True)
        after = {i: "forward" in L.__dict__ for i, L in enumerate(m.model.layers)}
        assert before == after


def test_duplicate_recipe_skipped(tmp_path):
    m = tiny_model()
    tr, ga = seqs(4, 64, 5), seqs(4, 64, 6)
    st = EditStore(tmp_path)
    class Same:
        def propose(self, h): return dict(expert_r=2, steps=3, window=32, batch=2), dict(source="grid")
    hist = []
    run_round(m, Same(), st, tr, [0] * 4, ga, [0] * 4, hist, log=lambda *_: None, boot=50)
    e = run_round(m, Same(), st, tr, [0] * 4, ga, [0] * 4, hist, log=lambda *_: None, boot=50)
    assert not e["passed"] and "duplicate" in e["error"]


def test_llm_proposer_unwraps_history_style_reply():
    m = tiny_model()
    p = LLMProposer(lambda _: 'x {"recipe": {"expert_r": 3, "rank": 2}, "passed": true, "agree_base": 0.5}', m.config)
    d, meta = p.propose([])
    assert meta["source"] == "llm" and d == {"expert_r": 3, "rank": 2}


def test_grid_quants():
    g = GridProposer(rs=(2,), ranks=(4,), lrs=(1e-3,), quants=("nvfp4", "int4"))
    assert [g.propose([])[0]["expert_quant"] for _ in range(2)] == ["nvfp4", "int4"]
