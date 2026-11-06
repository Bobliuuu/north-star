import pytest
import torch

from approx import DraftConfig, DraftPatch
from approx.config import EATParams, LSAParams
from fixtures import ids, param_hash, tiny_model


def logits(m, x):
    with torch.no_grad():
        return m(x).logits


def all_cfgs():
    return [
        DraftConfig(expert_r=2),
        DraftConfig(expert_r=2, weight_convention="renorm"),
        DraftConfig(expert_policy="eat", weight_convention="renorm"),
        DraftConfig(attn_budget=8, window=8, sink=2),
        DraftConfig(skip_layers={2, 3}),
        DraftConfig(attn_budget=8, window=8, sink=2, expert_r=2, skip_layers={5}),
    ]


@pytest.mark.parametrize("cfg", all_cfgs(), ids=lambda c: c.label())
def test_verifier_path_untouched(cfg):
    m = tiny_model()
    x = ids(40)
    h0, base = param_hash(m), logits(m, x)
    cls_fwd = {n: type(mod).forward for n, mod in m.named_modules()}
    with DraftPatch(m, cfg) as p:
        assert p._patched, "something must be patched"
        # installed, inactive -> bit-identical
        assert torch.equal(logits(m, x), base)
        # explicit verifier() inside draft() -> bit-identical
        with p.draft():
            with p.verifier():
                assert torch.equal(logits(m, x), base)
            d = logits(m, x)
            assert not torch.equal(d, base), "draft must actually differ for these configs"
            assert param_hash(m) == h0  # drafting never mutates weights/buffers
        assert torch.equal(logits(m, x), base)
    assert param_hash(m) == h0
    assert all("forward" not in mod.__dict__ for mod in m.modules())
    assert {n: type(mod).forward for n, mod in m.named_modules()} == cls_fwd
    assert torch.equal(logits(m, x), base)


def test_identity_cfg_is_noop_even_when_active():
    m = tiny_model()
    x = ids(40)
    base = logits(m, x)
    cfg = DraftConfig()
    assert cfg.is_identity
    with DraftPatch(m, cfg) as p:
        assert not p._patched
        with p.draft():
            assert torch.equal(logits(m, x), base)


def test_full_budget_draft_matches_verifier_numerically():
    m = tiny_model()
    x = ids(40)
    base = logits(m, x)
    cfg = DraftConfig(attn_budget=64, window=8, sink=2, attn_selector="oracle_topk", expert_r=4)  # r = native k
    with DraftPatch(m, cfg) as p, p.draft():
        d = logits(m, x)
    assert torch.allclose(d, base, atol=1e-4, rtol=1e-4), (d - base).abs().max()


def test_topr_matches_refnorth_draftmode():
    from northstar.refmodel import RefNorth
    from approx.hooks import to_ref_draft_mode
    m = tiny_model()
    x = ids(30)
    for conv in ("raw_sigmoid", "renorm"):
        cfg = DraftConfig(expert_r=2, weight_convention=conv, skip_layers={3})
        ref = RefNorth(m)
        cache = ref.new_cache(1, 64)
        want = ref.forward(x, cache, 0, to_ref_draft_mode(cfg))
        with DraftPatch(m, cfg) as p, p.draft():
            got = logits(m, x)
        assert torch.allclose(got, want, atol=1e-4, rtol=1e-4), (got - want).abs().max()


def test_skip_layers_counts_and_changes_output():
    m = tiny_model()
    x = ids(20)
    cfg = DraftConfig(skip_layers={2, 4})
    with DraftPatch(m, cfg) as p, p.draft():
        a = logits(m, x)
        assert p.stats.layers_skipped == 2
    assert not torch.allclose(a, logits(m, x))


def test_layer0_cannot_be_skipped():
    with pytest.raises(AssertionError):
        DraftConfig(skip_layers={0})


def test_attention_only_patches_global_layers():
    m = tiny_model()
    cfg = DraftConfig(attn_budget=4, window=8)
    with DraftPatch(m, cfg) as p:
        patched = {id(mod) for mod in p._patched}
        for i, layer in enumerate(m.model.layers):
            assert (id(layer.self_attn) in patched) == (m.config.layer_types[i] == "full_attention")
    cfg = DraftConfig(attn_budget=4, window=8, attn_layers="global_nope")
    with DraftPatch(m, cfg) as p:
        assert id(m.model.layers[0].self_attn) not in {id(x) for x in p._patched}


def test_hf_cache_decode_attention_consistent_with_sequence():
    """Draft attention through HF DynamicCache decode == draft in one sequence pass (same selection rule)."""
    m = tiny_model()
    x = ids(40)
    cfg = DraftConfig(attn_budget=6, window=8, sink=2, attn_selector="oracle_topk")
    with DraftPatch(m, cfg) as p, p.draft():
        full = logits(m, x)
        with torch.no_grad():
            out = m(x[:, :30], use_cache=True)
            pkv, last = out.past_key_values, []
            for t in range(30, 40):
                o = m(x[:, t : t + 1], past_key_values=pkv, use_cache=True)
                last.append(o.logits[0, 0])
    got = torch.stack(last)
    assert torch.allclose(got, full[0, 30:], atol=1e-3, rtol=1e-3), (got - full[0, 30:]).abs().max()
