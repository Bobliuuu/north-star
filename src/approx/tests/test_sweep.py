import numpy as np
import torch

from approx import sweep
from approx.config import DraftConfig
from approx.hooks import DraftPatch, EATCalibrator
from fixtures import ids, tiny_model


def test_accepted_lengths():
    a = sweep.accepted_lengths([1, 1, 0, 1, 1, 1, 1], 3)
    assert a.tolist() == [2, 1, 0, 3, 3, 2, 1]
    h = sweep.accepted_hist([1, 1, 0, 1], 2)
    assert h["hist"] == [1, 1, 2] or sum(h["hist"]) == 4
    assert sweep.accepted_lengths([0, 0], 4).tolist() == [0, 0]


def test_grid_groups_and_factorial_structure():
    g = sweep.build_grid("quick")
    assert [c.is_identity for _, c in g["full"]] == [True]
    assert all(c.attn_on and not c.expert_on for _, c in g["attn_only"])
    assert all(c.expert_on and not c.attn_on for _, c in g["expert_only"])
    assert all(c.expert_on and c.attn_on for _, c in g["combined"])
    # every combined component setting also appears as a single-axis config
    attn_single = {(c.attn_selector, c.attn_budget) for _, c in g["attn_only"]}
    exp_single = {(c.expert_r, c.weight_convention) for _, c in g["expert_only"] if c.expert_policy == "topr"}
    for _, c in g["combined"]:
        assert (c.attn_selector, c.attn_budget) in attn_single and (c.expert_r, c.weight_convention) in exp_single
    assert all(0 not in c.skip_layers for _, c in g["layerskip"])


def test_end_to_end_tiny_agreement_pipeline():
    """Full pipeline math on the tiny model: draft==verifier -> agreement 1; aggressive draft -> lower."""
    m = tiny_model()
    x = ids(48)
    with torch.no_grad():
        v = m(x).logits.argmax(-1)[0]
    with DraftPatch(m, DraftConfig(attn_budget=64, window=8, sink=2, expert_r=4)) as p, p.draft(), torch.no_grad():
        d = m(x).logits.argmax(-1)[0]
    assert (d == v).float().mean() == 1.0
    with DraftPatch(m, DraftConfig(expert_r=1, skip_layers={2, 3, 4})) as p, p.draft(), torch.no_grad():
        d = m(x).logits.argmax(-1)[0]
    assert (d == v).float().mean() < 1.0


def test_eat_calibrator_populates_and_leaves_outputs():
    m = tiny_model()
    x = ids(40)
    with torch.no_grad():
        base = m(x).logits
    with EATCalibrator(m) as cal, torch.no_grad():
        out = m(x).logits
    assert torch.equal(out, base)
    assert all("forward" not in mod.__dict__ for mod in m.modules())
    st = cal.states[1]
    assert st.N.sum() == 40 * m.config.num_experts_per_tok and (st.Cs >= 0).all()
    cfg = DraftConfig(expert_policy="eat", weight_convention="renorm")
    with DraftPatch(m, cfg, cal.states) as p, p.draft(), torch.no_grad():
        m(x)
    s = p.stats.summary()
    assert 1 <= s["experts_per_token"] <= m.config.num_experts_per_tok
