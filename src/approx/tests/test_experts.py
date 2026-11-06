import pytest
import torch

from approx.config import EATParams
from approx.experts import EATState, apply_weight_convention, eat_select, slot_output_norms, topr
from fixtures import tiny_model


def native(N=5, E=16, k=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(N, E, generator=g) * 2
    v, i = logits.topk(k, -1)
    return logits, torch.sigmoid(v), i


def test_conventions_numeric():
    w = torch.tensor([[0.8, 0.6, 0.4, 0.2]])
    assert torch.allclose(topr(w, None if False else torch.arange(4)[None], 2, "raw_sigmoid")[0], w[:, :2])
    r = topr(w, torch.arange(4)[None], 2, "renorm")[0]
    assert torch.allclose(r, torch.tensor([[0.8 / 1.4, 0.6 / 1.4]]))
    mp = topr(w, torch.arange(4)[None], 2, "mass_preserve")[0]
    assert torch.allclose(mp.sum(), w.sum()) and torch.allclose(mp[0, 0] / mp[0, 1], torch.tensor(0.8 / 0.6))


def test_topr_is_prefix_of_native_sorted():
    logits, w, i = native()
    w2, i2 = topr(w, i, 2, "raw_sigmoid")
    assert torch.equal(i2, i[:, :2]) and torch.equal(w2, w[:, :2])
    assert (w[:, :-1] >= w[:, 1:]).all()  # native order is descending


def test_ema_closed_form_matches_sequential():
    torch.manual_seed(0)
    st = EATState(6)
    idx = torch.randint(0, 6, (20, 3))
    # make slots unique per row
    idx = torch.stack([torch.randperm(6)[:3] for _ in range(20)])
    w, on = torch.rand(20, 3), torch.rand(20, 3) * 3
    a = 0.7
    st.update(idx[:10], w[:10], on[:10], a)
    st.update(idx[10:], w[10:], on[10:], a)
    ref_c, ref_n, ref_p = [None] * 6, torch.zeros(6), torch.zeros(6)
    for t in range(20):
        for s in range(3):
            e = idx[t, s].item()
            ref_c[e] = on[t, s].item() if ref_c[e] is None else (1 - a) * ref_c[e] + a * on[t, s].item()
            ref_n[e] += 1
            ref_p[e] += w[t, s]
    for e in range(6):
        assert abs(st.Cs[e].item() - (ref_c[e] or 0)) < 1e-5
    assert torch.allclose(st.N.float(), ref_n) and torch.allclose(st.Ptot.float(), ref_p, atol=1e-5)


def test_history_score_formula():
    st = EATState(3)
    st.N = torch.tensor([2.0, 6.0, 0.0], dtype=torch.float64)
    st.Ptot = torch.tensor([1.0, 3.0, 0.0], dtype=torch.float64)
    st.Cs = torch.tensor([2.0, 1.0, 0.0], dtype=torch.float64)
    p = EATParams()
    s = st.history_score(p)
    want = torch.tensor([0.3 * 2 / 8 + 0.3 * 0.5 + 0.4 * 2, 0.3 * 6 / 8 + 0.3 * 0.5 + 0.4 * 1, 0.0])
    assert torch.allclose(s, want, atol=1e-6)


def test_eat_count_bounds_and_renorm():
    logits, w, i = native(N=64, E=32, k=8)
    st = EATState(32)
    for scale in (0.0, 1.0, 3.0, 100.0):
        p = EATParams(tau_scale=scale, tau_bounds=None)
        ww, ii, act = eat_select(logits, w, i, st, p, "renorm")
        n = act.sum(-1)
        assert (n >= 1).all() and (n <= 8).all()
        assert torch.allclose(ww.sum(-1), torch.ones(64), atol=1e-5)
        assert (ww[~act] == 0).all()
        assert set(ii[act].unique().tolist()) <= set(range(32))
    p = EATParams(tau_scale=100.0, tau_bounds=None)  # huge tau -> argmax-only fallback
    _, ii, act = eat_select(logits, w, i, st, p, "renorm")
    assert (act.sum(-1) == 1).all()
    # fallback expert is the argmax of I = argmax of g (no history) among native
    assert torch.equal(ii[:, 0], i[:, 0])
    p = EATParams(tau_scale=0.0, tau_bounds=None)
    _, _, act = eat_select(logits, w, i, st, p, "renorm")
    assert (act.sum(-1) == 8).all()


def test_eat_native_scope_subset_of_native():
    logits, w, i = native(N=40, E=32, k=8, seed=3)
    st = EATState(32)
    ww, ii, act = eat_select(logits, w, i, st, EATParams(), "renorm")
    for n in range(40):
        assert set(ii[n][act[n]].tolist()) <= set(i[n].tolist())


def test_eat_scope_all_can_leave_native_set():
    logits, w, i = native(N=40, E=32, k=4, seed=3)
    st = EATState(32)
    st.N[:] = 1; st.Ptot[:] = 1; st.Cs[:] = 0
    st.Cs[7] = 1000.0  # history strongly favours expert 7
    p = EATParams(candidate_scope="all", tau_bounds=None, beta=0.9)
    ww, ii, act = eat_select(logits, w, i, st, p, "renorm")
    assert (ii == 7).any()


def test_slot_output_norms_match_experts():
    m = tiny_model()
    mlp = m.model.layers[1].mlp
    h = torch.randn(9, 64)
    with torch.no_grad():
        _, w, idx = mlp.gate(h)
        norms = slot_output_norms(mlp.experts, h, idx)
        for s in range(idx.shape[1]):
            out = mlp.experts(h, idx[:, s : s + 1], torch.ones(9, 1))
            assert torch.allclose(out.norm(dim=-1), norms[:, s], atol=1e-5)
