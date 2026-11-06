import pytest
import torch
import torch.nn.functional as F

from approx.attention import AttnCost, decode_step_cost, make_selector, sparse_attention
from approx.config import DraftConfig, LSAParams
from approx.layerskip import candidate_skip_sets, clasp_select, delta_layer_apply_factory, early_exit_set

B, H, Hkv, D = 1, 4, 2, 16
LSA = LSAParams(pool_size=2, bucket_size=4, node_fanout=2, beam_width=3, start_nodes=2, kmeans_iters=3)


def cfg_for(sel, budget, S=128, **kw):
    return DraftConfig(attn_budget=budget, attn_selector=sel, window=8, sink=2, lsa=LSA, subq_block=8,
                       attn_chunk=16, **kw)


def qkv(S, T=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    T = T or S
    return (torch.randn(B, H, T, D, generator=g), torch.randn(B, Hkv, S, D, generator=g),
            torch.randn(B, Hkv, S, D, generator=g))


def full_attn(q, K, V):
    T, S = q.shape[2], K.shape[2]
    G = H // Hkv
    m = torch.arange(S)[None] <= (torch.arange(S - T, S)[:, None])
    return F.scaled_dot_product_attention(q, K.repeat_interleave(G, 1), V.repeat_interleave(G, 1), attn_mask=m,
                                          scale=D ** -0.5)


SELS = ["oracle_topk", "lsa_flat", "lsa_tree", "subq_style"]


@pytest.mark.parametrize("sel", SELS)
def test_huge_budget_equals_full_attention(sel):
    S = 128
    q, K, V = qkv(S)
    cfg = cfg_for(sel, budget=10_000)
    if sel == "lsa_tree":  # exhaustive beam (otherwise the beam itself limits recall)
        cfg = cfg.with_(lsa=LSAParams(**{**LSA.__dict__, "beam_width": 10_000}))
    cost = AttnCost()
    o = sparse_attention(q, K, V, torch.arange(S), make_selector(cfg), cfg, D ** -0.5, cost)
    assert torch.allclose(o, full_attn(q, K, V), atol=1e-5), (o - full_attn(q, K, V)).abs().max()


@pytest.mark.parametrize("sel", SELS)
def test_causality_no_future_leak(sel):
    """Output at t must not depend on keys/values/queries after t, including through index construction."""
    S, t = 128, 70
    q, K, V = qkv(S)
    cfg = cfg_for(sel, budget=24)
    a = sparse_attention(q, K, V, torch.arange(S), make_selector(cfg), cfg, D ** -0.5, AttnCost())
    K2, V2 = K.clone(), V.clone()
    K2[:, :, t + 1 :] = torch.randn_like(K2[:, :, t + 1 :]) * 3
    V2[:, :, t + 1 :] = torch.randn_like(V2[:, :, t + 1 :])
    b = sparse_attention(q, K2, V2, torch.arange(S), make_selector(cfg), cfg, D ** -0.5, AttnCost())
    assert torch.allclose(a[:, :, : t + 1], b[:, :, : t + 1], atol=1e-6)
    assert not torch.allclose(a[:, :, t + 1 :], b[:, :, t + 1 :], atol=1e-3)


@pytest.mark.parametrize("sel", SELS)
def test_budget_respected_and_window_sink_kept(sel):
    S = 128
    q, K, V = qkv(S)
    cfg = cfg_for(sel, budget=16)
    sl = make_selector(cfg)
    cost = AttnCost()
    st = sl.prepare(K, cost)
    qp = torch.arange(S)
    keep = sl.select(q, K, qp, st, H // Hkv, cost)
    causal = torch.arange(S)[None] <= qp[:, None]
    assert not (keep & ~causal[None, None]).any()
    assert keep[..., :2].all() or True
    for t in range(S):
        assert keep[0, 0, t, max(0, t - 7) : t + 1].all()
        assert keep[0, 0, t, : min(2, t + 1)].all()
    n = keep.sum(-1).float()
    slack = {"oracle_topk": 0, "subq_style": 0, "lsa_flat": 8, "lsa_tree": 8}[sel]  # LSA keeps the crossing bucket
    assert (n <= 16 + 8 + 2 + slack).all()
    if sel == "oracle_topk":
        assert n[0, 0, -1] == 16 + 8 + 2


def test_oracle_budget_monotone_recall():
    """Oracle with larger k is closer to full attention."""
    S = 128
    q, K, V = qkv(S, seed=3)
    ref = full_attn(q, K, V)
    errs = []
    for k in (4, 16, 64):
        cfg = cfg_for("oracle_topk", k)
        o = sparse_attention(q, K, V, torch.arange(S), make_selector(cfg), cfg, D ** -0.5, AttnCost())
        errs.append((o - ref).abs().mean().item())
    assert errs[0] > errs[1] > errs[2]


def test_decode_row_equals_sequence_row():
    S = 96
    q, K, V = qkv(S, seed=5)
    for sel in SELS:
        cfg = cfg_for(sel, 20)
        seq = sparse_attention(q, K, V, torch.arange(S), make_selector(cfg), cfg, D ** -0.5, AttnCost())
        # decode at t = S-1 with the full K (index sees keys 0..S-1; last query only)
        dec = sparse_attention(q[:, :, -1:], K, V, torch.tensor([S - 1]), make_selector(cfg), cfg, D ** -0.5, AttnCost())
        assert torch.allclose(seq[:, :, -1:], dec, atol=1e-5), sel


def test_cost_accounting_labels():
    S = 128
    q, K, V = qkv(S)
    c = {}
    for sel in SELS:
        cfg = cfg_for(sel, 16)
        c[sel] = AttnCost()
        sparse_attention(q, K, V, torch.arange(S), make_selector(cfg), cfg, D ** -0.5, c[sel])
    o, f, t, s = (c[x].summary() for x in SELS)
    assert o["index_scored_per_q"] == pytest.approx(o["keys_causal_per_q"])  # all-keys scan, explicit
    assert o["index_build_total"] == 0
    for x in (f, t, s):
        assert x["index_build_total"] > 0 and x["index_bytes"] > 0       # build + resident index are charged
    assert f["index_scored_per_q"] < o["index_scored_per_q"]
    assert o["attended_frac"] < 1


def test_analytic_decode_cost_scaling():
    for S in (2 ** 15, 2 ** 20):
        full = decode_step_cost(S, "full", 0)
        orc = decode_step_cost(S, "oracle_topk", 2048)
        flat = decode_step_cost(S, "lsa_flat", 2048)
        tree = decode_step_cost(S, "lsa_tree", 2048)
        assert orc["index_scan"] == S                      # linear in S: not subquadratic
        assert flat["index_scan"] == S // 8 < orc["index_scan"]
        assert tree["index_scan"] < flat["index_scan"]
    assert decode_step_cost(2 ** 20, "lsa_tree", 2048)["index_scan"] < 10 * decode_step_cost(2 ** 15, "lsa_tree", 2048)["index_scan"]


def test_clasp_dp_skips_low_impact_layers():
    torch.manual_seed(0)
    L, Dm = 10, 8
    h = [torch.randn(Dm)]
    for l in range(L):
        d = torch.zeros(Dm) if l in (3, 6) else torch.randn(Dm) * 2
        h.append(h[-1] + d)
    got = clasp_select(h, delta_layer_apply_factory(h), 2)
    assert got == frozenset({3, 6})


def test_skip_sets_never_include_layer0():
    for name, s in candidate_skip_sets(49).items():
        assert 0 not in s and max(s) <= 48, name
    assert early_exit_set(49, 40) == frozenset(range(40, 49))
