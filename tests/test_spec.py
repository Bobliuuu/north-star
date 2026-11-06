import pytest
import torch

from fixtures import tiny_ref
from northstar.refmodel import FULL, DraftMode
from northstar.spec import SelfSpec, plain_greedy

CH = 8  # small ring chunk so sliding layers actually wrap (cap = window 8 + 8 = 16)


def test_layer_types_cover_both():
    m, _ = tiny_ref()
    assert "sliding_attention" in m.config.layer_types and "full_attention" in m.config.layer_types
    assert m.config.mlp_layer_types[0] == "dense"


@pytest.mark.parametrize("T", [1, 5, 30])
def test_ref_matches_hf_prefill_and_decode(T):
    m, ref = tiny_ref()
    ids = torch.randint(0, 97, (1, 40), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        want = m(ids).logits
    cache = ref.new_cache(1, 64, chunk=CH)
    ref.prefill(ids[:, :T], cache, FULL, CH, logits="none")
    got = []
    for s in range(T, 40):
        lg = ref.forward(ids[:, s : s + 1], cache, s, FULL)
        cache.len += 1
        got.append(lg[0, 0])
    got = torch.stack(got)
    assert torch.allclose(got, want[0, T:40], atol=1e-4, rtol=1e-4), (got - want[0, T:40]).abs().max()


def test_block_forward_and_rollback_match_hf():
    m, ref = tiny_ref()
    g = torch.Generator().manual_seed(2)
    ids = torch.randint(0, 97, (1, 40), generator=g)
    cache = ref.new_cache(1, 64, chunk=CH)
    ref.prefill(ids[:, :20], cache, FULL, CH, logits="none")
    junk = torch.randint(0, 97, (1, 6), generator=g)
    ref.forward(junk, cache, 20)  # speculative block that gets rolled back (cache.len stays 20)
    blk = ids[:, 20:26]
    lg = ref.forward(blk, cache, 20)  # overwrite with real tokens
    with torch.no_grad():
        want = m(ids[:, :26]).logits[0, 20:26]
    assert torch.allclose(lg[0], want, atol=1e-4, rtol=1e-4)
    cache.len = 26
    lg2 = ref.forward(ids[:, 26:27], cache, 26)
    with torch.no_grad():
        want2 = m(ids[:, :27]).logits[0, 26]
    assert torch.allclose(lg2[0, 0], want2, atol=1e-4, rtol=1e-4)


def test_moe_topr_is_prefix_of_native_topk_with_native_weights():
    m, ref = tiny_ref()
    mlp = m.model.layers[1].mlp
    x = torch.randn(5, 64)
    _, s, idx = mlp.gate(x)
    assert (idx[:, :-1] >= 0).all()
    # sorted descending => first r are the top-r; weights are sigmoid of the same logits (no renorm)
    assert (s[:, :-1] >= s[:, 1:]).all()
    logits = x @ mlp.gate.weight.T
    assert torch.allclose(s, torch.sigmoid(logits.gather(1, idx)))


MODES = [
    DraftMode(frozenset({2, 3})),
    DraftMode(frozenset(), 2),
    DraftMode(frozenset({3}), 1),
    DraftMode(frozenset({1, 2, 3, 4, 5, 6})),  # degenerate draft: nearly all layers skipped
]


@pytest.mark.parametrize("P", [3, 22])
@pytest.mark.parametrize("gamma", [1, 3, 5])
@pytest.mark.parametrize("mode", MODES, ids=lambda d: d.label())
def test_spec_identical_to_plain_greedy(P, gamma, mode):
    m, ref = tiny_ref(seed=3)
    prompt = torch.randint(0, 97, (P,), generator=torch.Generator().manual_seed(P)).tolist()
    base = plain_greedy(ref, prompt, 40, chunk=CH)["tokens"]
    res = SelfSpec(ref, mode, gamma, chunk=CH).generate(prompt, 40)
    assert res["tokens"] == base
    assert sum(r.committed for r in res["rounds"]) >= 40


def _forced(P, gamma, pattern):
    m, ref = tiny_ref(seed=5)
    prompt = torch.randint(0, 97, (P,), generator=torch.Generator().manual_seed(9)).tolist()
    N = 40
    truth = plain_greedy(ref, prompt, N + gamma + 4, chunk=CH)["tokens"]

    def override(pos, i, tok):
        k = pos - P
        if k >= len(truth):
            return tok
        if pattern == "all":
            return truth[k]
        j = {"first": 0, "middle": gamma // 2, "last": gamma - 1}[pattern]
        return truth[k] if i != j else (truth[k] + 1) % 97

    res = SelfSpec(ref, DraftMode(frozenset({2, 3})), gamma, chunk=CH, override=override).generate(prompt, N)
    assert res["tokens"] == truth[:N]
    return res


@pytest.mark.parametrize("P", [4, 21])
def test_all_accept(P):
    g = 4
    r = _forced(P, g, "all")
    assert all(s.accepted == g and s.committed == g + 1 for s in r["rounds"])


@pytest.mark.parametrize("pattern,exp", [("first", lambda g: 0), ("middle", lambda g: g // 2), ("last", lambda g: g - 1)])
@pytest.mark.parametrize("P", [4, 21])
def test_reject_positions(P, pattern, exp):
    g = 5
    r = _forced(P, g, pattern)
    assert all(s.accepted == exp(g) for s in r["rounds"]), [s.accepted for s in r["rounds"]]
    assert all(s.committed == exp(g) + 1 for s in r["rounds"])


def test_eos_truncation():
    m, ref = tiny_ref(seed=3)
    prompt = [1, 2, 3, 4]
    base = plain_greedy(ref, prompt, 30, chunk=CH)["tokens"]
    eos = base[7]
    b2 = plain_greedy(ref, prompt, 30, eos_id=eos, chunk=CH)["tokens"]
    s2 = SelfSpec(ref, MODES[0], 4, chunk=CH).generate(prompt, 30, eos_id=eos)["tokens"]
    assert b2 == s2 and b2[-1] == eos
