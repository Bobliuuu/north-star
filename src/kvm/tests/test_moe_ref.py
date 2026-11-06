import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from ref import moe_ref as m


def small(E=8, Tv=3, Td=4, seed=1):
    return m.make_inputs(E, Tv, Td, seed)


def test_grouped_equals_per_token():
    x, wg, wu, wd, idx, w = small()
    a = m.reference(x, wg, wu, wd, idx, w, 3, 3)
    b = m.reference_grouped(x, wg, wu, wd, idx, w, 3, 3)
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-4)


def test_r8_draft_equals_verify_semantics():
    x, wg, wu, wd, idx, w = small()
    full = m.reference(x, wg, wu, wd, idx, w, 7, 8)       # all tokens "verify"
    mixed = m.reference(x, wg, wu, wd, idx, w, 3, 8)      # draft with r=8 identical
    assert torch.allclose(full, mixed)


def test_topr_is_prefix_of_sorted_routing():
    _, _, _, _, idx, w = small()
    assert (w[:, :-1] >= w[:, 1:]).all()


def test_fused_bytes_never_exceed_two_pass():
    _, _, _, _, idx, w = small(E=16, Tv=4, Td=4)
    for r in range(1, 9):
        assert m.weight_bytes(idx, 4, r, 16, True) <= m.weight_bytes(idx, 4, r, 16, False)
    assert m.weight_bytes(idx, 4, 8, 16, True) < m.weight_bytes(idx, 4, 8, 16, False)  # overlap at E=16


def test_draft_truncation_changes_output_only_for_draft_rows():
    x, wg, wu, wd, idx, w = small()
    a = m.reference(x, wg, wu, wd, idx, w, 3, 8)
    b = m.reference(x, wg, wu, wd, idx, w, 3, 2)
    assert torch.equal(a[:3], b[:3]) and not torch.allclose(a[3:], b[3:])
