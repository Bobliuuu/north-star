import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "harness"))
import mk_block as mb
import pld


def test_mask_causal_and_window():
    m = mb.block_mask(5, 3)
    assert m.shape == (3, 8) and m[0].sum() == 6 and m[2].sum() == 8 and not m[0, 6]
    mw = mb.block_mask(5, 3, W=4)
    assert (mw.sum(1) == 4).all() and not mw[2, 3] and mw[2, 4]


def _run(W, p, T, a, tmax=8, d=8, seed=0):
    """block-verify T rows then roll back to p+a+1 and continue sequentially; must equal pure sequential decoding."""
    rng = np.random.default_rng(seed)
    n = p + 3 * T + 4
    Q, K, V = (rng.standard_normal((n, d)) for _ in range(3))
    ref = mb.LayerKV(d, W, tmax); seq = []
    for i in range(n):
        ref.write(i, K[i], V[i]); seq.append(mb.attend(Q[i], ref, i, W))
    kv = mb.LayerKV(d, W, tmax); out = []
    for i in range(p):
        kv.write(i, K[i], V[i]); out.append(mb.attend(Q[i], kv, i, W))
    kv.commit(p)
    o = mb.block_attend(Q[p:p + T], K[p:p + T], V[p:p + T], kv, p, W)
    assert np.allclose(o[: a + 1], np.stack(seq[p:p + a + 1]))  # accepted rows are exact
    # rejected rows get DIFFERENT tokens' K/V at the same positions after rollback
    kv.rollback(p + a + 1)
    for i in range(p + a + 1, n):
        kv.write(i, K[i], V[i]); out_i = mb.attend(Q[i], kv, i, W)
        assert np.allclose(out_i, seq[i]), (W, p, T, a, i)


def test_rollback_exact_global_and_ring():
    for W in (None, 6, 16):
        for p in (0, 3, 20, 41):
            for T in (1, 2, 5, 8):
                for a in range(T):
                    _run(W, p, T, a)


def test_ring_without_slack_breaks():
    # ring R=W (no slack) corrupts accepted rows of a block: proves the W+Tmax-1 requirement is real
    d, W, p, T = 4, 6, 20, 5
    rng = np.random.default_rng(1); n = p + T
    Q, K, V = (rng.standard_normal((n, d)) for _ in range(3))
    good = mb.LayerKV(d, W, tmax=T); bad = mb.LayerKV(d, W, tmax=1)
    for kv in (good, bad):
        for i in range(p):
            kv.write(i, K[i], V[i])
    og = mb.block_attend(Q[p:], K[p:], V[p:], good, p, W); ob = mb.block_attend(Q[p:], K[p:], V[p:], bad, p, W)
    assert not np.allclose(og, ob) and good.R == W + T - 1


def test_accept_matches_spec_loop():
    assert mb.accept([1, 2, 3], [1, 2, 9, 7]) == (2, [1, 2, 9])
    assert mb.accept([], [5]) == (0, [5])
    assert mb.accept([1, 2], [1, 2, 4]) == (2, [1, 2, 4])


def test_projection():
    cost = {T: mb.H11_VERIFY_MS.get(T, 6.68 + 0.23 * (T - 8)) for T in range(1, 30)}
    prompt = list(range(100, 160)); gen = list(range(100, 160))
    r = pld.simulate("sam", 7, prompt, gen, cost)
    assert r["speedup"] > 2.0  # fully copyable text on a 3.42 ms verify curve
    assert abs(mb.project([(0, 0)] * 10) - 1.0) < 1e-9
    assert mb.project([(7, 7)]) > 3.0 and mb.project([(7, 0)]) < 0.6


def test_adopt_blocks_keeps_history_and_adopts_new():
    V = np.array([[5, 6, -1, -1], [-1, 7, -1, -1]])      # layer1 block0 evicted (sliding)
    new = np.array([[9, 9, 11, 12], [9, 9, 11, -1]])      # last row's own pages
    out = mb.adopt_blocks(V, new, 2, 3)
    assert out.tolist() == [[5, 6, 11, 12], [-1, 7, 11, -1]]  # history kept, evicted not resurrected, new adopted
    assert mb.adopt_blocks(out, np.where(new >= 0, new + 1, new), 2, 3).tolist() == out.tolist()  # idempotent once adopted
