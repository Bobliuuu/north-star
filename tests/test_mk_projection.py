import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "harness"))
import mk_projection as mp
import mk_block as mb

C = {1: 3.0, 2: 4.0, 3: 5.0, 4: 6.0, 5: 7.0, 6: 8.0, 7: 9.0, 8: 10.0}


def test_fill_interp_and_extrapolate():
    f = mp.fill({1: 1.0, 3: 3.0, 5: 5.0}, 7)
    assert f[2] == 2.0 and f[4] == 4.0 and f[7] == 7.0


def test_tokps_matches_hand_arithmetic():
    rounds = [(3, 3), (0, 0)]  # 4 + 1 tokens for 6.0 + 3.0 ms
    assert abs(mp.tokps(rounds, C) - 5 / 9.0 * 1000) < 1e-9
    assert abs(mp.tokps([(0, 0)] * 4, C) - 1000 / 3.0) < 1e-9


def test_solve_a_roundtrip_and_extremes():
    for a in (0.2, 0.6, 0.9):
        assert abs(mp.solve_a(mp.e_acc(a, 5), 5) - a) < 1e-6
    assert mp.solve_a(0, 5) == 0.0 and mp.solve_a(5, 5) == 1.0


def test_agg_rounds_conserves_plain_and_caps_drafts():
    r = mp.agg_rounds(100, 20, 120, 40, 12)  # dbar 6, abar 2
    assert len(r) == 100 and sum(1 for d, _ in r if d == 0) == 80
    assert all(d <= 6 for d, _ in r)
    rc = mp.agg_rounds(100, 20, 120, 40, 3)
    assert all(d <= 3 for d, _ in rc) and sum(a for _, a in rc) < sum(a for _, a in r)


def test_breakeven_tokens_per_round_and_monotone():
    be = mp.breakeven(C, mp.fill(C, 13))
    assert abs(be["tokens_per_round_to_tie_own_plain"][4] - 2.0) < 1e-9
    assert be["a_vs_mk_plain"][2] < be["a_vs_mk_plain"][8]
    # same curve: tie at a=0 (both plain)
    assert be["a_intervals_mk_best_T_le8_ge_fast_pld_best_T_le13"][0][0] == 0.0


def test_s3_curve_loads():
    c = mp.mk_curves()
    assert sorted(c) == [2048, 8192] and sorted(c[2048]) == list(range(1, 9))
