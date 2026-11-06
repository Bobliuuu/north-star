import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import pytest
from async_spec.model import (batch_factor, length_dist, outcome_probs, hit_prob,
                              round_times, speedup, upper_bound)

APOS = [0.83, 0.73, 0.64, 0.57]


def test_batch_factor_measured_points():
    assert batch_factor(1) == 1.0
    assert batch_factor(8) == pytest.approx(7.39 / 3.32)
    assert batch_factor(2) == pytest.approx(4.10 / 3.32)
    assert batch_factor(16) > batch_factor(8)


def test_length_dist_sums_to_one_and_mean():
    p = length_dist(APOS)
    assert sum(p) == pytest.approx(1.0)
    assert sum(j * x for j, x in enumerate(p)) == pytest.approx(sum(APOS))


def test_outcome_probs_valid_and_hit_monotone():
    o = outcome_probs(APOS)
    assert all(p >= 0 for p, _ in o) and sum(p for p, _ in o) <= 1.0
    hs = [hit_prob(APOS, k) for k in range(0, 10)]
    assert hs[0] == 0 and all(b >= a for a, b in zip(hs, hs[1:])) and hs[-1] <= 1


def test_k0_equals_serial():
    Ts, Ta = round_times(4, 10.0, 14.0, 0, 0.0, 0.5, 0.5)
    assert Ts == Ta == 54.0


def test_zero_hit_is_never_better():
    Ts, Ta = round_times(4, 10.0, 14.0, 2, 0.0, 0.8, 0.8)
    assert Ta > Ts  # wasted contention on verify


def test_full_bandwidth_sharing_no_gain_even_with_perfect_hit():
    # beta=1: concurrency just time-slices HBM; perfect hit rate cannot beat serial
    Ts, Ta = round_times(4, 10.0, 14.0, 1, 1.0, 1.0, 1.0, fk=1.0)
    assert Ta >= Ts - 1e-9


def test_free_overlap_bound():
    # beta->0: verify unaffected, draft fully hidden when it fits in the verify window
    Ts, Ta = round_times(2, 2.0, 10.0, 1, 1.0, 0.0, 0.0, fk=1.0)
    assert Ta == pytest.approx(10.0)
    assert Ts / Ta == pytest.approx(upper_bound(2, 2.0, 10.0))


def test_more_branches_hurt_at_high_beta():
    s1 = speedup(4, 10.0, 14.0, 1, APOS, 0.8)[0]
    s16 = speedup(4, 10.0, 14.0, 16, APOS, 0.8)[0]
    assert s16 < s1 < 1.0 + 1e-9 or s16 < s1
