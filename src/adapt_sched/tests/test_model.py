import pathlib, sys, re, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from dataclasses import replace
from adapt_sched.model import (Cfg, simulate, step_ms, MK_BS_MS, memory_headroom_gb,
                               breakeven_horizon_s)

ROOT = pathlib.Path(__file__).resolve().parents[3]
FAST = Cfg(horizon_s=300.0, n_cand=2)


def test_step_ms_matches_measured_and_monotone():
    for b, v in MK_BS_MS.items():
        assert step_ms(b) == v
    assert step_ms(3) > step_ms(2) and step_ms(12) > step_ms(8)


def test_measured_constants_match_logs():
    txt = "".join(open(f).read() for f in ROOT.glob("results/megakernel_H100_bs_*.log"))
    vals = [float(x) for x in re.findall(r"mean/step\s+([0-9.]+) ms", txt)]
    for v in MK_BS_MS.values():
        assert any(abs(x - v) < 0.01 for x in vals)


def test_baseline_no_bg():
    r = simulate(replace(FAST, policy="none"))
    assert r["cands_done"] == 0 and r["p99_step_ms"] <= step_ms(8) + 1e-9
    assert r["slo_violation_frac"] == 0


def test_static_full_intensity_hurts_p99_more_than_low():
    hi = simulate(replace(FAST, policy="static", u_static=1.0))
    lo = simulate(replace(FAST, policy="static", u_static=0.1))
    assert hi["p99_step_ms"] >= lo["p99_step_ms"]


def test_sequential_pause_is_latency_spike_and_finishes():
    r = simulate(replace(FAST, policy="sequential", horizon_s=900.0))
    assert r["cands_done"] == 2
    assert r["pause_s"] == 2 * FAST.eval_s
    assert r["req_p99_s"] > 60  # requests queued behind a pause


def test_adaptive_respects_slo_better_than_static_full():
    c = replace(FAST, horizon_s=600.0, n_cand=3)
    a = simulate(replace(c, policy="adaptive"))
    s = simulate(replace(c, policy="static", u_static=1.0))
    assert a["slo_violation_frac"] <= s["slo_violation_frac"]


def test_no_slowdown_when_demands_fit():
    c = replace(FAST, beta_s=0.4, beta_b=0.4, c_sm=0.0, policy="static", u_static=1.0)
    assert simulate(c)["p99_step_ms"] <= step_ms(8) + 1e-9


def test_equal_work_charged_at_least_nominal():
    c = replace(FAST, horizon_s=1200.0, policy="static", u_static=0.5)
    r = simulate(c)
    assert r["cands_done"] == 2
    assert r["bg_gpu_s_consumed"] >= 2 * c.eval_s - 1e-6   # noise reps only add cost


def test_deterministic():
    c = replace(FAST, policy="static")
    assert simulate(c) == simulate(c)


def test_memory_headroom():
    m = memory_headroom_gb(4, 32768)
    assert m["benchmark_headroom_gb"] < 13 and m["kv_fits_in_claim"]
    assert not memory_headroom_gb(8, 131072)["kv_fits_in_claim"]


def test_breakeven():
    assert breakeven_horizon_s(100, 0.5, 0.1) == 2000
    assert breakeven_horizon_s(100, 0.5, 0) == math.inf
