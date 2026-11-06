import random

from suffixspec.evalsim import PLAIN_MS, cost_ms, evaluate, run_one
from tage.baselines import PromptLookup, SuffixLast


def test_cost_points_and_extrapolation():
    assert cost_ms(1) == PLAIN_MS and abs(cost_ms(3) - 4.7) < 1e-9 and abs(cost_ms(9) - 7.9) < 1e-9
    assert cost_ms(2) > cost_ms(1) and cost_ms(17) > cost_ms(9)


def test_no_match_equals_plain():
    rng = random.Random(0)
    out = [rng.randrange(10_000, 20_000) for _ in range(200)]  # all distinct-ish tokens
    s = run_one(PromptLookup(3, 2), [1, 2, 3], out)
    assert s["tokens"] == 200 and s["drafted"] <= 2 and abs(s["ms"] - 200 * PLAIN_MS) / (200 * PLAIN_MS) < 0.05


def test_loop_gives_speedup_and_exact_count():
    loop = [5, 6, 7, 8, 9, 10, 11, 12]
    out = loop * 40
    s = run_one(PromptLookup(3, 2), loop, out, kmax=8)
    assert s["tokens"] == len(out)           # every token committed exactly once
    assert s["tokens"] * PLAIN_MS / s["ms"] > 1.5
    rows, agg = evaluate(lambda: SuffixLast(1), [dict(id="a", prompt_ids=loop, out_ids=out)], kmax=8)
    assert agg["speedup_pooled"] > 1.5 and agg["precision"] > 0.9


def test_min_draft_gates_cost():
    out = [1, 2, 1, 3, 1, 4, 1, 5] * 10
    lo = run_one(SuffixLast(1), [1, 2], out, kmax=8, min_draft=1)
    hi = run_one(SuffixLast(1), [1, 2], out, kmax=8, min_draft=4)
    assert hi["verified_steps"] <= lo["verified_steps"]


def test_cost_models_and_kcount():
    from suffixspec.evalsim import COST_MODELS, speedup_from
    loop = [5, 6, 7, 8, 9, 10, 11, 12]
    out = loop * 40
    s = run_one(PromptLookup(3, 2), loop, out, kmax=8)
    assert sum(s["kcount"].values()) == s["steps"]
    assert abs(speedup_from(s["kcount"], s["tokens"], "megakernel") - s["tokens"] * PLAIN_MS / s["ms"]) < 1e-9
    assert speedup_from(s["kcount"], s["tokens"], "flat2") > speedup_from(s["kcount"], s["tokens"], "flat5") > 1.0


def test_warm_state_helps_second_request():
    a = [3, 4, 5, 6, 7, 8] * 5
    cold = run_one(SuffixLast(1), [1], a, kmax=8, keep_state=False)
    p = SuffixLast(1)
    run_one(p, [1], a, kmax=8, keep_state=True)
    warm = run_one(p, [1], a, kmax=8, keep_state=True)
    assert warm["steps"] <= cold["steps"]
