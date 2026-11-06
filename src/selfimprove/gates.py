"""Three independent gates. Thresholds come from EvalConfig (evaluator-owned, outside editable scope) and are logged."""
from __future__ import annotations
import math, random, statistics
from dataclasses import dataclass, asdict, field


@dataclass
class GateResult:
    name: str; passed: bool; reason: str; data: dict = field(default_factory=dict)
    def to_dict(self): return asdict(self)


# ---- (c1) numerical correctness vs reference -------------------------------------------------------------------------
@dataclass
class CorrectnessSpec:
    rel_l2_tol: float = 1e-3        # stated tolerance; bf16-in/fp32-acc kernels measured <=2e-4 vs torch ref (DESIGN.md)
    max_abs_tol: float | None = None
    min_cases: int = 8              # a gate that checks fewer cases than this is invalid (anti "--empty-ok")
    max_regress_vs_base: float = 4.0  # candidate error may not exceed this multiple of baseline error (floored at 1e-6)


def correctness_gate(cand: dict | None, base: dict | None, spec: CorrectnessSpec) -> GateResult:
    """cand/base: harness JSON {"cases":[{"name","rel_l2","max_abs","finite"}...]}"""
    if not cand or not isinstance(cand.get("cases"), list):
        return GateResult("correctness", False, "harness produced no parsable result")
    cases = cand["cases"]
    if len(cases) < spec.min_cases:
        return GateResult("correctness", False, f"only {len(cases)} cases < min_cases {spec.min_cases}")
    names = [c.get("name") for c in cases]
    if len(set(names)) != len(names):
        return GateResult("correctness", False, "duplicate case names")
    base_by = {c.get("name"): c for c in (base or {}).get("cases", [])}
    worst = 0.0
    for c in cases:
        try:
            r = float(c["rel_l2"]); a = float(c.get("max_abs", 0.0))
        except (KeyError, TypeError, ValueError):
            return GateResult("correctness", False, f"malformed case {c!r}"[:200])
        if not c.get("finite", True) or not (math.isfinite(r) and math.isfinite(a)):
            return GateResult("correctness", False, f"non-finite output in {c.get('name')}")
        if r > spec.rel_l2_tol:
            return GateResult("correctness", False, f"{c.get('name')}: rel_l2 {r:.3g} > tol {spec.rel_l2_tol:g}", {"worst": r})
        if spec.max_abs_tol is not None and a > spec.max_abs_tol:
            return GateResult("correctness", False, f"{c.get('name')}: max_abs {a:.3g} > {spec.max_abs_tol:g}")
        b = base_by.get(c.get("name"))
        if b is not None and r > spec.max_regress_vs_base * max(float(b["rel_l2"]), 1e-6):
            return GateResult("correctness", False, f"{c.get('name')}: error {r:.3g} regressed >{spec.max_regress_vs_base}x vs baseline {b['rel_l2']:.3g}")
        worst = max(worst, r)
    return GateResult("correctness", True, "ok", {"n_cases": len(cases), "worst_rel_l2": worst, "tol": spec.rel_l2_tol})


# ---- (c2) held-out task quality (non-inferiority) --------------------------------------------------------------------
@dataclass
class QualitySpec:
    margin: float = 0.0             # allowed drop in mean score (paired lower CI bound must be >= -margin)
    min_items: int = 20
    boot: int = 2000
    alpha: float = 0.05
    seed: int = 12345


def quality_gate(cand: dict | None, base: dict | None, spec: QualitySpec) -> GateResult:
    """harness JSON {"items":[score per held-out item, same order]}; paired non-inferiority via bootstrap."""
    try:
        c, b = [float(x) for x in cand["items"]], [float(x) for x in base["items"]]
    except Exception:
        return GateResult("quality", False, "harness produced no parsable item scores")
    if len(c) != len(b) or len(c) < spec.min_items:
        return GateResult("quality", False, f"need >= {spec.min_items} paired items (got {len(c)}/{len(b)})")
    if not all(math.isfinite(x) for x in c + b):
        return GateResult("quality", False, "non-finite scores")
    d = [x - y for x, y in zip(c, b)]
    rng = random.Random(spec.seed)
    n = len(d)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(spec.boot))
    lo = means[int(spec.alpha * spec.boot)]
    ok = lo >= -spec.margin
    return GateResult("quality", ok, "ok" if ok else f"paired CI lower bound {lo:.4f} < -{spec.margin}",
                      {"mean_delta": sum(d) / n, "ci_lo": lo, "n": n, "cand_mean": sum(c) / n, "base_mean": sum(b) / n})


# ---- (c3) repeatable speed ------------------------------------------------------------------------------------------
@dataclass
class SpeedSpec:
    n_runs: int = 9                 # interleaved A/B runs per split
    min_gain: float = 0.03          # promote only if speedup lower CI bound > 1 + max(min_gain, noise_floor)
    noise_mult: float = 2.0
    boot: int = 4000
    alpha: float = 0.05
    seed: int = 777
    require_heldout_split: bool = True


@dataclass
class TimerSpec:
    """Independent cross-check of the candidate-reported time (closes review risk 1). The BENCH HARNESS (protected, not the candidate)
    wall-clocks the child with a FIXED iteration count it chose, and derives ref_ms, a per-iteration time comparable to the
    candidate-reported time_ms (marginal wall of N iterations minus wall of 1, divided by N-1; the candidate cannot change N).
    The evaluator then (a) bounds the harness's claimed wall by ITS OWN clock around the whole child and (b) rejects when
    time_ms and ref_ms disagree by more than ratio_tol in either direction. ratio_tol is deliberately loose by default because
    ref_ms includes un-modelled work (cache flush, launch); calibrate it on the first live run, never widen it silently."""
    ratio_tol: float = 4.0
    wall_slack_ms: float = 250.0     # scheduling jitter between the harness's clock and the evaluator's clock
    wall_slack_rel: float = 0.05
    # calibrate=True (kernel track): the ABSOLUTE ratio t/ref is a property of the bench harness (ref_ms includes un-modelled work: fused AND
    # two-pass launches, L2 flush, process start), not of the kernel. MEASURED live: ~9.4x for the unmodified BASELINE, so the absolute 4x cap
    # rejected the baseline itself (run 4, rounds 0 and 2: gate_fail:timer on tag b). With calibrate, every run's rho = t/ref must match the
    # BASELINE's median rho within ratio_tol (relative, and tighter than before): a candidate that lies about its kernel time shifts rho and is
    # rejected; a genuinely faster/slower kernel moves rho only by its share of ref.
    calibrate: bool = False


def timer_crosscheck(j: dict | None, eval_wall_s: float, spec: TimerSpec) -> GateResult:
    if not isinstance(j, dict): return GateResult("timer", False, "no bench result")
    try:
        t, ref, wall, n = float(j["time_ms"]), float(j["ref_ms"]), float(j["harness_wall_ms"]), int(j["n_iters"])
    except (KeyError, TypeError, ValueError):
        return GateResult("timer", False, "bench result lacks harness-measured ref_ms/harness_wall_ms/n_iters")
    if not all(math.isfinite(x) and x > 0 for x in (t, ref, wall)) or n < 2:
        return GateResult("timer", False, "non-positive / non-finite timer fields")
    if wall > eval_wall_s * 1000.0 * (1 + spec.wall_slack_rel) + spec.wall_slack_ms:
        return GateResult("timer", False, f"harness wall {wall:.0f}ms exceeds evaluator-measured {eval_wall_s*1000:.0f}ms")
    r = max(t / ref, ref / t)
    if spec.calibrate: return GateResult("timer", True, "ok (relative check pending)", {"ratio": r, "rho": t / ref})
    if r > spec.ratio_tol:
        return GateResult("timer", False, f"self-reported {t:.4g}ms vs harness-measured {ref:.4g}ms disagree x{r:.2f} > {spec.ratio_tol}", {"ratio": r})
    return GateResult("timer", True, "ok", {"ratio": r})


def timer_calibrated(runs: list, spec: TimerSpec) -> GateResult:
    """runs: [(tag 'b'|'k', bench json)] of ONE split. Every run's rho = time_ms/ref_ms must lie within [rho_b/ratio_tol, rho_b*ratio_tol],
    rho_b = median rho of the baseline runs (the baseline is ground truth for the harness's own overhead)."""
    rb = [float(j["time_ms"]) / float(j["ref_ms"]) for t, j in runs if t == "b"]
    if not rb: return GateResult("timer", False, "no baseline runs to calibrate the timer cross-check")
    rho_b = statistics.median(rb)
    worst = 1.0
    # ref_ms is a difference of two process wall clocks and is noisy per run (A/A on A100: single-run rho spans 0.7x..1.6x of its median),
    # so: (a) the MEDIAN rho of each tag must be within ratio_tol of the baseline median (a candidate that mis-reports its time shifts every
    # run, hence the median); (b) every single run must be within 4*ratio_tol (catches a lone wild value without false-failing jitter).
    for tag in sorted({t for t, _ in runs}):
        med = statistics.median(float(j["time_ms"]) / float(j["ref_ms"]) for t, j in runs if t == tag)
        r = max(med / rho_b, rho_b / med); worst = max(worst, r)
        if r > spec.ratio_tol:
            return GateResult("timer", False, f"{tag}: median self-reported/harness-measured ratio {med:.3g} differs x{r:.2f} from the baseline's {rho_b:.3g} (> {spec.ratio_tol})",
                              {"rho_baseline": rho_b, "rho_median": med, "ratio": r})
    for t, j in runs:
        rho = float(j["time_ms"]) / float(j["ref_ms"]); r = max(rho / rho_b, rho_b / rho)
        if r > 4 * spec.ratio_tol:
            return GateResult("timer", False, f"{t}: single-run ratio {rho:.3g} differs x{r:.2f} from the baseline's {rho_b:.3g} (> {4 * spec.ratio_tol})", {"rho_baseline": rho_b, "rho": rho})
    return GateResult("timer", True, "ok", {"rho_baseline": rho_b, "worst_median_ratio": worst})


def _median(x): return statistics.median(x)


def _mad_rel(x):
    m = _median(x)
    return 1.4826 * _median([abs(v - m) for v in x]) / m if m else float("inf")


def speed_gate_split(base_ms: list[float], cand_ms: list[float], spec: SpeedSpec) -> dict:
    """Ratio of medians with bootstrap CI. Noise floor = noise_mult * robust relative spread of the runs."""
    if len(base_ms) != len(cand_ms) or len(base_ms) < 5:
        return {"valid": False, "reason": "need >=5 paired runs"}
    if not all(math.isfinite(x) and x > 0 for x in base_ms + cand_ms):
        return {"valid": False, "reason": "non-positive / non-finite timing"}
    rng = random.Random(spec.seed); n = len(base_ms)
    rs = []
    for _ in range(spec.boot):
        b = [base_ms[rng.randrange(n)] for _ in range(n)]
        c = [cand_ms[rng.randrange(n)] for _ in range(n)]
        rs.append(_median(b) / _median(c))
    rs.sort()
    lo, hi = rs[int(spec.alpha / 2 * spec.boot)], rs[int((1 - spec.alpha / 2) * spec.boot) - 1]
    noise = spec.noise_mult * max(_mad_rel(base_ms), _mad_rel(cand_ms))
    return {"valid": True, "speedup": _median(base_ms) / _median(cand_ms), "ci_lo": lo, "ci_hi": hi, "noise_floor": noise,
            "threshold": 1 + max(spec.min_gain, noise), "n": n, "base_med": _median(base_ms), "cand_med": _median(cand_ms)}


def speed_gate(splits: dict[str, tuple[list[float], list[float]]], spec: SpeedSpec) -> GateResult:
    """splits: {'dev': (base_ms, cand_ms), 'heldout': (...)}. Candidate must clear the threshold on dev AND must not be
    slower on held-out shapes; shape-specialized wins show up as dev-only gains."""
    res = {k: speed_gate_split(b, c, spec) for k, (b, c) in splits.items()}
    if any(not r["valid"] for r in res.values()):
        return GateResult("speed", False, "invalid timing data: " + "; ".join(f"{k}:{r.get('reason')}" for k, r in res.items() if not r["valid"]), res)
    if "dev" not in res:
        return GateResult("speed", False, "no dev split timed", res)
    if spec.require_heldout_split and "heldout" not in res:
        return GateResult("speed", False, "no held-out split timed", res)
    d = res["dev"]
    if d["ci_lo"] <= d["threshold"]:
        return GateResult("speed", False, f"dev speedup CI lo {d['ci_lo']:.3f} <= threshold {d['threshold']:.3f} (noise floor {d['noise_floor']:.3f})", res)
    h = res.get("heldout")
    if h and h["ci_lo"] < 1 - h["noise_floor"] - 1e-9:
        return GateResult("speed", False, f"held-out slowdown: CI lo {h['ci_lo']:.3f}", res)
    if h and h["speedup"] < 1 + 0.25 * (d["speedup"] - 1):
        return GateResult("speed", False, f"gain does not transfer to held-out shapes (dev {d['speedup']:.3f}, held-out {h['speedup']:.3f})", res)
    return GateResult("speed", True, f"dev speedup {d['speedup']:.3f} (CI lo {d['ci_lo']:.3f})", res)


# ---- cumulative gain vs the original baseline (selection effect, review risk 6) ---------------------------------------
def _z(alpha: float) -> float:
    return statistics.NormalDist().inv_cdf(1 - alpha / 2)


def cumulative_gain(steps: list[dict], alpha: float = 0.05, n_tested: int = 1) -> dict:
    """steps: per-promotion speed-split dicts {'speedup','ci_lo','ci_hi'}, each measured against ITS then-current baseline.
    Speedups multiply; each step's log-SE is recovered from its 95% bootstrap CI and, treating steps as independent, variances add.
    ci_* uses the nominal alpha; sel_adj_* uses alpha/n_tested (Bonferroni-style widening over every proposal that reached the speed
    gate: an upper bound on how many chances noise had to win). n_tested counts CANDIDATES that reached speed, not promotions."""
    n_tested = max(1, n_tested)
    if not steps:
        return dict(n_steps=0, speedup=1.0, ci_lo=1.0, ci_hi=1.0, sel_adj_ci_lo=1.0, sel_adj_ci_hi=1.0, n_tested=n_tested, alpha=alpha)
    ls = var = 0.0
    for s in steps:
        ls += math.log(s["speedup"])
        se = (math.log(s["ci_hi"]) - math.log(s["ci_lo"])) / (2 * _z(0.05))
        var += se * se
    sd = math.sqrt(var)
    za, zs = _z(alpha), _z(alpha / n_tested)
    return dict(n_steps=len(steps), speedup=math.exp(ls), ci_lo=math.exp(ls - za * sd), ci_hi=math.exp(ls + za * sd),
                sel_adj_ci_lo=math.exp(ls - zs * sd), sel_adj_ci_hi=math.exp(ls + zs * sd), n_tested=n_tested, alpha=alpha)
