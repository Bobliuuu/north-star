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
