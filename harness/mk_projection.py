"""CPU-only projection of PLD tok/s on the megakernel from the MEASURED S3 block cost curve (results/mk_block_s3_vast.json) and
MEASURED acceptance (H13 per-task traces replayed through pld.spec_loop; H15 per-task aggregates). Everything it outputs is INFERRED:
no end-to-end megakernel PLD run exists. Reuses mk_block.project() for all round-cost arithmetic."""
import json, math, pathlib
import mk_block as mb

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAPS = range(2, 9)  # verify-block cap T (draft length g = T-1)


def mk_curves(path=ROOT / "results/mk_block_s3_vast.json"):
    out = {}
    for r in json.load(open(path))["rows"]:
        if r.get("kind") == "s3":
            out.setdefault(r["ctx"], {})[r["T"]] = r["ms_median"]
    return out  # {2048: {1..8: ms}, 8192: {...}}


def fill(curve, tmax=13):
    """interpolate missing T linearly, extrapolate last slope (device ms per block)."""
    ks = sorted(curve)
    out = {}
    for T in range(1, tmax + 1):
        if T in curve:
            out[T] = curve[T]
        elif T < ks[-1]:
            lo = max(k for k in ks if k < T); hi = min(k for k in ks if k > T)
            out[T] = curve[lo] + (curve[hi] - curve[lo]) * (T - lo) / (hi - lo)
        else:
            out[T] = curve[ks[-1]] + (curve[ks[-1]] - curve[ks[-2]]) / (ks[-1] - ks[-2]) * (T - ks[-1])
    return out


def tokps(rounds, curve):
    """tok/s of a round list under a verify-ms curve (via mb.project; speedup * 1000 / plain_ms)."""
    return mb.project(rounds, curve, curve[1]) * 1000.0 / curve[1]


def geomean(xs):
    xs = list(xs)
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def e_acc(a, L):
    """expected accepted drafts of a length-L draft, iid per-position hit prob a (L may be real: linear mix of floor/ceil)."""
    f = lambda n: sum(a ** k for k in range(1, n + 1))
    lo = int(math.floor(L)); return f(lo) + (L - lo) * (f(lo + 1) - f(lo))


def solve_a(acc, L):
    if acc <= 0 or L <= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    if acc >= L:
        return 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if e_acc(mid, L) < acc else (lo, mid)
    return (lo + hi) / 2


def agg_rounds(rounds_total, draft_rounds, drafted, accepted, cap_g):
    """Round list from per-task aggregates (what H15 saved): plain rounds + draft rounds of mean length min(dbar, cap_g)
    with per-position hit prob fitted to the g12 accepted/draft-round. INFERRED model, validated against exact replay in the JSON."""
    rp = rounds_total - draft_rounds
    out = [(0, 0)] * rp
    if draft_rounds == 0:
        return out
    dbar, abar = drafted / draft_rounds, accepted / draft_rounds
    a = solve_a(abar, dbar)
    L = min(dbar, cap_g)
    lo = int(math.floor(L)); nhi = int(round((L - lo) * draft_rounds))
    acc_lo, acc_hi = e_acc(a, lo), e_acc(a, lo + 1)
    return out + [(lo + 1, acc_hi)] * nhi + [(lo, acc_lo)] * (draft_rounds - nhi)


def best_cap(rounds_by_cap, curve):
    return max(((c, tokps(r, curve)) for c, r in rounds_by_cap.items()), key=lambda t: t[1])


def breakeven(curve_mk, curve_fast, tmax_fast=13):
    """iid per-position hit prob a, a draft ALWAYS available (best case for PLD). E(a,T)=sum_{k<T} a^k tokens per T-row block."""
    E = lambda a, T: sum(a ** k for k in range(T))
    sp_mk = lambda a, T: E(a, T) * 1000 / curve_mk[T]
    sp_fast = lambda a, T: E(a, T) * 1000 / curve_fast[T]
    grid = [i / 1000 for i in range(0, 1001)]
    first = lambda pred: next((a for a in grid if pred(a)), None)
    res = {"tokens_per_round_to_tie_own_plain": {T: curve_mk[T] / curve_mk[1] for T in CAPS},
           "a_vs_mk_plain": {T: first(lambda a: sp_mk(a, T) >= 1000 / curve_mk[1]) for T in CAPS},
           "a_vs_fast_plain": {T: first(lambda a: sp_mk(a, T) >= 1000 / curve_fast[1]) for T in CAPS}}
    def intervals(pred):
        out, start = [], None
        for a in grid:
            if pred(a) and start is None: start = a
            if not pred(a) and start is not None: out.append([start, round(a - 0.001, 3)]); start = None
        if start is not None: out.append([start, 1.0])
        return out
    fbest = lambda a: max(sp_fast(a, T) for T in range(1, tmax_fast + 1))
    res["a_intervals_mk_best_T_le8_ge_fast_pld_best_T_le13"] = intervals(lambda a: max(sp_mk(a, T) for T in CAPS) >= fbest(a))
    res["a_intervals_mk_cap4_ge_fast_pld_best_T_le13"] = intervals(lambda a: max(sp_mk(a, T) for T in range(1, 5)) >= fbest(a))
    res["mk_over_fastpld_ratio_at_a"] = {str(a): max(sp_mk(a, T) for T in CAPS) / max(sp_fast(a, T) for T in range(1, tmax_fast + 1)) for a in (0.1, 0.3, 0.5, 0.7, 0.9)}
    return res
