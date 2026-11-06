"""EvalConfig factories for the two self-improvement tracks + the (dev-only) prompt material + the labeled MOCK dry-run diff.

Lives outside the editable scope and is hash-protected in every sandbox. Loaded by the evaluator side with
    runpy.run_path("harness/selfimprove/config.py")["make_config"](repo, track)
Held-out data (calib routing traces, held-out shapes/buckets) is NEVER referenced from profile_summary()/dev_results().
"""
import difflib, glob, os, sys
from pathlib import Path

DEV_KERNEL_SHAPES = [(1, 1, 2), (4, 4, 4), (8, 8, 4), (16, 16, 8)]               # (Tv, Td, r); mirrors kernel_bench.py DEV
HELDOUT_KERNEL_SHAPES = [(3, 5, 3), (6, 9, 5), (12, 7, 6), (13, 3, 7), (5, 11, 3), (9, 6, 5)]
DEV_POLICY_BUCKETS = [("code", 2048, 1, "corr75"), ("code", 8192, 1, "corr75"), ("code", 8192, 1, "uniform"), ("code", 32768, 1, "corr75")]
HELDOUT_POLICY_CTX = {3072, 5000, 12000, 20000}
MOCK_BANNER = "DRY RUN - MOCK PROPOSER - NOT NORTH-AUTHORED - never counted as North's work"


def _ints(shapes):
    return {int(x) for s in shapes for x in s}


def _calibration_is_measured(path):
    """True only if the file exists, is non-empty and EVERY row is provenance == 'measured' (build with scripts/build_accept_calibration.py)."""
    import json
    try:
        rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    except (OSError, ValueError):
        return False
    return bool(rows) and all(isinstance(r, dict) and r.get("provenance") == "measured" for r in rows)


def make_config(repo, track="kernel", python="python3", require_measured_calibration=False):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from selfimprove import gates
    from selfimprove.loop import EvalConfig
    from selfimprove.policy import Scope, ScopeRule, Limits
    repo = Path(repo); h = "harness/selfimprove"
    if track == "kernel":
        return EvalConfig(
            repo=repo, subdirs=["src/kvm", h],
            heldout_files={"routing_traces.json": repo / "results" / "routing_traces.json"}, timer=gates.TimerSpec(),
            build_cmd=[python, f"{h}/kernel_build.py"], correctness_cmd=[python, f"{h}/kernel_check.py"],
            quality_cmd=[python, f"{h}/kernel_quality.py"], bench_cmd=[python, f"{h}/kernel_bench.py"],
            scope=Scope([ScopeRule("src/kvm/moe_fused.cu", regions_only=True)], Limits(max_files=1, max_hunks=4, max_changed_lines=30)),
            correctness=gates.CorrectnessSpec(rel_l2_tol=1e-3, min_cases=16),
            quality=gates.QualitySpec(margin=1e-5, min_items=20),
            speed=gates.SpeedSpec(n_runs=9, min_gain=0.03),
            build_timeout_s=420, check_timeout_s=900, bench_timeout_s=330,
            dev_shapes=_ints(DEV_KERNEL_SHAPES), heldout_shapes=_ints(HELDOUT_KERNEL_SHAPES), usd_per_hour=2.5)
    if track == "policy":
        if require_measured_calibration and not _calibration_is_measured(repo / "results" / "accept_calibration.jsonl"):
            raise RuntimeError("policy track live run needs MEASURED acceptance calibration results/accept_calibration.jsonl "
                               "(only SYNTHETIC available): refusing, a predicted tok/s from invented acceptance means nothing")
        return EvalConfig(
            repo=repo, subdirs=["src/opt", h, "results"],
            correctness_cmd=[python, f"{h}/policy_eval.py", "check"], quality_cmd=[python, f"{h}/policy_eval.py", "quality"],
            bench_cmd=[python, f"{h}/policy_eval.py", "bench"],
            scope=Scope([ScopeRule("src/opt/policy_tables/spec_space.json")], Limits(max_files=1, max_hunks=3, max_changed_lines=12)),
            correctness=gates.CorrectnessSpec(rel_l2_tol=1e-3, min_cases=8),
            quality=gates.QualitySpec(margin=0.005, min_items=20),
            speed=gates.SpeedSpec(n_runs=5, min_gain=0.03),   # deterministic metric (cached per workspace): repeats cost ~nothing
            check_timeout_s=600, bench_timeout_s=240,
            dev_shapes={s for b in DEV_POLICY_BUCKETS for s in (b[1],)}, heldout_shapes=set(HELDOUT_POLICY_CTX), usd_per_hour=2.5)
    raise ValueError(f"unknown track {track!r}")


def profile_summary(track):
    if track == "kernel":
        return ("Kernel: fused draft/verify MoE expert GEMM (one North MoE layer, 128 experts, top-8, hidden 2048, expert dim 768, bf16 WMMA, "
                "fp32 atomics). Work item = (expert, 64-wide slice); a block streams gate/up/down weights of its slice once and applies them to "
                "all tokens routed to that expert. The kernel is HBM-bandwidth bound on A100 (measured ~0.7 of peak streaming read). "
                "Tunables you may change: KC (k-chunk of H staged in smem), XPAD/HPAD (smem row padding), MT_SPLIT (token count above which the "
                "2-row-tile path is used), MINB_PER_SM (launch bounds min blocks). Invalid values fail a protected static_assert.")
    return ("Policy table src/opt/policy_tables/spec_space.json: the search space of the Phase-3 speculative-decoding optimizer "
            "(gammas = candidate draft block lengths, modes = expert schedules, n_groups = layer-group granularity of the CP-SAT schedule). "
            "The score is the optimizer's PREDICTED tokens/s (cost model + acceptance calibration), not a measurement.")


def dev_results(track, baseline_dev_metric=None):
    """DEV information only. Held-out shapes/buckets/traces are never mentioned here."""
    if track == "kernel":
        s = "Dev shapes (Tv,Td,r), routing=corr75: " + ", ".join(map(str, DEV_KERNEL_SHAPES)) + ". Metric: mean fused-launch time (lower is better)."
    else:
        s = "Dev buckets (workload,ctx,batch,routing): " + ", ".join(map(str, DEV_POLICY_BUCKETS)) + ". Metric: predicted ms per 1000 tokens (lower is better)."
    if baseline_dev_metric is not None:
        s += f" Baseline dev metric: {baseline_dev_metric:.4g} ms."
    return s


def mock_structured(track):
    """Scripted MOCK structured-edit response for dry runs (labeled by the Loop as mock; never North-authored)."""
    if track == "kernel": return '{"edits":[{"name":"KC","value":128}],"rationale":"MOCK"}'
    return '{"edits":[{"name":"gammas","value":[1,2,3,4,5,6,8]}],"rationale":"MOCK"}'


def mock_diff(track, base_files):
    """Scripted MOCK proposal for dry runs: a tiny valid edit. Output is always labeled and never attributed to North."""
    if track == "kernel":
        path, old, new = "src/kvm/moe_fused.cu", "constexpr int KC = 256;", "constexpr int KC = 128;"
    else:
        path, old, new = "src/opt/policy_tables/spec_space.json", '"gammas": [1, 2, 3, 4, 6, 8]', '"gammas": [1, 2, 3, 4, 5, 6, 8]'
    a = base_files[path].splitlines(keepends=True)
    b = [l.replace(old, new) if old in l else l for l in a]
    d = "".join(difflib.unified_diff(a, b, f"a/{path}", f"b/{path}", n=3))
    return f"{MOCK_BANNER}\n```diff\n{d}```\n"
