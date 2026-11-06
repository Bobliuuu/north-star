"""Residual-risk closures from docs/SELFIMPROVE_REVIEW.md: timer cross-check, fd-delivered held-out data, anchored head CLI,
cumulative-vs-v0000 reporting, balanced ABBA order. CPU only, no GPU."""
import json, os, subprocess, sys, shutil, math
import pytest
from selfimprove import loop, gates as G, sandbox as S, proposer as P
from selfimprove.store import Store
from selfimprove import cli
from .test_loop import cu_diff
from .conftest import HARNESS_COMMON


# ---- (5) ABBA ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", range(1, 21))
def test_abba_balanced(n):
    o = loop.abba_order(n)
    assert len(o) == n + n % 2 and o.count(True) == o.count(False)       # odd counts are rounded up to stay balanced
    assert o[:4] in ([True, False, False, True], o[:4])                  # pattern starts b,k,k,b
    # linear drift cancels: sum of position index over base-first minus over cand-first is 0 for multiples of 4
    if len(o) % 4 == 0:
        assert sum(i for i, b in enumerate(o) if b) == sum(i for i, b in enumerate(o) if not b)


def test_abba_default_nine_runs_no_longer_5_vs_4():
    o = loop.abba_order(9)
    assert o.count(True) == o.count(False) == 5 and loop.abba_order(7) == loop.abba_order(8)


# ---- (1) timer cross-check --------------------------------------------------------------------------------------------
def good(**kw):
    j = dict(time_ms=10.0, ref_ms=12.0, harness_wall_ms=900.0, n_iters=21); j.update(kw); return j


def test_timer_crosscheck_unit():
    spec = G.TimerSpec(ratio_tol=4.0)
    assert G.timer_crosscheck(good(), 1.0, spec).passed
    assert not G.timer_crosscheck(good(time_ms=1.0), 1.0, spec).passed            # under-report 12x
    assert not G.timer_crosscheck(good(time_ms=100.0), 1.0, spec).passed          # wild over-report also rejected
    assert not G.timer_crosscheck(good(harness_wall_ms=60000.0), 1.0, spec).passed  # harness claims more wall than the evaluator saw
    for bad in ({"ref_ms": None}, {"harness_wall_ms": float("nan")}, {"n_iters": 1}, {"time_ms": -1.0}):
        assert not G.timer_crosscheck(good(**bad), 1.0, spec).passed
    j = good(); del j["ref_ms"]
    assert not G.timer_crosscheck(j, 1.0, spec).passed and not G.timer_crosscheck(None, 1.0, spec).passed


TIMED = HARNESS_COMMON + '''
import time
unit = 0.01 * (1 + 0.3 * abs(math.log2(SL / 512)))          # real work: sleep proportional to the kernel "cost"
def launch(n):
    t0 = time.perf_counter(); time.sleep(unit * n); return (time.perf_counter() - t0) * 1000
hi, lo = launch(8), launch(2)
reported = unit * 1000
if pol["dev_only_fast"]: reported /= 20                       # candidate-influenced self-report lies; real work unchanged
out({"time_ms": reported, "n_iters": 8, "ref_ms": (hi - lo) / 6, "harness_wall_ms": hi + lo})
'''


def timed_cfg(toy, timer=True):
    (toy.repo / "harness/timed.py").write_text(TIMED)
    toy.bench_cmd = [sys.executable, "harness/timed.py"]
    toy.timer = G.TimerSpec() if timer else None
    toy.speed = G.SpeedSpec(n_runs=5)
    return toy


def ev_files(toy, sl="256", lie=False):
    base = loop.editable_files_in_repo(toy.repo, toy.scope)
    new = {"src/kvm/moe_fused.cu": base["src/kvm/moe_fused.cu"].replace("SL = 64", f"SL = {sl}")}
    if lie: new["src/opt/policy_tables/sched.json"] = base["src/opt/policy_tables/sched.json"].replace("false", "true")
    return base, new


def test_timer_honest_candidate_passes(toy):
    toy = timed_cfg(toy); toy.bench_timeout_s = 20
    r = loop.Evaluator(toy).evaluate(*ev_files(toy))
    assert r["verdict"] == "pass", r


def test_timer_underreporting_candidate_rejected_even_in_diff_mode(toy):
    toy = timed_cfg(toy); toy.bench_timeout_s = 20
    r = loop.Evaluator(toy).evaluate(*ev_files(toy, sl="64", lie=True))     # SAME real speed as baseline, claims 20x
    assert r["verdict"] == "gate_fail:timer", r
    assert "disagree" in r["gates"]["timer"]["reason"]
    # without the cross-check the lie would have sailed through the speed gate: proves the check is what stops it
    toy.timer = None
    assert loop.Evaluator(toy).evaluate(*ev_files(toy, sl="64", lie=True))["verdict"] == "pass"


def test_timer_missing_fields_fail_closed(toy):
    toy.timer = G.TimerSpec()          # toy bench emits only time_ms
    r = loop.Evaluator(toy).evaluate(*ev_files(toy))
    assert r["verdict"] == "gate_fail:timer"


# ---- (2) held-out data never in the candidate workspace ---------------------------------------------------------------
SECRET = "HELDOUT-MARKER-7f3a9c"
PROBE_BUILD = HARNESS_COMMON + '''
import subprocess
MK = ("%s" + "%s").encode()      # the probe's own source must not contain the marker verbatim
found = []
for dp, dn, fn in os.walk(ws):
    for f in fn:
        try:
            if MK in open(os.path.join(dp, f), "rb").read(): found.append(os.path.join(dp, f))
        except OSError: pass
found += [k for k in os.environ if "HELDOUT" in k]
for fd in range(3, 64):
    try:
        if MK in os.pread(fd, 1 << 16, 0): found.append("fd%%d" %% fd)
    except OSError: pass
sys.stderr.write("PROBE:" + ("FOUND " + repr(found) if found else "CLEAN") + "\\n")
''' % (SECRET[:6], SECRET[6:])
QUALITY_FD = '''
import os, sys, json
nonce = os.environ.pop("NS_NONCE"); fds = json.loads(os.environ.pop("NS_HELDOUT_FDS"))
data = json.loads(os.pread(fds["held.json"], 1 << 20, 0))
import subprocess        # a spawned child (stand-in for the candidate-built binary) must NOT inherit the fd
kid = subprocess.run([sys.executable, "-c", "import os,sys\\ntry:\\n os.pread(%d,10,0);print('LEAK')\\nexcept OSError:print('closed')" % fds["held.json"]],
                     capture_output=True, text=True).stdout.strip()
print(json.dumps({"items": data["items"], "nonce": nonce, "child": kid}))
'''


@pytest.fixture
def hcfg(toy, tmp_path):
    sec = tmp_path / "private"; sec.mkdir()
    (sec / "held.json").write_text(json.dumps({"items": [1.0] * 30, "marker": SECRET}))
    (toy.repo / "harness/build.py").write_text(PROBE_BUILD)
    (toy.repo / "harness/qfd.py").write_text(QUALITY_FD)
    toy.build_cmd = [sys.executable, "harness/build.py"]
    toy.quality_cmd = [sys.executable, "harness/qfd.py"]
    toy.heldout_files = {"held.json": sec / "held.json"}
    return toy


def test_heldout_not_in_workspace_during_candidate_compile(hcfg):
    base = loop.editable_files_in_repo(hcfg.repo, hcfg.scope)
    hook = "import os,sys\n"            # candidate-controlled code that runs inside the compile/audit step
    r = loop.Evaluator(hcfg).evaluate(base, {"src/opt/policy_tables/hook.py": hook})
    assert "PROBE:CLEAN" in r["sandbox"]["build_cand"]["stderr"] and "PROBE:CLEAN" in r["sandbox"]["build_base"]["stderr"], r["sandbox"]


def test_candidate_hook_cannot_find_heldout_anywhere(hcfg):
    """The hook runs in-process in every toy harness command except quality (where only the evaluator-owned qfd.py runs)."""
    base = loop.editable_files_in_repo(hcfg.repo, hcfg.scope)
    hook = ("import os\nfor dp, dn, fn in os.walk(os.environ['NS_WORKSPACE']):\n for f in fn:\n"
            "  if (b'%s'+b'%s') in open(os.path.join(dp,f),'rb').read(): raise SystemExit('FOUND '+f)\n"
            "assert not [k for k in os.environ if 'HELDOUT' in k]\n" % (SECRET[:6], SECRET[6:]))
    r = loop.Evaluator(hcfg).evaluate(base, {"src/opt/policy_tables/hook.py": hook})
    assert r["verdict"] not in ("build_fail", "sandbox_violation", "evaluator_error"), r
    assert r["gates"]["correctness"]["passed"], r["gates"]      # hook exited cleanly (nothing found) in correctness runs too


def test_heldout_delivered_by_fd_to_quality_only_and_children_do_not_inherit(hcfg):
    ev = loop.Evaluator(hcfg)
    base = loop.editable_files_in_repo(hcfg.repo, hcfg.scope)
    ws = S.Workspace.create(hcfg.repo, hcfg.subdirs, base, set(base), hcfg.tmp_parent)
    try:
        r = ev._run(ws, hcfg.quality_cmd, 10, {"NS_SEED": "0"}, heldout=True)
        assert r.parsed and len(r.parsed["items"]) == 30 and r.parsed["child"] == "closed", (r.stderr, r.stdout)
        r2 = ev._run(ws, hcfg.quality_cmd, 10, {"NS_SEED": "0"})            # any other command: nothing delivered
        assert r2.parsed is None and r2.rc != 0
    finally:
        ws.cleanup()


def test_heldout_inside_workspace_is_a_violation(toy, tmp_path):
    toy.heldout_files = {"h": toy.repo / "harness/check.py"}              # someone configured held-out data inside a copied subtree
    base = loop.editable_files_in_repo(toy.repo, toy.scope)
    r = loop.Evaluator(toy).evaluate(base, {"src/opt/policy_tables/hook.py": "x=1\n"})
    assert r["verdict"] == "sandbox_violation" and "held-out data present in workspace" in r["sandbox"]["error"]


def test_real_kernel_config_keeps_traces_out_of_workspace():
    import runpy, pathlib
    repo = pathlib.Path(__file__).resolve().parents[3]
    cfg = runpy.run_path(str(repo / "harness/selfimprove/config.py"))["make_config"](repo, "kernel")
    assert not any("routing_traces" in s for s in cfg.subdirs) and "routing_traces.json" in cfg.heldout_files and cfg.timer is not None


# ---- (3) anchored head -------------------------------------------------------------------------------------------------
def mk_store(tmp_path):
    st = Store(tmp_path / "st"); st.init_baseline({"a.txt": "x"}); st.log(dict(event="proposal", verdict="no_change")); return st


def run_cli(*a):
    return cli.main(list(a))


def test_head_record_and_verify_cli(tmp_path, capsys):
    st = mk_store(tmp_path)
    assert run_cli("head", "--store", str(st.root), "--label", "r1") == 0
    line = capsys.readouterr().out.strip()
    assert line == f"selfimprove-head r1 n=2 head={st.head()}" == st.head_record("r1")
    h = line.split("head=")[1]
    assert run_cli("verify-log", "--store", str(st.root), "--expected-head", h) == 0
    st.log(dict(event="proposal", verdict="no_change"))                      # a round happened since the anchor
    assert run_cli("verify-log", "--store", str(st.root), "--expected-head", h) == 1           # exact: stale anchor
    assert run_cli("verify-log", "--store", str(st.root), "--expected-head", h, "--allow-newer") == 0
    assert run_cli("verify-log", "--store", str(st.root), "--expected-head", "0" * 64, "--allow-newer") == 1


def test_rewrite_and_truncation_detected_by_anchor(tmp_path):
    st = mk_store(tmp_path); h = st.head(); st.log(dict(event="proposal", verdict="x"))
    h2 = st.head()
    lines = st.log_path.read_text().splitlines()
    st.log_path.write_text("\n".join(lines[:-1]) + "\n")                         # truncate: chain still self-consistent
    assert st.verify_log() and not st.verify_anchor(h2) and not st.verify_anchor(h2, exact=False)
    # fully re-chained rewrite of the first proposal record: chain valid, anchor catches it
    recs = [json.loads(l) for l in lines]
    recs[1]["verdict"] = "promoted"; prev = recs[0]["hash"]
    from selfimprove.store import sha256_text
    for r in recs[1:]:
        r["prev_hash"] = prev; r.pop("hash", None); r["hash"] = sha256_text(json.dumps(r, sort_keys=True, default=str)); prev = r["hash"]
    st.log_path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    assert st.verify_log() and not st.verify_anchor(h, exact=False) and not st.verify_anchor(h2, exact=False)


def test_existing_result_stores_verify_with_their_own_head():
    import pathlib
    repo = pathlib.Path(__file__).resolve().parents[3]
    for d in sorted((repo / "results").glob("selfimprove_*")):
        if not (d / "proposals.jsonl").exists(): continue
        st = Store(d)
        assert st.verify_log(), d
        assert st.verify_anchor(st.head()) and st.head_record(d.name).startswith("selfimprove-head")


# ---- (4) cumulative vs v0000 ------------------------------------------------------------------------------------------
def step(s, lo, hi): return dict(speedup=s, ci_lo=lo, ci_hi=hi)


def test_cumulative_gain_math():
    c = G.cumulative_gain([step(1.2, 1.1, 1.3), step(1.1, 1.04, 1.16)], n_tested=1)
    assert c["speedup"] == pytest.approx(1.32) and c["ci_lo"] < 1.32 < c["ci_hi"] and c["n_steps"] == 2
    assert c["ci_lo"] < 1.1 * 1.04 + 1.0   # sanity: finite, positive
    w = G.cumulative_gain([step(1.2, 1.1, 1.3), step(1.1, 1.04, 1.16)], n_tested=40)
    assert w["sel_adj_ci_lo"] < c["ci_lo"] < c["ci_hi"] < w["sel_adj_ci_hi"]                 # more chances for noise -> wider
    assert G.cumulative_gain([])["speedup"] == 1.0


BENCH2 = HARNESS_COMMON + '''
rng = random.Random(seed)
t = 10 * (1 + 0.3 * abs(math.log2(SL / 512)))
out({"time_ms": t * (1 + 0.004 * rng.uniform(-1, 1))})
'''


def test_two_promotions_compare_to_current_and_report_cumulative(toy, tmp_path):
    (toy.repo / "harness/bench2.py").write_text(BENCH2); toy.bench_cmd = [sys.executable, "harness/bench2.py"]
    st = Store(tmp_path / "s2")
    lp = loop.Loop(toy, st, P.MockProposer([cu_diff(64, 128), cu_diff(128, 256)]), allow_mock_promotion=True)
    r1, r2 = lp.step(), lp.step()
    assert r1["verdict"] == r2["verdict"] == "promoted", (r1.get("gates"), r2.get("gates"))
    assert r2["base_version"] == "v0001"                                          # second round was judged against the CURRENT baseline
    s1, s2 = r1["gates"]["speed"]["data"]["dev"]["speedup"], r2["gates"]["speed"]["data"]["dev"]["speedup"]
    assert 1.1 < s2 < 1.3 and r2["gates"]["speed"]["data"]["dev"]["base_med"] == pytest.approx(r1["gates"]["speed"]["data"]["dev"]["cand_med"], rel=0.02)
    cum = r2["cumulative_vs_v0000"]["dev"]
    assert cum["speedup"] == pytest.approx(s1 * s2) and cum["n_steps"] == 2 and cum["ci_lo"] < cum["speedup"] < cum["ci_hi"]
    assert cum["sel_adj_ci_lo"] <= cum["ci_lo"]
    att = st.attribution()["cumulative_vs_v0000"]
    assert att["dev"]["speedup"] == pytest.approx(s1 * s2) and att["heldout"]["n_steps"] == 2 and att["baseline"] == "v0000"
    # a developer-only store has a neutral cumulative gain
    st0 = Store(tmp_path / "s0"); st0.init_baseline({"a": "b"})
    assert st0.attribution()["cumulative_vs_v0000"]["dev"]["speedup"] == 1.0


# ---- real kernel_bench.py wiring (fake sleeping binary: real wall time, tunable self-report) ---------------------------
SLEEPY = """#!%s
import sys, time, json
a = sys.argv; reps = int(a[a.index("--reps") + 1]); shapes = a[a.index("--shapes") + 1].split(";")
time.sleep(0.002 * reps * len(shapes))
with open(a[2], "w") as f:
    for s in shapes: f.write(json.dumps(dict(out_maxabs=1.0, fused_vs_two_maxabsdiff=0.0, v=dict(fused=dict(us_med=%s)))) + "\\n")
"""


@pytest.mark.parametrize("claimed_us,ok", [(2000.0, True), (100.0, False)])
def test_kernel_bench_timer_crosscheck_catches_underreport(tmp_path, claimed_us, ok):
    import pathlib
    repo = pathlib.Path(__file__).resolve().parents[3]
    ws = tmp_path / "ws"
    shutil.copytree(repo / "harness/selfimprove", ws / "harness/selfimprove", ignore=shutil.ignore_patterns("__pycache__", "tests"))
    (ws / "_scratch").mkdir(); (ws / "build").mkdir()
    b = ws / "build/moe_fused"; b.write_text(SLEEPY % (sys.executable, claimed_us)); b.chmod(0o755)
    r = S.run([sys.executable, "harness/selfimprove/kernel_bench.py"], ws, 60, {"NS_NONCE": "n", "NS_SPLIT": "dev"})
    assert r.rc == 0, r.stderr[-500:]
    j = r.json("n")
    assert j["n_iters"] == 21 and j["ref_ms"] and j["harness_wall_ms"] > 0
    g = G.timer_crosscheck(j, r.wall_s, G.TimerSpec())
    assert g.passed is ok, (j, g.reason)


def test_timer_calibrated_relative_to_baseline():
    spec = G.TimerSpec(calibrate=True, ratio_tol=1.5)
    mk = lambda t, ref: dict(time_ms=t, ref_ms=ref)
    base = [("b", mk(0.39, 3.7)), ("b", mk(0.40, 3.75)), ("b", mk(0.39, 3.7))]      # absolute ratio ~9.4: would fail the old absolute 4x cap
    assert G.timer_calibrated(base + [("k", mk(0.37, 3.7))], spec).passed             # genuinely faster kernel, same harness overhead
    assert not G.timer_calibrated(base + [("k", mk(0.10, 3.7))], spec).passed         # under-report 4x vs harness clock
    assert not G.timer_calibrated([("k", mk(0.39, 3.7))], spec).passed                # no baseline run to calibrate against


def test_timer_calibrated_tolerates_single_run_jitter_but_not_median_shift():
    spec = G.TimerSpec(calibrate=True, ratio_tol=1.5)
    base = [("b", dict(time_ms=0.40, ref_ms=3.0)) for _ in range(8)]
    jitter = [("k", dict(time_ms=0.40, ref_ms=3.0)) for _ in range(8)] + [("k", dict(time_ms=0.40, ref_ms=5.0))]     # one 1.7x ref outlier
    assert G.timer_calibrated(base + jitter, spec).passed
    shifted = [("k", dict(time_ms=0.25, ref_ms=3.0)) for _ in range(9)]                                         # every run under-reports 1.6x
    assert not G.timer_calibrated(base + shifted, spec).passed
