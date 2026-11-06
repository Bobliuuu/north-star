"""Tests for the REAL harness (harness/selfimprove/*), the CUDA markers and the Modal evaluator glue. No GPU, no network, no North:
the kernel binary is replaced by a python stand-in that mimics `moe_fused check|bench`; the dry-run proposer is MOCK."""
import json, os, pathlib, re, runpy, shutil, stat, sys
import pytest
from selfimprove import diffs as D, policy as P, sandbox as S, loop as L, proposer as PR
from selfimprove.store import Store

REPO = pathlib.Path(__file__).resolve().parents[3]
CU = REPO / "src/kvm/moe_fused.cu"
HCFG = runpy.run_path(str(REPO / "harness/selfimprove/config.py"))
SLOW = pytest.mark.skipif(not os.environ.get("NS_SLOW"), reason="set NS_SLOW=1 (minutes of CPU)")


# ---- (1) markers around safe tunables only, default values unchanged --------------------------------------------------
def test_markers_wrap_only_tunables_with_unchanged_defaults():
    text = CU.read_text(); mask = P.editable_line_mask(text); lines = text.split("\n")
    inside = [l.strip() for l, m in zip(lines, mask) if m]
    code = [l.split("//")[0] for l in inside]
    names = [re.match(r"constexpr int (\w+) = (\d+);", l).groups() for l in inside]
    assert dict(names) == {"KC": "256", "XPAD": "8", "HPAD": "8", "MT_SPLIT": "16", "MINB_PER_SM": "1"}
    for protected in ("H = 2048", "SL = 64", "MAXTOK = 32", "NS = I / SL", "LDX = KC + XPAD", "static_assert", "launch_bounds", "int main"):
        assert not any(protected in l for l in code)
    assert text.count("NS-EDITABLE-BEGIN") == 1 and text.count("NS-EDITABLE-END") == 1
    assert "__launch_bounds__(128, MINB_PER_SM)" in text and "t.pair_cnt > MT_SPLIT" in text
    assert "constexpr int LDX = KC + XPAD, LDH = SL + HPAD;" in text      # == KC + 8 / SL + 8 at the defaults


def _diff(path, old, new, text):
    a = text.splitlines(keepends=True); b = [new + "\n" if l.rstrip("\n") == old else l for l in a]
    import difflib
    return "".join(difflib.unified_diff(a, b, f"a/{path}", f"b/{path}", n=3))


def test_scope_lets_tunable_edit_apply_and_blocks_everything_else():
    cfg = HCFG["make_config"](REPO, "kernel"); text = CU.read_text(); get = {"src/kvm/moe_fused.cu": text}.get
    ok = D.parse(_diff("src/kvm/moe_fused.cu", "constexpr int KC = 256;                // k chunk of H staged in smem for stage 1 (multiple of 16 dividing H=2048)",
                       "constexpr int KC = 128;", text), cfg.scope)
    assert "KC = 128" in D.apply_all(ok, get, cfg.scope)["src/kvm/moe_fused.cu"]
    for old, new in (("constexpr int SL = 64;                 // intermediate columns per work item (fixed: 4 warps x 16-col WMMA tiles)", "constexpr int SL = 32;"),
                     ('static_assert(SL == 64, "SL is tied to the 4-warp x 16-column tiling");', "// gone")):
        with pytest.raises(D.DiffError):
            D.apply_all(D.parse(_diff("src/kvm/moe_fused.cu", old, new, text), cfg.scope), get, cfg.scope)


def test_mock_dry_run_diffs_apply_for_both_tracks_and_are_labeled():
    for track in ("kernel", "policy"):
        cfg = HCFG["make_config"](REPO, track)
        base = L.editable_files_in_repo(REPO, cfg.scope)
        raw = HCFG["mock_diff"](track, base)
        assert "NOT NORTH-AUTHORED" in raw
        d = PR.extract_diff(raw)
        L.D.apply_all(L.D.parse(d, cfg.scope), base.get, cfg.scope)


# ---- (2) held-out data stays out of the proposer prompt ---------------------------------------------------------------
def test_prompt_contains_no_heldout_material():
    traces = json.load(open(REPO / "results/routing_traces.json"))
    held_ids = [t for t, v in traces.items() if v["split"] == "calib"]
    for track in ("kernel", "policy"):
        cfg = HCFG["make_config"](REPO, track)
        req = PR.ProposalRequest("v0000", L.editable_files_in_repo(REPO, cfg.scope), L.scope_description(cfg.scope),
                                 HCFG["profile_summary"](track), HCFG["dev_results"](track, 1.23))
        prompt = PR.SYSTEM + PR.build_prompt(req)
        assert "routing_traces" not in prompt and "calib split" not in prompt
        assert not any(t in prompt for t in held_ids)
        for s in HCFG["HELDOUT_KERNEL_SHAPES"]: assert str(s) not in prompt
        for c in HCFG["HELDOUT_POLICY_CTX"]: assert str(c) not in prompt
        assert "5000" not in prompt and "12000" not in prompt


def test_dev_and_heldout_shape_ints_are_disjoint_for_audit():
    c = HCFG["make_config"](REPO, "kernel")
    assert c.dev_shapes and c.heldout_shapes and not (c.dev_shapes & c.heldout_shapes)
    k = (REPO / "harness/selfimprove/kernel_bench.py").read_text()
    assert str(HCFG["DEV_KERNEL_SHAPES"]) in k and str(HCFG["HELDOUT_KERNEL_SHAPES"]) in k


def test_policy_live_run_refuses_synthetic_calibration(tmp_path):
    if (REPO / "results/accept_calibration.jsonl").exists(): pytest.skip("measured calibration present")
    with pytest.raises(RuntimeError, match="MEASURED"):
        HCFG["make_config"](REPO, "policy", require_measured_calibration=True)
    HCFG["make_config"](REPO, "policy")          # dry run allowed


# ---- fake-binary workspace for the kernel scripts -----------------------------------------------------------------
FAKE = '''#!{py}
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.join({repo!r}, "src", "kvm")); torch.set_num_threads(2)
from ref import moe_ref as m
mode = sys.argv[1]
if mode == "check":
    d = sys.argv[2]; E, Tv, Td, r, rn = map(int, open(d + "/meta.txt").read().split())
    ld = lambda n, dt, sh: np.fromfile(f"{{d}}/{{n}}.bin", dtype=dt).reshape(sh)
    bf = lambda n, sh: torch.from_numpy(ld(n, np.int16, sh)).view(torch.bfloat16)
    x = bf("x", (Tv + Td, m.H)); wg, wu, wd = (bf(n, (E, m.I, m.H)) for n in ("wg", "wu", "wd"))
    idx = torch.from_numpy(ld("idx", np.int32, (Tv + Td, 8))); w = torch.from_numpy(ld("w", np.float32, (Tv + Td, 8)))
    ref = m.reference_grouped(x, wg, wu, wd, idx, w, Tv, r, bool(rn)).numpy()
    for name in ("fused", "two"):
        (ref * (1 + {noise})).astype(np.float32).tofile(f"{{d}}/out_{{name}}.bin")
else:
    outp = sys.argv[2]; shapes = [tuple(map(int, s.split(","))) for s in sys.argv[sys.argv.index("--shapes") + 1].split(";")]
    with open(outp, "w") as f:
        import json
        for tv, td, r in shapes:
            f.write(json.dumps(dict(out_maxabs=1.0, fused_vs_two_maxabsdiff={diff}, v=dict(fused=dict(us_med={us}*(tv+td)))) ) + "\\n")
'''


def fake(ws, noise=1e-5, diff=0.0, us=10.0):
    b = ws / "build/moe_fused"; b.write_text(FAKE.format(py=sys.executable, repo=str(REPO), noise=noise, diff=diff, us=us))
    b.chmod(b.stat().st_mode | stat.S_IXUSR)


@pytest.fixture
def kws(tmp_path):
    ws = tmp_path / "ws"
    for rel in ("src/kvm", "harness/selfimprove"): shutil.copytree(REPO / rel, ws / rel, ignore=shutil.ignore_patterns("__pycache__"))
    (ws / "results").mkdir(); shutil.copy(REPO / "results/routing_traces.json", ws / "results"); (ws / "_scratch").mkdir(); (ws / "build").mkdir()
    fake(ws)
    return ws


def run_h(ws, script, seed="123", split="dev", args=(), timeout=400):
    nonce = "n0nce"
    r = S.run([sys.executable, f"harness/selfimprove/{script}", *args], ws, timeout, {"NS_NONCE": nonce, "NS_SEED": seed, "NS_SPLIT": split})
    assert r.rc == 0, r.stderr[-800:]
    return r, r.json(nonce)


def test_kernel_check_protocol_cases_and_fresh_seed(kws):
    r, j = run_h(kws, "kernel_check.py", seed="1")
    cs = j["cases"]; names = [c["name"] for c in cs]
    assert len(cs) >= 16 and len(set(names)) == len(names) and all(c["finite"] and c["rel_l2"] < 1e-3 for c in cs)
    assert "nonce" in j and r.json("wrong") is None                      # forged / wrong nonce is ignored by the evaluator
    fake(kws, noise=0.05)                                                  # a kernel off by 5% must show up as large rel_l2
    _, j2 = run_h(kws, "kernel_check.py", seed="2")
    assert min(c["rel_l2"] for c in j2["cases"]) > 1e-2
    fake(kws, noise=1e-5)
    (kws / "build/moe_fused").write_text("#!/bin/sh\nexit 3\n")           # crashing binary => non-finite cases, not a pass
    _, j3 = run_h(kws, "kernel_check.py", seed="3")
    assert all(not c["finite"] for c in j3["cases"])


def test_kernel_bench_splits_and_invalid_timing(kws):
    _, d = run_h(kws, "kernel_bench.py", split="dev"); _, h = run_h(kws, "kernel_bench.py", split="heldout")
    assert d["n_shapes"] == 4 and h["n_shapes"] == 6 and d["time_ms"] > 0 and h["time_ms"] > 0
    fake(kws, diff=0.5)                                                    # fused disagrees with two-pass => timing invalid
    _, bad = run_h(kws, "kernel_bench.py"); assert bad["time_ms"] is None


def test_nonce_not_visible_to_child_and_required(kws):
    probe = kws / "harness/selfimprove/probe.py"
    probe.write_text("import common, os, subprocess, sys\nr = subprocess.run(['env'], capture_output=True, text=True, env=common.CHILD_ENV)\n"
                     "common.emit({'leak': 'NS_NONCE' in r.stdout or 'NS_SEED' in r.stdout, 'in_self': 'NS_NONCE' in os.environ})\n")
    _, j = run_h(kws, "probe.py"); assert j["leak"] is False and j["in_self"] is False
    r = S.run([sys.executable, "harness/selfimprove/probe.py"], kws, 20, {})          # no nonce => harness refuses
    assert r.rc == 97


@SLOW
def test_kernel_quality_items_heldout_only(kws):
    _, j = run_h(kws, "kernel_quality.py", timeout=900)
    assert len(j["items"]) >= 20 and all(0.99 < x <= 1.0 for x in j["items"])


def test_quality_items_come_from_heldout_split_only(kws):
    src = (kws / "harness/selfimprove/kernel_quality.py").read_text()
    assert 'HELDOUT_SPLIT = "calib"' in src
    traces = json.load(open(kws / "results/routing_traces.json"))
    n = sum(1 for v in traces.values() if v["split"] == "calib") * 5
    assert n >= 20


# ---- policy track: real optimizer on CPU ----------------------------------------------------------------------------
@pytest.fixture
def pws(tmp_path):
    ws = tmp_path / "pws"
    for rel in ("src/opt", "harness/selfimprove", "results"):
        shutil.copytree(REPO / rel, ws / rel, ignore=shutil.ignore_patterns("__pycache__", "tests", "selfimprove*"))
    (ws / "_scratch").mkdir(); return ws


def test_policy_bench_and_check_real_optimizer(pws):
    _, c = run_h(pws, "policy_eval.py", args=["check"], seed="9")
    assert len(c["cases"]) == 8 and all(x["finite"] and x["rel_l2"] < 1e-3 for x in c["cases"]) and "SYNTHETIC" in c["calibration"] + "SYNTHETIC"
    _, d = run_h(pws, "policy_eval.py", args=["bench"])
    assert d["time_ms"] > 0 and d["mean_pred_tok_s"] > 0
    t = pws / "src/opt/policy_tables/spec_space.json"
    t.write_text(json.dumps({"gammas": [1, 2], "modes": ["two_pass"], "n_groups": 6}))
    _, d2 = run_h(pws, "policy_eval.py", args=["bench"])
    assert d2["time_ms"] > d["time_ms"]                                   # smaller search space predicts slower
    t.write_text(json.dumps({"gammas": [1, 99], "modes": ["two_pass"], "n_groups": 6}))
    _, bad = run_h(pws, "policy_eval.py", args=["bench"]); assert bad["time_ms"] is None   # table validation fails closed
    t.write_text('{"gammas": [1], "modes": ["x"], "n_groups": 6, "evil": 1}')
    _, bad = run_h(pws, "policy_eval.py", args=["quality"]); assert bad["items"] == []


@SLOW
def test_full_dry_run_loop_policy_track_cpu_only(tmp_path):
    """End to end on the real harness with the MOCK proposer: gates run for real, nothing is promoted, label is mock."""
    cfg = HCFG["make_config"](REPO, "policy"); cfg.tmp_parent = tmp_path
    base = L.editable_files_in_repo(REPO, cfg.scope)
    st = Store(tmp_path / "selfimprove_dryrun_policy")
    rec = L.Loop(cfg, st, PR.MockProposer([HCFG["mock_diff"]("policy", base)])).step()
    assert rec["proposer"]["mock"] and "NOT NORTH" in rec["proposer"]["label"]
    assert rec["verdict"] in ("pass_mock_not_promoted", "gate_fail:speed"), rec["verdict"]
    assert not st.attribution()["reportable_as_north_authored"]


# ---- (3) Modal evaluator glue ---------------------------------------------------------------------------------------
def _modal_module():
    pytest.importorskip("modal")
    sys.path.insert(0, str(REPO / "modal"))
    import importlib; return importlib.import_module("selfimprove_eval")


def test_modal_function_is_gpu_network_blocked_and_cost_estimate():
    m = _modal_module(); src = (REPO / "modal/selfimprove_eval.py").read_text()
    assert 'GPU = "A100-80GB"' in src and "block_network=True" in src and "gpu=GPU" in src
    assert "nvcc" in src and callable(m.evaluate_candidate.remote)
    e1, e3 = m.estimate_cost("kernel", 1), m.estimate_cost("kernel", 3, cold_start=False)
    assert 0.5 < e1["usd"] < 3 and e3["usd"] > e1["usd"] * 1.5
    assert m.estimate_cost("kernel", 1, with_proposer=False)["usd"] < e1["usd"]


def test_remote_evaluator_error_is_a_non_promoting_verdict_and_label_guard(tmp_path):
    m = _modal_module()
    def boom(_): raise RuntimeError("container died")
    out = m.RemoteEvaluator("kernel", call=boom).evaluate({"a": "1"}, {"a": "2"})
    assert out["verdict"] == "evaluator_error" and "eval_wall_s" in out
    ok = m.RemoteEvaluator("kernel", call=lambda p: {"verdict": "pass", "gates": {}, "sandbox": {}, "seeds": {}}).evaluate({}, {})
    assert ok["verdict"] == "pass" and ok["eval_wall_s"] >= 0
    # store guard: mock and North lineages never share a store
    st = Store(tmp_path / "s"); st.init_baseline({"f": "x"})
    st.promote({"f": "y"}, "p1", "mock", False, "d", {}, True)
    with pytest.raises(SystemExit, match="MOCK"):
        m._prep_store_guard(tmp_path / "s", "north")


def test_loop_with_remote_evaluator_wired_to_local_toy_evaluator(toy, tmp_path):
    m = _modal_module()
    from test_loop import cu_diff
    local = L.Evaluator(toy)
    lp = L.Loop(toy, Store(tmp_path / "s_dryrun"), PR.MockProposer([cu_diff(64, 128)]), allow_mock_promotion=False)
    lp.ev = m.RemoteEvaluator("kernel", call=lambda p: local.evaluate(p["base_files"], p["new_files"]))
    rec = lp.step()
    assert rec["verdict"] == "pass_mock_not_promoted" and rec["proposer"]["label"].startswith("MOCK")
