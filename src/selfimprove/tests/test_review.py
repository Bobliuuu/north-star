"""Adversarial-review regression tests (docs/SELFIMPROVE_REVIEW.md). Each test failed before its fix. CPU only, MOCK proposers."""
import json, shutil, sys
import pytest
from selfimprove import audit as A, diffs as D, gates as G, loop, proposer as P, sandbox as S
from selfimprove.policy import Scope, ScopeRule
from selfimprove.remote_loop import EmittingStore, pull_tree
from selfimprove.store import Store, StoreCorrupt
from .test_loop import cu_diff

SC = Scope(rules=[ScopeRule("f.cu"), ScopeRule("p.json"), ScopeRule("t.py")])


def fd(path, added):
    body = "".join(f"+{a}\n" for a in added)
    return D.parse(f"--- a/{path}\n+++ b/{path}\n@@ -1,0 +1,{len(added)} @@\n{body}", SC)


# ---- audit: multi-line / preprocessor / syscall evasions ------------------------------------------------------------------
@pytest.mark.parametrize("lines,code", [
    (["#define EE cuda\\", "EventElapsedTime"], "timing_clock"),                 # backslash splice hides the identifier
    (["auto t = std::", "chrono::steady_clock::now();"], "timing_clock"),         # tokens split across lines
    (["#include \"/proc/self/cwd/results/routing_traces.json\""], "preprocessor"),
    (["asm volatile(\".incbin \\\"x\\\"\");"], "inline_asm_or_embed"),
    (["int fd = openat(AT_FDCWD, p, 0);"], "file_io"),
    (["std::fstream f(p);"], "file_io"),
    (["long r = syscall(SYS_open, p, 0);"], "file_io"),
    (["int s = socket(AF_INET, SOCK_STREAM, 0);"], "network"),
    (["connect(s, addr, len);"], "network"),
    (["posix_spawn(&pid, sh, 0, 0, argv, 0);"], "env_or_process"),
    (["pid_t p = vfork();"], "env_or_process"),
    (["#define CAT(a,b) a##b"], "preprocessor"),
])
def test_audit_evasions_rejected(lines, code):
    f = A.audit_diff(fd("f.cu", lines), dev_only_shapes={7, 13}, bench_shapes={7, 13, 96})
    assert any(x.code == code and x.severity == "reject" for x in f), f


def test_audit_multiline_shape_specialization():
    f = A.audit_diff(fd("f.cu", ["if (n_tokens ==", "7) { fast(); }"]), dev_only_shapes={7, 13}, bench_shapes={7, 13, 96})
    assert any(x.code == "shape_specialization" and x.severity == "reject" for x in f), f


def test_audit_benign_tunable_still_clean():
    assert A.verdict(A.audit_diff(fd("f.cu", ["constexpr int KC = 128;  // k chunk"]), {7}, {7})) == "clean"


# ---- sandbox: pyc planting and build/ tampering after the build -----------------------------------------------------------
def _ws(tmp_path):
    repo = tmp_path / "r"; (repo / "harness").mkdir(parents=True); (repo / "harness/h.py").write_text("print(1)\n")
    (repo / "src").mkdir(); (repo / "src/e.cu").write_text("x\n")
    return S.Workspace.create(repo, ["harness", "src"], {}, {"src/e.cu"}, tmp_path)


def test_planted_pyc_is_a_violation(tmp_path):
    ws = _ws(tmp_path)
    pc = ws.root / "harness/__pycache__"; pc.mkdir()
    (pc / "h.cpython-312.pyc").write_bytes(b"\0" * 16)
    with pytest.raises(S.SandboxViolation):
        ws.verify()


def test_planted_pyc_outside_pycache_is_a_violation(tmp_path):
    ws = _ws(tmp_path)
    (ws.root / "harness/h.pyc").write_bytes(b"x")
    with pytest.raises(S.SandboxViolation):
        ws.verify()


def test_build_dir_frozen_after_build(tmp_path):
    ws = _ws(tmp_path)
    (ws.root / "build").mkdir(); (ws.root / "build/bin").write_bytes(b"good")
    ws.freeze_build(); ws.verify()
    (ws.root / "build/bin").write_bytes(b"swapped")
    with pytest.raises(S.SandboxViolation):
        ws.verify()


def test_new_file_in_build_after_freeze_is_a_violation(tmp_path):
    ws = _ws(tmp_path); ws.freeze_build()
    (ws.root / "build").mkdir(); (ws.root / "build/extra").write_bytes(b"x")
    with pytest.raises(S.SandboxViolation):
        ws.verify()


# ---- nonce protocol: forged / duplicate lines, non-zero exit ----------------------------------------------------------------
def test_two_nonce_lines_are_ambiguous():
    r = S.RunResult(["x"], 0, '{"time_ms": 0.001, "nonce": "n"}\n{"time_ms": 10, "nonce": "n"}\n', "", 0.1, False)
    assert r.json("n") is None


def test_nonzero_exit_result_not_trusted(toy):
    ev = loop.Evaluator(toy)
    ws = S.Workspace.create(toy.repo, toy.subdirs, {}, set(), toy.tmp_parent)
    try:
        cmd = [sys.executable, "-c", "import os,json,sys; n=os.environ['NS_NONCE']; print(json.dumps({'time_ms':0.001,'nonce':n})); sys.exit(3)"]
        r = ev._run(ws, cmd, 5)
        assert r.rc == 3 and r.parsed is None
    finally:
        ws.cleanup()


# ---- evaluator robustness ----------------------------------------------------------------------------------------------------
def test_broken_baseline_is_baseline_broken(toy):
    base = {"src/kvm/moe_fused.cu": (toy.repo / "src/kvm/moe_fused.cu").read_text(),
            "src/opt/policy_tables/sched.json": "{not json"}              # baseline harness crashes -> no parsable baseline
    new = dict(base, **{"src/opt/policy_tables/sched.json": '{\n "skip_tail": 0,\n "dev_only_fast": false\n}\n'})
    out = loop.Evaluator(toy).evaluate(base, new)
    assert out["verdict"] == "baseline_broken", out


def test_evaluator_exception_is_a_verdict_not_a_crash(toy):
    base = {"src/kvm/moe_fused.cu": (toy.repo / "src/kvm/moe_fused.cu").read_text(),
            "src/opt/policy_tables/sched.json": (toy.repo / "src/opt/policy_tables/sched.json").read_text()}
    orig = G.correctness_gate
    G.correctness_gate = lambda *a, **k: (_ for _ in ()).throw(KeyError("rel_l2"))
    try:
        out = loop.Evaluator(toy).evaluate(base, dict(base))
    finally:
        G.correctness_gate = orig
    assert out["verdict"] == "evaluator_error", out


class FakeEval:
    def __init__(self, out): self.out = out
    def evaluate(self, b, n): return dict(self.out)


@pytest.mark.parametrize("out", [
    dict(verdict="pass", gates={}),
    dict(verdict="pass", gates={"correctness": dict(passed=True), "quality": dict(passed=True)}),
    dict(verdict="pass", gates={"correctness": dict(passed=True), "quality": dict(passed=True), "speed": dict(passed=False)}),
    dict(gates={}),
])
def test_promotion_requires_every_gate_recorded_and_passed(toy, tmp_path, out):
    st = Store(tmp_path / "s")
    lp = loop.Loop(toy, st, P.MockProposer([cu_diff(64, 128)]), allow_mock_promotion=True)
    lp.ev = FakeEval(out)
    rec = lp.step()
    assert rec["verdict"] != "promoted" and st.current() == "v0000", rec


# ---- store: integrity of versions/CURRENT vs the chained log ----------------------------------------------------------------
def _promoted(toy, tmp_path, name="s"):
    st = Store(tmp_path / name)
    lp = loop.Loop(toy, st, P.MockProposer([cu_diff(64, 128)]), allow_mock_promotion=True)
    assert lp.step()["verdict"] == "promoted"
    return st


def test_tampered_version_file_detected(toy, tmp_path):
    st = _promoted(toy, tmp_path)
    p = st.root / "versions" / st.current() / "files/src/kvm/moe_fused.cu"
    p.write_text(p.read_text().replace("SL = 128", "SL = 256"))
    with pytest.raises(StoreCorrupt):
        st.assert_healthy()


def test_tampered_meta_attribution_detected(toy, tmp_path):
    st = _promoted(toy, tmp_path)
    mp = st.root / "versions" / st.current() / "meta.json"
    m = json.loads(mp.read_text()); m["authored_by"] = "north"; m["north_attested"] = True; m["mock"] = False
    mp.write_text(json.dumps(m, indent=1))
    with pytest.raises(StoreCorrupt):
        st.assert_healthy()


def test_current_forged_ahead_of_log_detected(toy, tmp_path):
    st = _promoted(toy, tmp_path)
    st.rollback(reason="t")
    (st.root / "CURRENT").write_text(st.versions()[-1] + "\n")        # point CURRENT at a version the log does not justify
    with pytest.raises(StoreCorrupt):
        st.assert_healthy()


def test_loop_refuses_corrupt_store(toy, tmp_path):
    st = _promoted(toy, tmp_path)
    p = st.root / "versions" / st.current() / "files/src/kvm/moe_fused.cu"
    p.write_text(p.read_text() + "\n// x")
    with pytest.raises(StoreCorrupt):
        loop.Loop(toy, st, P.MockProposer(["x"]))


def test_verify_log_garbage_returns_false_and_head_anchor(toy, tmp_path):
    st = _promoted(toy, tmp_path)
    head = st.head()
    assert st.verify_log(expected_head=head)
    recs = st.read_log()[:-1]                                           # truncate the tail: chain still self-consistent
    st.log_path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    assert st.verify_log() and not st.verify_log(expected_head=head)
    with open(st.log_path, "a") as f: f.write("not json\n")
    assert st.verify_log() is False


def test_rollback_refuses_corrupt_target(toy, tmp_path):
    st = _promoted(toy, tmp_path)
    p = st.root / "versions/v0000/files/src/kvm/moe_fused.cu"
    p.write_text(p.read_text() + "\n// x")
    with pytest.raises(StoreCorrupt):
        st.rollback(reason="t")


def test_every_volume_commit_snapshot_is_consistent(toy, tmp_path):
    """The Volume snapshot at EVERY commit point must satisfy: CURRENT is never ahead of the chained log."""
    snaps = []
    def on_log():
        d = tmp_path / f"snap{len(snaps)}"; shutil.copytree(tmp_path / "s", d); snaps.append(d)
    st = EmittingStore(tmp_path / "s", emit=lambda l: None, on_log=on_log)
    lp = loop.Loop(toy, st, P.MockProposer([cu_diff(64, 128)]), allow_mock_promotion=True)
    assert lp.step()["verdict"] == "promoted"
    st.rollback(reason="t")
    assert len(snaps) >= 6
    for d in snaps:
        if (d / "CURRENT").exists() or (d / "proposals.jsonl").exists():
            Store(d).assert_healthy(allow_pending=True)


def test_recover_completes_logged_but_unapplied_promotion(toy, tmp_path):
    st = _promoted(toy, tmp_path)
    v = st.current()
    (st.root / "CURRENT").write_text("v0000\n")                         # crash between log append and CURRENT switch
    st.assert_healthy(allow_pending=True)
    with pytest.raises(StoreCorrupt):
        st.assert_healthy()
    assert st.recover() is True and st.current() == v
    st.assert_healthy()


# ---- pull: stage + verify before touching the destination; no traversal --------------------------------------------------
def _vol(tmp_path, toy):
    src = tmp_path / "vol" / "s"
    st = EmittingStore(src, emit=lambda l: None)
    loop.Loop(toy, st, P.MockProposer([cu_diff(64, 128)]), allow_mock_promotion=True).step()
    listdir = lambda p: [(f"{p}/{c.name}", c.is_dir()) for c in (tmp_path / "vol" / p).iterdir()]
    read = lambda p: (tmp_path / "vol" / p).read_bytes()
    return src, listdir, read


def test_pull_with_forged_tail_does_not_clobber_local(toy, tmp_path):
    src, listdir, read = _vol(tmp_path, toy)
    dest = tmp_path / "local"
    pull_tree(listdir, read, "s", dest)
    before = {p.relative_to(dest).as_posix(): p.read_bytes() for p in dest.rglob("*") if p.is_file()}
    forged = dict(event="promote", version="v0099", prev_hash="0" * 64, hash="f" * 64)
    with open(src / "proposals.jsonl", "a") as f: f.write(json.dumps(forged) + "\n")
    with pytest.raises(SystemExit):
        pull_tree(listdir, read, "s", dest)
    after = {p.relative_to(dest).as_posix(): p.read_bytes() for p in dest.rglob("*") if p.is_file()}
    assert before == after


def test_pull_rejects_path_traversal(toy, tmp_path):
    src, listdir, read = _vol(tmp_path, toy)
    evil = lambda p: listdir(p) + ([("s/../evil.txt", False)] if p == "s" else [])
    (tmp_path / "vol" / "evil.txt").write_text("x")
    dest = tmp_path / "local"
    with pytest.raises(SystemExit):
        pull_tree(evil, read, "s", dest)
    assert not (tmp_path / "evil.txt").exists()
