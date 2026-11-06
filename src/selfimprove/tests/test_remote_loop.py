"""Remote-loop plumbing: emitted JSON lines, commit hooks, store equivalence, pull. CPU only, MOCK proposer (nothing is North's)."""
import json
import pytest
from selfimprove import loop, proposer as P
from selfimprove.remote_loop import run_remote_loop, pull_tree, LOG_PREFIX
from selfimprove.store import Store
from .test_loop import cu_diff

P_ = dict(rounds=2, max_usd=5.0, mode="diff", proposer="mock")


def _run(toy, d, lines, commits, raw):
    return run_remote_loop(P_, d, toy, P.MockProposer([raw]), loop.Evaluator(toy), emit=lines.append,
                           on_log=lambda: commits.append(1))


def test_emits_every_record_and_commits(toy, tmp_path):
    lines, commits = [], []
    out = _run(toy, tmp_path / "s", lines, commits, cu_diff(64, 128))
    st = Store(tmp_path / "s")
    emitted = [json.loads(l[len(LOG_PREFIX):]) for l in lines if l.startswith(LOG_PREFIX)]
    assert emitted == st.read_log()                       # init + 2 proposals, identical to the durable chained log
    assert len(emitted) == 3 and out["log_ok"] and st.verify_log()
    assert all(v == "pass_mock_not_promoted" for _, v in out["verdicts"])
    assert len(commits) >= 3 and lines[-1].startswith("NS_LOOP_DONE")


def test_gate_semantics_unchanged_vs_plain_loop(toy, tmp_path):
    _run(toy, tmp_path / "r", [], [], cu_diff(64, 96))
    plain = Store(tmp_path / "p")
    loop.Loop(toy, plain, P.MockProposer([cu_diff(64, 96)])).run(2, 5.0)
    strip = lambda s: [(r.get("event"), r.get("verdict")) for r in s.read_log()]
    assert strip(Store(tmp_path / "r")) == strip(plain)


def test_failing_proposal_logged_and_emitted(toy, tmp_path):
    lines = []
    _run(toy, tmp_path / "s", lines, [], "no diff here")
    recs = [json.loads(l[len(LOG_PREFIX):]) for l in lines if l.startswith(LOG_PREFIX)]
    assert [r.get("verdict") for r in recs[1:]] == ["no_diff_in_response"] * 2


def test_store_guard_mixed_lineage(toy, tmp_path):
    d = tmp_path / "s"
    _run(toy, d, [], [], cu_diff(64, 128))
    st = Store(d)
    st.promote({"src/kvm/moe_fused.cu": "x"}, "p", "mock", False, "d", {}, True)
    with pytest.raises(SystemExit):
        run_remote_loop(dict(P_, proposer="north"), d, toy, P.MockProposer(["x"]), loop.Evaluator(toy), emit=lambda l: None)


def test_pull_tree_roundtrip_and_divergence(toy, tmp_path):
    src = tmp_path / "vol" / "s"
    _run(toy, src, [], [], cu_diff(64, 128))

    def listdir(p):
        base = tmp_path / "vol" / p
        return [(f"{p}/{c.name}", c.is_dir()) for c in base.iterdir()]
    read = lambda p: (tmp_path / "vol" / p).read_bytes()
    dest = tmp_path / "local"
    got = pull_tree(listdir, read, "s", dest)
    assert "proposals.jsonl" in got and "CURRENT" in got and any(g.startswith("versions/v0000/") for g in got)
    assert Store(dest).verify_log() and Store(dest).read_log() == Store(src).read_log()
    pull_tree(listdir, read, "s", dest)                   # idempotent re-pull
    (dest / "proposals.jsonl").write_text("diverged\n")
    with pytest.raises(SystemExit):
        pull_tree(listdir, read, "s", dest)
    with pytest.raises(SystemExit):
        pull_tree(lambda p: [], read, "nothing", tmp_path / "z")
