"""Structured-edit mode, POLICY track (spec_space.json): parsing/rendering, equivalence with the harness table loader, prompt hygiene,
and end-to-end through the unchanged diff parser / audit / loop with MOCK proposers (gates stubbed; no GPU, no network)."""
import importlib.util, json, os, pathlib, sys
import pytest
from selfimprove import structured_policy as SP, diffs as D, loop, proposer as P, audit as A
from selfimprove.policy import Scope, ScopeRule, Limits
from selfimprove.store import Store

REPO = pathlib.Path(__file__).resolve().parents[3]
PATH = SP.POLICY_PATH
SCOPE = Scope([ScopeRule(PATH)], Limits(max_files=1, max_hunks=3, max_changed_lines=12))      # == harness make_config("policy")
REAL = {PATH: (REPO / PATH).read_text()}


def j(edits, **kw):
    return json.dumps(dict(edits=[dict(name=n, value=v) for n, v in edits], rationale="r", **kw))


def err(raw):
    with pytest.raises(SP.StructuredEditError) as e: SP.parse_edits(raw)
    return str(e.value)


def test_constants_mirror_optimizer_and_harness():
    sys.path.insert(0, str(REPO / "src"))
    from opt import config as C
    assert SP.MODES == C.MODES
    cfg = (REPO / "harness/selfimprove/config.py").read_text()
    assert 'ScopeRule("src/opt/policy_tables/spec_space.json")' in cfg and "max_hunks=3, max_changed_lines=12" in cfg


def test_valid_roundtrip_through_strict_parser():
    pe = SP.parse_edits(j([("gammas", [8, 1, 4]), ("modes", ["fused", "two_pass"]), ("n_groups", 8)]))
    assert pe["edits"] == [("gammas", [1, 4, 8]), ("modes", ["two_pass", "fused"]), ("n_groups", 8)]      # set semantics: normalised order
    d = SP.render_diff(pe["edits"], REAL)
    new = D.apply_all(D.parse(d, SCOPE), REAL.get, SCOPE)[PATH]
    assert json.loads(new) == {"gammas": [1, 4, 8], "modes": ["two_pass", "fused"], "n_groups": 8}
    changed = [l for l in d.splitlines() if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    assert len(changed) == 6                                                              # only the three value lines
    assert new.count("\n") == REAL[PATH].count("\n")


def test_noop_edit_renders_empty():
    assert SP.render_diff([("gammas", [1, 2, 3, 4, 6, 8]), ("n_groups", 6)], REAL) == ""
    assert SP.render_diff([], REAL) == ""


def test_empty_edits_and_fences_and_think_tolerated():
    assert SP.parse_edits('{"edits":[],"rationale":"none"}')["edits"] == []
    assert SP.parse_edits('<think>{"x":1}</think>\n```json\n' + j([("n_groups", 4)]) + "\n```")["edits"] == [("n_groups", 4)]


@pytest.mark.parametrize("raw,frag", [
    ("no json here", "no_json_object"),
    ("{bad", "no_json_object"),
    ('{"edits":[{"name":"gammas","value":[NaN]}]}', "bad_json"),
    ("[1]", "no_json_object"),
    ('{"edits":[],"extra":1}', "unknown_keys"),
    ('{"edits":"x"}', "edits_not_a_list"),
    ('{"edits":[],"rationale":5}', "rationale_not_a_string"),
    ('{"edits":[1]}', "edit_not_an_object"),
    ('{"edits":[{"name":"gammas"}]}', "edit_keys"),
    ('{"edits":[{"name":"gammas","value":[1],"x":1}]}', "edit_keys"),
    ('{"edits":[{"name":"evil","value":1}]}', "unknown_name"),
    ('{"edits":[{"name":["gammas"],"value":[1]}]}', "unknown_name"),
    ('{"edits":[{"name":"gammas","value":[1]},{"name":"gammas","value":[2]}]}', "duplicate_name"),
    ('{"edits":[{"name":"gammas","value":[]}]}', "gammas_not_a_nonempty_list"),
    ('{"edits":[{"name":"gammas","value":4}]}', "gammas_not_a_nonempty_list"),
    ('{"edits":[{"name":"gammas","value":[1,"2"]}]}', "gammas_value_not_int"),
    ('{"edits":[{"name":"gammas","value":[1.0]}]}', "gammas_value_not_int"),
    ('{"edits":[{"name":"gammas","value":[true]}]}', "gammas_value_not_int"),
    ('{"edits":[{"name":"gammas","value":[0]}]}', "gammas_out_of_range"),
    ('{"edits":[{"name":"gammas","value":[13]}]}', "gammas_out_of_range"),
    ('{"edits":[{"name":"gammas","value":[2,2]}]}', "gammas_duplicate"),
    ('{"edits":[{"name":"modes","value":[]}]}', "modes_not_a_nonempty_list"),
    ('{"edits":[{"name":"modes","value":["rm -rf"]}]}', "modes_unknown"),
    ('{"edits":[{"name":"modes","value":[1]}]}', "modes_unknown"),
    ('{"edits":[{"name":"modes","value":["fused","fused"]}]}', "modes_duplicate"),
    ('{"edits":[{"name":"n_groups","value":1}]}', "n_groups_out_of_range"),
    ('{"edits":[{"name":"n_groups","value":13}]}', "n_groups_out_of_range"),
    ('{"edits":[{"name":"n_groups","value":"6"}]}', "n_groups_not_int"),
    ('{"edits":[{"name":"n_groups","value":6.0}]}', "n_groups_not_int"),
    ('{"edits":[{"name":"n_groups","value":true}]}', "n_groups_not_int"),
])
def test_rejections(raw, frag):
    assert frag in err(raw)


def test_too_many_edits_and_too_long_gammas():
    ed = json.dumps(dict(edits=[dict(name="n_groups", value=4)] * 4, rationale=""))
    assert "too_many_edits" in err(ed)
    assert "gammas_too_long" in err(j([("gammas", list(range(1, 13)) + [1])]))
    SP.parse_edits(j([("gammas", list(range(1, 13)))]))                                      # 12 is allowed


def test_protected_keys_unreachable_and_format_changes_fail_closed():
    with pytest.raises(SP.StructuredEditError, match="not_editable"):
        SP.render_diff([("evil", 1)], REAL)
    with pytest.raises(SP.StructuredEditError, match="file_missing"):
        SP.render_diff([("n_groups", 3)], {})
    with pytest.raises(SP.StructuredEditError, match="not_editable"):                       # key line not in the expected one-line form
        SP.render_diff([("gammas", [1])], {PATH: '{"gammas": [1,\n 2], "modes": ["fused"], "n_groups": 6}\n'})


def test_every_accepted_edit_is_accepted_by_the_harness_table_loader(tmp_path, monkeypatch):
    """The structured validator must be no looser than harness policy_eval.load_table (the evaluator's own whitelist)."""
    monkeypatch.setenv("NS_NONCE", "x"); monkeypatch.setenv("NS_WORKSPACE", str(REPO))
    monkeypatch.syspath_prepend(str(REPO / "harness/selfimprove")); monkeypatch.syspath_prepend(str(REPO / "src"))
    sys.modules.pop("common", None)
    spec = importlib.util.spec_from_file_location("policy_eval_t", REPO / "harness/selfimprove/policy_eval.py")
    pe = importlib.util.module_from_spec(spec); spec.loader.exec_module(pe)
    cases = [[("gammas", [12, 1])], [("gammas", list(range(1, 13)))], [("modes", ["fused"])], [("modes", ["fused", "spec_two_pass", "two_pass"])],
             [("n_groups", 2)], [("n_groups", 12)], [("gammas", [3]), ("modes", ["spec_two_pass"]), ("n_groups", 5)]]
    for edits in cases:
        pe_ = SP.parse_edits(j(edits))
        d = SP.render_diff(pe_["edits"], REAL)
        new = D.apply_all(D.parse(d, SCOPE), REAL.get, SCOPE)[PATH] if d else REAL[PATH]
        f = tmp_path / "t.json"; f.write_text(new); pe.TABLE = str(f)
        t, _ = pe.load_table()
        assert t == json.loads(new)
    sys.modules.pop("common", None)


def test_audit_does_not_flag_a_plain_policy_edit():
    pe = SP.parse_edits(j([("gammas", [1, 2, 3, 4, 5, 6, 8])]))
    fds = D.parse(SP.render_diff(pe["edits"], REAL), SCOPE)
    runpy = __import__("runpy")
    H = runpy.run_path(str(REPO / "harness/selfimprove/config.py"))
    cfg = H["make_config"](REPO, "policy")
    assert A.verdict(A.audit_diff(fds, cfg.dev_shapes - cfg.heldout_shapes, cfg.dev_shapes)) == "clean"


def test_prompt_lists_current_values_labels_predicted_and_no_heldout():
    req = P.ProposalRequest(base_version="v0", files=REAL, scope_description="s", dev_results="Dev buckets", mode="structured", track="policy",
                            history=["abc12345: gate_fail:speed | edits gammas=[1, 2] | PREDICTED dev speedup 0.9x"])
    p = P.build_prompt(req)
    for n in SP.KEYS: assert f"- {n} = " in p
    assert '"gammas": [1, 2, 3, 4, 6, 8]' in p and "PREDICTED" in p and "NOT a measured speedup" in p
    assert "Earlier proposals this session" in p and "abc12345" in p
    for h in ("5000", "12000", "20000", "3072", "identical"): assert h not in p
    assert "KC" not in p and "MT_SPLIT" not in p
    s = P.system_for("structured", "policy"); assert "policy search space" in s and "diff" in s.lower()
    assert P.system_for("structured", "kernel") == P.SYSTEM_STRUCTURED and P.system_for("diff", "policy") == P.SYSTEM


def test_kernel_prompt_unchanged_default_track():
    cu = {"src/kvm/moe_fused.cu": (REPO / "src/kvm/moe_fused.cu").read_text()}
    p = P.build_prompt(P.ProposalRequest("v0", cu, "s", mode="structured"))
    assert "KC" in p and "spec_space" not in p


# ---- end to end through the Loop (policy track, structured mode default path) --------------------------------------------------------
class _Ev:
    """Stub evaluator: returns a scripted verdict + a speed-gate record so history feeds can be checked."""
    def __init__(self, sp=1.0): self.sp = sp; self.calls = 0
    def evaluate(self, base, new):
        self.calls += 1
        g = {"speed": {"passed": False, "data": {"dev": {"valid": True, "speedup": self.sp, "ci_lo": self.sp, "threshold": 1.03}}}}
        return dict(verdict="gate_fail:speed", gates=g, sandbox={}, seeds={}, eval_wall_s=0.0)


def _loop(tmp_path, responses):
    H = __import__("runpy").run_path(str(REPO / "harness/selfimprove/config.py"))
    cfg = H["make_config"](REPO, "policy")
    st = Store(tmp_path / "selfimprove_dryrun_policy")
    lp = loop.Loop(cfg, st, P.MockProposer(responses), mode="structured", track="policy")
    lp.ev = _Ev(); return lp, st


def test_loop_policy_structured_end_to_end_and_history_feed(tmp_path):
    lp, st = _loop(tmp_path, [j([("gammas", [1, 2, 3, 4, 5, 6, 8])]), j([("n_groups", 4)])])
    r1 = lp.step()
    assert r1["verdict"] == "gate_fail:speed" and r1["proposer"]["mock"] and "NOT NORTH" in r1["proposer"]["label"]
    assert r1["structured"]["edits"] == [dict(name="gammas", value=[1, 2, 3, 4, 5, 6, 8])]
    assert r1["diff"] and "+ \"gammas\": [1, 2, 3, 4, 5, 6, 8]," in r1["diff"]
    r2 = lp.step()
    assert "PREDICTED dev speedup 1.000x" in r2["prompt"] and "gammas=[1, 2, 3, 4, 5, 6, 8]" in r2["prompt"]    # measured-history feed
    assert st.verify_log() and not st.attribution()["reportable_as_north_authored"]


def test_loop_policy_structured_invalid_noop_and_prior_history(tmp_path):
    lp, st = _loop(tmp_path, ['{"edits":[{"name":"gammas","value":[99]}]}', '{"edits":[],"rationale":"x"}'])
    lp.prior_history = ["old1: gate_fail:speed | edits n_groups=3 | PREDICTED dev speedup 0.990x"]
    a = lp.step(); assert a["verdict"] == "invalid_edit" and "gammas_out_of_range" in a["reason"] and lp.ev.calls == 0
    assert "old1: gate_fail:speed" in a["prompt"]
    b = lp.step(); assert b["verdict"] == "no_change" and lp.ev.calls == 0
    with pytest.raises(ValueError): loop.Loop(lp.cfg, st, P.MockProposer([""]), mode="structured", track="bogus")


def test_cli_and_modal_default_to_structured_for_policy():
    cli = (REPO / "src/selfimprove/cli.py").read_text()
    assert '"--mode", choices=["structured", "diff"], default="structured"' in cli and '"--track"' in cli
    m = (REPO / "modal/selfimprove_eval.py").read_text()
    assert 'mode = mode or "structured"' in m and "kernel track only" not in m and "track=track" in m
    H = __import__("runpy").run_path(str(REPO / "harness/selfimprove/config.py"))
    assert SP.parse_edits(H["mock_structured"]("policy"))["edits"] and "KC" in H["mock_structured"]("kernel")
