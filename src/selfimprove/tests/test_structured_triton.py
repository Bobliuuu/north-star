"""Track 'triton': structured-edit parse/render, scope, audit, prompt hygiene, and defaults == frozen reference config. MOCK only, no GPU."""
import json, pathlib, runpy
import pytest
from selfimprove import structured_triton as STR, diffs as D, audit as A, loop as L, proposer as P
from selfimprove.policy import Scope, ScopeRule, Limits

REPO = pathlib.Path(__file__).resolve().parents[3]
PATH = "src/tritonk/tuning.py"
SCOPE = Scope([ScopeRule(PATH, regions_only=True)], Limits(max_files=1, max_hunks=4, max_changed_lines=16))
HCFG = runpy.run_path(str(REPO / "harness/selfimprove/config.py"))


@pytest.fixture(scope="module")
def real():
    return {PATH: (REPO / PATH).read_text()}


def j(edits):
    return json.dumps(dict(edits=[dict(name=n, value=v) for n, v in edits], rationale="r"))


def test_defaults_match_frozen_reference_and_all_names_tunable(real):
    ns = runpy.run_path(str(REPO / PATH))
    cur = {k: v for k, v in ns.items() if k.isupper()}
    import re
    suite = (REPO / "harness/selfimprove/triton_suite.py").read_text()
    frozen = eval(re.search(r"DEFAULTS = (dict\(.*?\))   # FROZEN", suite, re.S).group(1))
    assert cur == frozen
    assert set(cur) == set(STR.TUNABLES)
    for n, v in cur.items(): assert v in STR.TUNABLES[n][0], n
    assert STR.current_values(real) == cur


def test_valid_edit_roundtrip_through_strict_parser_and_audit(real):
    pe = STR.parse_edits(j([("GU_BK", 128), ("ATT_NSPLIT", 64), ("XA_NS", 4)]))
    d = STR.render_diff(pe["edits"], real)
    fds = D.parse(d, SCOPE); new = D.apply_all(fds, real.get, SCOPE)[PATH]
    assert "GU_BK = 128" in new and "ATT_NSPLIT = 64" in new and "XA_NS = 4" in new
    assert A.verdict(A.audit_diff(fds, set(), set())) == "clean"
    assert new.count("\n") == real[PATH].count("\n")


def test_every_tunable_every_value_passes_audit(real):
    for n, (vals, _) in STR.TUNABLES.items():
        for v in vals:
            try: d = STR.render_diff([(n, v)], real)
            except STR.StructuredEditError: continue            # cap violations are rejected earlier
            if d: assert A.verdict(A.audit_diff(D.parse(d, SCOPE), set(), set())) == "clean", (n, v)


@pytest.mark.parametrize("raw,frag", [
    ('{"edits":[{"name":"GU_BK","value":100}]}', "value_not_allowed"),
    ('{"edits":[{"name":"GU_BK","value":"128"}]}', "value_not_int"),
    ('{"edits":[{"name":"DN_BK","value":512}]}', "value_not_allowed"),
    ('{"edits":[{"name":"EVIL","value":1}]}', "unknown_name"),
    ('{"edits":[{"name":"GU_BK","value":128},{"name":"GU_BK","value":64}]}', "duplicate_name"),
    ('{"edits":[{"name":"GU_BK","value":128}],"x":1}', "unknown_keys"),
    ('{"edits":[' + ",".join('{"name":"%s","value":%d}' % (n, STR.TUNABLES[n][0][0]) for n in list(STR.TUNABLES)[:5]) + ']}', "too_many_edits"),
])
def test_invalid_edits(raw, frag):
    with pytest.raises(STR.StructuredEditError) as e: STR.parse_edits(raw)
    assert frag in str(e.value)


def test_tile_cap_and_noop(real):
    with pytest.raises(STR.StructuredEditError): STR.render_diff([("GU_BN", 256), ("GU_BK", 1024)], real)
    assert STR.render_diff([("GU_BK", 256)], real) == ""


def test_editing_outside_region_impossible(real):
    bad = real[PATH].replace("# NS-EDITABLE-END", "# NS-EDITABLE-END\nGU_BK = 256")
    with pytest.raises(STR.StructuredEditError): STR.render_diff([("GU_BK", 128)], {PATH: bad})


def test_prompt_has_no_heldout_and_config_builds():
    cfg = HCFG["make_config"](REPO, "triton")
    base = L.editable_files_in_repo(REPO, cfg.scope)
    assert list(base) == [PATH]
    req = P.ProposalRequest(base_version="v0000", files=base, scope_description=L.scope_description(cfg.scope), profile_summary=HCFG["profile_summary"]("triton"),
                            dev_results=HCFG["dev_results"]("triton"), history=[], mode="structured", track="triton")
    pr = P.build_prompt(req)
    for held in (1500, 5000, 12000, 3000, 10000, 3072, 6144, 131072, 2560):
        assert str(held) not in pr, held
    assert "GU_BK" in pr and "SYSTEM" not in pr
    assert "triton" in P.system_for("structured", "triton").lower()


def test_mock_loop_step_runs_end_to_end_until_evaluator(tmp_path):
    """Mock proposer through the real Loop up to (stubbed) evaluation: labeled mock, rendered diff touches one line."""
    from selfimprove.store import Store
    cfg = HCFG["make_config"](REPO, "triton")
    lp = L.Loop(cfg, Store(tmp_path / "s"), P.MockProposer([HCFG["mock_structured"]("triton")]), mode="structured", track="triton")
    lp.ev = type("E", (), {"evaluate": lambda self, b, n: dict(verdict="gate_fail:speed", gates={}, sandbox={}, seeds={}, eval_wall_s=1.0)})()
    r = lp.step(profile_summary="p", dev_results="d")
    assert r["verdict"] == "gate_fail:speed" and r["proposer"]["mock"] and "GU_BK = 128" in r["diff"]
