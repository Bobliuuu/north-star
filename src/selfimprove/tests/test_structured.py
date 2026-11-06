"""Structured-edit mode: adversarial parsing/rendering + end-to-end through the unchanged diff parser/audit/gates. MOCK proposers only."""
import json, pathlib
import pytest
from selfimprove import structured as ST, diffs as D, loop, proposer as P, NORTH_MODEL_ID, NORTH_REVISION
from selfimprove.policy import Scope, ScopeRule, Limits
from selfimprove.store import Store

REAL = pathlib.Path(__file__).resolve().parents[2] / "kvm" / "moe_fused.cu"
SCOPE = Scope([ScopeRule("src/kvm/moe_fused.cu", regions_only=True)], Limits(max_files=1, max_hunks=4, max_changed_lines=30))
PATH = "src/kvm/moe_fused.cu"

@pytest.fixture(scope="module")
def real():
    return {PATH: REAL.read_text()}

def j(edits, **kw):
    return json.dumps(dict(edits=[dict(name=n, value=v) for n, v in edits], rationale="r", **kw))

def err(raw):
    with pytest.raises(ST.StructuredEditError) as e: ST.parse_edits(raw)
    return str(e.value)

def test_valid_roundtrip_through_strict_parser(real):
    pe = ST.parse_edits(j([("KC", 128), ("XPAD", 16), ("MINB_PER_SM", 2)]))
    d = ST.render_diff(pe["edits"], real)
    fds = D.parse(d, SCOPE); new = D.apply_all(fds, real.get, SCOPE)[PATH]
    assert "constexpr int KC = 128;" in new and "constexpr int XPAD = 16;" in new and "constexpr int MINB_PER_SM = 2;" in new
    assert "// k chunk of H" in new                                   # trailing comments preserved
    assert new.count("\n") == real[PATH].count("\n")
    changed = [l for l in d.splitlines() if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    assert len(changed) == 6                                          # only the three constant lines

def test_hunk_counts_always_consistent(real):
    for edits in ([("KC", 64)], [("KC", 64), ("MINB_PER_SM", 3)], [(n, v) for n, v in (("KC", 32), ("XPAD", 0), ("HPAD", 0), ("MT_SPLIT", 4), ("MINB_PER_SM", 8), ("UNROLL_KS", 2))]):
        D.parse(ST.render_diff(edits, real), SCOPE)                  # would raise 'hunk line counts do not match header'

def test_noop_edit_renders_empty(real):
    assert ST.render_diff([("KC", 256), ("XPAD", 8)], real) == ""     # current values
    assert ST.render_diff([], real) == ""

def test_extraction_variants():
    ok = j([("KC", 128)])
    for raw in (ok, f"```json\n{ok}\n```", f"Sure!\n```\n{ok}\n```\nthanks", f"<think>{{\"edits\":[{{\"name\":\"HPAD\",\"value\":8}}]}}</think>{ok}"):
        assert ST.parse_edits(raw)["edits"] == [("KC", 128)]
    assert ST.parse_edits(j([("KC", 128)]).replace("r\"", "has } brace {\""))["edits"] == [("KC", 128)]   # string-aware
    assert ST.parse_edits(f"{ok} {j([('KC', 64)])}")["edits"] == [("KC", 128)]                              # FIRST object only

@pytest.mark.parametrize("raw,frag", [
    ("no json at all", "no_json_object"),
    ('{"edits":[{"name":"KC","value":128}', "no_json_object"),         # truncated
    ("{'edits': []}", "bad_json"),
    ('[{"name":"KC","value":1}]', "unknown_keys"),   # first object found is the inner dict: still rejected
    ('{"edits":[],"rationale":"x","extra":1}', "unknown_keys"),
    ('{"edits":"KC=128"}', "edits_not_a_list"),
    ('{"edits":[{"name":"KC","value":128,"why":"x"}]}', "edit_keys"),
    ('{"edits":[{"name":"KC"}]}', "edit_keys"),
    ('{"edits":[{"name":"MAXTOK","value":16}]}', "unknown_name"),
    ('{"edits":[{"name":"SL","value":32}]}', "unknown_name"),
    ('{"edits":[{"name":"kc","value":128}]}', "unknown_name"),
    ('{"edits":[{"name":"KC; system(0)","value":128}]}', "unknown_name"),
    ('{"edits":[{"name":["KC"],"value":128}]}', "unknown_name"),
    ('{"edits":[{"name":"KC","value":128},{"name":"KC","value":64}]}', "duplicate_name"),
    ('{"edits":[{"name":"KC","value":"128"}]}', "value_not_int"),
    ('{"edits":[{"name":"KC","value":128.0}]}', "value_not_int"),
    ('{"edits":[{"name":"KC","value":true}]}', "value_not_int"),
    ('{"edits":[{"name":"KC","value":null}]}', "value_not_int"),
    ('{"edits":[{"name":"KC","value":NaN}]}', "bad_json"),
    ('{"edits":[{"name":"KC","value":4096}]}', "out_of_range"),
    ('{"edits":[{"name":"KC","value":0}]}', "out_of_range"),
    ('{"edits":[{"name":"KC","value":-16}]}', "out_of_range"),
    ('{"edits":[{"name":"KC","value":100}]}', "not_multiple_of_16"),
    ('{"edits":[{"name":"KC","value":48}]}', "KC_must_divide"),
    ('{"edits":[{"name":"XPAD","value":4}]}', "not_multiple_of_8"),
    ('{"edits":[{"name":"HPAD","value":72}]}', "out_of_range"),
    ('{"edits":[{"name":"MT_SPLIT","value":17}]}', "out_of_range"),
    ('{"edits":[{"name":"MINB_PER_SM","value":9}]}', "out_of_range"),
    ('{"edits":[{"name":"UNROLL_KS","value":0}]}', "out_of_range"),
    ('{"edits":[{"name":"UNROLL_KS","value":129}]}', "out_of_range"),
    ('{"edits":[{"name":"MINB_PER_SM","value":0}]}', "out_of_range"),
    ('{"edits":[],"rationale":5}', "rationale_not_a_string"),
])
def test_rejections(raw, frag):
    assert frag in err(raw)

def test_too_many_edits():
    ed = [("KC", 128), ("XPAD", 0), ("HPAD", 0), ("MT_SPLIT", 4), ("MINB_PER_SM", 2), ("UNROLL_KS", 4)]
    ST.parse_edits(j(ed))
    assert "too_many_edits" in err(json.dumps(dict(edits=[dict(name=n, value=v) for n, v in ed] + [dict(name="KC", value=64)], rationale="")))

def test_smem_overflow_rejected_before_build(real):
    with pytest.raises(ST.StructuredEditError, match="static_smem"):
        ST.render_diff([("KC", 1024), ("XPAD", 64)], real)            # 32*(1088)+... > 48KB

def test_protected_line_unreachable(real):
    # MAXTOK sits outside the region: even a hand-forced edit cannot be rendered, and a diff touching it is rejected by the parser
    with pytest.raises(ST.StructuredEditError, match="not_editable"):
        ST.render_diff([("MAXTOK", 16)], real)
    bad = ("--- a/src/kvm/moe_fused.cu\n+++ b/src/kvm/moe_fused.cu\n@@ -44,1 +44,1 @@\n"
           "-constexpr int MAXTOK = 32;             // max tokens routed to one expert (Tv+Td <= 32)\n"
           "+constexpr int MAXTOK = 16;             // max tokens routed to one expert (Tv+Td <= 32)\n")
    with pytest.raises(D.DiffError, match="outside editable"):
        D.apply_all(D.parse(bad, SCOPE), real.get, SCOPE)

def test_constant_outside_region_not_editable():
    f = {PATH: "constexpr int KC = 8;\n// NS-EDITABLE-BEGIN\nconstexpr int XPAD = 8;\n// NS-EDITABLE-END\n"}
    with pytest.raises(ST.StructuredEditError, match="not_editable"):
        ST.render_diff([("KC", 16)], f)

def test_prompt_lists_current_values_and_no_heldout(real):
    req = P.ProposalRequest(base_version="v0", files=real, scope_description="s", dev_results="Dev shapes (Tv,Td,r)", mode="structured")
    p = P.build_prompt(req)
    for n in ST.TUNABLES: assert f"- {n} = " in p
    assert "KC = 256" in p and "ONE JSON object" in p and "[16,2048]" in p and "unified diff" not in p.split("## Structured edit task")[1].split("Do NOT write a diff")[0]
    assert "heldout" not in p.lower() and "held-out" not in p.lower()
    assert P.system_for("structured") != P.system_for("diff")

# ---- end to end on the toy repo (editable region has SL and KC) -------------------------------------------------------
def run(toy, tmp_path, raw, attested=False):
    st = Store(tmp_path / "s")
    prop = (P.NorthProposer(transport=lambda p: dict(text=raw, model_id=NORTH_MODEL_ID, revision=NORTH_REVISION, backend="stub", response_id="x"))
            if attested else P.MockProposer([raw]))
    lp = loop.Loop(toy, st, prop, allow_mock_promotion=True, mode="structured")
    return st, lp.step()

def test_loop_invalid_edit_logged(toy, tmp_path):
    st, rec = run(toy, tmp_path, '{"edits":[{"name":"MAXTOK","value":16}],"rationale":"x"}')
    assert rec["verdict"] == "invalid_edit" and "unknown_name" in rec["reason"] and rec["raw_response"] and st.current() == "v0000"

def test_loop_noop_is_no_change(toy, tmp_path):
    st, rec = run(toy, tmp_path, j([("KC", 256)]))
    assert rec["verdict"] == "no_change" and st.current() == "v0000"
    assert rec["structured"]["edits"] == [dict(name="KC", value=256)]

def test_loop_renders_and_evaluates_logs_everything(toy, tmp_path):
    raw = "```json\n" + j([("KC", 128)]) + "\n```"
    st, rec = run(toy, tmp_path, raw)
    assert rec["diff"] and "-constexpr int KC = 256;" in rec["diff"] and "+constexpr int KC = 128;" in rec["diff"]
    assert rec["raw_response"] == raw and rec["structured"]["edits"] == [dict(name="KC", value=128)] and rec["proposer"]["mode"] == "structured"
    assert rec["gates"]["correctness"]["passed"]                       # went through the real gates
    assert rec["verdict"] in ("promoted", "gate_fail:speed")

def test_north_attribution_label_structured(toy, tmp_path):
    # PLUMBING ONLY (stub transport); real North output is never fabricated here
    st, rec = run(toy, tmp_path, j([("KC", 128)]), attested=True)
    assert rec["proposer"]["label"] == "North-authored (structured edit)" and rec["proposer"]["north_attested"]
    assert "harness only rendered" in rec["proposer"]["attribution_note"]
    if rec["verdict"] == "promoted":
        assert st.meta(st.current())["label"] == "North-authored (structured edit)"

def test_diff_mode_still_works(toy, tmp_path):
    st = Store(tmp_path / "s")
    rec = loop.Loop(toy, st, P.MockProposer(["```diff\n```"]), mode="diff").step()
    assert rec["verdict"] == "no_change"


def test_audit_shape_rule_ignores_comment_numbers():
    """Regression: '(<= 16)' inside a trailing // comment must not trigger shape_specialization (live North run, 2026-11-06)."""
    from selfimprove.audit import audit_diff
    from selfimprove.diffs import parse
    from selfimprove.policy import DEFAULT_SCOPE
    d = ("--- a/src/kvm/moe_fused.cu\n+++ b/src/kvm/moe_fused.cu\n@@ -1,1 +1,1 @@\n"
         "-constexpr int MT_SPLIT = 16;           // experts with pair_cnt > MT_SPLIT use the 2-row-tile path (<= 16)\n"
         "+constexpr int MT_SPLIT = 8;           // experts with pair_cnt > MT_SPLIT use the 2-row-tile path (<= 16)\n")
    f = audit_diff(parse(d, DEFAULT_SCOPE), dev_only_shapes={16}, bench_shapes={8, 16})
    assert not [x for x in f if x.code == "shape_specialization" and x.severity == "reject"]
    d2 = d.replace("+constexpr int MT_SPLIT = 8;           // experts", "+if (n == 16) return;  // experts")
    f2 = audit_diff(parse(d2, DEFAULT_SCOPE), dev_only_shapes={16}, bench_shapes={8, 16})
    assert [x for x in f2 if x.code == "shape_specialization" and x.severity == "reject"]


# ---- history + guidance (prompt improvements) ------------------------------------------------------------------------
def _rec(verdict="gate_fail:speed", edits=((("KC", 128),)), sp=0.964):
    return dict(event="proposal", proposal_id="abcdef0123456789", verdict=verdict,
                structured=dict(edits=[dict(name=n, value=v) for n, v in edits], rationale="r"),
                gates=dict(correctness=dict(passed=True), quality=dict(passed=True),
                           speed=dict(passed=False, data=dict(dev=dict(valid=True, speedup=sp, ci_lo=0.96, threshold=1.03),
                                                              heldout=dict(valid=True, speedup=0.5, ci_lo=0.5, threshold=1.03)))))

def test_history_line_has_edits_and_dev_speedup_but_never_heldout():
    h = ST.history_line(_rec())
    assert h.startswith("abcdef01: gate_fail:speed") and "KC=128" in h and "0.964x" in h and "SLOWER" in h
    assert "0.500" not in h and "heldout" not in h.lower()
    assert ST.history_line(dict(proposal_id="x" * 8, verdict="invalid_edit")) == "xxxxxxxx: invalid_edit"

def test_history_reaches_prompt_and_guidance_present(real):
    req = P.ProposalRequest(base_version="v0", files=real, scope_description="s", mode="structured", history=[ST.history_line(_rec())])
    p = P.build_prompt(req)
    assert "## Earlier proposals this session" in p and "KC=128" in p and "0.964x" in p
    assert "BASELINE" in p and "ODD multiple of 16" in p and "Dev launches route only" in p
    assert "heldout" not in p.lower() and "held-out" not in p.lower()

def test_loop_feeds_prior_rounds_into_next_prompt(toy, tmp_path):
    seen = []
    def tr(payload):
        seen.append(payload["prompt"])
        return dict(text=j([("KC", 128)]), model_id=NORTH_MODEL_ID, revision=NORTH_REVISION, backend="stub", response_id="x")
    st = Store(tmp_path / "s")
    lp = loop.Loop(toy, st, P.NorthProposer(transport=tr), mode="structured", prior_history=["old00000: gate_fail:speed | edits MT_SPLIT=8 | measured dev speedup 0.9x"])
    lp.step(); lp.step(rnd=1)
    assert "old00000" in seen[0] and "edits KC=128" not in seen[0]
    assert "edits KC=128" in seen[1]

def test_history_from_store_roundtrip(tmp_path):
    f = tmp_path / "proposals.jsonl"
    f.write_text(json.dumps(dict(event="init")) + "\n" + json.dumps(_rec()) + "\n")
    assert len(ST.history_from_store(f)) == 1 and "KC=128" in ST.history_from_store(f)[0]


def test_new_tunables_render(real):
    d = ST.render_diff([("UNROLL_KS", 4), ("MINB_PER_SM", 8)], real)
    D.parse(d, SCOPE)
    assert "+constexpr int UNROLL_KS = 4;" in d and "+constexpr int MINB_PER_SM = 8;" in d
    assert ST.render_diff([("UNROLL_KS", 16)], real) == ""            # baseline value is a no-op
