"""CPU tests for the demo renderer/replay/spend parser. Uses demo/fixtures/FIXTURE_*.jsonl: HAND-MADE, not North output."""
import copy, io, pathlib, sys
import pytest

D = pathlib.Path(__file__).resolve().parents[1] / "demo"
sys.path.insert(0, str(D))
import events as E, render as R, spend as S, view as V

FIX = D / "fixtures" / "FIXTURE_trace.jsonl"
NOPROV = D / "fixtures" / "FIXTURE_no_provenance.jsonl"


def test_fixture_requires_flag():
    evs = E.read_events(FIX)
    with pytest.raises(E.ProvenanceError, match="FIXTURE"):
        E.check_provenance(evs)
    assert E.check_provenance(evs, allow_fixture=True)["source"] == "FIXTURE"


def test_refuses_missing_provenance():
    with pytest.raises(E.ProvenanceError, match="no provenance"):
        E.check_provenance(E.read_events(NOPROV), allow_fixture=True)


def test_refuses_unknown_source_and_missing_keys_and_missing_step_fields():
    evs = E.read_events(FIX)
    a = copy.deepcopy(evs); a[0]["provenance"]["source"] = "made-up"
    with pytest.raises(E.ProvenanceError, match="unknown"):
        E.check_provenance(a, True)
    b = copy.deepcopy(evs); del b[0]["provenance"]["model_rev"]
    with pytest.raises(E.ProvenanceError, match="missing"):
        E.check_provenance(b, True)
    c = copy.deepcopy(evs); del c[2]["cum_cost_usd"]
    with pytest.raises(E.ProvenanceError, match="missing fields"):
        E.check_provenance(c, True)


def test_real_source_accepted():
    evs = copy.deepcopy(E.read_events(FIX)); evs[0]["provenance"]["source"] = E.REAL_SOURCE
    assert E.check_provenance(evs)["source"] == E.REAL_SOURCE


def test_render_contains_all_fields_and_fixture_banner():
    out = io.StringIO(); R.replay(E.read_events(FIX), out=out)
    t = out.getvalue()
    assert "FIXTURE: hand-made" in t
    assert "plain" in t and "self-spec g4 [skip8]" in t
    assert "2/4" in t and " 50%" in t          # acceptance this round / cumulative
    assert "30.0" in t and "25.0" in t         # inst / cumulative committed tok/s
    assert "2.3900" in t                       # cumulative optimisation cost
    assert "mode switch -> selfspec" in t and "done: 5 tokens" in t


def test_cli_replay_refusals(capsys):
    assert V.main(["--replay", str(FIX)]) == 2
    assert V.main(["--replay", str(NOPROV), "--allow-fixture"]) == 2
    assert V.main(["--replay", str(FIX), "--allow-fixture"]) == 0
    assert "FIXTURE" in capsys.readouterr().out


def test_spend_parse_real_file():
    total, n, cap = S.load_spend(pathlib.Path(__file__).resolve().parents[1] / "results" / "SPEND.md")
    assert n >= 10 and 2.0 < total < 30


def test_spend_parse_small():
    txt = "| t | task | gpu | secs | est USD | ok |\n|---|---|---|---|---|---|\n| a | x | A100 | 10 | 0.50 | |\n| b | y | H100 | 5 | $0.25 | True |\n"
    assert S.parse_spend(txt)[:2] == (0.75, 2)
