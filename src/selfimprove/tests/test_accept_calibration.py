"""scripts/build_accept_calibration.py: measured-only converter. CPU, no network. Nothing here invents acceptance data: synthetic inputs are
hand-built per-round counts used only to check the arithmetic and the refusals."""
import hashlib, importlib.util, json, pathlib, sys
import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
spec = importlib.util.spec_from_file_location("bac", REPO / "scripts/build_accept_calibration.py")
BAC = importlib.util.module_from_spec(spec); spec.loader.exec_module(BAC)
from opt.calibration import AcceptTable  # noqa: E402


def row(config="r4_g4", task="t1", kind="code", gamma=4, rounds=10, counts=(8, 6, 4, 2), r=4, skip=(), n=40, ident=True):
    return dict(phase="1_selfspec", config=config, task=task, kind=kind, gamma=gamma, rounds=rounds, expert_r=r, skip_layers=list(skip),
                accept_rate_per_pos=[c / rounds for c in counts], n_committed=n, identical=ident)


def write(tmp, rows, name="s.jsonl", gpu="Test GPU"):
    p = tmp / name; p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    if gpu: (tmp / (p.stem + ".meta.json")).write_text(json.dumps(dict(gpu=gpu, model="m", model_rev="rev")))
    return p


def run(tmp, rows, **kw):
    p = write(tmp, rows, **kw); out = tmp / "cal.jsonl"
    rc = BAC.main(["--spec", str(p), "--out", str(out)])
    return rc, out


def test_pooling_pmf_and_provenance(tmp_path):
    rc, out = run(tmp_path, [row(task="a"), row(task="b", rounds=20, counts=(10, 10, 0, 0), n=60)])
    assert rc == 0
    (r,) = [json.loads(l) for l in out.read_text().splitlines()]
    # pooled survival: [1, 18/30, 16/30, 4/30, 2/30]
    surv = [1, 18 / 30, 16 / 30, 4 / 30, 2 / 30]
    want = [surv[i] - surv[i + 1] for i in range(4)] + [surv[4]]
    assert r["accept_len_pmf"] == pytest.approx(want, abs=1e-9) and sum(r["accept_len_pmf"]) == pytest.approx(1.0, abs=1e-9)
    assert (r["provenance"], r["hardware"], r["n_tokens"], r["n_samples"], r["n_runs"], r["tasks"]) == ("measured", "Test GPU", 100, 30, 2, ["a", "b"])
    assert r["source_sha256"] == hashlib.sha256((tmp_path / "s.jsonl").read_bytes()).hexdigest()
    assert r["carry_hit_given_full"] == 0.0 and "NOT measured" in r["carry_note"]
    assert (r["workload"], r["attn_budget"], r["r"], r["skip_layers"], r["gamma_measured"]) == ("code", 0, 4, [], 4)
    t = AcceptTable.from_jsonl(out)                                          # accepted by the real loader
    assert t.provenance() == ["measured"] and not t.any_synthetic()
    assert t.get("code", (0, 4, ())).expected_committed(4) == pytest.approx(1 + sum(surv[1:]))


def test_deterministic_bytes(tmp_path):
    p = write(tmp_path, [row(task="b"), row(task="a")])
    for n in ("o1", "o2"): assert BAC.main(["--spec", str(p), "--out", str(tmp_path / n)]) == 0
    assert (tmp_path / "o1").read_bytes() == (tmp_path / "o2").read_bytes()


def test_native_topk_and_largest_gamma_only(tmp_path):
    rows = [row(config="skip8_g4", r=None, skip=(4, 10)), row(config="r4_g2", gamma=2, counts=(5, 1)),
            row(config="r4_g8", gamma=8, counts=(9, 8, 7, 6, 5, 4, 3, 2))]
    rc, out = run(tmp_path, rows); assert rc == 0
    rs = {(r["r"], tuple(r["skip_layers"])): r for r in map(json.loads, out.read_text().splitlines())}
    assert (8, (4, 10)) in rs                                                # expert_r None -> native top-8
    r4 = rs[(4, ())]; assert r4["gamma_measured"] == 8 and r4["config_names"] == ["r4_g8"] and r4["unused_configs"] == ["r4_g2(gamma=2)"]


def test_refuse_missing_fields_and_bad_counts(tmp_path):
    for i, mut in enumerate([dict(rounds=None), dict(accept_rate_per_pos=[0.5, 0.5]), dict(accept_rate_per_pos=[0.123, 0.1, 0.1, 0.1]),
                             dict(accept_rate_per_pos=[0.5, 0.6, 0.1, 0.0]), dict(accept_rate_per_pos=[1.5, 0.1, 0.1, 0.0]), dict(rounds=0)]):
        d = tmp_path / f"c{i}"; d.mkdir()
        bad = {**row(), **mut}
        if mut.get("rounds") is None and "rounds" in mut: bad.pop("rounds")
        rc, out = run(d, [bad]); assert rc == 2 and not out.exists(), mut


def test_refuse_mean_only_rows_like_h100_spec(tmp_path):
    rows = [dict(phase="g_spec", config="r4_g4", gamma=4, mean_accepted=2.4, committed_per_round=3.4, n_committed=64)]
    rc, out = run(tmp_path, rows); assert rc == 2 and not out.exists()


def test_refuse_unknown_hardware_and_cross_source_duplicate(tmp_path):
    rc, out = run(tmp_path, [row()], gpu=None); assert rc == 2 and not out.exists()
    a = write(tmp_path, [row()], "a.jsonl"); b = write(tmp_path, [row()], "b.jsonl", gpu="Other GPU")
    assert BAC.main(["--spec", str(a), str(b), "--out", str(tmp_path / "o")]) == 2


def test_refuse_one_bad_source_among_good(tmp_path):
    a = write(tmp_path, [row()], "a.jsonl"); b = write(tmp_path, [dict(phase="g_plain", tok_s=1.0)], "b.jsonl")
    assert BAC.main(["--spec", str(a), str(b), "--out", str(tmp_path / "o")]) == 2 and not (tmp_path / "o").exists()


def test_dry_run_writes_nothing(tmp_path):
    p = write(tmp_path, [row()])
    assert BAC.main(["--spec", str(p), "--out", str(tmp_path / "o"), "--dry-run"]) == 0 and not (tmp_path / "o").exists()


# ---- the committed artifact ------------------------------------------------------------------------------------------------
def test_committed_calibration_is_exactly_the_converter_output_of_the_committed_measurements(tmp_path):
    cal = REPO / "results/accept_calibration.jsonl"
    assert cal.exists()
    out = tmp_path / "re.jsonl"
    assert BAC.main(["--spec", str(REPO / "results/spec.jsonl"), "--out", str(out)]) == 0
    assert out.read_bytes() == cal.read_bytes()                              # not hand edited, reproducible
    rows = [json.loads(l) for l in cal.read_text().splitlines()]
    assert rows and all(r["provenance"] == "measured" and r["hardware"].startswith("NVIDIA A100") and r["n_tokens"] > 0 and r["source"] == "results/spec.jsonl" for r in rows)
    assert AcceptTable.from_jsonl(cal).provenance() == ["measured"]


def test_h100_spec_is_refused_by_the_converter(tmp_path):
    assert BAC.main(["--spec", str(REPO / "results/h100_spec.jsonl"), "--out", str(tmp_path / "o")]) == 2
