import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "harness"))
from selfimprove_bench import arms

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_random_arm_legal_unique_budget():
    r = arms.random_arm(33, 11)
    assert len(r) == 33 and len({arms.key(e) for e in r}) == 33
    for e in r:
        assert 1 <= len(e) <= arms.MAX_EDITS
        for k, v in e.items():
            lo, hi, m = arms.SPACE[k]
            assert lo <= v <= hi
        if "KC" in e:
            assert 2048 % e["KC"] == 0 and e["KC"] % 16 == 0
        for p in ("XPAD", "HPAD"):
            if p in e:
                assert e[p] % 8 == 0


def test_random_arm_deterministic_and_seed_disjoint():
    assert arms.random_arm(10, 5) == arms.random_arm(10, 5)
    assert arms.random_arm(10, 5) != arms.random_arm(10, 6)


def test_build_arms_no_overlap_between_arms():
    a, info = arms.build_arms(ROOT / "results")
    keys = [arms.key(e) for _, e in a]
    assert len(keys) == len(set(keys))
    assert info["distinct_valid"] >= 5 and info["proposals"] == 45
    names = {n for n, _ in a}
    assert {"base", "human", "north", "random11", "random12"} <= names


def test_analyze_gate_shape():
    from selfimprove_bench import analyze
    ok = dict(arm="x", build_rc=0, dev_speedup=1.05, held_speedup=1.04, check=dict(passed=True))
    assert analyze.passes(ok)
    assert not analyze.passes(dict(ok, dev_speedup=1.02))
    assert not analyze.passes(dict(ok, held_speedup=1.0))
    assert not analyze.passes(dict(ok, check=dict(passed=False)))
    assert not analyze.passes(dict(ok, build_rc=1))
    s = analyze.summarize([ok, dict(ok, arm="y", dev_speedup=0.9, held_speedup=0.9, check=None)])
    assert s["x"]["n_pass"] == 1 and s["y"]["frac_slower"] == 1.0


def test_build_unique_merges_llm_and_dedupes(tmp_path):
    import shutil, json
    res = tmp_path / "results"
    shutil.copytree(ROOT / "results" / "selfimprove_kernel_structured", res / "selfimprove_kernel_structured")
    shutil.copytree(ROOT / "results" / "selfimprove_kernel_structured_v5", res / "selfimprove_kernel_structured_v5")
    rows = [dict(model="m/x", edits={"KC": 64}), dict(model="m/x", edits={"KC": 64}), dict(model="m/x", edits={}), dict(model="m/x", error="invalid_edit:x")]
    (res / "selfimprove_bench_llm_m_x.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    items, info = arms.build_unique(res)
    kc64 = [u for u in items if arms.key(u["edits"]) == (("KC", 64),)]
    assert len(kc64) == 1 and set(kc64[0]["arms"]) == {"human", "llm:m/x"}
    assert arms.llm_samples(res)["m/x"] == [{"KC": 64}, {"KC": 64}, None, None]
    from selfimprove_bench import analyze
    meas = [dict(arm="human|llm:m/x", edits={"KC": 64}, build_rc=0, dev_speedup=1.06, held_speedup=1.05, check=dict(passed=True))]
    s = analyze.summarize_llm(meas, arms.llm_samples(res))["m/x"]
    assert s["samples"] == 4 and s["no_edit"] == 2 and s["n_pass"] == 2 and s["distinct_pass"] == 1
