import sys, pathlib, textwrap, json, os
import pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

CU = """\
// toy stand-in for src/kvm/moe_fused.cu
constexpr int H = 2048, I = 768;   // protected: model shape
// NS-EDITABLE-BEGIN tunables
constexpr int SL = 64;
constexpr int KC = 256;
// NS-EDITABLE-END
__global__ void k() {}
"""
POLICY = '{\n "skip_tail": 0,\n "dev_only_fast": false\n}\n'

HARNESS_COMMON = '''
import os, sys, re, json, random, math, runpy
ws = os.environ["NS_WORKSPACE"]; nonce = os.environ.pop("NS_NONCE")
seed = int(os.environ.get("NS_SEED", "0")); split = os.environ.get("NS_SPLIT", "dev")
src = open(os.path.join(ws, "src/kvm/moe_fused.cu")).read()
SL = int(re.search(r"SL = (\\d+);", src).group(1))
pol = json.load(open(os.path.join(ws, "src/opt/policy_tables/sched.json")))
hook = os.path.join(ws, "src/opt/policy_tables/hook.py")
if os.path.exists(hook): runpy.run_path(hook)
def out(o): o["nonce"] = nonce; print(json.dumps(o)); sys.stdout.flush()
'''
CHECK = HARNESS_COMMON + '''
rng = random.Random(seed)
cases = []
for i in range(10):
    bad = (768 % SL != 0)
    cases.append({"name": f"c{i}", "rel_l2": (0.5 if bad else 1e-4 * (1 + rng.random())), "max_abs": 0.01, "finite": True})
out({"cases": cases})
'''
QUALITY = HARNESS_COMMON + '''
out({"items": [1.0 - (0.3 if pol["skip_tail"] and i % 2 == 0 else 0.0) for i in range(30)]})
'''
BENCH = HARNESS_COMMON + '''
rng = random.Random(seed)
t = 10 * (1 + 0.3 * abs(math.log2(SL / 128)))
if pol["skip_tail"]: t *= 0.5
if pol["dev_only_fast"] and split == "dev": t *= 0.5
out({"time_ms": t * (1 + 0.005 * rng.uniform(-1, 1))})
'''

@pytest.fixture
def toy(tmp_path):
    from selfimprove import loop, gates, policy
    repo = tmp_path / "repo"
    for rel, text in {"src/kvm/moe_fused.cu": CU, "src/opt/policy_tables/sched.json": POLICY,
                      "harness/check.py": CHECK, "harness/quality.py": QUALITY, "harness/bench.py": BENCH}.items():
        p = repo / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)
    py = sys.executable
    cfg = loop.EvalConfig(
        repo=repo, subdirs=["src/kvm", "src/opt", "harness"],
        correctness_cmd=[py, "harness/check.py"], quality_cmd=[py, "harness/quality.py"], bench_cmd=[py, "harness/bench.py"],
        scope=policy.Scope(rules=[policy.ScopeRule("src/kvm/moe_fused.cu", regions_only=True),
                                  policy.ScopeRule("src/opt/policy_tables/*.json"),
                                  policy.ScopeRule("src/opt/policy_tables/*.py", allow_create=True)]),
        correctness=gates.CorrectnessSpec(min_cases=8), speed=gates.SpeedSpec(n_runs=7),
        check_timeout_s=4, bench_timeout_s=4, dev_shapes={7, 13, 96}, heldout_shapes={96, 200}, tmp_parent=tmp_path)
    return cfg
