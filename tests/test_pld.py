import sys, pathlib, random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "harness"))
import pld

COST = {T: 1.0 + 0.1 * (T - 1) for T in range(1, 30)}


def test_lookup_copies():
    p = pld.Lookup((3,)); p.extend([1, 2, 3, 4, 5, 6, 9, 1, 2, 3])
    assert p.propose(3) == ([4, 5, 6], 3)


def test_no_self_match():
    p = pld.Lookup((2,)); p.extend([1, 2]); assert p.propose(2) == ([], 0)


def test_ngc():
    p = pld.NgramCache((2,), 1); p.extend([1, 2, 3, 1, 2, 3, 1, 2]); assert p.propose(2)[0] == [3, 1]


def test_sim_exact_repeat():
    prompt = list(range(100, 140)); gen = list(range(100, 140))
    for n in ("pld3", "sam", "ngc"):
        r = pld.simulate(n, 8, prompt, gen, COST)
        assert r["identical_to_trace"] and r["tokens"] == 40 and r["speedup"] > 1.5, (n, r)


def test_sim_random_no_loss():
    random.seed(0)
    prompt = [random.randrange(5000) for _ in range(200)]; gen = [random.randrange(5000) for _ in range(100)]
    r = pld.simulate("sam", 8, prompt, gen, COST)
    assert r["identical_to_trace"] and 0.95 <= r["speedup"] <= 1.001
