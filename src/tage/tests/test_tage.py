import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tage import TagePredictor, TageConfig, PromptLookup, SuffixLast, SuffixFreq, simulate
from tage.baselines import SuffixCal


def feed(p, toks):
    for t in toks:
        p.commit(t)


def test_learns_repeated_sequence():
    p = TagePredictor()
    seq = [5, 9, 3, 7, 1, 8, 2, 6, 4, 11, 13, 17]
    for _ in range(6):
        p.start_doc(); feed(p, seq)
    p.start_doc(); feed(p, seq[:6])
    out, _ = p.propose(6)
    assert out == seq[6:12]


def test_longest_history_disambiguates():
    # same last-2 tokens, different earlier context -> different next token
    p = TagePredictor(TageConfig(lengths=(2, 4, 8)))
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]; b = [11, 12, 3, 4, 15, 16, 17, 18, 19]
    for _ in range(8):
        p.start_doc(); feed(p, a)
        p.start_doc(); feed(p, b)
    p.start_doc(); feed(p, [1, 2, 3, 4]); assert p.propose(1)[0] == [5]
    p.start_doc(); feed(p, [11, 12, 3, 4]); assert p.propose(1)[0] == [15]


def test_propose_does_not_mutate():
    p = TagePredictor(); feed(p, list(range(40)) * 3)
    snap = (list(p.hist), [list(x) for x in p.tok], [list(x) for x in p.ctr], dict(p.calib))
    p.propose(8)
    assert snap == (p.hist, p.tok, p.ctr, p.calib)


def test_calibration_gates_unreliable():
    import random
    rng = random.Random(0)
    p = TagePredictor()
    p.start_doc(); feed(p, [rng.randrange(50) for _ in range(5000)])  # noise: nothing reliable
    out, _ = p.propose(8, theta=0.5)
    assert out == []


def test_state_bytes_bounded():
    assert TagePredictor().state_bytes() < 1_000_000


def test_sim_oracle_exactness():
    doc = [1, 2, 3, 4] * 50
    for mk in (lambda: TagePredictor(), lambda: PromptLookup(), lambda: SuffixLast(), lambda: SuffixFreq(), lambda: SuffixCal()):
        r = simulate(mk(), [doc], kmax=8)
        assert r.tokens == len(doc)                 # every token committed exactly once
        assert sum(r.accept_hist.values()) == r.steps
        assert r.accepted <= r.drafted
    # periodic stream: TAGE must beat 1 token/step
    r = simulate(TagePredictor(), [doc], kmax=8)
    assert r.tokens / r.steps > 3


def test_pld_copy():
    p = PromptLookup(3, 1); p.start_doc(); feed(p, [1, 2, 3, 9, 9, 1, 2, 3])
    assert p.propose(2)[0] == [9, 9]


def test_cold_vs_warm():
    doc = [7, 8, 9, 10, 11, 12] * 30
    warm = simulate(TagePredictor(), [doc, doc], keep_state=True)
    cold = simulate(TagePredictor(), [doc, doc], keep_state=False)
    assert warm.steps <= cold.steps
