import torch

from tiny import seqs, tiny_model
from weightedit import EditedDraft, Recipe, RouterAdapters
from weightedit.distill import train
from weightedit.gates import fingerprint
from weightedit.quant import BITS, E2M1, bytes_frac, fake_quant


def test_nvfp4_values_on_grid_and_error_bounded():
    torch.manual_seed(0)
    w = torch.randn(4, 64)
    q = fake_quant(w, "nvfp4")
    g = q.reshape(4, 4, 16)
    for r in range(4):
        for k in range(4):
            blk = g[r, k]
            amax = blk.abs().max()
            if amax > 0:  # every value is (grid value) * (one shared scale) per block
                ratio = (blk.abs() / amax)
                grid = E2M1 / 6.0
                assert (ratio.unsqueeze(-1) - grid).abs().min(-1).values.max() < 0.02 + 1e-3
    rel = (q - w).norm() / w.norm()
    assert 0.0 < rel < 0.2


def test_int4_levels_and_none_identity():
    w = torch.randn(3, 32)
    q = fake_quant(w, "int4")
    assert torch.equal(fake_quant(w, "none"), w)
    for blk in q.reshape(3, 2, 16):
        lv = (blk / (blk.abs().max() / 7)).round()
        assert lv.abs().max() <= 7 + 1e-4
    assert (q - w).norm() / w.norm() < 0.2


def test_bytes_frac():
    assert bytes_frac("none") == 1.0 and abs(bytes_frac("nvfp4") - 4.5 / 16) < 1e-9 and BITS["int4"] == 5.0


def test_quant_draft_differs_and_verifier_untouched():
    m = tiny_model()
    x = torch.tensor(seqs(1, 40, 1))
    h0 = fingerprint(m)
    with torch.no_grad():
        base = m(x).logits
    r = Recipe(expert_r=2, expert_quant="nvfp4").validate(m.config)
    ad = RouterAdapters(m.config, r)
    with EditedDraft(m, ad, 2, "nvfp4") as e:
        with torch.no_grad():
            assert torch.equal(m(x).logits, base)
            with e.draft():
                dq = m(x).logits
    assert fingerprint(m) == h0 and not torch.equal(dq, base)


def test_quant_draft_trains_and_gates_report_bytes(tmp_path):
    from weightedit.loop import run_round
    from weightedit.store import EditStore
    m = tiny_model()
    tr, ga = seqs(8, 64, 5), seqs(8, 64, 6)
    class P:
        def propose(self, h):
            return dict(expert_r=2, expert_quant="nvfp4", steps=60, window=32, batch=4, lr=3e-3, rank=8), dict(source="grid")
    st = EditStore(tmp_path)
    e = run_round(m, P(), st, tr, [0] * 8, ga, [0] * 8, [], log=lambda *_: None, boot=50, margin=0.0)
    n = e["numbers"]
    assert e["gates"]["lossless"] and n["draft_expert_bytes_frac"] == (4.5 / 16) * 2 / 4
    assert e["train"]["last"] < e["train"]["first"]


def test_skip_layers_draft_untouched_verifier_and_trains(tmp_path):
    from weightedit.loop import run_round
    from weightedit.store import EditStore
    m = tiny_model()
    x = torch.tensor(seqs(1, 40, 1))
    with torch.no_grad():
        base = m(x).logits
    r = Recipe(expert_r=2, skip_n=2, steps=50, window=32, batch=4, lr=3e-3, rank=8).validate(m.config)
    assert len(r.skip_set(m.config)) == 2 and 0 not in r.skip_set(m.config)
    ad = RouterAdapters(m.config, r)
    assert all(int(k) not in r.skip_set(m.config) for k in ad.A.keys())
    h0 = fingerprint(m)
    with EditedDraft(m, ad, 2, "none", r.skip_set(m.config)) as e:
        with torch.no_grad():
            assert torch.equal(m(x).logits, base)
            with e.draft():
                assert not torch.equal(m(x).logits, base)
    assert fingerprint(m) == h0 and all("forward" not in mod.__dict__ for mod in m.modules())
    tr, ga = seqs(8, 64, 5), seqs(8, 64, 6)
    class P:
        def propose(self, h): return r.to_dict(), dict(source="grid")
    e = run_round(m, P(), EditStore(tmp_path), tr, [0] * 8, ga, [0] * 8, [], log=lambda *_: None, boot=50, margin=0.0)
    assert e["gates"]["lossless"] and e["numbers"]["draft_layers_frac"] < 1.0 and e["train"]["last"] < e["train"]["first"]
