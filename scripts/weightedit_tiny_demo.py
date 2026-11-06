"""CPU demo of the weight-edit loop on a RANDOM tiny Cohere2Moe (not North). Proves plumbing, not a result."""
import json, pathlib, sys, tempfile
sys.path.insert(0, "src"); sys.path.insert(0, "src/weightedit/tests")
from tiny import seqs, tiny_model
from weightedit.loop import GridProposer, run_loop
from weightedit.store import EditStore

m = tiny_model()
tr, ga = seqs(16, 64, 5), seqs(16, 64, 6)
st = EditStore(tempfile.mkdtemp())
prop = GridProposer(rs=(2, 3), ranks=(8,), lrs=(3e-3,), steps=300, window=32, batch=4)
hist = run_loop(m, prop, st, tr, [0] * 16, ga, [0] * 16, rounds=2, log=lambda *_: None)
out = dict(note="random tiny model, random tokens; plumbing demo only", history=hist, ledger=st.ledger())
pathlib.Path("results/weightedit_tiny_demo.json").write_text(json.dumps(out, indent=1, default=float))
for h in hist:
    n = h["numbers"]
    print(h["recipe"]["expert_r"], h["gates"], f"agree {n['agree_base']:.3f}->{n['agree_edit']:.3f} "
          f"ci_lo {n['gain_ci_lower_agree']:.3f} train_agree {n['agree_train']:.3f} kl {h['train']['first']:.3f}->{h['train']['last']:.3f}")
