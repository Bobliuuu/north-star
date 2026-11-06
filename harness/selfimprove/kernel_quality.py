"""Quality track (kernel): numerical agreement on HELD-OUT real routing traces (routing_traces.json, split 'calib', delivered by fd, never in the workspace).

These traces are never put in the proposer prompt (only the dev split is described there, see config.py). Each item takes a
real window of North's top-8 expert ids at one MoE layer (2 verify + 2 draft tokens, draft budget r=4), remaps the used experts
to a dense id space (weights are seeded random: a routing/work-list test, not a model-quality test) and scores
    item = 1 - rel_l2(fused kernel output, torch reference).
Fixed inputs (NS_SEED ignored) so base and candidate are paired item by item. Paired non-inferiority is judged by the gate.
"""
import common, os, sys, json, math
import numpy as np, torch, subprocess
common.limit_threads(2)
sys.path.insert(0, os.path.join(common.WS, "src", "kvm"))
from ref import moe_ref as m

TRACES = common.heldout_json("routing_traces.json")      # fd from the evaluator, not a workspace file
HELDOUT_SPLIT = "calib"
LAYER_IDX = (0, 11, 23, 35, 47)          # index into the 48 MoE layers
TV, TD, R = 2, 2, 4
W = [1.0 / (1.0 + math.exp(-(1.0 - 0.35 * k))) for k in range(8)]     # fixed descending router-weight profile


def items():
    for tid in sorted(t for t, v in TRACES.items() if v["split"] == HELDOUT_SPLIT):
        ex = TRACES[tid]["experts"]
        for li in LAYER_IDX:
            t0 = (li * 5 + len(tid)) % (len(ex[li]) - (TV + TD))
            yield tid, li, ex[li][t0:t0 + TV + TD]


def score(i, rows, d):
    used = sorted({e for row in rows for e in row})
    remap = {e: j for j, e in enumerate(used)}
    U = len(used)
    x, wg, wu, wd, _, _ = m.make_inputs(U, TV, TD, seed=1000 + i)
    idx = torch.tensor([[remap[e] for e in row] for row in rows], dtype=torch.int32)
    w = torch.tensor([W] * len(rows), dtype=torch.float32)
    os.makedirs(d, exist_ok=True)
    for n, t in (("x", x), ("wg", wg), ("wu", wu), ("wd", wd)):
        t.contiguous().view(torch.int16).numpy().tofile(f"{d}/{n}.bin")
    idx.numpy().astype(np.int32).tofile(f"{d}/idx.bin"); w.numpy().astype(np.float32).tofile(f"{d}/w.bin")
    open(f"{d}/meta.txt", "w").write(f"{U} {TV} {TD} {R} 0")
    p = subprocess.run([common.BINARY, "check", d], capture_output=True, env=common.CHILD_ENV, timeout=240)
    if p.returncode != 0:
        return 0.0
    ref = m.reference_grouped(x, wg, wu, wd, idx, w, TV, R, False)
    o = torch.from_numpy(np.fromfile(f"{d}/out_fused.bin", dtype=np.float32))
    if o.numel() != (TV + TD) * m.H or not bool(torch.isfinite(o).all()):
        return 0.0
    return max(0.0, 1.0 - float((o.reshape(TV + TD, m.H) - ref).norm() / ref.norm()))


scores = [score(i, rows, os.path.join(common.SCRATCH, f"q{i}")) for i, (_, _, rows) in enumerate(items())]
common.emit({"items": scores})
