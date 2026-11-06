"""Correctness track (kernel): candidate binary vs the torch CPU reference src/kvm/ref/moe_ref.py on FRESH inputs.

Inputs derive from NS_SEED, which the evaluator draws unpredictably per evaluation (and the candidate never sees), so the
kernel cannot memorise inputs. >= 8 cases; each case is checked for BOTH the fused and the two-pass launch lists.
"""
import common, os, random, subprocess, sys, json
import numpy as np, torch
common.limit_threads(2)
sys.path.insert(0, os.path.join(common.WS, "src", "kvm"))
from ref import moe_ref as m

E = 16
rng = random.Random(common.SEED * 7919 + 13)
cases = [(16, 16, 8, 0), (1, 1, 1, 0)]                    # always cover the 2-row-tile path and the minimal path
while len(cases) < 8:
    Tv_, Td_, r_, rn_ = rng.randint(1, 16), rng.randint(1, 16), rng.randint(1, 8), rng.randint(0, 1)
    # HARNESS FIX (found by A100 seed sweep, results/harness_diag_diag1_*): moe_ref.reference() defines renorm only for k_use < TOPK, but the
    # binary's check mode renormalizes draft weights even when r == TOPK. (r=8, renorm=1) made the UNMODIFIED BASELINE fail rel_l2 0.3-0.8 on
    # ~1/3 of seeds (candidate "failures" gate_fail:correctness in run 4 were this, not North). Exclude the semantically undefined combo.
    cases.append((Tv_, Td_, r_, 0 if r_ >= 8 else rn_))


def run_case(ci, Tv, Td, r, rn, d):
    x, wg, wu, wd, idx, w = m.make_inputs(E, Tv, Td, seed=(common.SEED * 1000 + ci) % (2**31 - 1))
    os.makedirs(d, exist_ok=True)
    for n, t in (("x", x), ("wg", wg), ("wu", wu), ("wd", wd)):
        t.contiguous().view(torch.int16).numpy().tofile(f"{d}/{n}.bin")
    idx.numpy().astype(np.int32).tofile(f"{d}/idx.bin"); w.numpy().astype(np.float32).tofile(f"{d}/w.bin")
    open(f"{d}/meta.txt", "w").write(f"{E} {Tv} {Td} {r} {rn}")
    p = subprocess.run([common.BINARY, "check", d], capture_output=True, env=common.CHILD_ENV, timeout=240)
    ref = m.reference_grouped(x, wg, wu, wd, idx, w, Tv, r, bool(rn))
    rows = []
    for name in ("fused", "two"):
        path = f"{d}/out_{name}.bin"
        o = None
        if p.returncode == 0 and os.path.exists(path):
            o = torch.from_numpy(np.fromfile(path, dtype=np.float32))
            o = o.reshape(Tv + Td, m.H) if o.numel() == (Tv + Td) * m.H else None
        if o is None or not bool(torch.isfinite(o).all()):
            rows.append({"name": f"c{ci}_{name}", "rel_l2": 1e9, "max_abs": 1e9, "finite": False}); continue
        rows.append({"name": f"c{ci}_{name}", "rel_l2": float((o - ref).norm() / ref.norm()),
                     "max_abs": float((o - ref).abs().max()), "finite": True})
    return rows


out = []
for ci, (Tv, Td, r, rn) in enumerate(cases):
    out += run_case(ci, Tv, Td, r, rn, os.path.join(common.SCRATCH, f"chk{ci}"))
common.emit({"cases": out})
