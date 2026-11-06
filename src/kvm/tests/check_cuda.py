"""Correctness: CUDA fused and two-pass outputs vs the PyTorch CPU reference. Usage: check_cuda.py <binary> <workdir>"""
import json, os, subprocess, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ref import moe_ref as m

binary, work = sys.argv[1], sys.argv[2]
E = 16
cases = [(4, 4, 3, 1), (1, 1, 1), (4, 4, 2), (16, 16, 8), (3, 13, 5), (16, 1, 4), (8, 8, 8), (1, 16, 3), (2, 2, 1)]
res, ok = [], True
for ci, cs in enumerate(cases):
    Tv, Td, r = cs[:3]; rn = cs[3] if len(cs) > 3 else 0
    d = os.path.join(work, f"case{ci}"); os.makedirs(d, exist_ok=True)
    x, wg, wu, wd, idx, w = m.make_inputs(E, Tv, Td, seed=ci)
    for n, t in (("x", x), ("wg", wg), ("wu", wu), ("wd", wd)):
        t.contiguous().view(torch.int16).numpy().tofile(f"{d}/{n}.bin")
    idx.numpy().astype(np.int32).tofile(f"{d}/idx.bin"); w.numpy().astype(np.float32).tofile(f"{d}/w.bin")
    open(f"{d}/meta.txt", "w").write(f"{E} {Tv} {Td} {r} {rn}")
    subprocess.run([binary, "check", d], check=True)
    ref = m.reference_grouped(x, wg, wu, wd, idx, w, Tv, r, bool(rn))
    row = dict(Tv=Tv, Td=Td, r=r)
    for name in ("fused", "two"):
        o = torch.from_numpy(np.fromfile(f"{d}/out_{name}.bin", dtype=np.float32).reshape(Tv + Td, m.H))
        rel = float((o - ref).norm() / ref.norm()); mx = float((o - ref).abs().max() / ref.abs().max())
        row[name] = dict(rel_l2=rel, max_rel=mx, pass_=bool(rel < 5e-3 and mx < 3e-2))
        ok &= row[name]["pass_"]
    res.append(row); print(json.dumps(row), flush=True)
print("CHECK", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
