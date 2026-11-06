"""Speed track (kernel): wall time of the fused launch on the DEV shape set or the HELD-OUT shape set (NS_SPLIT).

Shapes are (Tv, Td, r). Dev and held-out integer values are disjoint (config.py lists both for the static audit), so a win
that is specialised to the dev shapes shows up as a dev-only gain and fails the held-out transfer check in the speed gate.
Timing is the binary's own CUDA-event median (L2 flushed, 21 reps) over the routing mode 'corr75'; nothing the candidate
prints is parsed - we read the JSONL the binary writes and validate it (fused vs two-pass outputs must agree).
"""
import common, json, os, subprocess, sys

DEV = [(1, 1, 2), (4, 4, 4), (8, 8, 4), (16, 16, 8)]
HELDOUT = [(3, 5, 3), (6, 9, 5), (12, 7, 6), (13, 3, 7), (5, 11, 3), (9, 6, 5)]
shapes = DEV if common.SPLIT == "dev" else HELDOUT
outp = os.path.join(common.SCRATCH, f"bench_{os.getpid()}.jsonl")
arg = ";".join(f"{a},{b},{c}" for a, b, c in shapes)
p = subprocess.run([common.BINARY, "bench", outp, "--routing", "1", "--shapes", arg, "--reps", "21"],
                   capture_output=True, env=common.CHILD_ENV, timeout=170)
us, ok = [], p.returncode == 0 and os.path.exists(outp)
if ok:
    rows = [json.loads(l) for l in open(outp) if l.strip()]
    ok = len(rows) == len(shapes)
    for r in rows if ok else []:
        if r["out_maxabs"] <= 0 or r["fused_vs_two_maxabsdiff"] / r["out_maxabs"] > 1e-2:
            ok = False                                      # fused disagrees with the two-pass launch: invalid timing
        us.append(r["v"]["fused"]["us_med"])
common.emit({"time_ms": (sum(us) / len(us) / 1000.0) if ok and us else None, "n_shapes": len(us), "split": common.SPLIT})
