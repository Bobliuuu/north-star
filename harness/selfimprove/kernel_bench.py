"""Speed track (kernel): wall time of the fused launch on the DEV shape set or the HELD-OUT shape set (NS_SPLIT).

Shapes are (Tv, Td, r). Dev and held-out integer values are disjoint (config.py lists both for the static audit), so a win
that is specialised to the dev shapes shows up as a dev-only gain and fails the held-out transfer check in the speed gate.
Timing is the binary's own CUDA-event median (L2 flushed, 21 reps) over the routing mode 'corr75'; nothing the candidate
prints is parsed - we read the JSONL the binary writes and validate it (fused vs two-pass outputs must agree).
The harness ALSO wall-clocks the child at fixed rep counts (ref_ms) so the evaluator can reject a self-report that disagrees
with it (gates.timer_crosscheck): the candidate-built binary is not the only clock any more.
"""
import common, json, os, subprocess, sys, time

DEV = [(1, 1, 2), (4, 4, 4), (8, 8, 4), (16, 16, 8)]
HELDOUT = [(3, 5, 3), (6, 9, 5), (12, 7, 6), (13, 3, 7), (5, 11, 3), (9, 6, 5)]
shapes = DEV if common.SPLIT == "dev" else HELDOUT
arg = ";".join(f"{a},{b},{c}" for a, b, c in shapes)
REPS, REPS_LO = 21, 3                  # fixed here (protected harness): the candidate cannot change the iteration count


def launch(reps, tag):
    outp = os.path.join(common.SCRATCH, f"bench_{os.getpid()}_{tag}.jsonl")
    t0 = time.perf_counter()
    p = subprocess.run([common.BINARY, "bench", outp, "--routing", "1", "--shapes", arg, "--reps", str(reps)],
                       capture_output=True, env=common.CHILD_ENV, timeout=150)
    return p, outp, (time.perf_counter() - t0) * 1000.0


p, outp, wall_hi = launch(REPS, "hi")
us, ok = [], p.returncode == 0 and os.path.exists(outp)
if ok:
    rows = [json.loads(l) for l in open(outp) if l.strip()]
    ok = len(rows) == len(shapes)
    for r in rows if ok else []:
        if r["out_maxabs"] <= 0 or r["fused_vs_two_maxabsdiff"] / r["out_maxabs"] > 1e-2:
            ok = False                                      # fused disagrees with the two-pass launch: invalid timing
        us.append(r["v"]["fused"]["us_med"])
# independent harness-side timing: marginal wall per iteration per shape from a second, shorter run (same shapes)
ref_ms = wall_lo = None
if ok:
    p2, _, wall_lo = launch(REPS_LO, "lo")
    if p2.returncode == 0 and wall_hi > wall_lo:
        ref_ms = (wall_hi - wall_lo) / (REPS - REPS_LO) / len(shapes)
common.emit({"time_ms": (sum(us) / len(us) / 1000.0) if ok and us else None, "n_shapes": len(us), "split": common.SPLIT,
             "n_iters": REPS, "ref_ms": ref_ms, "harness_wall_ms": wall_hi + (wall_lo or 0.0)})
