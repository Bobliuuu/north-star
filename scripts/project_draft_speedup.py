"""INFERRED self-speculation speedup for edited/quantized drafts on a megakernel-class executor.
Inputs measured (RESULTS.md H5): plain step 3.32 ms; verify block K=3/5/9 -> 4.7/5.8/7.9 ms; expert bytes ~55% of the
weight stream; draft step ~ plain * bytes_fraction (byte-ratio model, same assumption H5 used: 2.4 ms for top-4 bf16).
Acceptance: measured committed tokens/block at gamma=4 from the gate (results/weightedit_*.json), extended to other gamma
with an i.i.d.-agreement model whose per-token agreement `a` is solved to reproduce the measured gamma-4 value.
NOT a measurement: no kernel for a 4-bit draft exists here; CPU lookup/launch overheads not modelled."""
import json, sys

PLAIN = 3.32
VER = [(1, 3.32), (3, 4.7), (5, 5.8), (9, 7.9)]
F_EXP = 0.55


def ver_ms(K):
    import numpy as np
    xs, ys = zip(*VER)
    if K > xs[-1]:
        return ys[-1] + (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) * (K - xs[-1])
    return float(np.interp(K, xs, ys))


def committed(a, g):  # expected tokens per block, iid agreement a, block of g drafts
    return (1 - a ** (g + 1)) / (1 - a) if a < 1 else g + 1


def solve_a(c4):
    lo, hi = 0.0, 0.999
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if committed(mid, 4) < c4 else (lo, mid)
    return (lo + hi) / 2


def draft_ms(r, qbytes, attn_frac=1.0, f_attn=0.0):
    """bytes model: non-expert stream fully read, experts scaled by r/8 and precision ratio qbytes."""
    nonexp = (1 - F_EXP) * (1 - f_attn + f_attn * attn_frac)
    return PLAIN * (nonexp + F_EXP * (r / 8) * qbytes)


def speedup(c4, d_ms, gammas=(1, 2, 3, 4, 6, 8)):
    a = solve_a(c4)
    best = max(((g, committed(a, g) * PLAIN / (g * d_ms + ver_ms(g + 1))) for g in gammas), key=lambda t: t[1])
    return a, best


if __name__ == "__main__":
    rows = [("bf16 top-2, unedited (r2b base)", 3.2047, 2, 1.0),
            ("bf16 top-2 + adapter (r2b v001)", 3.4739, 2, 1.0),
            ("nvfp4 top-2, unedited (r3q base)", 3.03, 2, 4.5 / 16),
            ("nvfp4 top-2 + adapter (r3q)", 3.47, 2, 4.5 / 16)]
    print(f"{'draft':36s} a_iid  draft_ms  best_gamma  speedup_vs_plain   [verifier precision: bf16 ratio]")
    for name, c4, r, q in rows:
        d = draft_ms(r, q)
        a, (g, s) = speedup(c4, d)
        print(f"{name:36s} {a:.3f}  {d:7.2f}   {g:5d}        {s:.3f}")
    print("\nSensitivity: if verifier experts are fp8 (nvfp4 ratio 4.5/8) and sparse attention halves the attention share (f_attn=.3):")
    for name, c4, r, q in rows[2:]:
        for qq, lab in ((4.5 / 8, "fp8 ref"),):
            for fa, afr in ((0.0, 1.0), (0.3, 0.2)):
                d = draft_ms(r, qq if "nvfp4" in name else 1.0, afr, fa)
                a, (g, s) = speedup(c4, d)
                print(f"{name:36s} {lab} f_attn={fa} -> draft {d:.2f} ms, best gamma {g}, {s:.3f}x")
