# Structured-mode analysis: why 6 North proposals were 2-13% slower

Source: `results/selfimprove_kernel_structured/proposals.jsonl` (two live sessions x 3 rounds, all base v0000, all verdict `gate_fail:speed`;
correctness worst rel_l2 ~2e-4 vs tol 1e-3, quality delta ~1e-10, so only speed failed). Gate threshold: dev speedup CI lower bound > 1.03.

## What North chose (baseline KC=256 XPAD=8 HPAD=8 MT_SPLIT=16 MINB_PER_SM=1)
| session/round | KC | XPAD | HPAD | MT_SPLIT | MINB | dev speedup | held-out speedup |
|---|---|---|---|---|---|---|---|
| A/0 | 128 | 16 | 16 | 8 | 2 | 0.964 | 0.976 |
| A/1 | 128 | 16 | 16 | 8 | 2 | 0.965 | 0.976 |
| A/2 | 128 | 0 | 0 | 8 | 2 | 0.872 | 0.861 |
| B/0 | 128 | 16 | 16 | 8 | 2 | 0.984 | 0.993 |
| B/1 | 128 | 16 | 16 | 8 | 2 | 0.983 | 0.993 |
| B/2 | 128 | 16 | 16 | 8 | 2 | 0.982 | 0.994 |

Findings:
1. North changed all 5 tunables every time (5/5 in all 6) and 5 of 6 proposals were the identical edit set. Nothing was attributable and
   the model never explored a single-axis move. Its history only contained `id: verdict` strings, no edits and no measured speedup, so
   it could not learn that the same set had already been measured slower; temperature 0.3 gave near-deterministic repetition.
2. Direction that hurt most: XPAD=HPAD=0 (A/2): a further -9% on top of the shared edits (0.872 vs 0.964). Row stride 128 elems = 256 B (and
   SL=64 -> 128 B for h) puts every WMMA row on the same banks (8-way conflict). Pad 8 (16 B, an odd multiple of 16 B) is conflict free, pad 16
   (32 B) is 2-way conflicted. North's rationale ("removing padding eliminates overhead", "16 improves bank conflicts") is backwards.
3. The shared edit set (KC 256->128, pads 16, MT_SPLIT 16->8, MINB 1->2) cost 2-4% (B session) to 3.5% (A session). The two sessions
   ran the same edit on different baseline timings (base_med 0.391 ms vs 0.407 ms), so ~2% is run-to-run/box variance. Because the five
   knobs moved together the data cannot separate them, but mechanistically: MT_SPLIT=8 sends experts with 9..16 tokens to the 2-row-tile
   path (double accumulators and padded-row MMAs for the same HBM weight bytes, in an HBM-bound kernel); KC=128 doubles the k-loop
   `__syncthreads` rounds; pad 16 adds 2-way smem conflicts; MINB=2 is only a register-cap hint with no occupancy gain (smem already
   permits several blocks). Every one of these moves is neutral-to-negative for a bandwidth-bound kernel. The baseline is near a local optimum.
4. Held-out was slightly less negative than dev (0.976 vs 0.964) so the regressions are real, not dev-shape specialization.

## Changes (all in the proposer/prompt path; gates, audit, protected lines untouched)
- `structured.history_line` / `Loop.step`: history now lists each earlier proposal's edits and measured DEV speedup (+CI lower bound, threshold,
  "<1 means SLOWER"). Held-out numbers are deliberately excluded (test enforces it). `Loop(prior_history=[...])` and
  `structured.history_from_store(proposals.jsonl)` let a new session be seeded with earlier sessions' measured outcomes.
- `structured.GUIDANCE` added to the structured prompt: baseline = hand-tuned strong start, HBM-bound, dev token/expert characteristics
  (2..32 tokens, top-8 of 128 experts), what MT_SPLIT/KC/pads/MINB actually do, the odd-multiple-of-16B padding rule, and instructions to
  change 1-2 tunables at a time, avoid combinations that measured < 1.0, and prefer an empty edit list to a guess.
- Tests: 4 new tests in `selfimprove/tests/test_structured.py` (history line, no held-out leakage, prompt contents, loop feeds round N-1 into round N,
  store seeding). `cd src && python -m pytest -q selfimprove/tests`: 158 passed, 2 skipped.

## Tunable space: not widened (deliberate)
The five tunables already span the legal range of their `static_assert`s (KC 16..2048 dividing H, pads 0..64, MT_SPLIT 1..16, MINB 1..4).
Further knobs (unroll depth, double-buffered x staging, warps per block, MINB > 4) need kernel-body or protected `static_assert` edits,
which are developer changes that cannot be validated here (no nvcc, no paid GPU runs allowed) and would also change what the correctness gate
exercises. Recommended next step, to be done with a GPU box and a compile: add a `UNROLL_KS` constexpr (inside the region, used by the
`#pragma unroll` in the k loop) and relax `MINB_PER_SM <= 4` to 8; both are pure performance knobs behind unchanged gates. Realistic expectation:
the baseline is already near the HBM roofline (~0.7 of peak), so even a good tunable choice may sit under the 3% gate; an honest `no_change`
(empty edits) is a valid outcome.

## Next run suggestion
Seed with `history_from_store` of both structured stores and run 3 rounds; cost as in docs/SELFIMPROVE.md (not run here).
