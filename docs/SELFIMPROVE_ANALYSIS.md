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

## Run 3 outcome (MEASURED, A100-80GB; app ap-B0wZK4IVQWioxxurZCKpLO, log logs/selfimprove_structured_run3.log, store rows 7-9)
This was the first live run after the history/guidance change. North still changed all 5 tunables and repeated the same edit set: round 0 gate_fail:speed (XPAD=HPAD=0 variant, dev 0.882x, held-out 0.877x), round 1 gate_fail:speed (KC128 XPAD16 HPAD16 MT8 MINB2, dev 0.983x, held-out 0.993x), round 2 gate_fail:correctness (rel_l2 0.78 on c7_fused; the v5 diagnosis below shows this failure class comes from the correctness harness bug and also hits the unmodified baseline, so it says nothing about the edit). 0 of 3 passed; the store has no North version. The later v5 section (harness fixes, 33 proposals, 0 passes) supersedes this as the current state.

## Next run suggestion (superseded by v5 below)
Seed with `history_from_store` of both structured stores and run 3 rounds; cost as in docs/SELFIMPROVE.md (not run here).

## Human OFAT sweep (MEASURED, A100-80GB, 2026-11-06; results/kernel_sweep_sweep1_20261106T065947Z.jsonl, app ap-kYajfWtgyO3OPmzJ1fSN0O)
One factor at a time from the baseline (KC256 XPAD8 HPAD8 MT16 MINB1 UNROLL16), dev shapes, 3 interleaved passes, run-to-run spread ~0.3%.
Correctness (check_cuda.py vs torch CPU ref, worst rel_l2 2.1e-4) PASS for baseline and every UNROLL_KS / MINB_PER_SM config. dev_x = baseline_time / time.

| change | dev_x | held-out_x | regs |
|---|---|---|---|
| KC=64 | 1.054 | 1.042 | 112 (base 252) |
| UNROLL_KS=4 | 1.040 | 1.037 | 118 |
| MINB_PER_SM=3 | 1.023 | 1.072 | 167 |
| KC=512 + UNROLL_KS=8 | 1.013 | 1.016 | 178 |
| KC=128, XPAD16/24, HPAD16/24, MT_SPLIT 4/8/12, MINB2, UNROLL8 | 0.99-1.003 | ~1.0 | |
| MINB4 / XPAD0 / HPAD0 | 0.981 / 0.961 / 0.942 | | MINB4 spills |
| UNROLL_KS=2 / MINB6 / MINB8 / UNROLL_KS=1 | 0.871 / 0.799 / 0.733 / 0.724 | | MINB6,8 spill |

So the baseline is NOT at a local optimum for speed: the 252-register full unroll costs ~4-5%; capping registers (KC=64, UNROLL_KS=4, MINB=3) recovers it. MINB>=6 and UNROLL_KS<=2 are strongly harmful, so the relaxed MINB cap of 8 is a trap, not a gain.

### Combo sweep (MEASURED, A100-80GB, results/kernel_sweep_combo1_20261106T074908Z.jsonl, app ap-N1q7OiAH5UWFH8vxV9xgO8, all 13 configs correctness PASS)
KC=64 alone gives dev 1.056 / held-out 1.043; adding UNROLL_KS or MINB (2/3, 4/8) changes it by <=0.4%, i.e. the effects do not stack (all just cap registers). Best: KC64+MINB3 1.057 (held 1.041). KC=32 (0.814) and KC64+UNROLL_KS2 (0.839) are strongly worse. Takeaway: KC=64 alone is a ~5.5% win; the tunable space is otherwise exhausted.

## v5 (branch selfimprove-v5, 2026-11-06): diagnosis of the 13 failures, harness fixes, live run 5+6

### Diagnosis (MEASURED, A100-80GB; results/harness_diag_diag{1,2,3}_*.json, modal/harness_diag.py)
Of the 13 proposals in `selfimprove_kernel_structured`: 9 were real `gate_fail:speed` (North's edit sets measured 2-13% slower), 2 were `gate_fail:correctness`, 2 were `gate_fail:timer`. The last four were HARNESS problems, not North's:
1. **Timer cross-check was miscalibrated.** The gate said "self-reported 0.392 ms vs harness-measured 3.67 ms, x9.4 > 4.0" and it fired on the unmodified BASELINE run (tag `b`). `ref_ms` is a wall-clock marginal that includes the fused AND two-pass launches, L2 flush, process work, so t/ref is ~0.13 (about 1/7.5) for any kernel. The absolute 4x cap could never pass. Fix: `TimerSpec(calibrate=True, ratio_tol=1.5)`: the median t/ref of each tag must be within 1.5x of the BASELINE's median t/ref (single runs within 6x, since A/A single-run rho spans 0.7x-1.6x). A candidate that mis-reports its own time shifts every run and is still rejected (unit-tested); the wall-vs-evaluator-clock bound is unchanged.
2. **Correctness harness bug.** `kernel_check.py` randomly generated cases with (r=8, renorm=1). `moe_ref` defines renorm only for k_use < TOPK, but the binary renormalizes draft weights even for r=8, so the UNMODIFIED BASELINE failed rel_l2 0.27-0.77 on those seeds (10 of 24 seeds, deterministic per seed, identical for baseline / North's edit set / MT_SPLIT=8; sequential repro, results/harness_diag_diag2_*.json). The candidate "correctness failures" (rows 9 and 11) were this. Fix: that undefined combination is no longer generated; after the fix 30/30 (10 seeds x 3 variants) pass, worst rel_l2 4.0e-4 vs tol 1e-3. Also new: if the baseline fails its own correctness gate the verdict is `baseline_broken` (a harness problem), never a North failure. Tolerance unchanged.
3. **Speed gate itself is fine.** A/A (baseline vs baseline, 18 interleaved pairs per split): ratio of medians 0.9994-1.0008, per-run spread 0.2% (one 5% outlier of 36). So the 3% threshold and the speed noise floor are sound; the speed failures were real slowdowns.
Gates NOT weakened: anchored head, ABBA order, held-out inputs, rel_l2 tolerance, 3% min gain, held-out transfer all unchanged.

### Task changes
UNROLL_S2 (1..4, default 4 = old full unroll) and UNROLL_NT (1..8, default 1) added to the editable region (8 tunables; compiled and passed the 30-seed correctness sweep at defaults); at most 3 edits per proposal (attributable); history lines carry base version, edits, measured dev speedup and reasons for invalid/duplicate; best-of-N per round (`--per-round`, temperature 0.3/0.55/0.8/1.0, greedy promotion of the first all-gates pass, duplicate edit sets skipped without GPU evaluation and resampled up to 3x); neutral "tunables not yet changed" hint (names only); prompt now includes the baseline ptxas profile (252 regs/thread, 0 spills, register-limited occupancy; a compile fact, no timings). North was NOT shown the human OFAT/combo sweep results. Proposer redeployed with scaledown 600 s (a 60 s window reloaded the 60 GB model between most proposals, runs 3-4 latencies of 660-1300 s).

### Live result (store results/selfimprove_kernel_structured_v5; runs 5 and 6, apps ap-FdLqlGKknNuRnVjg4II6PS and ap-pR9zZWc1qq2H2HHSOEodt7)
33 North proposals: 11 measured (all `gate_fail:speed`), 20 `duplicate_edit` (never evaluated), 2 `invalid_edit`. **0 passed all gates; no North-authored edit has a real speedup; `reportable_as_north_authored` stays false (no North version in the chain).** Best measured dev speedups: 1.006x (KC=512, UNROLL_KS=4, MINB=2), 1.003x (UNROLL_KS=32, XPAD=24, UNROLL_NT=4), all CI-lower-bound <= 1.006 versus the 1.03 threshold. No timer or correctness failures in 11 evaluations, so the harness fixes held. North still fixates on KC=128 (+ pads) despite seeing it measure 0.97-0.99x, and never proposed KC=64 (the human-sweep winner, 1.054x). Diagnosis: a model-diversity/search problem (temperature up to 1.0 mostly returned the same edit set: 20 of 33 proposals duplicates), not a harness problem any more.
