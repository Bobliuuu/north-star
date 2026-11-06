# Phase 5 async self-speculation: cost-model study (MODELED, not a hardware result)

Everything below is arithmetic on previously measured costs plus swept assumptions. No GPU was run. Code: `src/async_spec/` (`model.py`, `run.py`, CPU tests). Output: `results/async_spec_model_SIMULATED.json`. Reproduce: `python src/async_spec/run.py`.

## Question
Phase 5 (docs/PLAN.md): while full North verifies block n, approximate North drafts block n+1 for k guessed outcomes (accept length, bonus token), Saguaro-style. On a branch hit the next draft is already done. Compare serial vs concurrent on the SAME single device; kill on negative net benefit.

## Model
- Serial round: `T_s = gamma*d + V`. Commits `1 + a` tokens.
- Async round: verify runs concurrently with a k-wide, gamma-step branch draft. Hit (prob h): next draft done. Miss: branches discarded, draft redone serially after verify.
- Bandwidth sharing (the crux): a job of standalone time t keeps HBM busy for `beta*t`. Two concurrent jobs demanding `beta_V+beta_B > 1` both slow by `s = beta_V+beta_B`. Only non-bandwidth time (launch gaps) is hideable. This is optimistic: perfect interleaving, no SM or L2 contention, no KV-copy or memory cost for k branches, and (important) no SM partitioning cost for a persistent megakernel.
- k-wide branch step cost uses the measured megakernel batch curve (BS 1/2/4/8 = 3.32/4.10/5.29/7.39 ms).
- h(k) = sum of the top-k outcome probabilities from the measured accept-length distribution.

## Provenance of inputs
| Input | Tag |
|---|---|
| H100 graphed draft ms/step and verify ms/round, per config (r4_g2/g4/g8, skip8_g4) | MEASURED (`results/h100_spec.jsonl`) |
| Per-position accept `P(a>=i)` (sums to mean accepted) | MEASURED (`results/spec.jsonl`, A100; H100 mean accepted within 0.2) |
| Megakernel BS curve; plain 3.32 ms/token; BS1 beta=0.60 (1.97 ms floor / 3.25) | MEASURED (RESULTS.md H1) |
| Graphed-PyTorch beta ~0.45 (expert+lm_head bytes share of K=1 step) | INFERRED |
| Megakernel projection V(K=3/5/9)=4.7/5.8/7.9 ms, d=2.4 ms | INFERRED (RESULTS.md H5) |
| Full-accept bonus-guess hit = `apos[-1]/apos[-2]` | INFERRED proxy |
| Reject-bonus guess prob q_rej in {0,.05,.2,.4} (decay 0.5 per extra guess) | INVENTED, swept (docs/TAGE.md simulated 0.03-0.05) |
| beta in {0.45,0.6,0.8,1.0}, k in 0..16, branch-batch cost x{0.5,1,2}, BS>8 extrapolation | INVENTED, swept |

## Results (nominal q_rej=0.05, branch cost x1; best k per cell)
Speedup over SERIAL self-spec, and over PLAIN decode (the baseline that actually matters; serial self-spec already loses to plain on every measured config).

Serial self-spec vs plain decode: graphed H100 measured-input arithmetic 0.77 (g4), 0.80 (skip8_g4), 0.85 (g2), 0.64 (g8); megakernel projection 0.94 (g2), 0.83 (g4), 0.69 (g8).

| executor / config | beta=0.45 vs serial (vs plain) | beta=0.6 | beta=0.8 | beta=1.0 |
|---|---|---|---|---|
| graphed r4_g2 | 1.32 (1.12) | 1.20 (1.02) | 1.01 (0.86) | 0.88 (0.75) |
| graphed r4_g4 | 1.15 (0.89) | 1.09 (0.84) | 0.98 (0.75) | 0.89 (0.69) |
| graphed r4_g8 | 1.05 (0.67) | 1.01 (0.65) | 0.95 (0.61) | 0.90 (0.57) |
| megakernel r4_g2 | 1.45 (1.36) | 1.27 (1.19) | 1.02 (0.95) | 0.85 (0.79) |
| megakernel r4_g4 | 1.24 (1.04) | 1.14 (0.95) | 0.97 (0.81) | 0.85 (0.71) |
| megakernel r4_g8 | 1.09 (0.75) | 1.03 (0.71) | 0.92 (0.63) | 0.83 (0.57) |

Observations:
1. Best k is 1 in every cell. k=4 drops below serial at beta=0.6 (e.g. megakernel g4: 1.14 at k=1, 0.93 at k=4, 0.57 at k=16). Hit probability barely grows with k (r4_g4: 0.52 at k=1, 0.55 at k=4, 0.57 at k=16) because reject-side outcomes are low-probability and bonus-token guesses are the bottleneck, while branch cost grows with the measured batch curve. More branches are not free; they are net negative past k=1.
2. At beta=1.0 (fully HBM-bound) concurrency never beats serial: it only time-slices bandwidth, and misses add pure waste.
3. Break-even beta (max beta at which async beats serial self-spec, best k): g2 0.82, g4 0.76, g8 0.64. To beat plain decode: only graphed g2 (beta <= 0.62), megakernel g2 (<= 0.75) and megakernel g4 (<= 0.54); g8, skip8_g4 and graphed g4 never.
4. Even a free, perfect-hit async draft (upper bound = `(gamma*d+V)/V`: 2.0x to 6.2x vs serial) is not reachable: the hit probability caps at ~0.29-0.63 and the draft takes longer than the verify window in most configs.

## Honest conclusion
Net benefit is negative or marginal for North decode. The evidence we have says North verify is HBM-bound (RESULTS.md H2/H6), measured beta of the best executor is 0.60, and a 1.0-beta region is where concurrency is a pure loss. The positive cells above need beta <= 0.6 AND gamma=2 (the config where serial speculation is closest to break-even), AND k=1, AND perfect SM/bandwidth interleaving that a real kernel scheduler will not give, AND the INFERRED megakernel numbers. A gain of ~1.0-1.2x over plain decode at best is inside the model error of the INVENTED knobs. Recommendation: do NOT build Phase 5 on a single device for exact North decode. It becomes interesting only with a second device (not the same-budget comparison), or if a measured beta well under 0.6 for the verify block appears (the unmeasured block-verify beta is the key missing number).

Kill condition (as implemented): kill if best-k speedup over serial <= 1 at the measured beta, or over plain decode <= 1. At beta=0.6 for megakernel g4/g8, graphed g4/g8 and skip8_g4 this fires; only g2 survives, marginally.

## What would change the answer (measure these before any build)
1. Block-verify and draft-step HBM-busy fraction (beta) from a profiler, not inferred.
2. Real concurrent-kernel slowdown: run a verify graph and a draft graph on two streams on one H100 (a measured s, replacing the model s).
3. Real bonus-token hit rates from North traces (docs/TAGE.md numbers are simulated).
4. Memory cost of k KV branches (not modeled).

## Limits
Modeled not measured; A100 accept shapes reused for H100; the megakernel row stacks INFERRED costs; the contention model is a bandwidth-roofline, optimistic. Greedy exact decoding; no sampling.
