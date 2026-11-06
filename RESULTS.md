# RESULTS (live; updated as data lands)

Status 2026-11-05/06. Everything here is labeled MEASURED, SIMULATED or SYNTHETIC. No end-to-end North speedup is claimed.

## Measured (real hardware, synthetic routing)
Fused vs two-pass MoE expert GEMM, one North-shaped layer (2048 hidden, 768 inter, 128 experts, top-8), Modal A100-80GB, bf16, 6144 configs, 1 seed.
Raw: `results/kvm_fused_A100-80GB_fullsweep_*.jsonl`.

| routing (synthetic) | draft/verify expert overlap | median time saving | configs >=15% |
|---|---|---|---|
| uniform | 0.39 | 16.2% | 56% |
| corr75 | 0.84 | 27.4% | 89% |
| identical | 1.0 | 32.7% | 93% |

Caveats: one layer only; routing synthetic (real overlap unmeasured); part of the saving is removed launch cost, not shared loads; own kernel, not cohere-megakernel or KVM executor; A100 not H100.
Correctness: fused == two-pass on all 6144 configs; rel L2 vs torch ref 1.4e-5..1.8e-4.

## Real North numbers (MEASURED, Modal A100-80GB, North rev d11e61a8, 15 tasks dev+calib, HF/RefNorth trace)
Raw: `results/routing.jsonl`, `results/routing_traces.json`. Spend ~$0.19 so far.

Real expert routing (48 MoE layers, top-8 of 128):
| quantity | value |
|---|---|
| consecutive-token expert reuse (token t vs t-1, same layer) | **0.533** (Cohere blog: 0.38; independent: 0.0625) |
| distinct experts in a block of K consecutive tokens, K=1/2/4/8/16 | 8.0 / 11.7 / 17.1 / 24.7 / 35.1 (independent would be ~8 / 15 / 28 / 50 / 80) |

Verify blocks already share expert loads strongly. This is the real-routing counterpart to the synthetic "overlap" axis in the microbench (uniform 0.39 / corr75 0.84).

Draft-vs-verifier expert coverage, SAME tokens (teacher-forced). Metric = fraction of the verifier's experts that the draft also loads. This is NOT the same as the fraction of draft loads that fall inside the verify set for FUTURE draft tokens, which is what fusion needs; that cross-token number is not yet measured.
| draft | experts/token | verifier top-8 covered | verifier 8-token expert union covered |
|---|---|---|---|
| top-r=4 | 4 | 0.488 | 0.576 |
| top-r=2 | 2 | 0.236 | 0.344 |
| skip 8 layers | 8 | 0.826 | 0.845 |
| skip 16 layers | 8 | 0.724 | 0.766 |
| skip 8 + r=4 | 4 | 0.464 | 0.551 |

INFERENCE (not measured): real consecutive-token overlap ~0.5 sits between the synthetic uniform (0.39, ~16% MoE-layer saving) and corr75 (0.84, ~27%) rows, so an MoE-layer saving of very roughly 15-25% is plausible. Cycle-level G0 still needs attention/LM-head costs and the cross-token draft overlap.

### Phase 0/1 on real North (eager HF/PyTorch, A100-80GB, bf16, batch 1 unless noted). Raw: `results/*.jsonl`, summary `results/PHASE01.md`
IMPORTANT: eager PyTorch is LAUNCH-BOUND (~97 ms/token vs ~3.2 ms weight-stream floor, ~30x gap). Every timing below is an eager number, not a hardware limit and not a megakernel/CUDA-graph number. G0 (HBM- vs compute-bound) is NOT decided.

| item | result |
|---|---|
| plain greedy decode | 96.6 ms/token (10.4 tok/s) at 2K, 98.7 ms at 8K; batch 8: 120.6 ms/step |
| block-verify K=16 vs 16 decodes | 1.5x cost of one token => eager ceiling 11.8x (2K), 10.7x (8K); launch-bound, optimistic |
| distinct experts/layer by block K=1/2/4/8/16 | 8 / 12.3 / 18.5 / 26.4 / 36.9 |
| bandwidth-floor ceiling (computed from measured expert unions, perfect draft) | ~2.3x K=4, ~3.5x K=8, ~5.4x K=16 (analytic, not a GPU measurement) |
| decode components (126.9 ms CUDA-event timeline, 2K) | experts 71.5, attention 26.7, norm 8.6, router 5.5, lm_head 0.6 |

Self-speculation, greedy, 8 tasks x 64 tokens, gamma=4, zero new weights (eager => wall speedup <1x is expected and says nothing about a fused executor):
| draft | accepted/round | committed/round | wall vs plain | identical to plain |
|---|---|---|---|---|
| top-4 experts | 2.77 | 3.71 | 0.70x | 8/8 |
| top-2 experts | 1.80 | 2.74 | 0.54x | 7/8 |
| 8-layer skip | 2.45 | 3.37 | 0.71x | 5/8 |
| 16-layer skip | 1.50 | 2.48 | 0.59x | 5/8 |
| skip8 + top-4 | 1.94 | 2.87 | 0.52x | 4/8 |
| top-4, gamma=2 / gamma=8 | 1.57 / 4.23 | 2.54 / 5.05 | 0.76x / 0.54x | 5/8 / 4/8 |
8K, top-4, gamma=4: 3.0 accepted/round, token-identical, 6.5 vs 9.6 tok/s.
Divergences: mostly bf16 near-ties (top-1/top-2 margin <0.25), but one 8-layer-skip divergence is unexplained. RefNorth itself differs from HF generate on 3/8 tasks (first mismatch token 41-55), so "identical" holds only up to bf16 ties. Greedy only; no RNG/sampling path.

Reading: a training-free top-4-expert draft accepts ~2.8 of 4 on real North, so draft quality is not the blocker. Cost is: in eager mode a draft step costs about as much as a full step. Whether a fused/persistent executor changes that is the open question and the reason for the H100 phase.

NOT RUN: 32K/128K contexts, batch>1 self-spec, mixed lengths, sampling, vLLM and cohere-megakernel baselines, CUDA-graph timing.

## Simulated / modeled
- Joint optimizer (`docs/OPT.md`): on measured A100 MoE costs plus INVENTED acceptance, plain decode (~146 tok/s modeled) beats most speculative configs; only gamma=1 + fused wins (~+10%, entirely from invented carry-hit 0.6 / accept 0.9). Joint > independent by up to 11.5% in 36/48 synthetic scenarios. Not a speedup claim.
- TAGE (`docs/TAGE.md`): oracle-verifier simulation on local human-written code, Qwen3 tokenizer, NOT North. TAGE proposer does not beat suffix/prompt-lookup; a calibrated accept-gate gives modeled 1.10-1.33x at alpha 0.15 vs 0.92-1.07x gated baselines. Low novelty.

## Built, not yet run on North
- Draft approximations (`docs/APPROX.md`): top-r, EAT-inspired, sparse global attention (oracle top-k, LSA-inspired, SubQ-style approximation), layer skip. 43 CPU tests; verifier path proven untouched on tiny fixtures.
- Self-improvement loop (`docs/SELFIMPROVE.md`): North-only proposer, sandbox, gates, audit, rollback. 95 CPU tests. No North-authored result exists.

## Not done
H100 work, KVM executor port, Phase 5 async speculation live, Phase 7, live self-improvement run.

## Spend
See `results/SPEND.md`.
