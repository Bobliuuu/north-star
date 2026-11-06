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

## Real North numbers
PENDING (baseline agent): routing traces and real overlap, G0 ceiling, greedy tok/s, accepted lengths.

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
