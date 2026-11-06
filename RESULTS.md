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
- Self-improvement loop (`docs/SELFIMPROVE.md`): North-only proposer, sandbox, gates, audit, rollback. 95 CPU tests at first write (154 pass + 2 skipped on 2026-11-06; NS_SLOW subset: 21 pass). No North-authored result has passed the gates. Live kernel-track North rounds (`results/selfimprove_kernel*`): diff mode 4/4 `invalid_diff` and 3 attempts `no_diff_in_response`; structured mode run 1 3/3 `audit_reject` (audit rule later fixed, disclosed in `docs/SELFIMPROVE.md`); structured mode second run 6/6 `gate_fail:speed` with dev speedups 0.87-0.98x (CI lo below 1.03 threshold; all candidates slower than baseline). Policy track: CPU mock dry run verified end to end (verdict `gate_fail:speed`, 1.00x); live policy run blocked until `results/accept_calibration.jsonl` exists; see `docs/SELFIMPROVE_POLICY_RUNBOOK.md`.

## Not done
H100 work, KVM executor port, Phase 5 async speculation live, Phase 7, live self-improvement run.

## Spend
See `results/SPEND.md`.

## H100 (Gate G0 on real hardware; Modal H100 80GB HBM3; budget cap 4 GPU-h / ~$16, log in `results/SPEND.md`)
Labels: MEASURED = run on H100 in this section; INFERRED = arithmetic on measured numbers. Sub-sections are filled as runs land.

### H1. cohere-megakernel plain decode on North (MEASURED)
Engine `cohere-ai/cohere-megakernel` @ `67d0b9ca22ea` built unmodified (CUDA 13.0.1, torch 2.14+cu130, sm_90a), Modal H100 80GB, bf16, batch 1, `--cpp-decode-runtime`, 132 SMs, greedy, 512 new tokens. Runner `modal/mk.py`, raw `results/megakernel_H100_all_*.log`.
| setup | ms/token | tok/s |
|---|---|---|
| real weights, synthetic KV, ctx 2048 | 3.25 | 308 |
| real weights, synthetic KV, ctx 8192 | 3.38 | 296 |
| uniform-routing synthetic weights, ctx 2048 / 8192 | 3.27 / 3.39 | 306 / 295 |
| real prompt (129 tok, real prefill, real weights) | 3.18 | 314 |
Cohere's published BS=1 number is 292 tok/s at 8K; we reproduce 296 tok/s (+1%). Weight-stream floor 6.6 GB / 3.35 TB/s = 1.97 ms/token, so the megakernel is at ~60% of the HBM floor (Cohere: 62% of their speed-of-light). Eager HF/PyTorch on A100 was 96.6 ms/token, ~30x slower. Caveat: megakernel prefill here is the unoptimised PyTorch path (6.3 s for 129 tokens, includes first-call setup) and is irrelevant to decode numbers.

### H2. CUDA-graphed PyTorch decode and block verify on H100 (MEASURED; HF-module weights via `harness/graph_north.py`, bf16, b1)
Raw `results/h100_blocks.jsonl` (bmm attention; `results/h100_blocks_sdpamask.jsonl` is a first run with mask-SDPA attention, which made attention 2x slower: kept as a sensitivity row). Graph output == RefNorth greedy on 16/16 tokens and block-8 argmax == 8 sequential steps (both ctx). This is a graphed *PyTorch* executor, not a megakernel: K=1 costs 10.7 ms vs 3.25 ms for the megakernel, so its fixed costs (small-kernel latency, HF grouped_mm, unfused attention) are not the hardware floor.
| ctx 2K, K tokens in one forward | 1 | 2 | 4 | 5 | 8 | 9 | 16 |
|---|---|---|---|---|---|---|---|
| graph ms | 10.72 | 12.17 | 13.54 | 14.14 | 15.32 | 16.02 | 17.85 |
| cost vs 1 token | 1.00 | 1.13 | 1.26 | 1.32 | 1.43 | 1.49 | 1.66 |
| verify-only ceiling K*T1/T(K) | 1.0x | 1.8x | 3.2x | 3.8x | 5.6x | 6.0x | 9.6x |
8K ctx: T1 11.41 ms, K=16 20.49 ms (8.9x). The same forward run eagerly takes ~47-54 ms (graphs remove ~4.4x of launch cost). Component split by ablation (ms at K=1/5/9/16, 2K): experts 4.37/6.94/8.48/9.94, attention 4.12/3.96/4.30/3.99, lm_head 0.41 flat, router+norm+embed+resid 1.75/1.96/2.05/2.14.
HBM vs compute (INFERRED from the measured split): the expert term grows with the expert union (8 -> 37 distinct experts/layer, i.e. 3.6 -> 16.8 GB of expert weights) at a marginal ~2.4 TB/s (72% of 3.35 TB/s HBM3 peak); K=16 expert math is ~0.15 TFLOP, far below H100 compute. So verify is HBM-bandwidth-bound in its growing term, NOT compute-bound; the non-growing term (attention 4 ms, ~1.8 ms router/norm/etc.) is kernel-latency overhead of this executor.
Hardware-level data point (MEASURED, megakernel, real weights, ctx 2K, independent sequences = tokens in flight; `results/megakernel_H100_bs_*.log`): step time BS=1/2/4/8 = 3.32/4.10/5.29/7.39 ms (302/490/759/1087 tok/s). Each extra token adds ~0.58 ms avg (17% of BS1), BS8 = 2.2x BS1. A single-sequence verify block shares more experts than 8 independent sequences, so K=8 verify should cost <= 2.2x: ceiling for a perfect free draft is >= 8/2.2 = 3.6x at K=8 (INFERRED), vs 9.6x in the PyTorch-graph executor, whose big fixed floor inflates it.

### H3. REAL cross-token expert overlap (MEASURED, 22 prompts, 64 greedy tokens each, eager RefNorth traces; `harness/bench_overlap.py`, `harness/overlap.py`, raw `results/overlap_traces.json`, `results/overlap.jsonl`)
This is the quantity the earlier same-token table did NOT measure. Verify block = tokens s..s+K-1 (native top-8); draft = the FUTURE tokens s+K..s+K+Td-1 routed by the draft mode (teacher forced, all accepted). `pair` = fraction of draft (token, expert) loads whose expert is already in the verify block's union; `byte` = |U_D & U_V| / (|U_V| + |U_D|) = expert-weight bytes a fused pass avoids vs two passes; `shuf` = control with a distant verify block of the same prompt and layer.
| draft | K (verify) | Td (future draft toks) | pair in verify union | byte saving | shuf pair / byte | same-token pair / byte (old metric) | |U_V| / |U_D| per layer |
|---|---|---|---|---|---|---|---|
| top-4 | 5 | 4 | 0.78 | 0.235 | 0.71 / 0.21 | 0.99 / 0.33 | 19.2 / 9.4 |
| top-4 | 5 | 1 | 0.82 | 0.147 | 0.71 / 0.13 | 0.99 / 0.18 | 19.2 / 4.0 |
| top-4 | 9 | 8 | 0.83 | 0.255 | 0.79 / 0.24 | 0.99 / 0.35 | 26.1 / 14.0 |
| skip8 | 5 | 4 | 0.72 | 0.287 | 0.65 / 0.26 | 0.91 / 0.40 | 19.2 / 17.0 |
| skip8 | 5 | 1 | 0.76 | 0.228 | 0.65 / 0.20 | 0.91 / 0.27 | 19.2 / 8.0 |
Findings: (i) 72-83% of future draft expert loads already lie in the verify union, so the sharing the thesis needs exists, but (ii) most of that is generic expert popularity/skew, not token-specific: a distant block of the same prompt gives 65-79%; the real-minus-control gap in byte saving is only +1 to +4 points. (iii) The byte saving is capped by Ud/(Uv+Ud): ~15% for one top-4 draft token, ~24% for four. (iv) Per-layer spread: top-4 K5/Td4 byte saving is 0.20 in the first 12 MoE layers, 0.25 mid/late (range 0.16-0.28). The earlier "draft covers 0.49-0.83 of the verifier's experts" table is the same-token quantity and overstates what future tokens give.

### H4. Fused vs two-pass MoE with REAL routing (MEASURED, H100, `src/kvm/moe_fused.cu replay`; 25,344 layer-records = 22 prompts x 2 windows x 48 layers x 12 configs; `results/kvm_replay_H100_real_*.jsonl`)
Fused output == two-pass in all records (max rel diff 9.4e-7). Per-layer median time saving, graph-captured: top-4: 41% (K3/Td1), 39% (K5/Td1), 27% (K9/Td1), 37% (K5/Td4), 30% (K9/Td8); skip8: 41%/39%/28%, K5/Td4 only 8%. Fraction of layers >= 15%: 69-99% (skip8 K5/Td4 39%). BUT these per-layer time savings (27-41%) are 2-3x the weight-byte savings (12-30%): in this microbench most of the gain is the removed second launch and tail (the same caveat as the A100 table), which a persistent-kernel executor would not pay in either variant. Do not read 40% as an HBM saving.
Cycle-level (`harness/cycle_model.py`, `results/g0_cycle.json`). Saving = sum over 48 layers of (two-pass - fused) divided by cycle time. Cycle denominators: A = measured graphed-PyTorch cycle (verify K + gamma draft steps, measured H100 numbers from H2); B = measured non-MoE parts + microbench MoE time; C = INFERRED HBM-byte model (time ~ bytes; 2.97 GB non-expert + 9.44 MB per distinct expert; real per-layer union counts from the replay records). `Td=1` = realistic: draft tokens are autoregressive, so only ONE draft step can share a verify layer pass; `Td=gamma` = ideal batch (draft tokens in one pass; not how sequential self-spec works).
| draft, gamma | fused draft toks | A (graph cycle) | B | C: MoE only | C: + non-expert weight sharing (INFERRED, optimistic) |
|---|---|---|---|---|---|
| top-4, 2 | 1 | 11.7% | 11.3% | 7.5% | 23.1% |
| top-4, 4 | 1 | 6.8% | 6.4% | 4.8% | 14.5% |
| top-4, 8 | 1 | 4.0% | 3.6% | 3.0% | 8.6% |
| top-4, 4 | 4 (ideal batch) | 5.1% | 6.2% | 14.9% | 30.7% |
| top-4, 8 | 8 (ideal batch) | 4.2% | 5.4% | 18.7% | 30.9% |
| skip8, 4 | 1 | 5.3% | 5.3% | 6.7% | 14.5% |
| skip8, 4 | 4 (ideal batch) | 3.7% | 4.7% | 17.9% | 30.8% |
(Basis C ideal-batch rows use a baseline that batches all draft tokens in ONE pass, a different and much cheaper unfused system than sequential self-spec; shown only as an upper bound.)

### H5. Self-speculation wall speedup with CUDA-graphed draft + verify (MEASURED, H100, greedy, 64 tokens/task; `results/h100_spec.jsonl`)
Plain graphed decode: 94.4 tok/s (10.6 ms/tok) natural prompts, 92.6 tok/s at 2K ctx. All 6 task-runs per config (4 natural + 2 at 2K) token-identical to plain graphed decode (so no divergence here, unlike eager).
| config | accepted/round | committed/round | wall speedup vs plain graphed (range over tasks) |
|---|---|---|---|
| top-4, gamma=2 | 1.6-1.8 | 2.6-2.7 | 0.82x / 0.86x (natural / 2K) |
| top-4, gamma=4 | 2.7-3.1 | 3.7-4.0 | 0.73x / 0.80x (0.64-0.84) |
| top-4, gamma=8 | 4.6-4.8 | 5.3-5.7 | 0.60x / 0.64x |
| skip8, gamma=4 | 2.6-3.2 | 3.5-4.0 | 0.74x / 0.84x (0.61-0.94) |
No config beats plain decode (best 0.94x on one task). A graphed draft step costs 9.9 ms (top-4) / 9.1 ms (skip8) vs 10.7 ms for a full step, i.e. the draft is only 8-15% cheaper than the model because attention, router, norms, lm_head and the grouped_mm fixed costs do not shrink with top-r; expert bytes are ~55% of a step's weight stream and top-4 halves only that.
Projection to a megakernel-class executor (INFERRED, not measured; plain 3.32 ms, verify cost K=3/5/9 ~ 4.7/5.8/7.9 ms from the measured BS curve, draft step ~2.4 ms from the byte ratio, accept stats from this table): self-spec = 0.93x / 0.82x / 0.67x at gamma 2/4/8; adding fusion (realistic Td=1, basis C MoE-only) 1.00x / 0.86x / 0.69x; adding non-expert weight sharing as well 1.20x / 0.96x / 0.74x. Only the gamma=2 case that assumes free non-expert sharing clears 1x.

### H6. Gate G0 verdict (on this evidence)
- Verify is HBM-bound, not compute-bound (marginal expert-term bandwidth ~2.4 TB/s; megakernel BS8 = 2.2x BS1): that half of G0 passes.
- Fused saving >= 15% of cycle time: NOT met on any measured basis. Realistic single-draft-step fusion gives 3.6-11.7% (bases A and B) and 3.0-7.5% (byte basis, MoE only) at gamma 2-8; max measured 11.7% at gamma=2 where much of it is removed launch/tail. The 15% bar is reached only by upper-bound constructions: ideal-batch draft (not sequential) or sharing non-expert weights too (INFERRED), neither measured.
- Plain North decode on the megakernel is 3.25 ms/token (308 tok/s, ~60% of the HBM floor); graphed self-spec in PyTorch is 0.6-0.86x of plain graphed decode; the byte projection puts a megakernel self-spec at ~0.7-1.0x without fusion. The negative-result branch of the thesis (G0: fused saving < 15% of cycle time) is the supported reading.
Not done / caveats: ctx 32K/128K; batch>1 spec; the KVM executor (VM dispatch cost unmeasured); a fused kernel inside the real megakernel (our WMMA kernel is not Cohere's N=8 wgmma tile); sampling; only 22 short prompts x 64 tokens for routing (single seed), one microbench kernel; PyTorch executor attention at 8K is padded (over-reads sliding layers).
Spend this task: ~$1.4 / 0.35 H100-GPU-hours of the 4 h / $16 cap (see `results/SPEND.md`; includes ~$0.04 of tiny-model debug runs on A100-40GB/H100 and a failed 4 s replay launch).
