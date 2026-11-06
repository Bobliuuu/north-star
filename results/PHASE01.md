# Phase 0 / Phase 1 results (Modal A100-80GB, bf16, torch 2.10.0+cu128, transformers 5.22.1, HF experts impl grouped_mm, sdpa)
Model CohereLabs/North-Mini-Code-1.0 @ d11e61a8 (Apache-2.0, not gated; 56.8 GiB). Raw: results/*.jsonl, probe.json, routing_traces.json.
Executor for ceiling/spec = our RefNorth (tests/ + probe: logits bit-identical to HF on prompt, 32-token greedy identical).

## Baseline (HF, eager PyTorch, greedy, code prompts + stdlib repo filler)
batch1 2K: prefill 0.29 s (7.0k tok/s), decode 96.6 ms/tok (10.4 tok/s); 8K: 98.7 ms/tok. batch 8 2K: 120.6 ms/step = 66 tok/s; 8K b8 171 ms.
Decode latency is ~flat in batch => CPU/launch-bound, not HBM-bound. Weight-stream floor at b1 = 6.6 GB/2.0 TB/s ~ 3.2 ms (30x below measured).
Natural prompts (130-230 tok, 8 dev tasks): 94-101 ms/tok. Peak mem 57-64 GB. HF eager-loop experts: 168 ms/tok (b1).
32K / 128K cells NOT run (deadline). Batch>1 spec not run.

## Teacher-forced block-verify ceiling (real North tokens; RefNorth; ctx 2K, 8K b1; 2K b4)
K tokens in ONE forward vs K sequential decodes (b1, 2K): K=1 108.7 ms; K=4 134.0 (3.15x); K=8 147.1 (5.87x); K=16 165.2 ms (11.8x). 8K: 136/170/187/201 (K=1/4/8/16; 3.2x/5.8x/10.7x). b4 2K: K=16 10.3x.
These are EAGER ceilings: a K=16 block costs only 1.5x one token => launch-bound; the real GPU work is far below.
Distinct experts/layer over K real tokens (native top-8): K=1 8, 2 12.3, 4 18.5, 8 26.4, 16 36.9 (of 128); top-4: 4/6.5/10.5/15.9/22.4; top-2: 2/3.5/6/9.7/13.6.
Bandwidth-floor ceiling (weights streamed = 3.0 GB non-expert + 48*union*9.44 MB): K=4 2.3x, K=8 3.5x, K=16 5.4x for a free/perfect draft (accept all K). Not measured on GPU; analytic from measured unions.
Components, b1 2K single decode (CUDA-event timeline, includes launch gaps): experts 71.5, attn 26.7, router 5.5, norm 8.6, lm_head 0.6 ms of 126.9 (event run). Profiler device-time numbers in the JSONL are inflated (> wall) - do not use.
Compute-bound? No: K=16 verify moves ~20 GB for ~0.1 TFLOP-class work.

## Routing traces / draft-vs-verify overlap (15 dev+calib prompts, 48 greedy tokens, results/routing_traces.json: [48 MoE layers][token][8 ids])
Consecutive-token top-8 overlap (same layer): 0.53. Verifier top-8 covered by draft: r4 0.49, r2 0.24 (= prefix, by construction r/8 + chance), skip8 0.83, skip16 0.72, skip8+r4 0.46.
Block(8) union coverage (fraction of verifier's expert union that draft union already loads = fusable weight tiles): r4 0.58, r2 0.34, skip8 0.85, skip16 0.77.

## Phase 1 self-speculation (greedy, batch 1, 8 tasks x 64 tok, gamma=4 unless noted; skip layers evenly spaced in [4,46])
config: accepted/round | committed/round | wall speedup vs plain | identical-to-plain tasks
skip8: 2.45|3.37|0.71x|5/8 ; skip16: 1.50|2.48|0.59x|5/8 ; r4: 2.77|3.71|0.70x|8/8 ; r2: 1.80|2.74|0.54x|7/8 ; skip8+r4: 1.94|2.87|0.52x|4/8 ; r4 g=2: 1.57|2.54|0.76x|5/8 ; r4 g=8: 4.23|5.05|0.54x|4/8.
8K ctx (1 task): r4 g4 accepted 3.0, 6.5 vs 9.6 tok/s; identical. Divergence: all divergent runs have a bf16 near-tie (top1-top2 margin < 0.25) at the first mismatch except 1 of 3 in skip8 (see div_margin in JSONL); RefNorth vs HF itself diverges in 3/8 tasks (bf16 shape-dependent numerics) - so "identical" is only up to bf16 ties. fp32 tiny-fixture tests are exact (39 tests).
Speedup < 1 because in eager mode a draft step costs ~ a full step (launch-bound; skipping layers/experts barely changes kernel count) - this says nothing about a fused/graph executor.

## Gate G0 verdict (preliminary)
- Eager PyTorch is launch-bound (33x above weight-stream floor): no HBM-vs-compute bottleneck claim can be made from these timings; need CUDA-graph/megakernel timing (unmeasured).
- Block verification is not compute-bound; cost in the bandwidth-bound regime is driven by expert-union growth (8 -> 37 experts/layer at K=16), capping ideal speedup at ~5.4x (K=16), 3.5x (K=8).
- Fusion saving (draft weight tiles piggybacking on verify loads): draft top-4 experts are a strict subset of verifier's experts for the same token, but across a block draft union covers only 58% of verifier union (r4) / 85% (skip8) -> partial sharing; exact G0 15% test unmeasured (needs GPU-bound executor).
- Acceptance is non-trivial with zero training (r4: 2.8/4 accepted), so draft quality is not the blocker in this eager setup; execution cost model is.
Unmeasured: 32K/128K, batch>1 spec, mixed-length batches, sampling-mode spec, vLLM baseline, cohere-megakernel baseline.
