# Draft thread v2 (NOT published; for review). Negative result + what's next.

1/ Northstar: I tried to make Cohere's North-Mini-Code speculate against ITSELF (zero new weights, draft = same checkpoint with fewer experts/layers) and fuse the draft into verification's MoE weight loads. Pre-registered kill gate: fused saving >=15% of cycle time. It failed. Here is the data. github.com/Bobliuuu/north-star

2/ Baseline is real: cohere-megakernel on H100 reproduces 296-308 tok/s (3.25 ms/token, ~60% of the weight-stream floor). Eager PyTorch is ~97 ms/token on A100, 30x off the floor, so any eager speculation result is meaningless. We measured with CUDA graphs and the megakernel.

3/ Drafts are good: a training-free draft using 4 of North's 8 experts accepts ~2.8 of 4 tokens, token-identical to plain greedy on 8/8 tasks. Draft quality is not the problem.

4/ Cost is. A draft step is only 8-15% cheaper than a full step. Graphed self-spec on real North runs 0.60-0.86x of plain decode. Verify blocks are HBM-bound and ALREADY share expert loads: consecutive tokens reuse 53% of their top-8 experts (Cohere's blog: 38%).

5/ The fusion idea: put next-cycle draft tokens in the same expert weight-tile loads as verify. Real cross-token overlap is 72-83%, but a distant block of the same prompt gives 65-79%, so most of it is generic expert popularity. Fused byte saving 15-29% of expert bytes; cycle-level saving 4-12% (gamma 8..2). Gate fails.

6/ Per-layer kernel time saved 27-41%, 2-3x the byte saving. Most of it is just removing a second launch. Easy to mistake for a bandwidth win. Careful with microbenchmarks.

7/ Also negative: a TAGE-style branch predictor as proposer loses to suffix/prompt-lookup (simulation). Useful leftover: a calibrated accept-probability gate for block length.

8/ Caveats: 22 prompts x 64 tokens, one seed; my WMMA kernel, not Cohere's wgmma tile; no KVM executor; no 32K+ context or sampling. Everything labeled measured vs inferred in RESULTS.md. Raw JSONL in repo.

9/ Next: North runs at ~60% of its memory floor. Can North tune its own kernel, with North as the ONLY proposer and an independent held-out gate? Built a sandboxed loop (hash-protected harness, reward-hacking audit, rollback, hash-chained log). Live results so far: 12 North attempts. Ran out of tokens mid-reasoning (x2); malformed unified diffs (wrong hunk counts, x3); one edited a protected line and was rejected; then with a structured-edit format, North's parameter tunings were CORRECT and held-out-quality-neutral but 3-14% SLOWER, rejected by the speed gate. No improvement yet. [update with final rounds]

10/ Also: my own audit had a bug (it read "(<= 16)" in a code comment as shape-specialization) and rejected North's first valid patches. Fixed after the fact, disclosed, rejected run kept in the repo.
