# Draft thread v3 (NOT published; for review). Current result only. Tracks README.md top paragraph.

1/ Northstar: fast decode for Cohere's MoE "North" on one H100, greedy. A fast Triton decode executor (10.7 -> 3.9 ms/token) plus zero-cost prompt-lookup/suffix speculation reaches 3.2-4.4x tok/s over the original graphed baseline (MEASURED). github.com/Bobliuuu/north-star

2/ Exact mode: with batch-invariant kernels, speculative output equals the same executor's plain greedy on 70/70 runs, at 1.18x / 1.57x / 1.34x over its own plain decode (natural / ctx 2K / ctx 8K). It is NOT bit-identical to HF generate.

3/ Realistic prompts are more modest: on a fixed-seed mix of 99 real prompts from 11 public datasets, exact-mode PLD gives 1.15x geomean (82 wins / 15 ties / 2 losses, worst 0.96x). Copy-heavy QA and code-edit 1.3-1.4x; summarization and chat 1.0-1.1x.

4/ Negative result on the original thesis: model-draft self-speculation (the model drafts for itself, fused with verify) never beat plain decode on any executor, context (up to 128K) or batch size tested. Details: RESULTS.md H5-H8, H11, H12.

5/ Self-improvement: North as the only proposer of kernel edits, behind a sandboxed loop with hash-protected harness, held-out gates, reward-hacking audit and a hash-chained log. North has authored no gate-passing optimization yet, so there is no self-improvement result. Its structured edits were correct but measured slower than the baseline (docs/SELFIMPROVE_ANALYSIS.md). A human one-factor-at-a-time sweep found KC=64 alone is a ~5.5% kernel win (MEASURED), which North did not find.

6/ Caveats: single H100 (kernel sweeps on A100-80GB), greedy only, one model. Every claim is labeled MEASURED / INFERRED / SIMULATED in RESULTS.md; raw JSONL and logs are in the repo.

(Earlier runs, including the first malformed-diff attempts and an audit bug I fixed and disclosed, are kept in the repo history and RUNS.md, not in this thread.)
