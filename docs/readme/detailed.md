All numbers are MEASURED on one H100 80GB, greedy decoding, up to 192 new tokens. Full tables, methods and caveats are in [`RESULTS.md`](RESULTS.md).

| Result | Number | Where |
|---|---|---|
| Plain decode, Triton executor vs graphed PyTorch baseline | 10.7 → 3.9 ms/token | H9 |
| **Same launch: baseline vs old fast (non-exact) vs old exact vs exact v2**, ctx 2K / 8K | 10.70 / 3.95 / 4.58 / **3.12 ms** (2.71x / 2.34x / **3.43x**); 8K: 11.38 / 4.06 / 4.64 / **3.33 ms** (**3.42x**) | H29 |
| Executor plus speculation vs original graphed baseline | 3.2–4.4x tok/s | H9, H10, H13 |
| Exact-mode speculation vs the same executor's plain decode | 1.18x / 1.57x / 1.34x (natural / 2K / 8K ctx) | H14 |
| Same, on 99 real prompts from 11 public datasets | 1.15x geomean (82 wins, 15 ties, 2 losses, worst 0.96x) | H15 |
| Batched exact executor, aggregate throughput | 2012 tok/s at 32 concurrent sequences | H20 |
| **Exact v2: attention and MoE branches on concurrent streams + H100-tuned tiles** (same launch, 99 real prompts) | exact plain 237 → 387 tok/s (1.63x), tokens identical to v1 on 99/99; + PLD 449 tok/s, == plain on 99/99 | H24 |
| **Exact v2 vs stock vLLM 0.33, same H100 container** | plain 321.7 vs 248.8 tok/s (1.29x), + PLD 369.8 (1.49x); faster on 99/99 prompts | H24 |
| Exact v2 at long context (same launch per ctx) | plain 8K 214 → 298, 32K 211 → 322, 128K 90 → 187 tok/s | H24 |
| Exact v2, batched | 2219 tok/s at B=32, 308 at B=1; all exactness checks 64/64 | H24 |
| Exact v2 batched vs vLLM 0.33 batched, same container | 1.20x / 1.11x / 1.21x / 1.02x / 1.02x / 1.05x at B=1/2/4/8/16/32 | H25 |
| Speculation that does not lose (cost-aware cap + index built during prefill) | mix99 1.125x, worst 0.98x; 32K 1.08x, 128K 1.08x over v2 plain, all exact | H25 |
| Serving engine (`harness/serve.py`), end to end | 1 user: mix99 368 vs 321 tok/s plain (1.146x, worst 0.974x), all paths exact; batched rows == single-user outputs at B=2..32 | H26 |
| Continuous batching (`harness/cbatch.py`) vs vLLM 0.33 under Poisson arrivals, same container | 1.04-1.08x at 2-8 req/s, 0.99x at 16, 0.91x saturated; every request == its isolated decode (64/64 at every rate) | H27 |
| Continuous batching + batch-invariant batched prefill | saturated 2,246 vs vLLM 1,941 tok/s (1.16x), still 64/64 == isolated decode at every rate | H28 |
| Stock vLLM 0.33 at batch 1 (same prompt mix, separate launches) | plain 253.5 tok/s vs ours exact plain 221.4, ours exact + speculation 254.8 (superseded by the same-container H24 comparison) | H17 |
| North improving its own kernel, closed loop (3 sessions/arm) | North alone 1.026x mean best, North + 4 North helpers called from outside 1.026x (best session 1.056x); 1 of 3 sessions pass the 1.03x gate in each | H22 |

**Exact mode** uses batch-invariant kernels, so speculative output equals the same executor's plain greedy output on 70/70 runs (and 14/14 at every tested context up to 128K). It is **not** bit-identical to Hugging Face `generate`.
