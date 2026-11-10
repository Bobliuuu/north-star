# NEXT_STEPS

## STATUS UPDATE 2026-11-09 c (RESULTS.md H20; supersedes the batch items below)
MEASURED: batched exact executor, full stack bitwise (30/30 cells; 384/384 output checks per bench): plain agg tok/s B=32 2012 (H19 742, H18 636); step 3.9x (T=1) to 5.9x (T=13) vs the H18 path at B=32. PLD at batch>1 is now near break-even (capped 0.99-1.01x of plain at B>=4); B=1 unchanged (~220-233 tok/s, 1.27-1.29x with PLD).
NEXT: (a) cost model for the cap that accounts for masked rows (time the table with realistic valid fractions); (b) B=1: the remaining single-sequence cost is lin (1.8 ms of 4.4, 150 launches incl. lm_head) + MoE (1.6 ms): fuse qkv/o/router launches or port the 64-row lin tiles' lm_head gain to the B=1 path if bitwise; (c) larger B (64+) and longer contexts for the batched path (only <=32 x <=1.7K prompts measured); (d) ragged attention rows (attention still runs all T rows per sequence; masked rows still cost lin_bt and attention).

## STATUS UPDATE 2026-11-09 b (RESULTS.md H19; supersedes the batch items below)
MEASURED: expert-gathered batched MoE is bitwise exact and makes batched verify steps up to 2.2x cheaper; PLD at B=16/32 +29%/+50% agg tok/s. The H18 plain-B>1 divergence was a benchmark bug (unset graph replay overwrote prompt KV), fixed; 384/384 exact in every re-run. Cost-aware draft cap: 1.29x / 1.07x at B=1/2, 0.94-0.99x at B>=4 (plain PLD 0.64-0.92x).
NEXT: (a) ragged verify: rows carry their own T (flat row count, per-sequence attention already per-row), so short-draft rows stop paying for the longest draft (slot utilisation 0.18-0.46 now); (b) cheaper T=2 step at large B (the break-even barrier in H19 (3)); (c) carry the gathered MoE into FastNorth's single-sequence blk path for T>=8 at B=1 only if it wins there (abtile_ab1 says no below 48 rows); (d) serving-loop policy: capped PLD at B<=2, plain above.

## STATUS UPDATE 2026-11-09 (batch>1, RESULTS.md H18)
MEASURED: exact PLD on the batched exact executor wins only at B<=2 (1.27x B=1, 1.05x B=2) and loses from B=4 (0.90x) to B=32 (0.42x): rounds are padded to the longest draft (slot utilisation 1.00 -> 0.15). Plain aggregate saturates at B=16 (~640 tok/s). spec == plain within a batch size 384/384, but plain B>1 != plain B=1 on up to 62/64 rows at B=32 (unresolved; diag run ap-Jj4SEUzRYCVCrArm62vjoS left no data). The bs1 cost table is invalid for T>1 (timed on leftover padded inputs); fixed on branch batch-h18, not re-run.
NEXT (in order, each needs GPU authorization): (a) re-run `modal run --detach modal/batchspec_run.py::main --tag bs2` to get the corrected cost table + host/device split (~$0.65); (b) fetch or re-run batchdiag to locate the B>1 vs B=1 divergence; (c) batch-size gate for PLD (B<=2) in the serving loop; (d) MoE tile on a grid axis in fast_kernels_batch (keeps batch invariance) to lift the B=16 plain ceiling; (e) ragged per-row T verify so short-draft rows do not pay for the longest draft.

## STATUS UPDATE 2026-11-06 (read first; the older text below predates it)
POSITIVE RESULT (MEASURED, H100, RESULTS.md H9/H10/H13): fast Triton decode executor 10.7 -> 3.9 ms/token (plain decode, ~2.7x), plus zero-cost PLD/suffix speculation on top: 3.2-4.4x tok/s vs the original graphed baseline and 1.15-1.54x vs fast plain decode (natural 21 tasks / ctx 2K / ctx 8K). CAVEAT: outputs are NOT token-identical to plain greedy over 192 tokens; every one of 116 checked divergences is a verified near-tie (top-2 margin median 0.125, max 0.5). Quality-equivalent, not bit-exact. EXACT MODE (H14, branch exact-spec, MEASURED): batch-invariant Triton kernels make speculative output EQUAL to the same executor's plain greedy on 70/70 runs (192 tokens; old executor 12/70). Cost: plain 222 vs 259 tok/s (-13%); exact spec 263 / 347 / 289 tok/s (natural/2K/8K) = 1.18x / 1.57x / 1.34x over its own plain, 1.02x / 1.36x / 1.17x over the old non-exact plain. Still NOT equal to HF generate (plain exact == HF on 2/21): matching HF would need HF's own kernels.
NEGATIVE (MEASURED/INFERRED): model-draft self-speculation (H5-H8, H11 megakernel, H12 long-ctx 32K/128K and batch>1) never beats plain decode; best case ~1.0x. Gate G0 (self-spec fused saving >=15%) stays failed except under an unbuilt fused-draft upper bound at gamma<=3.
NEXT: exact (lossless) spec; PLD on the megakernel (3.25 ms; needs multi-token causal verify + KV rollback; design + CPU-tested reference in docs/PLD_MEGAKERNEL.md, kernel not built, no GPU run); fused-draft design; real-traffic task mix for PLD.


Every statement is taken from RESULTS.md, docs/*.md, results/*, RUNS.md or the code. Nothing here is a new measurement. Evidence classes: MEASURED, INFERRED (arithmetic on measured numbers), SIMULATED (invented inputs).

## Where the project stands (one paragraph)
Pre-registered gate G0 (docs/THESIS.md): fused saving >= 15% of cycle time, and verify not compute-bound. Verify is HBM-bound (passes). Fused saving is 3.6-11.7% of cycle time on the measured bases (RESULTS.md H4/H6), so the negative-result branch is the supported reading. No self-speculation config beats plain decode: graphed PyTorch 0.60-0.86x on average (best single task 0.94x), eager 0.52-0.76x. No end-to-end North speedup is claimed anywhere.

## Bottlenecks, in measured order, and what each implies
1. **The draft step is almost as expensive as a full step (largest measured blocker).** Graphed draft 9.9 ms (top-4) / 9.1 ms (skip8) vs 10.7 ms full step (H5): only 8-15% cheaper, because attention (~4 ms), router/norm/embed/residual (~1.8 ms), lm_head and fixed grouped_mm costs do not shrink with top-r; expert bytes are ~55% of the weight stream and top-4 halves only that. Eager is worse (a draft step costs about a full step). Consequence: with sequential drafting, gamma extra steps cost ~gamma full steps whatever fusion does.
2. **Executor fixed costs hide the hardware picture.** Eager HF is ~97 ms/token vs a ~2-3 ms weight-stream floor; the graphed PyTorch executor is 10.7 ms; cohere-megakernel is 3.25 ms (~60% of the HBM floor). All spec numbers so far are on the two slower executors. The megakernel-class projection (0.67-1.20x, RESULTS.md H5) is INFERRED, not measured.
3. **Fusion can only share ONE draft step with a verify pass** (draft tokens are autoregressive), so realistic savings are 3.0-11.7% (H4 table, Td=1 rows). The 15% bar is reached only by constructions that are not sequential self-spec (ideal-batch draft) or that assume non-expert weight sharing (INFERRED, optimistic).
4. **Much of the measured fusion gain is removed launch/tail, not HBM.** Per-layer time saving 27-41% vs weight-byte saving 12-30% (H4); the real-minus-shuffled-control byte saving is only +1 to +4 points (H3), i.e. overlap is mostly generic expert popularity.

## Ordered work (do in this order; each step can falsify the thesis, and that is an acceptable result)
1. **Decide whether to close G0 as a negative result.** The data in RESULTS.md H6 already supports it. Writing it up (docs/POST_DRAFT.md exists) costs no GPU. Items 2-6 only matter if the thesis is to be kept alive.
2. **[acceptance DONE, RESULTS.md H7; draft COST still unmeasured]** Draft step timed with gather-based sparse attention (RESULTS.md H8): NOT cheaper (+3-17%). Remaining: fused sparse kernel and 32K+ ctx (untested). Original text: **Measure a cheaper draft step, attacking blocker 1 first.** The draft approximations that shrink attention (sparse global attention, oracle top-k / LSA-inspired; `src/approx`, 43 CPU tests) are BUILT BUT NEVER RUN ON NORTH. Run `src/approx/modal_sweep.py` (quick preset, A100-80GB, ctx 2K/8K) to get acceptance and draft cost for attention x expert x layer-skip. G2 (docs/THESIS.md: combined must beat best single approximation) cannot be evaluated until this exists.
3. **Measure draft cost and fusion inside the megakernel executor**, not our WMMA microbench (not Cohere's N=8 wgmma tile). Requires modifying or extending cohere-megakernel (pinned 67d0b9ca22ea) or porting to the KVM executor; VM dispatch overhead is unmeasured (`src/kvm/PROVENANCE.md`: the executor was not used). This is the only way to turn the 0.67-1.20x projection into a measurement.
4. **Re-run overlap and spec on longer contexts and more prompts.** Only 22 short prompts x 64 tokens, single seed, for routing/overlap; one microbench kernel.
5. **Batch > 1, sampling.** Greedy only; no RNG/sampling path; distribution-preserving speculative sampling (PLAN.md) not implemented.
6. **Joint scheduling (PLAN.md step 5) only if 2-3 produce support.** The CP-SAT optimizer (`src/opt`, 19 tests) runs on invented acceptance (plain decode beat most speculative configs in the SIMULATED scan); recalibrate it with measured acceptance and the measured draft cost from item 2 before believing any joint-over-independent claim.

## Unrun experiments (explicit list)
- Contexts 32K and 128K (native-context feasibility not checked); batch > 1 self-speculation; mixed-length batches; sampling.
- vLLM baseline; cohere-megakernel self-speculation.
- Draft-approximation sweep on North (`src/approx/modal_sweep.py`): built, not run.
- KVM executor port; Phase 5 (asynchronous speculation live); Phase 7 (concurrent adaptation).
- Sparse-attention/SSA: only independent approximations exist (SubQ's SSA is proprietary; not reproduced).
- Live self-improvement: kernel track structured mode. Two live sessions x 3 rounds = 6 North proposals, all gate_fail:speed (2-13% slower; docs/SELFIMPROVE_ANALYSIS.md). Run 3 (after the prompt/history changes; app ap-B0wZK4IVQWioxxurZCKpLO, completed) gave 3 more proposals, 0 passes: 2 gate_fail:speed (dev 0.882x, 0.983x) and 1 gate_fail:correctness (harness renorm bug, see docs/SELFIMPROVE_ANALYSIS.md); outcome recorded in RUNS.md. No North-authored optimization has passed the gates, so no self-improvement result exists. The policy track has no live run.
- TAGE: oracle simulation on local code with the Qwen3 tokenizer, NOT on North; proposer does not beat suffix/prompt lookup. Not pursued on North.

## Blockers and limits
- **GPU budget.** H100 phase used ~$1.4 / 0.35 H100-h of a 4 h / ~$16 cap; total est. spend ~$2.35 on A100 plus H100 rows in `results/SPEND.md`, cap $30 for the A100 phase (docs/THESIS.md). Each new GPU run needs authorization; use `make bench` to see exact commands (always `--detach`).
- **Hardware not covered:** B200/GB200/NVL72 (mixture-of-kittens target), anything but A100-80GB and H100-80GB. The megakernel was run on H100 only.
- **Weight volume** (`northstar-weights`) must exist; `modal/download.py` populates it (CPU only).
- **Numerical caveats:** RefNorth differs from HF generate on 3/8 tasks (first mismatch token 41-55, bf16 ties); one 8-layer-skip divergence is unexplained. "Token-identical" holds up to bf16 near-ties.
- **Pinning gaps:** several GPU-image packages are unpinned (SETUP.md); GPU results are not bit-reproducible from the repo alone.
- **Plots** cover raw JSONL/JSON only; ceiling.jsonl block-verify rows, tage results and megakernel logs (`.log`, not JSONL) are not plotted.

## What would change the conclusion
A measured cycle-level saving >= 15% in the megakernel/KVM executor with real routing and real acceptance, or a draft step that costs well under the measured 8-15% discount (item 2). Absent both, report the negative result.
