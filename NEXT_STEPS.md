# NEXT_STEPS

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
- Live self-improvement: kernel track structured mode. Two live sessions x 3 rounds = 6 North proposals, all gate_fail:speed (2-13% slower; docs/SELFIMPROVE_ANALYSIS.md). Prompt/history changes were made after that analysis; run 3 was launched detached (RUNS.md, app ap-B0wZK4IVQWioxxurZCKpLO, log `logs/selfimprove_structured_run3.log`) but its outcome is NOT recorded in the repo. Pull it with `modal run modal/selfimprove_eval.py::pull` only if its store was a Volume store; otherwise check the app in the Modal dashboard. No North-authored optimization has passed the gates, so no self-improvement result exists. The policy track has no live run.
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
