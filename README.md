# Northstar
**Result (MEASURED, H100, Cohere MoE "North", greedy):** a fast Triton decode executor (10.7 -> 3.9 ms/token) plus zero-cost prompt-lookup/suffix speculation reaches 3.2-4.4x tok/s over the original graphed baseline. In EXACT mode (batch-invariant kernels) speculative output equals the same executor's plain greedy on 70/70 runs, at 1.18x / 1.57x / 1.34x over its own plain decode (natural / ctx 2K / ctx 8K). Not bit-identical to HF generate. Details: RESULTS.md H9, H10, H13, H14.
**Negative (the original thesis):** model-draft self-speculation (the model drafts for itself, fused with verify) never beat plain decode on any executor, context (up to 128K) or batch size tested: RESULTS.md H5-H8, H11, H12. North has authored no gate-passing optimization yet (docs/SELFIMPROVE_ANALYSIS.md).

Original aim: North improving North: weight-stream-fused self-speculation on a persistent-kernel VM (KVM).
Zero new weights. North is the only LLM that drafts, verifies, and proposes optimizations.
See docs/PLAN.md (brief), docs/THESIS.md (hypothesis + kill gates), PRIOR_ART.md, DESIGN.md, RESULTS.md.
