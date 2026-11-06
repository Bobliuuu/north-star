# Thesis
Decode/verify overlap on North gains ~0 (both HBM-bound, share weight streams; see inference-experiment north-branch-overlap doc).
Claim to test: fuse instead of overlap. Draft (top-r experts + sparse global attn, EAT/LSA-driven budgets) for cycle k+1
is scheduled by KVM into the SAME expert GEMM weight-tile loads as verification of cycle k. Draft is ~free in bandwidth.
Versioned KV buffers give rollback. CP-SAT picks budgets/schedule from measured acceptance. Phase 6/7: North proposes
kernel/schedule patches to itself (self-improvement loop), gated by independent harness.
## Kill gates
G0: teacher-forced ceiling; if fused saving < 15% of cycle time or verify compute-bound -> report negative result.
G2: combined (attn x expert) must beat best single approximation net.
## Budget
Modal A100-80GB first (test, <=$30). Then Modal H100 <=4h total (<=~$20). Stop idle. Log every spend in results/SPEND.md.
