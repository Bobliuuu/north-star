
## KVM fused draft/verify

Status: microbenchmark level (`src/kvm/moe_fused.cu`). Not yet run through the KVM executor; VM dispatch cost is unmeasured.
KVM sources (read-only): `~/kvm-wt/integ` @ `fc0ced7` (ABI v2 header copied to `src/kvm/include/kvm_task_abi.h`, see `src/kvm/PROVENANCE.md`).

### Idea
Verification of cycle k and drafting for cycle k+1 both run the same MoE layer. Verify tokens use all 8 routed experts, draft
tokens only the top-r of their own routed set. Instead of two passes (each streaming its experts' weights), one pass streams
each distinct expert once and applies it to every token (verify or draft) routed to it. Saving = weight bytes of experts that
appear in both sets (overlap), plus one fewer launch/tail. Draft routes are the draft token's own, so overlap is a measured
quantity, not assumed (`draft_expert_overlap` in the JSONL; routing modes uniform / corr75 / identical bound it).

### Task descriptors
One work item = (expert e, 64-wide slice s of the 768-wide intermediate dim) = op `MOE_EXPERT_SLICE`, a 64-byte ABI v2
`KvmTaskDesc` whose typed args are `{expert, slice, pair_off, pair_cnt}`; `pair_off/pair_cnt` index a (token_row, routing_weight)
list sorted by expert. Gate/up/down weights of the slice are read exactly once per item and applied to all `pair_cnt` tokens
(tensor-core M tiles of 16; N=8 megakernel tile modeled analytically as `wb_*_n8`). One item produces a partial down-projection
added into `out[token_row]` (fp32 atomics; order nondeterministic, tolerance check). Fusion is purely a work-list property:
the fused list is the union, the two-pass lists are separate batches. `MoeTask` in the .cu mirrors the descriptor.

### Dependency events (version variables, FRAMEWORK.md semantics)
Per layer L and cycle k: `x_L[k]` (input activations), `out_L[k]`. Verify block publishes `out_L[k]` version; the next layer's
items `wait(x_{L+1}[k] >= v)`. Within one fused layer items are independent (indegree 0) and write via atomics, so a per-layer
completion counter task `LAYER_DONE` (indegree = #items) publishes `out_L`. Draft items of cycle k+1 at layer L wait on the
draft-side activation version `xd_L[k+1]` (published by the draft's layer L-1 reduction) and on the *shared weight stream*
only by co-scheduling: the compiler places verify(k) and draft(k+1) items for the same expert adjacent in the ready queue / same
item. A fused item therefore waits on BOTH its token versions (two wait vars; ABI v2 has one wait per task, so the compiler must
insert a join task `JOIN(x_L[k], xd_L[k+1])` that publishes `xj_L`, or ABI v3 adds a second wait slot: open item).

### Buffer versions
KV and activations are versioned buffers: `kv[layer][pos]` has version = (cycle, branch). Verify writes the authoritative
version; draft writes a speculative version in a separate slot range (draft uses approximate attention/experts so its K/V are
NOT reusable by verify unless layers are exact prefix layers; reuse requires identical inputs, weights, masks, positions,
numerical contract). Commit = publish `kv_committed >= n_accepted` (monotone counter); readers use `at_least`. Rejected
speculative slots are never published, so they are simply overwritten.

### Cancellation / rollback
Cancel = bump a per-cycle `epoch` variable; every speculative task carries `exact` read of `epoch` at start (executor checks
exact reads at start/finish) and exits as a no-op when stale; there is no preemption, so an in-flight item finishes but its
publish is discarded (its pub var is a speculative-slot version nobody waits on). Rollback after rejection at position j:
`kv_committed := base + j + 1` (correction token), draft slots > j invalidated by epoch bump, next-cycle drafts conditioned on a
now-wrong prefix are cancelled. Verify results are never speculative; only draft state is discarded, so rollback costs no
recompute of committed state.

### Limits / honest notes
- ABI v2 has a single wait per task and `pub_ver == version+1`; the fused join needs a JOIN task or an ABI extension.
- Megakernel (cohere-megakernel) GEMM tile is N=8 token columns (H100 sm_90a); tokens per expert beyond 8 re-stream weights.
  `wb_fused_n8` / `wb_two_n8` quantify this; our own WMMA kernel handles up to 32 tokens/expert in one tile.
- Draft routing-weight convention is configurable (`--renorm`: w/sum(top-r) vs raw sigmoid scores). Default raw, as top-k is
  taken on raw logits then sigmoid with no renorm (docs/NORTH_ARCH.md).

### Measured (A100-80GB SXM4, bf16, one MoE layer, synthetic routing; results/kvm_fused_A100-80GB_fullsweep_*.jsonl, summary in results/kvm_fused_A100_summary.txt)
Median fused-vs-two-pass time saving over the full r(1..8) x Tv(1..16) x Td(1..16) grid: uniform independent routing 16.2%,
75%-correlated draft routing 27.4%, identical routing (upper bound) 32.7%. CUDA graphs change little (15.8 / 27.1 / 32.4%).
Fraction of grid points with saving >= 15% (gate G0 threshold, MoE layer only): 56% / 89% / 93%. Saving depends on
draft-expert overlap with the verify set (measured 0.39 / 0.84 / 1.0), not on r alone. Fused and two-pass outputs agree on all
6144 configs; CUDA vs torch reference rel L2 <= 2e-4.
