# Northstar — implementation brief for Claude

You are implementing a research prototype, not merely writing a proposal. Build an adaptive self-speculative executor for CohereLabs/North-Mini-Code-1.0. North is the only LLM used to generate optimization proposals, draft tokens, supervise training, or verify output. Normal compilers, profilers, solvers and tests are permitted. Human-written harness and kernel infrastructure are permitted. Do not silently use another coding model to generate candidate optimizations and attribute them to North.

## Central hypothesis
Jointly choosing North's approximate draft computation and its GPU execution schedule improves committed tokens/second over independently optimized drafting and scheduling. Draft decisions include attention budget, expert budget and optional layer skipping. Scheduling decisions include tile implementation, work ordering, legal overlap, checkpoint placement and memory reuse. These are hypotheses; neither combining papers nor adding a solver establishes novelty.

Target: same checkpoint, cheap draft execution and full authoritative verification. Preserve the chosen reference North computation. Begin with greedy decoding and explicitly defined numerical tolerances; later implement and validate standard distribution-preserving speculative sampling. Never claim bitwise identity from mathematical speculation alone. Approximate attention/expert pruning is allowed in the draft, never silently in the verifier.

## Constraints and architecture
Inspect pinned official model config, upstream model implementation, checkpoint license and Cohere's engine before designing changes. Record model revision, engine commit, dependency versions and GPU. North has parallel attention/MoE branches; preserve their dependencies. Distinguish sliding-window and global layers, positional conventions, sigmoid router/top-k weights, dense prefix layer and KV state. Do not infer the whole network from a generic Transformer diagram.

No second pretrained draft model. Initially no new learned weights: use subgraphs/modes of North. Later, optional adapters or small control heads trained only from North traces require explicit labeling as additional parameters. Freeze base/verification weights. If this violates the user's desired interpretation, retain the zero-new-weights path as primary.

## Sources and prior art to inspect
- Model: https://huggingface.co/CohereLabs/North-Mini-Code-1.0
- Engine: https://github.com/cohere-ai/cohere-megakernel
- Engine explanation: https://cohere.com/blog/megakernels
- LSA paper: https://arxiv.org/abs/2610.42147
- LSA code: https://github.com/Prefrontal-AI/latent-sparse-attention
- EAT (Expert Account Tracker): https://arxiv.org/abs/2609.33614
- SubQ: https://subq.ai/how-ssa-makes-long-context-practical and https://subq.ai/subq-1-1-small-technical-report
- SparseSpec: https://arxiv.org/abs/2512.01278
- LayerSkip: https://aclanthology.org/2024.acl-long.681.pdf
- Cascaded self-speculation/CAS-Spec, CLaSp and KnapSpec: locate original papers and implementations; compare explicitly.
- Speculative Speculative Decoding/Saguaro: https://arxiv.org/abs/2603.03251
- MoE cost-aware speculation/EcoSpec: https://arxiv.org/abs/2607.12696
- FlashMap: https://github.com/al-jshen/flashmap (related architecture ideas; separate drafter is outside primary constraints).
- MoE training kernels: https://github.com/cursor/mixture-of-kittens (NVL72 target; not a turnkey single-GPU dependency).
- Carryover Drafting and online speculative training: locate originals before claiming novelty.

Maintain PRIOR_ART.md: mechanism, assumptions, exact/approximate semantics, code availability, closest overlap, proposed distinction. SubQ's advertised SSA is proprietary; inspect public disclosure, identify reproducible pieces, and label independent approximations accurately. Do not claim an SSA reproduction from marketing material.

## Phase 0 — reproducible baseline and ceiling
Inventory compute and access; never spend on remote GPUs without existing authorization. Install compatible dependencies in isolation. Run official North and engine baselines on supported hardware. If no GPU, implement CPU correctness harness, graph simulator and reproducible GPU scripts, but report GPU performance as unmeasured. Never replace real execution with synthetic timing without labeling it.

Pin workloads: code completion, multi-turn terminal traces, reasoning/prose and long-context repository tasks. Split development, policy calibration and held-out evaluation by repository/task. Sweep batch 1/2/4/8 and feasible 2K/8K/32K/128K contexts; mark infeasible cells. Do not extend native context and attribute results to unchanged North.

Measure prefill, decode, expert operations, attention, LM head, routing, launch/control costs, memory, accepted-length distribution and full end-to-end latency. Use teacher-forced real-token block verification to estimate the optimistic speed ceiling before investing in draft adaptation. Report synthetic oracle separately.

Exit gate: identify bottleneck and plausible net speedup. If full verification dominates, increasing draft acceptance may have little value.

## Phase 1 — correct self-speculative reference
Build ordinary autoregressive North and a simple self-draft path (e.g. fixed layer skipping) sharing immutable weights but separate logically correct caches. Implement rejection rollback, correction token, bonus token, position/mask handling and independent RNG accounting. Validate all-accept, reject-first/middle/last, long-context transitions and mixed sequence lengths. Compare against reference on actual North, using tiny random fixtures only for exhaustive algorithm tests.

Instrument draft and verifier separately. Record raw time, committed tokens including correction/bonus, draft overhead and recovery overhead. A CPU/PyTorch reference comes before megakernel optimization.

## Phase 2 — two-axis approximation experiments
A: Expert budget. Start with top-r of North's native selected top-8, varying r, and explicit routing-weight conventions. Add EAT-inspired history/threshold policy only after reproducing its definitions. No assumption that fewer expert activations yield proportional latency savings.

B: Attention budget. Start with simple reference sparse selection, then LSA-inspired hierarchical indexing if justified. Account for building/updating the index, lookup, gather, metadata and training. An all-keys scan to select top-k is not subquadratic attention. Prefer global layers as an initial test; measure sliding-window layers separately because baseline attention is already bounded.

Run factorial full / attention-only / expert-only / combined configurations with identical proposals/evaluation protocol as appropriate. Measure interactions between acceptance loss and execution cost. Train nothing initially. If sparse draft attention destroys acceptance, investigate North-only adapter training as a secondary track, with costs counted.

Exit gate: combination must beat the best individual approximation in measured net performance or offer a demonstrable memory advantage. Publish negative findings.

## Phase 3 — joint optimizer
Represent each supported draft+verify cycle as a dependency DAG with implementation choices, task durations, SM/resource needs, scratch memory, checkpoint lifetimes and valid reuse edges. Acceptance estimates come from calibration data, not invented analytic probabilities. Profile contention; isolated task times are not additive under overlap.

Objective: expected committed tokens / expected total cycle time under memory and latency constraints. This ratio and empirical acceptance do not form a plain LP. For small candidate sets enumerate configurations and optimize schedules via MILP or CP-SAT, or use a justified fractional-programming transformation. Use LP relaxation for bounds where meaningful. Solve offline or occasionally, cache policies by workload bucket; do not put a large solver in every token step.

Compare: fixed baseline; draft-only optimization; schedule-only optimization; independently optimized combination; joint optimization. Report solve time and sensitivity to cost-model error. Validate predicted schedules on hardware; use prediction error to refine profiles.

## Phase 4 — KVM/megakernel implementation
Use actual existing KVM source only after locating and inspecting it. Do not pretend a fresh task-queue mock is the user's validated framework. Begin by extending Cohere's engine or one isolated North layer; preserve a working baseline.

Expose task descriptors, dependency events, buffer versions and cancellation. Implement supported fused expert operations, sparse attention/indexing, acceptance, KV commit/rollback and next-cycle selection incrementally. Start one speculative cycle per launch; persistent multi-cycle execution is a separate milestone. Measure against CUDA graphs and ordinary kernels, since VM dispatch, register pressure and barriers may lose.

Exact reuse requires identical inputs, weights, masks, positions and numerical contract. Early unchanged prefix layers can be reusable; arbitrary downstream states after an approximation generally cannot. Track invalidation explicitly. Never claim missing expert computation alone repairs downstream approximate activations.

## Phase 5 — optional asynchronous self-speculation
While full North verifies, approximate North prepares a bounded set of next continuations conditioned on possible outcomes (Saguaro-related). Use one checkpoint with versioned caches. Optimize branch allocation against GPU bandwidth, memory and opportunity cost. Compare serial vs concurrent using the same device budget. Kill on negative net benefit; more branches are not free.

## Phase 6 — North-authored kernel improvements
A fixed North agent receives source, profiles and development results. It proposes bounded changes to schedules or kernels. Compile and run candidates in isolated processes; enforce timeouts. Keep tests, benchmark harness and held-out data outside its editable scope. Independently evaluate numerical correctness, real-task quality and repeatable speed. Log all failed proposals, model responses, diffs and experiment costs. Promote validated versions between requests; retain rollback. Distinguish North-authored changes from developer infrastructure.

## Phase 7 — optional concurrent adaptation
Serve full North while training a North-derived adapter/control policy or searching configurations in the background. Stable serving versions plus candidate versions; publish memory accounting. Admission policy honors tail-latency constraints. Benchmark kernels in controlled windows before contention tests. Compare sequential adaptation, static sharing and adaptive sharing with equal GPU-time budgets. Claim benefit only after total saved inference time repays training/search cost over a stated horizon.

## Additional optimizations to evaluate after profiling
- Prefix reuse and code/prompt lookup as cheap proposal baselines.
- Adaptive draft block length; disable speculation when unprofitable.
- Fused routing, permutation and expert projections; weight reuse across verified tokens.
- LM-head/logit/sampling costs and fused acceptance kernels.
- KV cache precision/layout, paged sparse gathers and index maintenance.
- Supported low-precision expert kernels, with target precision held fixed across comparisons.
- Context/batch-specific compiled schedules and buffer plans.
Do not add every optimization. Select by measured bottleneck and retain ablations.

## Deliverables
Working source, tests, pinned setup, one-command correctness and benchmark runners, raw experiment JSONL, reproducible plots, PRIOR_ART.md, DESIGN.md, RESULTS.md, and NEXT_STEPS.md. Document unsupported hardware and unrun experiments. A live demo should show real North output, active execution mode, committed throughput, acceptance, memory and cumulative optimization cost. Do not fabricate traces or claim 10x from attention-only microbenchmarks. Report exact baseline, hardware, precision, context, batch, prompt distribution and whether prefill/training are included.

## Immediate work order
1. Inspect official source and closest papers; produce a novelty map.
2. Establish baseline and oracle ceiling.
3. Implement correct self-speculation.
4. Run attention x expert factorial sweep.
5. Proceed to joint scheduling only if measurements support it.
Continue implementation autonomously through runnable, reviewable milestones. Clearly state blockers and uncertainty; do not equate a simulation or untested adapter with a performance result.
