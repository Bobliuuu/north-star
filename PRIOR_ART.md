# PRIOR_ART.md

Retrieved 2026-11-05. Evidence level matters: for most arXiv items I read the abstract/landing page (through a fetch summarizer), NOT the full text; only the model card, config, upstream modeling code, megakernel README/blog and repo clone were read in full. LayerSkip is described from prior knowledge (the ACL PDF fetch returned unreadable binary). "Not stated" means not stated on the page I read, not that the paper omits it. Verify the full texts before any novelty claim goes in a paper.

Fields: Mechanism / Assumptions / Exact vs approx / Code / Closest overlap / Proposed distinction.

## Our claim under test
Fuse draft tokens (top-r experts, sparse global attention, optional layer skip, all from North's own weights) into the SAME expert-GEMM weight-tile loads used by full verification, scheduled by a task-graph VM (KVM/megakernel), with versioned KV, and budgets/schedule chosen by CP-SAT from measured acceptance. Verification stays full and authoritative.

## Novelty verdict (honest)
- I did not find a published or released system that fuses draft tokens and verify tokens into the same MoE expert weight-tile loads inside a single kernel / task graph. Searches (several phrasings, standard + extended) returned only the neighbours below. Absence of search hits is weak evidence; a full-text pass over S2-MoE, SpecMoE, SS-MoE, SparseSpec and Cohere's own follow-ups is still needed.
- The idea space is crowded and the "weight streams are shared" premise is already published as MOTIVATION: Cohere's blog (below) shows verification adds little weight loading at moderate batch, and SparseSpec batches draft and verify in one unified scheduler. So the defensible novelty is the specific combination (tile-level sharing in a persistent kernel + self-draft subgraph of the same checkpoint + joint solver), not "self-spec on MoE" or "share weight loads".
- Physics caveat that limits the claim: a draft token run through reduced/sparse North has different hidden states than the verify tokens, so its router picks experts from ITS activations. Sharing a tile load only helps where the draft token's chosen experts are already in the verify union. top-r experts is a subset of the draft's own top-8, not of the verifier's set. A "route draft to experts already loaded" policy (EcoSpec / S2-MoE / MoE-Spec style) changes the draft further. Also draft cycle k+1 depends on verify-k outcome, so same-pass fusion needs speculation over outcomes (Saguaro-like) or drafting from the pre-verify prefix.
- Platform caveat: megakernel GEMM tile is N=8 columns (`gemm-n8-wgmma.cuh`), sm_90a only. Fusion gain is bounded by the free columns left after verify tokens x batch.

## Model and engine

### North Mini Code 1.0 (HF CohereLabs/North-Mini-Code-1.0, rev d11e61a8)
- Mechanism: 30.5B BF16 sparse MoE, 49 layers, 128 experts top-8 sigmoid router, parallel attn+MoE block, 3:1 sliding/global, dense layer 0. Full details in docs/NORTH_ARCH.md.
- Exact/approx: n/a. Code: Apache-2.0 weights; modeling in huggingface/transformers `cohere2_moe` (v5.x). 
- Overlap: it is our target. Distinction: n/a.

### cohere-megakernel (github.com/cohere-ai/cohere-megakernel @67d0b9ca) + blog cohere.com/blog/megakernels
- Mechanism: one persistent kernel, one block per SM, host-built task list, tiles of GEMM/attention/routing/MoE, integer-counter dependencies in global memory, controller/producer/storer/consumer warp roles, static round-robin schedule plus work-stealing queues for attention and MoE, weight prefetch ahead of activations. TK-style wgmma N=8 tiles.
- Assumptions: decode is bandwidth-bound; H100 sm_90a, BF16, batch 1-8, decode only, North-specific schedule. Blog quotes 292 tok/s BS1 (62% of 470 tok/s SoL), 1.25-1.41x over vLLM; synthetic KV in microbenchmarks.
- Exact/approx: same math as reference (SciCode 38.9 vs 38.2, LCB 70.3 vs 70.3 reported; not bitwise).
- Code: Apache-2.0, public. No speculative decoding anywhere in README/blog/source (grep).
- Overlap: it is the substrate; its blog says MoE tiles fill SMs idle in the attention tail, i.e. intra-step overlap already exists. Distinction: we add cross-token (draft+verify) weight-tile sharing and cancellation/versioned KV; the engine has neither. Our thesis note says plain decode/verify overlap gains ~0; fusion is the proposal.

### Cohere blog "Why MoE Models Get More From Speculative Decoding" (cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding)
- Mechanism/analysis: for MoE, verifying K+1 tokens costs Tt(K+1)/Tt(1); in the moderate-batch sweet spot decode already loads nearly all experts so verification adds little weight traffic. Cohere MoE (N=128,k=8): adjacent-token expert overlap 0.381 vs 0.118 independent; ~20.4 unique experts for 4 verify tokens at BS4 (a different Cohere MoE, vLLM). Measured Tt(1)/Tt(4)=0.80 at BS1, 0.65 at BS2; separate small draft model costs ~15% of a target step.
- Exact: standard SD (lossless). Code: none (blog).
- Overlap: STRONGEST motivation overlap. It establishes that verify tokens already share expert weight loads (our "free" resource) and gives routing-correlation numbers to compare with our own North measurements. It does not fuse a draft into those loads and uses a separate draft model. Distinction: we put the draft computation itself into the shared loads.

## Sparse attention / expert budget

### Latent Sparse Attention (arXiv 2610.42147; github.com/Prefrontal-AI/latent-sparse-attention)
- Mechanism: keys hierarchically clustered in a learned latent space; query scores the cluster hierarchy and attends a selected subset. D-LSA distilled onto pretrained models (Qwen3.8-27B, GLM-5.3-Flash), N-LSA trained from scratch. Reports 10M-token per-layer speedups (decode 4.6x/42.5x, prefill 22.7x/72.4x), measured on H200.
- Assumptions: needs learned distilled maps per model (no North map exists); long contexts (>=1M) where gains show. Attention kernel unchanged ("only decides which keys it reads"). Python 3.12, CUDA 12.9 torch, SGLang fork.
- Exact/approx: approximate sparse selection; not used for speculation in the abstract.
- Code: Apache-2.0 repo with training/distill/eval, SGLang fork tag lsa-v0.1.0, D-LSA weights for two non-North models.
- Overlap: gives the indexing idea for draft attention on global layers. Distinction: we use it only inside the draft; verification stays full attention, so approximation never reaches output. LSA map for North would need new weights (extra-parameter track, must be labeled). North's 13 global layers (12 NoPE) are the only candidates; sliding layers are already bounded.

### EAT, Expert Account Tracker (arXiv 2609.33614)
- Mechanism: history-aware expert importance metrics + adaptive thresholds to activate fewer experts per token (>25% fewer activated experts vs vanilla; faster than Top-P baseline); pruned models recovered with ~9K data via OPD.
- Exact/approx: approximate (changes the model); abstract-only read, definitions not reproduced yet. Code: not found on page.
- Overlap: expert-budget policy for the draft. Distinction: used only in draft, verify keeps all 8; we first reproduce its definitions before using the name (per PLAN).

### SubQ SSA (subq.ai/how-ssa-makes-long-context-practical, subq-1-1-small-technical-report)
- Public disclosure: content-dependent selection of positions, attention exact over selected positions, selection independent of absolute position; trained by swapping SSA into an unnamed open-weight model then ~1T tokens long-context continued pretraining; 1M-token claims (e.g. 56x vs FlashAttention-2 on one layer, B200).
- Withheld: selection algorithm, architecture, kernels, params, base model. No code, no weights; hosted/partner access only. Page numbers are internally inconsistent (56.2x vs 52.2x at 1M).
- Reproducible pieces: only the generic idea (learned content-based sparse selection). Anything we build is an independent approximation and must be labeled "SSA-inspired", never "SSA reproduction".

## Self-speculation (layer skip / sparse draft)

### LayerSkip (Elhoushi et al., ACL 2024; from prior knowledge)
- Mechanism: layer dropout + early-exit loss during training, early exit as draft, remaining layers verify and reuse the draft's KV/activations (shared compute and cache). Lossless verification.
- Assumptions: requires continued training/finetuning; a stock checkpoint like North has no early-exit-trained layers. Code: released (Meta). 
- Overlap: exact reuse of early-layer computation between draft and verify. Distinction: North was not trained for it, so draft quality of skipping is an empirical question; our reuse is limited to unchanged prefix layers (PLAN Phase 4).

### SparseSpec (arXiv 2512.01278)
- Mechanism: same model drafts with sparse attention (PillarAttn, critical tokens picked from verification-stage info) and verifies; unified scheduler batching draft and verify, delayed verification overlapping CPU/GPU, dynamic KV management. Up to 2.13x throughput.
- Target: long CoT, dense models, attention-bound; exactness of output not stated on page (standard SD verification implies lossless). Code: not on page.
- Overlap: CLOSEST on "batch drafting and verification together" and sparse-attention self-draft with verify-derived selection. Distinction: it fuses at the scheduler/batch level for attention-bound dense models; it does not address MoE expert-tile sharing, expert budgets or a task-graph kernel. Must read full text to see whether "unified batching" shares weight loads (for dense GEMMs the weights are shared by construction in any batch).

### CAS-Spec (arXiv 2510.26843, NeurIPS 2025)
- Mechanism: cascade of self-drafts from Dynamically Switchable Inference Acceleration strategies (layer sparsity, activation quantization), Dynamic Tree Cascade routing by acceptance and latency predictions; 1.1-2.3x, Vicuna 7B-33B. Training-free, lossless. Code: not verified. Overlap: multi-level self-draft + adaptive routing. Distinction: dense models, no hardware-schedule co-optimization.

### CLaSp (arXiv 2505.24196, ACL 2025)
- Mechanism: per-step dynamic-programming choice of skipped layers using the last verify's hidden states as objective; training-free, lossless; 1.3-1.7x on LLaMA3. Code not verified. Overlap: layer-skip policy. Distinction: dense, DP not hardware-aware, no expert axis.

### KnapSpec (arXiv 2602.20217, ICML 2026)
- Mechanism: layer selection as knapsack maximizing tokens/time, attention and MLP modules skipped separately, context-length-aware, measures hardware speed of sub-components on the fly, parallel DP; up to 1.47x on Qwen3/Llama3. Training-free. Code not verified. 
- Overlap: CLOSEST on "optimize draft config by acceptance/time with hardware cost" (our joint objective). Distinction: it picks which modules to skip using timing, but not MoE expert budgets, not intra-kernel schedule/overlap/weight-tile sharing, and its cost model is additive per-module time, which PLAN says is invalid under overlap.

### Saguaro / Speculative Speculative Decoding (arXiv 2603.03251, ICLR 2026; Kumar, Dao, May)
- Mechanism: while target verifies, draft pre-computes speculations for likely verification outcomes; hit returns instantly. ~30% over optimized SD, up to 5x over AR. Separate draft model, runs on separate resources (parallel). Exactness not stated on page. Code not on page.
- Overlap: Phase 5 asynchronous branch preparation. Distinction: ours would share a single device's weight streams (fuse) rather than occupy separate hardware; thesis says same-device overlap gains ~0 on North.

### Carryover Drafting (arXiv 2609.14717)
- Mechanism: keeps rejected target hidden states as temporary KV context the drafter can attend to; one learned embedding marks them; bounded to one block; train with parallel draft-verify-draft rounds. +6.5-14.7% acceptance length, +7.9-14.4% vLLM speedup. Needs a trained separate drafter. Code not on page.
- Overlap: only the idea of recycling rejected verify states; relevant to whether a North-only control head could use them. Outside primary constraint (no new weights).
- Related online-training works found: Online Speculative Decoding (ICML 2024), Draft, Verify, and Improve (arXiv 2510.05421), TIDE (2602.05145), Draft-OPD (2605.29343), Online Draft Co-Training (2609.07108). All train a drafter; relevant only to the optional Phase 7.

### EcoSpec (arXiv 2607.12696)
- Mechanism: for MoE SD, add predicted marginal expert activation cost (lightweight expert predictor, dynamic expert buffer of experts already covered) to draft-tree selection to avoid "expert scattering"; up to 1.62x on DeepSeek-V3.1, Qwen3-235B, GPT-OSS-120B. Verification rule unchanged.
- Overlap: CLOSEST on "verification's expert weight traffic is the cost; choose drafts that reuse experts". Distinction: it selects WHICH draft tokens to submit; the draft source is external (EAGLE-style); no draft computation inside the shared expert tiles; no kernel co-design.

### Other MoE speculation neighbours found while searching for the fusion claim
- MoE-Spec (2602.16052): fixed per-layer expert budget at VERIFICATION (approximate; trades quality), 10-30% over EAGLE-3. Opposite stance to ours: we keep verifier full.
- AcceptMoE (2608.02989): verifier-side restricted expert sets, explicitly changes the distribution (-0.27 pt accuracy), 1.29x SGLang BS1. Same caveat.
- S2-MoE (2608.15018, code github.com/angerybob/S2-MoE): self-speculative MoE on edge; reuse-aware gating (soft routing bias so verify tokens activate fewer distinct experts) as a fused CUDA kernel; "controlled target-routing approximation, not lossless". It fuses gating, not draft/verify weight tiles.
- DraftExpert (2607.24434): self-speculation for offloaded MoE with shared+top-1+per-layer trained draft expert, target prefetching; verification exact; ~1.45x on DeepSeek-V2-Lite/Moonlight. Adds trained weights; offloading regime.
- SpecMoE (2604.10152) and SS-MoE (ACM 10.1145/3774904.3792218): draft = target non-expert layers + few pinned experts / subset of routed experts, sharing parameters with the target; offloading/edge; approximate draft. Closest in "draft = subset of experts of same model"; neither reported tile-level sharing of weight reads in what I read.
- SpecFit (2610.15417) and "Limits of Speculation" (2609.22156): cost of expert loading in verification; budget selection / bounds.
These are the papers a reviewer will cite against us. Required follow-up: read S2-MoE, SpecMoE, SS-MoE and SparseSpec full text for any kernel-level sharing.

### FlashMap (github.com/al-jshen/flashmap)
- Mechanism: separate ~0.5-0.8B drafter proposing a block in one pass via a flow map from noise (diffusion-style); lossless SD; up to 6x on Qwen3-8B. MIT code, Apache weights for Qwen. No MoE or tile-sharing discussion. Outside primary constraint (separate drafter); referenced only for block-draft ideas.

### Mixture-of-Kittens (github.com/cursor/mixture-of-kittens)
- Deterministic MoE TRAINING megakernel for NVL72 (SM100/103), fuses MoE compute and communication, BF16/MXFP8, Apache-2.0. README does not describe single-GPU decode or draft/verify. Not usable as a dependency on H100/A100; ideas only (overlap granularity).

## Gap map (what is and is not covered)
| Axis | Covered by | Not covered by anyone found |
|---|---|---|
| Self-draft from same checkpoint | LayerSkip, CLaSp, KnapSpec, CAS-Spec, SparseSpec, SpecMoE, SS-MoE, DraftExpert | North/cohere2_moe, parallel-block, sliding+NoPE-global specifics |
| MoE verify expert traffic | Cohere blog, EcoSpec, MoE-Spec, AcceptMoE, S2-MoE | draft tokens executed inside the same tiles |
| Joint draft config + hardware cost | KnapSpec (timing DP), CAS-Spec (latency predictor) | CP-SAT schedule with resource contention, versioned-KV rollback in a task VM |
| Persistent megakernel decode | cohere-megakernel | any speculative path in it |
| Async branch prep | Saguaro | same-device weight-sharing variant |

## What we must measure to make the distinction real
1. North's own routing overlap between draft and verify tokens (fraction of draft top-r experts already in the verify union), by layer.
2. Fraction of verify cycle time that is expert weight streaming vs compute at the chosen block size (Gate G0: >=15% saving or negative result).
3. Acceptance of top-r/sparse-global-attention draft of an unmodified North (zero new weights).

## TAGE-style tagged geometric-history predictors for self-speculation (added by the src/tage agent, 2026-11-05)
Evidence level: web-search snippets and abstract/landing pages only, NOT full texts. The Saguaro PDF fetch returned compressed bytes, so Saguaro cache/bonus-token details below are NOT verified beyond the abstract. Design in docs/TAGE.md; simulator in src/tage (oracle-verifier simulation, not a North result).

| Work | What it does | Overlap with TAGE-for-spec | Verdict |
|---|---|---|---|
| TAGE (Seznec & Michaud, JILP 2006; from prior knowledge, not re-read) | Branch predictor: tagless base table + tagged tables indexed by hashes of geometrically growing global-history lengths; longest tag-hit provides; allocate on mispredict in longer table; useful bits. | It is the mechanism we borrow. Patents/CBP papers apply it to branches and load-value prediction (search hits only). | Source of the idea, not prior art on LLMs. |
| Search: TAGE x speculative decoding (3 searches incl. one extended) | Returned only CPU branch-predictor/patent results. | No hit applying TAGE-like tagged geometric history tables to draft proposal, accept prediction, or expert/budget selection. | Not found. Weak evidence of novelty (absence of hits); a reviewer may find an unindexed paper. |
| Saguaro / SSD (2603.03251) | While verify runs, draft pre-computes speculations for likely verify outcomes; cache hit returns immediately. | Same two-ahead goal as our (d): choose which (accept length, bonus token) outcomes to pre-draft. Abstract does not say how outcomes are chosen/predicted, so I cannot say whether it uses a history predictor. | Closest on (d). Our delta (if any) is a zero-weight history predictor for the outcome set; must read the full text before claiming. |
| SpecBranch (2506.01979) | "Inspired by branch prediction in processors": launches speculative branches at low-confidence draft tokens, branch count/draft length scaled by draft-model confidence, rollback-aware. | Shares the branch-prediction metaphor and branch-at-uncertain-token idea. Uses a separate draft model's confidence, not tagged history tables. | Metaphor overlap is real; mechanism differs. Cite. |
| SpecDec++ (2405.19715), AdaSpec (2503.05096), Confidence-Modulated SD (2508.15371), FailFast (2512.20573) | Adaptive draft length from a trained acceptance-prediction head or draft-token confidence; stop when predicted rejection prob exceeds threshold (SpecDec++ claims threshold policy optimal; +7-11% over fixed length). | Our (b) is the same policy class (stop at a threshold on chain accept probability). Difference: our accept estimate is an online calibration table keyed by TAGE provider/counter, zero weights, vs. a trained head. | High overlap on policy, low on signal source. |
| SuffixDecoding (2411.04975; Snowflake/CMU, ArcticInference) | Model-free: suffix trees over prior outputs + current request, speculation tree from most frequent continuations, adaptive; up to 2.9x on agentic SQL. | Closest to our (a). Suffix-match ~ one very long history length. TAGE adds multiple history lengths with per-entry confidence and learned provider selection and bounded fixed-size state. Overlap in goal is high. | Strong baseline; our sim implements an approximation only (longest-match + frequency variant, not the real tree/scoring). |
| Prompt Lookup Decoding, LLMA, Lookahead decoding, REST, SSSD (2411.05894), N-gram-trie (EMNLP 2025), Cacheback (2511.21699), ngram-mod hash-table drafter (MLX-LM blog) | Retrieval/n-gram/cache drafts: copy continuation of a matched n-gram; hash-table O(1) context->next-token; trie variants. | (a): single-order or trie-structured n-gram proposals. The ngram-mod hash table is a single-history-length tagless ancestor of our tables. | Overlap on (a). None found with multi-length tagged tables + confidence counters + allocation-on-mispredict, but I only read snippets. |
| Expert prefetch predictors: ProMoE (2410.22134), Fate (2502.12224, cross-layer gate), MoE-Infinity/Lina (path/frequency), MoE-Beyond (2508.17137, learned) | Predict next-layer/next-token expert sets for offloading/prefetch; learned MLPs, cross-layer gate reuse, activation-path stats. ProMoE reports a "token-based" baseline weaker than learned predictors (figures unconfirmed). | Our (e) would predict North's expert sets from token-history tables. Prior work uses router-state/learned predictors, which are stronger and need router state or weights. | Low novelty and likely weaker. Unevaluated by us (no North routing trace). |

Honest overall verdict: the metaphor ("speculation as branch prediction") is already used by SpecBranch; adaptive length by acceptance prediction is established; n-gram/suffix proposals are established. The only possibly-new combination is a single zero-weight tagged-geometric-history structure that jointly serves proposal, accept confidence, and outcome prediction for two-ahead, with commit-only updates and bounded state. That is an engineering unification, not a new principle.
