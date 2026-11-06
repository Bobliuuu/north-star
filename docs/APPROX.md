# Draft-mode approximations (src/approx)

Scope: PLAN Phase 2 (axes A expert budget, B attention budget) plus axis C (layer skip) and the unified `DraftConfig`.
Approximations exist ONLY in draft mode. The verifier path is never modified (see "Verifier isolation").
Sources: HF `cohere2_moe` modeling code (transformers 5.22.1, same classes as the pinned revision d11e61a8), docs/NORTH_ARCH.md.
Evidence caveat: EAT and LSA method text was read through a fetch summarizer of the arXiv HTML pages (not the PDFs); equations
below are as extracted. Re-check against the PDFs before any paper claim.

## Layout
| file | content |
|---|---|
| `config.py` | `DraftConfig(attn_budget, attn_selector, attn_layers, sink, window, expert_r, weight_convention, expert_policy, skip_layers, ...)`, `LSAParams`, `EATParams` |
| `experts.py` | top-r + weight conventions, `EATState`, `eat_select`, `slot_output_norms` |
| `attention.py` | `OracleTopK`, `LSAIndex` (flat/tree), `SubQStyle`, `sparse_attention`, `AttnCost`, analytic `decode_step_cost` |
| `layerskip.py` | early-exit / uniform / middle skip sets, CLaSp-style DP |
| `hooks.py` | `DraftPatch` (HF model patching), `EATCalibrator`, `to_ref_draft_mode` (RefNorth subset) |
| `sweep.py`, `modal_sweep.py` | factorial sweep + Modal entrypoint |
| `tests/` | 43 tests on tiny random cohere2_moe configs (CPU, ~10 s) |

## A. Expert budget
North router: top-8 of 128 on RAW logits, weights = sigmoid(selected logits), not renormalised. Draft `topr` keeps the first r of the
sorted native top-8 (a subset of the DRAFT's own router choice, not of the verifier's) with weight convention:
- `raw_sigmoid`: native sigmoid weights, dropped mass lost (North's own convention applied to fewer experts)
- `renorm`: kept weights / their sum (changes gate scale vs. North's unnormalised sum; labelled variant)
- `mass_preserve`: renorm, then rescaled to the native top-8 weight sum

Fewer activated experts is not assumed to give proportional latency savings (PLAN). With the HF experts module `r<k` truncates
columns (real fewer expert GEMMs); EAT's variable counts are implemented as zero-weight slots (functionally dropped, compute NOT saved here;
`experts_per_token` is reported as the proxy).

### EAT (arXiv 2609.33614): reproduced vs approximated
Reproduced as printed (hyper-parameters = Table 1 defaults):
- history per expert: `N_i` activation count, `Ptot_i` cumulative gate weight, `C_s` EMA of expert output L2 norm, `C <- (1-a)C + a||o||`, a=0.95 (sequential EMA computed exactly in closed form, unit-tested against the sequential loop)
- `S_h,i = w1 N_i/sum(N) + w2 Ptot_i/N_i + w3 C_s,i`, w=(0.3,0.3,0.4) (no per-term normalisation, exactly as written; C_s is in raw norm units so it can dominate)
- `I_i = (1-beta) p_i + beta S_h,i`, beta=0.6, p_i = router selection probability (sigmoid(logit) for North)
- `tau = max(a1 mu_g, mu_g - a2 sigma_g)` over the per-token raw gating scores, a1=0.5, a2=1.0
- candidate set `E={I_i>tau}`; |E|=0 -> argmax I; |E|>K -> top-K by I (so 1..K experts/token)
- renormalisation `g*m / (sum g*m + eps)` is the `renorm` convention
Approximated / NOT reproduced:
- `tau_PPL` is used by the paper but never defined, and the PPL-feedback rule (+20% / -10% scale on tau, kappa smoothing 0.7/0.3) is therefore NOT implemented; replaced by a constant multiplier `tau_scale` (swept in the full preset).
- "tau bounded to 10%/90% of the routing weight distribution" is ambiguous; implemented as per-token quantiles (0.1, 0.9) of g (`tau_bounds`, can be disabled).
- Candidate scope: the paper's E ranges over all experts; default here is `native_topk` (E restricted to the router's own top-8, i.e. strictly a top-r-style subset). `candidate_scope="all"` implements the paper reading (draft may pick experts the router did not rank top-8).
- History comes from a calibration pass over the VERIFIER (native top-8) on the `calib` split (`EATCalibrator`; costs an extra pass of expert math for output norms), then is frozen. Paper's "all inference samples" statistics are collected the same way in spirit; online update during drafting is not done (`online_update` is a reserved flag, not implemented).
- No OPD recovery training (router fine-tuning with ~9K samples). This is a training-free, draft-only use; the paper prunes the model itself. Do not report results as "EAT" without these caveats ("EAT-inspired history/threshold policy").
- K=8 as in the paper's Qwen setting; no Mixtral/Phi settings.

## B. Attention budget (13 global layers: 0,4,..,48; layer 0 has RoPE, 12 are NoPE; sliding layers untouched)
`attn_layers="global"` (13) or `"global_nope"` (12). Selection is per KV head, shared by its 8 query heads; each query always keeps `sink` first keys and a `window` of recent keys; `attn_budget` = extra long-range keys. Attention over kept keys is exact softmax (SDPA with mask), so only the key set is approximate. Sequence-mode (prefill / teacher-forced) and HF-cache decode are both supported; the sweep uses sequence mode.

| selector | what it is | honest label |
|---|---|---|
| `oracle_topk` | top-k of q.k over every causal key | REFERENCE. All-keys scan: O(S) index work per query, **not subquadratic** (cost reports `index_scored == all causal keys`). Upper bound for index quality. |
| `lsa_flat`, `lsa_tree` | LSA-INSPIRED hierarchical latent index | see below; NOT the LSA artifact |
| `subq_style` | block-mean routing then exact rescoring in candidate blocks | SSA-INSPIRED independent approximation from public disclosure only (content-based selection, exact attention on selected positions). NOT a reproduction of SubQ SSA; never call it SSA. |

LSA (arXiv 2610.42147 + repo README/notation, read via summarizer): reproduced structure = L2-normalised keys, pools of 4 summed, cluster blocks (= window) closing into buckets (~8 keys avg), unit-norm bucket centroids + member table, tree levels of node buckets (fanout 4) with beam search from the lowest level having <= `start_nodes` nodes, candidates = children of kept nodes plus "open" nodes, buckets expanded best-first until k keys, sink + window always read, one index list shared across query heads (mean of the group's query heads), buckets built only when a block closes and never change. Differences: (1) no learned/distilled index projection (D-LSA) and North is not N-LSA-trained, so this is the "cluster own keys" idea applied to an untrained model; expect poor recall vs a trained map; (2) Lloyd k-means (or `position` contiguous) instead of Ward merging and no bucket-size cap; (3) tree nodes group consecutive bucket ids; (4) default window/block 128 and beam 64 (paper: 512, 2048) for 2k-32k contexts; (5) dense-tensor emulation: cost is COUNTED from candidate masks (what a real implementation scores), not from wall time; (6) causality: a bucket/node is usable by query i only when all its keys are <= i (tested: changing future K/V never changes earlier outputs). Layer 0 is a RoPE layer; keys carry position information there (paper's trained maps used RoPE-free dims).

Cost accounting (`AttnCost`, per (key-vector, KV-head)): `keys_causal` (full-attention reads), `keys_attended` (gather), `index_scored` (selection dot products), `index_build` (pooling + k-means + node sums / block means, per call = full build, i.e. prefill-style), `index_bytes` (resident bf16 centroids + int32 member table). `decode_step_cost(S, selector, budget)` gives the analytic per-decode-step cost with amortised build, and shows oracle scan = S (linear), LSA flat = S/8, tree ~ O(beam x fanout x levels). Training cost: none (no adapter training in this phase).

## C. Layer skip
`layerskip.py`: `early_exit_set` (LayerSkip inference shape only; North has no early-exit training), `uniform_skip_set`, `middle_block_set`, `candidate_skip_sets`; `clasp_select` = CLaSp-style DP (cosine-to-full-hidden objective, O(L*M)) re-implemented from the abstract-level description only; the exact DP needs the layer applied to arbitrary states with KV, so the provided `delta_layer_apply_factory` assumes input-independent residual increments (APPROXIMATION). Layer 0 is never skipped. Skipping removes the whole parallel block (attention+MoE) of that layer.

## D. DraftConfig and verifier isolation
`DraftPatch(model, cfg, eat_states)`: sets instance-level `forward` closures on the skipped decoder layers, global attention modules, and sparse MoE blocks; each closure begins `if not patch.active: return original(...)`. `with patch.draft():` activates, `with patch.verifier():` forces off, `remove()`/exit deletes the instance attributes. Tests (`tests/test_hooks.py::test_verifier_path_untouched`, 6 configs) assert: parameter+buffer SHA-256 identical before/during/after drafting; logits `torch.equal` to unpatched when installed-but-inactive and inside `verifier()`; no `forward` in any module `__dict__` and class forwards unchanged after removal; draft logits actually differ. Also: budget >= S with r = native k reproduces verifier logits to 1e-4; HF top-r/skip draft equals `RefNorth` `DraftMode` logits; HF-cache decode equals sequence-mode drafting. `to_ref_draft_mode(cfg)` maps the skip/top-r(raw|renorm) subset for RefNorth (which has no sparse-attention or EAT hook).

## Sweep
Teacher-forced on real tokens: teacher = verifier's greedy continuation; verifier targets = unpatched full-sequence argmax; draft = same sequence under `DraftPatch.draft()` (draft KV from draft computation, as in self-speculation). Accepted length at block gamma from position s = agreement run from s capped at gamma (exact under greedy verification while drafts agree). Reported: top-1 agreement, verifier-top-1-in-draft-top-5, accepted-length histogram and mean committed tokens/cycle (accepted+1) for gamma in {2,4,8}, cost proxies. Groups: `full` (sanity, must be 1.0), `attn_only`, `expert_only`, `combined` (subset product of the single-axis settings so interaction = combined vs best single is directly comparable), `layerskip`. EAT history uses the `calib` split, evaluation uses `dev`, `heldout` untouched. The tail of each window under-counts long runs (truncated at n_new). It measures acceptance only; execution cost/speed is a separate track, so the Phase-2 exit gate (net performance) needs these acceptances joined with measured draft cost.

Command (needs `northstar-weights` volume populated by `modal run modal/download.py`; GPU A100-80GB):
```
modal run src/approx/modal_sweep.py --smoke true                 # sanity, 8 configs
modal run src/approx/modal_sweep.py                              # quick preset, ctx 2048+8192, 2 prompts
modal run src/approx/modal_sweep.py --preset full --groups full,attn_only,expert_only,combined,layerskip --n-prompts 4 --ctxs 2048,8192,32768
```
Outputs `results/approx_sweep_<tag>.json` (rows + summary) and the same on the `northstar-results` volume; spend appended to `results/SPEND.md`. Local CPU tests: `PYTHONPATH=src python -m pytest src/approx/tests`.

## Known risks for the GPU run (untested at scale)
- HF `grouped_mm` experts path with r<8 columns / zero-weight slots is untested on GPU (tiny tests use the eager path). If it errors, run with `experts_implementation=None`.
- Sequence-mode sparse attention materialises [Hkv,chunk,S] masks; ctx 32k uses ~0.5 GB per chunk, fine on 80 GB after the 61 GB weights, but sync `.item()` cost counters slow it (tens of seconds per config at 32k).
- Expect negative results for untrained LSA/SubQ-style indexes and for `renorm` (changes gate scale); those are publishable per PLAN.
