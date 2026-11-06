# Phase 3 joint optimizer (`src/opt/`)

Code: `src/opt/` (pure Python + OR-tools CP-SAT; CPU only). Pins: `src/opt/requirements.txt`.
Tests: `cd src && python3 -m unittest opt.tests.test_opt` (19 tests, ~3 s; also `pytest src/opt/tests`).
Study: `cd src && python3 -m opt.cli` (about 3.5 min) writes `src/opt/results/regime_scan_SYNTHETIC.json`.

## Model
A steady-state speculative cycle is a DAG: `commit -> index update -> gamma' draft steps -> checkpoint -> verify(gamma+1 tokens) -> accept`.
Each draft step and the verify pass is a chain of layer groups. Inside a group the attention, MoE (and layer-0 dense) branches are independent
(North's parallel block) and join at the residual add. Draft skips `skip` layers; verify never does.

Decisions. Draft side: `gamma`, `attn_budget` (keys kept on global layers; 0 = full), `r` (top-r experts), `skip` set.
Schedule side: expert schedule `mode` in {`two_pass` (no cross-cycle speculation, verify_only MoE), `spec_two_pass` (a speculative next-cycle
draft token rides in the verify pass as a separate pass), `fused` (same token rides in one fused pass)}, attention tile variant per task,
and start times/overlap (CP-SAT).
With probability h = P(A>=gamma) * carry_hit_given_full the carried draft token is reusable and the next cycle runs gamma-1 draft steps
(scenario A); otherwise gamma (scenario B). T = h*T_A + (1-h)*T_B, each the CP-SAT makespan.

Constraints: two cumulative resources (HBM-bandwidth %, SM %, capacity 100) so overlapped tasks are not additive; memory cumulative over buffer
lifetimes (per-step draft KV scratch, sparse-index scratch, checkpoint); optional worst-case-cycle latency cap; static memory pre-filter.
Objective: max N/T, N = 1 + E[min(A, gamma)] from the calibration table (exact for greedy). Solver: Dinkelbach (lam <- N/T of incumbent),
exact schedule solves lazy, pruned by a valid lower bound LB <= T (critical path / bandwidth-area / SM-area): a config is only solved if
N - lam*LB > 0. The result is exact over the enumerated grid (up to CP-SAT gap 0.1%). Policies are cached per workload bucket
(`policy.PolicyCache`, JSON, ctx quantized to 2K/8K/32K/128K); serving is a dict lookup.

Comparators (`solve.run_comparators`): fixed baseline (in-order stream, two_pass); draft-only (optimize draft params under baseline
schedule policy); schedule-only (baseline draft params, optimize mode + CP-SAT); independent (draft-only choice, then schedule-optimize it);
joint (everything). Plus `nospec_reference` (plain decode) for context.

## Measured vs synthetic (read this before quoting any number)
| Component | Status |
|---|---|
| MoE expert layer time vs (r, Tv, Td, routing): verify_only, fused, two_pass | MEASURED, A100-80GB, `results/kvm_fused_A100-80GB_fullsweep_*.jsonl`, one layer, seed 0 only, assumed identical for all 48 layers |
| Standalone draft MoE pass (two_pass - verify_only, median over Tv) | DERIVED from the above |
| MoE bandwidth demand (cumulative resource) | MEASURED (fused GB/s / peak) |
| Attention time, dense layer 0, LM head | ANALYTIC roofline (bytes from NORTH_ARCH.md / 0.70 x measured peak BW); not validated on hardware |
| Launch overheads, sparse-gather penalty, index-update cost, checkpoint/accept cost, SM-demand profiles, tile-B variant, sparse-index memory | SYNTHETIC constants (fields in `NorthCostModel`) |
| Draft-vs-verify expert overlap | assumption: routing mode uniform/corr75/identical stands in for it (`Bucket.routing`) |
| Acceptance pmfs and `carry_hit_given_full` | NO MEASURED TABLE EXISTS YET. `synth.py` generates SYNTHETIC ones (arbitrary shapes), stamped `provenance: SYNTHETIC`, refused unless `i_understand_this_is_synthetic=True`; policies carry `acceptance_synthetic` |
Every task duration carries `src_us` by provenance; schedules report work by source.

## Interface other agents must feed
1. Cost data: JSONL rows in `results/*.jsonl` with the existing kvm microbench schema (`gpu, routing, r, Tv, Td, peak_read_GBs, v.{verify_only,two_pass,fused}.{us_med,GBs}`). Extend `NorthCostModel` methods (`attn_group`, `lm_head`, `index_update`, `ckpt_save`, ...) to swap analytic/synthetic parts for measured ones. Tv, Td <= 16, r <= 8; Tv = batch*(gamma+1).
2. Acceptance calibration, one JSONL row per (workload, attn_budget, r, skip set), loaded by `AcceptTable.from_jsonl`:
```
{"workload":"code","attn_budget":0,"r":4,"skip_layers":[20,22],"accept_len_pmf":[p0,...,pG],
 "n_samples":N,"carry_hit_given_full":0.5,"provenance":"measured","source":"run id"}
```
`accept_len_pmf[j]` = P(A=j), last bin = P(A>=G), where A = longest draft prefix accepted by the greedy verifier with lookahead G (>= max gamma searched; the solver truncates exactly). `carry_hit_given_full` (optional, default 0 so fused is never favoured without data) = P(speculative draft token carried in the verify pass is reusable | all gamma accepted). Layer 0 cannot be skipped. Configs without a row are never considered (no invented acceptance).
3. Output: `PolicyCache.solve(bucket, make_evaluator, space)` returns a JSON policy: config, predicted tok/s, per-scenario schedule (start times, tile variants, work by provenance), search stats, cost-model and calibration fingerprints.
Validate on hardware: compare `pred_cycle_us` and per-scenario start times with measured; feed back by editing cost-model parameters.

## Results (all acceptance SYNTHETIC; regime scan = 4 synthetic families x 3 routing x 4 ctx = 48 cells)
Full table: `src/opt/results/regime_scan_SYNTHETIC.json`.
- Plain decode (no speculation) on the measured A100 MoE costs: ~145-147 tok/s. The fixed baseline (gamma 4, r 4, two_pass) is 32-67 tok/s, i.e. slower than not speculating: a standalone draft MoE pass costs ~91-104 us per layer (launch/tail dominated) vs 121 us for a 1-token verify layer, and a 5-token verify layer is 2.7x a 1-token one (expert union). Classic multi-token self-speculation does not pay on this kernel regardless of acceptance; only the fused marginal draft cost (a few us per layer) is cheap.
- Draft-only opt reaches 80-105 tok/s (still < no-spec). Schedule-only 50-88. Independent 131-169. Joint 134-169; joint is >= independent in every cell (it is a superset; asserted in tests).
- Joint beats independent by >1% in 36/48 cells (range 0 to 11.5%, median ~5-6%); equal in 12. Mechanism: draft-only optimization under serial two_pass timing prefers cheap draft configs (layer skipping, attention sparsity); once the fused schedule makes the speculative draft token nearly free, those choices only lose acceptance, so the joint optimum is a different draft config (e.g. r6/no-skip vs r8/12-layer-skip). Winning configs are all gamma=1 + fused, only +0-15% over no-spec (160 vs 146 tok/s typical), and that gain depends entirely on the SYNTHETIC carry_hit=0.6 and acceptance 0.9. Do not cite as a speedup. The qualitative result "ranking of draft configs changes with the schedule" is a property of the measured fused/two_pass cost gap; its magnitude is not established until real calibration exists.
- Solve cost: joint enumeration 810 configs; Dinkelbach CP-SAT-solved 1-11 configs (rest pruned by LB); joint solve mean 2.9 s, max 9.4 s per bucket (4 workers, 3 s per-model limit). Per-cell DAG ~60-80 tasks, 6 layer groups.
- Cost-model sensitivity (32K ctx, attn-tolerant synthetic, 12 draws, lognormal per-kind errors: moe sigma=max(5%, measured repeat noise 3.5%), analytic kinds 25%, synthetic 50%; these sigmas are priors): mean regret of the nominal joint policy 0.8%, p90 0%, max 10%; argmax moved in 8% of draws; mean |predicted-actual tok/s| 6.3%. At sigma x2: mean regret 1.2%, max 14%, prediction error 19%. At x0.5: zero regret. Regret is vs a 25-config shortlist re-solved under the perturbed model, so it is a lower bound.

## Limits
Layer groups (6) are the scheduling granularity; contention is modeled by cumulative caps, not a measured slowdown; no overlap-contention profile exists (Phase 3 requires one from hardware). Same MoE cost for all 48 layers. Greedy only. Batch >1 limited by Tv<=16. Latency cap uses worst-case scenario. Draft attention uses analytic costs, so attn_budget conclusions are weak until attention is microbenchmarked.
