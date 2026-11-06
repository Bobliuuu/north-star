# Spend log (Modal; estimated = wall seconds x list price, excludes image build)

| utc | task | gpu | secs | est USD | ok |
|---|---|---|---|---|---|
| 20261106T021327Z | kvm-fused first full=False seeds=1 | A100-80GB | 26 | 0.018 | True |
| 20261106T021827Z | kvm-fused fullsweep full=True seeds=1 | A100-80GB | 263 | 0.183 | True |
| 2026-11-05 21:38 | probe | A100-80GB | 66 | 0.05 |  |
| 2026-11-05 21:41 | routing | A100-80GB | 196 | 0.14 |  |
| 2026-11-05 21:46 | baseline | A100-80GB | 245 | 0.17 |  |
| 2026-11-05 21:54 | ceiling | A100-80GB | 987 | 0.68 |  |
| 2026-11-05 22:08 | spec | A100-80GB | 868 | 0.60 |  |
| 2026-11-05 21:55 | spec (aborted attempt, slow volume load, killed) | A100-80GB | 740 | 0.51 | estimated from log timestamps (12 min); no results |

**Cumulative est. total: $2.35 of $30 cap** (CPU download/test containers not included, negligible).
| 20261106T033917Z | tiny graph test (debug) | A100-40GB | 8 | 0.00 | True |
| 20261106T034013Z | tiny graph test (debug) | A100-40GB | 9 | 0.01 | True |
| 20261106T034201Z | tiny graph test (debug) | A100-40GB | 9 | 0.01 | True |
| 20261106T034226Z | tiny graph test (debug, tiny model) | H100 | 9 | 0.01 | True |
| 20261106T034431Z | megakernel all | H100 | 101 | 0.11 | |
| 20261106T041953Z | megakernel all | H100 | 95 | 0.10 | |
| 20261106T042441Z | h100_blocks_overlap | H100 | 227 | 0.25 | True |
| 20261106T042620Z | tiny graph test (debug, tiny model) | H100 | 7 | 0.01 | True |
| 20261106T042838Z | h100_blocks_overlap | H100 | 119 | 0.13 | True |
| 20261106T043142Z | megakernel bs | H100 | 143 | 0.16 | |
| 20261106T043227Z | kvm-fused REAL-routing replay real | H100 | 4 | 0.004 | False |
| 20261106T043309Z | kvm-fused REAL-routing replay dbg | H100 | 4 | 0.004 | False |
| 20261106T044108Z | kvm-fused REAL-routing replay real | H100 | 436 | 0.478 | True |
| 20261106T044337Z | h100_spec | H100 | 88 | 0.10 | True |
| 20261106T064800Z | ctx_batch run1 (ap-QTZcDdQdbs3jmaGllyYwHa) | A100-80GB | 161 | 0.11 | True |
| 2026-11-06 01:56 | demo py_lru both | A100-80GB | 578 | 0.40 | wall incl. queue; trace in demo/traces |
| 20261106T065947Z | kernel tunable sweep sweep1 (build 22s check 156s) | A100-80GB | 285 | 0.198 | True |
| 20261106T074908Z | kernel tunable sweep combo1 (build 13s check 144s) | A100-80GB | 215 | 0.149 | True |
| 20261106T072300Z | ctx_batch run2 (ap-ib0aezTWzOUPOaSI9I3RkL) | A100-80GB | 482 | 0.33 | True (mixed cell OOM, rerun as run3) |
| 20261106T082100Z | ctx_batch run3 mixed (ap-ACwkUWV6NOmw6pSni19K3V, ap-X66dz8pjfyQZaZtcsnE9pU) | A100-80GB | 0 | 0.00 | never scheduled, stopped |
| 2026-11-06 09:00 | approx_sweep_smoke1 | A100-80GB | 97 | 0.07 |  |
| 2026-11-06 09:08 | approx_sweep_quick1 | A100-80GB | 148 | 0.10 |  |
| 2026-11-06 10:08 | draft_sparse_ds4 | H100 | 82 | nan |  |
| 2026-11-06 10:44 | weightedit_smoke1b | A100-80GB | 153 | 0.11 |  |
| 2026-11-06 11:16 | gen_dump_d1 | A100-80GB | 161 | 0.11 |  |
