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
