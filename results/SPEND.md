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
| 2026-11-06 11:36 | w1_gen_smoke | A100-80GB(or H100) | 114 | 0.08 |  |
| 2026-11-06 11:38 | selfedit_se_smoke2 | A100-80GB | 215 | 0.15 |  |
| 2026-11-06 10:39 | fast_exec_fx1 | H100 | 94 | nan |  |
| 2026-11-06 10:41 | fast_exec_fx2dbg | H100 | 38 | nan |  |
| 2026-11-06 10:45 | fast_exec_fx3c | H100 | 103 | nan |  |
| 2026-11-06 10:48 | fast_exec_fx4 | H100 | 113 | nan |  |
| 20261106T153920Z | mk_selfspec phase A | H100 | 234 | 0.26 | |
| 20261106T154507Z | mk_selfspec phase B | H100 | 312 | 0.34 | |
| 2026-11-06 | longctx lcA1 (ap-kiiBVXnLWGsnpa47tykm7c) | H100 | 442 | ~0.49 (est $0.0011/s) | 128K fits H100, peak 76.9 GB |
| 2026-11-06 | longctx lcB1 (ap-XROqUxe4vkuYkvgTak45Fg) | H100 | 456 | ~0.50 (est) | batch step costs |
| 2026-11-06 | longctx lcK1 (ap-nFhCdfbMfvUgE2GyBCx3pd) | H100 | ~900 | ~1.0 (est; stopped early) | kernel microbench |
| 2026-11-06 | longctx lcT1 (ap-DxAmLXiOiaFBPK3rOEPSY0) | H100 | ~420 | ~0.46 (est) | tuned kernel rerun |
| 2026-11-06 11:51 | weightedit_r2b | H100-80GB | 1658 | nan | H100 list ~$0.0011/s assumed; est $1.8 |
| 20261106T153422Z | harness diag diag1 | A100 | 203 | 0.141 | True |
| 20261106T160035Z | harness diag diag3 | A100 | 221 | 0.154 | True |
| 2026-11-08 | mk_block s2 vast1 (Vast.ai H100 SXM, 1 instance ~1h incl. build+57GB weights) | H100 | 55 (GPU run) | ~1.47 (Vast balance 454.92 -> 453.45) | instance destroyed, show instances empty |
| 2026-11-08 | mk_block s3 vast1 (Vast.ai H100 SXM, 1 instance, ~0.4 h incl. setup; 57GB weights parallel with build) | H100 | 61 (GPU run) | ~1.0 (est, 2.27/h rate; shared balance not isolable) | instance destroyed, show instances shows none of mine |
| 2026-11-08 | mk_block s2 vast2 + S4 partial (Vast.ai H100 NVL, 3.375/h, 13:58-14:20Z) | H100 NVL | 69 (S2 GPU) | ~1.25 (est from uptime ~22 min, destroyed) | S4 stopped by user order; show instances shows none of mine |
| 2026-11-08 | longexact le1 + le2 (Modal H100, 307 s + 517 s) | H100 | 824 | ~0.9 (Modal list $3.95/h x 824 s = 0.90) | both containers finished and exited; no Vast instance rented; no other resources left |
| 2026-11-08 | vllm_v1 (ap-xaefnZoiyA75jtFnHwR4Ve; 3 vLLM engine loads + plain) | H100 | 1307 | 1.43 (est 3.95/h) | OK; H17; earlier 2 failed launches 0 GPU-s; no Vast instance rented |
| 2026-11-08 | batchspec bs1 (ap-QOonWfL3UUuzGLPlYjPfBP) | H100 | 562 | 0.62 (est 3.95/h) | OK; H18 |
| 2026-11-08 | batchdiag d1 (ap-Jj4SEUzRYCVCrArm62vjoS) | H100 | unknown | unknown | outcome not recorded in repo |
