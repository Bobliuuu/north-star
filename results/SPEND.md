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
| 2026-11-09 | tilegrid kernel tests x3 (Modal H100, short) (workspace <workspace>) | H100 | ~240 (est) | ~0.26 (est) (est 3.95/h) | OK; H19 |
| 2026-11-09 | abtile ab1 (workspace <workspace>) | H100 | 210 | 0.23 (est 3.95/h) | OK; H19 |
| 2026-11-09 | batchspec bs2 (workspace <workspace>) | H100 | 638 | 0.70 (est 3.95/h) | OK; H19 |
| 2026-11-09 | batchspec bs2g (workspace <workspace>) | H100 | 630 | 0.69 (est 3.95/h) | OK; H19 |
| 2026-11-09 | batchdiag d2 (workspace <workspace>) | H100 | 77 | 0.08 (est 3.95/h) | OK; H19 |
| 2026-11-09 | batchspec bs3 (workspace <workspace>) | H100 | 728 | 0.80 (est 3.95/h) | OK; H19 |
| 2026-11-09 | weights download (workspace <workspace>) | CPU | ~600 | ~0.05 (est) | OK |
| 2026-11-09 | profbatch p1 + p2 (workspace <workspace>) | H100 | 158 | 0.17 (est 3.95/h) | OK; H20 |
| 2026-11-09 | abattn ba1 (workspace <workspace>) | H100 | 124 | 0.14 (est 3.95/h) | OK; H20 |
| 2026-11-09 | batchspec bs4 (workspace <workspace>) | H100 | 635 | 0.70 (est 3.95/h) | OK; H20 |
| 2026-11-09 | batchspec bs5 (workspace <workspace>) | H100 | 576 | 0.63 (est 3.95/h) | OK; H20 |
| 2026-11-09 | abstack st1 (workspace <workspace>) | H100 | 114 | 0.13 (est 3.95/h) | OK; H20 |
| 2026-11-09 | batchspec bs6 (workspace <workspace>) | H100 | 569 | 0.62 (est 3.95/h) | OK; H20 |
| 2026-11-09 | moe-tune agent (kernel sweeps + mt1) (workspace <workspace>) | H100 | see results/moe_tune_runs.md | ~1.0 (est 3.95/h) | OK; H20 |
| 20261111T062610Z | selfimprove bench sib1 (build 78s, 85 configs) | A100-80GB | 748 | 0.519 | True |
| 20261111T072827Z | selfimprove bench sib2 (build 65s, 96 configs) | A100-80GB | 702 | 0.488 | True |
| 20261112T124425Z | north-in-north nin0smoke (10 North calls) | A100-80GB | 197 | 0.137 | True |
| 20261112T131251Z | north-in-north nin1 (266 North calls) | A100-80GB | 1645 | 1.142 | True |
| 20261112T133950Z | embedded North-in-North smoke (1 runs) | A100-80GB | 124 | 0.086 | True |
| 20261112T144436Z | embedded North-in-North main (7 runs) | A100-80GB | 6254 | 4.343 | True |
| 2026-11-12 | x2_tune t1lin (no weights) (ap-XvSbyE9paS6he5AC4p7Xmj) | H100 | 126 | 0.14 (est 3.95/h) | True |
| 2026-11-12 | x2_tune t1moe (no weights) (ap-0Rb30RzTudp4cIhea5Pn19) | H100 | 141 | 0.15 (est 3.95/h) | True |
| 2026-11-12 | x2 x0smoke (ap-wP6EniN89K2KeBLgof7Lx8) | H100 | 150 | 0.16 (est 3.95/h) | True |
| 2026-11-12 | x2 x1prof (ap-SZB5IiYisldtj6GeFSAb73) | H100 | 60 | 0.07 (est 3.95/h) | True |
| 2026-11-12 | x2 x2step (ap-KDdbbERhmOVsN0RQnYkBpJ) | H100 | 140 | 0.15 (est 3.95/h) | True |
| 2026-11-12 | x2 x3 (99-prompt traffic) (ap-yxquFGZyx4Ta6R3C4ECVG9) | H100 | 543 | 0.60 (est 3.95/h) | True |
| 2026-11-12 | x2 x4prio (ap-BerPIPJYQNJW2TGERFgEZS) | H100 | 120 | 0.13 (est 3.95/h) | True |
| 2026-11-12 | bx2 b1 (ap-yiD2gDXqlkHugYQPoVLFj0) | H100 | 142 | 0.16 (est 3.95/h) | True |
| 2026-11-12 | batchspec bs7 (ap-8za1Ws6oTrJunObvu2WioP) | H100 | 484 | 0.53 (est 3.95/h) | True |
| 2026-11-12 | vllm v2 plain (ap-4RQtAPqaNHyPdlktZ1h5uz) | H100 | 389 | 0.43 (est 3.95/h) | True |
| 2026-11-12 | longexact le3 (ap-jtXJCq5jnIMNNPUaCrYMbx) | H100 | 299 | 0.33 (est 3.95/h) | True |
| 2026-11-12 | longexact le4 (ap-JwQrPtv8znQFKaXISA1q5C) | H100 | 515 | 0.57 (est 3.95/h) | True |
| 2026-11-12 | lcx2 lc1 (ap-z5I0h6C9akXwKBnTr1ZmPR) | H100 | 116 | 0.13 (est 3.95/h) | True |
| 2026-11-12 | longexact le5 (ap-aXGFj0swMEsnye2jcsLOTl) | H100 | 524 | 0.57 (est 3.95/h) | True |
| 2026-11-12 | longexact le6 (ap-qBbqM15nkgrMCrljxR29Lb) | H100 | 241 | 0.26 (est 3.95/h) | True |
| 2026-11-12 | vllm_vs_x2 vx1 (same container) (ap-hP2aZ9N2fXoFSslF5zBtVO) | H100 | 1095 | 1.20 (est 3.95/h) | True |
| 2026-11-12 | x2 x5prof (ap-wLrD5tl1tavxRyVvyEmEZv) | H100 | 69 | 0.08 (est 3.95/h) | True |
| 2026-11-12 | x2 x6 (ap-hi4VCUdLDsmdZhNCa7mMb5) | H100 | 614 | 0.67 (est 3.95/h) | True |
| 2026-11-12 | longexact le7 (ap-xAWIRHQ5U8w9Asql5zMGaF) | H100 | 254 | 0.28 (est 3.95/h) | True |
| 2026-11-12 | longexact le8 (ap-gyNvZZEkL0W9SJVyVGL5Le) | H100 | 557 | 0.61 (est 3.95/h) | True |
| 2026-11-12 | x2_tune t2moef (no weights, incl. failed first launch) (ap-2OgFw0fBGywr9MKSWDzipp) | H100 | 141 | 0.15 (est 3.95/h) | True |
| 2026-11-12 | x2 x7rf (ap-mwmmoeSfYxnNeTgvTmXhZ0) | H100 | 102 | 0.11 (est 3.95/h) | True |
| 2026-11-12 | x2 x8rf (ap-eMOiDZ26CHB6mHxImo5vas) | H100 | 112 | 0.12 (est 3.95/h) | True |
| 2026-11-12 | longexact le9 (ap-ekDA7RRIp3FCHgbnWqQfJu) | H100 | 255 | 0.28 (est 3.95/h) | True |
| 2026-11-12 | longexact le10 (ap-ciT7SYKCRqpZHrwl4r6ERd) | H100 | 579 | 0.64 (est 3.95/h) | True |
| 2026-11-12 | vllm_vs_x2 vxb1 (batched) (ap-4eMweePyOvy0Ug8RY12uqW) | H100 | 277 | 0.30 (est 3.95/h) | True |
| 2026-11-12 | serve sv0smoke (ap-Gny2DFpzBiQDd4OZEi2lwd) | H100 | 210 | 0.23 (est 3.95/h) | True |
| 2026-11-12 | serve sv1 (ap-0WWjsqmP8t9ej2DFgxM2Xn) | H100 | 519 | 0.57 (est 3.95/h) | True |
| 2026-11-12 | serve sv2b2 (ap-xzhBwYqhEU8QUQ7zpT0l3F) | H100 | 106 | 0.12 (est 3.95/h) | True |
| 2026-11-12 | prefill pf1 (ap-94EYUQ1HgO77T5R5QSMXOd) | H100 | 45 | 0.05 (est 3.95/h) | True |
| 2026-11-12 | vxcb cb0smoke (ap-uKtWF3aovIqBomjs5HTYnO) | H100 | 253 | 0.28 (est 3.95/h) | True |
| 2026-11-12 | vxcb cb0smoke2 (ap-UFTAgWtGDxteI3eJK2yCnA) | H100 | 332 | 0.36 (est 3.95/h) | True |
| 2026-11-12 | vxcb cb1 (ap-1BitGtaYxcZKibPplTwEvL) | H100 | 532 | 0.58 (est 3.95/h) | True |
| 2026-11-12 | vxcb cb2 (ap-U11oMwJWeT9dbqVgpfhm3s) | H100 | 319 | 0.35 (est 3.95/h) | True |
| 2026-11-12 | vxcb cb3 (ap-ul2ctFPWt9LizQdynmJ7Ng) | H100 | 557 | 0.61 (est 3.95/h) | True |
| 2026-11-12 | prefill inv1 (probe) (ap-BkRkt4juTIT9t0yiJ2MDc8) | H100 | 61 | 0.07 (est 3.95/h) | True |
| 2026-11-12 | vxcb cb4 (ap-F1426mDCLEUcZzSm3AVSzd) | H100 | 583 | 0.64 (est 3.95/h) | True |
| 2026-11-12 | headline hl1 (ap-6nmyBBq9bFnTn1r6ir2cN0) | H100 | 72 | 0.08 (est 3.95/h) | True |
