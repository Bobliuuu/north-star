# RUNS

| time (UTC) | app id | command | notes |
|---|---|---|---|
| 2026-11-06 ~06:40 | ap-B0wZK4IVQWioxxurZCKpLO | `modal run --detach modal/selfimprove_eval.py --track kernel --proposer north --mode structured --rounds 3 --max-usd 5 --confirm-cost --store results/selfimprove_kernel_structured` | live North structured run 3 (relaunched detached; first attached launch ap-zLGryfgHK5wOoi9nrTzaWA killed). Round loop is the local entrypoint; evaluate_candidate calls are remote. Log: logs/selfimprove_structured_run3.log |
| 2026-11-06 ~06:55 | ap-3VjFsGh7DZtv1FZWqGvumI | `modal run --detach modal/selfimprove_eval.py::remote --track kernel --proposer mock --rounds 1 --store selfimprove_dryrun_remote` | first test of remote Volume-backed loop (mock proposer, ~$0.7). Log: logs/selfimprove_dryrun_remote.log |
| 2026-11-06 ~07:10 | ap-YQcDqwl0vuA9m4NTXOWz0M | `modal run --detach modal/demo.py --task py_lru --mode both --gamma 4 --skip-k 8 --n-new 64` | first real North demo trace (est $0.25-0.45). Log: logs/demo_run1.log |
| 2026-11-06 ~06:45 | ap-QTZcDdQdbs3jmaGllyYwHa | `modal run --detach modal/ctx_batch.py::main` | ctx 8K/32K x batch 1/2/4 plain HF decode + top4 g4 self-spec vs plain at 8K/32K; result to Volume northstar-results:/ctx_batch_*.json |
| 2026-11-06 ~06:50 | ap-QTZcDdQdbs3jmaGllyYwHa | (result) | finished OK, 161 s GPU, ~$0.11. Log logs/ctx_batch_run1.log, results/ctx_batch_run1.jsonl |


| 2026-11-06 | ap-duOM69y4JrsPm52eVxoci9 | `modal run --detach src/approx/modal_sweep.py::main --smoke --tag smoke1` | first ever run of draft-approximation sweep on North (1 prompt, ctx 2048, 8 configs). Log logs/approx_sweep_smoke1_client.log |

| 2026-11-06 | ap-pqB0aAizeEKtPTfQ2sG2JC | `modal run --detach src/approx/modal_sweep.py::main --tag quick1` | approx sweep quick preset (full/attn/expert/combined, 2 prompts, ctx 2K/8K). smoke1 (ap-duOM69y4JrsPm52eVxoci9) finished OK. Log logs/approx_sweep_quick1_client.log |

| 2026-11-06 | $AP | `modal run --detach modal/draft_sparse.py::main --tag ds4` | H100 graphed draft-step cost with sparse-attn gather. Earlier tries: ds1 A100 failed (grouped_mm not graph-capturable on A100); ds3 oracle shape bug. Log logs/draft_sparse_ds4_client.log |

## Self-improve chain heads (external anchor for the unkeyed hash chain)

After EVERY self-improvement round run `PYTHONPATH=src python -m selfimprove.cli head --store results/<store> --label <store>` and commit the
printed line below (git history of this file is the anchor). Verify before trusting a store:
`PYTHONPATH=src python -m selfimprove.cli verify-log --store results/<store> --expected-head <hash>` (add `--allow-newer` if rounds were appended after the anchor).

| line |
|---|
| `selfimprove-head selfimprove_dryrun_kernel n=2 head=993ef5a14737ac29cf28b0f30fe70bf1ccc6d39f4ccd5cf0fd7c1e1aca5534e9` |
| `selfimprove-head selfimprove_dryrun_remote n=2 head=c4dd70196c72eb9a1ebc492b5928335cba454b0b29cf039e852b2e8341dd85c2` |
| `selfimprove-head selfimprove_kernel n=5 head=0d93828d4de11ae47fcda315026318b27370a86ba7ff671333c8cc20024c757b` |
| `selfimprove-head selfimprove_kernel_attempt1_nodiff n=2 head=b5316983c96edc91bf672a03e202526393164c1afc951b3989354de8f0cea7bd` |
| `selfimprove-head selfimprove_kernel_attempt2_nodiff_think8192 n=2 head=49ea40956add70c6dd3ac6fb4175a86e39abd0bd0ea8cd21ea5450c3313e0812` |
| `selfimprove-head selfimprove_kernel_attempt3_malformed_diff n=2 head=4e890627e3016ec24264d70a2ce678bdc405fc8445ff3d43f9675e10fe71e457` |
| `selfimprove-head selfimprove_kernel_structured n=9 head=f41d1dd4e33a4ff93acc9a9a679fc55e4275319ff45b6724c55cb168784bcc79` |
| `selfimprove-head selfimprove_kernel_structured_run1_audit_reject n=4 head=2ac750d88a869f40b7079464276c7ad2e8f229de7563122ea708146b666616a3` |
| 2026-11-06 | ap-6fmiTOjuNdsl9qTBQbYCKC | `modal run --detach modal/kernel_sweep.py --tag sweep1` | OFAT sweep of 6 structured tunables, A100-80GB, 3 interleaved passes; results/kernel_sweep_sweep1_*.jsonl; client log logs/kernel_sweep1_client.log |
| 2026-11-06 | ap-6fmiTOjuNdsl9qTBQbYCKC | (stopped) | sweep1 first attempt: A100-80GB never scheduled in ~55 min (0 containers, $0 spent); stopped |
| 2026-11-06 | ap-kYajfWtgyO3OPmzJ1fSN0O | `modal run --detach modal/kernel_sweep.py --tag sweep1` | relaunch with gpu fallback [A100-80GB, A100-40GB]; client log logs/kernel_sweep1b_client.log |
| 2026-11-06 | ap-N1q7OiAH5UWFH8vxV9xgO8 | `modal run --detach modal/kernel_sweep.py --tag combo1 --preset combo` | combo of winners (KC64/UNROLL4/MINB3), all configs correctness-checked; log logs/kernel_sweep2_client.log |
| 2026-11-06 ~08:05 | ap-c2wCUwGvoEKY9z4vuL9fPA | `modal run --detach modal/selfimprove_eval.py::main --track kernel --proposer north --mode structured --rounds 3 --max-usd 5 --confirm-cost --store results/selfimprove_kernel_structured` | live North structured run 4: history-aware prompt (own edits+speedups) + UNROLL_KS tunable; no human sweep data given to North. run 3 result: 2x gate_fail:speed, 1x gate_fail:correctness. Log: logs/selfimprove_structured_run4.log |
| 2026-11-06 ~07:20 | ap-ib0aezTWzOUPOaSI9I3RkL | `modal run --detach modal/ctx_batch.py::main2` | ctx_batch run2: 8K self-spec x3 prompts, batch 2/4 lock-step self-spec, mixed 2K+8K+32K left-padded plain decode, 128K plain + self-spec; incremental result Volume northstar-results:/ctx_batch2_*.json. cap $6. Log logs/ctx_batch_run2.log |
| 2026-11-06 02:23 EST | ap-ib0aezTWzOUPOaSI9I3RkL | (result) | finished, 482 s GPU (~$0.33 at run1 rate $0.00068/s). mixed cell OOM at prefill chunk 2048 (masked SDPA); rest OK. results/ctx_batch_run2.jsonl, logs/ctx_batch_run2.log |
| 2026-11-06 02:27 EST | ap-ACwkUWV6NOmw6pSni19K3V | `modal run --detach modal/ctx_batch.py::main2 --parts pmixed` | run3: mixed 2K+8K+32K left-padded plain decode with prefill chunk 512 (run2 OOM at 2048). Log logs/ctx_batch_run3.log |
| 2026-11-06 03:03 EST | ap-X66dz8pjfyQZaZtcsnE9pU | `modal run --detach modal/ctx_batch.py::main2 --parts pmixed` | run3 relaunch (first attempt ap-ACwkUWV6NOmw6pSni19K3V never scheduled; stopped, 0 GPU s). mixed 2K+8K+32K left-padded, prefill chunk 512 |
| 2026-11-06 03:21 EST | ap-ACwkUWV6NOmw6pSni19K3V, ap-X66dz8pjfyQZaZtcsnE9pU | (result) | run3 mixed: both launches never got a container (0 tasks, no logs for 40 min / 20 min), stopped by me; 0 GPU s. Mixed-length cell remains NOT RUN beyond the run2 OOM. |
| 2026-11-06 | ap-ri-diag (see logs/harness_diag{1,2,3}_client.log) | `modal run --detach modal/harness_diag.py::main --tag diag1/diag3`, `::main2 --tag diag2` | selfimprove-v5 harness diagnostics (A100): A/A bench+timer stats and correctness seed sweep for baseline/North edit set/mt8; ~$0.14+$0.15+small. results/harness_diag_*.json |
| 2026-11-06 | ap-FdLqlGKknNuRnVjg4II6PS | `modal run --detach modal/selfimprove_eval.py::main --track kernel --proposer north --mode structured --rounds 3 --per-round 4 --max-usd 5 --confirm-cost --store results/selfimprove_kernel_structured_v5` | live North structured run 5 (NEW store, branch selfimprove-v5): fixed harness (calibrated timer gate, kernel_check r=8/renorm bug), +UNROLL_S2/UNROLL_NT, 3-edit cap, best-of-4/round with duplicate skip, own-measured-results feedback. Proposer redeployed with scaledown 600 s. Log: logs/selfimprove_structured_run5.log |
