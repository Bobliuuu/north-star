# RUNS

| time (UTC) | app id | command | notes |
|---|---|---|---|
| 2026-11-06 ~06:40 | ap-B0wZK4IVQWioxxurZCKpLO | `modal run --detach modal/selfimprove_eval.py --track kernel --proposer north --mode structured --rounds 3 --max-usd 5 --confirm-cost --store results/selfimprove_kernel_structured` | live North structured run 3 (relaunched detached; first attached launch ap-zLGryfgHK5wOoi9nrTzaWA killed). Round loop is the local entrypoint; evaluate_candidate calls are remote. Log: logs/selfimprove_structured_run3.log |
| 2026-11-06 ~06:55 | ap-3VjFsGh7DZtv1FZWqGvumI | `modal run --detach modal/selfimprove_eval.py::remote --track kernel --proposer mock --rounds 1 --store selfimprove_dryrun_remote` | first test of remote Volume-backed loop (mock proposer, ~$0.7). Log: logs/selfimprove_dryrun_remote.log |
| 2026-11-06 ~07:10 | ap-YQcDqwl0vuA9m4NTXOWz0M | `modal run --detach modal/demo.py --task py_lru --mode both --gamma 4 --skip-k 8 --n-new 64` | first real North demo trace (est $0.25-0.45). Log: logs/demo_run1.log |
| 2026-11-06 ~06:45 | ap-QTZcDdQdbs3jmaGllyYwHa | `modal run --detach modal/ctx_batch.py::main` | ctx 8K/32K x batch 1/2/4 plain HF decode + top4 g4 self-spec vs plain at 8K/32K; result to Volume northstar-results:/ctx_batch_*.json |
| 2026-11-06 ~06:50 | ap-QTZcDdQdbs3jmaGllyYwHa | (result) | finished OK, 161 s GPU, ~$0.11. Log logs/ctx_batch_run1.log, results/ctx_batch_run1.jsonl |


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
