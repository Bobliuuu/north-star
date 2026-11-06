# RUNS

| time (UTC) | app id | command | notes |
|---|---|---|---|
| 2026-11-06 ~06:40 | ap-B0wZK4IVQWioxxurZCKpLO | `modal run --detach modal/selfimprove_eval.py --track kernel --proposer north --mode structured --rounds 3 --max-usd 5 --confirm-cost --store results/selfimprove_kernel_structured` | live North structured run 3 (relaunched detached; first attached launch ap-zLGryfgHK5wOoi9nrTzaWA killed). Round loop is the local entrypoint; evaluate_candidate calls are remote. Log: logs/selfimprove_structured_run3.log |
<<<<<<< HEAD
| 2026-11-06 ~06:55 | ap-3VjFsGh7DZtv1FZWqGvumI | `modal run --detach modal/selfimprove_eval.py::remote --track kernel --proposer mock --rounds 1 --store selfimprove_dryrun_remote` | first test of remote Volume-backed loop (mock proposer, ~$0.7). Log: logs/selfimprove_dryrun_remote.log |
| 2026-11-06 ~07:10 | ap-YQcDqwl0vuA9m4NTXOWz0M | `modal run --detach modal/demo.py --task py_lru --mode both --gamma 4 --skip-k 8 --n-new 64` | first real North demo trace (est $0.25-0.45). Log: logs/demo_run1.log |
=======
| 2026-11-06 ~06:45 | ap-QTZcDdQdbs3jmaGllyYwHa | `modal run --detach modal/ctx_batch.py::main` | ctx 8K/32K x batch 1/2/4 plain HF decode + top4 g4 self-spec vs plain at 8K/32K; result to Volume northstar-results:/ctx_batch_*.json |
| 2026-11-06 ~06:50 | ap-QTZcDdQdbs3jmaGllyYwHa | (result) | finished OK, 161 s GPU, ~$0.11. Log logs/ctx_batch_run1.log, results/ctx_batch_run1.jsonl |
>>>>>>> origin/ctx-batch
