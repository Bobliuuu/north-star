# RUNS

| time (UTC) | app id | command | notes |
|---|---|---|---|
| 2026-11-06 ~06:40 | ap-B0wZK4IVQWioxxurZCKpLO | `modal run --detach modal/selfimprove_eval.py --track kernel --proposer north --mode structured --rounds 3 --max-usd 5 --confirm-cost --store results/selfimprove_kernel_structured` | live North structured run 3 (relaunched detached; first attached launch ap-zLGryfgHK5wOoi9nrTzaWA killed). Round loop is the local entrypoint; evaluate_candidate calls are remote. Log: logs/selfimprove_structured_run3.log |
| 2026-11-06 | ap-6fmiTOjuNdsl9qTBQbYCKC | `modal run --detach modal/kernel_sweep.py --tag sweep1` | OFAT sweep of 6 structured tunables, A100-80GB, 3 interleaved passes; results/kernel_sweep_sweep1_*.jsonl; client log logs/kernel_sweep1_client.log |
