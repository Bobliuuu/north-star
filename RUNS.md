# RUNS

| time (UTC) | app id | command | notes |
|---|---|---|---|
| 2026-11-06 ~06:40 | ap-B0wZK4IVQWioxxurZCKpLO | `modal run --detach modal/selfimprove_eval.py --track kernel --proposer north --mode structured --rounds 3 --max-usd 5 --confirm-cost --store results/selfimprove_kernel_structured` | live North structured run 3 (relaunched detached; first attached launch ap-zLGryfgHK5wOoi9nrTzaWA killed). Round loop is the local entrypoint; evaluate_candidate calls are remote. Log: logs/selfimprove_structured_run3.log |
| 2026-11-06 | ap-6fmiTOjuNdsl9qTBQbYCKC | `modal run --detach modal/kernel_sweep.py --tag sweep1` | OFAT sweep of 6 structured tunables, A100-80GB, 3 interleaved passes; results/kernel_sweep_sweep1_*.jsonl; client log logs/kernel_sweep1_client.log |
| 2026-11-06 | ap-6fmiTOjuNdsl9qTBQbYCKC | (stopped) | sweep1 first attempt: A100-80GB never scheduled in ~55 min (0 containers, $0 spent); stopped |
| 2026-11-06 | ap-kYajfWtgyO3OPmzJ1fSN0O | `modal run --detach modal/kernel_sweep.py --tag sweep1` | relaunch with gpu fallback [A100-80GB, A100-40GB]; client log logs/kernel_sweep1b_client.log |
| 2026-11-06 | ap-N1q7OiAH5UWFH8vxV9xgO8 | `modal run --detach modal/kernel_sweep.py --tag combo1 --preset combo` | combo of winners (KC64/UNROLL4/MINB3), all configs correctness-checked; log logs/kernel_sweep2_client.log |
