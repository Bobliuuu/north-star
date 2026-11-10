# moe-tune runs (Modal H100, workspace arnold-17059; est at $3.95/h)

| date | app id | command | seconds | est $ | notes |
|---|---|---|---|---|---|
| 2026-11-09 | ap-DwG6pJlatDMGGBIyJvpK8O | `modal run modal/moe_tune_test.py` | ~300 (est) | ~0.33 | kernel sweep 1: modes 0/3/5, warps/stages/BN; all bitwise; results/moe_tune_kernel.json |
| 2026-11-09 | ap-xZqQvQGZPtH2M0V8FhVetD | `modal run modal/moe_tune_test2.py` | ~360 (est) | ~0.40 | kernel sweep 2: mode-5 tile width x BN x warps, lin_bt TW 16/32/64/128; tw64+bn128 not bitwise, TW128 out of smem; results/moe_tune_kernel2.json |
| 2026-11-09 | ap-VW2WoDeHVby7ClDa7JZ4K7 | `modal run modal/ab_moetune_run.py::main --tag mt1` | 227 | 0.25 | real North A/B ref/main/moe5/lin64/new; 30/30 bitwise; results/abmoetune_mt1.jsonl, logs/abmoetune_mt1_client.log |

Total est ~$1.0.
