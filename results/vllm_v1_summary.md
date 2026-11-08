# vLLM baseline summary (v1); vLLM 0.33.0, kwargs {'dtype': 'bfloat16', 'max_model_len': 8192, 'max_num_seqs': 1, 'enable_prefix_caching': False, 'gpu_memory_utilization': 0.88}

## Geomean decode tok/s (per-task best of 2 reps; decode-only = (n-1)/(t_full - t_prefill+1tok))

| cell | config | n tasks | decode tok/s | e2e tok/s (incl. prefill) | vs vLLM plain | exact == vLLM plain output | mean n |
|---|---|---|---|---|---|---|---|
| natural | vLLM plain | 21 | 255.5 | 238.8 | 1.000x | 21/21 | 179 |
| natural | vLLM ngram_k8_n3 | 21 | 224.9 | 210.7 | 0.880x | 21/21 | 179 |
| natural | vLLM ngram_k12_n2to12 | 21 | 226.6 | 212.3 | 0.887x | 21/21 | 179 |
| natural | vLLM ngram_k5_n2to5 | 21 | 244.5 | 228.3 | 0.957x | 21/21 | 179 |
| mix99 | vLLM plain | 99 | 253.5 | 233.0 | 1.000x | 99/99 | 176 |
| mix99 | vLLM ngram_k8_n3 | 99 | 221.5 | 204.6 | 0.874x | 86/99 | 176 |
| mix99 | vLLM ngram_k12_n2to12 | 99 | 224.1 | 206.1 | 0.884x | 87/99 | 176 |
| mix99 | vLLM ngram_k5_n2to5 | 99 | 243.2 | 222.6 | 0.959x | 87/99 | 176 |

## Ours (from existing data, NOT re-run; different launches/hosts)

- natural (21), ours exact plain (H14): 222.4 tok/s geomean
- natural (21), ours exact sam2 g12 (H14): 280.6 tok/s geomean
- natural (21), ours old-fast plain (H13 fp2 / fp3): 254.1 / 301.6 tok/s geomean
- mix99 (H15), exact plain: 221.4; exact sam2 g12: 254.8; exact pld3 g8: 248.6 tok/s geomean

## Task-level ratios vs ours (vLLM decode tok/s / ours), geomean over tasks

- natural, vLLM plain / ours exact plain: 1.149x; / ours old-fast plain fp2: 1.006x; fp3: 0.847x
- mix99, vLLM plain / ours exact plain: 1.145x (n=99)
- natural, vLLM ngram_k8_n3 / ours exact sam2 g12: 0.802x (n=21)
- mix99, vLLM ngram_k8_n3 / ours exact sam2 g12: 0.869x (n=99)
- mix99, vLLM ngram_k8_n3 / ours exact pld3 g8: 0.891x (n=99)
- natural, vLLM ngram_k12_n2to12 / ours exact sam2 g12: 0.808x (n=21)
- mix99, vLLM ngram_k12_n2to12 / ours exact sam2 g12: 0.880x (n=99)
- mix99, vLLM ngram_k12_n2to12 / ours exact pld3 g8: 0.902x (n=99)
- natural, vLLM ngram_k5_n2to5 / ours exact sam2 g12: 0.871x (n=21)
- mix99, vLLM ngram_k5_n2to5 / ours exact sam2 g12: 0.954x (n=99)
- mix99, vLLM ngram_k5_n2to5 / ours exact pld3 g8: 0.978x (n=99)

## Per-category (mix99), vLLM spec / vLLM plain geomean

| category | ngram_k8_n3 | ngram_k12_n2to12 | ngram_k5_n2to5 |
|---|---|---|---|
| canitedit | 1.16 | 1.21 | 1.23 |
| cnn_summ | 0.76 | 0.76 | 0.84 |
| glaive_tool | 0.82 | 0.81 | 0.89 |
| gov4k | 0.76 | 0.75 | 0.84 |
| gov8k | 0.77 | 0.75 | 0.84 |
| humaneval | 0.86 | 0.87 | 0.95 |
| mbpp | 0.88 | 0.89 | 0.96 |
| quality_qa | 1.11 | 1.19 | 1.25 |
| squad_qa | 1.03 | 1.09 | 1.14 |
| ultrachat | 0.76 | 0.75 | 0.82 |
| xsum_summ | 0.81 | 0.82 | 0.92 |

task errors: 0
