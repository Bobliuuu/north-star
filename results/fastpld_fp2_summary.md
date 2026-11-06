## Wall tok/s (geomean over tasks; each task = mean of its reps; reps pooled across files)
| cell | n tasks | graphed PyTorch plain | fast plain | pld3 g8 | sam2 g12 |
|---|---|---|---|---|---|
| natural | 21 | 92.5 | 254.0 | 288.1 | 293.0 |
| ctx2048 | 7 | 92.4 | 248.2 | 371.2 | 381.7 |
| ctx8192 | 7 | 87.0 | 243.2 | 344.9 | 354.2 |

## Speedup (geomean over tasks of mean-over-reps tok/s ratio) vs graphed PyTorch plain (93.7 orig baseline) and vs fast plain
| cell | config | vs graphed PyTorch plain | vs fast plain | tok/round | rep-to-rep spread (max |r1-r2|/mean over tasks) | reps/task | worst task vs fast | best task vs fast |
|---|---|---|---|---|---|---|---|---|
| natural | pld3 g8 | 3.11x | 1.13x | 1.24 | 3.9% | 2 | 1.00x | 1.84x |
| natural | sam2 g12 | 3.17x | 1.15x | 1.28 | 0.8% | 2 | 1.02x | 1.77x |
| ctx2048 | pld3 g8 | 4.02x | 1.50x | 1.78 | 2.3% | 2 | 1.07x | 2.86x |
| ctx2048 | sam2 g12 | 4.13x | 1.54x | 1.89 | 0.8% | 2 | 1.05x | 3.78x |
| ctx8192 | pld3 g8 | 3.96x | 1.42x | 1.67 | 0.2% | 2 | 1.06x | 2.87x |
| ctx8192 | sam2 g12 | 4.07x | 1.46x | 1.85 | 0.7% | 2 | 1.04x | 3.54x |

## By kind (geomean speedup vs graphed PyTorch plain / vs fast plain; n tasks)
| cell | config | code | edit | extract | prose | summarize | terminal |
|---|---|---|---|---|---|---|---|
| natural | pld3 g8 | 3.18 / 1.16 (n=7) | 2.97 / 1.08 (n=2) | 3.96 / 1.45 (n=2) | 2.93 / 1.06 (n=4) | 3.03 / 1.11 (n=2) | 2.93 / 1.07 (n=4) |
| natural | sam2 g12 | 3.20 / 1.16 (n=7) | 3.12 / 1.14 (n=2) | 3.87 / 1.42 (n=2) | 3.04 / 1.10 (n=4) | 3.17 / 1.16 (n=2) | 2.97 / 1.08 (n=4) |
| ctx2048 | pld3 g8 | 3.92 / 1.45 (n=2) | 7.70 / 2.86 (n=1) | 5.35 / 1.99 (n=1) | 2.87 / 1.07 (n=1) | 2.92 / 1.09 (n=1) | 3.18 / 1.19 (n=1) |
| ctx2048 | sam2 g12 | 3.86 / 1.43 (n=2) | 10.17 / 3.78 (n=1) | 5.64 / 2.09 (n=1) | 2.81 / 1.05 (n=1) | 2.95 / 1.10 (n=1) | 2.91 / 1.09 (n=1) |
| ctx8192 | pld3 g8 | 3.62 / 1.30 (n=2) | 8.02 / 2.87 (n=1) | 4.81 / 1.72 (n=1) | 2.97 / 1.06 (n=1) | 3.03 / 1.08 (n=1) | 3.39 / 1.21 (n=1) |
| ctx8192 | sam2 g12 | 3.44 / 1.23 (n=2) | 9.89 / 3.54 (n=1) | 4.80 / 1.72 (n=1) | 2.90 / 1.04 (n=1) | 3.06 / 1.10 (n=1) | 3.73 / 1.33 (n=1) |

## Token identity (all reps of all files; counts of live runs)
| cell | config | identical to fast-plain | identical to graphed-plain | identical to HF (where run) | runs |
|---|---|---|---|---|---|
| natural | pld3 g8 | 4/42 | 6/42 | 10/42 | 42 |
| natural | sam2 g12 | 6/42 | 4/42 | 4/42 | 42 |
| ctx2048 | pld3 g8 | 6/14 | 4/14 | 2/14 | 14 |
| ctx2048 | sam2 g12 | 4/14 | 4/14 | 2/14 | 14 |
| ctx8192 | pld3 g8 | 4/14 | 0/14 | n/a | 14 |
| ctx8192 | sam2 g12 | 2/14 | 0/14 | n/a | 14 |

## Plain executors vs each other / HF (traces)
- natural: fast-plain == graphed-plain on 0/21 tasks; fast-plain == HF on 1/21; graphed-plain == HF on 3/21
- ctx2048: fast-plain == graphed-plain on 2/7 tasks; fast-plain == HF on 1/7; graphed-plain == HF on 1/7
- ctx8192: fast-plain == graphed-plain on 0/7 tasks; fast-plain == HF on n/a; graphed-plain == HF on n/a

## Logit margins at divergences (teacher-forced on common prefix; a = reference token, b = diverging token; logit_a_minus_b = how far a beat b)
| cell | task | what | at token | top2 margin (fast T=1) | logit a-b (fast T=1) | top2 margin (HF) | logit a-b (HF) |
|---|---|---|---|---|---|---|---|
| natural | py_lru | fast_vs_base | 0 | 0.000 | 0.000 | 0.125 | 0.125 |
| natural | py_lru | fast_vs_hf | 0 | 0.000 | 0.000 | 0.125 | 0.125 |
| natural | py_lru | spec_sam2_g12_vs_fast | 70 | 0.000 | 0.000 | 0.125 | -0.125 |
| natural | py_lru | spec_pld3_g8_vs_fast | 68 | 0.125 | -0.125 | 0.000 | 0.000 |
| natural | py_interval | fast_vs_base | 5 | 0.000 | 0.000 | 0.125 | 0.125 |
| natural | py_interval | fast_vs_hf | 5 | 0.000 | 0.000 | 0.125 | 0.125 |
| natural | py_dataclass | fast_vs_base | 121 | 0.125 | -0.125 | 0.125 | -0.125 |
| natural | py_dataclass | fast_vs_hf | 108 | 0.000 | 0.000 | 0.000 | 0.000 |
| natural | py_dataclass | spec_sam2_g12_vs_fast | 121 | 0.125 | 0.125 | 0.125 | 0.125 |
| natural | py_dataclass | spec_pld3_g8_vs_fast | 108 | 0.000 | 0.000 | 0.000 | 0.000 |
| natural | rs_stack | fast_vs_base | 119 | 0.125 | 0.125 | 0.250 | 0.250 |
| natural | rs_stack | spec_sam2_g12_vs_fast | 137 | 0.125 | 0.125 | 0.125 | -0.125 |
| natural | go_worker | fast_vs_base | 116 | 0.250 | -0.250 | 0.250 | -0.250 |
| natural | go_worker | fast_vs_hf | 3 | 0.125 | -0.125 | 0.125 | -0.125 |
| natural | go_worker | spec_sam2_g12_vs_fast | 119 | 0.000 | 0.000 | 0.000 | 0.000 |
| natural | go_worker | spec_pld3_g8_vs_fast | 119 | 0.000 | 0.000 | 0.000 | 0.000 |
| natural | sql_topn | fast_vs_base | 34 | 0.000 | 0.000 | 0.000 | 0.000 |
| natural | sql_topn | fast_vs_hf | 34 | 0.000 | 0.000 | 0.000 | 0.000 |
| natural | sql_topn | spec_sam2_g12_vs_fast | 34 | 0.000 | 0.000 | 0.000 | 0.000 |
| natural | sql_topn | spec_pld3_g8_vs_fast | 65 | 0.125 | -0.125 | 0.000 | 0.000 |
| natural | sh_logs | fast_vs_base | 172 | 0.125 | -0.125 | 0.250 | -0.250 |
| natural | sh_logs | fast_vs_hf | 54 | 0.125 | -0.125 | 0.125 | 0.125 |
| natural | sh_logs | spec_sam2_g12_vs_fast | 54 | 0.125 | 0.125 | 0.125 | -0.125 |
| natural | sh_logs | spec_pld3_g8_vs_fast | 54 | 0.125 | 0.125 | 0.125 | -0.125 |
| natural | term_pytest_fail | fast_vs_base | 1 | 0.250 | 0.250 | 0.125 | -0.125 |
| natural | term_pytest_fail | fast_vs_hf | 51 | 0.125 | -0.125 | 0.125 | -0.125 |
| natural | term_pytest_fail | spec_sam2_g12_vs_fast | 64 | 0.000 | 0.000 | 0.500 | 0.500 |
| natural | term_pytest_fail | spec_pld3_g8_vs_fast | 53 | 0.125 | 0.125 | 0.125 | 0.125 |
| natural | term_git_conflict | fast_vs_base | 102 | 0.000 | 0.000 | 0.125 | 0.125 |
| natural | term_git_conflict | fast_vs_hf | 102 | 0.000 | 0.000 | 0.125 | 0.125 |
| natural | term_git_conflict | spec_sam2_g12_vs_fast | 64 | 0.250 | 0.250 | 0.375 | 0.375 |
| natural | term_git_conflict | spec_pld3_g8_vs_fast | 102 | 0.000 | 0.000 | 0.125 | -0.125 |
| natural | term_make | fast_vs_base | 98 | 0.500 | 0.500 | 0.375 | -0.375 |
| natural | term_make | fast_vs_hf | 47 | 0.125 | -0.125 | 0.500 | 0.500 |
| natural | term_make | spec_sam2_g12_vs_fast | 98 | 0.500 | -0.500 | 0.375 | 0.375 |
| natural | term_make | spec_pld3_g8_vs_fast | 98 | 0.500 | -0.500 | 0.375 | 0.375 |
| natural | term_docker | fast_vs_base | 150 | 0.125 | -0.125 | 0.125 | 0.250 |
| natural | term_docker | fast_vs_hf | 150 | 0.125 | -0.125 | 0.125 | 0.250 |
| natural | term_docker | spec_pld3_g8_vs_fast | 16 | 0.375 | 0.375 | 0.625 | 1.000 |
| natural | pr_hbm | fast_vs_base | 40 | 0.000 | 0.000 | 0.000 | 0.000 |
| natural | pr_hbm | fast_vs_hf | 40 | 0.000 | 0.000 | 0.000 | 0.000 |
| natural | pr_hbm | spec_sam2_g12_vs_fast | 55 | 0.125 | 0.125 | 0.125 | 0.125 |
| natural | pr_hbm | spec_pld3_g8_vs_fast | 41 | 0.250 | 0.250 | 0.125 | 0.125 |
| natural | pr_story | fast_vs_base | 130 | 0.125 | 0.125 | 0.000 | 0.000 |
| natural | pr_story | fast_vs_hf | 132 | 0.125 | 0.125 | 0.125 | 0.125 |
| natural | pr_story | spec_sam2_g12_vs_fast | 54 | 0.125 | 0.125 | 0.125 | -0.125 |
| natural | pr_story | spec_pld3_g8_vs_fast | 81 | 0.125 | 0.125 | 0.000 | 0.000 |
| natural | pr_tradeoff | fast_vs_base | 51 | 0.000 | 0.000 | 0.000 | 0.000 |
| natural | pr_tradeoff | fast_vs_hf | 50 | 0.125 | 0.125 | 0.000 | 0.000 |
| natural | pr_tradeoff | spec_sam2_g12_vs_fast | 51 | 0.000 | 0.000 | 0.000 | 0.000 |
| natural | pr_tradeoff | spec_pld3_g8_vs_fast | 50 | 0.125 | -0.125 | 0.000 | 0.000 |
| natural | pr_reason | fast_vs_base | 110 | 0.250 | 0.250 | 0.250 | 0.250 |
| natural | pr_reason | fast_vs_hf | 86 | 0.250 | -0.250 | 0.125 | 0.125 |
| natural | pr_reason | spec_sam2_g12_vs_fast | 86 | 0.250 | 0.375 | 0.125 | 0.375 |
| natural | pr_reason | spec_pld3_g8_vs_fast | 110 | 0.250 | -0.250 | 0.250 | -0.250 |
| natural | edit_shutil | fast_vs_base | 60 | 0.000 | 0.000 | 0.125 | 0.125 |
| natural | edit_shutil | fast_vs_hf | 53 | 0.125 | -0.125 | 0.250 | -0.250 |
| natural | edit_shutil | spec_sam2_g12_vs_fast | 45 | 0.000 | 0.000 | 0.000 | 0.000 |
| natural | edit_shutil | spec_pld3_g8_vs_fast | 67 | 0.125 | -0.125 | 0.125 | -0.125 |
| natural | edit_difflib | fast_vs_base | 96 | 0.000 | 0.000 | 0.000 | 0.000 |
| natural | edit_difflib | fast_vs_hf | 86 | 0.125 | -0.125 | 0.000 | 0.000 |
| natural | edit_difflib | spec_sam2_g12_vs_fast | 184 | 0.125 | 0.125 | 0.125 | 0.125 |
| natural | edit_difflib | spec_pld3_g8_vs_fast | 92 | 0.125 | 0.125 | 0.000 | 0.000 |
| natural | extract_textwrap | fast_vs_base | 84 | 0.125 | 0.125 | 0.250 | 0.250 |
| natural | extract_textwrap | fast_vs_hf | 84 | 0.125 | 0.125 | 0.250 | 0.250 |
| natural | extract_textwrap | spec_pld3_g8_vs_fast | 84 | 0.125 | -0.125 | 0.250 | -0.250 |
| natural | extract_heapq | fast_vs_base | 78 | 0.000 | 0.000 | 0.125 | -0.125 |
| natural | extract_heapq | fast_vs_hf | 70 | 0.000 | 0.000 | 0.125 | 0.125 |
| natural | extract_heapq | spec_sam2_g12_vs_fast | 133 | 0.000 | 0.000 | 0.000 | 0.000 |
| natural | extract_heapq | spec_pld3_g8_vs_fast | 70 | 0.000 | 0.000 | 0.125 | -0.125 |
| natural | summ_argparse | fast_vs_base | 158 | 0.125 | -0.125 | 0.000 | 0.000 |
| natural | summ_argparse | fast_vs_hf | 35 | 0.000 | 0.000 | 0.125 | 0.125 |
| natural | summ_argparse | spec_sam2_g12_vs_fast | 35 | 0.000 | 0.000 | 0.125 | -0.125 |
| natural | summ_argparse | spec_pld3_g8_vs_fast | 162 | 0.000 | 0.000 | 0.000 | 0.000 |
| natural | summ_shutil | fast_vs_base | 182 | 0.125 | 0.125 | 0.250 | 0.250 |
| natural | summ_shutil | fast_vs_hf | 182 | 0.125 | 0.125 | 0.250 | 0.250 |
| natural | summ_shutil | spec_sam2_g12_vs_fast | 182 | 0.125 | -0.125 | 0.250 | -0.250 |
| natural | summ_shutil | spec_pld3_g8_vs_fast | 182 | 0.125 | -0.125 | 0.250 | -0.250 |
| ctx2048 | py_lru | fast_vs_hf | 101 | 0.125 | -0.125 | 0.000 | 0.000 |
| ctx2048 | py_dataclass | fast_vs_base | 129 | 0.000 | 0.000 | 0.000 | 0.000 |
| ctx2048 | py_dataclass | fast_vs_hf | 116 | 0.125 | 0.125 | 0.000 | 0.000 |
| ctx2048 | py_dataclass | spec_sam2_g12_vs_fast | 116 | 0.125 | -0.125 | 0.000 | 0.000 |
| ctx2048 | py_dataclass | spec_pld3_g8_vs_fast | 116 | 0.125 | -0.125 | 0.000 | 0.000 |
| ctx2048 | pr_story | fast_vs_base | 62 | 0.000 | 0.000 | 0.125 | -0.125 |
| ctx2048 | pr_story | fast_vs_hf | 62 | 0.000 | 0.000 | 0.125 | -0.125 |
| ctx2048 | pr_story | spec_sam2_g12_vs_fast | 50 | 0.375 | 0.375 | 0.500 | 0.500 |
| ctx2048 | pr_story | spec_pld3_g8_vs_fast | 62 | 0.000 | 0.250 | 0.125 | 0.375 |
| ctx2048 | term_make | fast_vs_base | 18 | 0.125 | -0.125 | 0.125 | -0.125 |
| ctx2048 | term_make | fast_vs_hf | 1 | 0.250 | -0.250 | 0.000 | 0.000 |
| ctx2048 | term_make | spec_sam2_g12_vs_fast | 18 | 0.125 | 0.125 | 0.125 | 0.125 |
| ctx2048 | term_make | spec_pld3_g8_vs_fast | 32 | 0.000 | 0.000 | 0.125 | -0.125 |
| ctx2048 | extract_textwrap | fast_vs_base | 42 | 0.125 | -0.250 | 0.000 | -0.125 |
| ctx2048 | extract_textwrap | fast_vs_hf | 33 | 0.000 | 0.000 | 0.125 | 0.125 |
| ctx2048 | extract_textwrap | spec_sam2_g12_vs_fast | 42 | 0.125 | 0.125 | 0.000 | 0.000 |
| ctx2048 | extract_textwrap | spec_pld3_g8_vs_fast | 42 | 0.125 | 0.125 | 0.000 | 0.000 |
| ctx2048 | summ_argparse | fast_vs_base | 8 | 0.000 | 0.000 | 0.000 | 0.000 |
| ctx2048 | summ_argparse | fast_vs_hf | 8 | 0.000 | 0.000 | 0.000 | 0.000 |
| ctx2048 | summ_argparse | spec_sam2_g12_vs_fast | 116 | 0.125 | 0.125 | 0.000 | 0.000 |
| ctx8192 | py_lru | fast_vs_base | 141 | 0.125 | -0.125 | n/a | n/a |
| ctx8192 | py_lru | spec_sam2_g12_vs_fast | 135 | 0.000 | 0.000 | n/a | n/a |
| ctx8192 | py_lru | spec_pld3_g8_vs_fast | 71 | 0.125 | -0.125 | n/a | n/a |
| ctx8192 | py_dataclass | fast_vs_base | 132 | 0.000 | 0.000 | n/a | n/a |
| ctx8192 | py_dataclass | spec_sam2_g12_vs_fast | 132 | 0.000 | 0.000 | n/a | n/a |
| ctx8192 | py_dataclass | spec_pld3_g8_vs_fast | 132 | 0.000 | 0.000 | n/a | n/a |
| ctx8192 | pr_story | fast_vs_base | 150 | 0.500 | 0.500 | n/a | n/a |
| ctx8192 | pr_story | spec_sam2_g12_vs_fast | 65 | 0.000 | 0.000 | n/a | n/a |
| ctx8192 | pr_story | spec_pld3_g8_vs_fast | 102 | 0.125 | -0.125 | n/a | n/a |
| ctx8192 | term_make | fast_vs_base | 24 | 0.000 | 0.000 | n/a | n/a |
| ctx8192 | term_make | spec_sam2_g12_vs_fast | 127 | 0.250 | 0.250 | n/a | n/a |
| ctx8192 | edit_shutil | fast_vs_base | 6 | 0.125 | 0.125 | n/a | n/a |
| ctx8192 | extract_textwrap | fast_vs_base | 0 | 0.125 | -0.125 | n/a | n/a |
| ctx8192 | extract_textwrap | spec_sam2_g12_vs_fast | 48 | 0.125 | -0.125 | n/a | n/a |
| ctx8192 | extract_textwrap | spec_pld3_g8_vs_fast | 48 | 0.125 | -0.125 | n/a | n/a |
| ctx8192 | summ_argparse | fast_vs_base | 28 | 0.125 | -0.125 | n/a | n/a |
| ctx8192 | summ_argparse | spec_sam2_g12_vs_fast | 28 | 0.125 | 0.125 | n/a | n/a |
| ctx8192 | summ_argparse | spec_pld3_g8_vs_fast | 28 | 0.125 | 0.125 | n/a | n/a |

errors: 0 []
