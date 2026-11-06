## Wall tok/s (geomean over tasks; each task = mean of its reps; reps pooled across files)
| cell | n tasks | graphed PyTorch plain | fast plain | pld3 g8 | sam2 g12 |
|---|---|---|---|---|---|
| natural | 21 | 96.5 | 301.6 | 338.5 | 346.9 |
| ctx2048 | 7 | 97.0 | 293.9 | 417.5 | 425.3 |
| ctx8192 | 7 | 91.0 | 287.4 | 378.9 | 377.1 |

## Speedup (geomean over tasks of mean-over-reps tok/s ratio) vs graphed PyTorch plain (93.7 orig baseline) and vs fast plain
| cell | config | vs graphed PyTorch plain | vs fast plain | tok/round | rep-to-rep spread (max |r1-r2|/mean over tasks) | reps/task | worst task vs fast | best task vs fast |
|---|---|---|---|---|---|---|---|---|
| natural | pld3 g8 | 3.51x | 1.12x | 1.23 | 0.1% | 2 | 0.97x | 1.83x |
| natural | sam2 g12 | 3.59x | 1.15x | 1.30 | 0.4% | 2 | 0.96x | 1.90x |
| ctx2048 | pld3 g8 | 4.30x | 1.42x | 1.69 | 0.1% | 2 | 0.99x | 2.86x |
| ctx2048 | sam2 g12 | 4.38x | 1.45x | 1.81 | 0.5% | 2 | 1.01x | 3.66x |
| ctx8192 | pld3 g8 | 4.16x | 1.32x | 1.57 | 0.1% | 2 | 0.99x | 2.84x |
| ctx8192 | sam2 g12 | 4.14x | 1.31x | 1.74 | 1.0% | 2 | 0.94x | 3.28x |

## By kind (geomean speedup vs graphed PyTorch plain / vs fast plain; n tasks)
| cell | config | code | edit | extract | prose | summarize | terminal |
|---|---|---|---|---|---|---|---|
| natural | pld3 g8 | 3.56 / 1.14 (n=7) | 3.37 / 1.08 (n=2) | 4.50 / 1.45 (n=2) | 3.30 / 1.05 (n=4) | 3.44 / 1.11 (n=2) | 3.30 / 1.05 (n=4) |
| natural | sam2 g12 | 3.62 / 1.16 (n=7) | 3.70 / 1.19 (n=2) | 4.52 / 1.46 (n=2) | 3.41 / 1.09 (n=4) | 3.54 / 1.14 (n=2) | 3.31 / 1.06 (n=4) |
| ctx2048 | pld3 g8 | 4.12 / 1.36 (n=2) | 8.68 / 2.86 (n=1) | 6.21 / 2.05 (n=1) | 3.24 / 1.07 (n=1) | 3.01 / 0.99 (n=1) | 3.07 / 1.01 (n=1) |
| ctx2048 | sam2 g12 | 3.91 / 1.29 (n=2) | 11.10 / 3.66 (n=1) | 6.16 / 2.03 (n=1) | 3.13 / 1.03 (n=1) | 3.12 / 1.03 (n=1) | 3.05 / 1.01 (n=1) |
| ctx8192 | pld3 g8 | 3.68 / 1.17 (n=2) | 8.96 / 2.84 (n=1) | 5.08 / 1.61 (n=1) | 3.33 / 1.05 (n=1) | 3.12 / 0.99 (n=1) | 3.38 / 1.07 (n=1) |
| ctx8192 | sam2 g12 | 3.47 / 1.10 (n=2) | 10.35 / 3.28 (n=1) | 5.22 / 1.65 (n=1) | 3.14 / 1.00 (n=1) | 3.09 / 0.98 (n=1) | 3.32 / 1.05 (n=1) |

## Token identity (all reps of all files; counts of live runs)
| cell | config | identical to fast-plain | identical to graphed-plain | identical to HF (where run) | runs |
|---|---|---|---|---|---|
| natural | pld3 g8 | 16/42 | 6/42 | n/a | 42 |
| natural | sam2 g12 | 4/42 | 6/42 | n/a | 42 |
| ctx2048 | pld3 g8 | 4/14 | 2/14 | n/a | 14 |
| ctx2048 | sam2 g12 | 4/14 | 2/14 | n/a | 14 |
| ctx8192 | pld3 g8 | 6/14 | 0/14 | n/a | 14 |
| ctx8192 | sam2 g12 | 4/14 | 0/14 | n/a | 14 |

## Plain executors vs each other / HF (traces)
- natural: fast-plain == graphed-plain on 3/21 tasks; fast-plain == HF on n/a; graphed-plain == HF on n/a
- ctx2048: fast-plain == graphed-plain on 1/7 tasks; fast-plain == HF on n/a; graphed-plain == HF on n/a
- ctx8192: fast-plain == graphed-plain on 0/7 tasks; fast-plain == HF on n/a; graphed-plain == HF on n/a

## Logit margins at divergences (teacher-forced on common prefix; a = reference token, b = diverging token; logit_a_minus_b = how far a beat b)
| cell | task | what | at token | top2 margin (fast T=1) | logit a-b (fast T=1) | top2 margin (HF) | logit a-b (HF) |
|---|---|---|---|---|---|---|---|
| natural | py_lru | fast_vs_base | 100 | 0.000 | 0.000 | n/a | n/a |
| natural | py_lru | spec_sam2_g12_vs_fast | 100 | 0.000 | 0.000 | n/a | n/a |
| natural | py_lru | spec_pld3_g8_vs_fast | 100 | 0.000 | 0.000 | n/a | n/a |
| natural | py_interval | fast_vs_base | 17 | 0.125 | 0.125 | n/a | n/a |
| natural | py_interval | spec_sam2_g12_vs_fast | 17 | 0.125 | -0.125 | n/a | n/a |
| natural | py_interval | spec_pld3_g8_vs_fast | 30 | 0.000 | 0.000 | n/a | n/a |
| natural | py_dataclass | fast_vs_base | 121 | 0.125 | -0.125 | n/a | n/a |
| natural | py_dataclass | spec_sam2_g12_vs_fast | 157 | 0.000 | 0.000 | n/a | n/a |
| natural | py_dataclass | spec_pld3_g8_vs_fast | 157 | 0.000 | 0.000 | n/a | n/a |
| natural | rs_stack | fast_vs_base | 153 | 0.125 | 0.125 | n/a | n/a |
| natural | rs_stack | spec_sam2_g12_vs_fast | 153 | 0.125 | -0.125 | n/a | n/a |
| natural | rs_stack | spec_pld3_g8_vs_fast | 153 | 0.125 | -0.125 | n/a | n/a |
| natural | go_worker | fast_vs_base | 135 | 0.250 | -0.250 | n/a | n/a |
| natural | go_worker | spec_sam2_g12_vs_fast | 153 | 0.125 | 0.125 | n/a | n/a |
| natural | go_worker | spec_pld3_g8_vs_fast | 119 | 0.125 | 0.125 | n/a | n/a |
| natural | sql_topn | fast_vs_base | 34 | 0.000 | 0.000 | n/a | n/a |
| natural | sql_topn | spec_sam2_g12_vs_fast | 84 | 0.125 | -0.125 | n/a | n/a |
| natural | sql_topn | spec_pld3_g8_vs_fast | 74 | 0.000 | 0.000 | n/a | n/a |
| natural | sh_logs | fast_vs_base | 97 | 0.000 | 0.000 | n/a | n/a |
| natural | sh_logs | spec_sam2_g12_vs_fast | 155 | 0.250 | 0.250 | n/a | n/a |
| natural | sh_logs | spec_pld3_g8_vs_fast | 69 | 0.000 | 0.000 | n/a | n/a |
| natural | term_pytest_fail | fast_vs_base | 1 | 0.250 | 0.250 | n/a | n/a |
| natural | term_pytest_fail | spec_sam2_g12_vs_fast | 53 | 0.125 | 0.125 | n/a | n/a |
| natural | term_pytest_fail | spec_pld3_g8_vs_fast | 53 | 0.125 | 0.125 | n/a | n/a |
| natural | term_git_conflict | fast_vs_base | 106 | 0.125 | -0.125 | n/a | n/a |
| natural | term_git_conflict | spec_sam2_g12_vs_fast | 106 | 0.125 | 0.125 | n/a | n/a |
| natural | term_make | fast_vs_base | 57 | 1.000 | 1.000 | n/a | n/a |
| natural | term_make | spec_sam2_g12_vs_fast | 57 | 1.000 | -1.000 | n/a | n/a |
| natural | term_make | spec_pld3_g8_vs_fast | 57 | 1.000 | -1.000 | n/a | n/a |
| natural | term_docker | fast_vs_base | 1 | 0.125 | 0.125 | n/a | n/a |
| natural | term_docker | spec_sam2_g12_vs_fast | 33 | 0.000 | 0.000 | n/a | n/a |
| natural | pr_hbm | spec_sam2_g12_vs_fast | 41 | 0.250 | -0.250 | n/a | n/a |
| natural | pr_hbm | spec_pld3_g8_vs_fast | 41 | 0.250 | -0.250 | n/a | n/a |
| natural | pr_story | fast_vs_base | 79 | 0.250 | -0.250 | n/a | n/a |
| natural | pr_story | spec_sam2_g12_vs_fast | 99 | 0.000 | 0.000 | n/a | n/a |
| natural | pr_tradeoff | fast_vs_base | 51 | 0.000 | 0.000 | n/a | n/a |
| natural | pr_tradeoff | spec_sam2_g12_vs_fast | 50 | 0.125 | -0.125 | n/a | n/a |
| natural | pr_tradeoff | spec_pld3_g8_vs_fast | 50 | 0.125 | -0.125 | n/a | n/a |
| natural | pr_reason | fast_vs_base | 86 | 0.125 | -0.125 | n/a | n/a |
| natural | pr_reason | spec_sam2_g12_vs_fast | 86 | 0.125 | 0.375 | n/a | n/a |
| natural | edit_shutil | fast_vs_base | 183 | 0.000 | 0.000 | n/a | n/a |
| natural | edit_shutil | spec_sam2_g12_vs_fast | 60 | 0.125 | -0.125 | n/a | n/a |
| natural | edit_difflib | fast_vs_base | 34 | 0.000 | 0.000 | n/a | n/a |
| natural | edit_difflib | spec_sam2_g12_vs_fast | 86 | 0.250 | -0.250 | n/a | n/a |
| natural | edit_difflib | spec_pld3_g8_vs_fast | 86 | 0.250 | -0.250 | n/a | n/a |
| natural | extract_heapq | fast_vs_base | 70 | 0.125 | -0.125 | n/a | n/a |
| natural | extract_heapq | spec_sam2_g12_vs_fast | 70 | 0.125 | 0.125 | n/a | n/a |
| natural | summ_argparse | fast_vs_base | 35 | 0.000 | 0.000 | n/a | n/a |
| natural | summ_argparse | spec_sam2_g12_vs_fast | 121 | 0.125 | 0.125 | n/a | n/a |
| natural | summ_argparse | spec_pld3_g8_vs_fast | 35 | 0.000 | 0.000 | n/a | n/a |
| ctx2048 | py_lru | fast_vs_base | 19 | 0.000 | 0.000 | n/a | n/a |
| ctx2048 | py_lru | spec_sam2_g12_vs_fast | 128 | 0.125 | 0.125 | n/a | n/a |
| ctx2048 | py_lru | spec_pld3_g8_vs_fast | 128 | 0.125 | 0.125 | n/a | n/a |
| ctx2048 | py_dataclass | fast_vs_base | 114 | 0.250 | 0.250 | n/a | n/a |
| ctx2048 | py_dataclass | spec_sam2_g12_vs_fast | 114 | 0.250 | -0.250 | n/a | n/a |
| ctx2048 | py_dataclass | spec_pld3_g8_vs_fast | 114 | 0.250 | -0.250 | n/a | n/a |
| ctx2048 | pr_story | fast_vs_base | 62 | 0.125 | 0.125 | n/a | n/a |
| ctx2048 | pr_story | spec_sam2_g12_vs_fast | 62 | 0.125 | -0.125 | n/a | n/a |
| ctx2048 | pr_story | spec_pld3_g8_vs_fast | 62 | 0.125 | -0.125 | n/a | n/a |
| ctx2048 | term_make | fast_vs_base | 1 | 0.125 | 0.125 | n/a | n/a |
| ctx2048 | term_make | spec_sam2_g12_vs_fast | 15 | 0.000 | 0.000 | n/a | n/a |
| ctx2048 | term_make | spec_pld3_g8_vs_fast | 41 | 0.000 | 0.000 | n/a | n/a |
| ctx2048 | extract_textwrap | fast_vs_base | 42 | 0.000 | 0.000 | n/a | n/a |
| ctx2048 | summ_argparse | fast_vs_base | 116 | 0.125 | 0.125 | n/a | n/a |
| ctx2048 | summ_argparse | spec_sam2_g12_vs_fast | 129 | 0.000 | 0.000 | n/a | n/a |
| ctx2048 | summ_argparse | spec_pld3_g8_vs_fast | 129 | 0.000 | 0.000 | n/a | n/a |
| ctx8192 | py_lru | fast_vs_base | 25 | 0.125 | -0.125 | n/a | n/a |
| ctx8192 | py_lru | spec_sam2_g12_vs_fast | 25 | 0.125 | 0.125 | n/a | n/a |
| ctx8192 | py_lru | spec_pld3_g8_vs_fast | 25 | 0.125 | 0.125 | n/a | n/a |
| ctx8192 | py_dataclass | fast_vs_base | 190 | 0.125 | 0.125 | n/a | n/a |
| ctx8192 | py_dataclass | spec_sam2_g12_vs_fast | 106 | 0.000 | 0.000 | n/a | n/a |
| ctx8192 | pr_story | fast_vs_base | 49 | 0.250 | -0.250 | n/a | n/a |
| ctx8192 | pr_story | spec_sam2_g12_vs_fast | 105 | 0.125 | -0.125 | n/a | n/a |
| ctx8192 | pr_story | spec_pld3_g8_vs_fast | 147 | 0.250 | 0.250 | n/a | n/a |
| ctx8192 | term_make | fast_vs_base | 111 | 0.125 | -0.125 | n/a | n/a |
| ctx8192 | term_make | spec_sam2_g12_vs_fast | 95 | 0.875 | 1.375 | n/a | n/a |
| ctx8192 | term_make | spec_pld3_g8_vs_fast | 133 | 0.625 | -0.625 | n/a | n/a |
| ctx8192 | edit_shutil | fast_vs_base | 6 | 0.125 | -0.125 | n/a | n/a |
| ctx8192 | extract_textwrap | fast_vs_base | 54 | 0.125 | 0.125 | n/a | n/a |
| ctx8192 | extract_textwrap | spec_sam2_g12_vs_fast | 61 | 0.125 | -0.125 | n/a | n/a |
| ctx8192 | extract_textwrap | spec_pld3_g8_vs_fast | 61 | 0.125 | -0.125 | n/a | n/a |
| ctx8192 | summ_argparse | fast_vs_base | 9 | 0.125 | -0.125 | n/a | n/a |

errors: 0 []
