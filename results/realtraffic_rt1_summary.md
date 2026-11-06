## Tasks 99; skipped: []

Prompt tokens: min 160 p25 249 median 466 p75 2421 max 7161; generated n (mean 176, <48 tokens: 0 tasks)

| config | geomean | arith mean | median | wins(>1.02) | ties | losses(<0.98) | worst | best | exact (runs) | tokens in draft rounds | accepted draft tok / generated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pld3 g8 fixed | 1.123x | 1.136x | 1.059x | 70 | 26 | 3 | 0.961x | 1.857x | 121/121 | 25.5% | 17.3% |
| sam2 g12 fixed | 1.151x | 1.167x | 1.083x | 82 | 15 | 2 | 0.960x | 1.965x | 121/121 | 40.0% | 21.2% |
| sam2 g12 bo | 1.150x | 1.164x | 1.089x | 84 | 13 | 2 | 0.965x | 1.948x | 121/121 | 40.0% | 20.8% |
| sam2 g12 bo+off | 1.136x | 1.150x | 1.073x | 82 | 15 | 2 | 0.977x | 1.938x | 121/121 | 33.8% | 18.3% |
| sam2 g12 off | 1.138x | 1.152x | 1.066x | 77 | 19 | 3 | 0.974x | 1.973x | 121/121 | 33.8% | 18.8% |

### Per-category geomean speedup vs exact plain (n tasks, median prompt tokens)

| category | n | med prompt | pld3 g8 fixed | sam2 g12 fixed | sam2 g12 bo | sam2 g12 bo+off | sam2 g12 off |
|---|---|---|---|---|---|---|---|
| canitedit | 9 | 397 | 1.30x | 1.36x | 1.36x | 1.34x | 1.35x |
| cnn_summ | 9 | 1098 | 1.02x | 1.04x | 1.04x | 1.03x | 1.03x |
| glaive_tool | 9 | 293 | 1.07x | 1.09x | 1.08x | 1.08x | 1.09x |
| gov4k | 9 | 2709 | 1.01x | 1.01x | 1.02x | 1.01x | 1.01x |
| gov8k | 9 | 5143 | 1.02x | 1.04x | 1.04x | 1.03x | 1.02x |
| humaneval | 9 | 232 | 1.13x | 1.18x | 1.17x | 1.16x | 1.16x |
| mbpp | 9 | 177 | 1.13x | 1.18x | 1.18x | 1.17x | 1.17x |
| quality_qa | 9 | 6526 | 1.37x | 1.38x | 1.36x | 1.32x | 1.34x |
| squad_qa | 9 | 275 | 1.28x | 1.32x | 1.30x | 1.26x | 1.28x |
| ultrachat | 9 | 1031 | 1.03x | 1.04x | 1.04x | 1.03x | 1.03x |
| xsum_summ | 9 | 429 | 1.06x | 1.11x | 1.12x | 1.11x | 1.10x |

### Worst 8 tasks per config

- pld3 g8 fixed: gov8k_03_452 0.96x (P=4697,n=192), gov4k_07_870 0.97x (P=2925,n=192), cnn_summ_06_10117 0.97x (P=1176,n=192), gov4k_08_332 0.98x (P=2883,n=192), ultrachat_05_250 0.99x (P=274,n=192), ultrachat_03_4377 0.99x (P=688,n=192), gov8k_02_389 0.99x (P=2925,n=192), gov8k_04_827 0.99x (P=5192,n=192)
- sam2 g12 fixed: ultrachat_01_1583 0.96x (P=1404,n=192), gov4k_08_332 0.97x (P=2883,n=192), ultrachat_03_4377 0.98x (P=688,n=192), cnn_summ_01_791 0.99x (P=1098,n=173), gov4k_00_664 0.99x (P=2942,n=192), gov8k_04_827 0.99x (P=5192,n=192), gov4k_03_136 1.00x (P=2686,n=192), ultrachat_05_250 1.00x (P=274,n=192)
- sam2 g12 bo: ultrachat_01_1583 0.97x (P=1404,n=192), gov4k_08_332 0.97x (P=2883,n=192), ultrachat_03_4377 0.98x (P=688,n=192), cnn_summ_01_791 0.99x (P=1098,n=173), gov4k_00_664 1.00x (P=2942,n=192), gov4k_03_136 1.00x (P=2686,n=192), ultrachat_05_250 1.00x (P=274,n=192), gov4k_07_870 1.01x (P=2925,n=192)
- sam2 g12 bo+off: gov4k_08_332 0.98x (P=2883,n=192), ultrachat_01_1583 0.98x (P=1404,n=192), gov8k_08_622 0.98x (P=5519,n=192), ultrachat_03_4377 0.99x (P=688,n=192), gov4k_03_136 0.99x (P=2686,n=192), gov8k_04_827 0.99x (P=5192,n=192), cnn_summ_01_791 1.00x (P=1098,n=173), gov4k_00_664 1.00x (P=2942,n=192)
- sam2 g12 off: gov4k_08_332 0.97x (P=2883,n=192), ultrachat_01_1583 0.98x (P=1404,n=192), gov8k_04_827 0.98x (P=5192,n=192), gov8k_08_622 0.98x (P=5519,n=192), gov4k_03_136 0.99x (P=2686,n=192), ultrachat_03_4377 0.99x (P=688,n=192), gov4k_07_870 0.99x (P=2925,n=192), cnn_summ_01_791 0.99x (P=1098,n=173)

### Variance: 2-rep subset (rep-to-rep spread of spec tok/s, and of the speedup ratio using each rep's own plain)

- pld3 g8 fixed: 22 tasks, per-task rep-to-rep ratio difference mean 0.2% max 0.7%; geomean rep0 1.158x rep1 1.158x
- sam2 g12 fixed: 22 tasks, per-task rep-to-rep ratio difference mean 0.5% max 1.5%; geomean rep0 1.179x rep1 1.176x

### Sensitivity: geomean excluding tasks with n<48 generated tokens, and excluding the 22 rep2-subset tasks (single-rep tasks only)

- pld3 g8 fixed: n>=48 1.123x (99 tasks); single-rep tasks 1.113x (77)
- sam2 g12 fixed: n>=48 1.151x (99 tasks); single-rep tasks 1.143x (77)
- sam2 g12 bo: n>=48 1.150x (99 tasks); single-rep tasks 1.142x (77)
- sam2 g12 bo+off: n>=48 1.136x (99 tasks); single-rep tasks 1.131x (77)
- sam2 g12 off: n>=48 1.138x (99 tasks); single-rep tasks 1.132x (77)

### Non-identical runs (exactness failures)

none
