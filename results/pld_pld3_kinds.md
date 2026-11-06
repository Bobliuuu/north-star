
#### natural (MEASURED live wall speedup vs plain graphed decode, geomean over tasks; n per kind in header)
| config | all | code (n=7) | edit (n=2) | extract (n=2) | prose (n=4) | summarize (n=2) | terminal (n=4) | worst task | identical |
|---|---|---|---|---|---|---|---|---|---|---|
| ngc g8 | 1.196 | 1.19 | 1.19 | 1.49 | 1.16 | 1.25 | 1.09 | 1.001 | 15/21 |
| pld3b g8 | 1.205 | 1.20 | 1.19 | 1.53 | 1.16 | 1.25 | 1.10 | 1.000 | 16/21 |
| pld3b g12 | 1.200 | 1.20 | 1.20 | 1.58 | 1.15 | 1.23 | 1.09 | 0.984 | 16/21 |
| sam g12 | 1.177 | 1.18 | 1.16 | 1.61 | 1.11 | 1.20 | 1.06 | 0.989 | 17/21 |
| sam2 g8 | 1.222 | 1.21 | 1.22 | 1.56 | 1.18 | 1.26 | 1.12 | 1.008 | 16/21 |
| sam2 g12 | 1.227 | 1.22 | 1.23 | 1.61 | 1.18 | 1.26 | 1.11 | 1.008 | 16/21 |

#### ctx2048 (MEASURED live wall speedup vs plain graphed decode, geomean over tasks; n per kind in header)
| config | all | code (n=2) | edit (n=1) | extract (n=1) | prose (n=1) | summarize (n=1) | terminal (n=1) | worst task | identical |
|---|---|---|---|---|---|---|---|---|---|---|
| ngc g8 | 1.443 | 1.52 | 3.44 | 1.19 | 1.06 | 1.15 | 1.13 | 1.059 | 6/7 |
| pld3b g8 | 1.481 | 1.55 | 3.87 | 1.20 | 1.06 | 1.15 | 1.16 | 1.062 | 6/7 |
| pld3b g12 | 1.500 | 1.56 | 4.32 | 1.18 | 1.06 | 1.13 | 1.14 | 1.062 | 6/7 |
| sam g12 | 1.502 | 1.57 | 4.75 | 1.19 | 1.09 | 1.09 | 1.06 | 1.058 | 7/7 |
| sam2 g8 | 1.504 | 1.54 | 4.13 | 1.22 | 1.07 | 1.19 | 1.14 | 1.070 | 7/7 |
| sam2 g12 | 1.548 | 1.57 | 4.89 | 1.22 | 1.07 | 1.18 | 1.14 | 1.071 | 7/7 |

#### ctx8192 (MEASURED live wall speedup vs plain graphed decode, geomean over tasks; n per kind in header)
| config | all | code (n=2) | edit (n=1) | extract (n=1) | prose (n=1) | summarize (n=1) | terminal (n=1) | worst task | identical |
|---|---|---|---|---|---|---|---|---|---|---|
| ngc g8 | 1.288 | 1.27 | 2.64 | 1.17 | 1.05 | 1.13 | 0.99 | 0.991 | 3/7 |
| pld3b g8 | 1.368 | 1.29 | 3.71 | 1.18 | 1.05 | 1.16 | 1.01 | 1.005 | 3/7 |
| pld3b g12 | 1.372 | 1.29 | 4.11 | 1.16 | 1.05 | 1.11 | 0.99 | 0.991 | 3/7 |
| sam g12 | 1.428 | 1.36 | 4.45 | 1.20 | 1.07 | 1.12 | 1.02 | 1.016 | 5/7 |
| sam2 g8 | 1.394 | 1.33 | 3.91 | 1.16 | 1.07 | 1.17 | 1.02 | 1.024 | 5/7 |
| sam2 g12 | 1.432 | 1.34 | 4.58 | 1.16 | 1.07 | 1.18 | 1.03 | 1.025 | 5/7 |
