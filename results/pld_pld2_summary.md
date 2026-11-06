
### LIVE ctx2048: wall speedup by config (all tasks)
| ngc g4 | n=7 | geomean 1.394 | min 1.057 | max 2.797 |
| pld2 g4 | n=7 | geomean 1.382 | min 1.061 | max 2.563 |
| pld3 g4 | n=7 | geomean 1.365 | min 1.064 | max 2.835 |
| pld3 g8 | n=7 | geomean 1.451 | min 1.071 | max 3.726 |
| sam g4 | n=7 | geomean 1.374 | min 1.063 | max 2.998 |
| sam g8 | n=7 | geomean 1.466 | min 1.067 | max 4.057 |
| samw g8 | n=7 | geomean 1.453 | min 1.072 | max 3.959 |

### LIVE ctx2048: by kind x config
| code      ngc g4 | n=2 | geomean 1.454 | min 1.326 | max 1.595 |
| code      pld2 g4 | n=2 | geomean 1.450 | min 1.303 | max 1.614 |
| code      pld3 g4 | n=2 | geomean 1.447 | min 1.312 | max 1.597 |
| code      pld3 g8 | n=2 | geomean 1.542 | min 1.374 | max 1.730 |
| code      sam g4 | n=2 | geomean 1.443 | min 1.311 | max 1.588 |
| code      sam g8 | n=2 | geomean 1.538 | min 1.369 | max 1.727 |
| code      samw g8 | n=2 | geomean 1.497 | min 1.355 | max 1.654 |
| edit      ngc g4 | n=1 | geomean 2.797 | min 2.797 | max 2.797 |
| edit      pld2 g4 | n=1 | geomean 2.563 | min 2.563 | max 2.563 |
| edit      pld3 g4 | n=1 | geomean 2.835 | min 2.835 | max 2.835 |
| edit      pld3 g8 | n=1 | geomean 3.726 | min 3.726 | max 3.726 |
| edit      sam g4 | n=1 | geomean 2.998 | min 2.998 | max 2.998 |
| edit      sam g8 | n=1 | geomean 4.057 | min 4.057 | max 4.057 |
| edit      samw g8 | n=1 | geomean 3.959 | min 3.959 | max 3.959 |
| extract   ngc g4 | n=1 | geomean 1.215 | min 1.215 | max 1.215 |
| extract   pld2 g4 | n=1 | geomean 1.228 | min 1.228 | max 1.228 |
| extract   pld3 g4 | n=1 | geomean 1.178 | min 1.178 | max 1.178 |
| extract   pld3 g8 | n=1 | geomean 1.191 | min 1.191 | max 1.191 |
| extract   sam g4 | n=1 | geomean 1.182 | min 1.182 | max 1.182 |
| extract   sam g8 | n=1 | geomean 1.193 | min 1.193 | max 1.193 |
| extract   samw g8 | n=1 | geomean 1.186 | min 1.186 | max 1.186 |
| prose     ngc g4 | n=1 | geomean 1.057 | min 1.057 | max 1.057 |
| prose     pld2 g4 | n=1 | geomean 1.061 | min 1.061 | max 1.061 |
| prose     pld3 g4 | n=1 | geomean 1.072 | min 1.072 | max 1.072 |
| prose     pld3 g8 | n=1 | geomean 1.082 | min 1.082 | max 1.082 |
| prose     sam g4 | n=1 | geomean 1.074 | min 1.074 | max 1.074 |
| prose     sam g8 | n=1 | geomean 1.080 | min 1.080 | max 1.080 |
| prose     samw g8 | n=1 | geomean 1.081 | min 1.081 | max 1.081 |
| summarize ngc g4 | n=1 | geomean 1.188 | min 1.188 | max 1.188 |
| summarize pld2 g4 | n=1 | geomean 1.190 | min 1.190 | max 1.190 |
| summarize pld3 g4 | n=1 | geomean 1.107 | min 1.107 | max 1.107 |
| summarize pld3 g8 | n=1 | geomean 1.105 | min 1.105 | max 1.105 |
| summarize sam g4 | n=1 | geomean 1.101 | min 1.101 | max 1.101 |
| summarize sam g8 | n=1 | geomean 1.104 | min 1.104 | max 1.104 |
| summarize samw g8 | n=1 | geomean 1.121 | min 1.121 | max 1.121 |
| terminal  ngc g4 | n=1 | geomean 1.133 | min 1.133 | max 1.133 |
| terminal  pld2 g4 | n=1 | geomean 1.153 | min 1.153 | max 1.153 |
| terminal  pld3 g4 | n=1 | geomean 1.064 | min 1.064 | max 1.064 |
| terminal  pld3 g8 | n=1 | geomean 1.071 | min 1.071 | max 1.071 |
| terminal  sam g4 | n=1 | geomean 1.063 | min 1.063 | max 1.063 |
| terminal  sam g8 | n=1 | geomean 1.067 | min 1.067 | max 1.067 |
| terminal  samw g8 | n=1 | geomean 1.072 | min 1.072 | max 1.072 |

identical to plain: 49/49; first_div values: []
mean accepted/verify (rounds with a draft), tok/round, draft_hit_rate by config:
| ngc g4 | acc/verify 1.70 | tok/round 1.67 | hit 0.32 | sim-vs-live 1.408/1.394 |
| pld2 g4 | acc/verify 1.66 | tok/round 1.63 | hit 0.32 | sim-vs-live 1.392/1.382 |
| pld3 g4 | acc/verify 2.50 | tok/round 1.60 | hit 0.21 | sim-vs-live 1.377/1.365 |
| pld3 g8 | acc/verify 3.95 | tok/round 1.84 | hit 0.18 | sim-vs-live 1.450/1.451 |
| sam g4 | acc/verify 2.56 | tok/round 1.63 | hit 0.21 | sim-vs-live 1.389/1.374 |
| sam g8 | acc/verify 4.11 | tok/round 1.90 | hit 0.17 | sim-vs-live 1.469/1.466 |
| samw g8 | acc/verify 3.60 | tok/round 1.86 | hit 0.19 | sim-vs-live 1.460/1.453 |

### LIVE ctx8192: wall speedup by config (all tasks)
| ngc g4 | n=7 | geomean 1.352 | min 1.005 | max 2.669 |
| pld2 g4 | n=7 | geomean 1.360 | min 1.016 | max 2.507 |
| pld3 g4 | n=7 | geomean 1.381 | min 1.027 | max 2.776 |
| pld3 g8 | n=7 | geomean 1.388 | min 1.024 | max 3.593 |
| sam g4 | n=7 | geomean 1.390 | min 1.021 | max 2.899 |
| sam g8 | n=7 | geomean 1.395 | min 1.016 | max 3.870 |
| samw g8 | n=7 | geomean 1.376 | min 1.017 | max 3.784 |

### LIVE ctx8192: by kind x config
| code      ngc g4 | n=2 | geomean 1.216 | min 1.178 | max 1.254 |
| code      pld2 g4 | n=2 | geomean 1.261 | min 1.187 | max 1.340 |
| code      pld3 g4 | n=2 | geomean 1.295 | min 1.222 | max 1.372 |
| code      pld3 g8 | n=2 | geomean 1.360 | min 1.268 | max 1.459 |
| code      sam g4 | n=2 | geomean 1.289 | min 1.220 | max 1.362 |
| code      sam g8 | n=2 | geomean 1.348 | min 1.260 | max 1.441 |
| code      samw g8 | n=2 | geomean 1.325 | min 1.244 | max 1.410 |
| edit      ngc g4 | n=1 | geomean 2.669 | min 2.669 | max 2.669 |
| edit      pld2 g4 | n=1 | geomean 2.507 | min 2.507 | max 2.507 |
| edit      pld3 g4 | n=1 | geomean 2.776 | min 2.776 | max 2.776 |
| edit      pld3 g8 | n=1 | geomean 3.593 | min 3.593 | max 3.593 |
| edit      sam g4 | n=1 | geomean 2.899 | min 2.899 | max 2.899 |
| edit      sam g8 | n=1 | geomean 3.870 | min 3.870 | max 3.870 |
| edit      samw g8 | n=1 | geomean 3.784 | min 3.784 | max 3.784 |
| extract   ngc g4 | n=1 | geomean 1.688 | min 1.688 | max 1.688 |
| extract   pld2 g4 | n=1 | geomean 1.676 | min 1.676 | max 1.676 |
| extract   pld3 g4 | n=1 | geomean 1.655 | min 1.655 | max 1.655 |
| extract   pld3 g8 | n=1 | geomean 1.198 | min 1.198 | max 1.198 |
| extract   sam g4 | n=1 | geomean 1.689 | min 1.689 | max 1.689 |
| extract   sam g8 | n=1 | geomean 1.191 | min 1.191 | max 1.191 |
| extract   samw g8 | n=1 | geomean 1.130 | min 1.130 | max 1.130 |
| prose     ngc g4 | n=1 | geomean 1.061 | min 1.061 | max 1.061 |
| prose     pld2 g4 | n=1 | geomean 1.067 | min 1.067 | max 1.067 |
| prose     pld3 g4 | n=1 | geomean 1.072 | min 1.072 | max 1.072 |
| prose     pld3 g8 | n=1 | geomean 1.079 | min 1.079 | max 1.079 |
| prose     sam g4 | n=1 | geomean 1.069 | min 1.069 | max 1.069 |
| prose     sam g8 | n=1 | geomean 1.070 | min 1.070 | max 1.070 |
| prose     samw g8 | n=1 | geomean 1.072 | min 1.072 | max 1.072 |
| summarize ngc g4 | n=1 | geomean 1.163 | min 1.163 | max 1.163 |
| summarize pld2 g4 | n=1 | geomean 1.189 | min 1.189 | max 1.189 |
| summarize pld3 g4 | n=1 | geomean 1.130 | min 1.130 | max 1.130 |
| summarize pld3 g8 | n=1 | geomean 1.126 | min 1.126 | max 1.126 |
| summarize sam g4 | n=1 | geomean 1.129 | min 1.129 | max 1.129 |
| summarize sam g8 | n=1 | geomean 1.128 | min 1.128 | max 1.128 |
| summarize samw g8 | n=1 | geomean 1.143 | min 1.143 | max 1.143 |
| terminal  ngc g4 | n=1 | geomean 1.005 | min 1.005 | max 1.005 |
| terminal  pld2 g4 | n=1 | geomean 1.016 | min 1.016 | max 1.016 |
| terminal  pld3 g4 | n=1 | geomean 1.027 | min 1.027 | max 1.027 |
| terminal  pld3 g8 | n=1 | geomean 1.024 | min 1.024 | max 1.024 |
| terminal  sam g4 | n=1 | geomean 1.021 | min 1.021 | max 1.021 |
| terminal  sam g8 | n=1 | geomean 1.016 | min 1.016 | max 1.016 |
| terminal  samw g8 | n=1 | geomean 1.017 | min 1.017 | max 1.017 |

identical to plain: 41/49; first_div values: [61, 124, 189]
mean accepted/verify (rounds with a draft), tok/round, draft_hit_rate by config:
| ngc g4 | acc/verify 1.60 | tok/round 1.67 | hit 0.32 | sim-vs-live 1.378/1.352 |
| pld2 g4 | acc/verify 1.53 | tok/round 1.63 | hit 0.33 | sim-vs-live 1.357/1.360 |
| pld3 g4 | acc/verify 2.67 | tok/round 1.63 | hit 0.20 | sim-vs-live 1.392/1.381 |
| pld3 g8 | acc/verify 3.99 | tok/round 1.76 | hit 0.14 | sim-vs-live 1.476/1.388 |
| sam g4 | acc/verify 2.78 | tok/round 1.66 | hit 0.19 | sim-vs-live 1.409/1.390 |
| sam g8 | acc/verify 4.15 | tok/round 1.82 | hit 0.14 | sim-vs-live 1.496/1.395 |
| samw g8 | acc/verify 3.53 | tok/round 1.78 | hit 0.15 | sim-vs-live 1.487/1.376 |

### LIVE natural: wall speedup by config (all tasks)
| ngc g4 | n=21 | geomean 1.196 | min 1.000 | max 1.789 |
| pld2 g4 | n=21 | geomean 1.192 | min 0.995 | max 1.745 |
| pld3 g4 | n=21 | geomean 1.157 | min 0.992 | max 1.824 |
| pld3 g8 | n=21 | geomean 1.171 | min 0.990 | max 2.024 |
| sam g4 | n=21 | geomean 1.159 | min 0.995 | max 1.814 |
| sam g8 | n=21 | geomean 1.174 | min 0.990 | max 2.053 |
| samw g8 | n=21 | geomean 1.174 | min 0.998 | max 2.004 |

### LIVE natural: by kind x config
| code      ngc g4 | n=7 | geomean 1.193 | min 1.000 | max 1.369 |
| code      pld2 g4 | n=7 | geomean 1.188 | min 0.995 | max 1.339 |
| code      pld3 g4 | n=7 | geomean 1.157 | min 0.992 | max 1.317 |
| code      pld3 g8 | n=7 | geomean 1.174 | min 0.990 | max 1.383 |
| code      sam g4 | n=7 | geomean 1.160 | min 0.995 | max 1.317 |
| code      sam g8 | n=7 | geomean 1.176 | min 0.990 | max 1.374 |
| code      samw g8 | n=7 | geomean 1.170 | min 0.998 | max 1.363 |
| edit      ngc g4 | n=2 | geomean 1.189 | min 1.165 | max 1.213 |
| edit      pld2 g4 | n=2 | geomean 1.190 | min 1.166 | max 1.215 |
| edit      pld3 g4 | n=2 | geomean 1.138 | min 1.121 | max 1.155 |
| edit      pld3 g8 | n=2 | geomean 1.139 | min 1.118 | max 1.160 |
| edit      sam g4 | n=2 | geomean 1.140 | min 1.124 | max 1.157 |
| edit      sam g8 | n=2 | geomean 1.143 | min 1.127 | max 1.160 |
| edit      samw g8 | n=2 | geomean 1.161 | min 1.146 | max 1.177 |
| extract   ngc g4 | n=2 | geomean 1.441 | min 1.161 | max 1.789 |
| extract   pld2 g4 | n=2 | geomean 1.431 | min 1.173 | max 1.745 |
| extract   pld3 g4 | n=2 | geomean 1.462 | min 1.172 | max 1.824 |
| extract   pld3 g8 | n=2 | geomean 1.556 | min 1.196 | max 2.024 |
| extract   sam g4 | n=2 | geomean 1.455 | min 1.167 | max 1.814 |
| extract   sam g8 | n=2 | geomean 1.565 | min 1.193 | max 2.053 |
| extract   samw g8 | n=2 | geomean 1.545 | min 1.191 | max 2.004 |
| prose     ngc g4 | n=4 | geomean 1.161 | min 1.019 | max 1.304 |
| prose     pld2 g4 | n=4 | geomean 1.161 | min 1.020 | max 1.294 |
| prose     pld3 g4 | n=4 | geomean 1.114 | min 1.015 | max 1.201 |
| prose     pld3 g8 | n=4 | geomean 1.112 | min 1.011 | max 1.200 |
| prose     sam g4 | n=4 | geomean 1.116 | min 1.015 | max 1.198 |
| prose     sam g8 | n=4 | geomean 1.113 | min 1.010 | max 1.200 |
| prose     samw g8 | n=4 | geomean 1.118 | min 1.010 | max 1.218 |
| summarize ngc g4 | n=2 | geomean 1.256 | min 1.167 | max 1.351 |
| summarize pld2 g4 | n=2 | geomean 1.222 | min 1.170 | max 1.278 |
| summarize pld3 g4 | n=2 | geomean 1.183 | min 1.098 | max 1.276 |
| summarize pld3 g8 | n=2 | geomean 1.198 | min 1.091 | max 1.316 |
| summarize sam g4 | n=2 | geomean 1.195 | min 1.099 | max 1.299 |
| summarize sam g8 | n=2 | geomean 1.201 | min 1.091 | max 1.323 |
| summarize samw g8 | n=2 | geomean 1.212 | min 1.101 | max 1.333 |
| terminal  ngc g4 | n=4 | geomean 1.103 | min 1.049 | max 1.196 |
| terminal  pld2 g4 | n=4 | geomean 1.108 | min 1.050 | max 1.223 |
| terminal  pld3 g4 | n=4 | geomean 1.064 | min 1.034 | max 1.114 |
| terminal  pld3 g8 | n=4 | geomean 1.069 | min 1.040 | max 1.123 |
| terminal  sam g4 | n=4 | geomean 1.067 | min 1.036 | max 1.121 |
| terminal  sam g8 | n=4 | geomean 1.070 | min 1.039 | max 1.127 |
| terminal  samw g8 | n=4 | geomean 1.067 | min 1.041 | max 1.132 |

identical to plain: 121/147; first_div values: [60, 116, 134, 152, 185]
mean accepted/verify (rounds with a draft), tok/round, draft_hit_rate by config:
| ngc g4 | acc/verify 1.27 | tok/round 1.28 | hit 0.21 | sim-vs-live 1.194/1.196 |
| pld2 g4 | acc/verify 1.24 | tok/round 1.28 | hit 0.21 | sim-vs-live 1.190/1.192 |
| pld3 g4 | acc/verify 1.71 | tok/round 1.21 | hit 0.11 | sim-vs-live 1.162/1.157 |
| pld3 g8 | acc/verify 2.29 | tok/round 1.24 | hit 0.09 | sim-vs-live 1.172/1.171 |
| sam g4 | acc/verify 1.72 | tok/round 1.21 | hit 0.11 | sim-vs-live 1.163/1.159 |
| sam g8 | acc/verify 2.35 | tok/round 1.24 | hit 0.09 | sim-vs-live 1.173/1.174 |
| samw g8 | acc/verify 1.90 | tok/round 1.22 | hit 0.10 | sim-vs-live 1.175/1.174 |

### SIM ctx2048: speedup by proposer x gamma
| ngc g12 | n=7 | geomean 1.473 | min 1.047 | max 4.224 |
| ngc g2 | n=7 | geomean 1.297 | min 1.048 | max 2.096 |
| ngc g3 | n=7 | geomean 1.369 | min 1.059 | max 2.479 |
| ngc g4 | n=7 | geomean 1.408 | min 1.060 | max 2.892 |
| ngc g6 | n=7 | geomean 1.428 | min 1.056 | max 3.279 |
| ngc g8 | n=7 | geomean 1.441 | min 1.049 | max 3.480 |
| ngc2 g12 | n=7 | geomean 1.087 | min 0.982 | max 1.312 |
| ngc2 g2 | n=7 | geomean 1.073 | min 0.990 | max 1.227 |
| ngc2 g3 | n=7 | geomean 1.085 | min 0.990 | max 1.275 |
| ngc2 g4 | n=7 | geomean 1.084 | min 0.988 | max 1.267 |
| ngc2 g6 | n=7 | geomean 1.087 | min 0.985 | max 1.288 |
| ngc2 g8 | n=7 | geomean 1.089 | min 0.983 | max 1.295 |
| pld2 g12 | n=7 | geomean 1.469 | min 1.048 | max 3.816 |
| pld2 g2 | n=7 | geomean 1.283 | min 1.048 | max 1.942 |
| pld2 g3 | n=7 | geomean 1.364 | min 1.059 | max 2.402 |
| pld2 g4 | n=7 | geomean 1.392 | min 1.060 | max 2.634 |
| pld2 g6 | n=7 | geomean 1.427 | min 1.056 | max 3.056 |
| pld2 g8 | n=7 | geomean 1.459 | min 1.049 | max 3.587 |
| pld3 g12 | n=7 | geomean 1.486 | min 1.057 | max 4.343 |
| pld3 g2 | n=7 | geomean 1.268 | min 1.058 | max 2.052 |
| pld3 g3 | n=7 | geomean 1.333 | min 1.071 | max 2.455 |
| pld3 g4 | n=7 | geomean 1.377 | min 1.067 | max 2.919 |
| pld3 g6 | n=7 | geomean 1.423 | min 1.073 | max 3.411 |
| pld3 g8 | n=7 | geomean 1.450 | min 1.066 | max 3.753 |
| pld3b g12 | n=7 | geomean 1.496 | min 1.048 | max 4.381 |
| pld3b g2 | n=7 | geomean 1.300 | min 1.048 | max 2.096 |
| pld3b g3 | n=7 | geomean 1.371 | min 1.059 | max 2.479 |
| pld3b g4 | n=7 | geomean 1.413 | min 1.060 | max 2.892 |
| pld3b g6 | n=7 | geomean 1.450 | min 1.056 | max 3.447 |
| pld3b g8 | n=7 | geomean 1.477 | min 1.049 | max 3.906 |
| pld4 g12 | n=7 | geomean 1.465 | min 1.051 | max 4.306 |
| pld4 g2 | n=7 | geomean 1.253 | min 1.049 | max 2.088 |
| pld4 g3 | n=7 | geomean 1.307 | min 1.048 | max 2.470 |
| pld4 g4 | n=7 | geomean 1.354 | min 1.045 | max 2.888 |
| pld4 g6 | n=7 | geomean 1.400 | min 1.056 | max 3.377 |
| pld4 g8 | n=7 | geomean 1.427 | min 1.054 | max 3.717 |
| sam g12 | n=7 | geomean 1.511 | min 1.057 | max 4.881 |
| sam g2 | n=7 | geomean 1.282 | min 1.058 | max 2.193 |
| sam g3 | n=7 | geomean 1.354 | min 1.071 | max 2.712 |
| sam g4 | n=7 | geomean 1.389 | min 1.067 | max 3.103 |
| sam g6 | n=7 | geomean 1.439 | min 1.073 | max 3.691 |
| sam g8 | n=7 | geomean 1.469 | min 1.066 | max 4.117 |
| sam2 g12 | n=7 | geomean 1.557 | min 1.072 | max 5.035 |
| sam2 g2 | n=7 | geomean 1.315 | min 1.048 | max 2.244 |
| sam2 g3 | n=7 | geomean 1.383 | min 1.054 | max 2.744 |
| sam2 g4 | n=7 | geomean 1.423 | min 1.060 | max 3.148 |
| sam2 g6 | n=7 | geomean 1.469 | min 1.065 | max 3.662 |
| sam2 g8 | n=7 | geomean 1.512 | min 1.073 | max 4.215 |
| samw g12 | n=7 | geomean 1.503 | min 1.071 | max 4.784 |
| samw g2 | n=7 | geomean 1.282 | min 1.058 | max 2.193 |
| samw g3 | n=7 | geomean 1.344 | min 1.072 | max 2.668 |
| samw g4 | n=7 | geomean 1.383 | min 1.070 | max 3.048 |
| samw g6 | n=7 | geomean 1.431 | min 1.074 | max 3.622 |
| samw g8 | n=7 | geomean 1.460 | min 1.073 | max 4.038 |

### SIM ctx8192: speedup by proposer x gamma
| ngc g12 | n=7 | geomean 1.417 | min 0.964 | max 3.527 |
| ngc g2 | n=7 | geomean 1.285 | min 1.024 | max 2.037 |
| ngc g3 | n=7 | geomean 1.336 | min 1.017 | max 2.331 |
| ngc g4 | n=7 | geomean 1.378 | min 1.006 | max 2.768 |
| ngc g6 | n=7 | geomean 1.394 | min 0.995 | max 2.971 |
| ngc g8 | n=7 | geomean 1.400 | min 0.983 | max 3.105 |
| ngc2 g12 | n=7 | geomean 1.051 | min 0.996 | max 1.246 |
| ngc2 g2 | n=7 | geomean 1.057 | min 0.999 | max 1.239 |
| ngc2 g3 | n=7 | geomean 1.064 | min 0.999 | max 1.279 |
| ngc2 g4 | n=7 | geomean 1.061 | min 0.998 | max 1.261 |
| ngc2 g6 | n=7 | geomean 1.058 | min 0.997 | max 1.251 |
| ngc2 g8 | n=7 | geomean 1.056 | min 0.996 | max 1.268 |
| pld2 g12 | n=7 | geomean 1.437 | min 0.981 | max 3.642 |
| pld2 g2 | n=7 | geomean 1.271 | min 1.031 | max 1.938 |
| pld2 g3 | n=7 | geomean 1.341 | min 1.024 | max 2.367 |
| pld2 g4 | n=7 | geomean 1.357 | min 1.013 | max 2.569 |
| pld2 g6 | n=7 | geomean 1.396 | min 1.005 | max 2.971 |
| pld2 g8 | n=7 | geomean 1.426 | min 0.995 | max 3.469 |
| pld3 g12 | n=7 | geomean 1.519 | min 1.021 | max 4.167 |
| pld3 g2 | n=7 | geomean 1.283 | min 1.025 | max 2.047 |
| pld3 g3 | n=7 | geomean 1.344 | min 1.024 | max 2.420 |
| pld3 g4 | n=7 | geomean 1.392 | min 1.028 | max 2.851 |
| pld3 g6 | n=7 | geomean 1.444 | min 1.025 | max 3.323 |
| pld3 g8 | n=7 | geomean 1.476 | min 1.023 | max 3.636 |
| pld3b g12 | n=7 | geomean 1.466 | min 0.981 | max 4.192 |
| pld3b g2 | n=7 | geomean 1.291 | min 1.031 | max 2.091 |
| pld3b g3 | n=7 | geomean 1.351 | min 1.024 | max 2.443 |
| pld3b g4 | n=7 | geomean 1.386 | min 1.013 | max 2.822 |
| pld3b g6 | n=7 | geomean 1.426 | min 1.005 | max 3.353 |
| pld3b g8 | n=7 | geomean 1.446 | min 0.995 | max 3.779 |
| pld4 g12 | n=7 | geomean 1.501 | min 1.002 | max 4.142 |
| pld4 g2 | n=7 | geomean 1.270 | min 1.005 | max 2.083 |
| pld4 g3 | n=7 | geomean 1.333 | min 1.010 | max 2.436 |
| pld4 g4 | n=7 | geomean 1.377 | min 1.009 | max 2.824 |
| pld4 g6 | n=7 | geomean 1.423 | min 1.006 | max 3.293 |
| pld4 g8 | n=7 | geomean 1.458 | min 1.004 | max 3.607 |
| sam g12 | n=7 | geomean 1.545 | min 1.021 | max 4.694 |
| sam g2 | n=7 | geomean 1.296 | min 1.025 | max 2.188 |
| sam g3 | n=7 | geomean 1.363 | min 1.024 | max 2.675 |
| sam g4 | n=7 | geomean 1.409 | min 1.028 | max 3.032 |
| sam g6 | n=7 | geomean 1.461 | min 1.025 | max 3.598 |
| sam g8 | n=7 | geomean 1.496 | min 1.023 | max 3.993 |
| sam2 g12 | n=7 | geomean 1.567 | min 1.028 | max 4.849 |
| sam2 g2 | n=7 | geomean 1.307 | min 1.031 | max 2.239 |
| sam2 g3 | n=7 | geomean 1.375 | min 1.030 | max 2.707 |
| sam2 g4 | n=7 | geomean 1.416 | min 1.030 | max 3.077 |
| sam2 g6 | n=7 | geomean 1.468 | min 1.029 | max 3.570 |
| sam2 g8 | n=7 | geomean 1.514 | min 1.029 | max 4.091 |
| samw g12 | n=7 | geomean 1.538 | min 1.022 | max 4.616 |
| samw g2 | n=7 | geomean 1.296 | min 1.025 | max 2.188 |
| samw g3 | n=7 | geomean 1.357 | min 1.025 | max 2.633 |
| samw g4 | n=7 | geomean 1.395 | min 1.025 | max 2.981 |
| samw g6 | n=7 | geomean 1.457 | min 1.024 | max 3.536 |
| samw g8 | n=7 | geomean 1.487 | min 1.023 | max 3.924 |

### SIM natural: speedup by proposer x gamma
| ngc g12 | n=21 | geomean 1.179 | min 0.958 | max 2.080 |
| ngc g2 | n=21 | geomean 1.168 | min 1.010 | max 1.545 |
| ngc g3 | n=21 | geomean 1.188 | min 1.005 | max 1.689 |
| ngc g4 | n=21 | geomean 1.194 | min 0.996 | max 1.795 |
| ngc g6 | n=21 | geomean 1.195 | min 0.983 | max 1.885 |
| ngc g8 | n=21 | geomean 1.186 | min 0.972 | max 1.961 |
| ngc2 g12 | n=21 | geomean 1.040 | min 0.997 | max 1.136 |
| ngc2 g2 | n=21 | geomean 1.036 | min 0.999 | max 1.100 |
| ngc2 g3 | n=21 | geomean 1.039 | min 0.999 | max 1.118 |
| ngc2 g4 | n=21 | geomean 1.041 | min 0.999 | max 1.123 |
| ngc2 g6 | n=21 | geomean 1.042 | min 0.999 | max 1.135 |
| ngc2 g8 | n=21 | geomean 1.041 | min 0.999 | max 1.131 |
| pld2 g12 | n=21 | geomean 1.188 | min 0.952 | max 2.157 |
| pld2 g2 | n=21 | geomean 1.164 | min 1.003 | max 1.501 |
| pld2 g3 | n=21 | geomean 1.186 | min 0.998 | max 1.670 |
| pld2 g4 | n=21 | geomean 1.190 | min 0.990 | max 1.752 |
| pld2 g6 | n=21 | geomean 1.194 | min 0.976 | max 1.859 |
| pld2 g8 | n=21 | geomean 1.191 | min 0.966 | max 1.992 |
| pld3 g12 | n=21 | geomean 1.175 | min 0.985 | max 2.171 |
| pld3 g2 | n=21 | geomean 1.133 | min 0.999 | max 1.561 |
| pld3 g3 | n=21 | geomean 1.151 | min 0.997 | max 1.693 |
| pld3 g4 | n=21 | geomean 1.162 | min 0.995 | max 1.828 |
| pld3 g6 | n=21 | geomean 1.169 | min 0.992 | max 1.935 |
| pld3 g8 | n=21 | geomean 1.172 | min 0.989 | max 2.025 |
| pld3b g12 | n=21 | geomean 1.189 | min 0.952 | max 2.157 |
| pld3b g2 | n=21 | geomean 1.170 | min 1.003 | max 1.545 |
| pld3b g3 | n=21 | geomean 1.188 | min 0.998 | max 1.670 |
| pld3b g4 | n=21 | geomean 1.196 | min 0.990 | max 1.818 |
| pld3b g6 | n=21 | geomean 1.199 | min 0.976 | max 1.968 |
| pld3b g8 | n=21 | geomean 1.194 | min 0.966 | max 1.992 |
| pld4 g12 | n=21 | geomean 1.157 | min 0.992 | max 2.123 |
| pld4 g2 | n=21 | geomean 1.108 | min 0.999 | max 1.567 |
| pld4 g3 | n=21 | geomean 1.129 | min 0.999 | max 1.719 |
| pld4 g4 | n=21 | geomean 1.140 | min 0.998 | max 1.816 |
| pld4 g6 | n=21 | geomean 1.148 | min 0.998 | max 1.924 |
| pld4 g8 | n=21 | geomean 1.153 | min 0.997 | max 2.015 |
| sam g12 | n=21 | geomean 1.176 | min 0.985 | max 2.171 |
| sam g2 | n=21 | geomean 1.134 | min 0.999 | max 1.593 |
| sam g3 | n=21 | geomean 1.154 | min 0.997 | max 1.751 |
| sam g4 | n=21 | geomean 1.163 | min 0.995 | max 1.828 |
| sam g6 | n=21 | geomean 1.170 | min 0.992 | max 1.963 |
| sam g8 | n=21 | geomean 1.173 | min 0.989 | max 2.058 |
| sam2 g12 | n=21 | geomean 1.225 | min 1.003 | max 2.140 |
| sam2 g2 | n=21 | geomean 1.172 | min 1.003 | max 1.576 |
| sam2 g3 | n=21 | geomean 1.192 | min 1.003 | max 1.731 |
| sam2 g4 | n=21 | geomean 1.201 | min 1.003 | max 1.829 |
| sam2 g6 | n=21 | geomean 1.213 | min 1.003 | max 1.938 |
| sam2 g8 | n=21 | geomean 1.219 | min 1.003 | max 2.030 |
| samw g12 | n=21 | geomean 1.178 | min 0.999 | max 2.113 |
| samw g2 | n=21 | geomean 1.134 | min 0.999 | max 1.593 |
| samw g3 | n=21 | geomean 1.147 | min 0.999 | max 1.713 |
| samw g4 | n=21 | geomean 1.157 | min 0.999 | max 1.809 |
| samw g6 | n=21 | geomean 1.168 | min 0.999 | max 1.943 |
| samw g8 | n=21 | geomean 1.175 | min 0.999 | max 2.006 |
