"""Correctness command, track 'triton': launch parameters from src/tritonk/tuning.py vs torch references on FRESH inputs derived from NS_SEED,
plus BITWISE row-invariance of the exact kernels (T=1 vs block rows). Output {"cases":[{name, rel_l2, max_abs, finite}]}; bitwise cases are 0.0 or 1.0."""
import common
import triton_suite as TS

common.emit({"cases": TS.run_check(common.SEED)})
