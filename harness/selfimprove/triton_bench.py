"""Speed command, track 'triton': CUDA-graph replay timing (CUDA events) of per-kernel microbenchmarks and full synthetic T=1 / T=B steps on
the DEV or the disjoint HELD-OUT shape set (NS_SPLIT). time_ms = sum of per-case ms. ref_ms = the same replays timed with an INDEPENDENT host clock
(perf_counter around the replay batch incl. a device sync); the evaluator's TimerSpec cross-checks time_ms against it."""
import common
import triton_suite as TS

r = TS.run_bench(common.SPLIT)
common.emit({"time_ms": r["time_ms"], "ref_ms": r["ref_ms"], "n_iters": r["n_iters"], "harness_wall_ms": r["harness_wall_ms"],
             "split": r["split"], "percase": {k: round(v, 5) for k, v in r["percase"].items()}})
