# Phase 7 concurrent adaptation: scheduling study (MODELED, not a hardware result)

No GPU was run. Everything below is a simulation over previously measured step times plus swept, INVENTED knobs. Code: `src/adapt_sched/` (`model.py`, `run.py`, CPU tests: `python -m pytest src/adapt_sched/tests`). Output: `results/adapt_sched_model_SIMULATED.json`. Reproduce: `python src/adapt_sched/run.py` (~30 s, deterministic). Log: `logs/adapt_sched_run.log`.

## Question
docs/PLAN.md Phase 7: serve full North on one GPU while a background job (here: kernel-candidate benchmarking as in `src/selfimprove`) runs. Compare sequential adaptation (pause serving), static sharing, and adaptive sharing (admission control honoring a tail-latency SLO) at equal GPU-time budget.

## Model (short)
- Serving: Poisson requests, continuous batching up to Bmax=8, decode only (prefill not modeled). Step time at batch b is the MEASURED megakernel curve 3.32/4.10/5.29/7.39 ms (b=1/2/4/8). A secondary HF-eager curve (85.8/96.6/116.3 ms at b=1/2/4, ctx 8K, A100) is run as a sensitivity.
- Background: n=6 candidate evaluations of MEASURED 130.5 standalone GPU-s each (mean of `eval_wall_s`, range 120-154, `results/selfimprove_kernel_structured`). Equal GPU-time budget = every policy must complete the same 6 evaluations (783 standalone GPU-s). Anything above 783 in `bg_gpu_s_consumed` is wasted work (extra reps under noise).
- Contention (same fluid style as `docs/ASYNC_SPEC.md`): serving keeps HBM busy for beta_s = 0.60 (MEASURED, megakernel BS1); background at intensity u demands beta_b*u; both slow by `max(1, beta_s + beta_b*u) * (1 + c_sm*u)`.
- Timing-sensitive share of each evaluation (`f_timing`, INVENTED 0.5): timing done while contended is worth 1/`noise_mult` (INVENTED 2) because it needs extra reps for the same CI. Pure idle-gap timing is clean.
- Policies: `sequential` pauses serving for each evaluation (1 candidate per window with 60 s serving gaps, or all 6 at once); `static` runs the benchmark at fixed u in {0.25, 0.5, 1.0} whenever work remains; `adaptive` is AIMD on u from the measured p99 step latency each 1 s, targeting 0.95 x SLO (SLO = 1.25 x unloaded Bmax step = 9.24 ms, INVENTED slack). Two adaptive variants: timing phase only in idle gaps ("controlled window", conservative) or shared like static.
- Memory: separate arithmetic, not part of the simulation.

| Input | Tag |
|---|---|
| Megakernel step ms by batch; beta_s = 0.60; eval wall per candidate; HF step ms; weights 61.0 GB; 79.3 GB card; megakernel `--frac-vram-utilization 0.85`; KV 3.5 GB per 128K sequence | MEASURED / derived from measured (results/, NORTH_ARCH.md) |
| beta_s = 0.45 for graphed PyTorch | INFERRED (async_spec) |
| beta_b (0.1-0.8), c_sm (0-0.3), load rho (0.3-0.85), 200 tok/request, f_timing, noise_mult (1-4), SLO slack (1.1-1.5), BS>8 extrapolation, controller constants, payback gains | INVENTED, swept |

## Results (megakernel curve, nominal beta_b=0.5, rho=0.5, 3000 s horizon, one seed)
| policy | p99 step ms | tokens over SLO (of 1.6M) | longest stall | request p99 | 6 evals done at (s) | background GPU-s |
|---|---|---|---|---|---|---|
| no background | 7.39 | 0 | 7 ms | 1.5 s | n/a | 0 |
| sequential, 1 eval per pause | 7.39 | 40 | 130 s | 462 s | 1143 | 783 |
| sequential, all 6 at once | 7.39 | 4 | 783 s | 770 s | 843 | 783 |
| static u=0.25 | 7.57 | 0 | 8 ms | 1.6 s | not done (5/6) | 955 |
| static u=0.5 | 7.76 | 0 | 8 ms | 1.7 s | 2006 | 1056 |
| static u=1.0 | 8.94 | 0 | 9 ms | 2.0 s | 1338 | 1120 |
| adaptive, timing idle-only | 7.80 | 0 | 9 ms | 1.6 s | not done (4/6) | 597 |
| adaptive, timing shared | 8.46 | 0 | 9 ms | 1.8 s | 1434 | 1100 |

Note the token-weighted p99 of the sequential rows is blind to pauses (a pause touches only the few in-flight tokens), so request p99 and longest stall are the honest columns for them.

At a hostile setting (beta_b=0.8, rho=0.5): static u=1.0 p99 11.4 ms with 558K tokens over SLO (35%); static u<=0.5 stays under SLO; adaptive-shared p99 9.41 ms with 19.6K tokens over SLO (1.2%) and finishes at 1966 s; adaptive idle-only 8.51 ms, 6.8K over, 4/6 done. At rho=0.85 the idle-only variant starves (0 of 6 done in 3000 s: serving is never idle), while static u=1.0 puts the request p99 at 413 s (queue collapse, since serving capacity drops by `s` near saturation).

## Findings (about the model, with its stated assumptions)
1. Sharing is nearly free only while `beta_s + beta_b*u <= 1`. With the measured beta_s = 0.60 that means beta_b <= 0.4 at u=1; there the only cost is the INVENTED SM factor. Whether benchmark kernels are that light is unmeasured.
2. When demand exceeds 1, static full-intensity sharing breaks the SLO and, at high load, collapses queues. Static low-u meets the SLO at every setting tried but needs a hand-picked u and finishes later (u=0.25 did not finish 6 evals in 3000 s at nominal load).
3. Adaptive sharing meets the SLO at the nominal and benign settings and reduces violations at hostile ones (1.2% vs 35% at beta_b=0.8, rho=0.5) while finishing the evals sooner than static u=0.5 at nominal (1434 s vs 2006 s) but later than static u=1.0 (1338 s). It is not SLO-clean in hostile cells: the 1 s AIMD loop overshoots, so the p99 target is missed there. Restricting the timing phase to idle gaps protects benchmark quality but starves under load.
4. Sequential adaptation has zero decode degradation and the smallest total time (843 s all at once), but multi-minute stalls (request p99 460-780 s). It is only acceptable for offline/maintenance windows, not for live serving with a tail-latency SLO.
5. Contended timing wastes budget: at noise_mult=4 the static and adaptive-shared runs spend 1.6-1.7k GPU-s for 783 s of useful work (2x). If contended timings are not just noisier but biased, they cannot be used for the speed gate at all, which pushes toward the idle-only or sequential variant for the timing step. The model does not capture bias.
6. HF-eager executor (launch-bound, beta_s 0.45): same ordering; slowdown at u=1 is ~10% (127.9 vs 116.3 ms), SLO met everywhere.

## Memory accounting (INFERRED from measured constants)
Weights 61.0 GB on a 79.3 GB card. The megakernel runs claim `0.85 x 79.3 = 67.4 GB`, leaving 6.4 GB for KV inside the claim (one 32K-context sequence at ~3.5 GB per 128K is fine; batch 8 at 128K needs 28 GB and does not fit) and about 11.9 GB outside the claim for a co-resident benchmark. Compile, candidate buffers and a second CUDA context must fit in that ~12 GB; their size is not measured. Stable plus candidate versions of anything weight-sized do not fit (a second 61 GB copy is impossible), so candidates must be kernels/policies, not weight variants.

## Payback (Phase 7 gate)
Measured yield: 0 of 8 structured-run candidates passed the speed gate (dev speedup 0.96-0.98). The measured saved inference time is therefore zero and payback is undefined (infinite). Under an INVENTED assumption of one accepted candidate cutting decode time by g, break-even serving horizon for the 783 GPU-s of background work (cost counts background GPU-s only, not serving slowdown) ranges from 2.4 h (g=10%, serving 90% busy) to 72 h (g=1%, 30% busy); g=3%, 60% busy gives 12 h. Claiming a Phase 7 benefit requires a real accepted candidate with a measured gain, not this table.

## Honest conclusion
This is a model. It says that, if a benchmark workload shares HBM the way the fluid model assumes, an adaptive controller can run it alongside serving with small SLO cost, while sequential pausing is unusable for live traffic and fixed sharing needs tuning. It does not say that this holds on an H100. The largest unmodeled risk: the cohere megakernel is persistent and launched with `--num-sms 132`, i.e. it occupies the whole GPU. A second process's kernels may get no SMs (time-slicing between contexts rather than concurrent execution) or only coarse time slices, which would make real behavior look like sequential-with-short-pauses, with step-latency spikes the fluid model cannot show. Also unmodeled: L2 contention, kernel-granularity head-of-line blocking, prefill, and benchmark measurement bias. Given the measured selfimprove yield of 0/8, the expected benefit of the background job itself is currently unsupported; recommendation is to not build Phase 7 until at least one candidate has passed the isolated speed gate, then run the three measurements below.

## Three measurements needed on hardware
1. Decode p99 under a co-running benchmark. On one H100, megakernel serving at BS 1/4/8 with the real selfimprove benchmark process running in a second process (and again via CUDA MPS and with CUDA streams if the engine allows), at benchmark intensities off/25/50/100%. Record per-step latency distribution (p50/p99/max), whether the benchmark's kernels run concurrently or only in time slices, and the step-time stall length. This replaces beta_b, c_sm and the contention formula.
2. Memory accounting. Measured `nvidia-smi`/`torch.cuda.mem_get_info` at: megakernel idle, serving at batch x context grid, benchmark compile, benchmark run; confirm the real free memory outside the 0.85 claim and whether an OOM in the benchmark can disturb the serving process.
3. Benchmark noise under contention. Same candidate and baseline kernel timed 30+ times in (a) exclusive, (b) serving idle, (c) serving at rho 0.3/0.7: report the CI width, the speedup bias against exclusive, and the false-accept/false-reject rate of the existing speed gate. This replaces `noise_mult` and `f_timing` and decides whether timing may ever run concurrently.
