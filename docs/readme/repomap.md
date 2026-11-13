| Path | What is in it |
|---|---|
| `harness/` | Serving engine (`serve.py`), continuous batching (`cbatch.py`), Triton executors (`fast_north.py`, `fast_kernels*.py`; exact v2 = `FastNorth.set_x2(**fast_kernels_x2.recommended(Lmax))`), batched exact executor, PLD / suffix proposers, benchmarks |
| `modal/` | Modal entry points for every GPU experiment |
| `src/` | Subprojects: `northstar` reference model, `kvm` fused-MoE kernels, `approx`, `opt`, `selfedit`, `adapt_sched`, `async_spec`, `mk_selfspec` |
| `tests/` | CPU unit tests for the core package |
| `results/` | Raw JSONL / JSON results and plots, one file per run |
| `logs/` | Client-side run logs |
| `docs/` | Design notes: [`THESIS.md`](docs/THESIS.md), [`PLAN.md`](docs/PLAN.md), [`NORTH_ARCH.md`](docs/NORTH_ARCH.md), self-improvement and kernel notes |
| `RESULTS.md` | Every experiment (H1–H28): setup, tables, findings, caveats |
| `NEXT_STEPS.md` | Current status and the next experiments |
