# Phase 6: North-authored self-improvement loop (`src/selfimprove/`)

Rule: only North (`CohereLabs/North-Mini-Code-1.0`, rev `d11e61a8...`, served on Modal) may author optimization proposals.
`MockProposer` exists for unit tests only; its output is labeled `MOCK - NOT NORTH-AUTHORED` and is never counted as North's.

## Flow (`loop.Loop.step`)
1. **Propose** (`proposer.NorthProposer` -> Modal `NorthProposerService.propose`): prompt = scope description, editable files, profile
   summary, dev-split results, one-line history. Response must attest model id + revision or the step ends `proposer_error`.
   `extract_diff` deterministically takes the first ```diff fence (after stripping `<think>`).
2. **Parse/apply** (`diffs.py`): strict unified-diff parser, exact context (no fuzz); rejects traversal, absolute paths, renames, deletions,
   mode/binary/symlink ops, hidden/bidi chars, oversize diffs, files outside the allowlist, lines outside `NS-EDITABLE-BEGIN/END`
   regions, altered markers. Files in `regions_only` rules without markers are uneditable (fail closed).
3. **Static audit** (`audit.py`): `reject` (timing/clock APIs, env/process/file I/O, harness references, result forging, exit/abort,
   grader-directed text, conditionals on dev-only shape constants or policy-table keys) or `flag` (output caches, tolerance edits,
   precision downgrades, removed guards, zeroing/skipping work). Flags that pass all gates end `needs_review`, never auto-promoted.
4. **Sandboxed evaluation** (`sandbox.py`, `loop.Evaluator`): baseline and candidate each run in a throw-away copy of the repo subset;
   the evaluator (not the candidate) writes the diffed files. sha256 manifest of the whole tree is verified after EVERY command: any
   change to a protected file, any new file, any rewrite of an editable file is `sandbox_violation`. Real-repo protected files are
   hashed before/after. Subprocess: new session, rlimits (CPU, file size), scrubbed env (no API keys), hard timeout + process-group kill,
   stdin closed. Result lines must carry a per-run nonce (`NS_NONCE`), so a forged JSON line printed by candidate code is ignored.
5. **Gates** (`gates.py`, thresholds in `EvalConfig`, evaluator-owned): short-circuit order correctness, quality, speed.
   - correctness: every case finite, `rel_l2 <= 1e-3` (default, stated), >= 8 distinct cases, not > 4x baseline's own error; fresh
     unpredictable `NS_SEED` per evaluation, so the candidate cannot memorise inputs.
   - quality: paired bootstrap non-inferiority on held-out items (lower 95% bound of mean delta >= -margin, >= 20 items).
   - speed: interleaved A/B runs (order alternated), `n_runs`=9 per split, ratio of medians with bootstrap CI; must clear
     `1 + max(min_gain 3%, 2 x robust relative noise)` on dev with CI lower bound, must not regress on held-out shapes and must keep
     >= 25% of the dev gain there (catches shape specialization). Zero/non-finite timings invalid.
6. **Promote** (`store.py`): `versions/vNNNN/{files,meta.json}`, atomic `CURRENT` pointer, `rollback(to=None)` (versions never deleted).
   `Store.attribution()` splits lineage into developer / North (attested) / mock; `reportable_as_north_authored` is false if any
   mock version is in the lineage. `promote` refuses a `north` label without attestation.
7. **Log** (`proposals.jsonl`, fsync, hash-chained, `Store.verify_log()`): every proposal incl. failures: prompt, raw response + sha,
   diff + sha, audit findings, gate data, seeds, sandbox info, verdict, promoted version, cost (proposer usd/GPU s/tokens, eval wall s
   and usd estimate). `Loop.run(max_total_usd=...)` stops on budget.

Verdicts: `promoted`, `needs_review`, `pass_mock_not_promoted`, `no_change`, `no_diff_in_response`, `invalid_diff`, `audit_reject`,
`build_fail`, `build_timeout`, `baseline_broken`, `timeout`, `sandbox_violation`, `gate_fail:{correctness,quality,speed}`, `proposer_error`.

## Interface
```python
cfg = EvalConfig(repo, subdirs, correctness_cmd, quality_cmd, bench_cmd, build_cmd=None, scope=DEFAULT_SCOPE, ...)
loop = Loop(cfg, Store("results/selfimprove"), NorthProposer())
loop.run(rounds=3, max_total_usd=5.0, profile_summary="...", dev_results="...")
Store.attribution(); Store.cost_summary(); Store.rollback(); Store.verify_log()
```
Harness command protocol (commands run with cwd = workspace root; env `NS_WORKSPACE`, `NS_NONCE`, `NS_SEED`, `NS_SPLIT`=dev|heldout;
last stdout JSON line must contain `"nonce": os.environ.pop("NS_NONCE")` read BEFORE any candidate code is imported):
- correctness: `{"cases":[{"name","rel_l2","max_abs","finite"}...]}`; quality: `{"items":[score...]}` (same order every run); bench: `{"time_ms": x}`.
CLI: `PYTHONPATH=src python -m selfimprove.cli {run,rollback,attribution,verify-log,cost} --store DIR [--config cfg.py]`.

## Running it live on Modal (not done in this task; no GPU was launched)
Needs, in order:
1. `modal run modal/download.py` already populated Volume `northstar-weights` (the proposer reuses it and `modal/common.py`; it fails
   loudly if the weights are absent rather than downloading anything).
2. `modal deploy src/selfimprove/modal_north.py` (app `northstar-selfimprove`, class `NorthProposerService`, A100-80GB, ~$2.5/h,
   scaledown 60 s, 1 container). Default backend is HF transformers 5.22.1 (the stack in `modal/`); vLLM is opt-in
   (`NS_BACKEND=vllm`, build with `NS_WITH_VLLM=1 NS_VLLM_SPEC=vllm==<pinned>`) and UNVERIFIED for `cohere2_moe`. Smoke:
   `modal run src/selfimprove/modal_north.py --prompt ...` (confirm spend first).
3. Developer (not North) must add `// NS-EDITABLE-BEGIN` / `// NS-EDITABLE-END` markers around ONLY the tunables in
   `src/kvm/moe_fused.cu` (e.g. `SL`, `KC`; not `H`, `I`, `MAXTOK`, the bench/check `main`), and create `src/opt/policy_tables/*.json`
   if a policy table is wanted. Until then no kernel edit can apply (fail closed).
4. Real harness scripts that follow the protocol above, placed outside the editable scope (e.g. `harness/si_check.py`, `si_quality.py`,
   `si_bench.py`): correctness vs the torch reference in `src/kvm/ref/moe_ref.py` on fresh `NS_SEED` inputs, held-out task scoring
   on data kept out of the proposer's prompt, timing on dev vs held-out shape sets (held-out shapes must differ from the benchmark
   shapes; list them in `EvalConfig.dev_shapes/heldout_shapes` for the audit). `build_cmd` must write only under `build/` (nvcc).
5. A GPU evaluator process: the evaluator is plain Python and must run where nvcc + the GPU are (e.g. a Modal A100 function that
   imports `selfimprove.loop` and calls the deployed proposer via `NorthProposer`). Not yet written; the CPU loop is tested with a toy harness.

## Honest limits
- Process isolation is a subprocess + rlimits + hash checks + nonce, not a container/seccomp. Candidate code that shares a process with
  the scoring code (a Python hook) could read `/proc/self/environ`; for CUDA, compile the candidate to a separate binary and compare
  outputs in a separate harness process. Network is not blocked on macOS/dev; on Modal, run the evaluator in a network-blocked function.
- Static audit is a heuristic first line; the real defenses are fresh seeds, held-out shapes/data and protected-tree hashes.
- Attribution rests on the endpoint attesting its model id/revision; the Modal service is developer infrastructure and is trusted.
- All 95 unit tests use a toy harness and mock/stub proposers. No North-authored result exists yet.

Tests: `cd src && python -m pytest -q selfimprove/tests` (about 30 s, CPU).
