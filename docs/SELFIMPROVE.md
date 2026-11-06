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

## Running it live on Modal (developer infrastructure now in place; NO live run has happened)
Done (developer-authored, outside North's reach):
- `src/kvm/moe_fused.cu`: `NS-EDITABLE-BEGIN/END` around `KC, XPAD, HPAD, MT_SPLIT, MINB_PER_SM, UNROLL_KS` only (defaults identical to the old
  literals: 256, 8, 8, 16, launch bounds default, full unroll=16; MINB cap 8, UNROLL_KS 1..128). `H, I, SL, MAXTOK`, kernel body, host code and a protected block of `static_assert`s
  (tiling, <=48KB smem, ranges) stay outside, so a bad value fails the build. Added bench flags `--shapes tv,td,r;...` and `--routing k`
  (defaults unchanged). NOT compiled here (no nvcc on the dev box): the first nvcc build happens on Modal; fix compile errors there.
- `src/opt/policy_tables/spec_space.json` (gammas, modes, n_groups = the Phase-3 search space; baseline = current defaults).
- `harness/selfimprove/` (protected): `kernel_build.py` (nvcc sm_80 -> `build/moe_fused`), `kernel_check.py` (binary vs `moe_ref.py`, 8 cases x
  fused/two-pass, inputs from fresh `NS_SEED`), `kernel_quality.py` (held-out routing traces, split `calib` of `results/routing_traces.json`,
  35 items, item = 1 - rel_l2), `kernel_bench.py` (dev shapes vs disjoint held-out shapes, binary's own CUDA-event medians, rejects timing if
  fused != two-pass), `policy_eval.py check|quality|bench` (policy track: dev metric = Phase-3 PREDICTED tok/s; held-out = 24 unseen buckets),
  `config.py` (`make_config(repo, track)`, prompt text that mentions dev material only, labeled `mock_diff` for dry runs).
- `modal/selfimprove_eval.py`: `evaluate_candidate` (A100-80GB, `block_network=True`, nvcc image, runs the Loop's own `Evaluator`),
  `RemoteEvaluator` (drop-in; crash/timeouts become verdict `evaluator_error`, never promoted) and the `modal run` launcher.

Launch (confirm cost first; est. in `estimate_cost`: kernel track ~USD 1.1 for round 1 incl. proposer cold start, ~USD 0.9 per later round; dry run ~USD 0.7):
```
modal deploy src/selfimprove/modal_north.py          # proposer (once; weights already in Volume northstar-weights)
modal run modal/selfimprove_eval.py::main --track kernel --proposer mock  --rounds 1                      # DRY RUN, mock, store results/selfimprove_dryrun_kernel
modal run modal/selfimprove_eval.py::main --track kernel --proposer north --rounds 1 --confirm-cost --max-usd 3   # LIVE round 1
```
Live runs write `results/selfimprove_kernel/`; mock runs refuse any store whose name lacks `dryrun` and a store with a mixed lineage.
Policy track (`--track policy`) refuses a live run unless `results/accept_calibration.jsonl` exists and every row is `provenance: measured`; it is built from `results/spec.jsonl` by
`scripts/build_accept_calibration.py` (A100, 15 rows). Policy defaults to `--mode structured` (`structured_policy.py`). Its metric is a PREDICTED tok/s: a gate pass is NOT a measured speedup.
See `docs/SELFIMPROVE_POLICY_RUNBOOK.md`. Policy `check` gate = solver self-consistency only.
Tests: `NS_SLOW=1` also runs the full CPU dry-run loop on the policy track and the fake-binary quality harness.

## Honest limits
- Process isolation is a subprocess + rlimits + hash checks + nonce, not a container/seccomp. Candidate code that shares a process with
  the scoring code (a Python hook) could read `/proc/self/environ`; for CUDA, compile the candidate to a separate binary and compare
  outputs in a separate harness process. Network is not blocked on macOS/dev; on Modal, run the evaluator in a network-blocked function.
- Static audit is a heuristic first line; the real defenses are fresh seeds, held-out shapes/data and protected-tree hashes.
- Attribution rests on the endpoint attesting its model id/revision; the Modal service is developer infrastructure and is trusted.
- 95 original unit tests use a toy harness; the new ones exercise the real harness with a python stand-in for the CUDA binary (no GPU). No North-authored result exists yet, and no nvcc build/GPU timing of the harness has been run.
- Held-out data lives inside the sandbox copy (hash-protected, readable): the kernel candidate is compiled code whose file I/O the audit rejects, and the policy candidate is a JSON data file parsed with a strict whitelist; neither is executed as Python.

Tests: `cd src && python -m pytest -q selfimprove/tests` (about 30 s, CPU).

## Structured-edit proposal mode (kernel tunables track; default)

Four live diff-mode rounds (results/selfimprove_kernel*) all died in the strict diff parser: North named the right intent but got
hunk line counts wrong, emitted whitespace-only edits, or edited a protected line (MAXTOK). In `--mode structured` North outputs ONLY

    {"edits":[{"name":"KC|XPAD|HPAD|MT_SPLIT|MINB_PER_SM|UNROLL_KS","value":<int>}], "rationale":"..."}

(at most 6 edits, each name once, ints within the ranges in `src/selfimprove/structured.py`, which mirror the source comments and static_asserts;
unknown keys/names, non-ints, out-of-range values, KC not dividing 2048 and combined static smem > 48KB are rejected as `invalid_edit`).
The harness rewrites only the value on `constexpr int NAME = <n>;` lines inside NS-EDITABLE-BEGIN/END and renders the unified diff with difflib,
then the unchanged strict parser, audit, sandbox, gates, store and attribution run on it. A no-op is `no_change`. The log keeps the raw North
response, the parsed edits, the rendered diff, and the label `North-authored (structured edit)`: North chose every value, the harness only formatted.
Diff mode remains available (`--mode diff`). The prompt shows the dev profile and current values only, never held-out data.

Launch (live, GPU spend; confirm cost first; proposer must be deployed, `modal deploy src/selfimprove/modal_north.py`):

    modal run modal/selfimprove_eval.py::main --track kernel --proposer north --mode structured --rounds 3 --max-usd 5 --confirm-cost --store results/selfimprove_kernel_structured

## Harness changes made after live North runs (disclosure)
- 2026-11-06: `audit.py` shape-specialization rule now ignores `//` comment text. Live structured run 1 (`results/selfimprove_kernel_structured/`, 3 rounds, all `audit_reject`) was rejected because the `MT_SPLIT` line's comment contains `(<= 16)` and 16 is a dev-only shape; any edit to that line would have been rejected. This was changed AFTER seeing the rejection; the rejected run is kept as-is. Real conditionals on dev-only shapes are still rejected (regression test in `test_structured.py`). The held-out-shape speed gate is unchanged.

## Remote (laptop-proof) loop
The local `::main` entrypoint dies with the client. `::remote` runs the loop in a CPU Modal function (`run_loop_remote`, 24 h timeout) that calls `evaluate_candidate` (GPU, network blocked) and the deployed `NorthProposerService`. The store (`proposals.jsonl`, `versions/`, `CURRENT`) lives on Volume `northstar-selfimprove-store:/<store>`, committed after every log/version write; each proposal record is also printed to app logs as `NS_PROPOSAL_JSON {...}` (final summary `NS_LOOP_DONE`). Loop/Store/audit/gates/attribution/hash chain are unchanged (`src/selfimprove/remote_loop.py` only adds emit/commit hooks). Same guards as `::main` (`--confirm-cost`; mock needs a `dryrun` store name; mixed-lineage refusal). Re-running with the same `--store` continues the chain.
```
modal run --detach modal/selfimprove_eval.py::remote --track kernel --proposer mock --rounds 1     # dry run (mock proposer; still uses the GPU evaluator, ~USD 0.7)
modal run --detach modal/selfimprove_eval.py::remote --track kernel --proposer north --mode structured --rounds 3 --max-usd 5 --confirm-cost --store selfimprove_kernel_structured2   # LIVE, GPU spend
modal run modal/selfimprove_eval.py::pull --store selfimprove_kernel_structured2   # Volume -> results/<store>, verifies chain, refuses diverged local log
modal app logs <app-id> | grep NS_PROPOSAL_JSON
```
