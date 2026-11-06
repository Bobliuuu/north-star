# Policy-track runbook (draft-length / speculation policy table)

Status 2026-11-06. Nothing in this file has been launched on a paid GPU. Everything under "Verified" was run on CPU only.

**Both blockers are fixed (branch `policy-structured`)**: (1) `--mode structured` exists for the policy track and is the default; (2) `results/accept_calibration.jsonl` now exists, built from REAL measured acceptance only.
**A policy-track "win" is NOT a measured speedup.** The gate metric is the Phase-3 optimizer's PREDICTED tok/s (measured cost-model tables + the measured acceptance table). Never report a promoted policy table as an end-to-end speedup; to claim one, run the table on the real decoder.
Background: `docs/SELFIMPROVE.md` (loop, gates, store), `harness/selfimprove/policy_eval.py` (policy gates), `src/opt/policy_tables/spec_space.json` (the editable file).

## What the policy track is
North edits `src/opt/policy_tables/spec_space.json` (`gammas` = candidate draft block lengths 1..12, `modes` = subset of the expert schedules, `n_groups` 2..12).
The table is parsed as data with a strict whitelist; nothing in it is executed. Gates:
- check: 8 fresh-seed buckets, solver self-consistency only (feasible, finite, Dinkelbach lambda == N/T, prediction <= ceiling).
- quality: 24 held-out buckets (context 3072/5000/12000/20000 x routing x batch 1/2), non-inferiority on predicted speedup over no-speculation.
- speed: dev buckets (ctx 2048/8192/32768, code) vs held-out buckets (ctx 5000/12000/20000); metric = predicted ms per 1000 tokens.

## Read this before spending money: what a "win" would and would not mean
1. The metric is the Phase-3 optimizer's PREDICTED tok/s (measured A100 MoE cost model + an acceptance table). It is not a measured end-to-end speedup.
2. `results/accept_calibration.jsonl` (MEASURED draft acceptance) now exists: 15 rows, A100-SXM4-80GB, built by `scripts/build_accept_calibration.py` from `results/spec.jsonl` (see "Measured calibration" below).
   It is small (4 code tasks, 26-97 verify rounds per draft config, 64-token generations, one hardware), `carry_hit_given_full` is NOT measured (set to 0.0, carry disabled), and no attention-budget (long-context) draft was measured (attn_budget 0 only).
   So predicted numbers rest on a thin acceptance sample. `make_config(..., require_measured_calibration=True)` now requires the file to exist and EVERY row to be `provenance == "measured"`; the Modal entrypoint passes it for every non-mock run.
3. The metric is deterministic (noise floor 0.000 in the CPU dry run), so the 3% speed threshold is a hard 1.03x on the model prediction.
4. The policy search space is small (a handful of gammas/modes). Per `RESULTS.md`, the modeled optimum is already plain decode or gamma=1 + fused, so the likely outcome is no gain.
5. Proposal mode: the policy track now defaults to `--mode structured` (`src/selfimprove/structured_policy.py`), as the kernel track does. North emits ONLY
   `{"edits":[{"name":"gammas|modes|n_groups","value":...}],"rationale":"..."}` (<= 3 edits, each name once; gammas = 1..12 distinct ints in [1,12], modes = distinct subset of
   `two_pass, spec_two_pass, fused`, n_groups = int in [2,12]; gammas/modes are sets, so the harness renders them in canonical order). The harness validates strictly and renders the unified diff
   (touching only those three value lines); the diff then goes through the unchanged strict parser, scope (`spec_space.json` only, <= 3 hunks / 12 changed lines), audit, sandbox, gates, store.
   North chose every value; the record carries the label `North-authored (structured edit)`. Any other key, out-of-range value, wrong type or extra field is `invalid_edit` and nothing is evaluated.
   Measured-history feed: every prompt lists earlier proposals of the session (and `prior_history` from earlier stores) as `id: verdict | edits ... | PREDICTED dev speedup Nx (model prediction, not a measurement)`;
   dev split only, held-out numbers never. `--mode diff` remains available (kernel-track diff mode failed the strict parser 7/7 live; expect the same here).

## Verified on CPU (no GPU, no Modal)
- `cd src && python -m pytest -q selfimprove/tests`: 218 passed, 2 skipped, ~80 s (adds `test_structured_policy.py`, `test_accept_calibration.py`, measured-calibration guard tests).
- `cd src && NS_SLOW=1 python -m pytest -q selfimprove/tests -k "policy or slow or real"`: 21 passed, ~200 s (includes the full policy dry-run loop and the fake-binary quality harness).
- Direct end-to-end dry run, policy track, MockProposer, local `Evaluator` (propose -> parse -> audit -> sandbox -> correctness -> quality -> speed -> store -> hash-chained log):
  verdict `gate_fail:speed` (the mock adds gamma 5; dev speedup 1.000, held-out 1.000, threshold 1.03), correctness 8 cases rel_l2 0.0, quality 24 items delta 0.0,
  `verify_log()` True, attribution: developer v0000 only, `reportable_as_north_authored` False. Wall time 103 s on CPU. No bugs found; no code changes were needed.

Reproduce the free CPU dry run (writes a store under `/tmp`-style scratch dir; the store name must contain `dryrun` by convention):
```
cd <repo root>
PYTHONPATH=src python - <<'EOF'
import sys, runpy, pathlib, json
R = pathlib.Path.cwd(); sys.path.insert(0, str(R / "src"))
from selfimprove.loop import Loop, editable_files_in_repo
from selfimprove.proposer import MockProposer
from selfimprove.store import Store
H = runpy.run_path(str(R / "harness/selfimprove/config.py"))
cfg = H["make_config"](R, "policy")                      # synthetic calibration allowed (dry run)
base = editable_files_in_repo(R, cfg.scope)
st = Store(pathlib.Path("/tmp/store_dryrun_policy"))
res = Loop(cfg, st, MockProposer([H["mock_diff"]("policy", base)]), mode="diff").run(
    1, max_total_usd=1, profile_summary=H["profile_summary"]("policy"), dev_results=H["dev_results"]("policy"))
print([r["verdict"] for r in res], st.verify_log(), st.attribution())
EOF
```

## Commands
All from the repo root, Modal CLI logged in. `--detach` goes between `run` and the file so the job survives a laptop crash.
Per `keep-runs-alive-and-in-repo`: after any run, copy the store and Modal logs into the repo `logs/` and `~/north-star-work/agent-traces/`, commit and push.

0. Prerequisite (done): `results/accept_calibration.jsonl` (see below). Re-run `python scripts/build_accept_calibration.py` after any new measured spec run; it is byte-deterministic.
1. Deploy the proposer once (weights already in Volume `northstar-weights`):
   ```
   modal deploy src/selfimprove/modal_north.py
   ```
2. Optional Modal dry run (mock, A100 evaluator even though the policy track is CPU-bound; ~USD 0.7):
   ```
   modal run --detach modal/selfimprove_eval.py --track policy --proposer mock --rounds 1 --store results/selfimprove_dryrun_policy
   ```
   Prefer the free CPU dry run above.
3. Live North run (spends money; needs steps 0 and 1, and `--confirm-cost`):
   ```
   modal run --detach modal/selfimprove_eval.py --track policy --proposer north --mode structured --rounds 3 --max-usd 3 --confirm-cost --store results/selfimprove_policy
   ```
4. Inspect (local, free):
   ```
   PYTHONPATH=src python -m selfimprove.cli attribution --store results/selfimprove_policy
   PYTHONPATH=src python -m selfimprove.cli cost --store results/selfimprove_policy
   PYTHONPATH=src python -m selfimprove.cli verify-log --store results/selfimprove_policy
   ```
   Caveat: the Loop (store, log) runs in the local entrypoint; `--detach` keeps the Modal app alive if the client disconnects, but the store is written locally, so keep the terminal/session alive and copy the store to `logs/` right after.

## Expected cost (from `estimate_cost` in `modal/selfimprove_eval.py`; ESTIMATES, A100-80GB at USD 2.50/h list, unmeasured on Modal)
| run | minutes | USD |
|---|---|---|
| dry run, mock, 1 round | ~10.5 | ~0.44 (estimate_cost: evaluator 8 + 1 + 1.5 min; docs/SELFIMPROVE.md quotes ~0.7) |
| live north, 1 round | ~20 | ~0.83 |
| live north, 3 rounds | ~45 | ~1.88 |
| live north, 5 rounds | ~70 | ~2.92 |
Includes ~6 min proposer cold start/model load once. The 8 min/round evaluator figure is conservative: the policy evaluation took 103 s on a laptop CPU.
The `--max-usd` flag is a hard cap checked between rounds. Actual North proposer cost per round has never been measured (no live round has recorded a usable cost); real figures come from `Store.cost_summary()` after round 1.

## Recommended order
1. Both prerequisites are done. Expect the likely outcome to be "no gain": with the measured table a small space (e.g. gammas [1,2], two_pass) already predicts the SAME tok/s as the full baseline space on the dev buckets (CPU check), so the speed gate (>1.03x predicted) has little room.
2. Then 1 round live (~USD 0.83), inspect the log, then extend.

## Measured calibration (`results/accept_calibration.jsonl`)
Built by `python scripts/build_accept_calibration.py` (optional `--spec FILE...`, `--out`, `--dry-run`); never hand-edited (a test regenerates it and compares bytes).
- Input: `phase == "1_selfspec"` rows of `results/spec.jsonl` (A100-SXM4-80GB, `harness/bench_spec.py`). Each run records `accept_rate_per_pos[i] = #(rounds with accepted > i) / rounds`, i.e. the
  empirical P(A >= i+1) with A capped at the run's gamma. That is exactly the calibration pmf (`pmf[j] = surv[j]-surv[j+1]`, last bin = P(A >= G)); runs of one (workload kind, draft config) are pooled round-weighted.
- Draft key = (attn_budget 0, r = expert_r or 8 for native top-8, sorted skipped layers). A key measured at several gammas (`r4_g2/g4/g8`) uses only the largest gamma run set (g8 for r4); lookahead beyond a row's measured gamma is unavailable, so the optimizer skips gamma > measured (e.g. skip configs only support gamma <= 4).
- Provenance per row: `provenance: "measured"`, `source`, `source_sha256`, `hardware` (from `spec.meta.json`), `model`, `model_rev`, `n_tokens`, `n_rounds` (= `n_samples`), `n_runs`, `tasks`, `gamma_measured`, `config_names`, `unused_configs`, `n_identical_runs`.
- Refuses (exit 2, writes nothing) on: a source without per-position acceptance, missing/non-integer counts, non-monotone survival, rate lists whose length != gamma, missing hardware metadata, duplicate keys across sources.
  `results/h100_spec.jsonl` is REFUSED: its `g_spec` rows only carry `mean_accepted` / `committed_per_round`, so a pmf would have to be invented. It remains a throughput record, not a calibration source.
- Known limits: the last verify round of a 64-token run can be cut by the token budget (slightly lowers accepted counts); acceptance is in-sample for the same 4 code tasks; calibration is per-hardware (A100 only; the H100 speed numbers are not mixed in).
