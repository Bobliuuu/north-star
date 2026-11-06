# Policy-track runbook (draft-length / speculation policy table)

Status 2026-11-06. Nothing in this file has been launched on a paid GPU. Everything under "Verified" was run on CPU only.
Background: `docs/SELFIMPROVE.md` (loop, gates, store), `harness/selfimprove/policy_eval.py` (policy gates), `src/opt/policy_tables/spec_space.json` (the editable file).

## What the policy track is
North edits `src/opt/policy_tables/spec_space.json` (`gammas` = candidate draft block lengths 1..12, `modes` = subset of the expert schedules, `n_groups` 2..12).
The table is parsed as data with a strict whitelist; nothing in it is executed. Gates:
- check: 8 fresh-seed buckets, solver self-consistency only (feasible, finite, Dinkelbach lambda == N/T, prediction <= ceiling).
- quality: 24 held-out buckets (context 3072/5000/12000/20000 x routing x batch 1/2), non-inferiority on predicted speedup over no-speculation.
- speed: dev buckets (ctx 2048/8192/32768, code) vs held-out buckets (ctx 5000/12000/20000); metric = predicted ms per 1000 tokens.

## Read this before spending money: what a "win" would and would not mean
1. The metric is the Phase-3 optimizer's PREDICTED tok/s (measured A100 MoE cost model + an acceptance table). It is not a measured end-to-end speedup.
2. `results/accept_calibration.jsonl` (MEASURED draft acceptance, Phase 2) does not exist. Today only SYNTHETIC acceptance exists, so predicted numbers are a plumbing check.
   `make_config(..., require_measured_calibration=True)` raises for a live run; the Modal entrypoint passes that flag for every non-mock run.
   So the live commands below fail fast (RuntimeError, no GPU spent on the evaluator) until that file is produced.
3. The metric is deterministic (noise floor 0.000 in the CPU dry run), so the 3% speed threshold is a hard 1.03x on the model prediction.
4. The policy search space is small (a handful of gammas/modes). Per `RESULTS.md`, the modeled optimum is already plain decode or gamma=1 + fused, so the likely outcome is no gain.
5. Proposal mode: the policy track runs in `--mode diff` (structured edits exist for the kernel track only). In every logged kernel diff-mode run
   North's diffs failed the strict parser (`invalid_diff` 4/4, `no_diff_in_response` 3/3 in `results/selfimprove_kernel*`). Expect the same on a JSON file unless
   a structured mode is added for the policy table (not done; recommended before any live policy run).

## Verified on CPU (no GPU, no Modal)
- `cd src && python -m pytest -q selfimprove/tests`: 154 passed, 2 skipped, ~70 s.
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

0. Prerequisite (not yet done): produce `results/accept_calibration.jsonl` from a measured Phase-2 acceptance run on real North (separate cost; see `docs/OPT.md`).
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
   modal run --detach modal/selfimprove_eval.py --track policy --proposer north --mode diff --rounds 3 --max-usd 3 --confirm-cost --store results/selfimprove_policy
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
1. Do not launch the policy live run yet. First: (a) measured acceptance calibration, (b) a structured JSON edit mode for the policy table so North's output is not lost to diff formatting.
2. Then 1 round live (~USD 0.83), inspect the log, then extend.
