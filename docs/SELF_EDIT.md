# SELF_EDIT (track 2): North edits its OWN (verifier) weights, gated independently

PRE-REGISTERED before any GPU run (2026-11-06). Status: see "Results" at the bottom (empty until a run is recorded).
Sibling track: docs/WEIGHT_EDIT.md (draft-only router adapter; verifier untouched). This track CHANGES the served model, so it needs quality gates, not just a losslessness check.

## Experiment A: per-layer expert precision, chosen by North
Edit: fake-quantize (round-to-nearest, symmetric int4, group g along the input dim, bf16 scale) the routed-expert weights (`gate_up_proj`, `down_proj`) of a SET of MoE layers; everything else stays bf16/original. "Fake" = weights are rounded in place and stay bf16 tensors, so MEASURED quality is the real quality of that int4 grid, but there is NO speed measurement (no int4 kernel is run here).
Why this edit: expert weights are ~55% of North's per-step weight stream (RESULTS.md H5); the inference-experiment campaign found W4A16 experts raise decode $/h ~+13% but North W4A16 passed its HE+ gate once and failed on two repeats, and an ILP found layers 0-3 and the last-layer MLP sensitive (~/inference-experiment research/equiv-solver/mixprec, docs/...-north-fp8attn). A *per-layer* choice (which layers tolerate int4) is exactly the decision North could make itself.

Action space (src/selfedit/recipe.py): `{"int4_layers": [MoE layer ids], "group": 16|32|64|128}`. Harness validates; ids must be MoE layers; group must divide hidden_size and intermediate_size.

### Splits and data
Teacher = the ORIGINAL North (bf16 weights, same loader), logits cached before any edit. Sequences: harness workloads, `dev` split for evaluation (n_eval = 6 prompts, ctx 1024 + 48 greedy continuation tokens, teacher-forced), `calib` split for the sensitivity table and for recovery training. `heldout` is never touched. Metrics are computed on every position >= 128 of each evaluation sequence (prompt AND continuation): mean forward KL(teacher || edited) in nats/token, and top-1 agreement with the teacher. Proxy only: NOT HumanEval+/BFCL, so a pass here is necessary, not sufficient.

### Cost bookkeeping (INFERRED, never claimed as measured)
expert_bytes_saved = (|int4_layers|/48) * (1 - bits_per_weight/8), bits_per_weight = 4 + 16/group, relative to the FP8-expert incumbent (8 bits). Speed gain from bytes is a derived estimate (decode is expert-byte bound per H5/inference-experiment), not a measurement.

### Arms (all at the SAME layer count K = 24 of 48, group 32, so bytes are matched; expert_bytes_saved = 21.9% of expert stream)
1. `north_blind`: North proposes K layers seeing only the layer count, architecture facts and its own past results (3 rounds, history of KL fed back).
2. `north_table`: same, plus the harness's per-layer sensitivity table (single-layer KL deltas on the calib split).
3. `greedy_table`: harness picks the K least-sensitive layers from the table (non-LLM baseline).
4. `random`: 5 random K-subsets (seeds 0-4).
5. `uniform_all`: all 48 layers (reference point, not matched).
Frontier: greedy_table at K = 8, 16, 24, 32, 48 to show quality vs bytes.

### Gates (a recipe PASSES only if all hold)
- G-quality: mean KL <= 0.03 nats/token AND top-1 agreement >= 0.96 on dev (n = 6 sequences, ~6k positions).
- G-bytes: expert_bytes_saved >= 15%.
- G-nonexpert: every non-expert parameter fingerprint-equal to the original (sampled hash), i.e. nothing but routed-expert weights of the listed layers changed; layers not listed are bit-identical to the original.
- G-selfauthored (for the self-improvement CLAIM only): the recipe came from North's own reply (source=llm, no fallback), valid, K layers exactly.
Claims allowed:
- "North improved itself" requires a `north_*` arm recipe that passes ALL gates incl. G-selfauthored AND beats the random-arm median KL (strictly lower than >= 3 of 5 random subsets; paired, same K). Beating greedy_table is NOT required, but we report the ratio KL(north)/KL(greedy).
- If only greedy/random pass: report "weights-edit gives bytes at passing quality, North did not add value" (honest partial).
Kill criteria:
- K1: greedy_table at K=24 has KL > 0.03 -> int4 RTN g32 is too lossy for half the layers; report the frontier and the largest passing K, no North claim at K=24 (re-run North at the largest passing K only if budget allows).
- K2: all `north_*` rounds fall back (no valid JSON) -> no self-authored claim; record raw replies.
- K3: `north_*` KL not below random median -> North's layer choice has no demonstrated value.

## Experiment B: recovery by self-distillation (only if A leaves a gap)
Take the best passing-or-nearest recipe from A. Train, on `calib` windows with the ORIGINAL North's cached logits as targets, an always-on router adapter (low-rank router logit delta + per-layer gain, same parameterization as src/weightedit, applied on the VERIFIER path now) to reduce KL(teacher || edited+adapter). Trained on calib, gated on dev:
- G-recover: dev KL reduced by >= 15% relative, paired-bootstrap (over sequences) 5th percentile of the KL reduction > 0, AND dev top-1 agreement not lower.
- Cost: adapter FLOPs <= 5% of expert FLOPs/token (unchanged experts count). Note this adds ~0.4M params and changes the served model, so it needs the same G-quality.
Kill: no relative KL reduction >= 15% on dev -> record negative.

## Honesty rules
MEASURED = numbers produced by a run in results/selfedit_*. INFERRED = byte arithmetic. Seeds, n, and splits stated with every number. North's raw replies are kept in the results JSON. Fallback proposals never count as self-authored.

## Budget
Hard cap $6 on A100-80GB (~$0.0007/GPU-s). Plan: smoke (~$0.3), full A (+B) (~$1-2). Every launch logged in RUNS.md, spend in results/SPEND.md, detached Modal runs only, results on Volume `northstar-results` and in app logs.

## Results
(none yet)
