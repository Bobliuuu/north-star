# WEIGHT_EDIT: North improves North by editing its own (draft) weights

Status: PROTOTYPE (CPU-tested on a tiny Cohere2Moe config). No GPU run, no North result yet. Nothing here is a measurement on North.

## Idea
The kernel self-improve loop (docs/SELFIMPROVE.md) lets North propose kernel/schedule patches: 0 of 6 live proposals passed the speed gate. This track gives North a different action space: **weights**. North does not write weight deltas as text (hopeless: huge, no gradient). North proposes a *recipe* (small JSON); the harness executes it (distillation), gates the result independently, and versions the outcome with rollback.

## Why the first target is the draft, not the verifier
Measured blocker (RESULTS.md H5/H7): the draft step is only 8-15% cheaper than a full step, and top-4 experts with raw-sigmoid gates loses agreement (0.86-0.88 combined; renorm 0.63-0.69), while sparse attention alone keeps agreement >= 0.96.

Edit = a **draft-only router adapter**. It is applied only inside `draft()` mode. Consequences:
* The verifier path is never touched, so output stays exactly the verifier's greedy output (lossless speculation). Safety is by construction and is also *tested* (params hash + bit-identical verifier logits with the patch installed).
* The metric is offline and cheap: top-1 agreement / committed tokens per block vs the unedited verifier (same protocol as src/approx/sweep.py). It is not noisy wall-clock like the kernel gate.
* Cost-neutral: adapter FLOPs per token per layer = hidden*rank + rank*E (rank <= 16) vs expert math, and the number of active experts is unchanged (`expert_r`). Report it anyway.

Thesis change to state honestly: "zero new weights" no longer holds. The artifact is a *derived* delta (router adapter, ~MBs), same base model. Edits to the verifier weights are out of scope until the draft track shows signal.

## Action space (bounded; harness validates, rejects out-of-range)
Recipe JSON (src/weightedit/recipe.py):
| field | range | meaning |
|---|---|---|
| expert_r | 1..K-1 | draft keeps top-r experts (K = native top-k) |
| kind | `router_lowrank` / `router_bias` | logits += (h A) B (rank r_a) / logits += b |
| rank | 1..16 | rank of A,B (lowrank only) |
| layers | subset of MoE layers or `all` | which routers get an adapter |
| steps | 1..2000 | optimizer steps |
| lr | 1e-5..1e-2 | AdamW lr |
| window | 16..2048 | tokens per training window |
| temperature | 0.5..2 | KL temperature |
| gain | bool | learn per-layer scalar gate gain |
| seed | int | |
Initialisation is identity (B = 0, bias 0, gain 1), so step 0 equals the unedited top-r draft.

## Data and splits
* Teacher = the unedited North verifier (self-distillation; its own greedy continuations, same as the sweep).
* **train**: harness `calib` split. **gate**: `dev` split (never used for training). **heldout**: untouched until a human decides to run the final confirmation once.
* Training sequences are North's own outputs; drift/self-reinforcement risk is bounded because the target is the *frozen* verifier distribution, not the edited model's.

## Gates (src/weightedit/gates.py), all must pass to commit a version
1. **G-lossless**: base params hash unchanged; verifier-mode logits bit-identical with patch installed (inactive) vs unpatched. Hard fail.
2. **G-gain**: gate-split top-1 agreement improves over the unedited baseline draft by >= margin (default +0.02 abs) AND paired-bootstrap lower bound of the gain > 0 over blocks.
3. **G-overfit**: train agreement minus gate agreement <= 0.10.
4. **G-cost**: active experts/token unchanged; adapter overhead FLOPs <= 5% of expert FLOPs/token (reported).
Wall-clock speed is NOT a gate here (agreement is the proxy); it enters only in the follow-up (see below).

## Loop (src/weightedit/loop.py)
round: proposer(history) -> recipe JSON -> validate -> train (frozen base, adapters only) -> gates -> commit version to the store (or ledger-only reject) -> history grows. Proposers: `GridProposer` (deterministic baseline), `LLMProposer` (prompts North in verifier mode with its own history, parses JSON, falls back to Grid on parse/validation failure; every fallback is recorded). Store (src/weightedit/store.py): versioned dirs `v001/...`, append-only `ledger.jsonl`, `rollback(n)`, content hash per version.

## Kill criteria (pre-registered)
* K1: best recipe over >= 6 rounds does not lift gate-split committed tokens/block (gamma 4) by >= 0.3 over the unedited top-4 draft -> stop; report negative.
* K2: gain vanishes at ctx 8K versus 2K (train windows are short) -> note as limit; do not claim.
* K3: even with recovered acceptance, measured draft+verify cycle (megakernel/graphed executor) is not > 1.0x plain decode -> acceptance win is real but not an end-to-end speedup; say so. This is likely given H5 (draft ~ full step cost); the edit is what makes a *much cheaper* draft (r=2 / layer-skip + sparse attention) viable, so the follow-up sweep should lower r after training.
No claim of self-improvement unless a version passes the gates AND North authored the recipe (LLMProposer without fallback).

## What this does NOT show yet
Actual North numbers, 8K+ context behaviour, wall-clock, verifier-weight edits, online/continual adaptation. All unrun.

## Run plan (needs authorization; GPU budget separate from the $30 A100 phase)
1. `make test` (CPU, tiny config) - done in prototype.
2. Smoke: `modal run --detach modal/weightedit.py::main --smoke true` (A100-80GB, 1 recipe, few steps, ctx 2K). Training memory for 49 layers needs activation checkpointing; `train.grad_ckpt` is implemented, untested on North.
3. Round loop with LLMProposer, `--rounds 6`, cap $8. Results to Volume `northstar-results` + app logs + `results/weightedit_*.json`; log launch in RUNS.md.
