1. **Outside helpers (what H22 tested).** North is called several times from outside, each call with a different role prompt (planner, four specialists, selector). This is how most agent frameworks make "subagents". Result above: no gain on average.
2. **Embedded (H23, tested on North).** Small trainable worker modules live *inside* one frozen North forward pass: private recurrent state, a learned router, and a return channel into the residual stream (tapped after layer 12 of 49); no prompts, no extra calls. This is what our sister project [qwenagent](https://github.com/Bobliuuu/qwenagent) built on Qwen3-0.6B; H23 ports it to North. It was tested on a state-tracking task (ledger), **not** on the kernel-speedup task, so it says nothing yet about North making itself faster.

### H23 results: embedded workers inside frozen North (MEASURED, Modal A100-80GB, 1,500 steps, 100 eval items per cell)
Trained on 2-3 entities, tested on 2-8. Accuracy (chance is 0.10):

| Mode | seeds | P=2 | P=3 | P=4 | P=6 | P=8 | zero the return signal (P=4) |
|---|---|---|---|---|---|---|---|
| frozen North, nothing added | 1 | 0.15 | 0.14 | 0.14 | 0.10 | 0.14 | n/a |
| adapter (stateless add-on, 970k params) | 2 | 1.00 | 0.98 / 1.00 | 0.97 / 0.99 | 0.96 / 0.95 | 0.90 / 0.89 | 0.14 |
| single shared state (837k) | 2 | 1.00 | 0.98 / 0.97 | 0.95 / 0.97 | 0.93 / 0.95 | 0.85 / 0.92 | 0.14 |
| **multi: private slots + learned router (the embedded "subagents", 837k)** | 2 | 0.99 / 0.96 | 0.97 / 0.94 | 0.94 / 0.90 | 0.93 / 0.93 | 0.85 / 0.84 | 0.14 |

(two numbers = seed 0 / seed 1.) Findings: (1) Embedding works on North: frozen North is at chance (0.10-0.15) and every trained mode reaches 0.84-1.00 on up to 8 entities, even though training used 2-3. (2) The answer flows through the workers: zeroing the return signal sends every mode back to 0.14. (3) Private slots do not help: mean at P=8 is adapter 0.895, single 0.885, multi 0.845; mean at P=4 is 0.98 / 0.96 / 0.92. The router is only partly clean (purity at P=4: 0.84 seed 0, 0.54 seed 1) and the multi arm is the least stable, as in qwenagent on Qwen. (4) Long sequences (P=4, more updates per entity) are hard for all modes (0.31-0.54), same wall as in qwenagent.
Caveats: 2 seeds per trained mode, 1 for frozen; one task family; accuracy cells have about +-0.05 binomial noise at n=100; capability test only, no speed measurement. Code: modal/embedded_north_run.py, src/embedded/. Raw: results/embedded_main_20261112T144436Z.json, logs/embedded_main_vol/ (per-run logs, results, worker checkpoints). Cost about $4.3 GPU time.

These are results from the embedded approach on a different model (frozen Qwen3-0.6B up to 4B) and a different task (interleaved register / ledger tracking). They are the nearest evidence for what embedding would do, not evidence about North.

| Benchmark | Result | Win? |
|---|---|---|
| Embedding trainable workers in a frozen Qwen works at all | chance → well above chance; zeroing the return channel sends it back to chance | ✅ WIN |
| Random register names make tracking generalise (q7, 5 seeds) | trained on 2-3 registers, still well above chance at 16 | ✅ WIN |
| Private slots vs a shared state or stateless adapter (q7, q9, 5 seeds) | no advantage: 16 registers `adapter` 28.4, `single` 27.3, `multi` 25.6, `oracle` 19.6 (%) | ❌ NO WIN |
| Workers vs LoRA of about 441k parameters (q10, 5 seeds) | mean accuracy `multi` 0.410 vs LoRA 0.394 vs `adapter` 0.458 | ➖ TIE |
| Realistic ledger task (q12) | trained modes ≥0.88 at P=2-3 (frozen Qwen at chance); at P=12 `single` 0.53 vs `multi` 0.39 | 🟡 PARTIAL |
| Scale (q13) | `adapter` at P=4: 54% / 58% / 72% for 0.6B / 1.7B / 4B; `multi` falls further behind at 4B | 🟡 PARTIAL |

Takeaway: embedding extra modules inside a frozen model is feasible, but the "private subagent slots" part did not beat simpler baselines. Source: qwenagent `README.md` and `results/`.
