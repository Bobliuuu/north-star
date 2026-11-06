# TAGE-style predictor for North self-speculation

Status: design plus a CPU oracle-verifier simulation (`src/tage`). No North run, no GPU, no speed claim.
Prior art and overlap verdicts: PRIOR_ART.md section "TAGE-style tagged geometric-history predictors".

## 1. Idea
Treat each decode step as a branch: "will the cheap guess for the next token(s) match what full North commits?"
CPU branch predictors solved the analogous problem with TAGE (Seznec): several tables indexed by hashes of
geometrically longer global-history lengths, tagged to avoid aliasing, longest tag-hit wins, a confidence counter
per entry, allocate in a longer table on a mispredict. Here history = North's committed token stream (and optionally
routing summaries). Zero new weights: every table is built online from committed tokens.

One structure serves five consumers:

| id | consumer | what the structure supplies |
|---|---|---|
| a | next-token proposals | provider entry token (longest-history hit), chained speculatively for k tokens |
| b | accept/reject confidence -> block length / disable | calibrated P(provider correct) per (provider table, counter); chain product vs threshold theta; theta from the cost model |
| c | draft-config selection (expert r, attention budget) | per-history-context bandit/counter: table maps context hash -> best config id (see 3.c) |
| d | two-ahead: which continuation to pre-draft while verify runs | predict the outcome (accepted length a, bonus token) from provider confidence and alt-prediction; pre-draft the top-B outcomes |
| e | expert-set prediction for weight-tile prefetch | same tagged tables, entry holds a 128-bit expert mask (or top-8 ids) per layer group instead of a token |

## 2. Predictor (implemented: `src/tage/predictor.py`)
- T0: tagless base table indexed by last token (bimodal analogue), 2^14 entries.
- T1..T7: history lengths L = 2,3,5,8,13,21,34 tokens; 2^12 entries each; index and 10-bit tag from a polynomial
  rolling hash of the last L committed tokens (prefix hashes, O(1) per length).
- Entry: predicted token (32 b), tag (10 b), confidence counter (3 b, 0..7), useful bit (1 b).
- Predict: provider = longest tag hit; alt = next-longest hit, else base.
- Draft block: repeat predict on the speculatively extended history (no table writes). Stop at kmax, on no prediction,
  or when the running product of calibrated per-token accept probabilities drops below theta.
- Calibration: `calib[(provider, counter)] = (#correct, #seen)` updated at every COMMITTED position (including positions
  that were never drafted), so it is unbiased by selection. Prior 1/3.
- Commit-time update (per committed token t at history h): provider counter +1 if provider token == t else -1; at counter 0
  the entry's token is replaced; a useful bit is set when counter >= 2 and cleared on a miss. On a provider misprediction,
  allocate (tag, token=t, counter=1) in up to 2 longer tables whose useful bit is 0; if the slot is useful, clear its bit
  (ageing). All useful bits reset every 2^14 commits.

### Update / rollback rules
- Tables are updated ONLY from tokens the verifier committed (accepted draft prefix plus correction/bonus token). Rejected draft
  tokens never touch tables. Therefore rollback of table state is a no-op; the only discarded state is the speculative history
  extension used while drafting (a local copy; the committed history pointer is the only durable state).
- Updates may be batched after verify (one pass over the a+1 committed positions); order is the stream order, so results equal
  token-at-a-time updates.
- KV/versioned-cache interplay is unaffected: the predictor reads token ids only. If it later reads routing summaries (3.e) those must
  come from the verify pass that wrote the committed KV version, never from draft activations.
- Doc/request boundary: history restarts, tables persist (warm session) or are cleared (cold request); both are evaluated.
- Determinism: updates are integer and order-fixed, so the predictor is bit-reproducible given the committed stream. It can only
  change WHICH tokens are proposed, never what is committed (verification stays authoritative).

### State cost (`TagePredictor.state_bytes`)
7 x 4096 x 6 B + 16384 x 5 B ~= 254 KB with the default config (token 4 B + tag/ctr/useful packed in 2 B; base 5 B). Fits an SM-cluster L2
slice or host cache; the lookup is 7 hash+compare operations per drafted token. It can live on the CPU (hash lookups of ~1 us) with
proposals delivered over PCIe/pinned memory; only the draft token ids are transferred. Calibration table: <1 KB.
Compute cost is not included in any number below (taken as zero); it must be measured on CPU alongside the GPU step before use.

## 3. Consumers in detail
### a. Proposals
Overlaps SuffixDecoding / prompt-lookup / n-gram hash tables (PRIOR_ART.md). Differences: multiple history lengths with per-entry
confidence, bounded memory, allocation-on-mispredict instead of unbounded suffix trees. No claim of better raw hit rate; the
simulation below shows it is NOT better at fixed k.
### b. Block length / disable
Policy: draft token j iff prod_{i<=j} p_i >= theta where p_i is the calibrated accept probability for the provider of draft token i.
Break-even: with verify cost 1 + alpha*w for width w, one extra drafted token is worth it iff p_chain >= alpha / (1 + ...) roughly alpha
in the small-alpha regime; theta should be set from the measured marginal cost (G0 profile), not tuned on accuracy. If the first token
fails the gate the step runs plain decode (speculation disabled for that step at no extra cost).
### c. Draft-config selection (expert r, attention budget)
Not simulated (needs North drafts). Design: a second table family with the same tagged geometric contexts, entry = argmax over a small config
set {(r, attn budget, skip)} of a running per-config "accepted tokens per unit cost" estimate, updated by the SAME verifier outcome that
commits tokens (bandit feedback: only the chosen config is observed, so epsilon-explore 5%). The context could be the provider id + counter +
last token class, i.e. coarse; long contexts dilute statistics. Expected benefit is unknown; tables here are an alternative to a CP-SAT
schedule from measured mean acceptance, and should be compared against a plain per-bucket mean (a bandit with no history) first.
### d. Two-ahead (Phase 5 / Saguaro-related)
While verify of block B_k runs, pre-draft the next block for the most likely outcomes. Outcomes = (a, bonus). Predictor supplies:
(i) full-accept outcome: bonus = TAGE prediction at position k+1 (one more chained lookup), P(full) = chain probability;
(ii) a rejection at position j: bonus = alt prediction at j (next-longest hit or base).
Pre-draft only outcome sets whose predicted probability exceeds a threshold; "branch" budget B bounded by free draft columns/bandwidth
(PLAN Phase 5: kill on negative net benefit). Simulated coverage below.
### e. Expert-set prediction for weight-tile prefetch
Same tagged tables, value = per-layer expert set learned from committed tokens' recorded routing (verify pass only). Not evaluated: no North
routing trace exists in this repo and I did not fabricate one. Prior work (ProMoE, Fate, MoE-Infinity) uses router state or learned predictors and is likely
stronger. Open question for G0 data: how stable is the expert set given the last n token ids? It must beat "previous token's experts" (Cohere reports
adjacent-token overlap 0.381 vs 0.118 independent for a different MoE) before this is worth building. Note prefetching only helps when weights are NOT resident;
on an 80 GB H100 North's 61 GB weights are resident, so (e) matters mainly for tile scheduling / L2 residency inside the megakernel, a much smaller prize.

## 4. Simulation (src/tage/sim.py, run_eval.py, tests)
ORACLE-VERIFIER SIMULATION, NOT A NORTH RESULT. The "verifier" is the true next token of a recorded stream; a draft token is accepted iff it equals the stream
token. The streams are human-written source/text files on this Mac, NOT North generations. Tokenizer is Qwen3-0.6B BPE (not North's). So absolute rates say nothing about
North output; code written by a model in an agentic loop (repeated edits, tool output echo) is likely MORE repetitive than these files. Costs: verify pass = 1 + alpha*(drafted tokens), baseline AR = 1 per token, alpha in {0.05, 0.15, 0.30}; draft compute and proposer CPU cost = 0. alpha is a free parameter of the model, NOT a measured North number.

Corpora (label, files, tokens): see results/tage/*.log. Modes: warm-session (tables persist across files in a repo, files in random order, seed 0, exact duplicates dropped)
and cold-per-file (state reset every file). Baselines: pld = prompt lookup on current file only (n-gram 3..1 or 3..2); suffix-last = longest match among orders 1..32 over all history, most recent
continuation (approximation of SuffixDecoding, no tree/frequency scoring); suffix-freq = frequency argmax variant; suffix+gate = suffix-last with a calibrated start gate by matched order (ablation to separate
gating from TAGE structure). All use kmax 8 unless noted.

All numbers: warm-session, kmax 8, ORACLE-VERIFIER SIMULATION on local human-written files (see above). Raw logs: results/tage/oracle_sim.log (kittens-vm, north-star-src, qwentpu_v2; the combined JSON was not written before the push deadline) and results/tage/sim_{cakebench,bobliuos,doomsrc}.{log,json}. Regenerate: `cd src && python3 -m tage.run_eval`; tables: `python3 -m tage.report <json...>`.
Corpora (files/Qwen3-BPE tokens): kittens-vm 5/17.9k, north-star-src 46/76.3k, qwentpu_v2 39/75.9k, cakebench 7/9.2k, bobliu-os 36/62.1k, doom-src (C) 60/65.5k.

Modeled speedup tokens/(steps + alpha*drafted), alpha = 0.15 (free parameter, NOT measured North cost), warm-session:

| corpus | pld n=3..2 | suffix-last (ungated) | suffix+gate 0.3 | TAGE fixed k=8 | TAGE theta=0.3 | TAGE theta=0.5 |
|---|---|---|---|---|---|---|
| kittens-vm | 0.92 | 0.70 | 0.99 | 0.67 | 1.10 | 1.07 |
| north-star-src | 0.99 | 0.76 | 0.91 | 0.70 | 1.16 | 1.11 |
| qwentpu_v2 | 0.95 | 0.74 | 1.01 | 0.68 | 1.15 | 1.12 |
| cakebench | 0.98 | 0.79 | 0.95 | 0.75 | 1.16 | 1.13 |
| bobliu-os | 1.05 | 0.92 | 1.07 | 0.87 | 1.33 | 1.27 |
| doom-src (C) | 1.02 | 0.88 | 0.98 | 0.83 | 1.29 | 1.25 |

Other warm-session facts (theta=0.3 TAGE vs ungated suffix-last): tokens/step 1.15-1.51 vs 1.31-1.84; draft fraction of steps 20-52% vs 76-90%; first-token proposal hit 0.50-0.60 vs 0.20-0.34 (gated, so selection-biased); token precision 0.49-0.56 vs 0.05-0.13. Ungated, TAGE fixed-k has a LOWER hit rate than suffix-last (e.g. 1.46 vs 1.53 tokens/step on north-star-src). At alpha=0.05 all methods are 1.0-1.4x and ungated suffix-last is competitive (bobliu-os 1.38 vs TAGE theta=0.3 1.44). Accepted-length distributions per method are in `python3 -m tage.report` output; for all methods 70-90% of steps accept 0 draft tokens, the tail is dominated by a spike at the full kmax=8 (copy/boilerplate runs). kmax 16 gives the same result as 8 under gating (chains die before 8).
Cold-per-file (tables reset per file) is within ~0.02-0.05 speedup of warm for TAGE; the state persistence gain is small on these corpora.

Honest reading:
1. The gain is mostly the ACCEPT-CONFIDENCE GATE (consumer b), not the tagged-geometric structure. Adding the same kind of calibrated gate keyed by match order to suffix-last (suffix+gate) recovers part of it, but TAGE-gated is still better by ~0.05-0.2 modeled speedup at alpha 0.15 because the calibration key (provider table x counter) is richer than match length. At fixed block length TAGE is not better than suffix matching.
2. Absolute modeled gains are small (up to 1.33x at alpha 0.15, 1.1-1.2x typical) on human-written code. Real agentic North output may be much more or less repetitive; this sim cannot say.
3. Two-ahead (consumer d), TAGE theta=0.5, warm: P(full accept)~0.58; given a full accept the bonus token is predicted correctly ~0.49-0.53 of the time; the alt prediction hits at a rejection only ~0.03-0.05. Outcome coverage of the best single pre-draft branch (full-accept + predicted bonus) is ~0.29-0.32 of drafted steps (theta=0.3: ~0.2); adding alt branches adds <0.02. So one pre-draft branch is useful on roughly a third of the (already minority) drafted steps; rejected-branch coverage is essentially nil. Expect Saguaro-style gains to be capped by this unless the verifier-side distribution (top-2 logits) supplies the bonus candidates.
4. Consumers (c) config selection and (e) expert prefetch were NOT evaluated (no North drafts or routing traces).

## 5. Decision
Is it worth GPU time? Only a cheap, narrow version.
- Worth a small experiment (hours of H100/A100, no new kernels): run the TAGE proposer + calibrated gate on CPU beside the Phase 2/3 self-speculation loop on REAL North agentic-coding generations, log (proposal, verify outcome) pairs, and replay offline. Cost is mostly CPU. Compare against prompt-lookup/suffix with and without a gate, at the MEASURED alpha from G0. Decision rule: proceed only if gated TAGE beats the best gated baseline by >=5% modeled tokens/s on North traces.
- Not worth building yet: (c) config selection and (e) expert prefetch (no data; prior art stronger; weights resident on 80 GB) and (d) multi-branch two-ahead (alt coverage ~0 in sim; one-branch coverage ~30% of drafted steps).
- The most defensible piece is (b): a zero-weight online accept-calibration gate that disables speculation when the draft is unlikely to pay. It is cheap, composes with any proposer (model-free or North self-draft), and the sim shows it matters most when verification width costs anything (alpha >= 0.15). That belongs in Phase 3 adaptive block length regardless of the TAGE structure.
- Novelty: low (see PRIOR_ART.md). Do not claim a new principle; at most "unified zero-weight proposal+confidence+outcome predictor".
- Caveats: Qwen3 tokenizer, human-written files, oracle verifier, alpha assumed, proposer CPU latency ignored, single seed per corpus ordering, no confidence intervals.
