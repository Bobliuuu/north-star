# North live demo

Terminal viewer for REAL North generation (same `RefNorth` / `SelfSpec` / `plain_greedy` path as `modal/run.py`).
Per step it shows: token text, active execution mode (plain / self-spec gamma N + draft), committed tok/s (instant and cumulative),
acceptance (this round and cumulative), GPU memory, and cost (this run + cumulative optimisation cost = parsed `results/SPEND.md` total + live run cost).
Nothing is synthesised: every number is measured inside the Modal container (perf counters, `torch.cuda.memory_allocated`, token ids, wall clock x list price).

## Launch (PAID; not run by the author of this code)

```
modal run --detach modal/demo.py --task <workload id from harness/workloads> --mode both --gamma 4 --skip-k 8 --n-new 64
```
Never plain `modal run`. One A100-80GB, `--mode both` runs plain then self-spec on the same prompt.
Expected cost: about $0.25-0.45 (spec-run precedent: 868 s = $0.60 for 7 configs x 8 tasks; this is one load plus two 64-token generations; model load dominates, ~$0.0007/s).
The local entrypoint writes `demo/traces/<utc>.jsonl`, appends a row to `results/SPEND.md`; the remote also keeps a copy on the `northstar-results` volume under `/results/demo/`.
Throughput is the reference executor (not an optimised engine) and self-spec may be slower than plain; the demo reports what happens.

## Replay (free, CPU)

```
python demo/view.py --replay demo/traces/<utc>.jsonl [--delay 0.05]
```
Refuses (exit 2) traces with no `provenance`, missing required fields, an unknown `provenance.source`, or `source = FIXTURE` unless `--allow-fixture` (then a banner is printed).

Tests: `python -m pytest tests/test_demo.py` using `demo/fixtures/FIXTURE_*.jsonl` (hand-made, labeled FIXTURE, not North output).

## Event schema (`northstar-demo/1`, JSON Lines, one object per line)

1. `{"type":"run", "schema":"northstar-demo/1", "task", "prompt_len", "n_new", "mode_request", "spend_prior_usd", "spend_rows", "provenance":{...}}`
   Must be first. `provenance` required keys: `source` (`modal-live` for real runs, `FIXTURE` for tests), `model_id`, `model_rev`, `gpu`, `created_utc`; real runs also add `torch`, `executor`, `load_s`.
2. `{"type":"mode_switch","mode":"plain|selfspec"}` (only with `--mode both`).
3. `{"type":"step", ...}` one per committed token (plain) or per speculation round (self-spec):
   - `i` step index within the mode segment; `tokens` committed token ids; `text` their decoded text
   - `mode` `plain|selfspec`; `gamma` (0 for plain); `draft` label e.g. `skip8` or `skip8_r4` (null for plain)
   - `accepted`, `drafted`: drafts accepted / proposed this round (0/0 plain); `accept_cum` cumulative accepted/drafted (null for plain)
   - `committed_total` tokens committed in the segment; `gen_s` summed decode time (prefill excluded)
   - `inst_tok_s` = len(tokens)/step time; `cum_tok_s` = committed_total/gen_s
   - `gpu_mem_gb` allocated, `gpu_peak_gb` max allocated
   - `run_cost_usd` = container wall seconds (incl. model load) x $0.000694; `cum_cost_usd` = `spend_prior_usd` + `run_cost_usd`
4. `{"type":"end","mode", "committed_total","gen_s","cum_tok_s","gpu_peak_gb","run_cost_usd","cum_cost_usd"}` closes a segment.
5. `{"type":"error","error"}` remote failure; shown, never papered over.

Cost caveats: list-price estimate, excludes image build and CPU containers; the prior total sums the `est USD` column of every `results/SPEND.md` row (the file's own "Cumulative" line is not trusted).
