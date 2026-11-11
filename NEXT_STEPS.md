# NEXT_STEPS

## STATUS UPDATE 2026-11-09 c (RESULTS.md H20; supersedes the batch items below)
MEASURED: batched exact executor, full stack bitwise (30/30 cells; 384/384 output checks per bench): plain agg tok/s B=32 2012 (H19 742, H18 636); step 3.9x (T=1) to 5.9x (T=13) vs the H18 path at B=32. PLD at batch>1 is now near break-even (capped 0.99-1.01x of plain at B>=4); B=1 unchanged (~220-233 tok/s, 1.27-1.29x with PLD).
NEXT: (a) cost model for the cap that accounts for masked rows (time the table with realistic valid fractions); (b) B=1: the remaining single-sequence cost is lin (1.8 ms of 4.4, 150 launches incl. lm_head) + MoE (1.6 ms): fuse qkv/o/router launches or port the 64-row lin tiles' lm_head gain to the B=1 path if bitwise; (c) larger B (64+) and longer contexts for the batched path (only <=32 x <=1.7K prompts measured); (d) ragged attention rows (attention still runs all T rows per sequence; masked rows still cost lin_bt and attention).

Older notes: docs/archive/NEXT_STEPS_pre_2026-11-09.md (historical; RESULTS.md is authoritative).
