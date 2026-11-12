# Thread draft: North inside North (H22)

1/ We let North-Mini-Code improve its own MoE kernel, three ways, same budget (24 measured builds on an A100):
- a plain autotuner
- North alone
- North orchestrating 4 North subagents (only North, no other model)

2/ Open loop (no feedback) North found nothing: 0 of 13 edit sets passed our 1.03x gate. Random search and GPT-4.1/5-mini did no better.

3/ Closed loop (measured speedups fed back each round, 3 sessions per arm):
autotuner: 1.002x best, no pass
North alone: 1.018 / 1.018 / 1.041x, 1 of 3 sessions pass
North + North subagents: 1.056 / 1.006 / 1.017x, 1 of 3 pass

4/ Honest read: subagents did not help on average (both arms 1.026x mean best, n=3). The one standout: a North-in-North session found KC=64, the same move our human sweep found (1.056x vs human 1.054-1.062x). No earlier North run did.

5/ Caveats: tiny n, ~4% timing noise in the feedback loop, North-in-North uses ~1.4x the model calls. Everything correctness-checked, raw traces for all 266 North calls in the repo.
Code + results: github.com/Bobliuuu/north-star (H22)

[attach docs/img/north_in_north.png to tweet 3]
