| Metric | Gain | Win? | Numbers |
|---|---|---|---|
| ⚡ Speed (plain decode) | **3.4x** (exact), up from 2.7x (not exact) | ✅ WIN | one launch, same H100 (H29): 10.70 → 3.12 ms per token at 2K context (93 → 319 tok/s live), 3.42x at 8K; the old non-exact executor re-measures at 2.71x and the old exact one at 2.34x |
| 🎯 Same answers | **1.00x** (no quality lost) | ✅ WIN | identical to the model's own normal decoding on every run checked: 70/70 (H14), 99/99 real prompts (H24), every serving and continuous-batching request (H26-H28) |
| 👥 Many users at once | **7.2x** total throughput | ✅ WIN | 2,219 tokens/s with 32 users together vs 308 tok/s for one user (H24 bs7; was 2,012 in H20) |
| 🏁 vs vLLM (popular tool) | **1.29x** plain, **1.49x** with speculation | ✅ WIN | same H100, same container, 99 real prompts: 321.7 / 369.8 vs 248.8 tok/s, faster on 99/99 prompts (H24; was a tie in H17) |
| 🏁 vs vLLM, many users | **1.02-1.21x** at 1-32 users | ✅ WIN (small at 8+) | same container, same 32 prompts: 2,325 vs 2,217 tok/s at 32 users, 637 vs 528 at 4 (H25) |
| 🚦 vs vLLM, requests arriving over time | **1.04-1.05x** up to 8 req/s, 0.98x at 16, **1.16x** when all arrive at once | ✅ WIN (except ~16 req/s) | continuous batching with batched prefill, same container, 64 requests; answers never depend on other requests (H27, H28) |
| 📜 Long context (8K-128K) | **1.4-2.1x** plain, **2.2x** with speculation at 128K | ✅ WIN | exact plain 90 → 187 tok/s at 128K, 211 → 322 at 32K; with the speculation index built during prefill 203 tok/s at 128K (H24, H25; was the weakest regime) |
| 🤖 North improves its own kernel | **1.056x** best, **1.026x** average | 🟡 PARTIAL | best of 3 sessions; only 1 of 3 sessions clears our 1.03x bar; 4 extra North helper copies called from outside did not change the average (1.026x with or without) |
| 🧬 North EMBEDDED inside North: workers live inside one forward pass (qwenagent-style) | **6.6x** accuracy at 4 entities (0.14 → 0.92); **6.0x** at 8 (0.14 → 0.85) | ✅ WIN (capability) / ❌ no edge over simpler adapters | a different task from the kernel test: tracking who owns what across a text. Frozen North is at chance; embedded workers get it to 0.85-0.99. The "private slots" version is 0.94x of a plain adapter (0.92 vs 0.98 at 4 entities). See H23 |

Legend: ✅ clear win · 🟡 small or partial win · ➖ tie · ❌ no win. The x is the ratio against the stated baseline (higher is better; 1.00x = no change).

### What that means
| Metric | In plain English |
|---|---|
| ⚡ Speed | Replies show up roughly 3.4x sooner. Same hardware, same model, same answers. |
| 🎯 Same answers | The speed-up is a free lunch: no quality traded away. |
| 👥 Many users at once | One GPU can serve far more people for the same cost. |
| 🏁 vs vLLM | On the same GPU we are now clearly faster than the industry standard for one user, and level-to-ahead of it with up to 32 users, with identical answers to our own normal decoding. |
| 📜 Long context | Long documents (up to 128K tokens) are now 1.4-2.1x faster too. |
| 🤖 Self-improvement | Yes, a little, and only sometimes. A human found the same size of win (+5.4-6.2%) with a simple sweep. Not a self-improvement explosion, but a real first sign. |
| 🧬 Embedded North-in-North | We really did put small trainable workers inside North's forward pass. North went from guessing to mostly right at tracking several things in a story, so embedding works. But the fancy "separate private workers" design was no better than a simple add-on, same as we saw on Qwen. Note this tests tracking ability, not making North faster. |

**Honest bottom line:** ✅ big, verified speed-ups for running North, now ahead of stock vLLM on the same H100 for one user, for static batches up to 32, and under most arrival rates we tested. ❌ no evidence yet that the model can make itself dramatically faster. Details, failures and caveats below. 👇
