"""README summary figure. Numbers are copied from RESULTS.md (section in each panel title); no new measurement.
Run: python scripts/plots/readme_figure.py  ->  docs/img/summary.png"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "img", "summary.png")
BLUE, GREY, ORANGE, GREEN = "#2a6fdb", "#8d96a3", "#e0792b", "#1b7f5c"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left"})
fig, ax = plt.subplots(2, 2, figsize=(11, 7.2), dpi=150)

a = ax[0, 0]  # H9 / H24
v = [10.73, 3.9, 4.52, 3.05]
b = a.bar(["graphed PyTorch\nbaseline (H2)", "Triton executor\n(H9, not exact)", "exact v1\n(H14)", "exact v2\n(H24)"], v,
          color=[GREY, GREY, BLUE, GREEN], width=0.6)
a.bar_label(b, fmt="%.2f ms", padding=3)
a.set_ylabel("ms per token (plain decode, graph step)"); a.set_ylim(0, 13)
a.set_title("A. Decode step on one H100 (H9, H24)")
a.tick_params(axis="x", labelsize=8)

a = ax[0, 1]  # H24 long context: exact v1 vs exact v2 plain, same launch per context (le3 8K, le6 32K, le5 128K)
lab = ["ctx 8K", "ctx 32K", "ctx 128K"]
v1, v2 = [214.2, 211.0, 90.1], [298.1, 322.1, 186.6]
xs = range(len(lab))
b1 = a.bar([x - 0.18 for x in xs], v1, width=0.36, color=BLUE, label="exact v1")
b2 = a.bar([x + 0.18 for x in xs], v2, width=0.36, color=GREEN, label="exact v2")
a.bar_label(b1, fmt="%.0f", padding=2, fontsize=8); a.bar_label(b2, fmt="%.0f", padding=2, fontsize=8)
for x, p, q in zip(xs, v1, v2):
    a.annotate("%.2fx" % (q / p), (x + 0.18, q), textcoords="offset points", xytext=(0, 14), ha="center", fontsize=8, color=GREEN, fontweight="bold")
a.set_xticks(list(xs)); a.set_xticklabels(lab)
a.set_ylabel("plain decode tok/s (exact)"); a.set_ylim(0, 400)
a.legend(frameon=False, loc="upper right", fontsize=8)
a.set_title("B. Long context, exact plain decode (H24)")

a = ax[1, 0]  # H20 bs6 plain, H24 bs7 plain
B = [1, 2, 4, 8, 16, 32]; t = [218, 314, 514, 822, 1314, 2012]; t2 = [308, 431, 676, 984, 1508, 2219]
a.plot(B, t, marker="o", color=GREY, lw=2, label="H20 (bs6)")
a.plot(B, t2, marker="o", color=GREEN, lw=2, label="exact v2, H24 (bs7)")
for x, y in zip(B, t2):
    a.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
a.set_xscale("log", base=2); a.set_xticks(B); a.set_xticklabels(B)
a.set_xlabel("concurrent sequences"); a.set_ylabel("aggregate tok/s (plain, exact)"); a.set_ylim(0, 2500)
a.legend(frameon=False, loc="upper left", fontsize=8)
a.set_title("C. Batched exact executor (H20, H24)")

a = ax[1, 1]  # H24 vx1: same container, back to back
lab = ["vLLM 0.33\nplain", "ours exact v1\nplain", "ours exact v2\nplain", "ours exact v2\n+ suffix spec"]
val = [248.8, 220.8, 321.7, 369.8]
b = a.bar(lab, val, color=[GREY, BLUE, GREEN, GREEN], width=0.6)
a.bar_label(b, fmt="%.0f", padding=3)
a.set_ylabel("decode tok/s, geomean, 99-prompt mix"); a.set_ylim(0, 430)
a.set_title("D. Against stock vLLM, batch 1, same H100 (H24)")
a.text(0.5, -0.27, "One container, run back to back. v2 beats vLLM plain on 99/99 prompts (1.29x; 1.49x with spec).", transform=a.transAxes,
       ha="center", fontsize=7.5, color="#555")
a.tick_params(axis="x", labelsize=8)

fig.suptitle("Northstar on one H100: Cohere MoE North, greedy decoding (all MEASURED)", x=0.01, ha="left",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))


# ---- second figure: self-improvement benchmark (H21 / H21b, launch sib2) ----
fig2, a = plt.subplots(figsize=(8.5, 3.9), dpi=150)
arms = [("human sweep\n(5 sets)", 1.054, GREEN), ("random\n(66 draws)", 1.052, GREY), ("gpt-5-mini\n(33)", 1.014, GREY),
        ("North\n(45 proposals)", 1.003, ORANGE), ("gpt-4.1\n(33)", 1.001, GREY), ("gpt-4.1-mini\n(33)", 0.999, GREY)]
b = a.bar([x[0] for x in arms], [x[1] for x in arms], color=[x[2] for x in arms], width=0.6)
a.bar_label(b, fmt="%.3fx", padding=3)
a.axhline(1.03, color="k", ls="--", lw=0.9); a.text(5.45, 1.0315, "gate: 1.03x", ha="right", fontsize=8)
a.axhline(1.0, color="k", lw=0.8)
a.set_ylim(0.97, 1.075); a.set_ylabel("best dev speedup vs baseline")
a.set_title("Self-improvement on the kernel-tunable task: best edit set per proposer (H21, H21b)")
a.tick_params(axis="x", labelsize=8)
fig2.text(0.01, -0.02, "Same task, gates and A100, measured in one interleaved launch. Only the human sweep and 1 of 66 random draws pass all gates.",
          fontsize=7.5, color="#555")
fig2.tight_layout()
fig2.savefig(os.path.join(os.path.dirname(OUT), "selfimprove.png"), bbox_inches="tight")
print("wrote docs/img/selfimprove.png")
