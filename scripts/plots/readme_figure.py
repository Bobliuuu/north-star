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

a = ax[0, 0]  # H9
v = [10.73, 3.9]
b = a.bar(["graphed PyTorch\nbaseline", "Triton executor"], v, color=[GREY, BLUE], width=0.55)
a.bar_label(b, fmt="%.1f ms", padding=3)
a.set_ylabel("ms per token (plain decode)"); a.set_ylim(0, 13)
a.set_title("A. Fast decode executor (H9)")

a = ax[0, 1]  # H14 / H15 / H16
lab = ["natural\n(21 tasks)", "ctx 2K", "ctx 8K", "real mix\n(99 prompts)", "ctx 32K", "ctx 128K"]
val = [1.18, 1.57, 1.34, 1.15, 1.03, 0.97]
col = [BLUE] * 4 + [GREY] * 2
b = a.bar(lab, val, color=col, width=0.6)
a.bar_label(b, fmt="%.2fx", padding=3)
a.axhline(1, color="k", lw=0.8)
a.set_ylabel("tok/s vs same executor's plain decode"); a.set_ylim(0, 1.85)
a.set_title("B. Exact-mode speculation (H14, H15, H16)")
a.tick_params(axis="x", labelsize=8)

a = ax[1, 0]  # H20 bs6 plain
B = [1, 2, 4, 8, 16, 32]; t = [218, 314, 514, 822, 1314, 2012]
a.plot(B, t, marker="o", color=GREEN, lw=2)
for x, y in zip(B, t):
    a.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
a.set_xscale("log", base=2); a.set_xticks(B); a.set_xticklabels(B)
a.set_xlabel("concurrent sequences"); a.set_ylabel("aggregate tok/s"); a.set_ylim(0, 2300)
a.set_title("C. Batched exact executor (H20)")

a = ax[1, 1]  # H17 / H15 mix99
lab = ["vLLM 0.33\nplain", "vLLM\nngram (best)", "ours exact\nplain", "ours exact\n+ suffix spec"]
val = [253.5, 243.2, 221.4, 254.8]
b = a.bar(lab, val, color=[GREY, GREY, BLUE, BLUE], width=0.6)
a.bar_label(b, fmt="%.0f", padding=3)
a.set_ylabel("decode tok/s, geomean, 99-prompt mix"); a.set_ylim(0, 300)
a.set_title("D. Against stock vLLM, batch 1 (H17)")
a.text(0.5, -0.27, "Separate launches; cross-system ratios carry about +-10% host noise.", transform=a.transAxes,
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
