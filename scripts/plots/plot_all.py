"""Regenerate every plot from raw results/ files into results/plots/.  Run: python scripts/plots/plot_all.py"""
import os, sys, collections
sys.path.insert(0, os.path.dirname(__file__))
from _common import *

plots = []


def plot(f):
    plots.append(f)
    return f


@plot
def kvm_a100_synthetic():
    p = glob1("kvm_fused_A100-80GB_fullsweep_", ".jsonl")
    rows = jsonl(p)
    if rows is None:
        return skip("kvm_a100_synthetic", "results/kvm_fused_A100-80GB_fullsweep_*.jsonl")
    by = collections.defaultdict(list)
    for r in rows:
        v = r["v"]
        by[r["routing"]].append(1 - v["fused_graph"]["us_med"] / v["two_pass_graph"]["us_med"])
    pref = ("uniform", "corr75", "identical")
    order = [k for k in pref if k in by] + sorted(k for k in by if k not in pref)
    fig, ax = new_fig("Fused vs two-pass MoE layer, A100-80GB, SYNTHETIC routing (1 layer, 1 seed)", "MEASURED", os.path.basename(p))
    ax.boxplot([[100 * x for x in by[k]] for k in order], tick_labels=[f"{k}\n(n={len(by[k])})" for k in order], showfliers=False)
    ax.axhline(15, color="k", ls="--", lw=0.8)
    ax.text(0.55, 15.8, "15% (layer-level only; gate G0 is on cycle time)", fontsize=7)
    ax.set_ylabel("graph-captured time saving, %")
    save(fig, "kvm_a100_synthetic_routing")


@plot
def kvm_h100_real():
    p = glob1("kvm_replay_H100_real_", ".jsonl")
    rows = jsonl(p)
    if rows is None:
        return skip("kvm_h100_real", "results/kvm_replay_H100_real_*.jsonl")
    t = collections.defaultdict(list)
    b = collections.defaultdict(list)
    for r in rows:
        v = r["v"]
        t[r["tag"]].append(1 - v["fused_graph"]["us_med"] / v["two_pass_graph"]["us_med"])
        b[r["tag"]].append(1 - v["fused"]["weight_bytes"] / v["two_pass"]["weight_bytes"])
    tags = sorted(t)
    fig, ax = new_fig("Fused vs two-pass MoE, H100, REAL North routing replay (median over layer records)", "MEASURED", os.path.basename(p), w=9)
    x = list(range(len(tags)))
    w = 0.4
    ax.bar([i - w / 2 for i in x], [100 * median(t[k]) for k in tags], w, label="time saving (graph)", color="#2166ac")
    ax.bar([i + w / 2 for i in x], [100 * median(b[k]) for k in tags], w, label="weight-byte saving", color="#999")
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("%")
    ax.legend(fontsize=8)
    ax.text(0.01, 0.93, "time saving > byte saving: part is removed launch/tail, not HBM", transform=ax.transAxes, fontsize=7)
    save(fig, "kvm_h100_real_routing")


@plot
def g0_cycle():
    d = jload(path("g0_cycle.json"))
    if d is None:
        return skip("g0_cycle", "results/g0_cycle.json")
    fig, ax = new_fig("Cycle-level fused saving vs the 15% G0 gate (arithmetic on measured parts)", "INFERRED", "g0_cycle.json", w=9)
    keys = [("savingA", "A graph cycle"), ("savingB", "B microbench MoE"), ("savingC_moe_only", "C bytes, MoE only"),
            ("savingC_plus_nonexpert_share", "C + non-expert share (optimistic)")]
    n = len(d)
    w = 0.2
    for j, (k, lab) in enumerate(keys):
        ax.bar([i + (j - 1.5) * w for i in range(n)], [100 * r[k] for r in d], w, label=lab)
    ax.axhline(15, color="k", ls="--", lw=1)
    ax.text(-0.4, 15.5, "G0 gate 15%", fontsize=7)
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"{r['tag']}\ng={r['gamma']}" for r in d], rotation=60, fontsize=6)
    ax.set_ylabel("% of cycle time")
    ax.legend(fontsize=7)
    save(fig, "g0_cycle_saving")


@plot
def h100_blocks():
    rows = jsonl(path("h100_blocks.jsonl"))
    if rows is None:
        return skip("h100_blocks", "results/h100_blocks.jsonl")
    fig, ax = new_fig("CUDA-graphed PyTorch block verify, H100, batch 1: cost vs 1 token", "MEASURED", "h100_blocks.jsonl (bmm attention)")
    blk = [r for r in rows if r.get("phase") == "g_block" and r.get("attn_impl") == "bmm"]
    for ctx in sorted({r["ctx"] for r in blk}):
        pts = sorted((r["K"], r["cost_vs_one_token"]) for r in blk if r["ctx"] == ctx)
        ax.plot(*zip(*pts), marker="o", label=f"ctx {ctx}")
    ax.set_xlabel("K tokens per forward")
    ax.set_ylabel("block time / 1-token time")
    ax.legend()
    save(fig, "h100_graph_block_cost")


@plot
def h100_spec():
    rows = jsonl(path("h100_spec.jsonl"))
    if rows is None:
        return skip("h100_spec", "results/h100_spec.jsonl")
    by = collections.defaultdict(list)
    for r in rows:
        if r.get("phase") == "g_spec":
            by[r["config"]].append(r["speedup_wall"])
    cfg = sorted(by)
    fig, ax = new_fig("Self-speculation wall speedup vs plain graphed decode, H100 (greedy, 64 tok/task)", "MEASURED", "h100_spec.jsonl")
    for i, c in enumerate(cfg):
        ax.scatter([i] * len(by[c]), by[c], color="#2166ac", s=18)
    ax.bar(range(len(cfg)), [mean(by[c]) for c in cfg], alpha=0.3, color="#2166ac")
    ax.axhline(1, color="k", ls="--")
    ax.set_xticks(range(len(cfg)))
    ax.set_xticklabels(cfg)
    ax.set_ylabel("speedup (x), <1 is slower")
    save(fig, "h100_selfspec_speedup")


@plot
def eager_selfspec():
    rows = jsonl(path("spec.jsonl"))
    if rows is None:
        return skip("eager_selfspec", "results/spec.jsonl")
    by = collections.defaultdict(list)
    acc = collections.defaultdict(list)
    for r in rows:
        if r.get("phase") == "1_selfspec":
            by[r["config"]].append(r["speedup_wall"])
            acc[r["config"]].append(r["mean_committed_per_round"])
    cfg = sorted(by)
    fig, ax = new_fig("Self-spec on real North, EAGER HF/PyTorch A100 (launch-bound; speedup<1 expected)", "MEASURED", "spec.jsonl", w=9)
    ax.bar(range(len(cfg)), [mean(by[c]) for c in cfg], color="#2166ac", alpha=0.7)
    for i, c in enumerate(cfg):
        ax.text(i, mean(by[c]) + 0.01, f"{mean(acc[c]):.1f} tok/rd", ha="center", fontsize=6)
    ax.axhline(1, color="k", ls="--")
    ax.set_xticks(range(len(cfg)))
    ax.set_xticklabels(cfg, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("wall speedup vs plain (x)")
    save(fig, "eager_selfspec_speedup")


@plot
def overlap():
    rows = jsonl(path("overlap.jsonl"))
    if rows is None:
        return skip("overlap", "results/overlap.jsonl")
    g = collections.defaultdict(lambda: ([], []))
    for r in rows:
        k = f"{r['mode']} K{r['K']}/Td{r['Td']}"
        g[k][0].append(r["byte_saving"])
        g[k][1].append(r["shuf_byte"])
    ks = sorted(g)
    fig, ax = new_fig("Expert-byte saving of fusing future draft tokens into a verify block: real vs shuffled control", "MEASURED", "overlap.jsonl", w=9)
    w = 0.4
    ax.bar([i - w / 2 for i in range(len(ks))], [100 * mean(g[k][0]) for k in ks], w, label="real adjacent block")
    ax.bar([i + w / 2 for i in range(len(ks))], [100 * mean(g[k][1]) for k in ks], w, label="distant-block control", color="#999")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(ks, rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("byte saving, %")
    ax.legend(fontsize=8)
    save(fig, "overlap_byte_saving")


@plot
def routing_union():
    rows = jsonl(path("routing.jsonl"))
    if rows is None:
        return skip("routing_union", "results/routing.jsonl")
    u = collections.defaultdict(list)
    for r in rows:
        if r.get("phase") == "route_stats":
            for k, v in r["union_experts_by_K"].items():
                u[int(k)].append(v)
    Ks = sorted(u)
    indep = [128 * (1 - (1 - 8 / 128) ** K) for K in Ks]  # analytic expectation, independent uniform top-8 of 128
    fig, ax = new_fig("Distinct experts per layer in a block of K tokens: real North routing vs independent-uniform analytic", "MEASURED", "routing.jsonl")
    ax.plot(Ks, [mean(u[K]) for K in Ks], "o-", label="measured (15 tasks)")
    ax.plot(Ks, indep, "s--", color="#999", label="independent uniform (analytic reference)")
    ax.set_xlabel("K")
    ax.set_ylabel("distinct experts / layer (of 128)")
    ax.legend(fontsize=8)
    save(fig, "routing_union_by_K")


@plot
def baseline():
    rows = jsonl(path("baseline.jsonl"))
    if rows is None:
        return skip("baseline", "results/baseline.jsonl")
    fig, ax = new_fig("Eager HF decode ms/step vs batch, A100-80GB bf16 (launch-bound; NOT a hardware floor)", "MEASURED", "baseline.jsonl")
    ok = [r for r in rows if r.get("phase") == "0.2_baseline" and r.get("status") == "ok"]
    for ctx in sorted({r["ctx"] for r in ok}):
        pts = sorted((r["batch"], r["decode_ms_median"]) for r in ok if r["ctx"] == ctx)
        ax.plot(*zip(*pts), "o-", label=f"ctx {ctx}")
    ax.set_xlabel("batch")
    ax.set_ylabel("median ms / decode step")
    ax.legend()
    save(fig, "eager_baseline_decode")


@plot
def opt_synthetic():
    d = jload(os.path.join(ROOT, "src", "opt", "results", "regime_scan_SYNTHETIC.json"))
    if d is None:
        return skip("opt_synthetic", "src/opt/results/regime_scan_SYNTHETIC.json")
    rows = [r for r in d["regime_scan"] if "no_spec_tok_s" in r]
    names = ["fixed_baseline", "draft_only", "schedule_only", "joint"]
    fig, ax = new_fig("Joint optimizer on SYNTHETIC (invented) acceptance: modeled tok/s / plain decode", "SIMULATED", "src/opt/results/regime_scan_SYNTHETIC.json", w=9)
    w = 0.2
    for j, n in enumerate(names):
        vals = [r[n]["tok_s"] / r["no_spec_tok_s"] if isinstance(r.get(n), dict) and "tok_s" in r[n] else float("nan") for r in rows]
        ax.bar([i + (j - 1.5) * w for i in range(len(vals))], vals, w, label=n)
    ax.axhline(1, color="k", ls="--")
    ax.set_xlabel("scenario index")
    ax.set_ylabel("modeled tok/s / plain")
    ax.legend(fontsize=7)
    ax.text(0.01, 0.93, "NOT a speedup claim: acceptance is invented", transform=ax.transAxes, fontsize=8, color="#b2182b")
    save(fig, "opt_synthetic_regime_scan")


if __name__ == "__main__":
    failed = 0
    for f in plots:
        try:
            f()
        except Exception as e:  # one broken plot must not hide the rest; exit nonzero at the end
            print(f"[plots] FAIL {f.__name__}: {type(e).__name__}: {e}")
            failed += 1
    sys.exit(1 if failed else 0)
