"""Cycle-level G0 accounting.  Inputs: results/h100_blocks.jsonl (graphed, measured), results/kvm_replay_H100_*.jsonl (REAL routing).
Saving fraction = (sum over 48 MoE layers of [two_pass_graph - fused_graph]) / cycle time.
Two denominators (both reported):
  A  HF-graph basis: cycle = T_verify_graph(K) + Td_total * T_draft_step_graph(mode)   [measured; includes the HF grouped_mm cost]
  B  microbench basis: replace the HF MoE time by the microbench kernel time (the fused kernel replaces the experts anyway):
     cycle_B = nonMoE_verify(K) + sum_L verify_only_graph + Td_total*(nonMoE_draft_step + mean per-step draft MoE)
Variants: Td=1 'realistic' (draft is sequential, only ONE draft step can share a verify layer pass) vs Td=gamma 'ideal batch'."""
import glob, json, sys, pathlib, collections, statistics as st
ROOT = pathlib.Path(__file__).resolve().parent.parent


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def windows(rows):
    """group consecutive records into windows of one (tag, task, start): new window when layer id does not increase."""
    W, cur, last = [], [], None
    for r in rows:
        key = (r["tag"], r["task"])
        if cur and (key != last[0] or r["layer"] <= last[1]):
            W.append(cur); cur = []
        cur.append(r); last = (key, r["layer"])
    if cur: W.append(cur)
    return W


def main():
    blk = load(ROOT / "results/h100_blocks.jsonl")
    rep = load(sorted(glob.glob(str(ROOT / "results/kvm_replay_H100_*.jsonl")))[-1])
    ctx = sorted({b["ctx"] for b in blk if b["phase"] == "g_block"})[0]
    gb = {b["K"]: b for b in blk if b["phase"] == "g_block" and b["ctx"] == ctx}
    comp = {c["K"]: c for c in blk if b_ok(c, ctx)}
    dstep = {(d["mode"], d["T"]): d["graph_ms"] for d in blk if d["phase"] == "g_draft_step" and d["ctx"] == ctx}
    out = []
    print(f"ctx {ctx}; graph T1 {gb[1]['graph_ms']:.3f} ms")
    by_tag = collections.defaultdict(list)
    for w in windows(rep): by_tag[w[0]["tag"]].append(w)
    for tag, ws in sorted(by_tag.items()):
        K = ws[0][0]["Tv"]; Td = max(r["Td"] for r in ws[0]); mode = tag.split("_")[0]
        tdtot = K - 1 if Td == 1 else Td  # drafts per cycle (gamma)
        saved = [sum(r["v"]["two_pass_graph"]["us_med"] - r["v"]["fused_graph"]["us_med"] for r in w) for w in ws]
        two = [sum(r["v"]["two_pass_graph"]["us_med"] for r in w) for w in ws]
        ver = [sum(r["v"]["verify_only_graph"]["us_med"] for r in w) for w in ws]
        dr = [sum(r["v"]["draft_only_graph"]["us_med"] for r in w) for w in ws]
        s_ms = st.mean(saved) / 1e3; two_ms = st.mean(two) / 1e3; ver_ms = st.mean(ver) / 1e3; dr_ms = st.mean(dr) / 1e3
        # basis A
        Tver = gb[K]["graph_ms"] if K in gb else None
        dname = "r4" if mode == "r4" else "skip8"
        Tdr = dstep.get((dname, 1))
        cycA = Tver + tdtot * Tdr
        # basis B: non-MoE part of verify at K = full - experts (measured components nearest K); draft step non-MoE ~ comp at K=1
        cK = comp.get(K) or comp.get(min(comp, key=lambda k: abs(k - K)))
        c1 = comp[1]
        nonmoe_v = cK["full_ms"] - cK["experts_ms"]
        nonmoe_d = (c1["full_ms"] - c1["experts_ms"]) * (1.0 if mode == "r4" else (49 - 8) / 49)
        draft_moe_per_step = dr_ms / max(1, Td) if Td else 0   # microbench draft-only pass for Td tokens, per token-step approx
        cycB = nonmoe_v + ver_ms + tdtot * (nonmoe_d + (dr_ms if Td == 1 else dr_ms / Td))
        row = dict(tag=tag, mode=mode, K=K, Td_fused=Td, gamma=tdtot, n_windows=len(ws), moe_saved_ms=s_ms, moe_two_pass_ms=two_ms,
                   moe_layer_saving_frac=s_ms / two_ms, cycleA_ms=cycA, savingA=s_ms / cycA, cycleB_ms=cycB, savingB=s_ms / cycB,
                   verify_graph_ms=Tver, draft_step_ms=Tdr)
        # ---- basis C (INFERRED, HBM-byte model; time ~ bytes) from the REAL per-layer expert counts in the replay records ----
        EXP = 3 * 2048 * 768 * 2  # bytes per expert
        NONEXP = 2.97e9           # attention + dense + embed/lm_head weights streamed per full step (derived, docs/NORTH_ARCH.md)
        nonexp_d = NONEXP if mode == "r4" else NONEXP - (8 / 49) * 1.86e9
        r_d = ws[0][0]["r"]
        Ld = 48 if mode == "r4" else 40
        ev = st.mean(sum(x["experts_verify"] for x in w) for w in ws)
        ed = st.mean(sum(x["experts_draft"] for x in w) for w in ws)
        eu = st.mean(sum(x["experts_union"] for x in w) for w in ws)
        single_draft_exp = r_d * Ld                      # one draft token alone: r distinct experts per kept layer
        if Td == 1:   # sequential draft: gamma separate steps; ONE shares the verify pass
            base = NONEXP + ev * EXP + tdtot * (nonexp_d + single_draft_exp * EXP)
            save_moe = (ev + single_draft_exp - eu) * EXP
        else:         # ideal batch: all Td draft tokens in one pass (needs a non-sequential draft)
            base = NONEXP + ev * EXP + (nonexp_d + ed * EXP)
            save_moe = (ev + ed - eu) * EXP
        row.update(bytes_cycle_GB=base / 1e9, savingC_moe_only=save_moe / base, savingC_plus_nonexpert_share=(save_moe + nonexp_d) / base)
        out.append(row)
        print(f"{tag:14s} K={K} Tdf={Td} g={tdtot} win={len(ws):3d} | MoE-layer-sum saving {100*s_ms/two_ms:5.1f}% ({s_ms:.3f} ms) | cycle A {cycA:.2f} ms -> {100*s_ms/cycA:5.2f}% | cycle B {cycB:.2f} ms -> {100*s_ms/cycB:5.2f}% | bytes C {row['bytes_cycle_GB']:.1f} GB -> MoE-only {100*row['savingC_moe_only']:.1f}%  +nonexp {100*row['savingC_plus_nonexpert_share']:.1f}%")
    json.dump(out, open(ROOT / "results/g0_cycle.json", "w"), indent=1)


def b_ok(c, ctx):
    return c["phase"] == "g_components" and c["ctx"] == ctx


if __name__ == "__main__":
    main()
