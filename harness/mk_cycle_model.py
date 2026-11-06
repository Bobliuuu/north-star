"""Megakernel-executor self-spec cycle model (INFERRED unless an end-to-end loop was run; none was).

Inputs (all MEASURED in-repo):
  results/mk_selfspec_A.jsonl   draft-step ms inside cohere-megakernel (top-r experts, layer-skip), BS=1, ctx 2048, H100
  results/mk_selfspec_B.jsonl   verify-block ms: K tokens emulated as BS=K rows (padded to BS 2/4/8) with the REAL per-layer
                                expert-union size U(K) (results/mk_union_by_K.json from results/overlap_traces.json)
  results/h100_spec.jsonl       measured mean accepted tokens/round of the real draft+verify loop (graphed PyTorch executor)
  results/approx_sweep_quick1.json  teacher-forced acceptance (ctx 2K, 2 prompts) for r2/r4/r6 raw_sigmoid
Round time = gamma * T_draft + T_verify(K=gamma+1); committed/round = mean_accepted + 1; speedup = (committed / round_ms) * T_plain.
Optimistic: ignores host launch gaps between the two kernel launches, KV catch-up of the draft layers, and the argmax/accept logic.
"""
import json, pathlib, statistics as st, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def jl(p):
    return [json.loads(l) for l in open(ROOT / p) if l.strip()]


A = {r["name"]: r["tpot_ms"] for r in jl("results/mk_selfspec_A.jsonl") if r["tpot_ms"]}
B = {r["name"]: r["tpot_ms"] for r in jl("results/mk_selfspec_B.jsonl") if r["tpot_ms"]}
plain = A["base_R8_real"]
# measured acceptance from the real loop (h100_spec, graphed executor): mean accepted per round, r4 and skip8
spec = [r for r in jl("results/h100_spec.jsonl") if r.get("phase") == "g_spec"]


def acc(cfg):
    v = [r["mean_accepted"] for r in spec if r["config"] == cfg]
    return st.mean(v) if v else None


# teacher-forced acceptance, approx sweep (ctx 2K, mean of 2 prompts): expected accepted per round
sw = json.load(open(ROOT / "results/approx_sweep_quick1.json"))["rows"]


def sweep(name, g):
    v = [r["accepted"][str(g)]["mean_accepted"] for r in sw if r["name"] == name and r["ctx"] == 2048]
    return st.mean(v) if v else None


# verify cost V(K): use union-emulated rows when present, else independent-routing rows as pessimistic bound
def V(K, mode="U"):
    keys = {"U": [f"K{K}_U{u}" for u in (12, 15, 17, 19, 21, 23, 24)], "indep": [f"K{K}_U0"], "low": [f"K{K}_U8_lowerbound"]}[mode]
    for k in keys:
        if k in B:
            return B[k]
    if K == 1:
        return plain
    if K == 9 and mode == "U" and V(8) and V(7):
        return V(8) + (V(8) - V(7))   # EXTRAPOLATED (BS=8 is the largest megakernel batch)
    if K == 9 and mode == "low" and V(8):
        return V(8) + 0.12
    return None


def Vint(K):
    """verify cost for K tokens; K not natively measured is taken from the nearest measured K above (monotone upper bound)."""
    for k in range(K, 9):
        v = V(k)
        if v is not None:
            return v
    return None


drafts = {  # name -> (draft ms, acceptance source label, {gamma: mean accepted})
    "top4": (A["R4_real"], "h100_spec r4 (real loop)", {2: acc("r4_g2"), 4: acc("r4_g4"), 8: acc("r4_g8")}),
    "top4_sweep": (A["R4_real"], "approx_sweep raw_sigmoid r4, teacher-forced", {g: sweep("topr:r4:raw_sigmoid", g) for g in (2, 4, 8)}),
    "top2_sweep": (A["R2_real"], "approx_sweep r2", {g: sweep("topr:r2:raw_sigmoid", g) for g in (2, 4, 8)}),
    "skip8_L41(R8)": (A["L41_R8_syn"] * plain / A["base_R8_syn"], "h100_spec skip8 (only gamma=4)", {4: acc("skip8_g4")}),
}
out = []
print(f"plain megakernel step {plain:.3f} ms")
for name, (td, lab, accs) in drafts.items():
    print(f"\n{name}: draft step {td:.3f} ms ({td/plain:.2f} of full), acceptance = {lab}")
    for g, a in sorted(accs.items()):
        if a is None:
            continue
        for label, mode in (("verify@real-union", "U"), ("verify@indep-routing", "indep"), ("verify@lower-bound(union=8)", "low")):
            v = V(g + 1, mode)
            if v is None:
                continue
            rnd = g * td + v
            sp = (a + 1) / rnd * plain
            out.append(dict(draft=name, gamma=g, accepted=a, draft_ms=td, verify_ms=v, verify_basis=label, round_ms=rnd, speedup=sp,
                            label="INFERRED"))
            print(f"  gamma={g} acc={a:.2f} verify(K={g+1}) {v:.2f} ms [{label}] round {rnd:.2f} ms -> speedup {sp:.2f}x")
json.dump(out, open(ROOT / "results/mk_selfspec_model.json", "w"), indent=1)


# ---- G0 inside the megakernel (INFERRED, upper bound): fuse ONE draft step into the verify pass as an extra row.
# Marginal cost of an extra row = V(K+1) - V(K) from the union-emulated BS curve; saving = T_draft - marginal; assumes the fused
# draft token is always usable (i.e. optimistic: a real fused row is wasted whenever an earlier draft token is rejected).
print("\nG0 upper bound inside megakernel: fuse one top-4 draft step as an extra verify row")
g0 = []
for g in (2, 3, 4, 5, 6, 7):
    K = g + 1
    td = A["R4_real"]
    marg = V(K + 1) - V(K)
    rnd = g * td + V(K)
    saved = td - marg
    acc_g = acc(f"r4_g{g}") or {2: 1.68, 4: 2.87, 8: 4.63}.get(g)
    row = dict(gamma=g, round_ms=rnd, marginal_row_ms=marg, saved_ms=saved, saving_frac_of_cycle=saved / rnd, label="INFERRED upper bound")
    if acc_g is not None:
        row["speedup_if_fused"] = (acc_g + 1) / (rnd - saved) * plain
        row["speedup_unfused"] = (acc_g + 1) / rnd * plain
    g0.append(row)
    print(f"  gamma={g}: round {rnd:.2f} ms, marginal row {marg:.2f} ms, saving {saved:.2f} ms = {100*saved/rnd:.1f}% of cycle"
          + (f", speedup {row['speedup_unfused']:.2f}x -> {row['speedup_if_fused']:.2f}x" if "speedup_if_fused" in row else ""))
json.dump(dict(rows=out, g0_fused_bound=g0), open(ROOT / "results/mk_selfspec_model.json", "w"), indent=1)
