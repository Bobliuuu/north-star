"""Summarize results/kvm_fused_*.jsonl: fusion saving vs two-pass (time, bytes), BW, graph variants. Usage: summarize_kvm.py file.jsonl"""
import json, sys, collections, statistics as st
rows = [json.loads(l) for l in open(sys.argv[1])]
def agg(key, rs):
    out = {}
    for v in ("verify_only", "two_pass", "fused", "two_pass_graph", "fused_graph"):
        out[v] = st.median(r["v"][v]["us_med"] for r in rs)
    sv = [1 - r["v"]["fused"]["us_med"] / r["v"]["two_pass"]["us_med"] for r in rs]
    sg = [1 - r["v"]["fused_graph"]["us_med"] / r["v"]["two_pass_graph"]["us_med"] for r in rs]
    sb = [1 - r["v"]["fused"]["weight_bytes"] / r["v"]["two_pass"]["weight_bytes"] for r in rs]
    ov = st.mean(r["draft_expert_overlap"] for r in rs)
    bw = st.median(r["v"]["fused"]["GBs"] for r in rs); bw2 = st.median(r["v"]["two_pass"]["GBs"] for r in rs)
    dr = [(r["v"]["fused"]["us_med"] - r["v"]["verify_only"]["us_med"]) / r["v"]["verify_only"]["us_med"] for r in rs]
    return dict(n=len(rs), time_save_med=st.median(sv), time_save_graph_med=st.median(sg), bytes_save_med=st.median(sb), overlap=ov,
                GBs_fused=bw, GBs_two=bw2, draft_marginal_cost_med=st.median(dr), us=out)
def show(title, groups):
    print("##", title)
    for k, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        a = agg(k, rs)
        print(f"{str(k):28s} n={a['n']:4d} ovl={a['overlap']:.2f} time_save={a['time_save_med']*100:5.1f}% graph_save={a['time_save_graph_med']*100:5.1f}% "
              f"bytes_save={a['bytes_save_med']*100:5.1f}% GB/s fused={a['GBs_fused']:.0f} two={a['GBs_two']:.0f} draft_marginal={a['draft_marginal_cost_med']*100:5.1f}%")
print("peak read GB/s", rows[0]["peak_read_GBs"], rows[0]["gpu"])
g = collections.defaultdict(list)
for r in rows: g[r["routing"]].append(r)
show("by routing", g)
for rt in sorted(g):
    g2 = collections.defaultdict(list)
    for r in g[rt]: g2[("r", r["r"])].append(r)
    show(f"routing={rt} by r", g2)
    g3 = collections.defaultdict(list)
    for r in g[rt]: g3[("Tv", r["Tv"])].append(r)
    show(f"routing={rt} by Tv", g3)
    g4 = collections.defaultdict(list)
    for r in g[rt]: g4[("Td", r["Td"])].append(r)
    show(f"routing={rt} by Td", g4)
bad = [r for r in rows if r["fused_vs_two_maxabsdiff"] > 1e-3 * max(r["out_maxabs"], 1e-9)]
print("fused-vs-two mismatches:", len(bad), "of", len(rows))
