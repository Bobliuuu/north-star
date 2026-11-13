"""Derive README / video numbers from raw files in results/. Writes results/derived/readme_data.json."""
import json, os, collections
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
L = lambda f: [json.loads(l) for l in open(os.path.join(R, f))]
out = {}

# H29: one launch, four decoders, graph ms per token at ctx 2048 / 8192
rows = [r for r in L("headline_hl1.jsonl") if r.get("phase") == "headline"]
name = {"base": "PyTorch baseline (graphed)", "fast": "Triton executor (H9, not exact)", "exact_v1": "Exact v1", "exact_v2": "Exact v2"}
base = {r["ctx"]: r["graph_ms"] for r in rows if r["decoder"] == "base"}
out["headline"] = [{"ctx": f"ctx {r['ctx']}", "decoder": name[r["decoder"]], "ms": r["graph_ms"], "x": base[r["ctx"]] / r["graph_ms"],
                    "live_tok_s": r["live_tok_s"]} for r in rows]
out["v2"] = {str(r["ctx"]): {"ms": r["graph_ms"], "x": base[r["ctx"]] / r["graph_ms"], "tok_s": r["live_tok_s"]} for r in rows if r["decoder"] == "exact_v2"}
out["base_ms_2048"] = base[2048]

# batched aggregate throughput vs stock vLLM, same container (H25)
def mean(xs): return sum(xs) / len(xs)
ours = collections.defaultdict(list); vl = collections.defaultdict(list)
for f in ("vxb_vxb1_ours1.jsonl", "vxb_vxb1_ours2.jsonl"):
    for r in L(f):
        if r.get("phase") == "batch": ours[int(r["B"])].append(r["agg_tok_s"])
for r in L("vxb_vxb1_vllm.jsonl"):
    if r.get("phase") == "batch": vl[int(r["B"])].append(r["decode_agg_tok_s"])
bs = sorted(set(ours) & set(vl))
out["batched"] = [{"users": f"{b}", "impl": "Northstar exact v2", "tok_s": mean(ours[b])} for b in bs] + \
                 [{"users": f"{b}", "impl": "vLLM 0.33", "tok_s": mean(vl[b])} for b in bs]
out["batched_ratio"] = [{"users": b, "ours": mean(ours[b]), "vllm": mean(vl[b]), "ratio": mean(ours[b]) / mean(vl[b])} for b in bs]
out["b32"] = out["batched_ratio"][-1]
out["b1"] = out["batched_ratio"][0]
out["scale"] = mean(ours[32]) / mean(ours[1])

# H23: embedded workers inside frozen North, accuracy by number of entities (chance 0.10)
emb = json.load(open(os.path.join(R, "embedded_main_20261112T144436Z.json")))
modes = collections.defaultdict(list)
for r in emb: modes[r["mode"]].append(r["final"])
lab = {"none": "frozen North, nothing added", "adapter": "adapter (stateless)", "single": "single shared state", "multi": "multi (private slots + router)"}
out["embedded"] = [{"mode": lab[m], "seeds": len(v), **{p: mean([x[p] for x in v]) for p in ("P2", "P4", "P8")}} for m, v in modes.items() if m in lab]
out["embedded_bars"] = [{"entities": p, "mode": lab[m], "acc": mean([x[p] for x in modes[m]])} for p in ("P2", "P4", "P8") for m in ("none", "adapter", "multi")]
out["emb_none_p4"] = mean([x["P4"] for x in modes["none"]]); out["emb_adapter_p4"] = mean([x["P4"] for x in modes["adapter"]])
out["emb_multi_p4"] = mean([x["P4"] for x in modes["multi"]]); out["emb_multi_p8"] = mean([x["P8"] for x in modes["multi"]])
out["emb_none_p8"] = mean([x["P8"] for x in modes["none"]])
os.makedirs(os.path.join(R, "derived"), exist_ok=True)
json.dump(out, open(os.path.join(R, "derived", "readme_data.json"), "w"), indent=1)
print(json.dumps({k: out[k] for k in ("v2", "b32", "b1", "scale", "embedded")}, indent=1))
