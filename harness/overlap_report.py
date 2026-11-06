"""CPU: turn results/overlap_traces.json into results/overlap.jsonl (+ replay input for the H100 microbench)."""
import json, sys, numpy as np, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from overlap import analyze

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRID = ((2, 3), (4, 5), (8, 9), (1, 3), (1, 5), (1, 9))


def main(tr_path=ROOT / "results/overlap_traces.json"):
    D = json.load(open(tr_path))
    out = []
    agg = {}
    for tid, e in D.items():
        full = np.array(e["full"], dtype=np.int64)
        lidx = {l: i for i, l in enumerate(e["layers"])}
        for name, md in e["modes"].items():
            dr = np.array(md["arr"], dtype=np.int64)
            li = [lidx[l] for l in md["layers"]]
            for Td, K in GRID:
                if full.shape[1] < K + Td + 2: continue
                a = analyze(full, dr, li, md["r"], K, Td)
                a.update(task=tid, mode=name, r=md["r"], K=K, Td=Td, n_layers=len(li))
                out.append(a)
                agg.setdefault((name, K, Td), []).append(a)
    with open(ROOT / "results/overlap.jsonl", "w") as f:
        for a in out: f.write(json.dumps(a) + "\n")
    print("mode K Td | pair_in_union uniq_in_union byte_saving | same-token pair uniq byte | shuffled pair uniq byte | |Uv| |Ud|")
    for (name, K, Td), v in sorted(agg.items()):
        mean = lambda k: np.mean([x[k] for x in v])
        print(f"{name:9s} {K:2d} {Td:2d} | {mean('pair_in_union'):.3f} {mean('uniq_in_union'):.3f} {mean('byte_saving'):.3f} | "
              f"{mean('same_pair'):.3f} {mean('same_uniq'):.3f} {mean('same_byte'):.3f} | {mean('shuf_pair'):.3f} {mean('shuf_uniq'):.3f} {mean('shuf_byte'):.3f} | {mean('n_union_v'):.1f} {mean('n_union_d'):.1f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
