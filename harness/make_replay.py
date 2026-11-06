"""CPU: build the replay input for `moe_fused replay` from real North traces (results/overlap_traces.json).
One record per (task, window start, layer, config). Verify = tokens s..s+K-1 native top-8; draft = FUTURE tokens s+K..s+K+Td-1 under
the draft mode (r = number of experts used; skip modes have no draft at skipped layers -> record with Td=0 = verify-only)."""
import json, sys, pathlib, numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
CFGS = [("r4", 4), ("skip8", 8)]
GRID = ((2, 3), (4, 5), (8, 9), (1, 3), (1, 5), (1, 9))  # (Td, K); Td=1 = only ONE draft step shares the verify pass (sequential-draft realistic case)
WINDOWS_PER_TASK = 2


def main(maxrec=None):
    D = json.load(open(ROOT / "results/overlap_traces.json"))
    out = open(ROOT / "src/kvm/replay_input.txt", "w"); n = 0
    meta = []
    for ti, (tid, e) in enumerate(D.items()):
        full = np.array(e["full"]); T = full.shape[1]
        for name, _ in CFGS:
            md = e["modes"][name]; dr = np.array(md["arr"]); dl = {l: i for i, l in enumerate(md["layers"])}; r = md["r"]
            for Td, K in GRID:
                span = T - K - Td
                if span < 1: continue
                starts = sorted(set(int(x) for x in np.linspace(0, span, WINDOWS_PER_TASK)))
                for s in starts:
                    for li, l in enumerate(e["layers"]):
                        has = l in dl
                        td = Td if has else 0
                        out.write(f"{name}_k{K}_td{Td} {r} {K} {td} {l} {ti}\n")
                        for row in full[li, s:s + K]: out.write(" ".join(map(str, row)) + "\n")
                        if has:
                            for row in dr[dl[l], s + K:s + K + Td]: out.write(" ".join(map(str, row)) + "\n")
                        n += 1
    out.close(); print("records", n)


if __name__ == "__main__":
    main()
