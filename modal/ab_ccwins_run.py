"""A/B of moe_fused's per-arch codegen defaults (sass-lane compiler sweep wins) against the pre-sweep build, per chip.

  modal run --detach modal/ab_ccwins_run.py::main --gpus H100,B200,RTX-PRO-6000 --tag cw1
  modal volume get ship-lane cw1 results/ccwins_cw1

Per chip, in one container, both builds from the same source (`-DNS_TUNE_UPSTREAM` = pre-sweep values):
  1. correctness: tests/check_cuda.py (fused and two-pass vs the PyTorch CPU reference) for each build
  2. kernel: `bench` on the sweep's shapes, BLOCKS interleaved blocks (base, new, base, new, ...), fused_graph us_med
     per shape; block ratio = geomean(base)/geomean(new); spread = max - min of block ratios.
     Claimed only if (median ratio - 1) > 2 * spread.
  3. workload: `replay` on REAL North routing (harness/make_replay.py, an even sample of MAXREC records), two
     interleaved rounds per build, paired per record on fused_graph us_med; per-record quantiles, the share of
     records that got slower, and the time-weighted ratio with its spread over the rounds.
Every step writes its JSON to the volume and commits, so a killed container keeps what it measured.
"""
import json
import os
import pathlib
import modal

HERE = pathlib.Path(__file__).resolve().parent.parent
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("numpy", "torch==2.8.0", index_url="https://download.pytorch.org/whl/cpu")
    .add_local_dir(HERE / "src" / "kvm", "/root/kvm", ignore=["replay_input.txt", "**/__pycache__/**"])
    .add_local_file(HERE / "harness" / "make_replay.py", "/root/harness/make_replay.py")
    .add_local_file(HERE / "results" / "overlap_traces.json", "/root/results/overlap_traces.json")
)
app = modal.App("ship-moefused-ab", image=image)
vol = modal.Volume.from_name("ship-lane", create_if_missing=True)
SHAPES = "1,1,8;4,4,8;8,8,4;16,16,8"
BLOCKS = 6
MAXREC = 2400


def _sh(cmd, **kw):
    import subprocess
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return r.returncode, r.stdout, r.stderr


def _job(tag, chip, sm):
    import statistics
    import subprocess
    import sys
    import time
    out = pathlib.Path(f"/vol/{tag}/{chip}")
    out.mkdir(parents=True, exist_ok=True)
    res = {"chip": chip, "sm": sm, "smi": _sh(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])[1].strip()}

    def save():
        (out / "result.json").write_text(json.dumps(res, indent=1))
        vol.commit()

    t0 = time.time()
    exes = {}
    for name, extra in (("base", ["-DNS_TUNE_UPSTREAM"]), ("new", [])):
        exe = f"/root/ns_{name}"
        rc, o, e = _sh(["nvcc", "-O3", "-std=c++17", f"-arch=sm_{sm}", "-Xptxas", "-v", "-I", "/root/kvm/include", *extra,
                        "/root/kvm/moe_fused.cu", "-o", exe])
        regs = [l for l in e.splitlines() if "moe_kernel" in l or "registers" in l]
        res.setdefault("build", {})[name] = {"rc": rc, "ptxas": regs[-6:], "err": e[-1500:] if rc else ""}
        if rc:
            save()
            return res
        exes[name] = exe
    save()

    # 1. correctness vs the CPU reference
    for name, exe in exes.items():
        rc, o, e = _sh([sys.executable, "/root/kvm/tests/check_cuda.py", exe, f"/tmp/chk_{name}"], cwd="/root/kvm")
        res.setdefault("check", {})[name] = {"rc": rc, "verdict": "PASS" if rc == 0 else "FAIL", "tail": (o + e)[-1200:]}
    save()
    if any(v["rc"] for v in res["check"].values()):
        return res

    # 2. kernel bench, interleaved blocks
    per = {"base": [], "new": []}
    for b in range(BLOCKS):
        for name in (("base", "new") if b % 2 == 0 else ("new", "base")):
            jl = f"/tmp/bench_{name}_{b}.jsonl"
            rc, o, e = _sh([exes[name], "bench", jl, "--shapes", SHAPES, "--reps", "30"], timeout=900)
            if rc:
                res["bench_err"] = (o + e)[-1500:]
                save()
                return res
            rows = [json.loads(l) for l in open(jl)]
            per[name].append([r["v"]["fused_graph"]["us_med"] for r in rows])
    geo = lambda xs: statistics.geometric_mean(xs)
    ratios = [geo(bb) / geo(nn) for bb, nn in zip(per["base"], per["new"])]
    med = statistics.median(ratios)
    spread = max(ratios) - min(ratios)
    res["kernel"] = {"shapes": SHAPES, "base_us_per_shape": [statistics.median(c) for c in zip(*per["base"])],
                     "new_us_per_shape": [statistics.median(c) for c in zip(*per["new"])], "block_ratios": ratios,
                     "speedup_median": med, "spread": spread, "passes_2x_spread": (med - 1) > 2 * spread}
    save()

    # 3. real-routing replay
    pathlib.Path("/root/src/kvm").mkdir(parents=True, exist_ok=True)
    rc, o, e = _sh([sys.executable, "/root/harness/make_replay.py"], cwd="/root")
    if rc:
        res["replay_err"] = (o + e)[-1500:]
        save()
        return res
    inp = pathlib.Path("/root/src/kvm/replay_input.txt")
    lines, recs, i = inp.read_text().splitlines(), [], 0
    while i < len(lines):
        hdr = lines[i].split()
        k = 1 + int(hdr[2]) + int(hdr[3])
        recs.append(lines[i:i + k])
        i += k
    stride = max(1, len(recs) // MAXREC)
    pick = recs[::stride][:MAXREC]                      # even sample over every prompt, layer and config
    small = "/tmp/replay_small.txt"
    pathlib.Path(small).write_text("\n".join(l for r in pick for l in r) + "\n")
    rounds = []
    for rd, order in enumerate((("base", "new"), ("new", "base"))):
        rep = {}
        for name in order:
            jl = f"/tmp/replay_{name}_{rd}.jsonl"
            rc, o, e = _sh([exes[name], "replay", small, jl, "--reps", "9"], timeout=2400)
            if rc:
                res["replay_err"] = (o + e)[-1500:]
                save()
                return res
            rep[name] = [json.loads(l) for l in open(jl)]
        pr = [a["v"]["fused_graph"]["us_med"] / b["v"]["fused_graph"]["us_med"] for a, b in zip(rep["base"], rep["new"])]
        tb = sum(a["v"]["fused_graph"]["us_med"] for a in rep["base"])
        tn = sum(b["v"]["fused_graph"]["us_med"] for b in rep["new"])
        rounds.append({"per_record_speedup_median": statistics.median(pr), "per_record_p10": sorted(pr)[len(pr) // 10],
                       "per_record_p90": sorted(pr)[9 * len(pr) // 10], "frac_records_slower": sum(x < 1 for x in pr) / len(pr),
                       "time_weighted_speedup": tb / tn, "base_total_us": tb, "new_total_us": tn})
        res["replay"] = {"records": len(pick), "of_records": len(recs), "rounds": rounds}
        save()
    tw = [r["time_weighted_speedup"] for r in rounds]
    res["replay"]["time_weighted_speedup_median"] = statistics.median(tw)
    res["replay"]["time_weighted_spread"] = max(tw) - min(tw)
    res["secs"] = time.time() - t0
    save()
    return res


@app.function(gpu="H100", timeout=3 * 3600, volumes={"/vol": vol})
def h100(tag: str):
    return _job(tag, "h100", "90")


@app.function(gpu="B200", timeout=3 * 3600, volumes={"/vol": vol})
def b200(tag: str):
    return _job(tag, "b200", "100")


@app.function(gpu="RTX-PRO-6000", timeout=3 * 3600, volumes={"/vol": vol})
def rtxpro(tag: str):
    return _job(tag, "rtxpro6000", "120")


@app.local_entrypoint()
def main(gpus: str = "H100,B200,RTX-PRO-6000", tag: str = "cw1"):
    fns = {"H100": h100, "B200": b200, "RTX-PRO-6000": rtxpro}
    calls = [(g, fns[g].spawn(tag)) for g in gpus.split(",")]
    for g, c in calls:
        r = c.get()
        print(g, json.dumps({k: r.get(k) for k in ("check", "kernel", "replay", "secs")}, default=str)[:3000], flush=True)
