"""Self-improvement benchmark: base vs human vs North vs budget-matched random edit sets on the 8-tunable moe_fused.cu task.

  modal run --detach modal/selfimprove_bench_run.py --tag sib1        # ALWAYS --detach

All arms are built and benchmarked in ONE launch, in interleaved shuffled passes on the same A100, with the same DEV / HELD-OUT
shape sets and timing method as modal/kernel_sweep.py. Correctness (src/kvm/tests/check_cuda.py) runs for every config that clears
a 1.0 dev speedup. Rows are also printed as `NS_SIB_JSON {...}` so `modal app logs <id>` recovers them if the client dies.
Analysis (CPU): python harness/selfimprove_bench/analyze.py results/selfimprove_bench_<tag>_*.jsonl
"""
import datetime, json, os, random, re, statistics, subprocess, sys, time
import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "harness"))
RATE = 2.50
DEV = [(1, 1, 2), (4, 4, 4), (8, 8, 4), (16, 16, 8)]
HELD = [(3, 5, 3), (6, 9, 5), (12, 7, 6), (13, 3, 7), (5, 11, 3), (9, 6, 5)]
BASE = dict(KC=256, XPAD=8, HPAD=8, MT_SPLIT=16, MINB_PER_SM=1, UNROLL_KS=16, UNROLL_S2=4, UNROLL_NT=1)

image = (modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
         .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
         .pip_install("numpy<2")
         .add_local_dir(os.path.join(ROOT, "src/kvm"), "/work/kvm"))
app = modal.App("northstar-selfimprove-bench", image=image)


def _src(c):
    s = open("/work/kvm/moe_fused.cu").read()
    for k, v in c.items():
        s, n = re.subn(rf"(?m)^(constexpr int {k} = )\d+(;)", rf"\g<1>{v}\g<2>", s)
        assert n == 1, k
    return s


def _build(item):
    cid, c = item
    d = f"/work/b/{cid}"; os.makedirs(d, exist_ok=True)
    open(f"{d}/moe_fused.cu", "w").write(_src(c))
    os.system(f"cp -r /work/kvm/include {d}/include")
    p = subprocess.run(["nvcc", "-O3", "-std=c++17", "-arch=sm_80", "--resource-usage", "-o", f"{d}/bin", f"{d}/moe_fused.cu"], capture_output=True, text=True)
    regs = re.findall(r"Used (\d+) registers", p.stderr)
    spill = re.findall(r"(\d+) bytes spill stores", p.stderr)
    return cid, dict(rc=p.returncode, err=p.stderr[-400:] if p.returncode else "", regs=regs, spill_stores=spill)


def _bench(cid, shapes):
    out = f"/work/b/{cid}/bench.jsonl"
    arg = ";".join(f"{a},{b},{c}" for a, b, c in shapes)
    p = subprocess.run([f"/work/b/{cid}/bin", "bench", out, "--routing", "1", "--shapes", arg, "--reps", "21"], capture_output=True, text=True, timeout=300)
    if p.returncode: return None
    rows = [json.loads(l) for l in open(out) if l.strip()]
    for r in rows:
        if r["out_maxabs"] <= 0 or r["fused_vs_two_maxabsdiff"] / r["out_maxabs"] > 1e-2: return None
    return [r["v"]["fused"]["us_med"] for r in rows]


@app.function(gpu=["A100-80GB", "A100-40GB"], timeout=7000, cpu=8, memory=32768, scaledown_window=2, max_containers=1)
def run(items: list, passes: int = 3):
    """items: [[cid, arm, edits_dict], ...]"""
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    gpu = subprocess.run("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip()
    print("GPU", gpu, flush=True)
    os.makedirs("/work/b", exist_ok=True)
    cfg = {cid: dict(BASE, **e) for cid, arm, e in items}
    res = {cid: dict(arm=arm, edits=e, dev=[], held=[]) for cid, arm, e in items}
    with ThreadPoolExecutor(8) as ex:
        for cid, info in ex.map(_build, list(cfg.items())):
            res[cid]["build"] = info
    nb = sum(1 for r in res.values() if r["build"]["rc"])
    print(f"built {len(res) - nb}/{len(res)} ok, {nb} failed to compile", flush=True)
    t_build = time.time() - t0
    ok = [cid for cid in cfg if res[cid]["build"]["rc"] == 0]
    rng = random.Random(1)
    for ps in range(passes):
        order = ok[:]; rng.shuffle(order)
        for cid in order:
            d = _bench(cid, DEV); h = _bench(cid, HELD)
            if d: res[cid]["dev"].append(d)
            if h: res[cid]["held"].append(h)
        print(f"pass {ps + 1}/{passes} done t={time.time() - t0:.0f}s", flush=True)
    med = lambda L: [statistics.median(col) for col in zip(*L)] if L else None
    base_id = next(cid for cid, arm, e in items if arm.split("|")[0] == "base")
    mean = lambda v: sum(v) / len(v) if v else None
    bd, bh = mean(med(res[base_id]["dev"])), mean(med(res[base_id]["held"]))
    rows = []
    for cid, r in res.items():
        dv, hv = med(r["dev"]), med(r["held"])
        row = dict(cid=cid, arm=r["arm"], edits=r["edits"], build_rc=r["build"]["rc"], build_err=r["build"]["err"], regs=r["build"]["regs"],
                   spill_stores=r["build"]["spill_stores"], dev_mean_us=mean(dv), held_mean_us=mean(hv), n_passes=len(r["dev"]),
                   dev_pass_means=[sum(x) / len(x) for x in r["dev"]])
        row["dev_speedup"] = bd / row["dev_mean_us"] if row["dev_mean_us"] else None
        row["held_speedup"] = bh / row["held_mean_us"] if row["held_mean_us"] and bh else None
        rows.append(row)
    # correctness only where it can matter: anything not slower than baseline on dev
    for row in rows:
        if row["dev_speedup"] and row["dev_speedup"] >= 1.0 and row["build_rc"] == 0:
            p = subprocess.run(["python", "/work/kvm/tests/check_cuda.py", f"/work/b/{row['cid']}/bin", f"/work/b/{row['cid']}/chk"], capture_output=True, text=True)
            rels = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
            worst = max([x[k]["rel_l2"] for x in rels for k in ("fused", "two")] or [None]) if rels else None
            row["check"] = dict(rc=p.returncode, passed="CHECK PASS" in p.stdout, worst_rel_l2=worst)
        print("NS_SIB_JSON " + json.dumps(row), flush=True)
    return dict(rows=rows, gpu=gpu, secs=time.time() - t0, secs_build=t_build)


@app.local_entrypoint()
def main(tag: str = "sib1", passes: int = 3, n_random: int = 33):
    from selfimprove_bench import arms
    u, info = arms.build_unique(os.path.join(ROOT, "results"), n_random=n_random)
    items = [[f"c{i:03d}", "base" if x["arms"] == ["base"] else "|".join(x["arms"]), x["edits"]] for i, x in enumerate(u)]
    print("unique configs:", len(items), "info:", info, flush=True)
    r = run.remote(items, passes)
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(ROOT, "results", f"selfimprove_bench_{tag}_{ts}.jsonl")
    with open(path, "w") as f:
        for row in r["rows"]: f.write(json.dumps(row) + "\n")
    cost = r["secs"] / 3600 * RATE
    with open(os.path.join(ROOT, "results", "SPEND.md"), "a") as f:
        f.write(f"| {ts} | selfimprove bench {tag} (build {r['secs_build']:.0f}s, {len(items)} configs) | A100-80GB | {r['secs']:.0f} | {cost:.3f} | True |\n")
    print("est $%.3f ->" % cost, path)
