"""One-factor-at-a-time sweep of the moe_fused.cu structured tunables on a Modal A100 (human-side baseline, not North).

  modal run --detach modal/kernel_sweep.py --tag sweep1        # ALWAYS --detach

Per config: sed the constexpr into a copy of the kernel, nvcc -O3 sm_80 (same flags as harness/selfimprove/kernel_build.py), optional
correctness check (src/kvm/tests/check_cuda.py vs torch CPU reference), then `bench` on the DEV and HELD-OUT shape sets (routing corr75,
21 reps, same as harness/selfimprove/kernel_bench.py). Configs are benched in `passes` interleaved passes (shuffled order) and the
per-config result is the median over passes of the dev-mean fused us_med, to cancel box drift.
Every result row is also printed as `NS_SWEEP_JSON {...}` so `modal app logs <id>` recovers it if the client dies.
Spend is ESTIMATED (wall secs x list price) and appended to results/SPEND.md.
"""
import datetime, json, os, random, re, statistics, subprocess, time
import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RATE = 2.50
DEV = [(1, 1, 2), (4, 4, 4), (8, 8, 4), (16, 16, 8)]
HELD = [(3, 5, 3), (6, 9, 5), (12, 7, 6), (13, 3, 7), (5, 11, 3), (9, 6, 5)]
BASE = dict(KC=256, XPAD=8, HPAD=8, MT_SPLIT=16, MINB_PER_SM=1, UNROLL_KS=16)
# one-factor-at-a-time variants of the baseline
OFAT = ([("KC", v) for v in (64, 128, 512)] + [("XPAD", v) for v in (0, 16, 24)] + [("HPAD", v) for v in (0, 16, 24)]
        + [("MT_SPLIT", v) for v in (4, 8, 12)] + [("MINB_PER_SM", v) for v in (2, 3, 4, 6, 8)]
        + [("UNROLL_KS", v) for v in (1, 2, 4, 8)])
CONFIGS = [dict(BASE)] + [dict(BASE, **{k: v}) for k, v in OFAT] + [dict(BASE, KC=512, UNROLL_KS=8), dict(BASE, KC=128, UNROLL_KS=8)]
NEW = {"UNROLL_KS", "MINB_PER_SM"}

image = (modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
         .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
         .pip_install("numpy<2")
         .add_local_dir(os.path.join(ROOT, "src/kvm"), "/work/kvm"))
app = modal.App("northstar-kernel-sweep", image=image)


def _name(c):
    d = [f"{k}{v}" for k, v in c.items() if BASE[k] != v]
    return "base" if not d else "_".join(d)


def _src(c):
    s = open("/work/kvm/moe_fused.cu").read()
    for k, v in c.items():
        s, n = re.subn(rf"(?m)^(constexpr int {k} = )\d+(;)", rf"\g<1>{v}\g<2>", s)
        assert n == 1, k
    return s


def _build(c):
    nm = _name(c); d = f"/work/b/{nm}"; os.makedirs(d, exist_ok=True)
    shutil_src = f"{d}/moe_fused.cu"
    open(shutil_src, "w").write(_src(c))
    os.system(f"cp -r /work/kvm/include {d}/include")
    p = subprocess.run(["nvcc", "-O3", "-std=c++17", "-arch=sm_80", "--resource-usage", "-o", f"{d}/bin", shutil_src], capture_output=True, text=True)
    regs = re.findall(r"Used (\d+) registers", p.stderr)
    spill = re.findall(r"(\d+) bytes spill stores", p.stderr)
    return nm, dict(rc=p.returncode, err=p.stderr[-600:] if p.returncode else "", regs=regs, spill_stores=spill)


def _bench(nm, shapes):
    out = f"/work/b/{nm}/bench.jsonl"
    arg = ";".join(f"{a},{b},{c}" for a, b, c in shapes)
    p = subprocess.run([f"/work/b/{nm}/bin", "bench", out, "--routing", "1", "--shapes", arg, "--reps", "21"], capture_output=True, text=True, timeout=300)
    if p.returncode: return None
    rows = [json.loads(l) for l in open(out) if l.strip()]
    for r in rows:
        if r["out_maxabs"] <= 0 or r["fused_vs_two_maxabsdiff"] / r["out_maxabs"] > 1e-2: return None
    return [r["v"]["fused"]["us_med"] for r in rows]


@app.function(gpu="A100-80GB", timeout=2400, cpu=8, memory=32768, scaledown_window=2, max_containers=1)
def sweep(passes: int = 3):
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time(); res = {}
    gpu = subprocess.run("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip()
    print("GPU", gpu, flush=True)
    os.makedirs("/work/b", exist_ok=True)
    with ThreadPoolExecutor(8) as ex:
        for nm, info in ex.map(_build, CONFIGS):
            res[nm] = dict(build=info); print(f"built {nm} rc={info['rc']} regs={info['regs']} spill={info['spill_stores']}", flush=True)
    t_build = time.time() - t0
    cfg = {_name(c): c for c in CONFIGS}
    # correctness (A100 binary vs torch CPU reference, same cases as tests/check_cuda.py) for baseline and every config touching a NEW tunable
    for nm, c in cfg.items():
        if res[nm]["build"]["rc"] or not (nm == "base" or any(BASE[k] != v and k in NEW for k, v in c.items())): continue
        p = subprocess.run(["python", "/work/kvm/tests/check_cuda.py", f"/work/b/{nm}/bin", f"/work/b/{nm}/chk"], capture_output=True, text=True)
        rels = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
        worst = max([r[k]["rel_l2"] for r in rels for k in ("fused", "two")] or [None]) if rels else None
        res[nm]["check"] = dict(rc=p.returncode, pass_="CHECK PASS" in p.stdout, worst_rel_l2=worst, tail=p.stdout[-200:] + p.stderr[-300:])
        print(f"check {nm}: {res[nm]['check']['pass_']} worst_rel_l2={worst}", flush=True)
    t_check = time.time() - t0 - t_build
    ok = [nm for nm in cfg if res[nm]["build"]["rc"] == 0 and res[nm].get("check", {}).get("pass_", True)]
    for nm in cfg: res[nm].update(dev=[], held=[])
    rng = random.Random(1)
    for ps in range(passes):
        order = ok[:]; rng.shuffle(order)
        for nm in order:
            d = _bench(nm, DEV); h = _bench(nm, HELD)
            if d: res[nm]["dev"].append(d)
            if h: res[nm]["held"].append(h)
        print(f"pass {ps + 1}/{passes} done t={time.time() - t0:.0f}s", flush=True)
    rows = []
    for nm, c in cfg.items():
        r = res[nm]
        med = lambda L: [statistics.median(col) for col in zip(*L)] if L else None
        dv, hv = med(r["dev"]), med(r["held"])
        row = dict(name=nm, cfg=c, build_rc=r["build"]["rc"], build_err=r["build"]["err"], regs=r["build"]["regs"], spill_stores=r["build"]["spill_stores"],
                   check=r.get("check"), dev_us=dv, held_us=hv, dev_mean_us=sum(dv) / len(dv) if dv else None, held_mean_us=sum(hv) / len(hv) if hv else None,
                   dev_pass_means=[sum(x) / len(x) for x in r["dev"]], n_passes=len(r["dev"]))
        rows.append(row)
    b = next(r for r in rows if r["name"] == "base")
    for r in rows:
        r["dev_speedup"] = b["dev_mean_us"] / r["dev_mean_us"] if r["dev_mean_us"] else None
        r["held_speedup"] = b["held_mean_us"] / r["held_mean_us"] if r["held_mean_us"] and b["held_mean_us"] else None
        print("NS_SWEEP_JSON " + json.dumps(r), flush=True)
    return dict(rows=rows, gpu=gpu, secs=time.time() - t0, secs_build=t_build, secs_check=t_check)


@app.local_entrypoint()
def main(tag: str = "sweep1", passes: int = 3):
    r = sweep.remote(passes)
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(ROOT, "results", f"kernel_sweep_{tag}_{ts}.jsonl")
    with open(path, "w") as f:
        for row in r["rows"]: f.write(json.dumps(row) + "\n")
    cost = r["secs"] / 3600 * RATE
    with open(os.path.join(ROOT, "results", "SPEND.md"), "a") as f:
        f.write(f"| {ts} | kernel tunable sweep {tag} (build {r['secs_build']:.0f}s check {r['secs_check']:.0f}s) | A100-80GB | {r['secs']:.0f} | {cost:.3f} | True |\n")
    print(f"{'name':28s} {'dev_us':>8s} {'dev_x':>6s} {'held_x':>6s} chk")
    for x in sorted(r["rows"], key=lambda x: -(x["dev_speedup"] or 0)):
        print(f"{x['name']:28s} {x['dev_mean_us'] or 0:8.1f} {x['dev_speedup'] or 0:6.3f} {x['held_speedup'] or 0:6.3f} {x['check'] and x['check']['pass_']}")
    print("est $%.3f ->" % cost, path)
