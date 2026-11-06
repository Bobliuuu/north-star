"""HUMAN-FREE (script) baseline for the 'triton' self-improve track. NOT North, NOT shown to North (its output never enters a prompt).

  modal run --detach modal/triton_sweep.py::main --tag sw1

Learns, on the H100, (a) the true achievable speedup of the launch-parameter space and (b) how it distributes:
  stage A  one-factor-at-a-time: every knob moved to its +-1 neighbours in its allowed list, measured on the micro cases of its kernel group (dev split);
  stage B  greedy per-group combination + top single moves, full dev+held-out suites, ABBA, same speed statistic as the gate (selfimprove.gates.speed_gate_split);
  stage C  random 4-knob neighbours of the default (North's action space: <= 4 edits), full suites, ABBA x2.
Every row is also printed as `NS_SWEEP_JSON {...}`. Spend is estimated from wall time at the H100 list price.
"""
import json, os, pathlib, random, statistics, sys, time
import modal

ROOT = pathlib.Path(__file__).resolve().parents[1]
IGN = ["**/__pycache__", "**/*.pyc", "**/.pytest_cache"]
image = (modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.10.0", "numpy==2.2.5").env({"PYTHONUNBUFFERED": "1"})
         .add_local_dir(str(ROOT / "src/tritonk"), "/repo/src/tritonk", copy=False, ignore=IGN)
         .add_local_dir(str(ROOT / "src/selfimprove"), "/repo/src/selfimprove", copy=False, ignore=IGN + ["**/tests/**"])
         .add_local_dir(str(ROOT / "harness"), "/repo/harness", copy=False, ignore=IGN))
app = modal.App("northstar-triton-sweep", image=image)
cache = modal.Volume.from_name("northstar-triton-cache", create_if_missing=True)
RATE = 3.95

GROUPS = {   # knob prefix -> micro-case prefixes that exercise it
    "GU_": ["moe_t1"], "DN_": ["moe_t1"], "MB_": ["moe_blk"], "ATT_": ["attn_t1"], "CMB_": ["attn_t1"], "ROPE_NW": ["attn_t1"],
    "XA_": ["attn_x"], "XCMB_": ["attn_x"], "ROPEB_": ["attn_x"], "LIN_": ["lin_"], "RN_": ["small_"], "TK_": ["small_"], "AM_": ["small_"],
}


def group_of(knob):
    for p, g in GROUPS.items():
        if knob.startswith(p): return tuple(g)


@app.function(gpu="H100", timeout=3 * 3600, volumes={"/root/.triton": cache}, max_containers=1)
def sweep(tag: str, n_random: int = 8, seed: int = 0):
    sys.path.insert(0, "/repo/src"); sys.path.insert(0, "/repo/harness/selfimprove")
    os.environ["NS_WORKSPACE"] = "/repo"
    import torch, triton_suite as TS
    from selfimprove import structured_triton as STR, gates as G
    t_start = time.time(); rows = []; rng = random.Random(seed)
    S = TS.State(0)

    def emit(**kw):
        kw["t_s"] = round(time.time() - t_start, 1); rows.append(kw); print("NS_SWEEP_JSON " + json.dumps(kw), flush=True)

    emit(kind="env", gpu=torch.cuda.get_device_name(), torch=torch.__version__)
    base = TS.cfg()
    assert base == TS.DEFAULTS, "sweep must start from the default launch config"

    def valid(c):
        for a, b, cap in STR.CAPS:
            if c[a] * c[b] > cap: return False
        return all(STR.DIVIDES[k] % c[k] == 0 for k in STR.DIVIDES)

    def tm(c, split, only=None):
        return TS.run_bench(split, c, S, only=only)["time_ms"]

    def pair(c, split, only, reps=3):
        """interleaved default/candidate, returns (median default ms, median cand ms)"""
        d, k = [], []
        for i in range(reps):
            for tag_ in (("d", "k") if i % 2 == 0 else ("k", "d")):
                (d if tag_ == "d" else k).append(tm(base if tag_ == "d" else c, split, only))
        return statistics.median(d), statistics.median(k)

    # ---- stage A: OFAT +-1 neighbours, micro cases of the knob's group, dev split
    best = {}
    for knob, (vals, _) in STR.TUNABLES.items():
        i0 = vals.index(base[knob]); only = list(group_of(knob))
        for j in (i0 - 1, i0 + 1):
            if not (0 <= j < len(vals)): continue
            c = dict(base, **{knob: vals[j]})
            if not valid(c): continue
            try:
                d, k = pair(c, "dev", only, reps=2)
            except Exception as e:
                emit(kind="ofat", knob=knob, value=vals[j], error=repr(e)[:200]); continue
            emit(kind="ofat", knob=knob, value=vals[j], group=only, base_ms=d, cand_ms=k, speedup=d / k)
            if knob not in best or d / k > best[knob][1]: best[knob] = (vals[j], d / k)
    cache.commit()

    # ---- stage B: greedy per-group combination + top singles on the FULL suites, ABBA
    def abba_full(c, n=8, splits=("dev", "heldout")):
        res = {}
        for sp in splits:
            b, k = [], []
            for first_base in [i % 4 in (0, 3) for i in range(n)]:
                for t_ in (("b", "k") if first_base else ("k", "b")):
                    (b if t_ == "b" else k).append(tm(base if t_ == "b" else c, sp))
            res[sp] = G.speed_gate_split(b, k, G.SpeedSpec(n_runs=n, min_gain=0.02))
        return res

    def qc(c):
        """gate-side numbers of a config: worst non-bitwise rel_l2, #bitwise failures, mean/min quality item (1 - drift vs default)"""
        cs = TS.run_check(12345, c); it = TS.run_quality(c)
        bw = [x for x in cs if "rowinv" in x["name"] or x["name"] in ("argmax_lowest_tie", "topk_T1", "topk_T13")]
        return dict(worst_rel_l2=max(x["rel_l2"] for x in cs if x not in bw), bitwise_fail=sum(1 for x in bw if x["rel_l2"] > 0), q_mean=sum(it) / len(it), q_min=min(it))

    singles = sorted(((s, k, v) for k, (v, s) in best.items() if s > 1.003), reverse=True)
    emit(kind="ofat_best", moves=[dict(knob=k, value=v, speedup=s) for s, k, v in singles])
    combos = {}
    if singles:
        combos["all_positive"] = dict(base, **{k: v for s, k, v in singles})
        for g_ in sorted({group_of(k) for s, k, v in singles}):
            combos["group_" + "+".join(g_)] = dict(base, **{k: v for s, k, v in singles if group_of(k) == g_})
        for s, k, v in singles[:3]: combos[f"single_{k}={v}"] = dict(base, **{k: v})
    for name, c in combos.items():
        if not valid(c): emit(kind="combo", name=name, error="cap violation"); continue
        try:
            r = abba_full(c)
        except Exception as e:
            emit(kind="combo", name=name, error=repr(e)[:200]); continue
        emit(kind="combo", name=name, edits={k: v for k, v in c.items() if base[k] != v}, dev=r["dev"], heldout=r["heldout"], gates_side=qc(c))
    cache.commit()

    # ---- stage C: random <=4-knob neighbours of the default (North's action space)
    names = list(STR.TUNABLES)
    for i in range(n_random):
        for _ in range(100):
            ks = rng.sample(names, rng.randint(1, 4)); c = dict(base)
            for k in ks:
                vals = STR.TUNABLES[k][0]; c[k] = rng.choice([v for v in vals if v != base[k]])
            if valid(c): break
        try:
            r = abba_full(c, n=4)
        except Exception as e:
            emit(kind="random", edits={k: c[k] for k in ks}, error=repr(e)[:200]); continue
        emit(kind="random", edits={k: v for k, v in c.items() if base[k] != v}, dev=r["dev"], heldout=r["heldout"], gates_side=qc(c))
    # ---- A/A noise of the in-process timer (default vs default)
    r = abba_full(dict(base), n=8)
    emit(kind="aa", dev=r["dev"], heldout=r["heldout"])
    cache.commit()
    wall = time.time() - t_start
    emit(kind="done", wall_s=wall, est_usd=wall / 3600 * RATE)
    return rows


@app.local_entrypoint()
def main(tag: str = "sw1", n_random: int = 8):
    rows = sweep.remote(tag, n_random)
    p = ROOT / "results" / f"selfimprove_triton_sweep_{tag}.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n"); print(f"wrote {p}")
