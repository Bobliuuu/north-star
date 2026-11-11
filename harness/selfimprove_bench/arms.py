"""Edit-set arms for the self-improvement benchmark: same 8-tunable kernel task, same legal space, same budget.

Arms: base (no edit), human (OFAT/combo sweep winners), north (every distinct edit set North proposed in the v1 and v5 live
stores), random (budget-matched uniform draws from the legal space, 2 seeds). Pure Python, CPU only; unit-tested.
"""
import json, random, pathlib

BASE = dict(KC=256, XPAD=8, HPAD=8, MT_SPLIT=16, MINB_PER_SM=1, UNROLL_KS=16, UNROLL_S2=4, UNROLL_NT=1)
# legal values mirror src/selfimprove/structured.TUNABLES (lo, hi, multiple_of)
SPACE = {"KC": (16, 2048, 16), "XPAD": (0, 64, 8), "HPAD": (0, 64, 8), "MT_SPLIT": (1, 16, 1), "MINB_PER_SM": (1, 8, 1),
         "UNROLL_KS": (1, 128, 1), "UNROLL_S2": (1, 4, 1), "UNROLL_NT": (1, 8, 1)}
H = 2048
MAX_EDITS = 3
HUMAN = [dict(KC=64), dict(KC=64, MINB_PER_SM=3), dict(UNROLL_KS=4), dict(MINB_PER_SM=3), dict(KC=512, UNROLL_KS=8)]


def legal_values(name):
    lo, hi, m = SPACE[name]
    vals = [v for v in range(lo, hi + 1) if v % m == 0 or (name in ("MT_SPLIT", "MINB_PER_SM", "UNROLL_KS", "UNROLL_S2", "UNROLL_NT"))]
    if name == "KC":
        vals = [v for v in vals if H % v == 0]
    return vals


def key(edits):
    return tuple(sorted((k, v) for k, v in edits.items() if BASE[k] != v))


def random_arm(n, seed, exclude=()):
    rng = random.Random(seed)
    seen, out = set(exclude), []
    while len(out) < n:
        k = rng.randint(1, MAX_EDITS)
        names = rng.sample(sorted(SPACE), k)
        e = {nm: rng.choice(legal_values(nm)) for nm in names}
        kk = key(e)
        if not kk or kk in seen:
            continue
        seen.add(kk); out.append(e)
    return out


def north_arm(results_dir):
    out, seen, n_total, n_dup = [], set(), 0, 0
    for d in ("selfimprove_kernel_structured", "selfimprove_kernel_structured_v5"):
        p = pathlib.Path(results_dir) / d / "proposals.jsonl"
        for l in p.read_text().splitlines():
            r = json.loads(l)
            if r.get("event") != "proposal":
                continue
            n_total += 1
            s = r.get("structured") or {}
            edits = {e["name"]: e["value"] for e in s.get("edits", []) if e.get("name") in SPACE}
            if not edits or not isinstance(next(iter(edits.values())), int):
                continue
            kk = key(edits)
            if not kk or kk in seen:
                n_dup += 1
                continue
            seen.add(kk); out.append(edits)
    return out, dict(proposals=n_total, distinct_valid=len(out), dup_or_empty=n_dup)


def build_arms(results_dir, n_random=33, seeds=(11, 12)):
    north, info = north_arm(results_dir)
    arms = [("base", dict())] + [("human", e) for e in HUMAN] + [("north", e) for e in north]
    used = {key(e) for _, e in arms}
    for s in seeds:
        r = random_arm(n_random, s, used)
        used |= {key(e) for e in r}
        arms += [(f"random{s}", e) for e in r]
    return arms, info


def llm_samples(results_dir):
    """{model: [edits_or_None per sample (None = invalid/empty/failed)]} from results/selfimprove_bench_llm_*.jsonl"""
    out = {}
    for f in sorted(pathlib.Path(results_dir).glob("selfimprove_bench_llm_*.jsonl")):
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        if not rows:
            continue
        m = rows[0]["model"]
        out[m] = [({k: v for k, v in r["edits"].items() if k in SPACE} or None) if r.get("edits") else None for r in rows]
    return out


def build_unique(results_dir, n_random=33, seeds=(11, 12)):
    """Unique edit sets to measure; each carries every arm label that produced it. Returns (items, info)."""
    arms, info = build_arms(results_dir, n_random, seeds)
    uniq = {}
    def add(arm, e):
        k = key(e)
        u = uniq.setdefault(k, dict(edits=dict(e), arms=[]))
        if arm not in u["arms"]:
            u["arms"].append(arm)
    for arm, e in arms:
        add(arm, e)
    for model, samples in llm_samples(results_dir).items():
        for e in samples:
            if e and key(e):
                add("llm:" + model, e)
    base_k = key({})
    items = [dict(edits=u["edits"], arms=u["arms"]) for k, u in uniq.items()]
    items.sort(key=lambda u: (u["arms"] != ["base"], 0))
    info["unique_configs"] = len(items)
    return items, info
