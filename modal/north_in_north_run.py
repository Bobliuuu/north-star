"""North-in-North experiment: three ways to improve the moe_fused.cu kernel tunables, same task, same measured-eval budget.

  modal run --detach modal/north_in_north_run.py --tag nin1            # ALWAYS --detach; smoke: --rounds 1 --sessions 1

Arms (all closed loop: every round's edits are BUILT and TIMED on the A100 and the measured DEV speedups are fed back):
  A  standard  : greedy coordinate-descent autotuner (no LLM), 3 evals/round.
  B  north     : ONE North (CohereLabs/North-Mini-Code-1.0) proposes, 3 distinct edit sets/round (best-of-3 sampling).
  C  north_in_north : a North ORCHESTRATOR plans 4 focus areas, 4 North SUBAGENTS each propose edits for one area, the
                 orchestrator selects/merges <=3 sets to evaluate. Every call is the same North weights; no other model.
Each arm gets the same number of measured evaluations (3/round x rounds). North runs in-process on the same A100 (HF generate,
reasoning off); generation and timing never overlap. Final phase re-measures every unique config interleaved (3 passes), dev + held-out
shapes, and runs the correctness check on everything not slower than base. Raw rows are also printed as NIN_JSON lines.
"""
import datetime, json, os, random, re, statistics, subprocess, sys, time, uuid
import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RATE = 2.50
DEV = [(1, 1, 2), (4, 4, 4), (8, 8, 4), (16, 16, 8)]
HELD = [(3, 5, 3), (6, 9, 5), (12, 7, 6), (13, 3, 7), (5, 11, 3), (9, 6, 5)]
BASE = dict(KC=256, XPAD=8, HPAD=8, MT_SPLIT=16, MINB_PER_SM=1, UNROLL_KS=16, UNROLL_S2=4, UNROLL_NT=1)
WEIGHTS, MODEL_DIR = "/weights", "/weights/north-mini-code-d11e61a8"
MODEL_ID, MODEL_REV = "CohereLabs/North-Mini-Code-1.0", "d11e61a842617a22dc328552fa5bb86231ee4f37"

weights_vol = modal.Volume.from_name("northstar-weights")
results_vol = modal.Volume.from_name("northstar-results", create_if_missing=True)
image = (modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
         .pip_install("torch==2.10.0", "transformers==5.22.1", "accelerate==1.15.0", "numpy==2.2.5", "safetensors")
         .add_local_dir(os.path.join(ROOT, "src/kvm"), "/work/kvm")
         .add_local_dir(os.path.join(ROOT, "src/selfimprove"), "/work/pysrc/selfimprove"))
app = modal.App("northstar-nin", image=image)

# ---------------------------------------------------------------- kernel build / bench (same method as selfimprove_bench_run)


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
    return cid, dict(rc=p.returncode, err=p.stderr[-300:] if p.returncode else "", regs=regs, spill_stores=spill)


def _bench(cid, shapes):
    out = f"/work/b/{cid}/bench.jsonl"
    arg = ";".join(f"{a},{b},{c}" for a, b, c in shapes)
    p = subprocess.run([f"/work/b/{cid}/bin", "bench", out, "--routing", "1", "--shapes", arg, "--reps", "21"], capture_output=True, text=True, timeout=300)
    if p.returncode: return None
    rows = [json.loads(l) for l in open(out) if l.strip()]
    for r in rows:
        if r["out_maxabs"] <= 0 or r["fused_vs_two_maxabsdiff"] / r["out_maxabs"] > 1e-2: return None
    return [r["v"]["fused"]["us_med"] for r in rows]


def key(edits):
    return tuple(sorted((k, v) for k, v in edits.items() if BASE[k] != v))


class Evaluator:
    """Builds + times edit sets. Loop-time feedback = DEV only (base timed before and after every batch, interleaved)."""

    def __init__(self):
        from concurrent.futures import ThreadPoolExecutor
        self.ex = ThreadPoolExecutor(8)
        self.cache, self.n = {}, 0                      # key -> record
        self.cfgs = {}                                  # cid -> full config (for the final phase)
        os.makedirs("/work/b", exist_ok=True)
        cid, info = _build(("base", dict(BASE)))
        assert info["rc"] == 0, info
        self.cfgs["base"] = dict(BASE)

    def new_cid(self):
        self.n += 1
        return f"c{self.n:04d}"

    def measure(self, edit_sets):
        """edit_sets: list of dicts. Returns list of records (cached by key). Each: cid, edits, dev_speedup (None if build/bench fail)."""
        todo = []
        for e in edit_sets:
            k = key(e)
            if k not in self.cache and not any(k == key(t) for _, t in todo):
                todo.append((self.new_cid(), e))
        items = [(cid, dict(BASE, **e)) for cid, e in todo]
        for (cid, e), (_, info) in zip(todo, self.ex.map(_build, items)):
            self.cache[key(e)] = dict(cid=cid, edits=dict(e), build_rc=info["rc"], build_err=info["err"], regs=info["regs"], dev_speedup=None)
            self.cfgs[cid] = dict(BASE, **e)
        ok = [self.cache[key(e)] for _, e in todo if self.cache[key(e)]["build_rc"] == 0]
        if ok:
            order = ok[:]; random.shuffle(order)
            b0 = _bench("base", DEV)
            times = {r["cid"]: _bench(r["cid"], DEV) for r in order}
            b1 = _bench("base", DEV)
            bm = (sum(b0) + sum(b1)) / (len(b0) + len(b1)) if b0 and b1 else None
            for r in ok:
                t = times[r["cid"]]
                r["dev_speedup"] = (bm / (sum(t) / len(t))) if (t and bm) else None
        return [self.cache[key(e)] for e in edit_sets]


# ---------------------------------------------------------------- North (in-process HF)


class North:
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        assert os.path.exists(os.path.join(MODEL_DIR, "config.json")), "weights missing"
        t = time.time()
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa",
                                                          experts_implementation="grouped_mm").eval()
        self.load_s = time.time() - t
        self.calls, self.tok_out, self.gen_s = 0, 0, 0.0

    def gen(self, system, user, temp=0.5, max_new=350):
        t0 = time.time()
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, reasoning_effort="none")
        x = self.tok(ids, return_tensors="pt", add_special_tokens=False).to("cuda")
        with self.torch.no_grad():
            y = self.model.generate(**x, max_new_tokens=max_new, do_sample=temp > 0, temperature=temp if temp > 0 else None)
        n_in = x["input_ids"].shape[1]
        text = self.tok.decode(y[0, n_in:], skip_special_tokens=True)
        self.calls += 1; self.tok_out += y.shape[1] - n_in; self.gen_s += time.time() - t0
        return text


# ---------------------------------------------------------------- prompts


def hist_text(hist):
    if not hist: return "(none yet)"
    return "\n".join(hist)


def hist_line(edits, rec):
    e = ", ".join(f"{k}={v}" for k, v in edits.items()) or "(none)"
    if rec is None: return f"edits {e}: not evaluated (invalid or duplicate)"
    if rec["build_rc"]: return f"edits {e}: BUILD FAILED"
    if rec["dev_speedup"] is None: return f"edits {e}: bench failed"
    return f"edits {e}: measured dev speedup {rec['dev_speedup']:.3f}x (<1 means SLOWER than base; need > 1.03)"


SYS_NORTH = (
    "You are North, optimizing your own inference kernel tunables. Reply with ONLY one JSON object of the form "
    '{"edits":[{"name":"<TUNABLE>","value":<integer>}],"rationale":"<short>"} and nothing else. Choose values from the allowed '
    "ranges listed in the task. You never write code or diffs; the harness applies your values. If you see no safe improvement reply with "
    '{"edits":[],"rationale":"..."}.')
SYS_ORCH = ("You are North acting as the ORCHESTRATOR of a team of North subagents that tune your own inference kernel. "
            "Reply with ONLY one JSON object and nothing else.")
SYS_SUB = ("You are a North SUBAGENT tuning your own inference kernel. The orchestrator gave you one focus area and the only tunables you may change. "
           "Reply with ONLY one JSON object of the form "
           '{"edits":[{"name":"<TUNABLE>","value":<integer>}],"rationale":"<short>"} and nothing else, using only your assigned tunables. '
           'If you see no safe improvement reply with {"edits":[],"rationale":"..."}.')


def parse_set(S, obj):
    """obj: {'edits': [{name,value}], 'rationale'} -> validated dict or None."""
    try:
        p = S.parse_edits(json.dumps(obj))
    except Exception:
        return None
    return {n: v for n, v in p["edits"]}


def run_session(arm, sess, rounds, per_round, ev, north, S, files, log, seed):
    """Returns the list of unique evaluated records (in order) for this session."""
    random.seed(seed)
    if north is not None: north.torch.manual_seed(seed)
    base_prompt = S.prompt_section(files)
    hist, evaluated, seen = [], [], set()
    best_cfg, best_sp = {}, 1.0                          # arm A state
    fb = dict(planner_fallback=0, selector_fallback=0, sub_invalid=0, sub_empty=0, sub_dup=0, b_invalid=0, b_empty=0, b_dup=0)
    names = list(S.TUNABLES)

    def emit(kind, **kw):
        log(dict(kind=kind, arm=arm, session=sess, **kw))

    for rd in range(rounds):
        cands = []                                       # edit dicts to evaluate this round
        if arm == "standard":
            # greedy coordinate descent: neighbours (x2, /2, then +-1 step) of the current best, one tunable at a time, cycling
            moves = []
            for off in range(len(names)):
                n = names[(rd * 3 + off) % len(names)]
                cur = dict(BASE, **best_cfg)[n]
                lo, hi, m, _d = S.TUNABLES[n]
                for v in (cur * 2, cur // 2 if cur // 2 else cur, cur + max(m, 1), cur - max(m, 1)):
                    if lo <= v <= hi and v % m == 0 and v != cur and not (n == "KC" and 2048 % v):
                        e = dict(best_cfg); e[n] = v
                        if key(e) not in seen and e not in moves: moves.append(e)
                if len(moves) >= per_round: break
            cands = moves[:per_round]
        elif arm == "north":
            user = base_prompt + "\n## Measured history this session (dev split)\n" + hist_text(hist)
            tries = 0
            while len(cands) < per_round and tries < per_round * 3:
                tries += 1
                raw = north.gen(SYS_NORTH, user, temp=[0.3, 0.55, 0.8, 1.0][tries % 4])
                try:
                    e = {n: v for n, v in S.parse_edits(raw)["edits"]}
                except Exception as ex:
                    fb["b_invalid"] += 1; emit("north_call", round=rd, role="single", raw=raw[:600], status="invalid"); continue
                st = "ok"
                if not e: fb["b_empty"] += 1; st = "empty"
                elif key(e) in seen or any(key(e) == key(c) for c in cands): fb["b_dup"] += 1; st = "dup"
                emit("north_call", round=rd, role="single", raw=raw[:600], status=st)
                if st == "ok": cands.append(e)
        else:  # north_in_north
            tun = "\n".join(f"- {n} (current {BASE[n]}): allowed [{lo},{hi}]" + (f", multiple of {m}" if m > 1 else "") + f"; {d}"
                            for n, (lo, hi, m, d) in S.TUNABLES.items())
            # 1) orchestrator plans 4 focus areas
            plan_user = (f"Tunables:\n{tun}\n\n{S.GUIDANCE}\n## Measured history this session (dev split)\n{hist_text(hist)}\n\n"
                         'Split the search among exactly 4 subagents. Output: {"subtasks":[{"focus":"<short>","tunables":["NAME",...]} x4]} '
                         "(each subagent gets 1-3 tunable names from the list; areas may overlap; prefer areas not yet measured or that measured >= 1.0).")
            raw = north.gen(SYS_ORCH, plan_user, temp=0.4, max_new=300)
            subs = None
            try:
                o = json.loads(S.extract_json_object(raw))
                st = o["subtasks"]
                subs = [dict(focus=str(s["focus"])[:200], tunables=[t for t in s["tunables"] if t in S.TUNABLES][:3]) for s in st][:4]
                if len(subs) < 4 or any(not s["tunables"] for s in subs): subs = None
            except Exception:
                subs = None
            if subs is None:
                fb["planner_fallback"] += 1
                subs = [dict(focus="smem tiling", tunables=["KC", "XPAD", "HPAD"]), dict(focus="occupancy / register cap", tunables=["MINB_PER_SM", "UNROLL_KS"]),
                        dict(focus="down-projection loops", tunables=["UNROLL_S2", "UNROLL_NT"]), dict(focus="tile path split", tunables=["MT_SPLIT", "KC"])]
            emit("north_call", round=rd, role="orchestrator_plan", raw=raw[:800], status="ok", subtasks=subs)
            # 2) four subagents
            proposals = []
            for si, s in enumerate(subs):
                sub_user = (f"Your focus: {s['focus']}\nYou may ONLY change: {', '.join(s['tunables'])}\n\nTunables:\n{tun}\n\n{S.GUIDANCE}\n"
                            f"## Measured history this session (dev split)\n{hist_text(hist)}")
                raw = north.gen(SYS_SUB, sub_user, temp=0.6)
                try:
                    p = S.parse_edits(raw); e = {n: v for n, v in p["edits"]}
                    if any(n not in s["tunables"] for n in e): raise ValueError("outside_assignment")
                except Exception:
                    fb["sub_invalid"] += 1; emit("north_call", round=rd, role=f"subagent{si}", raw=raw[:600], status="invalid"); continue
                st = "ok"
                if not e: fb["sub_empty"] += 1; st = "empty"
                elif key(e) in seen: fb["sub_dup"] += 1; st = "dup"
                emit("north_call", round=rd, role=f"subagent{si}", raw=raw[:600], status=st)
                if st == "ok": proposals.append(dict(by=si, focus=s["focus"], edits=e))
            # 3) orchestrator selects / merges up to per_round sets
            chosen = []
            if proposals:
                plist = "\n".join(f"[{i}] ({p['focus']}) " + ", ".join(f"{k}={v}" for k, v in p["edits"].items()) for i, p in enumerate(proposals))
                sel_user = (f"Subagent proposals:\n{plist}\n\n## Measured history this session (dev split)\n{hist_text(hist)}\n\n"
                            f'Choose up to {per_round} edit sets to BUILD AND MEASURE this round. You may use a proposal as is or merge several (at most 3 edits per set, '
                            'each tunable once, no set already measured). Output: {"evaluate":[{"edits":[{"name":"..","value":N}],"rationale":".."}]}')
                raw = north.gen(SYS_ORCH, sel_user, temp=0.3, max_new=350)
                try:
                    o = json.loads(S.extract_json_object(raw))
                    for it in o["evaluate"][:per_round]:
                        e = parse_set(S, it)
                        if e and key(e) not in seen and not any(key(e) == key(c) for c in chosen): chosen.append(e)
                except Exception:
                    pass
                emit("north_call", round=rd, role="orchestrator_select", raw=raw[:800], status="ok" if chosen else "fallback", n_chosen=len(chosen))
                if not chosen: fb["selector_fallback"] += 1
            # top up with subagent proposals so every arm gets the same eval budget when possible
            for p in proposals:
                if len(chosen) >= per_round: break
                if key(p["edits"]) not in seen and not any(key(p["edits"]) == key(c) for c in chosen): chosen.append(p["edits"])
            cands = chosen[:per_round]
        # evaluate (measured feedback)
        cands = [c for c in cands if key(c) and key(c) not in seen]
        recs = ev.measure(cands) if cands else []
        for c, r in zip(cands, recs):
            seen.add(key(c)); evaluated.append(r); hist.append(hist_line(c, r))
            if arm == "standard" and r["dev_speedup"] and r["dev_speedup"] > best_sp: best_sp, best_cfg = r["dev_speedup"], dict(c)
            emit("eval", round=rd, edits=c, cid=r["cid"], build_rc=r["build_rc"], dev_speedup=r["dev_speedup"])
        best = max([r["dev_speedup"] or 0 for r in evaluated] or [0])
        print(f"[{arm} s{sess}] round {rd + 1}/{rounds}: evaluated {len(evaluated)} total, best dev so far {best:.3f}x", flush=True)
    return evaluated, fb


@app.cls(gpu="A100-80GB", volumes={WEIGHTS: weights_vol, "/results": results_vol}, timeout=6 * 3600, cpu=8, memory=65536, scaledown_window=2, max_containers=1)
class NIN:
    @modal.method()
    def run(self, tag: str, rounds: int, per_round: int, sessions: int, arms: list):
        sys.path.insert(0, "/work/pysrc")
        from selfimprove import structured as S
        t0 = time.time()
        gpu = subprocess.run("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip()
        print("GPU", gpu, flush=True)
        logf = open(f"/results/nin_{tag}_trace.jsonl", "a")

        def log(d):
            d["t"] = round(time.time() - t0, 1)
            logf.write(json.dumps(d) + "\n"); logf.flush()
            print("NIN_TRACE " + json.dumps(d)[:1500], flush=True)

        files = {S.KERNEL_PATH: open("/work/kvm/moe_fused.cu").read()}
        ev = Evaluator()
        north = North()
        print(f"North loaded in {north.load_s:.0f}s", flush=True)
        out = dict(arms={}, fallbacks={})
        for arm in arms:
            for sess in range(1 if arm == "standard" else sessions):
                tt = time.time()
                recs, fb = run_session(arm, sess, rounds, per_round, ev, None if arm == "standard" else north, S, files, log, seed=1000 * (1 + sess) + len(arm))
                out["arms"][f"{arm}#{sess}"] = [dict(cid=r["cid"], edits=r["edits"], dev_speedup=r["dev_speedup"], build_rc=r["build_rc"]) for r in recs]
                out["fallbacks"][f"{arm}#{sess}"] = fb
                print(f"== {arm}#{sess} done {time.time() - tt:.0f}s, north calls so far {north.calls}", flush=True)
                results_vol.commit()
        # ----- final phase: re-measure every unique config + base, interleaved, dev + held-out, then correctness
        t_loop = time.time() - t0
        ids = ["base"] + [r["cid"] for r in ev.cache.values() if r["build_rc"] == 0]
        res = {cid: dict(dev=[], held=[]) for cid in ids}
        rng = random.Random(1)
        for ps in range(3):
            order = ids[:]; rng.shuffle(order)
            for cid in order:
                d, h = _bench(cid, DEV), _bench(cid, HELD)
                if d: res[cid]["dev"].append(d)
                if h: res[cid]["held"].append(h)
            print(f"final pass {ps + 1}/3 t={time.time() - t0:.0f}s", flush=True)
        med = lambda L: [statistics.median(c) for c in zip(*L)] if L else None
        mean = lambda v: sum(v) / len(v) if v else None
        bd, bh = mean(med(res["base"]["dev"])), mean(med(res["base"]["held"]))
        final = {}
        for r in ev.cache.values():
            cid = r["cid"]; row = dict(cid=cid, edits=r["edits"], build_rc=r["build_rc"], loop_dev_speedup=r["dev_speedup"], regs=r["regs"])
            if r["build_rc"] == 0:
                dv, hv = mean(med(res[cid]["dev"])), mean(med(res[cid]["held"]))
                row["dev_speedup"] = bd / dv if dv else None; row["held_speedup"] = bh / hv if hv and bh else None
                if row["dev_speedup"] and row["dev_speedup"] >= 1.0:
                    p = subprocess.run(["python", "/work/kvm/tests/check_cuda.py", f"/work/b/{cid}/bin", f"/work/b/{cid}/chk"], capture_output=True, text=True)
                    rels = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
                    worst = max([x[k]["rel_l2"] for x in rels for k in ("fused", "two")] or [None]) if rels else None
                    row["check"] = dict(rc=p.returncode, passed="CHECK PASS" in p.stdout, worst_rel_l2=worst)
            final[cid] = row
            print("NIN_JSON " + json.dumps(row), flush=True)
        out.update(final=final, gpu=gpu, secs=time.time() - t0, secs_loop=t_loop, north=dict(calls=north.calls, tokens_out=north.tok_out, gen_s=north.gen_s, load_s=north.load_s),
                   model=MODEL_ID, revision=MODEL_REV, per_round=per_round, rounds=rounds, sessions=sessions)
        json.dump(out, open(f"/results/nin_{tag}_summary.json", "w")); results_vol.commit()
        return out


@app.local_entrypoint()
def main(tag: str = "nin1", rounds: int = 8, per_round: int = 3, sessions: int = 2, arms: str = "standard,north,north_in_north"):
    r = NIN().run.remote(tag, rounds, per_round, sessions, arms.split(","))
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(ROOT, "results", f"nin_{tag}_{ts}.json")
    json.dump(r, open(path, "w"), indent=1)
    cost = r["secs"] / 3600 * RATE
    with open(os.path.join(ROOT, "results", "SPEND.md"), "a") as f:
        f.write(f"| {ts} | north-in-north {tag} ({r['north']['calls']} North calls) | A100-80GB | {r['secs']:.0f} | {cost:.3f} | True |\n")
    print("est $%.3f ->" % cost, path)
