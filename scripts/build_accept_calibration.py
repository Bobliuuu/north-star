#!/usr/bin/env python3
"""Deterministic converter: MEASURED self-speculation runs -> results/accept_calibration.jsonl (src/opt/calibration.py schema).

    python scripts/build_accept_calibration.py [--spec results/spec.jsonl ...] [--out results/accept_calibration.jsonl] [--dry-run]

What it uses (and nothing else; nothing is invented):
  rows with phase == "1_selfspec" written by harness/bench_spec.py. Each row carries, for ONE (task, draft config) run with block length
  gamma G and R verify rounds, `accept_rate_per_pos[i] = (#rounds with accepted > i) / R`, i = 0..G-1, i.e. the empirical survival
  P(A >= i+1) with A = longest accepted draft prefix (greedy verification), capped at G. That is exactly the calibration pmf with the
  last bin = P(A >= G):  pmf[j] = surv[j] - surv[j+1],  surv[0] = 1.
  Pooling: rows of the same (workload kind, draft key) are pooled round-weighted:  surv[i] = sum_rows(rate_i * R_row) / sum_rows(R_row).
  rate_i * R must be an integer count (checked), survival must be non-increasing (checked).
  When one draft key was measured at several gammas (r4_g2/g4/g8) only the LARGEST gamma is used (longest lookahead, one run per task,
  no mixing); the smaller-gamma configs are listed in `unused_configs`.
  Draft key = (attn_budget=0 [full attention, all runs], r = expert_r or 8 [None = native top-8], sorted skip_layers).
  `carry_hit_given_full` was NOT measured: written as 0.0 (carry disabled, conservative), flagged in `carry_note`.

Refusals (exit 2, no file written): a source with no usable rows (e.g. results/h100_spec.jsonl: its g_spec rows only have the mean
`mean_accepted`/`committed_per_round`, no per-position acceptance, so no pmf can be derived without inventing a shape); a row missing
rounds/accept_rate_per_pos/gamma; non-integer counts; rate lists whose length != gamma; non-monotone survival; missing hardware
(sibling <stem>.meta.json `gpu`); pmf failing calibration.parse_row; any output row that is not provenance == "measured".

Provenance fields per output row: source (repo-relative file), source_sha256, hardware, model, model_rev, n_tokens (sum of n_committed
over pooled runs), n_rounds (== n_samples), n_runs, tasks, gamma_measured, config_names, unused_configs, n_identical_runs.
Known bias: the last verify round of a 64-token run can be cut by the token budget, which slightly lowers its accepted count.
Output is byte-deterministic for a given input (sorted keys, no timestamps).
"""
from __future__ import annotations
import argparse, collections, hashlib, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from opt.calibration import parse_row  # noqa: E402

NATIVE_TOPK = 8   # North top-8 of 128 experts; DraftMode.expert_r=None means native top-k


class Refuse(Exception):
    pass


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _meta(path):
    mp = os.path.splitext(path)[0] + ".meta.json"
    if not os.path.exists(mp):
        raise Refuse(f"{path}: hardware unknown (missing {mp})")
    m = json.load(open(mp))
    if not m.get("gpu"):
        raise Refuse(f"{mp}: no 'gpu' field")
    return m


def _relpath(path):
    ap = os.path.abspath(path)
    return os.path.relpath(ap, ROOT) if ap.startswith(ROOT + os.sep) else path


def load_runs(path):
    """Return usable per-run records from one source file; raise Refuse if there are none or a row is malformed."""
    rows = [json.loads(l) for l in open(path) if l.strip()]
    sel = [r for r in rows if r.get("phase") == "1_selfspec"]
    if not sel:
        phases = sorted({str(r.get("phase")) for r in rows})
        raise Refuse(f"{path}: no phase '1_selfspec' rows with per-position acceptance (phases present: {phases}); "
                     "only mean accepted length is available, a pmf cannot be derived without inventing it")
    out = []
    for i, r in enumerate(sel):
        w = f"{path} row(1_selfspec #{i}, config={r.get('config')}, task={r.get('task')})"
        for k in ("accept_rate_per_pos", "rounds", "gamma", "kind", "skip_layers", "n_committed", "config"):
            if k not in r or r[k] is None:
                raise Refuse(f"{w}: missing field {k}")
        rates, R, G = r["accept_rate_per_pos"], r["rounds"], r["gamma"]
        if not isinstance(rates, list) or len(rates) != G or not isinstance(R, int) or R <= 0:
            raise Refuse(f"{w}: accept_rate_per_pos must be a list of length gamma={G} and rounds a positive int")
        counts = []
        for x in rates:
            c = x * R
            if not isinstance(x, (int, float)) or not 0 <= x <= 1 or abs(c - round(c)) > 1e-6:
                raise Refuse(f"{w}: rate {x} x rounds {R} is not an integer count: not a measured per-round fraction")
            counts.append(int(round(c)))
        if any(counts[j] < counts[j + 1] for j in range(G - 1)) or counts[0] > R:
            raise Refuse(f"{w}: survival counts not non-increasing / exceed rounds")
        r_exp = r.get("expert_r")
        out.append(dict(kind=r["kind"], r=NATIVE_TOPK if r_exp is None else int(r_exp), skip=tuple(sorted(int(s) for s in r["skip_layers"])),
                        gamma=G, rounds=R, counts=counts, n_committed=int(r["n_committed"]), task=r.get("task"),
                        config=r["config"], identical=bool(r.get("identical"))))
    return out


def build_rows(path):
    meta = _meta(path)
    runs = load_runs(path)
    groups = collections.defaultdict(list)
    for x in runs:
        groups[(x["kind"], x["r"], x["skip"])].append(x)
    rows = []
    for (kind, r, skip), xs in sorted(groups.items()):
        gmax = max(x["gamma"] for x in xs)
        use = [x for x in xs if x["gamma"] == gmax]
        unused = sorted({f"{x['config']}(gamma={x['gamma']})" for x in xs if x["gamma"] != gmax})
        R = sum(x["rounds"] for x in use)
        surv = [1.0] + [sum(x["counts"][i] for x in use) / R for i in range(gmax)]
        pmf = [surv[j] - surv[j + 1] for j in range(gmax)] + [surv[gmax]]
        row = {"workload": kind, "attn_budget": 0, "r": r, "skip_layers": list(skip),
               "accept_len_pmf": [round(p, 12) for p in pmf],
               "n_samples": R, "carry_hit_given_full": 0.0,
               "carry_note": "NOT measured: 0.0 (carry disabled, conservative)",
               "provenance": "measured", "source": _relpath(path), "source_sha256": _sha(path),
               "hardware": meta["gpu"], "model": meta.get("model"), "model_rev": meta.get("model_rev"),
               "n_tokens": sum(x["n_committed"] for x in use), "n_rounds": R, "n_runs": len(use),
               "n_identical_runs": sum(x["identical"] for x in use),
               "tasks": sorted(x["task"] for x in use), "gamma_measured": gmax,
               "config_names": sorted({x["config"] for x in use}), "unused_configs": unused}
        # renormalise the 1e-12 rounding so parse_row's sum check is exact-ish, then validate through the real loader
        s = sum(row["accept_len_pmf"]); row["accept_len_pmf"][-1] = round(row["accept_len_pmf"][-1] + (1.0 - s), 12)
        parse_row(row, f"{path}:{kind}/{r}/{list(skip)}")
        if row["provenance"] != "measured" or row["n_tokens"] <= 0:
            raise Refuse("internal: non-measured output row")
        rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", nargs="+", default=[os.path.join(ROOT, "results", "spec.jsonl")],
                    help="measured source files (each needs a sibling <stem>.meta.json)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "accept_calibration.jsonl"))
    ap.add_argument("--dry-run", action="store_true", help="validate and print a summary, write nothing")
    a = ap.parse_args(argv)
    try:
        rows = []
        for p in a.spec:
            rows += build_rows(p)
        seen = set()
        for r in rows:       # AcceptTable forbids duplicate (workload, key): two sources must not collide silently
            k = (r["workload"], r["attn_budget"], r["r"], tuple(r["skip_layers"]))
            if k in seen:
                raise Refuse(f"duplicate calibration key {k} across sources; pass one source per hardware")
            seen.add(k)
    except Refuse as e:
        sys.stderr.write(f"REFUSED: {e}\n")
        return 2
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    for r in rows:
        print(f"{r['workload']:9s} r={r['r']} skip={len(r['skip_layers']):2d} gamma<={r['gamma_measured']} rounds={r['n_rounds']:3d} "
              f"tokens={r['n_tokens']:4d} E[commit@G]={1 + sum(sum(r['accept_len_pmf'][j:]) for j in range(1, len(r['accept_len_pmf']))):.3f} hw={r['hardware']}")
    if a.dry_run:
        print(f"[dry-run] {len(rows)} rows, nothing written"); return 0
    with open(a.out, "w") as f:
        f.write(text)
    print(f"wrote {len(rows)} measured rows -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
