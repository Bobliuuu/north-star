"""STRUCTURED-EDIT proposal mode, POLICY track (src/opt/policy_tables/spec_space.json).

Same contract as structured.py (kernel): North emits ONLY {"edits":[{"name":..,"value":..}], "rationale":".."}; this module
(developer infra) validates strictly and renders the unified diff deterministically by rewriting ONLY the value of the three
whitelisted keys. North chose every value; the harness only formatted. The diff then goes through the unchanged strict diff
parser, audit, sandbox, gates, store.

The three keys and their ranges mirror harness/selfimprove/policy_eval.py::load_table (a test pins the equivalence).
gammas and modes are SETS semantically, so order is normalised on render (gammas ascending, modes in opt.config.MODES order);
duplicates are rejected.

REMINDER: the policy metric is the Phase-3 optimizer's PREDICTED tok/s. A "win" here is a better model prediction, never a
measured speedup.
"""
from __future__ import annotations
import difflib, json, re
from .structured import StructuredEditError, extract_json_object, _THINK  # noqa: F401  (shared error type and JSON extraction)

POLICY_PATH = "src/opt/policy_tables/spec_space.json"
MODES = ("two_pass", "spec_two_pass", "fused")            # mirrors opt.config.MODES (pinned by a test)
GAMMA_LO, GAMMA_HI, GAMMA_MAX_LEN = 1, 12, 12
NG_LO, NG_HI = 2, 12
KEYS = ("gammas", "modes", "n_groups")
MAX_EDITS = 3
DESCR = {
    "gammas": f"list of DISTINCT integers in [{GAMMA_LO},{GAMMA_HI}] (1..{GAMMA_MAX_LEN} entries): candidate draft block lengths the optimizer may pick from",
    "modes": f"non-empty list of DISTINCT strings from {list(MODES)}: expert schedules the optimizer may pick from",
    "n_groups": f"integer in [{NG_LO},{NG_HI}]: layer-group granularity of the CP-SAT schedule",
}


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def parse_edits(raw: str) -> dict:
    """Strict schema validation. Returns {'edits': [(name, value)...], 'rationale': str}. Raises StructuredEditError."""
    txt = extract_json_object(raw)
    if txt is None: raise StructuredEditError("no_json_object")
    try:
        o = json.loads(txt, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    except ValueError as e:
        raise StructuredEditError(f"bad_json: {str(e)[:80]}")
    if not isinstance(o, dict): raise StructuredEditError("not_an_object")
    extra = set(o) - {"edits", "rationale"}
    if extra: raise StructuredEditError(f"unknown_keys: {sorted(extra)}")
    ed = o.get("edits")
    if not isinstance(ed, list): raise StructuredEditError("edits_not_a_list")
    rat = o.get("rationale", "")
    if not isinstance(rat, str): raise StructuredEditError("rationale_not_a_string")
    if len(ed) > MAX_EDITS: raise StructuredEditError(f"too_many_edits: {len(ed)} > {MAX_EDITS}")
    out, seen = [], set()
    for e in ed:
        if not isinstance(e, dict): raise StructuredEditError("edit_not_an_object")
        if set(e) != {"name", "value"}: raise StructuredEditError(f"edit_keys: {sorted(e)}")
        n, v = e["name"], e["value"]
        if not isinstance(n, str) or n not in KEYS: raise StructuredEditError(f"unknown_name: {str(n)[:40]!r}")
        if n in seen: raise StructuredEditError(f"duplicate_name: {n}")
        seen.add(n)
        if n == "gammas":
            if not isinstance(v, list) or not v: raise StructuredEditError("gammas_not_a_nonempty_list")
            if len(v) > GAMMA_MAX_LEN: raise StructuredEditError(f"gammas_too_long: {len(v)}")
            if any(not _is_int(x) for x in v): raise StructuredEditError("gammas_value_not_int")
            if any(not GAMMA_LO <= x <= GAMMA_HI for x in v): raise StructuredEditError(f"gammas_out_of_range: [{GAMMA_LO},{GAMMA_HI}]")
            if len(set(v)) != len(v): raise StructuredEditError("gammas_duplicate")
            v = sorted(v)
        elif n == "modes":
            if not isinstance(v, list) or not v: raise StructuredEditError("modes_not_a_nonempty_list")
            if any(not isinstance(x, str) or x not in MODES for x in v): raise StructuredEditError(f"modes_unknown: allowed {list(MODES)}")
            if len(set(v)) != len(v): raise StructuredEditError("modes_duplicate")
            v = [m for m in MODES if m in v]
        else:
            if not _is_int(v): raise StructuredEditError("n_groups_not_int")
            if not NG_LO <= v <= NG_HI: raise StructuredEditError(f"n_groups_out_of_range: [{NG_LO},{NG_HI}]")
        out.append((n, v))
    return dict(edits=out, rationale=rat[:1000])


def _line_pat(name):
    return re.compile(rf'^(\s*"{name}":\s*)(\[[^\]]*\]|\d+|"[^"]*")(\s*,?\s*)$')


def current_values(files: dict[str, str], path: str = POLICY_PATH) -> dict:
    try: t = json.loads(files.get(path, ""))
    except ValueError: return {k: None for k in KEYS}
    return {k: t.get(k) for k in KEYS}


def render_diff(edits, files: dict[str, str], path: str = POLICY_PATH) -> str:
    """Deterministic unified diff (difflib, 3 context lines) touching only the `"key": value` lines of the whitelisted keys.
    Returns '' for a no-op. Raises StructuredEditError if the file is missing or a key is not found as exactly one plain line."""
    if path not in files: raise StructuredEditError(f"file_missing: {path}")
    old = files[path]; lines = old.split("\n"); new = list(lines)
    for name, val in edits:
        pat = _line_pat(name)
        hits = [i for i, l in enumerate(lines) if pat.match(l)]
        if len(hits) != 1: raise StructuredEditError(f"not_editable_in_file: {name}")
        m = pat.match(lines[hits[0]])
        new[hits[0]] = f"{m[1]}{json.dumps(val)}{m[3]}"
    nt = "\n".join(new)
    try: t = json.loads(nt)
    except ValueError: raise StructuredEditError("rendered_file_not_json")
    if set(t) != set(KEYS): raise StructuredEditError("rendered_keys_changed")
    if new == lines: return ""
    return "".join(difflib.unified_diff([l + "\n" for l in lines], [l + "\n" for l in new], f"a/{path}", f"b/{path}", n=3))


def history_line(rec: dict) -> str:
    """One-line summary of an earlier proposal for the prompt (measured-history feed, like the kernel structured mode). The
    metric is a PREDICTED tok/s so the numbers are labelled 'predicted'. DEV split only; held-out numbers are never included."""
    line = f"{rec['proposal_id'][:8]}: {rec['verdict']}"
    ed = (rec.get("structured") or {}).get("edits")
    if ed is not None:
        line += " | edits " + (", ".join(f"{e['name']}={json.dumps(e['value'])}" for e in ed) if ed else "(none)")
    g = rec.get("gates") or {}
    sp = (g.get("speed") or {}).get("data", {}).get("dev") if isinstance(g.get("speed"), dict) else None
    if sp and sp.get("valid") and isinstance(sp.get("speedup"), (int, float)):
        line += (f" | PREDICTED dev speedup {sp['speedup']:.3f}x (model prediction, not a measurement; need > {sp.get('threshold', 1.03):.2f}; "
                 f"<1 means predicted SLOWER than base)")
    return line


GUIDANCE = (
    "Facts and strategy:\n"
    "- The score is the optimizer's PREDICTED tokens/s (measured MoE cost model + an acceptance table), deterministic for a given table; it is NOT a measured speedup.\n"
    "- The optimizer picks the best (gamma, mode) pair FROM your lists, so adding an option can only help the prediction or leave it unchanged, "
    "while removing an option can only hurt or leave it unchanged. A larger gamma needs the acceptance table to extend that far; unsupported values simply never win.\n"
    "- n_groups changes the CP-SAT schedule granularity: finer groups can find better schedules but cost solve time (2 s limit per bucket).\n"
    "- The BASELINE table is already a reasonable starting point; an empty edit list is better than a guess you expect to be worse.\n"
    "- Earlier proposals this session list their predicted dev speedup. Do NOT repeat a combination that was predicted <= 1.0; "
    "change one thing per proposal so the effect is attributable. The speed gate needs a predicted dev speedup above the stated threshold.\n")


def prompt_section(files: dict[str, str], dev_results: str = "") -> str:
    cur = current_values(files)
    rows = "\n".join(f"- {n} = {json.dumps(cur[n])}  allowed: {DESCR[n]}" for n in KEYS)
    return ("## Structured edit task\nChange the search space in src/opt/policy_tables/spec_space.json. Keys with CURRENT values:\n"
            f"{rows}\n"
            "Any other key is protected and cannot be edited.\n"
            "Output format (strict): ONE JSON object and nothing else (no prose, no markdown fences):\n"
            '{"edits":[{"name":"gammas","value":[1,2,3,4,5,6,8]}],"rationale":"one or two sentences"}\n'
            f"{GUIDANCE}"
            f"Rules: at most {MAX_EDITS} edits; each name at most once; values have exactly the JSON types shown (lists of ints / strings, int; no expressions); "
            "no other keys. Do NOT write a diff; the harness renders it. If you see no safe improvement output {\"edits\":[],\"rationale\":\"...\"}.")
