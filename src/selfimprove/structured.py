"""STRUCTURED-EDIT proposal mode (tunables track).

North emits ONLY a JSON object {"edits":[{"name":..,"value":<int>}], "rationale":".."}. This module (developer infra) validates
it strictly and renders the unified diff deterministically by rewriting ONLY the value of constant lines that sit inside the
NS-EDITABLE region. Authorship: North chose every value; the harness only formatted. The rendered diff then goes through the
unchanged strict parser, audit, sandbox, gates, store.
"""
from __future__ import annotations
import difflib, json, re
from .policy import editable_line_mask

KERNEL_PATH = "src/kvm/moe_fused.cu"
H_DIM = 2048
# name -> (lo, hi, multiple_of, human description). Mirrors the comments + static_asserts in src/kvm/moe_fused.cu.
TUNABLES = {
    "KC": (16, 2048, 16, "k chunk of H staged in smem; multiple of 16 that divides H=2048"),
    "XPAD": (0, 64, 8, "smem row padding (bf16 elems) for the x tile; multiple of 8"),
    "HPAD": (0, 64, 8, "smem row padding (bf16 elems) for the h tile; multiple of 8"),
    "MT_SPLIT": (1, 16, 1, "experts with pair_cnt > MT_SPLIT use the 2-row-tile path"),
    "MINB_PER_SM": (1, 8, 1, "min resident blocks per SM hint for __launch_bounds__"),
    "UNROLL_KS": (1, 128, 1, "#pragma unroll factor of the stage-1 k-step loop (>= KC/16 = full unroll, 1 = none)"),
    "UNROLL_S2": (1, 4, 1, "#pragma unroll factor of the stage-2 (down projection) k-step loop; 4 = full unroll"),
    "UNROLL_NT": (1, 8, 1, "#pragma unroll factor of the stage-2 output-tile loop (each warp walks H/16/4 = 32 tiles); 1 = none"),
}
MAX_EDITS = 3
STATIC_SMEM_LIMIT = 48 * 1024


class StructuredEditError(ValueError):
    """Machine-readable reason in args[0]."""


_THINK = re.compile(r"<think>.*?</think>", re.S)


def extract_json_object(raw: str) -> str | None:
    """Strip <think>, return the text of the FIRST balanced top-level {...} (markdown fences tolerated, string-aware)."""
    s = _THINK.sub("", raw)
    i = s.find("{")
    if i < 0: return None
    depth, in_str, esc = 0, False, False
    for j in range(i, len(s)):
        c = s[j]
        if in_str:
            if esc: esc = False
            elif c == "\\": esc = True
            elif c == '"': in_str = False
        elif c == '"': in_str = True
        elif c == "{": depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0: return s[i:j + 1]
    return None


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
        if not isinstance(n, str) or n not in TUNABLES: raise StructuredEditError(f"unknown_name: {str(n)[:40]!r}")
        if n in seen: raise StructuredEditError(f"duplicate_name: {n}")
        seen.add(n)
        if isinstance(v, bool) or not isinstance(v, int): raise StructuredEditError(f"value_not_int: {n}")
        lo, hi, mult, _ = TUNABLES[n]
        if not (lo <= v <= hi): raise StructuredEditError(f"out_of_range: {n}={v} not in [{lo},{hi}]")
        if v % mult: raise StructuredEditError(f"not_multiple_of_{mult}: {n}={v}")
        if n == "KC" and H_DIM % v: raise StructuredEditError(f"KC_must_divide_{H_DIM}: {v}")
        out.append((n, v))
    return dict(edits=out, rationale=rat[:1000])


def _const(src: str, name: str):
    m = re.search(rf"(?m)^\s*constexpr\s+int\s+{name}\s*=\s*(\d+)\s*;", src)
    return int(m.group(1)) if m else None


def render_diff(edits, files: dict[str, str], path: str = KERNEL_PATH) -> str:
    """Deterministic unified diff (difflib, 3 context lines) touching only `constexpr int NAME = <int>;` lines inside the region.
    Returns '' for a no-op. Raises StructuredEditError if a name is not found as a plain constant inside the editable region."""
    if path not in files: raise StructuredEditError(f"file_missing: {path}")
    old = files[path]; lines = old.split("\n"); mask = editable_line_mask(old); new = list(lines)
    for name, val in edits:
        pat = re.compile(rf"^(\s*constexpr\s+int\s+{name}\s*=\s*)(\d+)(\s*;.*)$")
        hits = [i for i, l in enumerate(lines) if pat.match(l)]
        if len(hits) != 1 or not mask[hits[0]]:
            raise StructuredEditError(f"not_editable_in_region: {name}")
        m = pat.match(lines[hits[0]]); new[hits[0]] = f"{m[1]}{val}{m[3]}"
    # combined smem check (static_assert would fail the build anyway; fail early and cheaply)
    nt = "\n".join(new); mt, sl = _const(nt, "MAXTOK"), _const(nt, "SL")
    kc, xp, hp = (_const(nt, k) for k in ("KC", "XPAD", "HPAD"))
    if None not in (mt, sl, kc, xp, hp) and (mt * (kc + xp) + mt * (sl + hp)) * 2 + 4 * 256 * 4 + mt * 8 > STATIC_SMEM_LIMIT:
        raise StructuredEditError("static_smem_exceeds_48KB")
    if new == lines: return ""
    d = "".join(difflib.unified_diff([l + "\n" for l in lines], [l + "\n" for l in new], f"a/{path}", f"b/{path}", n=3))
    return d


def current_values(files: dict[str, str], path: str = KERNEL_PATH) -> dict[str, int | None]:
    src = files.get(path, "")
    return {n: _const(src, n) for n in TUNABLES}


def history_line(rec: dict) -> str:
    """One-line summary of an earlier proposal record for the prompt: id, verdict and, when present, the chosen edits and the
    measured DEV-split speedup. Held-out numbers are never included (prompt must carry dev data only)."""
    line = f"{rec['proposal_id'][:8]} (vs {rec.get('base_version', '?')}): {rec['verdict']}"
    if rec["verdict"] in ("invalid_edit", "duplicate_edit", "build_fail") and rec.get("reason"): line += f" [{str(rec['reason'])[:160]}]"
    if rec["verdict"] == "gate_fail:correctness":
        cg = (rec.get("gates") or {}).get("correctness") or {}
        line += f" [{str(cg.get('reason', ''))[:100]}]"
    ed = (rec.get("structured") or {}).get("edits")
    if ed is not None:
        line += " | edits " + (", ".join(f"{e['name']}={e['value']}" for e in ed) if ed else "(none)")
    g = rec.get("gates") or {}
    sp = (g.get("speed") or {}).get("data", {}).get("dev") if isinstance(g.get("speed"), dict) else None
    if sp and sp.get("valid") and isinstance(sp.get("speedup"), (int, float)):
        line += f" | measured dev speedup {sp['speedup']:.3f}x (95% CI lower {sp.get('ci_lo', float('nan')):.3f}; need > {sp.get('threshold', 1.03):.2f}; <1 means SLOWER than base)"
    return line


def history_from_store(log_path) -> list[str]:
    """Dev-only history lines from an earlier store's proposals.jsonl (to seed a new session with prior measured outcomes)."""
    import pathlib
    out = []
    for l in pathlib.Path(log_path).read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            if r.get("event") == "proposal": out.append(history_line(r))
    return out


# Developer-written, measurement-free facts about the kernel (no held-out information).
GUIDANCE = (
    "Facts and strategy:\n"
    "- The BASELINE values above are the hand-tuned defaults and are a strong starting point; the kernel is HBM-bandwidth bound.\n"
    "- Dev launches route only 2..32 tokens in total (Tv+Td), top-8 of 128 experts, so most experts see only a few tokens "
    "(uniform expectation = 8*T/128 pairs per expert). The 2-row-tile path (pair_cnt > MT_SPLIT) doubles accumulator registers and "
    "MMA work on padded rows for the SAME weight traffic; lowering MT_SPLIT moves more experts onto it. Raising MT_SPLIT back to 16 is the maximum.\n"
    "- smem rows are read by 16x16 WMMA loads in 16-byte chunks. A row stride (in bytes) that is an ODD multiple of 16 spreads 8 rows over "
    "all 8 bank groups (conflict free); stride = 128B*k + 16B*odd works, e.g. pad 8 or 24 or 40. Pad 0 gives 8-way conflicts and pad 16 gives 2-way conflicts.\n"
    "- KC trades smem footprint and sync count: smaller KC means more __syncthreads rounds and shorter weight-load bursts; larger KC means more smem.\n"
    "- Compile profile of the BASELINE (ptxas resource usage, developer-recorded, no timing): 252 registers/thread, 0 spills, 128 threads/block, "
    "~35 KB static smem. Resident blocks per SM are therefore limited by REGISTERS (64K per SM), not smem. Tunables that shrink the fully "
    "unrolled register-hungry loops (smaller KC, smaller UNROLL_KS / UNROLL_S2, UNROLL_NT stays 1) or cap registers (MINB_PER_SM) can raise "
    "occupancy; a register cap that is too tight forces local-memory spills and is slower. UNROLL_S2 (max 4) and UNROLL_NT (max 8) tune the stage-2 down-projection loops.\n"
    "- Earlier proposals this session list their measured dev speedup, including ones from this same round. Do NOT repeat a combination that "
    "was already measured; a repeated edit set is skipped as duplicate_edit and wastes your turn. Output AT MOST 3 edits (more is rejected as invalid_edit). "
    "Prefer ONE-factor-at-a-time moves on a tunable you have not tried yet, so each result is attributable, then combine moves that each measured >= 1.0, and move AWAY from directions that already measured slower. "
    "The speed gate needs a dev speedup above the stated threshold, so an empty edit list is better than a guess you expect to be slower.\n")


def prompt_section(files: dict[str, str], dev_results: str = "") -> str:
    cur = current_values(files)
    rows = "\n".join(f"- {n} = {cur[n]}  allowed: integers in [{lo},{hi}]" + (f", multiple of {m}" if m > 1 else "") + f"  ({d})"
                     for n, (lo, hi, m, d) in TUNABLES.items())
    return ("## Structured edit task\nChange tunables of src/kvm/moe_fused.cu (inside NS-EDITABLE-BEGIN/END). Tunables with CURRENT values:\n"
            f"{rows}\n"
            "Also protected by static_assert: (MAXTOK*(KC+XPAD) + MAXTOK*(SL+HPAD))*2 + 4096 + MAXTOK*8 <= 49152 bytes (MAXTOK=32, SL=64). "
            "Anything not listed (MAXTOK, SL, H, I, TOPK, ...) is protected and cannot be edited.\n"
            "Output format (strict): ONE JSON object and nothing else (no prose, no markdown fences):\n"
            '{"edits":[{"name":"KC","value":128}],"rationale":"one or two sentences"}\n'
            f"{GUIDANCE}"
            f"Rules: at most {MAX_EDITS} edits; each name at most once; values are integers (no strings, no expressions); no other keys. "
            "Do NOT write a diff; the harness renders it. If you see no safe improvement output {\"edits\":[],\"rationale\":\"...\"}.")
