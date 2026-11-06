"""STRUCTURED-EDIT proposal mode, TRITON track (src/tritonk/tuning.py: launch parameters of the fast Triton decode kernels).

Same contract as structured.py (CUDA tunables track): North emits ONLY {"edits":[{"name":..,"value":<int>}], "rationale":".."}; this
module (developer infra) validates every name/value against an explicit allowed set and renders the unified diff by rewriting ONLY the
value of `NAME = <int>` lines inside the NS-EDITABLE region. North chose every value; the harness only formatted. The diff then goes
through the unchanged strict diff parser, audit, sandbox, gates, store.
"""
from __future__ import annotations
import difflib, json, re
from .policy import editable_line_mask
from .structured import StructuredEditError, extract_json_object, history_line as _hist_cuda  # noqa: F401

PATH = "src/tritonk/tuning.py"
MAX_EDITS = 4


def _p2(lo, hi):
    return [2 ** i for i in range(0, 20) if lo <= 2 ** i <= hi]


_W = [1, 2, 4, 8, 16]
_S = [1, 2, 3, 4, 5, 6]
# name -> (allowed values, description)
TUNABLES: dict[str, tuple[list[int], str]] = {
    "GU_BN": (_p2(2, 256), "T=1 expert gate_up GEMV: output rows per program (grid = 8 experts x 768/BN)"),
    "GU_BK": (_p2(16, 1024), "T=1 gate_up: k-tile of the reduction over 2048 (loop of 2048/BK iterations; tile = BN x BK fp32 for gate AND up)"),
    "GU_NW": (_W, "T=1 gate_up: num_warps"),
    "GU_NS": (_S, "T=1 gate_up: num_stages (software pipelining depth; Triton default 3)"),
    "DN_BN": (_p2(2, 512), "T=1 expert down GEMV: output rows per program (grid = 8 experts x 2048/BN)"),
    "DN_BK": (_p2(16, 256), "T=1 down: k-tile of the reduction over 768 (must divide 768)"),
    "DN_NW": (_W, "T=1 down: num_warps"),
    "DN_NS": (_S, "T=1 down: num_stages"),
    "ATT_NSPLIT": (_p2(4, 128), "T=1 split-K flash-decode: number of key splits per kv-head (grid = 4 kv heads x NSPLIT); combine kernel reduces them"),
    "ATT_BN": (_p2(16, 256), "T=1 attention: keys per inner iteration (tl.dot tile)"),
    "ATT_NW": (_W[:4], "T=1 attention split kernel: num_warps"),
    "ATT_NS": (_S[:5], "T=1 attention split kernel: num_stages"),
    "CMB_NW": (_W[:4], "T=1 attention combine kernel: num_warps"),
    "ROPE_NW": (_W[:3], "rope + KV-cache write kernel: num_warps"),
    "RN_NW": (_W, "fused residual-add + RMSNorm kernel (one program per row of 2048): num_warps"),
    "TK_NW": (_W[:3], "router top-8 + sigmoid kernel (128 logits): num_warps"),
    "XA_CHUNK": (_p2(64, 1024), "EXACT-mode attention: absolute key chunk per split program (more chunks = more parallelism, bigger combine)"),
    "XA_BN": (_p2(16, 128), "EXACT-mode attention: keys per inner iteration (tl.dot tile, query rows padded to 128)"),
    "XA_NW": ([2, 4, 8, 16], "EXACT-mode attention split kernel: num_warps"),
    "XA_NS": (_S[:5], "EXACT-mode attention split kernel: num_stages"),
    "XCMB_NW": (_W[:4], "EXACT-mode attention combine kernel: num_warps"),
    "ROPEB_NW": (_W[:3], "EXACT-mode rope + KV write kernel: num_warps"),
    "LIN_BN_S": (_p2(16, 128), "EXACT-mode linear (tl.dot, tokens padded to 16) for output width N <= 4096: output rows per program"),
    "LIN_BN_L": (_p2(16, 256), "EXACT-mode linear for N > 4096 (qkv 5120, lm_head 262144): output rows per program"),
    "LIN_BK_A": (_p2(32, 512), "EXACT-mode linear: k-tile when K <= 2048"),
    "LIN_BK_B": (_p2(32, 512), "EXACT-mode linear: k-tile when K > 2048 (o_proj K=4096)"),
    "LIN_NW_S": (_W[:4], "EXACT-mode linear N <= 4096: num_warps"),
    "LIN_NS_S": (_S[:5], "EXACT-mode linear N <= 4096: num_stages"),
    "LIN_NW_L": (_W[:4], "EXACT-mode linear N > 4096: num_warps"),
    "LIN_NS_L": (_S[:5], "EXACT-mode linear N > 4096: num_stages"),
    "MB_BN1": (_p2(16, 64), "EXACT/verify-block expert gate_up (tl.dot, tokens padded to 16): output rows per program"),
    "MB_BK1": (_p2(32, 256), "EXACT/verify-block gate_up: k-tile over 2048"),
    "MB_NW1": ([2, 4, 8], "EXACT/verify-block gate_up: num_warps"),
    "MB_NS1": (_S[:5], "EXACT/verify-block gate_up: num_stages"),
    "MB_BN2": (_p2(16, 64), "EXACT/verify-block expert down: output rows per program"),
    "MB_BK2": (_p2(32, 256), "EXACT/verify-block down: k-tile over 768 (must divide 768)"),
    "MB_NW2": ([2, 4, 8], "EXACT/verify-block down: num_warps"),
    "MB_NS2": (_S[:5], "EXACT/verify-block down: num_stages"),
    "AM_CH": (_p2(1024, 16384), "argmax over the 262144-wide logits: chunk per program of stage 1"),
    "AM_NW": (_W, "argmax stage 1: num_warps"),
}
# cross-knob resource caps (a violation would fail Triton compilation or spill massively; rejected early and cheaply)
CAPS = [("GU_BN", "GU_BK", 16384), ("DN_BN", "DN_BK", 16384), ("LIN_BN_S", "LIN_BK_A", 32768), ("LIN_BN_S", "LIN_BK_B", 32768),
        ("LIN_BN_L", "LIN_BK_A", 32768), ("LIN_BN_L", "LIN_BK_B", 32768), ("MB_BN1", "MB_BK1", 8192), ("MB_BN2", "MB_BK2", 8192)]
DIVIDES = {"DN_BK": 768, "MB_BK2": 768}


def current_values(files: dict[str, str], path: str = PATH) -> dict[str, int | None]:
    src = files.get(path, "")
    out = {}
    for n in TUNABLES:
        m = re.search(rf"(?m)^{n}\s*=\s*(\d+)\s*$", src)
        out[n] = int(m.group(1)) if m else None
    return out


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
        if v not in TUNABLES[n][0]: raise StructuredEditError(f"value_not_allowed: {n}={v} not in {TUNABLES[n][0]}")
        if n in DIVIDES and DIVIDES[n] % v: raise StructuredEditError(f"{n}_must_divide_{DIVIDES[n]}: {v}")
        out.append((n, v))
    return dict(edits=out, rationale=rat[:1000])


def render_diff(edits, files: dict[str, str], path: str = PATH) -> str:
    """Deterministic unified diff (difflib, 3 context lines) touching only `NAME = <int>` lines inside the region. '' for a no-op."""
    if path not in files: raise StructuredEditError(f"file_missing: {path}")
    old = files[path]; lines = old.split("\n"); mask = editable_line_mask(old); new = list(lines)
    for name, val in edits:
        pat = re.compile(rf"^({name}\s*=\s*)(\d+)(\s*)$")
        hits = [i for i, l in enumerate(lines) if pat.match(l)]
        if len(hits) != 1 or not mask[hits[0]]:
            raise StructuredEditError(f"not_editable_in_region: {name}")
        m = pat.match(lines[hits[0]]); new[hits[0]] = f"{m[1]}{val}{m[3]}"
    nv = {}
    for l in new:
        m = re.match(r"^([A-Z_0-9]+)\s*=\s*(\d+)\s*$", l)
        if m: nv[m[1]] = int(m[2])
    for a, b, cap in CAPS:
        if nv.get(a, 0) * nv.get(b, 0) > cap: raise StructuredEditError(f"tile_too_large: {a}*{b}={nv[a] * nv[b]} > {cap}")
    if new == lines: return ""
    return "".join(difflib.unified_diff([l + "\n" for l in lines], [l + "\n" for l in new], f"a/{path}", f"b/{path}", n=3))


def history_line(rec: dict) -> str:
    return _hist_cuda(rec)


GUIDANCE = (
    "Facts and strategy (developer-written; they contain no measurements):\n"
    "- The BASELINE values are the hand-set launch parameters of the fast executor and are a strong starting point. The expert GEMV kernels and the "
    "split-K attention are memory-bandwidth bound; the lm_head and router linears are small and latency bound.\n"
    "- T=1 path: GU_* / DN_* (expert gate_up and down GEMV), ATT_* / CMB_NW / ROPE_NW (split-K attention), RN_NW, TK_NW. EXACT path (T=1 and verify blocks T=2..16 run the "
    "same batch-invariant kernels): XA_* / XCMB_NW / ROPEB_NW, LIN_*, MB_*, AM_*. The benchmark times both paths, at several context lengths and verify-block sizes.\n"
    "- Correctness is checked against a torch reference, and the exact path must stay BITWISE row-invariant: any value is fine as long as it is the SAME for T=1 and block rows, "
    "which these knobs guarantee, so a bitwise failure means the value is unsound for some shape.\n"
    "- The speed gate is measured on context lengths and block sizes you do not see; do not tune to one context length. Values that are only good at one size lose on the others.\n"
    "- num_stages is software-pipelining depth (more stages need more shared memory); num_warps trades parallelism per program against register pressure; "
    "larger BN/BK tiles reuse more of each loaded row but reduce the number of programs.\n"
    "- Earlier proposals this session list their measured dev speedup. Do NOT repeat a combination already measured (it is skipped as duplicate_edit and wastes your turn). "
    f"Output AT MOST {MAX_EDITS} edits. Prefer ONE-factor-at-a-time moves on a knob you have not tried yet, so results are attributable, then combine moves that each measured >= 1.0, "
    "and move AWAY from directions that measured slower. An empty edit list is better than a guess you expect to be slower.\n")


def prompt_section(files: dict[str, str], dev_results: str = "") -> str:
    cur = current_values(files)
    rows = "\n".join(f"- {n} = {cur[n]}  allowed: {TUNABLES[n][0]}  ({TUNABLES[n][1]})" for n in TUNABLES)
    return ("## Structured edit task\nChange launch parameters in src/tritonk/tuning.py (inside NS-EDITABLE-BEGIN/END). Parameters with CURRENT values:\n"
            f"{rows}\n"
            f"Resource caps (violations are rejected): {', '.join(f'{a}*{b} <= {c}' for a, b, c in CAPS)}.\n"
            "Output format (strict): ONE JSON object and nothing else (no prose, no markdown fences):\n"
            '{"edits":[{"name":"GU_BK","value":128}],"rationale":"one or two sentences"}\n'
            f"{GUIDANCE}"
            f"Rules: at most {MAX_EDITS} edits; each name at most once; values are integers from the allowed list; no other keys. "
            "Do NOT write a diff; the harness renders it. If you see no safe improvement output {\"edits\":[],\"rationale\":\"...\"}.")
