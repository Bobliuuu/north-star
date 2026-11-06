"""Proposers. NorthProposer is the ONLY production proposer (remote North on Modal). MockProposer is test-only."""
from __future__ import annotations
import json, re, time
from dataclasses import dataclass, field, asdict
from . import NORTH_MODEL_ID, NORTH_REVISION


class ProvenanceError(RuntimeError):
    pass


@dataclass
class ProposalRequest:
    base_version: str
    files: dict[str, str]            # editable files (full text of current promoted version)
    scope_description: str           # allowlist + region rules + limits, human-readable
    profile_summary: str = ""
    dev_results: str = ""            # dev-split results ONLY (never held-out data)
    history: list[str] = field(default_factory=list)   # one-line summaries of earlier verdicts
    max_new_tokens: int = 2048
    temperature: float = 0.3
    track: str = "kernel"            # which structured-edit module renders/prompts: kernel | policy
    mode: str = "diff"               # 'diff' (unified diff) | 'structured' (JSON tunable edits, harness renders the diff)


@dataclass
class ProposalResponse:
    raw_response: str
    kind: str                        # 'north' | 'mock'
    model_id: str = ""
    revision: str = ""
    backend: str = ""
    endpoint: str = ""
    response_id: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    gpu_seconds: float = 0.0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    prompt: str = ""
    extra: dict = field(default_factory=dict)


SYSTEM = (
    "You are North, optimizing your own inference kernels and schedules. Reply with ONE unified diff in a ```diff fenced "
    "block, preceded by at most 5 lines of rationale. Rules: edit ONLY the files shown; change ONLY lines inside "
    "NS-EDITABLE-BEGIN/END regions (or whole files where stated); keep exact context lines; never touch tests, harness, "
    "timing, tolerances, held-out data, or the evaluator; do not special-case shapes, cache outputs, or change numerics "
    "beyond the stated tolerance. Your change is judged on numerical correctness, held-out quality, and repeatable speed "
    "on shapes you have not seen. If you see no safe improvement reply with an empty ```diff block."
)


SYSTEM_STRUCTURED = (
    "You are North, optimizing your own inference kernel tunables. Reply with ONLY one JSON object of the form "
    '{"edits":[{"name":"<TUNABLE>","value":<integer>}],"rationale":"<short>"} and nothing else. Choose values from the allowed '
    "ranges listed in the task. You never write code or diffs; the harness applies your values. Your change is judged on numerical "
    "correctness, held-out quality, and repeatable speed on shapes you have not seen. If you see no safe improvement reply with "
    '{"edits":[],"rationale":"..."}.')


SYSTEM_STRUCTURED_POLICY = (
    "You are North, optimizing your own speculative-decoding policy search space. Reply with ONLY one JSON object of the form "
    '{"edits":[{"name":"<KEY>","value":<value>}],"rationale":"<short>"} and nothing else. Choose values from the allowed '
    "types and ranges listed in the task. You never write code or diffs; the harness applies your values. Your change is judged on "
    "solver consistency, held-out quality and a repeatable PREDICTED-throughput gain on buckets you have not seen. If you see no safe "
    'improvement reply with {"edits":[],"rationale":"..."}.')


SYSTEM_STRUCTURED_TRITON = (
    "You are North, optimizing the launch parameters (tile sizes, num_warps, num_stages, split-K counts) of your own fast Triton decode kernels. "
    'Reply with ONLY one JSON object of the form {"edits":[{"name":"<PARAM>","value":<integer>}],"rationale":"<short>"} and nothing else. '
    "Choose values from the allowed lists in the task. You never write code or diffs; the harness applies your values. Your change is judged on "
    "numerical correctness against a torch reference (including a bitwise row-invariance check), drift of the full step, and repeatable CUDA-graph "
    "speed on context lengths and block sizes you have not seen. If you see no safe improvement reply with "
    '{"edits":[],"rationale":"..."}.')


def system_for(mode: str, track: str = "kernel") -> str:
    if mode != "structured": return SYSTEM
    return {"policy": SYSTEM_STRUCTURED_POLICY, "triton": SYSTEM_STRUCTURED_TRITON}.get(track, SYSTEM_STRUCTURED)


def build_prompt(req: ProposalRequest) -> str:
    if req.mode == "structured":
        from . import structured as ST, structured_policy as SP, structured_triton as STR
        parts = [f"## Scope\n{req.scope_description}", f"## Base version\n{req.base_version}"]
        if req.profile_summary: parts.append(f"## Profile summary\n{req.profile_summary}")
        if req.dev_results: parts.append(f"## Dev results (dev split only)\n{req.dev_results}")
        if req.history: parts.append("## Earlier proposals this session\n" + "\n".join(f"- {h}" for h in req.history[-24:]))
        for p, t in sorted(req.files.items()):
            parts.append(f"## File: {p}\n```\n{t}\n```")
        parts.append({"policy": SP, "triton": STR}.get(req.track, ST).prompt_section(req.files))
        return "\n\n".join(parts)
    parts = [f"## Scope\n{req.scope_description}", f"## Base version\n{req.base_version}"]
    if req.profile_summary: parts.append(f"## Profile summary\n{req.profile_summary}")
    if req.dev_results: parts.append(f"## Dev results (dev split only)\n{req.dev_results}")
    if req.history: parts.append("## Earlier proposals this session\n" + "\n".join(f"- {h}" for h in req.history[-24:]))
    for p, t in sorted(req.files.items()):
        parts.append(f"## File: {p}\n```\n{t}\n```")
    parts.append("## Task\nPropose one bounded improvement as a unified diff against the files above.\n"
                 "Output format (strict): one ```diff fenced block and nothing else. It must start with file headers, e.g.\n"
                 "--- a/<path>\n+++ b/<path>\n@@ -<old_start>,<old_len> +<new_start>,<new_len> @@\n"
                 " <context line>\n-<old line>\n+<new line>\n <context line>\n"
                 "Change as few lines as possible, only lines inside the editable markers, with at most 3 context lines per hunk. "
                 "Do not rewrite or repeat unchanged lines. Close the fence.")
    return "\n\n".join(parts)


_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"```(?:diff|patch)[^\n]*\n(.*?)(?:```|\Z)", re.S)   # tolerate an unclosed fence (truncated output); the strict parser then rejects it


def extract_diff(raw: str) -> str | None:
    """Deterministic: strip <think>, take the FIRST ```diff fence, else text from the first '--- ' header line. None if absent."""
    s = _THINK.sub("", raw)
    m = _FENCE.search(s)
    if m:
        d = m.group(1)
    else:
        i = re.search(r"(?m)^--- ", s)
        if not i: return None
        d = s[i.start():]
    d = d.strip("\n")
    return d + "\n" if d.strip() else ""


class Proposer:
    kind = "abstract"
    def propose(self, req: ProposalRequest) -> ProposalResponse: raise NotImplementedError


class NorthProposer(Proposer):
    """Calls the Modal North endpoint (src/selfimprove/modal_north.py). `transport(payload: dict) -> dict` is injectable;
    default resolves the deployed Modal class. The response MUST carry the pinned model id + revision or it is rejected."""
    kind = "north"

    def __init__(self, transport=None, app_name="northstar-selfimprove", cls_name="NorthProposerService"):
        self.transport, self.app_name, self.cls_name = transport, app_name, cls_name

    def _modal_transport(self, payload):
        import modal
        cls = modal.Cls.from_name(self.app_name, self.cls_name)
        return cls().propose.remote(payload)

    def propose(self, req: ProposalRequest) -> ProposalResponse:
        prompt = build_prompt(req)
        payload = dict(system=system_for(req.mode, req.track), prompt=prompt, max_new_tokens=req.max_new_tokens, temperature=req.temperature)
        t0 = time.time()
        out = (self.transport or self._modal_transport)(payload)
        if out.get("model_id") != NORTH_MODEL_ID or out.get("revision") != NORTH_REVISION:
            raise ProvenanceError(f"endpoint did not attest pinned North: {out.get('model_id')}@{out.get('revision')}")
        if not isinstance(out.get("text"), str):
            raise ProvenanceError("endpoint returned no text")
        return ProposalResponse(raw_response=out["text"], kind="north", model_id=out["model_id"], revision=out["revision"],
                                backend=out.get("backend", ""), endpoint=f"modal:{self.app_name}/{self.cls_name}",
                                response_id=out.get("response_id", ""), tokens_in=out.get("tokens_in", 0),
                                tokens_out=out.get("tokens_out", 0), gpu_seconds=out.get("gpu_seconds", 0.0),
                                cost_usd=out.get("cost_usd", 0.0), latency_s=time.time() - t0, prompt=system_for(req.mode, req.track) + "\n\n" + prompt,
                                extra=dict(out.get("extra", {})))


class MockProposer(Proposer):
    """TEST ONLY. Replays scripted raw responses. kind='mock' => never North-authored, never attested."""
    kind = "mock"
    LABEL = "MOCK-PROPOSER-NOT-NORTH"

    def __init__(self, responses: list[str]):
        self.responses, self.i = list(responses), 0

    def propose(self, req: ProposalRequest) -> ProposalResponse:
        raw = self.responses[self.i % len(self.responses)]; self.i += 1
        return ProposalResponse(raw_response=raw, kind="mock", model_id=self.LABEL, revision="", backend="mock",
                                endpoint="mock", prompt=system_for(req.mode, req.track) + "\n\n" + build_prompt(req))
