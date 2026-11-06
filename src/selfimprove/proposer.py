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


def build_prompt(req: ProposalRequest) -> str:
    parts = [f"## Scope\n{req.scope_description}", f"## Base version\n{req.base_version}"]
    if req.profile_summary: parts.append(f"## Profile summary\n{req.profile_summary}")
    if req.dev_results: parts.append(f"## Dev results (dev split only)\n{req.dev_results}")
    if req.history: parts.append("## Earlier proposals this session\n" + "\n".join(f"- {h}" for h in req.history[-10:]))
    for p, t in sorted(req.files.items()):
        parts.append(f"## File: {p}\n```\n{t}\n```")
    parts.append("## Task\nPropose one bounded improvement as a unified diff against the files above.")
    return "\n\n".join(parts)


_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"```(?:diff|patch)[^\n]*\n(.*?)```", re.S)


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
        payload = dict(system=SYSTEM, prompt=prompt, max_new_tokens=req.max_new_tokens, temperature=req.temperature)
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
                                cost_usd=out.get("cost_usd", 0.0), latency_s=time.time() - t0, prompt=SYSTEM + "\n\n" + prompt,
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
                                endpoint="mock", prompt=SYSTEM + "\n\n" + build_prompt(req))
