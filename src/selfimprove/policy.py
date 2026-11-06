"""Editable-scope policy: default deny. Only files/regions named here may change; everything else is protected."""
from __future__ import annotations
import fnmatch, re
from dataclasses import dataclass, field

BEGIN_RE = re.compile(r"NS-EDITABLE-BEGIN\b")
END_RE = re.compile(r"NS-EDITABLE-END\b")


@dataclass
class ScopeRule:
    pattern: str                 # fnmatch on repo-relative posix path
    regions_only: bool = False   # True: only lines between NS-EDITABLE-BEGIN/END markers may change
    allow_create: bool = False


@dataclass
class Limits:
    max_diff_bytes: int = 16_000
    max_files: int = 3
    max_hunks: int = 12
    max_changed_lines: int = 120


@dataclass
class Scope:
    rules: list[ScopeRule] = field(default_factory=list)
    limits: Limits = field(default_factory=Limits)

    def rule_for(self, path: str):
        for r in self.rules:
            if fnmatch.fnmatchcase(path, r.pattern):
                return r
        return None


# Default for this repo. src/kvm/moe_fused.cu is regions_only: the developer must wrap the tunables in
# `// NS-EDITABLE-BEGIN` / `// NS-EDITABLE-END` markers (until then NO edit to it can apply: fail closed).
DEFAULT_SCOPE = Scope(rules=[
    ScopeRule("src/kvm/moe_fused.cu", regions_only=True),
    ScopeRule("src/opt/policy_tables/*.json"),
    ScopeRule("src/opt/policy_tables/*.py"),
])


def editable_line_mask(text: str) -> list[bool]:
    """mask[i] True if 0-based line i is strictly inside an editable region (marker lines themselves are not)."""
    mask, inside = [], False
    for ln in text.split("\n"):
        if BEGIN_RE.search(ln):
            inside = True; mask.append(False)
        elif END_RE.search(ln):
            inside = False; mask.append(False)
        else:
            mask.append(inside)
    return mask
