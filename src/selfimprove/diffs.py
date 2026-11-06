"""Strict unified-diff parser + applier (no `patch` binary, no fuzz, no renames/modes/binary/symlinks)."""
from __future__ import annotations
import posixpath, re
from dataclasses import dataclass, field
from .policy import Scope, editable_line_mask

class DiffError(ValueError):
    """Diff is malformed or violates scope; message is the machine-readable reason."""

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
BAD_CHARS = re.compile("[\x00​-‏‪-‮⁦-⁩﻿]")


@dataclass
class Hunk:
    old_start: int; old_len: int; new_start: int; new_len: int
    lines: list[tuple[str, str]]   # (' '|'+'|'-', text)

@dataclass
class FileDiff:
    path: str
    created: bool
    hunks: list[Hunk] = field(default_factory=list)

    def added(self):
        return [t for h in self.hunks for o, t in h.lines if o == "+"]
    def removed(self):
        return [t for h in self.hunks for o, t in h.lines if o == "-"]


def _clean(p: str) -> str:
    p = p.split("\t")[0].strip()
    if p == "/dev/null":
        return p
    if p.startswith(("a/", "b/")):
        p = p[2:]
    return p


def check_path(p: str):
    if not p or p.startswith("/") or "\\" in p or ":" in p.split("/")[0]:
        raise DiffError(f"bad path: {p!r}")
    n = posixpath.normpath(p)
    if n != p or n.startswith("..") or "/../" in "/" + n or n == ".":
        raise DiffError(f"path traversal / non-normal path: {p!r}")


def parse(diff: str, scope: Scope) -> list[FileDiff]:
    lim = scope.limits
    if not diff.strip():
        raise DiffError("empty diff")
    if len(diff.encode()) > lim.max_diff_bytes:
        raise DiffError("diff too large")
    if BAD_CHARS.search(diff):
        raise DiffError("NUL/zero-width/bidi control characters in diff")
    lines = diff.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    i, out = 0, []
    while i < len(lines):
        ln = lines[i]
        if ln.startswith(("diff --git", "index ")):
            i += 1; continue
        if ln.startswith(("old mode", "new mode", "rename ", "copy ", "similarity", "deleted file", "new file mode", "Binary", "GIT binary", "symlink")):
            raise DiffError(f"unsupported git extension: {ln[:40]!r}")
        if not ln.startswith("--- "):
            raise DiffError(f"unexpected line outside file header: {ln[:60]!r}")
        old = _clean(ln[4:]); i += 1
        if i >= len(lines) or not lines[i].startswith("+++ "):
            raise DiffError("missing +++ header")
        new = _clean(lines[i][4:]); i += 1
        if new == "/dev/null":
            raise DiffError("file deletion not allowed")
        if old == "/dev/null":
            created, path = True, new
        else:
            created, path = False, new
            if old != new:
                raise DiffError(f"rename/mismatched paths {old!r} -> {new!r}")
        if new == "/dev/null":
            raise DiffError("file deletion not allowed")
        check_path(path)
        fd = FileDiff(path, created)
        while i < len(lines) and lines[i].startswith("@@"):
            m = HUNK_RE.match(lines[i])
            if not m:
                raise DiffError(f"bad hunk header {lines[i][:60]!r}")
            os_, ol, ns, nl = int(m[1]), int(m[2] or 1), int(m[3]), int(m[4] or 1)
            i += 1; hl = []; co = cn = 0
            while i < len(lines) and (co < ol or cn < nl):
                l = lines[i]
                if l.startswith("\\"):
                    i += 1; continue
                if not l or l[0] not in " +-":
                    if l == "":      # blank context line with stripped space
                        l = " "
                    else:
                        raise DiffError(f"bad hunk line {l[:60]!r}")
                op, t = l[0], l[1:]
                hl.append((op, t))
                if op in " -": co += 1
                if op in " +": cn += 1
                i += 1
            if co != ol or cn != nl:
                raise DiffError("hunk line counts do not match header")
            while i < len(lines) and lines[i].startswith("\\"):
                i += 1
            fd.hunks.append(Hunk(os_, ol, ns, nl, hl))
        if not fd.hunks:
            raise DiffError(f"no hunks for {path}")
        out.append(fd)
    if not out:
        raise DiffError("no file diffs")
    if len(out) > lim.max_files:
        raise DiffError("too many files")
    if len({f.path for f in out}) != len(out):
        raise DiffError("duplicate file entries")
    if sum(len(f.hunks) for f in out) > lim.max_hunks:
        raise DiffError("too many hunks")
    if sum(len(f.added()) + len(f.removed()) for f in out) > lim.max_changed_lines:
        raise DiffError("too many changed lines")
    return out


def apply_one(fd: FileDiff, original: str | None, scope: Scope) -> str:
    """Apply with exact context (no fuzz). Enforces scope rule and editable-region membership of every changed line."""
    rule = scope.rule_for(fd.path)
    if rule is None:
        raise DiffError(f"path outside editable allowlist: {fd.path}")
    if fd.created:
        if not rule.allow_create or original is not None:
            raise DiffError(f"file creation not allowed: {fd.path}")
        if rule.regions_only:
            raise DiffError("cannot create regions_only file")
        return "\n".join(t for h in fd.hunks for o, t in h.lines if o == "+") + "\n"
    if original is None:
        raise DiffError(f"file does not exist: {fd.path}")
    src = original.split("\n")
    mask = editable_line_mask(original) if rule.regions_only else [True] * len(src)
    res, pos = [], 0   # pos: 0-based index into src
    for h in fd.hunks:
        start = h.old_start - 1 if h.old_len else h.old_start
        if start < pos:
            raise DiffError("overlapping/out-of-order hunks")
        res.extend(src[pos:start]); pos = start
        for op, t in h.lines:
            if op in " -":
                if pos >= len(src) or src[pos] != t:
                    raise DiffError(f"context mismatch in {fd.path} at old line {pos + 1}")
                if op == "-":
                    if not mask[pos]:
                        raise DiffError(f"edit outside editable region in {fd.path} line {pos + 1}")
                else:
                    res.append(src[pos])
                pos += 1
            else:
                # an insertion must sit between two editable lines (or adjacent to one) of an editable region
                before_ok = pos - 1 >= 0 and pos - 1 < len(mask) and mask[pos - 1]
                after_ok = pos < len(mask) and mask[pos]
                if rule.regions_only and not (before_ok or after_ok):
                    raise DiffError(f"insertion outside editable region in {fd.path} near line {pos + 1}")
                res.append(t)
    res.extend(src[pos:])
    new = "\n".join(res)
    if rule.regions_only:
        # markers must survive and regions must stay balanced
        from .policy import BEGIN_RE, END_RE
        if len(BEGIN_RE.findall(new)) != len(BEGIN_RE.findall(original)) or len(END_RE.findall(new)) != len(END_RE.findall(original)):
            raise DiffError("editable-region markers altered")
    return new


def apply_all(diffs: list[FileDiff], read, scope: Scope) -> dict[str, str]:
    """read(path)->str|None. Returns {path: new_text}; nothing written."""
    return {fd.path: apply_one(fd, read(fd.path), scope) for fd in diffs}
