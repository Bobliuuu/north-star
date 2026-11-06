"""Versioned edit store with append-only ledger and rollback. Layout:
  root/ledger.jsonl            every round (committed or rejected)
  root/CURRENT                 version id currently active ('' = none = unedited)
  root/vNNN/{adapters.pt,recipe.json,report.json}"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time

import torch


def state_hash(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().float().contiguous().numpy().tobytes())
    return h.hexdigest()


class EditStore:
    def __init__(self, root):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def versions(self):
        return sorted(p.name for p in self.root.glob("v[0-9][0-9][0-9]*") if p.is_dir())

    def current(self):
        p = self.root / "CURRENT"
        return p.read_text().strip() if p.exists() else ""

    def _ledger(self, entry):
        entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **entry}
        with (self.root / "ledger.jsonl").open("a") as f:
            f.write(json.dumps(entry, default=float) + "\n")
        return entry

    def ledger(self):
        p = self.root / "ledger.jsonl"
        return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []

    def commit(self, adapters, recipe, report, source):
        vid = f"v{len(self.versions()) + 1:03d}"
        d = self.root / vid
        d.mkdir()
        sd = {k: v.detach().cpu() for k, v in adapters.state_dict().items()}
        torch.save(sd, d / "adapters.pt")
        (d / "recipe.json").write_text(json.dumps(recipe.to_dict(), indent=1))
        (d / "report.json").write_text(json.dumps(report, indent=1, default=float))
        (self.root / "CURRENT").write_text(vid)
        self._ledger(dict(event="commit", version=vid, source=source, hash=state_hash(sd), passed=True,
                          numbers=report["numbers"]))
        return vid

    def reject(self, recipe, report, source, reason=""):
        return self._ledger(dict(event="reject", source=source, recipe=recipe.to_dict() if recipe else None,
                                 passed=False, gates=report.get("gates") if report else None,
                                 numbers=report.get("numbers") if report else None, reason=reason))

    def load_state(self, vid=None):
        vid = vid or self.current()
        return torch.load(self.root / vid / "adapters.pt") if vid else None

    def load_recipe(self, vid):
        from .recipe import Recipe
        return Recipe.from_dict(json.loads((self.root / vid / "recipe.json").read_text()))

    def rollback(self, to=""):
        """to='' -> unedited. Versions are kept on disk; only the pointer moves."""
        assert to == "" or to in self.versions(), to
        (self.root / "CURRENT").write_text(to)
        self._ledger(dict(event="rollback", to=to))
