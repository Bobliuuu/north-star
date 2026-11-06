"""Version store (promote / rollback) + hash-chained JSONL log of EVERY proposal and attribution accounting."""
from __future__ import annotations
import hashlib, json, os, shutil, time
from pathlib import Path


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


class Store:
    def __init__(self, root: Path):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "versions").mkdir(exist_ok=True)
        self.log_path = self.root / "proposals.jsonl"

    # ---- JSONL log (append-only, fsync, hash chained) -------------------------------------------------------------
    def _last_hash(self) -> str:
        if not self.log_path.exists(): return "0" * 64
        last = None
        with open(self.log_path) as f:
            for ln in f:
                if ln.strip(): last = ln
        return json.loads(last)["hash"] if last else "0" * 64

    def log(self, record: dict) -> dict:
        rec = dict(record); rec["ts"] = rec.get("ts", time.time()); rec["prev_hash"] = self._last_hash()
        rec.pop("hash", None)
        rec["hash"] = sha256_text(json.dumps(rec, sort_keys=True, default=str))
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n"); f.flush(); os.fsync(f.fileno())
        return rec

    def read_log(self) -> list[dict]:
        if not self.log_path.exists(): return []
        return [json.loads(l) for l in open(self.log_path) if l.strip()]

    def verify_log(self) -> bool:
        prev = "0" * 64
        for r in self.read_log():
            h = r.pop("hash")
            if r["prev_hash"] != prev or sha256_text(json.dumps(r, sort_keys=True, default=str)) != h:
                return False
            prev = h
        return True

    # ---- versions -----------------------------------------------------------------------------------------------------
    def versions(self) -> list[str]:
        return sorted(p.name for p in (self.root / "versions").iterdir() if p.is_dir() and not p.name.startswith("."))

    def current(self) -> str | None:
        p = self.root / "CURRENT"
        return p.read_text().strip() if p.exists() else None

    def meta(self, v: str) -> dict:
        return json.loads((self.root / "versions" / v / "meta.json").read_text())

    def files(self, v: str) -> dict[str, str]:
        base = self.root / "versions" / v / "files"
        return {p.relative_to(base).as_posix(): p.read_text() for p in sorted(base.rglob("*")) if p.is_file()}

    def _set_current(self, v: str):
        tmp = self.root / "CURRENT.tmp"; tmp.write_text(v + "\n"); os.replace(tmp, self.root / "CURRENT")

    def _commit_version(self, files: dict[str, str], meta: dict) -> str:
        n = len(self.versions()); v = f"v{n:04d}"
        tmp = self.root / "versions" / f".tmp_{v}"
        if tmp.exists(): shutil.rmtree(tmp)
        for rel, text in files.items():
            p = tmp / "files" / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)
        meta = dict(meta, version=v, created=time.time(), files_sha256={k: sha256_text(t) for k, t in files.items()})
        (tmp / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
        os.rename(tmp, self.root / "versions" / v)
        return v

    def init_baseline(self, files: dict[str, str], note="developer baseline") -> str:
        if self.versions(): raise RuntimeError("store already initialised")
        v = self._commit_version(files, dict(parent=None, authored_by="developer", north_attested=False, note=note))
        self._set_current(v); self.log(dict(event="init", version=v)); return v

    def promote(self, new_files: dict[str, str], proposal_id: str, authored_by: str, north_attested: bool, diff: str,
                gates: dict, mock: bool, extra: dict | None = None) -> str:
        assert authored_by in ("north", "mock"), authored_by
        if authored_by == "north" and (mock or not north_attested):
            raise RuntimeError("refusing to label a version North-authored without attestation")
        parent = self.current()
        files = dict(self.files(parent)); files.update(new_files)
        v = self._commit_version(files, dict(parent=parent, authored_by=authored_by, north_attested=north_attested, mock=mock,
                                             proposal_id=proposal_id, diff=diff, diff_sha256=sha256_text(diff), gates=gates, **(extra or {})))
        self._set_current(v); self.log(dict(event="promote", version=v, parent=parent, proposal_id=proposal_id, authored_by=authored_by))
        return v

    def rollback(self, to: str | None = None, reason: str = "") -> str:
        cur = self.current()
        target = to or self.meta(cur)["parent"]
        if target is None or target not in self.versions():
            raise RuntimeError("nothing to roll back to")
        self._set_current(target); self.log(dict(event="rollback", from_version=cur, to_version=target, reason=reason))
        return target

    # ---- attribution --------------------------------------------------------------------------------------------------
    def attribution(self) -> dict:
        """Lineage of CURRENT: which versions/lines are North-authored vs developer infra vs mock (test-only)."""
        chain, v = [], self.current()
        while v:
            m = self.meta(v); chain.append(m); v = m["parent"]
        chain.reverse()
        out = dict(current=self.current(), north_versions=[], mock_versions=[], developer_versions=[], north_lines_added=0,
                   north_lines_removed=0, note="'north' only counts versions whose proposer was attested as the pinned North endpoint")
        for m in chain:
            if m["authored_by"] == "developer": out["developer_versions"].append(m["version"])
            elif m["authored_by"] == "north" and m.get("north_attested"):
                out["north_versions"].append(m["version"])
                for ln in m.get("diff", "").splitlines():
                    if ln.startswith("+") and not ln.startswith("+++"): out["north_lines_added"] += 1
                    elif ln.startswith("-") and not ln.startswith("---"): out["north_lines_removed"] += 1
            else: out["mock_versions"].append(m["version"])
        out["any_mock_in_lineage"] = bool(out["mock_versions"])
        out["reportable_as_north_authored"] = bool(out["north_versions"]) and not out["mock_versions"]
        return out

    def cost_summary(self) -> dict:
        tot = dict(proposals=0, proposer_usd=0.0, proposer_gpu_s=0.0, eval_wall_s=0.0, eval_usd=0.0, mock_proposals=0)
        for r in self.read_log():
            if r.get("event") != "proposal": continue
            c = r.get("cost", {}); tot["proposals"] += 1
            tot["proposer_usd"] += c.get("proposer_usd", 0) or 0; tot["proposer_gpu_s"] += c.get("proposer_gpu_s", 0) or 0
            tot["eval_wall_s"] += c.get("eval_wall_s", 0) or 0; tot["eval_usd"] += c.get("eval_usd", 0) or 0
            tot["mock_proposals"] += r.get("proposer", {}).get("kind") == "mock"
        return tot
