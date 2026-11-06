"""Version store (promote / rollback) + hash-chained JSONL log of EVERY proposal and attribution accounting."""
from __future__ import annotations
import hashlib, json, os, shutil, time
from pathlib import Path


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


class StoreCorrupt(RuntimeError):
    """Chained log, version contents/meta and CURRENT disagree (tampering or a torn write)."""


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

    def head(self) -> str:
        """Hash of the newest log record. Anchor it OUTSIDE the store (app logs print every record's hash): the chain alone cannot
        detect truncation or a full rewrite by someone who can write the file."""
        recs = self.read_log()
        return recs[-1]["hash"] if recs else "0" * 64

    def head_record(self, label: str = "") -> str:
        """One line to commit to RUNS.md after every round (the external anchor): `selfimprove-head <store> n=<records> head=<hash>`."""
        n = len(self.read_log())
        return f"selfimprove-head {label or self.root.name} n={n} head={self.head()}"

    def verify_anchor(self, anchored_head: str, exact: bool = True) -> bool:
        """Chain verifies AND the externally anchored head matches. exact=True: it must be the CURRENT head (nothing appended/removed
        since the anchor was committed). exact=False: it may be any earlier record hash in the chain (the log may have grown since);
        a rewrite or truncation below the anchor still fails."""
        if not self.verify_log(): return False
        hs = [r["hash"] for r in self.read_log()] or ["0" * 64]
        return hs[-1] == anchored_head if exact else anchored_head in hs

    def verify_log(self, expected_head: str | None = None) -> bool:
        prev = "0" * 64
        try:
            recs = self.read_log()
            for r in recs:
                h = r.pop("hash")
                if r["prev_hash"] != prev or sha256_text(json.dumps(r, sort_keys=True, default=str)) != h:
                    return False
                prev = h
        except (ValueError, KeyError, TypeError, AttributeError):
            return False
        return expected_head is None or prev == expected_head

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

    # ---- integrity: the chained log is the write-ahead log. Order of every state change is
    #      (1) write the version dir (unreferenced, harmless if we crash)  (2) append the log record, which pins the version's
    #      meta.json hash (and through it every file hash)  (3) switch CURRENT.  CURRENT is therefore never ahead of the log;
    #      a crash between (2) and (3) leaves the log ahead, which recover() completes.
    def _meta_hash(self, v: str) -> str:
        return sha256_text((self.root / "versions" / v / "meta.json").read_text())

    def check_version(self, v: str, log_meta_sha256: str | None = None):
        d = self.root / "versions" / v
        if not (d / "meta.json").is_file(): raise StoreCorrupt(f"{v}: meta.json missing")
        if log_meta_sha256 is not None and self._meta_hash(v) != log_meta_sha256:
            raise StoreCorrupt(f"{v}: meta.json does not match the hash pinned in the chained log")
        want = self.meta(v).get("files_sha256") or {}
        base = d / "files"
        have = {p.relative_to(base).as_posix(): sha256_text(p.read_text()) for p in sorted(base.rglob("*")) if p.is_file()} if base.exists() else {}
        if have != want: raise StoreCorrupt(f"{v}: files differ from meta.files_sha256")

    def _expected_state(self):
        """Walk the log: (expected CURRENT, previous expected CURRENT, {version: pinned meta sha or None})."""
        cur = prev = None; pins = {}
        for r in self.read_log():
            ev = r.get("event")
            if ev in ("init", "promote"):
                pins[r["version"]] = r.get("meta_sha256"); prev, cur = cur, r["version"]
            elif ev == "rollback":
                prev, cur = cur, r["to_version"]
        return cur, prev, pins

    def assert_healthy(self, allow_pending: bool = False):
        """Raise StoreCorrupt unless: the hash chain verifies, every version referenced by the log matches the meta hash the log
        pinned and its own file hashes, and CURRENT equals what the log says (or, with allow_pending, the state just before the
        newest logged switch). Legacy stores whose log records carry no meta_sha256 skip only the meta pin."""
        if not self.verify_log(): raise StoreCorrupt("hash chain broken")
        cur, prev, pins = self._expected_state()
        if cur is None and not self.versions() and not (self.root / "CURRENT").exists(): return
        for v, pin in pins.items(): self.check_version(v, pin)
        have = self.current()
        if have != cur and not (allow_pending and have == prev):
            raise StoreCorrupt(f"CURRENT={have} but the chained log says {cur}")

    def recover(self) -> bool:
        """Complete a switch that was logged but not applied. Returns True if CURRENT changed."""
        if not self.verify_log(): raise StoreCorrupt("hash chain broken")
        cur, prev, pins = self._expected_state()
        if cur is None or self.current() == cur: return False
        if self.current() != prev: raise StoreCorrupt(f"CURRENT={self.current()} is neither the logged target {cur} nor its predecessor {prev}")
        self.check_version(cur, pins.get(cur))
        self._set_current(cur)
        return True

    def init_baseline(self, files: dict[str, str], note="developer baseline") -> str:
        if self.versions(): raise RuntimeError("store already initialised")
        v = self._commit_version(files, dict(parent=None, authored_by="developer", north_attested=False, note=note))
        self.log(dict(event="init", version=v, meta_sha256=self._meta_hash(v))); self._set_current(v); return v

    def promote(self, new_files: dict[str, str], proposal_id: str, authored_by: str, north_attested: bool, diff: str,
                gates: dict, mock: bool, extra: dict | None = None) -> str:
        assert authored_by in ("north", "mock"), authored_by
        if authored_by == "north" and (mock or not north_attested):
            raise RuntimeError("refusing to label a version North-authored without attestation")
        parent = self.current()
        files = dict(self.files(parent)); files.update(new_files)
        v = self._commit_version(files, dict(parent=parent, authored_by=authored_by, north_attested=north_attested, mock=mock,
                                             proposal_id=proposal_id, diff=diff, diff_sha256=sha256_text(diff), gates=gates, **(extra or {})))
        self.log(dict(event="promote", version=v, parent=parent, proposal_id=proposal_id, authored_by=authored_by, meta_sha256=self._meta_hash(v)))
        self._set_current(v)
        return v

    def rollback(self, to: str | None = None, reason: str = "") -> str:
        cur = self.current()
        target = to or self.meta(cur)["parent"]
        if target is None or target not in self.versions():
            raise RuntimeError("nothing to roll back to")
        _, _, pins = self._expected_state()
        self.check_version(target, pins.get(target))      # never roll back onto a tampered version
        self.log(dict(event="rollback", from_version=cur, to_version=target, reason=reason))
        self._set_current(target)
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
        out["cumulative_vs_v0000"] = self.cumulative(chain)
        out["any_mock_in_lineage"] = bool(out["mock_versions"])
        out["reportable_as_north_authored"] = bool(out["north_versions"]) and not out["mock_versions"]
        return out

    def cumulative(self, chain: list[dict] | None = None, extra_tested: int = 0) -> dict:
        """Cumulative speed gain of CURRENT vs v0000 (compounded per-promotion gains, each measured vs its then-current baseline)
        with CIs; n_tested = every logged proposal that reached the speed gate (selection-effect widening)."""
        from .gates import cumulative_gain
        if chain is None:
            chain, v = [], self.current()
            while v:
                m = self.meta(v); chain.append(m); v = m["parent"]
            chain.reverse()
        n_tested = sum(1 for r in self.read_log() if r.get("event") == "proposal" and isinstance((r.get("gates") or {}).get("speed"), dict)
                       and (r["gates"]["speed"].get("data") or {}))
        n_tested += extra_tested
        out = dict(baseline="v0000" if chain else None, n_tested=n_tested,
                   note="steps compound; CI assumes independent steps; sel_adj_* widens alpha by n_tested proposals that reached the speed gate")
        for split in ("dev", "heldout"):
            steps = []
            for m in chain:
                d = (((m.get("gates") or {}).get("speed") or {}).get("data") or {}).get(split)
                if isinstance(d, dict) and all(k in d for k in ("speedup", "ci_lo", "ci_hi")): steps.append(d)
            out[split] = cumulative_gain(steps, n_tested=n_tested)
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
