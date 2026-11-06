"""Laptop-death-proof round loop: pure-python pieces (no modal import) used by modal/selfimprove_eval.py::run_loop_remote.

The orchestrator (a CPU Modal function) runs the unchanged Loop against a Store that lives on a Modal Volume. `EmittingStore`
only adds side effects AFTER the base Store has written+fsynced a record: it prints the record as one JSON line (app logs) and
calls `on_log` (Volume commit). Hash chain, gates, audit and promotion semantics are untouched. `pull_tree` copies the Volume
store down to results/<store>; the chain is re-verified locally.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from .store import Store

LOG_PREFIX = "NS_PROPOSAL_JSON "


class EmittingStore(Store):
    def __init__(self, root, emit=None, on_log=None):
        super().__init__(root)
        self.emit = emit or (lambda line: print(line, flush=True))
        self.on_log = on_log

    def log(self, record: dict) -> dict:
        rec = super().log(record)                  # durable + chained first
        try: self.emit(LOG_PREFIX + json.dumps(rec, default=str))
        except Exception: pass                     # logging must never break the loop
        if self.on_log: self.on_log()
        return rec

    def _commit_version(self, files, meta):
        v = super()._commit_version(files, meta)
        if self.on_log: self.on_log()
        return v

    def _set_current(self, v):
        super()._set_current(v)
        if self.on_log: self.on_log()


def store_guard(store_dir: Path, proposer: str):
    if not Path(store_dir).exists(): return
    att = Store(store_dir).attribution()
    if proposer == "north" and att.get("mock_versions"):
        raise SystemExit(f"refusing: store {store_dir} has MOCK versions in its lineage; use a fresh --store for North runs")
    if proposer == "mock" and att.get("north_versions"):
        raise SystemExit(f"refusing: store {store_dir} holds North-authored versions; mock dry runs need their own store")


def run_remote_loop(p: dict, store_dir: Path, cfg, proposer, evaluator, profile_summary="", dev_results="",
                    emit=None, on_log=None) -> dict:
    """p: {rounds, max_usd, mode, proposer ('mock'|'north')}. cfg/proposer/evaluator are injected (real ones in the Modal function)."""
    from .loop import Loop
    store_guard(store_dir, p["proposer"])
    st = EmittingStore(store_dir, emit, on_log)
    lp = Loop(cfg, st, proposer, mode=p["mode"], track=p.get("track", "kernel"))
    lp.ev = evaluator
    if on_log: on_log()
    res = lp.run(p["rounds"], max_total_usd=p["max_usd"], proposals_per_round=int(p.get("per_round", 1)), profile_summary=profile_summary, dev_results=dev_results)
    out = dict(verdicts=[(r["proposal_id"], r["verdict"]) for r in res], attribution=st.attribution(), cost=st.cost_summary(),
               log_ok=st.verify_log())
    (emit or print)("NS_LOOP_DONE " + json.dumps(out, default=str))
    if on_log: on_log()
    return out


def pull_tree(listdir, read_file, remote_dir: str, dest: Path) -> list[str]:
    """Generic recursive pull. listdir(path)->[(path, is_dir)]; read_file(path)->bytes. Returns written relative paths.
    Safety: (1) refuses to overwrite a dest whose proposals.jsonl is not a prefix of the incoming one (never clobber local
    history); (2) every incoming path must stay under dest (no `..`/absolute); (3) the incoming store is STAGED in a temp dir and
    must pass the chain + version/CURRENT consistency check BEFORE anything in dest is touched; (4) files are installed versions
    first, log next, CURRENT last, so an interrupted install never leaves CURRENT pointing at a missing version."""
    import shutil, tempfile
    dest = Path(dest); remote_dir = remote_dir.strip("/")
    files, stack = [], [remote_dir]
    while stack:
        d = stack.pop()
        for path, is_dir in listdir(d):
            path = path.strip("/")
            (stack if is_dir else files).append(path)
    if not files: raise SystemExit(f"nothing found under {remote_dir} in the Volume")
    blobs = {}
    for f in files:
        rel = f[len(remote_dir):].lstrip("/")
        parts = rel.split("/")
        if not rel or rel.startswith("/") or ".." in parts or "" in parts or "\\" in rel:
            raise SystemExit(f"refusing: unsafe path in Volume listing: {f!r}")
        blobs[rel] = read_file(f)
    old = dest / "proposals.jsonl"
    if old.exists() and "proposals.jsonl" in blobs and not blobs["proposals.jsonl"].startswith(old.read_bytes()):
        raise SystemExit(f"refusing: local {old} is not a prefix of the remote log (diverged); pull into a fresh --dest")
    stage = Path(tempfile.mkdtemp(prefix="ns_pull_"))
    try:
        for rel, b in blobs.items():
            t = stage / rel; t.parent.mkdir(parents=True, exist_ok=True); t.write_bytes(b)
        try:
            Store(stage).assert_healthy(allow_pending=True)
        except Exception as e:
            print(f"BAD INCOMING STORE: {e}", file=sys.stderr)
            raise SystemExit(1)
        rank = lambda r: (2 if r == "CURRENT" else 1 if r == "proposals.jsonl" else 0, r)
        for rel in sorted(blobs, key=rank):
            t = dest / rel; t.parent.mkdir(parents=True, exist_ok=True); t.write_bytes(blobs[rel])
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    ok = Store(dest).verify_log()
    print("log chain OK" if ok else "LOG CHAIN BROKEN", file=sys.stderr)
    if not ok: raise SystemExit(1)
    return sorted(blobs)
