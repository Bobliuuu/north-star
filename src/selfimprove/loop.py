"""Orchestrator: propose -> validate/apply -> audit -> sandboxed build -> correctness -> quality -> speed -> promote/log."""
from __future__ import annotations
import secrets, time, uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

from . import audit as A, diffs as D, gates as G, sandbox as S
from .policy import Scope, DEFAULT_SCOPE
from . import structured as ST
from .proposer import Proposer, ProposalRequest, extract_diff
from .store import Store, sha256_text


@dataclass
class EvalConfig:
    repo: Path
    subdirs: list[str]                         # repo subtrees copied into every sandbox workspace
    correctness_cmd: list[str]                 # prints JSON {"cases":[...]}; reads NS_SEED
    quality_cmd: list[str]                     # prints JSON {"items":[...]}; fixed held-out data, protected
    bench_cmd: list[str]                       # prints JSON {"time_ms": x}; reads NS_SPLIT in {dev,heldout}
    build_cmd: list[str] | None = None         # must write only under build/
    scope: Scope = field(default_factory=lambda: DEFAULT_SCOPE)
    correctness: G.CorrectnessSpec = field(default_factory=G.CorrectnessSpec)
    quality: G.QualitySpec = field(default_factory=G.QualitySpec)
    speed: G.SpeedSpec = field(default_factory=G.SpeedSpec)
    build_timeout_s: float = 300; check_timeout_s: float = 300; bench_timeout_s: float = 120
    dev_shapes: set[int] = field(default_factory=set)        # integers appearing in dev benchmark shapes
    heldout_shapes: set[int] = field(default_factory=set)
    usd_per_hour: float = 2.5                  # evaluator hardware, for cost accounting only
    tmp_parent: Path | None = None
    keep_workspaces: bool = False


def scope_description(scope: Scope) -> str:
    L = [f"- {r.pattern}" + (" (only lines between NS-EDITABLE-BEGIN/END markers)" if r.regions_only else " (whole file)") for r in scope.rules]
    l = scope.limits
    return ("Editable files:\n" + "\n".join(L) + f"\nLimits: diff <= {l.max_diff_bytes} bytes, <= {l.max_files} files, "
            f"<= {l.max_hunks} hunks, <= {l.max_changed_lines} changed lines. Everything else is read-only.")


def editable_files_in_repo(repo: Path, scope: Scope) -> dict[str, str]:
    import fnmatch
    out = {}
    for r in scope.rules:
        for p in sorted(Path(repo).glob(r.pattern)):
            if p.is_file(): out[p.relative_to(repo).as_posix()] = p.read_text()
    return out


class Evaluator:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg

    def _run(self, ws: S.Workspace, cmd, timeout, env=None):
        nonce = secrets.token_hex(16)
        r = S.run(cmd, ws.root, timeout, {**(env or {}), "NS_NONCE": nonce})
        ws.verify()                       # raises SandboxViolation
        r.parsed = r.json(nonce) if (r.rc == 0 and not r.timed_out) else None   # crashed/hung harness: any line it printed is untrusted
        return r

    def evaluate(self, base_files: dict[str, str], new_files: dict[str, str]) -> dict:
        """Gates in cheap-to-expensive order, short-circuiting. Returns {'verdict', 'gates', 'sandbox', 'eval_wall_s', ...}."""
        c, t0 = self.cfg, time.time()
        out = dict(verdict=None, gates={}, sandbox={}, seeds={})
        # fixed external repo manifest, excluding editable paths: proves we never touched protected files in the real repo
        edit_paths = set(new_files) | set(base_files)
        repo_before = {k: v for k, v in self._repo_manifest().items() if k not in edit_paths}
        wb = wc = None
        try:
            wb = S.Workspace.create(c.repo, c.subdirs, base_files, edit_paths, c.tmp_parent)
            wc = S.Workspace.create(c.repo, c.subdirs, {**base_files, **new_files}, edit_paths, c.tmp_parent)
            # protected trees of baseline and candidate must be identical
            if wb.protected_hashes != wc.protected_hashes:
                out["verdict"] = "sandbox_violation"; out["sandbox"] = {"error": "protected trees differ"}; return out
            out["sandbox"]["protected_digest"] = S.manifest_digest(wc.protected_hashes)
            if c.build_cmd:
                for tag, ws in (("base", wb), ("cand", wc)):
                    r = self._run(ws, c.build_cmd, c.build_timeout_s)
                    out["sandbox"][f"build_{tag}"] = dict(rc=r.rc, wall_s=r.wall_s, timed_out=r.timed_out, stderr=r.stderr[-1500:])
                    if r.rc != 0:
                        out["verdict"] = ("build_timeout" if r.timed_out else "build_fail") if tag == "cand" else "baseline_broken"
                        return out
            wb.freeze_build(); wc.freeze_build()      # build/ is read-only evidence from here on (binary swap / planted files = violation)
            # (1) correctness: fresh unpredictable seed, same for base and cand within this evaluation
            seed = secrets.randbits(48); out["seeds"]["correctness"] = seed
            rc_ = self._run(wc, c.correctness_cmd, c.check_timeout_s, {"NS_SEED": str(seed)})
            rb_ = self._run(wb, c.correctness_cmd, c.check_timeout_s, {"NS_SEED": str(seed)})
            if rc_.timed_out:
                out["verdict"] = "timeout"; out["gates"]["correctness"] = G.GateResult("correctness", False, "timeout").to_dict(); return out
            if rb_.timed_out or not rb_.parsed:       # without a measured baseline the regress-vs-baseline check would silently vanish
                out["verdict"] = "baseline_broken"; out["sandbox"]["error"] = "baseline correctness run failed/unparsable"; return out
            g = G.correctness_gate(rc_.parsed, rb_.parsed, c.correctness); out["gates"]["correctness"] = g.to_dict()
            if not g.passed: out["verdict"] = "gate_fail:correctness"; return out
            # (2) held-out quality (fixed seed, paired)
            qc = self._run(wc, c.quality_cmd, c.check_timeout_s, {"NS_SEED": "0"})
            qb = self._run(wb, c.quality_cmd, c.check_timeout_s, {"NS_SEED": "0"})
            g = G.quality_gate(qc.parsed, qb.parsed, c.quality); out["gates"]["quality"] = g.to_dict()
            if not g.passed: out["verdict"] = "gate_fail:quality"; return out
            # (3) speed: interleaved A/B per split, order alternated to cancel drift
            splits = {}
            for split in ("dev", "heldout"):
                b, k = [], []
                for i in range(c.speed.n_runs):
                    order = (("b", wb), ("k", wc)) if i % 2 == 0 else (("k", wc), ("b", wb))
                    for tag, ws in order:
                        r = self._run(ws, c.bench_cmd, c.bench_timeout_s, {"NS_SPLIT": split, "NS_SEED": str(secrets.randbits(32))})
                        j = r.parsed
                        if r.timed_out or not j or not isinstance(j.get("time_ms"), (int, float)):
                            out["gates"]["speed"] = G.GateResult("speed", False, f"{split}: bench failed/timeout/unparsable").to_dict()
                            out["verdict"] = "gate_fail:speed"; return out
                        (b if tag == "b" else k).append(float(j["time_ms"]))
                splits[split] = (b, k)
            g = G.speed_gate(splits, c.speed); out["gates"]["speed"] = g.to_dict()
            out["verdict"] = "pass" if g.passed else "gate_fail:speed"
            return out
        except S.SandboxViolation as e:
            out["verdict"] = "sandbox_violation"; out["sandbox"]["error"] = str(e); return out
        except Exception as e:                    # a gate/harness bug must be a logged non-promotion, never a crash that loses the record
            out["verdict"] = "evaluator_error"; out["sandbox"]["error"] = repr(e)[:500]; return out
        finally:
            for w in (wb, wc):
                if w is not None and not c.keep_workspaces: w.cleanup()
            after = {k: v for k, v in self._repo_manifest().items() if k not in edit_paths}
            if after != repo_before:
                out["verdict"] = "sandbox_violation"; out["sandbox"]["error"] = "REAL repo protected files changed during evaluation"
            out["eval_wall_s"] = time.time() - t0

    def _repo_manifest(self):
        m = {}
        for s in self.cfg.subdirs:
            p = self.cfg.repo / s
            if p.is_file(): m[s] = S.sha256_file(p)
            elif p.exists():
                sub = S.manifest(p); m.update({f"{s}/{k}": v for k, v in sub.items()})
        return m


class Loop:
    def __init__(self, cfg: EvalConfig, store: Store, proposer: Proposer, allow_mock_promotion: bool = False, mode: str = "diff",
                 prior_history: list[str] | None = None):
        if mode not in ("diff", "structured"): raise ValueError(mode)
        self.mode = mode
        self.prior_history = list(prior_history or [])      # dev-only one-liners from earlier stores (see ST.history_from_store)
        self.cfg, self.store, self.proposer, self.ev = cfg, store, proposer, Evaluator(cfg)
        self.allow_mock_promotion = allow_mock_promotion
        if store.current() is None:
            store.init_baseline(editable_files_in_repo(cfg.repo, cfg.scope))
        else:
            store.recover()                           # finish a promotion/rollback that was logged but not applied (crash)
            store.assert_healthy()

    def step(self, profile_summary: str = "", dev_results: str = "", rnd: int = 0) -> dict:
        cfg, st = self.cfg, self.store
        st.assert_healthy()                           # chained log, version contents and CURRENT must agree before we trust the baseline
        base_v = st.current(); base_files = st.files(base_v)
        history = list(self.prior_history) + [ST.history_line(r) for r in st.read_log() if r.get("event") == "proposal"]
        req = ProposalRequest(base_version=base_v, files=base_files, scope_description=scope_description(cfg.scope),
                              profile_summary=profile_summary, dev_results=dev_results, history=history, mode=self.mode)
        pid = uuid.uuid4().hex
        rec = dict(event="proposal", proposal_id=pid, round=rnd, base_version=base_v, verdict=None, findings=[], gates={}, cost={})
        t0 = time.time()
        try:
            resp = self.proposer.propose(req)
        except Exception as e:                       # provenance failures, endpoint errors: log, never promote
            rec.update(verdict="proposer_error", error=repr(e)[:500], proposer=dict(kind=self.proposer.kind))
            return st.log(rec)
        mock = resp.kind != "north"
        attested = resp.kind == "north" and resp.model_id == "CohereLabs/North-Mini-Code-1.0" and bool(resp.revision)
        structured_err = None
        if self.mode == "structured":
            diff = None
            try:
                se = ST.parse_edits(resp.raw_response)
                rec["structured"] = dict(edits=[dict(name=n, value=v) for n, v in se["edits"]], rationale=se["rationale"])
                diff = ST.render_diff(se["edits"], base_files)       # '' for a no-op
            except ST.StructuredEditError as e:
                structured_err = str(e)
        else:
            diff = extract_diff(resp.raw_response)
        label = "MOCK - NOT NORTH-AUTHORED" if mock else ("North-authored (structured edit)" if self.mode == "structured" else "North-authored (attested)")
        rec.update(proposer=dict(kind=resp.kind, model_id=resp.model_id, revision=resp.revision, backend=resp.backend,
                                 endpoint=resp.endpoint, response_id=resp.response_id, mock=mock, north_attested=attested,
                                 label=label, mode=self.mode,
                                 attribution_note=("North chose the values; harness only rendered the diff" if self.mode == "structured" and not mock else None)),
                   prompt=resp.prompt, prompt_sha256=sha256_text(resp.prompt),
                   raw_response=resp.raw_response, raw_response_sha256=sha256_text(resp.raw_response),
                   diff=diff, diff_sha256=sha256_text(diff) if diff else None, base_files_sha256={k: sha256_text(v) for k, v in base_files.items()})
        cost = dict(proposer_usd=resp.cost_usd, proposer_gpu_s=resp.gpu_seconds, tokens_in=resp.tokens_in, tokens_out=resp.tokens_out,
                    proposer_latency_s=resp.latency_s, eval_wall_s=0.0, eval_usd=0.0)
        rec["cost"] = cost

        def finish(verdict, promoted=None):
            rec["verdict"] = verdict; rec["promoted_version"] = promoted
            cost["total_wall_s"] = time.time() - t0
            return st.log(rec)

        if structured_err is not None:
            rec["reason"] = structured_err; return finish("invalid_edit")
        if not diff:
            return finish("no_change" if diff == "" else "no_diff_in_response")
        try:
            fds = D.parse(diff, cfg.scope)
            new_files = D.apply_all(fds, base_files.get, cfg.scope)
        except D.DiffError as e:
            rec["reason"] = str(e); return finish("invalid_diff")
        findings = A.audit_diff(fds, cfg.dev_shapes - cfg.heldout_shapes, cfg.dev_shapes)
        rec["findings"] = [f.to_dict() for f in findings]
        av = A.verdict(findings)
        if av == "reject":
            return finish("audit_reject")
        ev = self.ev.evaluate(base_files, new_files)
        rec.update(gates=ev.get("gates") or {}, sandbox=ev.get("sandbox") or {}, seeds=ev.get("seeds") or {})
        cost["eval_wall_s"] = ev.get("eval_wall_s", 0.0); cost["eval_usd"] = cost["eval_wall_s"] / 3600 * cfg.usd_per_hour
        verdict = ev.get("verdict")
        if verdict == "pass":
            # defense in depth (the evaluator may be remote): a bare "pass" is not enough, every gate must be recorded AND passed
            gs = ev.get("gates") or {}
            if not all(isinstance(gs.get(k), dict) and gs[k].get("passed") is True for k in ("correctness", "quality", "speed")):
                rec["reason"] = "evaluator said pass but not all of correctness/quality/speed are recorded as passed"
                return finish("gate_missing")
        if verdict != "pass":
            return finish(verdict or "evaluator_error")
        if av == "flag":
            return finish("needs_review")          # passed all gates but static audit flagged: human decides, never auto-promoted
        if mock and not self.allow_mock_promotion:
            return finish("pass_mock_not_promoted")
        v = st.promote(new_files, pid, "mock" if mock else "north", attested, diff, ev["gates"], mock,
                       extra=dict(proposal_mode=self.mode, label=label))
        return finish("promoted", v)

    def run(self, rounds: int, max_total_usd: float = 5.0, **kw) -> list[dict]:
        out = []
        for i in range(rounds):
            spent = self.store.cost_summary()
            if spent["proposer_usd"] + spent["eval_usd"] >= max_total_usd:
                self.store.log(dict(event="budget_stop", spent=spent)); break
            out.append(self.step(rnd=i, **kw))
        return out
