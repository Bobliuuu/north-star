"""Isolated workspace + subprocess runner + tree hash verification.

The candidate runs only inside a throw-away COPY of the repo subset. Everything not in the editable scope is protected:
we record a sha256 manifest of the whole tree (minus ignorable build dirs) before and after every command, and any change
outside the evaluator's own writes is a sandbox violation. Protected files are also chmod a-w (defense in depth, not
trusted: same-uid code can chmod back, which the hash check would catch)."""
from __future__ import annotations
import hashlib, os, shutil, signal, subprocess, tempfile, time, json
from dataclasses import dataclass
from pathlib import Path

IGNORE_DIRS = {"__pycache__", ".git", "build", "_scratch", ".pytest_cache"}
SCRUB_ENV_KEEP = {"PATH", "HOME", "LANG", "LC_ALL", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "PYTHONPATH", "TMPDIR"}


class SandboxViolation(RuntimeError):
    pass


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def manifest(root: Path) -> dict[str, str]:
    m = {}
    for dp, dns, fns in os.walk(root, followlinks=False):
        dns[:] = sorted(d for d in dns if d not in IGNORE_DIRS)
        for fn in sorted(fns):
            p = Path(dp) / fn
            rel = p.relative_to(root).as_posix()
            if p.is_symlink():
                m[rel] = "symlink:" + os.readlink(p)
            elif fn.endswith(".pyc"):
                continue
            else:
                m[rel] = sha256_file(p)
    return m


def manifest_digest(m: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()


def copy_tree(repo: Path, subdirs: list[str], dest: Path):
    for s in subdirs:
        src = repo / s
        if not src.exists():
            continue
        if src.is_file():
            (dest / s).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dest / s); continue
        shutil.copytree(src, dest / s, symlinks=False, ignore=shutil.ignore_patterns(*IGNORE_DIRS, "*.pyc"), dirs_exist_ok=True)


@dataclass
class Workspace:
    root: Path
    editable: set[str]                 # rel paths the evaluator is allowed to have written
    protected_hashes: dict[str, str]   # frozen at creation (after diff application)
    editable_hashes: dict = None

    @classmethod
    def create(cls, repo: Path, subdirs: list[str], overrides: dict[str, str], editable_paths: set[str], tmp_parent: Path | None = None):
        root = Path(tempfile.mkdtemp(prefix="ns_ws_", dir=tmp_parent))
        copy_tree(repo, subdirs, root)
        for rel, text in overrides.items():       # evaluator applies the diff result; the candidate never writes
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        (root / "_scratch").mkdir(exist_ok=True)
        ws = cls(root, set(editable_paths), {})
        ws.protected_hashes = {k: v for k, v in manifest(root).items() if k not in ws.editable}
        ws.editable_hashes = {k: v for k, v in manifest(root).items() if k in ws.editable}
        ws.lock()
        return ws

    def lock(self):
        for rel in self.protected_hashes:
            try:
                os.chmod(self.root / rel, 0o444)
            except OSError:
                pass

    def verify(self):
        """Raise SandboxViolation if any protected file changed/vanished, any unexpected file appeared, or editable
        files differ from what the evaluator wrote (a build must not rewrite its own source)."""
        cur = manifest(self.root)
        bad = []
        for k, v in self.protected_hashes.items():
            if cur.get(k) != v:
                bad.append(("protected_changed", k))
        for k, v in self.editable_hashes.items():
            if cur.get(k) != v:
                bad.append(("editable_rewritten", k))
        known = set(self.protected_hashes) | set(self.editable_hashes)
        for k in cur:
            if k not in known:
                bad.append(("unexpected_file", k))
        if bad:
            raise SandboxViolation(json.dumps(bad[:20]))

    def cleanup(self):
        for dp, dns, fns in os.walk(self.root):
            for n in dns + fns:
                try: os.chmod(os.path.join(dp, n), 0o755 if n in dns else 0o644)
                except OSError: pass
        shutil.rmtree(self.root, ignore_errors=True)


@dataclass
class RunResult:
    cmd: list[str]; rc: int | None; stdout: str; stderr: str; wall_s: float; timed_out: bool

    def json(self):
        """Last JSON object line on stdout (harness protocol)."""
        for ln in reversed(self.stdout.strip().splitlines()):
            ln = ln.strip()
            if ln.startswith("{") and ln.endswith("}"):
                try: return json.loads(ln)
                except json.JSONDecodeError: pass
        return None


def _limits(cpu_s: int, fsize_mb: int, nproc: int | None):
    def f():
        import resource
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_mb << 20, fsize_mb << 20))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if nproc:
            try: resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
            except (ValueError, OSError): pass
    return f


def run(cmd: list[str], cwd: Path, timeout_s: float, env_extra: dict[str, str] | None = None, max_out: int = 1 << 20,
        fsize_mb: int = 256, nproc: int | None = None) -> RunResult:
    env = {k: v for k, v in os.environ.items() if k in SCRUB_ENV_KEEP}
    env["NS_WORKSPACE"] = str(cwd)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(env_extra or {})
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         preexec_fn=_limits(int(timeout_s) + 5, fsize_mb, nproc))
    timed_out = False
    try:
        out, err = p.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        try: os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        out, err = p.communicate()
    else:
        try: os.killpg(p.pid, signal.SIGKILL)     # reap stragglers
        except (ProcessLookupError, PermissionError): pass
    return RunResult(cmd, None if timed_out else p.returncode, out[-max_out:].decode("utf8", "replace"),
                     err[-max_out:].decode("utf8", "replace"), time.time() - t0, timed_out)
