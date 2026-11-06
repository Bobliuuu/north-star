"""Generates the pinned small workload files (run once; outputs are committed + sha256 in MANIFEST.json).
All prompts are hand-written here (no external data). Split is by TASK id (no task appears in two splits).
Long-context repository filler is built at run time from CPython stdlib sources (see harness/common.py)."""
import hashlib, json, pathlib

OUT = pathlib.Path(__file__).parent / "workloads"
OUT.mkdir(exist_ok=True)

CODE = [
 ("py_lru", "dev", "Complete this Python function. Output only code.\n\n```python\nfrom collections import OrderedDict\n\nclass LRUCache:\n    \"\"\"Fixed-capacity LRU cache with O(1) get/put.\"\"\"\n    def __init__(self, capacity: int):\n"),
 ("py_interval", "dev", "Complete this Python function. Output only code.\n\n```python\ndef merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:\n    \"\"\"Merge overlapping closed intervals; return sorted merged list.\"\"\"\n"),
 ("py_tokenize", "calib", "Complete this Python function. Output only code.\n\n```python\nimport re\n\ndef tokenize_arith(expr: str) -> list[str]:\n    \"\"\"Split an arithmetic expression into numbers, operators and parentheses.\"\"\"\n"),
 ("py_dataclass", "heldout", "Complete this Python module. Output only code.\n\n```python\nfrom dataclasses import dataclass, field\nimport json\n\n@dataclass\nclass Config:\n    name: str\n    retries: int = 3\n    tags: list[str] = field(default_factory=list)\n\n    @classmethod\n    def from_json(cls, s: str) -> \"Config\":\n"),
 ("js_debounce", "calib", "Complete this JavaScript function. Output only code.\n\n```js\n/** Returns a debounced version of fn that fires after `wait` ms of silence, with cancel() and flush(). */\nfunction debounce(fn, wait) {\n"),
 ("rs_stack", "dev", "Complete this Rust code. Output only code.\n\n```rust\n/// A generic fixed-capacity stack backed by an array-like Vec.\npub struct BoundedStack<T> {\n    items: Vec<T>,\n    cap: usize,\n}\n\nimpl<T> BoundedStack<T> {\n"),
 ("go_worker", "heldout", "Complete this Go function. Output only code.\n\n```go\n// RunPool runs jobs on n workers and returns results in input order.\nfunc RunPool[T, R any](n int, jobs []T, f func(T) R) []R {\n"),
 ("c_strtrim", "calib", "Complete this C function. Output only code.\n\n```c\n/* Trim leading and trailing ASCII whitespace of s in place; return s. */\nchar *str_trim(char *s) {\n"),
 ("sql_topn", "heldout", "Write the SQL query. Output only SQL.\n\nTables: orders(id, customer_id, total, created_at), customers(id, name, country).\nTask: for each country, the top 3 customers by total spend in 2025, with rank."),
 ("sh_logs", "dev", "Write a bash script. Output only code.\n\nTask: given a directory of *.log files, print the 10 most frequent ERROR message prefixes (text before the first ':' after 'ERROR') with counts, ignoring case."),
]
TERM = [
 ("term_pytest_fail", "dev", [
  ("user", "You are in /repo. Run the tests and fix failures."),
  ("assistant", "I'll run the test suite first.\n```bash\npytest -x -q\n```"),
  ("user", "$ pytest -x -q\nF\n=================================== FAILURES ===================================\n_______________________________ test_parse_date ________________________________\n    def test_parse_date():\n>       assert parse_date('2025-13-01') is None\nE       ValueError: month must be in 1..12\nsrc/dates.py:14: ValueError\n1 failed in 0.21s"),
 ]),
 ("term_git_conflict", "dev", [
  ("user", "Resolve the merge conflict in README.md and continue the rebase."),
  ("assistant", "```bash\ngit status\n```"),
  ("user", "$ git status\ninteractive rebase in progress; onto 4f2a1c9\nUnmerged paths:\n  both modified:   README.md"),
 ]),
 ("term_disk", "calib", [
  ("user", "The build box says 'No space left on device'. Find what is using space under /var and clean safely."),
  ("assistant", "```bash\ndf -h /var && du -xh /var --max-depth=2 | sort -rh | head -15\n```"),
  ("user", "$ df -h /var\nFilesystem  Size  Used Avail Use% Mounted on\n/dev/nvme0n1p2 50G  50G  0 100% /\n$ du ...\n31G /var/lib/docker\n9.2G /var/log\n4.1G /var/cache/apt"),
 ]),
 ("term_port", "calib", [
  ("user", "Something is already listening on port 8080 and my server won't start. Diagnose."),
  ("assistant", "```bash\nss -ltnp | grep 8080\n```"),
  ("user", "$ ss -ltnp | grep 8080\nLISTEN 0 511 0.0.0.0:8080 0.0.0.0:* users:((\"node\",pid=4412,fd=19))"),
 ]),
 ("term_make", "heldout", [
  ("user", "make fails; figure out why and fix it."),
  ("assistant", "```bash\nmake 2>&1 | tail -20\n```"),
  ("user", "$ make 2>&1 | tail -20\ngcc -O2 -c src/net.c -o build/net.o\nsrc/net.c:88:5: error: implicit declaration of function 'htonl' [-Wimplicit-function-declaration]\nmake: *** [Makefile:12: build/net.o] Error 1"),
 ]),
 ("term_docker", "heldout", [
  ("user", "The container exits immediately. Debug it."),
  ("assistant", "```bash\ndocker logs api --tail 30\n```"),
  ("user", "$ docker logs api --tail 30\nError: Cannot find module '/app/dist/server.js'\n    at Module._resolveFilename (node:internal/modules/cjs/loader:1140:15)\n$ docker run --rm api ls /app\npackage.json  src  node_modules"),
 ]),
]
PROSE = [
 ("pr_hbm", "dev", "Explain, for a systems engineer, why LLM decoding at batch size 1 is memory-bandwidth bound, and when speculative decoding helps and when it does not."),
 ("pr_moe", "calib", "Write a clear explanation of mixture-of-experts routing: top-k selection, sigmoid versus softmax gating, load balancing, and what changes at inference time."),
 ("pr_story", "heldout", "Write a 300-word short story about a lighthouse keeper who discovers the light has been signalling to someone else."),
 ("pr_tradeoff", "dev", "Compare optimistic and pessimistic concurrency control in databases. Give a concrete example where each wins."),
 ("pr_summary", "calib", "Summarise the key ideas behind attention with a sliding window plus periodic global layers, and the consequences for KV-cache size at long context."),
 ("pr_reason", "heldout", "A train leaves at 9:40 travelling 72 km/h; a second leaves the same station at 10:10 travelling 90 km/h on the same track. When and where does the second catch up? Reason step by step."),
]

def w(name, rows):
    p = OUT / f"{name}.jsonl"
    p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    return p

code = [dict(id=i, kind="code", split=s, messages=[dict(role="user", content=t)]) for i, s, t in CODE]
term = [dict(id=i, kind="terminal", split=s, messages=[dict(role=r, content=c) for r, c in m]) for i, s, m in TERM]
prose = [dict(id=i, kind="prose", split=s, messages=[dict(role="user", content=t)]) for i, s, t in PROSE]
files = [w("code", code), w("terminal", term), w("prose", prose)]
man = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
man["split_counts"] = {s: sum(r["split"] == s for r in code + term + prose) for s in ("dev", "calib", "heldout")}
man["note"] = "terminal traces are hand-written synthetic multi-turn snippets, not real agent logs."
(OUT / "MANIFEST.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
print(man)
