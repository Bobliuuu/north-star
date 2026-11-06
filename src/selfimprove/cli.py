"""python -m selfimprove.cli {run,rollback,attribution,verify-log} --store DIR  (run from repo root with PYTHONPATH=src)"""
import argparse, json, sys
from pathlib import Path
from .store import Store


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "rollback", "attribution", "verify-log", "cost"])
    ap.add_argument("--store", default="results/selfimprove")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-usd", type=float, default=5.0)
    ap.add_argument("--config", help="python file defining make_config(repo: Path)->EvalConfig")
    ap.add_argument("--mock", action="store_true", help="TEST ONLY: mock proposer; output labeled and never North-authored")
    ap.add_argument("--mode", choices=["structured", "diff"], default="structured", help="proposal mode (structured = JSON tunable edits rendered to a diff by the harness)")
    ap.add_argument("--to")
    a = ap.parse_args(argv)
    st = Store(Path(a.store))
    if a.cmd == "attribution": print(json.dumps(st.attribution(), indent=1)); return 0
    if a.cmd == "verify-log": ok = st.verify_log(); print("log chain OK" if ok else "LOG TAMPERED"); return 0 if ok else 1
    if a.cmd == "cost": print(json.dumps(st.cost_summary(), indent=1)); return 0
    if a.cmd == "rollback": print(st.rollback(a.to, "cli")); return 0
    import runpy
    from .loop import Loop
    from .proposer import NorthProposer, MockProposer
    cfg = runpy.run_path(a.config)["make_config"](Path.cwd())
    prop = MockProposer(["```diff\n```" if a.mode == "diff" else '{"edits":[],"rationale":"mock"}']) if a.mock else NorthProposer()
    res = Loop(cfg, st, prop, mode=a.mode).run(a.rounds, a.max_usd)
    for r in res: print(r["proposal_id"][:8], r["verdict"])
    return 0

if __name__ == "__main__":
    sys.exit(main())
