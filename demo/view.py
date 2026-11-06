#!/usr/bin/env python3
"""Terminal viewer for real North generation.
  python demo/view.py --replay demo/traces/<file>.jsonl [--delay 0.05]
  modal run --detach modal/demo.py --task <id> ...   (live; see demo/README.md; do not use plain `modal run`)"""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import events as E, render as R


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--allow-fixture", action="store_true")
    a = ap.parse_args(argv)
    evs = E.read_events(a.replay)
    try:
        E.check_provenance(evs, a.allow_fixture)
    except E.ProvenanceError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    R.replay(evs, delay=a.delay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
