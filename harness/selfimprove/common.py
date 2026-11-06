"""Shared harness plumbing for the self-improvement evaluator. OUTSIDE the editable scope (hash-protected in every sandbox).

Nonce protocol: the per-run nonce is read from the environment and REMOVED before anything else happens, so processes we
spawn (the candidate-built CUDA binary) never see it; the last stdout line is the only thing the evaluator trusts and it
must carry the nonce. Candidate-controlled code (the compiled binary) is only ever run as a child process whose stdout is
captured and discarded; it is never imported into this process.
"""
import json, os, sys

_NONCE = os.environ.pop("NS_NONCE", None)          # must stay the first side effect of this module
if _NONCE is None:
    sys.stderr.write("NS_NONCE missing: harness must be run by the evaluator\n"); sys.exit(97)

WS = os.path.abspath(os.environ.get("NS_WORKSPACE", os.getcwd()))
SEED = int(os.environ.get("NS_SEED", "0"))
SPLIT = os.environ.get("NS_SPLIT", "dev")
SCRATCH = os.path.join(WS, "_scratch")
BINARY = os.path.join(WS, "build", "moe_fused")
CHILD_ENV = {k: v for k, v in os.environ.items() if not k.startswith("NS_")}   # children get no NS_* (nonce, seed, split)


def emit(obj: dict):
    obj = dict(obj); obj["nonce"] = _NONCE
    sys.stdout.write(json.dumps(obj, allow_nan=False) + "\n"); sys.stdout.flush()


def limit_threads(n: int = 2):
    """The sandbox caps process CPU seconds; many torch threads burn that budget. Keep CPU reference work small."""
    import torch
    torch.set_num_threads(n)
