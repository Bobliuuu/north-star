"""Event schema, provenance gate and JSONL IO for the North demo. Pure stdlib (CPU-testable). See demo/README.md."""
import json

SCHEMA = "northstar-demo/1"
REAL_SOURCE = "modal-live"          # produced by modal/demo.py on real North weights
FIXTURE_SOURCE = "FIXTURE"          # hand-made, tests only
REQUIRED_PROV = ("source", "model_id", "model_rev", "gpu", "created_utc")
STEP_FIELDS = ("i", "tokens", "text", "mode", "gamma", "accepted", "drafted", "committed_total",
               "gen_s", "inst_tok_s", "cum_tok_s", "accept_cum", "gpu_mem_gb", "gpu_peak_gb",
               "run_cost_usd", "cum_cost_usd")


class ProvenanceError(ValueError):
    pass


def read_events(path):
    out = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{n}: bad JSON ({e})")
    return out


def write_event(f, ev):
    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    f.flush()


def check_provenance(events, allow_fixture=False):
    """Refuse traces without a valid provenance record on the first `run` event. Returns the provenance dict."""
    if not events or events[0].get("type") != "run":
        raise ProvenanceError("first event must be type=run carrying a provenance object")
    prov = events[0].get("provenance")
    if not isinstance(prov, dict):
        raise ProvenanceError("refusing trace: no provenance field (possibly synthetic)")
    missing = [k for k in REQUIRED_PROV if not prov.get(k)]
    if missing:
        raise ProvenanceError(f"refusing trace: provenance missing {missing}")
    if events[0].get("schema") != SCHEMA:
        raise ProvenanceError(f"unsupported schema {events[0].get('schema')!r}, want {SCHEMA}")
    if prov["source"] == FIXTURE_SOURCE:
        if not allow_fixture:
            raise ProvenanceError("refusing FIXTURE trace (hand-made test data); pass --allow-fixture to render it, it will be bannered")
    elif prov["source"] != REAL_SOURCE:
        raise ProvenanceError(f"refusing trace: unknown provenance source {prov['source']!r}")
    for ev in events:
        if ev.get("type") == "step":
            miss = [k for k in STEP_FIELDS if k not in ev]
            if miss:
                raise ProvenanceError(f"step {ev.get('i')} missing fields {miss}")
    return prov
