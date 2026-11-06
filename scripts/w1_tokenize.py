"""Tokenize the W1 OpenHands panel (built by ~/inference-experiment/scripts/tiramisu/build_panel.py from
nebius/SWE-rebench-openhands-trajectories@35455389) with North's chat template (reasoning_effort=none) exactly as
prepare_agent_panel.measure_ids does. Writes an int32 npz of prompt ids + a small manifest (committed)."""
import copy, json, sys, hashlib
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

panel, tokdir, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
tok = AutoTokenizer.from_pretrained(tokdir)
meta = json.loads((panel / "panel.json").read_text())
sess = sorted((panel / "sessions").glob("*.jsonl"))
# panel order = prepared requests order; recover from panel.json if it lists sessions, else file order
order = meta.get("session_order") or meta.get("sessions_list")
files = [panel / "sessions" / f"{s}.jsonl" for s in order] if isinstance(order, list) else sess
ids, man = {}, []
for f in files:
    for t, line in enumerate(f.read_text().splitlines()):
        body = json.loads(line)
        msgs = copy.deepcopy(body["messages"])
        for m in msgs:
            if m["role"] == "assistant":
                for c in m.get("tool_calls") or []:
                    if isinstance(c["function"]["arguments"], str):
                        c["function"]["arguments"] = json.loads(c["function"]["arguments"])
        v = tok.apply_chat_template(msgs, tools=body.get("tools"), tokenize=True, add_generation_prompt=True,
                                    reasoning_effort="none")
        v = v["input_ids"] if hasattr(v, "keys") else v
        rid = f"{f.stem}-turn{t}"
        ids[rid] = np.array(v, dtype=np.int32)
        man.append(dict(id=rid, session=f.stem, turn=t, prompt_tokens=len(v),
                        sha=hashlib.sha256(ids[rid].tobytes()).hexdigest()[:16]))
np.savez_compressed(out, **ids)
Path(str(out).replace(".npz", "_manifest.json")).write_text(json.dumps(man, indent=0))
L = [m["prompt_tokens"] for m in man]
print(len(man), "requests; mean", sum(L) / len(L), "max", max(L), "min", min(L))
