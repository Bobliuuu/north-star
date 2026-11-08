"""S4 prompt set: chat-templated ids, shared by the megakernel and fast-executor drivers (same token ids on both sides).
Cells: natural (all dev+heldout+custom tasks, natural length) + ctx2048 / ctx8192 (7-task copy/edit/prose/terminal subset at padded context)."""
import json, pathlib, sys

sys.path.insert(0, "/work/harness") if pathlib.Path("/work/harness").exists() else None
from common import build_ids, load_workloads
from bench_pld import custom_tasks, pick_ctx_subset


def build(tok, n_new=192, ctxs=(2048, 8192)):
    tasks = [r for r in load_workloads() if r["split"] in ("dev", "heldout")] + custom_tasks()
    cells = [("natural", tasks, None)] + [("ctx%d" % c, pick_ctx_subset(tasks), c) for c in ctxs]
    out = []
    for cname, ts, tl in cells:
        for t in ts:
            ids = build_ids(tok, t, tl) if tl else build_ids(tok, t)
            out.append(dict(cell=cname, task=t["id"], kind=t["kind"], ids=ids))
    return out


def load_or_build(path, model_dir):
    p = pathlib.Path(path)
    if p.exists():
        return json.loads(p.read_text())
    from transformers import AutoTokenizer
    rows = build(AutoTokenizer.from_pretrained(model_dir))
    p.write_text(json.dumps(rows))
    return rows


if __name__ == "__main__":
    rows = load_or_build(sys.argv[1], sys.argv[2])
    print(len(rows), [(r["cell"], r["task"], len(r["ids"])) for r in rows][:60])
