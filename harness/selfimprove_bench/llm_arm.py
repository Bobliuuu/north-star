"""LLM-proposer arm: other models answer North's exact round-0 structured prompt (v5 store) N times; edits are validated by the
same strict parser (selfimprove.structured.parse_edits). Open loop: no measured feedback between samples, matching North's round 0.
These are BENCHMARK ARMS ONLY: nothing here is promoted or counted as North-authored.

  python harness/selfimprove_bench/llm_arm.py --model anthropic/claude-haiku-5.5 --n 33 --out results/selfimprove_bench_llm_<tag>.jsonl

Key: OPENROUTER_API_KEY from ~/.config/openrouter/or.env (never printed). Hard cap via --cap-usd (default 1.5 per model).
"""
import argparse, json, os, pathlib, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from selfimprove import structured  # noqa: E402

TEMPS = [0.3, 0.55, 0.8, 1.0]   # same cycle North's v5 best-of-N used


def north_prompt():
    for l in (ROOT / "results/selfimprove_kernel_structured_v5/proposals.jsonl").read_text().splitlines():
        r = json.loads(l)
        if r.get("event") == "proposal" and str(r.get("round")) == "0":
            return r["prompt"]
    raise SystemExit("no round-0 prompt")


def api_key():
    for l in open(os.path.expanduser("~/.config/openrouter/or.env")):
        if l.startswith("OPENROUTER_API_KEY"):
            return l.split("=", 1)[1].strip().strip("\"'")
    raise SystemExit("no key")


def call(key, model, prompt, temp):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temp, "max_tokens": 1500,
                       "usage": {"include": True}}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    for attempt in range(3):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=180))
            u = d.get("usage", {})
            return d["choices"][0]["message"]["content"] or "", float(u.get("cost") or 0), u.get("prompt_tokens"), u.get("completion_tokens")
        except urllib.error.HTTPError as e:
            if e.code in (402, 401, 403):
                raise SystemExit(f"HTTP {e.code}: stopping (do not top up; ask the user)")
            time.sleep(3 * (attempt + 1))
        except Exception:
            time.sleep(3 * (attempt + 1))
    return None, 0.0, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--n", type=int, default=33)
    ap.add_argument("--out", required=True); ap.add_argument("--cap-usd", type=float, default=1.5); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    key, prompt = api_key(), north_prompt()
    spent = [0.0]

    def one(i):
        if spent[0] >= a.cap_usd:
            return dict(i=i, model=a.model, error="cap_reached")
        t = TEMPS[i % len(TEMPS)]
        raw, cost, tin, tout = call(key, a.model, prompt, t)
        spent[0] += cost
        rec = dict(i=i, model=a.model, temperature=t, cost_usd=cost, tokens_in=tin, tokens_out=tout, raw=(raw or "")[:2000])
        if raw is None:
            rec["error"] = "api_failed"; return rec
        try:
            p = structured.parse_edits(raw)
            rec["edits"] = {n: v for n, v in p["edits"]}; rec["rationale"] = p["rationale"]
        except structured.StructuredEditError as e:
            rec["error"] = f"invalid_edit:{e.args[0]}"
        return rec

    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, range(a.n)))
    pathlib.Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    ok = [r for r in rows if r.get("edits")]
    print(f"{a.model}: {len(rows)} calls, valid_nonempty={len(ok)}, distinct={len({tuple(sorted(r['edits'].items())) for r in ok})}, "
          f"empty={sum(1 for r in rows if r.get('edits') == {})}, invalid={sum(1 for r in rows if str(r.get('error','')).startswith('invalid'))}, spent=${spent[0]:.3f}")


if __name__ == "__main__":
    main()
