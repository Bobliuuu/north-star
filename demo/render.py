"""Render demo events to terminal lines. Pure functions; no GPU."""
import sys

BAR = "-" * 100


def mode_label(ev):
    return "plain" if ev["mode"] == "plain" else f"self-spec g{ev['gamma']} [{ev.get('draft', '?')}]"


def render_run(ev):
    p = ev["provenance"]
    lines = []
    if p["source"] == "FIXTURE":
        lines.append("#" * 100)
        lines.append("# FIXTURE: hand-made test data, NOT a North measurement. Numbers below are meaningless.")
        lines.append("#" * 100)
    lines += [f"North demo  model={p['model_id']}@{str(p['model_rev'])[:8]}  gpu={p['gpu']}  source={p['source']}  {p['created_utc']}",
              f"task={ev.get('task')}  prompt_len={ev.get('prompt_len')}  baseline_spend_prior=${ev.get('spend_prior_usd', 0):.2f}"
              f" ({ev.get('spend_rows', 0)} logged runs)",
              BAR,
              f"{'step':>4} {'mode':<28} {'tok/s':>7} {'cum':>7} {'acc':>5} {'accC':>5} {'mem GB':>7} {'run $':>7} {'total $':>8}  text",
              BAR]
    return lines


def render_step(ev):
    acc = "  -  " if ev["mode"] == "plain" else f"{ev['accepted']}/{ev['drafted']}".rjust(5)
    accc = "  -  " if ev["accept_cum"] is None else f"{ev['accept_cum']*100:4.0f}%"
    text = ev["text"].replace("\n", "\\n")
    return (f"{ev['i']:>4} {mode_label(ev):<28} {ev['inst_tok_s']:>7.1f} {ev['cum_tok_s']:>7.1f} {acc} {accc} "
            f"{ev['gpu_mem_gb']:>7.1f} {ev['run_cost_usd']:>7.4f} {ev['cum_cost_usd']:>8.4f}  {text}")


def render_end(ev):
    return [BAR, f"done: {ev['committed_total']} tokens in {ev['gen_s']:.2f}s gen = {ev['cum_tok_s']:.1f} tok/s committed; "
            f"peak GPU mem {ev['gpu_peak_gb']:.1f} GB; this run ${ev['run_cost_usd']:.4f}; cumulative optimisation cost ${ev['cum_cost_usd']:.4f}",
            "Throughput is single-stream decode on one A100/H100 with the reference executor (not an optimised engine); see RESULTS.md."]


def render_event(ev):
    t = ev.get("type")
    if t == "run":
        return render_run(ev)
    if t == "step":
        return [render_step(ev)]
    if t == "end":
        return render_end(ev)
    if t == "mode_switch":
        return [f"--- mode switch -> {ev.get('mode')} ---"]
    if t == "error":
        return [f"!!! REMOTE ERROR: {ev.get('error')}"]
    return []


def replay(events, out=None, delay=0.0, sleep=None):
    out = out or sys.stdout
    import time
    sleep = sleep or time.sleep
    for ev in events:
        for line in render_event(ev):
            print(line, file=out)
        if delay and ev.get("type") == "step":
            sleep(delay)
