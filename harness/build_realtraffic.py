"""Build the REAL-traffic PLD workload mix (no cherry-picking): fixed seed, N random draws per public dataset, mechanical truncation only.
  python3 harness/build_realtraffic.py   ->  harness/workloads_real/realtraffic.jsonl + MANIFEST.json
Selection rule: for each source, rng=random.Random(SEED); idx = rng.sample(range(len(split)), N). NO quality/length filtering of which examples are
chosen; only (a) the mechanical truncation listed per category (long text cut at a word boundary to a char budget) and (b) glaive chats must contain an
ASSISTANT turn. The first REP2 draws of each category are the '2-rep variance subset' (20%+; first draws of a seeded sample = unbiased)."""
import json, pathlib, random, re, hashlib

SEED = 20261106
N = 9
REP2 = 2
OUT = pathlib.Path(__file__).resolve().parent / "workloads_real"


def trunc(s, n):
    if len(s) <= n:
        return s
    s = s[:n]
    return s[: s.rfind(" ")] if " " in s[-200:] else s


def parse_glaive(system, chat):
    """glaive chat string -> messages up to (and excluding) the first ASSISTANT turn that follows a FUNCTION RESPONSE; if no tool response,
    up to the first ASSISTANT turn. 'system' is kept as the system message."""
    parts = re.split(r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)", chat)
    turns = []
    for i in range(1, len(parts) - 1, 2):
        role = {"USER:": "user", "ASSISTANT:": "assistant", "FUNCTION RESPONSE:": "user"}[parts[i]]
        txt = parts[i + 1].replace("<|endoftext|>", "").strip()
        turns.append((parts[i], role, txt))
    sysmsg = re.sub(r"^SYSTEM:\s*", "", system.strip())
    cut = None
    for k, (tag, role, txt) in enumerate(turns):
        if tag == "FUNCTION RESPONSE:":
            if k + 1 < len(turns) and turns[k + 1][0] == "ASSISTANT:":
                cut = k + 1
                break
    if cut is None:
        cut = next((k for k, t in enumerate(turns) if t[0] == "ASSISTANT:"), None)
    if cut is None or cut == 0:
        return None
    msgs = [dict(role="system", content=sysmsg)]
    for tag, role, txt in turns[:cut]:
        msgs.append(dict(role=role, content=("[function response] " + txt) if tag == "FUNCTION RESPONSE:" else txt))
    return msgs


def main():
    from datasets import load_dataset
    rows = []

    def draw(name, kw, cat, license_, n=N, filt=None, salt=0, **extra):
        d = load_dataset(name, **kw)
        rng = random.Random(SEED + salt)
        order = rng.sample(range(len(d)), min(len(d), n * 4 if filt else n))
        k = 0
        for i in order:
            m = filt(d[i]) if filt else None
            yield_ = (m if filt else None)
            if filt and yield_ is None:
                continue
            yield d[i], i, k
            k += 1
            if k >= n:
                break

    def add(cat, source, license_, split, i, k, msgs, **kw):
        rows.append(dict(id=f"{cat}_{k:02d}_{i}", kind=cat, source=source, license=license_, split=split, src_index=i, rep2=k < REP2, messages=msgs, **kw))

    for ex, i, k in draw("openai/openai_humaneval", dict(split="test"), "humaneval", "MIT"):
        add("humaneval", "openai/openai_humaneval", "MIT", "test", i, k, [dict(role="user", content="Complete the following Python function. Output only the full function in a code block.\n\n```python\n" + ex["prompt"] + "```")])
    for ex, i, k in draw("google-research-datasets/mbpp", dict(name="full", split="test"), "mbpp", "CC-BY-4.0"):
        add("mbpp", "google-research-datasets/mbpp", "CC-BY-4.0", "test", i, k, [dict(role="user", content=ex["text"] + " Your code should pass these tests:\n\n" + "\n".join(ex["test_list"]))])
    for ex, i, k in draw("nuprl/CanItEdit", dict(split="test"), "canitedit", "MIT"):
        add("canitedit", "nuprl/CanItEdit", "MIT", "test", i, k, [dict(role="user", content="Edit the following code according to the instruction. Output the complete edited code.\n\nInstruction: " + ex["instruction_descriptive"] + "\n\n```python\n" + ex["before"] + "\n```")])
    for ex, i, k in draw("abisee/cnn_dailymail", dict(name="3.0.0", split="test"), "cnn_summ", "Apache-2.0"):
        add("cnn_summ", "abisee/cnn_dailymail 3.0.0", "Apache-2.0", "test", i, k, [dict(role="user", content="Summarize the following news article in a few sentences.\n\n" + trunc(ex["article"], 5000))], trunc_chars=5000)
    for ex, i, k in draw("EdinburghNLP/xsum", dict(split="test"), "xsum_summ", "see dataset card (BBC articles, research use)"):
        add("xsum_summ", "EdinburghNLP/xsum", "see dataset card", "test", i, k, [dict(role="user", content="Summarize the following article in one or two sentences.\n\n" + trunc(ex["document"], 5000))], trunc_chars=5000)
    for ex, i, k in draw("rajpurkar/squad_v2", dict(split="validation"), "squad_qa", "CC-BY-SA-4.0"):
        add("squad_qa", "rajpurkar/squad_v2", "CC-BY-SA-4.0", "validation", i, k, [dict(role="user", content="Answer the question based on the context. If it cannot be answered from the context, say so.\n\nContext:\n" + ex["context"] + "\n\nQuestion: " + ex["question"])])
    for ex, i, k in draw("HuggingFaceH4/ultrachat_200k", dict(split="test_sft"), "ultrachat", "MIT"):
        m = ex["messages"]
        # prompt = every turn up to the last user turn before the first assistant reply... keep multi-turn context: cut before the LAST assistant turn
        last_a = max(j for j, x in enumerate(m) if x["role"] == "assistant")
        add("ultrachat", "HuggingFaceH4/ultrachat_200k", "MIT", "test_sft", i, k, [dict(role=x["role"], content=trunc(x["content"], 6000)) for x in m[:last_a]], turns=last_a)
    # glaive: filter = must parse (has an assistant turn); draw extra candidates in seeded order and take the first N that parse
    d = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    rng = random.Random(SEED); k = 0
    for i in rng.sample(range(len(d)), 200):
        msgs = parse_glaive(d[i]["system"], d[i]["chat"])
        if msgs is None:
            continue
        add("glaive_tool", "glaiveai/glaive-function-calling-v2", "Apache-2.0", "train", i, k, msgs, turns=len(msgs) - 1)
        k += 1
        if k >= N:
            break
    # long-document: GovReport (4K, 8K token budgets via chars ~ 4 chars/token) and QuALITY (full article + question)
    for budget, tag, salt in ((14000, "gov4k", 0), (28000, "gov8k", 1)):
        for ex, i, k in draw("ccdv/govreport-summarization", dict(split="test"), tag, "see dataset card (CRS/GAO public reports)", salt=salt):
            add(tag, "ccdv/govreport-summarization", "public-domain US gov reports", "test", i, k, [dict(role="user", content="Summarize the key findings and recommendations of the following report.\n\n" + trunc(ex["report"], budget))], trunc_chars=budget)
    for ex, i, k in draw("emozilla/quality", dict(split="validation"), "quality_qa", "CC-BY-4.0 (QuALITY)"):
        opts = "\n".join(f"({chr(65 + j)}) {o}" for j, o in enumerate(ex["options"]))
        add("quality_qa", "emozilla/quality", "CC-BY-4.0", "validation", i, k, [dict(role="user", content="Read the article and answer the question. Explain your reasoning briefly, then give the letter.\n\n" + trunc(ex["article"], 32000) + "\n\nQuestion: " + ex["question"] + "\n" + opts)], trunc_chars=32000)
    OUT.mkdir(exist_ok=True)
    txt = "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n"
    (OUT / "realtraffic.jsonl").write_text(txt)
    cats = {}
    for r in rows:
        cats.setdefault(r["kind"], dict(n=0, source=r["source"], license=r["license"]))["n"] += 1
    (OUT / "MANIFEST.json").write_text(json.dumps(dict(seed=SEED, n=len(rows), per_category=cats, sha256=hashlib.sha256(txt.encode()).hexdigest()), indent=1, sort_keys=True))
    print(len(rows), cats)


if __name__ == "__main__":
    main()
