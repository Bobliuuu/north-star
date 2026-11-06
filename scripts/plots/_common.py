"""Shared helpers. Plots read ONLY raw files in results/. A missing raw file => the plot is skipped (never invented)."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "plots")
LABEL_COL = {"MEASURED": "#1b7837", "INFERRED": "#b8860b", "SIMULATED": "#b2182b"}


def path(*p):
    return os.path.join(RES, *p)


def glob1(prefix, suffix):
    """Newest file in results/ matching prefix*suffix (names carry UTC timestamps); None if absent."""
    c = sorted(f for f in os.listdir(RES) if f.startswith(prefix) and f.endswith(suffix))
    return os.path.join(RES, c[-1]) if c else None


def jsonl(p):
    if not p or not os.path.exists(p):
        return None
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def jload(p):
    if not p or not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def skip(name, what):
    print(f"[plots] SKIP {name}: raw file missing ({what})")
    return None


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def new_fig(title, label, source, w=7.5, h=4.4):
    """label is printed in the title so a figure can never be mistaken for another evidence class."""
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_title(f"[{label}] {title}", fontsize=9, color=LABEL_COL[label], loc="left")
    fig.text(0.01, 0.01, f"source: {source}", fontsize=6.5, color="#555")
    ax.grid(alpha=0.25)
    return fig, ax


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name + ".png")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(p, dpi=130, metadata={"Software": None})  # no timestamps => deterministic bytes
    plt.close(fig)
    print(f"[plots] wrote {os.path.relpath(p, ROOT)}")
    return p
