"""Parse results/SPEND.md (several row formats) into a cumulative estimated optimisation cost."""
import re

_NUM = re.compile(r"^\$?\s*([0-9]+(?:\.[0-9]+)?)$")


def parse_spend(text):
    """Sum 'est USD' over table rows. Rows are `| time | task | gpu | secs | est USD | ... |`.
    Returns (total_usd, n_rows, cap_usd or None)."""
    total, n = 0.0, 0
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5:
            continue
        m = _NUM.match(cells[4])
        if m and re.match(r"^\d", cells[3] or ""):
            total += float(m.group(1))
            n += 1
    cap = re.search(r"of \$([0-9]+(?:\.[0-9]+)?) cap", text) or re.search(r"cap this task: \$([0-9]+)", text)
    return round(total, 4), n, (float(cap.group(1)) if cap else None)


def load_spend(path):
    with open(path) as f:
        return parse_spend(f.read())
