"""Interleaved register programs (mod 10). P registers (letters a..p), each with its own op stream, interleaved.
Modes: 'query' value of one register | 'sum' sum of all finals | 'subset' sum of 3 queried registers.
make() also returns word-level register ids (for oracle routing / visualization)."""
import random
NAMES = "abcdefghijklmnop"
LETTERS = "abcdefghijklmnopqrstuvwxyz"

def make_perm(P, n_swaps, rng=random):
    """pure state tracking: P distinct digits in positions 1..P, n_swaps swaps 'sIJ', answer = digit at queried position"""
    arr = rng.sample(range(10), P); cur = arr[:]
    words = [("".join(map(str, arr)), -1)]
    for _ in range(n_swaps):
        i, j = rng.sample(range(P), 2); cur[i], cur[j] = cur[j], cur[i]
        words.append((f"s{i+1}{j+1}", -1))
    q = rng.randrange(P); words.append(("|", -1)); words.append((f"q{q+1}?", -1)); words.append(("=", -1))
    return words, cur[q]

def make_permn(P, n_swaps, rng=random):
    """named-box permutation tracking: boxes a..  hold distinct digits, ops 'a<>c' swap two boxes, query by box name"""
    arr = rng.sample(range(10), P); cur = arr[:]
    words = [(f"{NAMES[i]}={arr[i]}", i) for i in range(P)]
    for _ in range(n_swaps):
        i, j = rng.sample(range(P), 2); cur[i], cur[j] = cur[j], cur[i]
        words.append((f"{NAMES[i]}<>{NAMES[j]}", i))
    q = rng.randrange(P); words += [("|", -1), (f"{NAMES[q]}?", -1), ("=", -1)]
    return words, cur[q]

# ---- realistic interleaved-entity-state tasks (q12). Natural-language-ish, non-modular, answer = digit 0-9 ----
ENT_NAMES = ("mira bo ana juno kai lena omar pia ravi sol tess uri vera wil xena yuri zed ola "
             "nina otto rhea saul tomas ines hugo dara eli fay gus ida jax kit lou max neo "
             "oz pru quin rex sky tam una val wes yan zoe abe bea cy dot").split()
FILE_NAMES = ("api db ui auth cache cli docs infra jobs log mail net orm perf queue sdk test util web "
              "build core deploy edge form graph hash io json kernel lint math nav").split()

def make_ledger(P, n_ops, rng=random, files=False):
    """P entities (random names) each with a small integer state (0-9, no wraparound), updates interleaved in one text.
    ledger: 'mira starts with 5 .' 'bo deposits 2 .' 'ana withdraws 3 .'  query 'how much does mira have ?'
    files:  'api is at version 5 .' 'db gets 2 new commits .' 'ui reverts 3 commits .' query 'what version is api at ?'"""
    names = rng.sample(FILE_NAMES if files else ENT_NAMES, P)
    val = [rng.randrange(2, 8) for _ in range(P)]
    order = list(range(P)); rng.shuffle(order)
    words = [((f"{names[i]} is at version {val[i]} ." if files else f"{names[i]} starts with {val[i]} ."), i) for i in order]
    up = ("gets {d} new commits", "bumps {d} versions") if files else ("deposits {d}", "receives {d}", "earns {d}")
    dn = ("reverts {d} commits", "drops {d} versions") if files else ("withdraws {d}", "spends {d}", "loses {d}")
    for _ in range(n_ops):
        i = rng.randrange(P); v = val[i]
        can_up, can_dn = v < 9, v > 0
        if can_up and (not can_dn or rng.random() < 0.5):
            d = rng.randrange(1, min(4, 9 - v) + 1); val[i] += d; t = rng.choice(up)
        else:
            d = rng.randrange(1, min(4, v) + 1); val[i] -= d; t = rng.choice(dn)
        words.append((f"{names[i]} {t.format(d=d)} .", i))
    q = rng.randrange(P)
    words.append((f"what version is {names[q]} at ?" if files else f"how much does {names[q]} have ?", -1))
    words.append(("=", -1))
    return words, val[q]

def make(P, n_ops, mode="query", rng=random, rename=False, shuffle_init=False, long_names=False):
    if mode == "ledger": return make_ledger(P, n_ops, rng)
    if mode == "files": return make_ledger(P, n_ops, rng, files=True)
    names = NAMES if not rename else "".join(rng.sample("abcdefghijklmnopqrstuvwxyz", 16))
    if long_names:   # q11: random distinct 2-letter names (676 available) so P can go to 64+; same length at train and test
        names = rng.sample([x + y for x in LETTERS for y in LETTERS], max(P, 1))
    if mode == "perm": return make_perm(P, n_ops, rng)
    if mode == "permn": return make_permn(P, n_ops, rng)
    init = [rng.randrange(10) for _ in range(P)]
    val = init[:]
    order = list(range(P)); shuffle_init and rng.shuffle(order)
    words = [(f"{names[i]}={init[i]}", i) for i in order]
    for _ in range(n_ops):
        i = rng.randrange(P); op = rng.choice("+-*")
        d = rng.randrange(1, 5) if op == "*" else rng.randrange(1, 10)
        val[i] = (val[i] + d) % 10 if op == "+" else (val[i] - d) % 10 if op == "-" else (val[i] * d) % 10
        words.append((f"{names[i]}{op}{d}", i))
    words.append(("|", -1))
    if mode == "query":
        q = rng.randrange(P); ans = val[q]; words.append((f"{names[q]}?", -1))
    elif mode == "subset":
        k = min(3, P); qs = rng.sample(range(P), k); ans = sum(val[i] for i in qs) % 10
        words.append(("+".join(names[i] for i in qs) + "?", -1))
    else:
        ans = sum(val) % 10; words.append(("sum?", -1))
    words.append(("=", -1))
    return words, ans

def render(words):
    text, regs, pos = "", [], 0
    spans = []
    for w, r in words:
        if text: text += " "
        spans.append((len(text), len(text) + len(w), r)); text += w
    return text, spans

def batch(B, Ps, nops_per_reg, mode, rng=random, **kw):
    xs, ys, sp = [], [], []
    for _ in range(B):
        P = rng.choice(Ps); n = P * rng.randrange(nops_per_reg[0], nops_per_reg[1] + 1)
        words, y = make(P, n, mode, rng, **kw); t, s = render(words); xs.append(t); ys.append(y); sp.append(s)
    return xs, ys, sp
