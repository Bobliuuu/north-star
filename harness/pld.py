"""Zero-cost drafting for North: prompt-lookup / suffix-match / n-gram-cache proposers + the shared speculative loop.
No model forward is ever used to draft. A proposer only sees token ids (prompt + committed output).
`spec_loop` is used both by the offline simulator (verify = lookup in a plain-greedy trace) and by the live GPU loop
(verify = graphed block step), so simulated and live round structure are identical."""
from collections import defaultdict


class Lookup:
    """Longest-suffix match over prompt+generated text. ns = descending n-gram lengths tried (ns=(3,) is classic prompt lookup n=3).
    Dict n-gram -> position right after its MOST RECENT earlier occurrence (O(1) incremental; a hashed longest-match, not a true suffix automaton).
    weak_n/weak_cap: if the match found has n <= weak_n, cap the draft at weak_cap tokens."""

    def __init__(self, ns=(3,), weak_n=0, weak_cap=2):
        self.ns, self.weak_n, self.weak_cap = tuple(ns), weak_n, weak_cap
        self.h = []
        self.d = {n: {} for n in self.ns}

    def extend(self, toks):
        h = self.h
        for t in toks:
            p = len(h)
            h.append(t)
            for n in self.ns:  # the n-gram ENDING at p-1 now has its continuation start p
                if p >= n:
                    self.d[n][tuple(h[p - n:p])] = p

    def propose(self, g):
        h, L = self.h, len(self.h)
        for n in self.ns:
            if L < n:
                continue
            s = self.d[n].get(tuple(h[L - n:]))
            if s is not None:
                dr = h[s:s + g]
                if n <= self.weak_n:
                    dr = dr[: self.weak_cap]
                if dr:
                    return dr, n
        return [], 0


class NgramCache:
    """Adaptive n-gram cache: counts[n][context n-gram][next token]. Draft greedily follows the most frequent continuation chain,
    backing off over ns; stops when top count < mincount. Learns online from prompt + committed output."""

    def __init__(self, ns=(3, 2), mincount=1):
        self.ns, self.mincount = tuple(ns), mincount
        self.h = []
        self.c = {n: defaultdict(lambda: defaultdict(int)) for n in self.ns}

    def extend(self, toks):
        h = self.h
        for t in toks:
            p = len(h)
            for n in self.ns:
                if p >= n:
                    self.c[n][tuple(h[p - n:p])][t] += 1
            h.append(t)

    def _next(self, ctx):
        for n in self.ns:
            if len(ctx) >= n:
                m = self.c[n].get(tuple(ctx[-n:]))
                if m:
                    tok, cnt = max(m.items(), key=lambda kv: kv[1])
                    if cnt >= self.mincount:
                        return tok, n
        return None, 0

    def propose(self, g):
        ctx, dr, n0 = list(self.h[-max(self.ns):]), [], 0
        for _ in range(g):
            t, n = self._next(ctx)
            if t is None:
                break
            dr.append(t); ctx.append(t); n0 = max(n0, n)
        return dr, n0


def make(name):
    """pld{n}: exact n-gram lookup; pld3b: n=3 then 2; sam: hashed longest match 12..3; samw: sam but weak matches (n<=4) draft <=2;
    sam2: down to n=2 with weak cap; ngc/ngc2: n-gram cache."""
    if name in ("pld2", "pld3", "pld4"):
        return Lookup((int(name[3:]),))
    if name == "pld3b":
        return Lookup((3, 2))
    if name == "sam":
        return Lookup((12, 8, 6, 5, 4, 3))
    if name == "samw":
        return Lookup((12, 8, 6, 5, 4, 3), weak_n=4, weak_cap=2)
    if name == "sam2":
        return Lookup((12, 8, 6, 5, 4, 3, 2), weak_n=3, weak_cap=2)
    if name == "ngc":
        return NgramCache((3, 2), 1)
    if name == "ngc2":
        return NgramCache((4, 3, 2), 2)
    raise KeyError(name)


PROPOSERS = ("pld2", "pld3", "pld4", "pld3b", "sam", "samw", "sam2", "ngc", "ngc2")


def spec_loop(prop, prompt_ids, n_new, g, verify, eos=None):
    """Greedy speculative decode. verify(tcur, drafts) -> list of len(drafts)+1 greedy next-tokens
    (pred[i] = model argmax after tcur+drafts[:i]). Returns (out, rounds), rounds = [(n_drafts, n_accepted), ...]."""
    prop.extend(prompt_ids)
    tcur, out, rounds = list(prompt_ids), [], []
    done = False
    while len(out) < n_new and not done:
        drafts, _ = prop.propose(g)
        pred = verify(tcur, drafts)
        a = 0
        while a < len(drafts) and drafts[a] == pred[a]:
            a += 1
        new = drafts[:a] + [pred[a]]
        if eos is not None and eos in new:
            new = new[: new.index(eos) + 1]; done = True
        tcur += new; out += new; prop.extend(new)
        rounds.append((len(drafts), a))
    return out[:n_new], rounds


def summarize_rounds(rounds, n):
    nd = [r for r in rounds if r[0] > 0]
    return dict(tokens=n, rounds=len(rounds), verify_rounds=len(nd), tok_per_round=n / len(rounds),
                mean_accepted_per_verify=(sum(a for _, a in nd) / len(nd)) if nd else 0.0,
                draft_hit_rate=len(nd) / len(rounds))


def simulate(name, g, prompt_ids, gen, cost_ms, eos=None):
    """Offline: acceptance against the plain greedy trace `gen`; cost = sum cost_ms[T] per round (T=1+n_drafts, T=1 = plain step)."""
    def verify(tcur, drafts):
        k = len(tcur) - len(prompt_ids)
        return [gen[k + i] if k + i < len(gen) else -1 for i in range(len(drafts) + 1)]
    out, rounds = spec_loop(make(name), prompt_ids, len(gen), g, verify, eos)
    ms = sum(cost_ms[1 + d] for d, _ in rounds)
    n = len(out)
    r = summarize_rounds(rounds, n)
    r.update(ms=ms, plain_ms=n * cost_ms[1], speedup=n * cost_ms[1] / ms, identical_to_trace=out == list(gen[:n]))
    return r
