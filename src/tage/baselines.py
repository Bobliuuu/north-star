"""Baseline proposers sharing the TagePredictor interface (start_doc/commit/propose)."""
from __future__ import annotations
from collections import defaultdict


class _Hist:
    def __init__(self):
        self.hist = []      # current doc
    def start_doc(self, keep_state=True):
        self.hist = []


class PromptLookup(_Hist):
    """Prompt-lookup decoding: match the last n tokens (n from nmax down to nmin) against the
    CURRENT document only, copy the continuation after the most recent earlier occurrence."""
    def __init__(self, nmax=3, nmin=1, name=None):
        super().__init__()
        self.nmax, self.nmin = nmax, nmin
        self.name = name or f"pld(n={nmax}..{nmin})"
        self.idx = {}   # n -> {ngram tuple: last end position (index of the token after the gram)}
    def start_doc(self, keep_state=True):
        super().start_doc()
        self.idx = {n: {} for n in range(self.nmin, self.nmax + 1)}
    def commit(self, tok):
        h = self.hist
        h.append(tok)
        L = len(h)
        for n in range(self.nmin, self.nmax + 1):
            if L > n:
                # gram ending at L-2 (before tok) -> next token position L-1
                self.idx[n][tuple(h[L - 1 - n:L - 1])] = L - 1
    def propose(self, kmax=8, theta=0.0, **kw):
        h = self.hist
        for n in range(self.nmax, self.nmin - 1, -1):
            if len(h) < n:
                continue
            pos = self.idx[n].get(tuple(h[-n:]))
            if pos is not None:
                out = h[pos:pos + kmax]
                if out:
                    return list(out), [1.0] * len(out)
        return [], []


class SuffixLast(_Hist):
    """Suffix-match proposer over ALL history (session + current doc): longest match among
    orders {1,2,3,4,6,8,12,16,24,32}, continuation after most recent occurrence.
    Approximates SuffixDecoding's longest-suffix match without the frequency-scored tree."""
    ORDERS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
    def __init__(self, min_len=1, name=None):
        super().__init__()
        self.min_len = min_len
        self.name = name or f"suffix-last(min={min_len})"
        self.glob = []
        self.tabs = {n: {} for n in self.ORDERS}
        self.docstart = 0
    def start_doc(self, keep_state=True):
        super().start_doc()
        if not keep_state:
            self.glob = []; self.tabs = {n: {} for n in self.ORDERS}
        self.sep = len(self.glob)
        self.glob.append(-1)  # doc separator, never matches real tokens
    def commit(self, tok):
        g = self.glob
        g.append(tok)
        self.hist.append(tok)
        L = len(g)
        for n in self.ORDERS:
            if L > n:
                self.tabs[n][tuple(g[L - 1 - n:L - 1])] = L - 1
    def propose(self, kmax=8, theta=0.0, **kw):
        g = self.glob
        for n in reversed(self.ORDERS):
            if n < self.min_len or len(self.hist) < n:
                continue
            pos = self.tabs[n].get(tuple(g[-n:]))
            if pos is not None:
                out = []
                for t in g[pos:pos + kmax]:
                    if t < 0:
                        break
                    out.append(t)
                if out:
                    return out, [1.0] * len(out)
        return [], []


class SuffixFreq(SuffixLast):
    """Like SuffixLast but picks the most FREQUENT next token for the longest seen context
    (frequency-scored, closer to SuffixDecoding's counted tree), drafting greedily token by token."""
    def __init__(self, min_len=1, name=None):
        super().__init__(min_len, name or f"suffix-freq(min={min_len})")
        self.cnt = {n: defaultdict(lambda: defaultdict(int)) for n in self.ORDERS}
    def start_doc(self, keep_state=True):
        super().start_doc(keep_state)
        if not keep_state:
            self.cnt = {n: defaultdict(lambda: defaultdict(int)) for n in self.ORDERS}
    def commit(self, tok):
        g = self.glob
        L = len(g)
        for n in self.ORDERS:
            if L >= n:
                self.cnt[n][tuple(g[L - n:L])][tok] += 1
        super().commit(tok)
    def propose(self, kmax=8, theta=0.0, **kw):
        ctx = list(self.glob)
        out = []
        for _ in range(kmax):
            nxt = None
            for n in reversed(self.ORDERS):
                if n < self.min_len or len(ctx) < n:
                    continue
                d = self.cnt[n].get(tuple(ctx[-n:]))
                if d:
                    nxt = max(d.items(), key=lambda kv: kv[1])[0]
                    break
            if nxt is None:
                break
            out.append(nxt); ctx.append(nxt)
        return out, [1.0] * len(out)


class SuffixCal(SuffixLast):
    """Ablation: SuffixLast + a calibrated start gate keyed by the matched order (learned online
    on committed tokens exactly like TAGE's calibration). Isolates 'gating' from 'TAGE structure'."""
    def __init__(self, name=None):
        super().__init__(1, name or "suffix-last+gate")
        self.cal = {}
    def _match(self):
        g = self.glob
        for n in reversed(self.ORDERS):
            if len(self.hist) < n:
                continue
            pos = self.tabs[n].get(tuple(g[-n:]))
            if pos is not None and g[pos] >= 0:
                return n, pos
        return 0, None
    def commit(self, tok):
        n, pos = self._match()
        if pos is not None:
            a, c = self.cal.get(n, (1, 3))
            self.cal[n] = (a + (self.glob[pos] == tok), c + 1)
        super().commit(tok)
    def propose(self, kmax=8, theta=0.0, **kw):
        n, pos = self._match()
        if pos is None:
            return [], []
        a, c = self.cal.get(n, (1, 3))
        if a / c < theta:
            return [], []
        out = []
        for t in self.glob[pos:pos + kmax]:
            if t < 0:
                break
            out.append(t)
        return out, [a / c] * len(out)
