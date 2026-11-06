"""TAGE-like predictor over token history.

Tables T1..Tn are indexed by a hash of the last L_i committed tokens (L_i geometric) and carry a
partial tag. An entry stores a predicted next token, a saturating confidence counter and a
useful bit. T0 is a tagless base table indexed by the last token (bimodal analogue).
Prediction = longest-history tag hit (provider); alt = next-longest hit (or base).
Update only on COMMITTED tokens (never on rejected draft tokens), so rollback = no-op for
table state; only the speculative draft extension is discarded.
"""
from __future__ import annotations
from dataclasses import dataclass, field

MASK = (1 << 61) - 1
BASE = 1_000_003


@dataclass
class TageConfig:
    lengths: tuple = (2, 3, 5, 8, 13, 21, 34)   # geometric-ish history lengths
    log_size: int = 12                           # entries per tagged table = 2**log_size
    tag_bits: int = 10
    ctr_max: int = 7                             # 3-bit confidence 0..7
    base_log_size: int = 14
    u_reset_period: int = 1 << 14                # graceful useful-bit decay (commits)
    alloc_tables: int = 2                        # try to allocate in up to this many longer tables


class TagePredictor:
    name = "tage"

    def __init__(self, cfg: TageConfig | None = None):
        self.cfg = cfg or TageConfig()
        c = self.cfg
        n = len(c.lengths)
        sz = 1 << c.log_size
        # tagged tables: parallel arrays
        self.tag = [[-1] * sz for _ in range(n)]
        self.tok = [[0] * sz for _ in range(n)]
        self.ctr = [[0] * sz for _ in range(n)]
        self.u = [[0] * sz for _ in range(n)]
        bsz = 1 << c.base_log_size
        self.btok = [-1] * bsz
        self.bctr = [0] * bsz
        self.hist: list[int] = []
        self.pref: list[int] = [0]
        self.pw = [1]
        self.commits = 0
        # calibration: P(provider correct) by (provider id, ctr), learned on committed positions
        self.calib = {}
        self._pwL = {}

    # --- session / doc management
    def start_doc(self, keep_state: bool = True):
        """New document / request. History restarts; tables persist if keep_state."""
        self.hist = []
        self.pref = [0]
        if not keep_state:
            self.__init__(self.cfg)

    # --- hashing
    def _ensure_pw(self, n):
        while len(self.pw) <= n:
            self.pw.append(self.pw[-1] * BASE & MASK)

    def _hash(self, L, pref, hlen):
        # hash of last L tokens of a history of length hlen
        self._ensure_pw(L)
        return (pref[hlen] - pref[hlen - L] * self.pw[L]) & MASK

    def _idx_tag(self, t, h):
        c = self.cfg
        x = (h ^ (h >> 29) ^ (t * 0x9E3779B1)) & MASK
        idx = (x * 0x2545F4914F6CDD1D >> 20) & ((1 << c.log_size) - 1)
        tag = (x >> 7 ^ x >> 31) & ((1 << c.tag_bits) - 1)
        return idx, tag

    def _lookup(self, hist, pref):
        """Return list of hits [(table_id, idx)] longest first, plus base idx. hist length = len(pref)-1."""
        hl = len(pref) - 1
        hits = []
        for t in range(len(self.cfg.lengths) - 1, -1, -1):
            L = self.cfg.lengths[t]
            if hl < L:
                continue
            idx, tag = self._idx_tag(t, self._hash(L, pref, hl))
            if self.tag[t][idx] == tag:
                hits.append((t, idx))
        return hits

    def _base_idx(self, hist):
        last = hist[-1] if hist else 0
        return (last * 0x9E3779B1) & ((1 << self.cfg.base_log_size) - 1)

    def _predict_from(self, hist, pref):
        """-> (token or -1, provider_id (0=base, t+1), ctr, alt_token)"""
        hits = self._lookup(hist, pref)
        bi = self._base_idx(hist)
        if hits:
            t, i = hits[0]
            alt = self.tok[hits[1][0]][hits[1][1]] if len(hits) > 1 else self.btok[bi]
            return self.tok[t][i], t + 1, self.ctr[t][i], alt
        return self.btok[bi], 0, self.bctr[bi], -1

    # --- speculative proposal (no state mutation)
    def calibrated(self, prov, ctr):
        a, n = self.calib.get((prov, ctr), (1, 3))  # weak prior 1/3
        return a / n

    def propose(self, kmax: int = 8, theta: float = 0.0, with_alt: bool = False):
        """Draft up to kmax tokens by repeated lookup on the speculatively extended history.
        Stops when the cumulative calibrated accept-chain probability falls below theta
        (adaptive block length; theta=0 -> fixed kmax). Returns (tokens, probs[, alts])."""
        hist = list(self.hist)
        pref = list(self.pref)
        out, probs, alts = [], [], []
        chain = 1.0
        for _ in range(kmax):
            tok, prov, ctr, alt = self._predict_from(hist, pref)
            if tok < 0:
                break
            p = self.calibrated(prov, ctr)
            chain *= p
            if chain < theta:
                break
            out.append(tok); probs.append(chain); alts.append(alt)
            hist.append(tok)
            pref.append((pref[-1] * BASE + tok + 1) & MASK)
        return (out, probs, alts) if with_alt else (out, probs)

    # --- commit (training)
    def commit(self, tok: int):
        c = self.cfg
        hist, pref = self.hist, self.pref
        pred, prov, ctr, _ = self._predict_from(hist, pref)
        if pred >= 0:
            a, n = self.calib.get((prov, ctr), (1, 3))
            self.calib[(prov, ctr)] = (a + (pred == tok), n + 1)
        hits = self._lookup(hist, pref)
        bi = self._base_idx(hist)
        # update provider (or base)
        if hits:
            t, i = hits[0]
            if self.tok[t][i] == tok:
                self.ctr[t][i] = min(c.ctr_max, self.ctr[t][i] + 1)
                if self.ctr[t][i] >= 2:
                    self.u[t][i] = 1
            else:
                if self.ctr[t][i] > 0:
                    self.ctr[t][i] -= 1
                else:
                    self.tok[t][i] = tok
                    self.ctr[t][i] = 1
                self.u[t][i] = 0
            correct = (pred == tok)
            start = t + 1
        else:
            if self.btok[bi] == tok:
                self.bctr[bi] = min(c.ctr_max, self.bctr[bi] + 1)
            elif self.bctr[bi] > 0:
                self.bctr[bi] -= 1
            else:
                self.btok[bi] = tok
                self.bctr[bi] = 1
            correct = (pred == tok)
            start = 0
        # allocate on misprediction in longer tables (only if history long enough)
        if not correct:
            hl = len(hist)
            n_alloc = 0
            for t2 in range(start, len(c.lengths)):
                L = c.lengths[t2]
                if hl < L:
                    break
                idx, tag = self._idx_tag(t2, self._hash(L, pref, hl))
                if self.tag[t2][idx] == tag:
                    continue
                if self.u[t2][idx] == 0:
                    self.tag[t2][idx] = tag
                    self.tok[t2][idx] = tok
                    self.ctr[t2][idx] = 1
                    n_alloc += 1
                    if n_alloc >= c.alloc_tables:
                        break
                else:
                    self.u[t2][idx] = 0  # age so a later alloc can take it
        self.commits += 1
        if self.commits % c.u_reset_period == 0:
            for t in range(len(c.lengths)):
                self.u[t] = [0] * len(self.u[t])
        # extend history
        hist.append(tok)
        pref.append((pref[-1] * BASE + tok + 1) & MASK)

    def state_bytes(self) -> int:
        """Hardware-style cost: token 4B + tag(10b)+ctr(3b)+u(1b) packed in 2B per tagged entry."""
        c = self.cfg
        tagged = len(c.lengths) * (1 << c.log_size) * 6
        base = (1 << c.base_log_size) * 5
        return tagged + base
