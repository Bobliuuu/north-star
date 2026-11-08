"""CPU reference + planning model for multi-token causal verify and KV rollback on the cohere-megakernel (RESULTS.md H11 blocker).
Pure numpy, tiny; no GPU, no model. Defines the contract the new megakernel attention instructions must meet:
  * block layout: T=1+n_drafts rows of ONE sequence at positions p..p+T-1, row j attends keys <= p+j (and > p+j-W on sliding layers);
  * KV store: global layers append-only with a logical length (rollback = reset length); sliding layers a ring of R >= W+Tmax-1 slots,
    which makes rollback free (rejected rows can only overwrite slots no committed row still needs);
  * accept rule: identical to pld.spec_loop (longest matching draft prefix + 1 bonus token);
  * projection: expected PLD speedup from a verify-cost curve T_verify(T) (default = H11 megakernel measured K=1..8 curve)."""
import numpy as np

H11_VERIFY_MS = {1: 3.42, 2: 4.04, 3: 4.55, 4: 4.95, 5: 5.47, 6: 6.05, 7: 6.45, 8: 6.68}  # RESULTS.md H11 (real-union emulation)


def block_mask(p, T, W=None):
    """Boolean [T, p+T]: row j (position p+j) may attend key position k iff k <= p+j and (W is None or k > p+j-W)."""
    k = np.arange(p + T)[None, :]
    q = (p + np.arange(T))[:, None]
    m = k <= q
    if W is not None:
        m &= k > q - W
    return m


class LayerKV:
    """One layer's single-sequence KV. W=None: linear (global) cache. W=int: ring of R=W+tmax-1 slots (sliding layer).
    `length` is the committed length; positions >= length are speculative and invisible after rollback()."""

    def __init__(self, d, W=None, tmax=16, cap=4096):
        self.W, self.d = W, d
        self.R = (W + tmax - 1) if W else cap
        self.k = np.zeros((self.R, d)); self.v = np.zeros((self.R, d))
        self.length = 0

    def write(self, pos, k, v):
        s = pos % self.R if self.W else pos
        self.k[s], self.v[s] = k, v

    def read(self, pos):
        s = pos % self.R if self.W else pos
        return self.k[s], self.v[s]

    def commit(self, n_total):
        self.length = n_total

    def rollback(self, n_total):  # global: nothing to restore; ring: slack guarantees nothing needed was clobbered
        self.length = n_total


def attend(q, kv, pos, W=None):
    """Single-row attention of query q at absolute position pos over kv (positions max(0,pos-W+1)..pos)."""
    lo = 0 if W is None else max(0, pos - W + 1)
    K = np.stack([kv.read(i)[0] for i in range(lo, pos + 1)]); V = np.stack([kv.read(i)[1] for i in range(lo, pos + 1)])
    s = K @ q / np.sqrt(len(q)); s = np.exp(s - s.max())
    return (s / s.sum()) @ V


def block_attend(Q, Kn, Vn, kv, p, W=None):
    """Verify-block attention: write all T new K/V rows at p..p+T-1 FIRST (as the kernel would, one pass), then each row attends causally."""
    for j in range(len(Q)):
        kv.write(p + j, Kn[j], Vn[j])
    return np.stack([attend(Q[j], kv, p + j, W) for j in range(len(Q))])


def accept(drafts, pred):
    """pred[i] = argmax after prefix+drafts[:i], len(drafts)+1 entries. Returns (n_accepted, committed tokens)."""
    a = 0
    while a < len(drafts) and drafts[a] == pred[a]:
        a += 1
    return a, list(drafts[:a]) + [pred[a]]


def project(rounds, verify_ms=H11_VERIFY_MS, plain_ms=None):
    """Speedup for a list of (n_drafts, n_accepted) rounds; T>8 extrapolates the last marginal slope."""
    plain = plain_ms or verify_ms[1]
    last = max(verify_ms)
    slope = verify_ms[last] - verify_ms[last - 1]
    cost = lambda T: verify_ms[T] if T in verify_ms else verify_ms[last] + slope * (T - last)
    ms = sum(cost(1 + d) for d, _ in rounds)
    n = sum(a + 1 for _, a in rounds)
    return n * plain / ms


# ---- engine mapping (S1 finding, 2026-11-07): the pinned megakernel already has per-row cache_seqlens (RoPE position + attention
# length = pos+1, i.e. causal mask) and per-row page tables, and QKV writes ALL rows' K/V (one BM=8 tile) before the per-(layer,kv_head)
# barrier the attention waits on. So a T-row block of ONE sequence = BS=T rows with cache_seqlens p..p+T-1 and page-table rows that all
# alias one sequence's pages. No new instruction is needed; only the page-table aliasing below. Sliding layers use linear paged
# addressing with host block eviction (not a ring); rollback is free because positions >= committed length are overwritten next round.
# Engine window_size=W attends W+1 tokens (pos-W..pos); mk_block's W counts tokens, so W_mk = W_engine + 1.
def adopt_blocks(V, new, lo, hi, xp=np):
    """Persistent virtual page table V[L, P] shared by all T rows. Adopt (once) the pages the last row's host allocation made for
    logical blocks lo..hi that V does not have yet; never overwrite or drop an existing entry (history pages stay valid)."""
    idx = (xp.arange(V.shape[-1], device=V.device) if xp is not np else xp.arange(V.shape[-1]))[None, :]
    sel = (V < 0) & (new >= 0) & (idx >= lo) & (idx <= hi)
    return xp.where(sel, new, V)
