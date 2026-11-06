"""Attention-budget selectors for draft mode (axis B), sequence (prefill / teacher-forced) mode.

All selectors return a boolean keep-mask [B,Hkv,Tc,S] (True = key is attended, causal, includes sink+window) and
charge an `AttnCost`.  Attention over the kept keys is EXACT softmax attention (kernel unchanged).

Selectors (labels matter - see docs/APPROX.md):
  OracleTopK   : "reference". Scores q.k against EVERY causal key and keeps the top-k.  This is an ALL-KEYS SCAN:
                 O(S) index work per query - NOT subquadratic.  It upper-bounds what any index can achieve.
  LSAIndex     : LSA-INSPIRED hierarchical latent index (flat = beam over all closed buckets; tree = beam search over
                 node levels).  NOT the paper's trained map (uses raw L2-normalised keys, like the paper's N-LSA
                 'cluster own keys', but applied to a model not trained for it) and Lloyd k-means instead of Ward.
  SubQStyle    : independent, public-disclosure-only 'content-dependent selection + exact attention over selected
                 positions' approximation (two-stage block routing + exact rescoring).  NOT SSA.  Never call it SSA.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .config import DraftConfig, LSAParams

NEG = float("-inf")


@dataclass
class AttnCost:
    """Counts in units of (key-vector, kv-head) so they add across batch and heads.
    keys_causal   : keys a full-attention query would read (sum over queries)
    keys_attended : keys actually attended (incl. sink+window)           -> gather/KV traffic
    index_scored  : vectors dot-producted by the selection step (centroids, node centroids, or ALL keys for oracle)
    index_build   : multiply-adds spent building/updating the index (pooling, k-means, node sums, block means)
    index_bytes   : resident index size (bf16 centroids + int32 member table), summed over layers calls (last build)
    """
    calls: int = 0
    queries: int = 0
    keys_causal: float = 0.0
    keys_attended: float = 0.0
    index_scored: float = 0.0
    index_build: float = 0.0
    index_bytes: float = 0.0

    def merge(self, o: "AttnCost"):
        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(self, f) + getattr(o, f))

    def summary(self):
        q = max(self.queries, 1)
        return dict(calls=self.calls, queries=self.queries,
                    keys_causal_per_q=self.keys_causal / q, keys_attended_per_q=self.keys_attended / q,
                    index_scored_per_q=self.index_scored / q, index_build_total=self.index_build,
                    index_bytes=self.index_bytes,
                    attended_frac=self.keys_attended / max(self.keys_causal, 1))


def _agg_scores(q, K, group, how):
    """q [B,H,T,D]; K [B,Hkv,S,D] -> [B,Hkv,T,S] selection scores (shared across the G query heads of a KV head)."""
    B, H, T, D = q.shape
    Hkv = K.shape[1]
    qg = q.view(B, Hkv, group, T, D).float()
    if how == "mean_q":
        return torch.einsum("bhtd,bhsd->bhts", qg.mean(2), K.float())
    s = torch.einsum("bhgtd,bhsd->bhgts", qg, K.float())
    return s.max(2).values if how == "max" else s.mean(2)


def _masks(qpos, S, sink, window, device):
    kpos = torch.arange(S, device=device)
    causal = kpos[None, :] <= qpos[:, None]
    force = causal & ((kpos[None, :] < sink) | (kpos[None, :] > qpos[:, None] - window))
    return causal, force, causal & ~force  # avail: selectable long-range keys


def _topk_mask(scores, avail, k):
    s = scores.masked_fill(~avail, NEG)
    kk = min(k, s.shape[-1])
    v, i = s.topk(kk, dim=-1)
    return torch.zeros_like(s, dtype=torch.bool).scatter_(-1, i, torch.isfinite(v))


class Selector:
    name = "base"

    def __init__(self, cfg: DraftConfig):
        self.cfg = cfg

    def prepare(self, K, cost: AttnCost):
        """Build index over all keys K [B,Hkv,S,D] (counted in cost.index_build)."""
        return None

    def select(self, q, K, qpos, state, group, cost: AttnCost):
        raise NotImplementedError


class OracleTopK(Selector):
    name = "oracle_topk"

    def __init__(self, cfg, head_agg="max"):
        super().__init__(cfg)
        self.head_agg = head_agg

    def select(self, q, K, qpos, state, group, cost):
        c = self.cfg
        S = K.shape[2]
        causal, force, avail = _masks(qpos, S, c.sink, c.window, K.device)
        sc = _agg_scores(q, K, group, self.head_agg)
        cost.index_scored += float(causal.sum()) * K.shape[0] * K.shape[1]  # ALL-KEYS SCAN
        keep = _topk_mask(sc, avail[None, None], c.attn_budget) | force[None, None]
        return keep


# ------------------------------------------------------------------------------------------------ LSA-style
@dataclass
class _Level:
    ksum: torch.Tensor      # [B,H,N,D] unnormalised sum of keys under node
    cent: torch.Tensor      # [B,H,N,D] L2-normalised
    size: torch.Tensor      # [B,H,N] keys under node (0 = empty)
    end: torch.Tensor       # [N] exclusive end position; node usable by query at pos i iff end <= i+1


@dataclass
class _LSAState:
    levels: list            # level 0 = buckets
    bucket_of_key: torch.Tensor  # [B,H,S] long, -1 = unclustered
    top: int                # beam start level


def _l2n(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


class LSAIndex(Selector):
    name = "lsa"

    def __init__(self, cfg: DraftConfig, tree: bool):
        super().__init__(cfg)
        self.tree = tree
        self.p: LSAParams = cfg.lsa
        assert cfg.window >= self.block, "window must cover the open cluster block (paper: window == cluster block)"

    @property
    def block(self):
        return self.cfg.window  # cluster block == window (LSA paper)

    # ---- build -------------------------------------------------------------------------------
    def prepare(self, K, cost):
        p, Bc = self.p, self.block
        B, H, S, D = K.shape
        P = p.pool_size
        assert Bc % P == 0 and Bc % p.bucket_size == 0
        nb = S // Bc
        dev = K.device
        if nb == 0:
            return _LSAState([], torch.full((B, H, S), -1, device=dev, dtype=torch.long), 0)
        Kn = _l2n(K.float())
        Sc = nb * Bc
        npool = Bc // P
        mb = Bc // p.bucket_size
        pools = Kn[:, :, :Sc].reshape(B, H, nb, npool, P, D).sum(4)  # [B,H,nb,np,D]
        cost.index_build += B * H * Sc * D  # pooling adds
        pn = _l2n(pools)
        if p.cluster == "position":
            a = (torch.arange(npool, device=dev) * mb // npool)[None, None, None].expand(B, H, nb, npool)
        else:
            init = (torch.arange(mb, device=dev) * npool // mb)
            cen = pn[:, :, :, init]  # [B,H,nb,mb,D]
            for _ in range(p.kmeans_iters):
                a = torch.einsum("bhnpd,bhnmd->bhnpm", pn, cen).argmax(-1)
                oh = F.one_hot(a, mb).float()
                new = _l2n(torch.einsum("bhnpm,bhnpd->bhnmd", oh, pn))
                empty = oh.sum(3) == 0
                cen = torch.where(empty[..., None], cen, new)
            a = torch.einsum("bhnpd,bhnmd->bhnpm", pn, cen).argmax(-1)
            cost.index_build += (p.kmeans_iters + 1) * B * H * nb * npool * mb * D * 2
        oh = F.one_hot(a, mb).float()                                   # [B,H,nb,np,mb]
        ksum = torch.einsum("bhnpm,bhnpd->bhnmd", oh, pools).reshape(B, H, nb * mb, D)
        size = (oh.sum(3) * P).reshape(B, H, nb * mb)
        gid = a + (torch.arange(nb, device=dev) * mb)[None, None, :, None]
        bok = torch.full((B, H, S), -1, device=dev, dtype=torch.long)
        bok[:, :, :Sc] = gid.repeat_interleave(P, dim=-1).reshape(B, H, Sc)
        end = ((torch.arange(nb, device=dev) + 1) * Bc).repeat_interleave(mb)
        levels = [_Level(ksum, _l2n(ksum), size, end)]
        top = 0
        if self.tree:
            f = p.node_fanout
            cur = levels[0]
            while cur.size.shape[-1] > p.start_nodes:
                N = cur.size.shape[-1]
                padn = (-N) % f
                ks = F.pad(cur.ksum, (0, 0, 0, padn)).reshape(B, H, -1, f, D).sum(3)
                sz = F.pad(cur.size, (0, padn)).reshape(B, H, -1, f).sum(3)
                en = F.pad(cur.end, (0, padn), value=0).reshape(-1, f).max(1).values
                cost.index_build += B * H * (N + padn) * D
                cur = _Level(ks, _l2n(ks), sz, en)
                levels.append(cur)
            top = len(levels) - 1
        cost.index_bytes += sum(l.cent.numel() * 2 + l.size.numel() * 4 for l in levels) + bok.numel() * 4
        return _LSAState(levels, bok, top)

    # ---- lookup ----------------------------------------------------------------------------------
    def _fill(self, sc, cand, size, kbudget, pmax):
        """From candidate buckets [B,H,Tc,M] pick best-first until cumulative size >= kbudget (crossing bucket kept)."""
        s = sc.masked_fill(~cand, NEG)
        v, order = s.topk(min(pmax, s.shape[-1]), dim=-1)
        sz = size.unsqueeze(2).expand_as(s).gather(-1, order).float()
        before = sz.cumsum(-1) - sz
        take = torch.isfinite(v) & (before < kbudget)
        return torch.zeros_like(cand).scatter_(-1, order, take)

    def select(self, q, K, qpos, st: _LSAState, group, cost):
        c, p = self.cfg, self.p
        B, H, S, D = K.shape
        causal, force, avail = _masks(qpos, S, c.sink, c.window, K.device)
        if not st.levels:
            return force[None, None].expand(B, H, -1, -1)
        qs = _l2n(_agg_scores_q(q, group, p.head_agg))  # [B,H,Tc,D]
        Tc = qs.shape[2]
        lv = st.levels
        vis = [(l.end[None, :] <= (qpos[:, None] + 1))[None, None] & (l.size > 0)[:, :, None, :] for l in lv]  # closed
        sc = [torch.einsum("bhtd,bhnd->bhtn", qs, l.cent) for l in lv]
        scored = 0.0
        if st.top == 0:
            cand0 = vis[0]
            scored += cand0.sum().item()
        else:
            L = st.top
            f = p.node_fanout
            cand = vis[L]
            scored += cand.sum().item()
            kept = self._beam(sc[L], cand)
            for l in range(L - 1, -1, -1):
                par = (torch.arange(lv[l].size.shape[-1], device=K.device) // f)
                parent_closed = vis[l + 1][..., par]
                cand = (kept[..., par] & vis[l]) | (vis[l] & ~parent_closed)  # children of kept + open nodes
                scored += cand.sum().item()
                if l > 0:
                    kept = self._beam(sc[l], cand)
            cand0 = cand
        pmax = max(1, math.ceil(c.attn_budget / p.pool_size))
        picked = self._fill(sc[0], cand0, lv[0].size, c.attn_budget, pmax)
        idx = st.bucket_of_key.clamp_min(0)[:, :, None, :].expand(B, H, Tc, S)
        keep = picked.gather(-1, idx) & (st.bucket_of_key >= 0)[:, :, None, :] & avail[None, None]
        cost.index_scored += scored
        return keep | force[None, None]

    def _beam(self, sc, cand):
        s = sc.masked_fill(~cand, NEG)
        v, i = s.topk(min(self.p.beam_width, s.shape[-1]), dim=-1)
        return torch.zeros_like(cand).scatter_(-1, i, torch.isfinite(v))


def _agg_scores_q(q, group, how):
    B, H, T, D = q.shape
    Hkv = H // group
    qg = q.view(B, Hkv, group, T, D).float()
    return qg.mean(2) if how == "mean_q" else qg.sum(2)


class SubQStyle(Selector):
    """SSA-INSPIRED, independent approximation from public disclosure only (content-dependent selection, exact
    attention over selected positions, selection independent of absolute position).  Two stages:
    (1) route to candidate blocks by q . mean-key of closed blocks, (2) exact q.k rescoring of all keys inside the
    candidates, keep top-k.  Stage 2 scans cand_mult*budget keys per query."""
    name = "subq_style"

    def prepare(self, K, cost):
        bs = self.cfg.subq_block
        B, H, S, D = K.shape
        nb = S // bs
        cost.index_build += B * H * nb * bs * D
        cost.index_bytes += B * H * nb * D * 2
        if nb == 0:
            return None
        m = K[:, :, : nb * bs].float().reshape(B, H, nb, bs, D).mean(3)
        return m

    def select(self, q, K, qpos, mean, group, cost):
        c = self.cfg
        B, H, S, D = K.shape
        bs = c.subq_block
        causal, force, avail = _masks(qpos, S, c.sink, c.window, K.device)
        if mean is None:
            return force[None, None].expand(B, H, -1, -1)
        nb = mean.shape[2]
        bs_score = torch.einsum("bhtd,bhnd->bhtn", _agg_scores_q(q, group, "mean_q"), mean)
        closed = ((torch.arange(nb, device=K.device) + 1) * bs)[None, :] <= (qpos[:, None] + 1)
        ncand = max(1, math.ceil(c.subq_cand_mult * c.attn_budget / bs))
        cand_blk = _topk_mask(bs_score, closed[None, None], ncand)             # [B,H,Tc,nb]
        cost.index_scored += closed.sum().item() * B * H
        key_cand = cand_blk.repeat_interleave(bs, dim=-1)
        key_cand = F.pad(key_cand, (0, S - key_cand.shape[-1]))
        cost.index_scored += key_cand.sum().item() * 1.0  # stage-2 exact rescoring of candidate keys
        sc = _agg_scores(q, K, group, "max")
        return _topk_mask(sc, key_cand & avail[None, None], c.attn_budget) | force[None, None]


def make_selector(cfg: DraftConfig) -> Selector:
    n = cfg.attn_selector
    if n == "oracle_topk":
        return OracleTopK(cfg)
    if n == "lsa_flat":
        return LSAIndex(cfg, tree=False)
    if n == "lsa_tree":
        return LSAIndex(cfg, tree=True)
    if n == "subq_style":
        return SubQStyle(cfg)
    raise ValueError(n)


def sparse_attention(q, K, V, qpos, selector: Selector, cfg: DraftConfig, scaling, cost: AttnCost):
    """q [B,H,T,D], K/V [B,Hkv,S,D], qpos [T] absolute positions of the queries (keys are at 0..S-1).
    Returns attention output [B,H,T,D] (exact softmax over the selected keys)."""
    B, H, T, D = q.shape
    Hkv, S = K.shape[1], K.shape[2]
    G = H // Hkv
    state = selector.prepare(K, cost)
    cost.calls += 1
    outs = []
    for s in range(0, T, cfg.attn_chunk):
        e = min(T, s + cfg.attn_chunk)
        qc, qp = q[:, :, s:e], qpos[s:e]
        keep = selector.select(qc, K, qp, state, G, cost)               # [B,Hkv,Tc,S]
        Tc = e - s
        cost.queries += Tc * B * Hkv
        cost.keys_causal += float((qp + 1).sum()) * B * Hkv
        cost.keys_attended += keep.sum().item()
        qg = qc.reshape(B, Hkv, G, Tc, D).reshape(B, Hkv, G * Tc, D)
        o = F.scaled_dot_product_attention(qg, K, V, attn_mask=keep.repeat(1, 1, G, 1), scale=scaling)
        outs.append(o.reshape(B, Hkv, G, Tc, D).reshape(B, H, Tc, D))
    return torch.cat(outs, 2)


# ---------------------------------------------------------------------------------------------- analytic cost
def decode_step_cost(S: int, selector: str, budget: int, lsa: LSAParams | None = None, window: int = 128,
                     sink: int = 4, subq_block: int = 64, subq_cand_mult: int = 2, head_dim: int = 128):
    """Per-decode-step, per (layer, KV head) cost in vector reads/ops at context S.  Units: key/centroid vectors of
    head_dim. 'full' reads S keys + S values.  Index build is amortised per generated token.
    Honest accounting: index_scan is what the *selection* reads; attend is the gather of K+V for the kept keys."""
    lsa = lsa or LSAParams()
    full = dict(index_scan=0, attend_kv=2 * S, build_per_token=0, index_resident=0)
    k = budget + sink + window
    if selector == "full":
        return full
    if selector == "oracle_topk":  # ALL-KEYS SCAN: O(S), not subquadratic overall
        return dict(index_scan=S, attend_kv=2 * k, build_per_token=0, index_resident=0)
    Bc, P, nb_ = window, lsa.pool_size, lsa.bucket_size
    M = S // nb_
    mb = Bc // nb_
    # amortised build: pooling (D adds/key) + kmeans (iters+1)*np*mb*2 per block / Bc keys
    build = 1 + ((lsa.kmeans_iters + 1) * (Bc // P) * mb * 2) / Bc if lsa.cluster == "kmeans" else 1
    if selector == "lsa_flat":
        return dict(index_scan=M, attend_kv=2 * k, build_per_token=build, index_resident=M + S // 4)
    if selector == "lsa_tree":
        f, n, levels, scan = lsa.node_fanout, M, 0, 0
        counts = [M]
        while counts[-1] > lsa.start_nodes:
            counts.append(math.ceil(counts[-1] / f))
        scan = counts[-1] + sum(min(lsa.beam_width, c) * f for c in counts[:-1]) + 2 * f * len(counts)  # + open nodes
        return dict(index_scan=scan, attend_kv=2 * k, build_per_token=build + 1 / f,
                    index_resident=sum(counts) + S // 4)
    if selector == "subq_style":
        ncand = subq_cand_mult * budget
        return dict(index_scan=S // subq_block + ncand, attend_kv=2 * k, build_per_token=1 / subq_block * 1,
                    index_resident=S // subq_block)
    raise ValueError(selector)
