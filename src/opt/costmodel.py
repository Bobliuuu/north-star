"""Pluggable cost model.

Provenance tags on every duration (`src_us` maps tag -> microseconds):
  measured  : read straight from a microbench JSONL row (MoE expert GEMM, one layer, A100).
  derived   : arithmetic on measured rows (e.g. draft pass = two_pass - verify_only).
  analytic  : roofline from docs/NORTH_ARCH.md byte counts / measured peak bandwidth.
  synthetic : invented constants (launch overheads, resource-demand profiles, index cost). NOT data.
Only the MoE expert layer is measured today. Attention, LM head, index, checkpoint are analytic/synthetic
until a microbench for them lands in results/*.jsonl; plug it in by subclassing/replacing the methods.
"""
from __future__ import annotations

import json
import math
import random
import statistics as st
from collections import defaultdict
from dataclasses import dataclass

from . import config as C

SOURCES = ("measured", "derived", "analytic", "synthetic")


@dataclass(frozen=True)
class Variant:
    """One implementation choice (e.g. tile) of a task: duration and resource demand (percent of A100
    HBM bandwidth / SM capacity, integers 0..100)."""
    name: str
    us: float
    bw: int
    sm: int
    src_us: tuple = ()               # ((source, us), ...)

    def src(self):
        return dict(self.src_us)


def _v(name, parts, bw, sm):
    parts = {k: v for k, v in parts.items() if v > 0}
    return Variant(name, sum(parts.values()), max(1, min(100, int(round(bw)))), max(1, min(100, int(round(sm)))),
                   tuple(sorted(parts.items())))


_MB_KEYS = ("gpu", "routing", "r", "Tv", "Td", "v", "peak_read_GBs")


def is_microbench_row(r) -> bool:
    """True only for kvm_fused microbench rows; results/*.jsonl also holds rows from unrelated benches (skipped, not an error)."""
    if not isinstance(r, dict) or any(k not in r for k in _MB_KEYS):
        return False
    v = r["v"]
    return (isinstance(v, dict) and all(isinstance(v.get(m), dict) for m in ("verify_only", "two_pass", "fused"))
            and "us_med" in v["verify_only"] and "us_med" in v["two_pass"] and "us_med" in v["fused"] and "GBs" in v["fused"])


class MicrobenchTable:
    """Fit from results/*.jsonl rows of the kvm_fused microbench. Median over seeds/repeats."""

    def __init__(self, rows):
        self.rows = [r for r in rows if is_microbench_row(r)]
        if not self.rows:
            raise ValueError("no microbench rows")
        self.gpus = sorted({r["gpu"] for r in self.rows})
        self.peak = st.median(r["peak_read_GBs"] for r in self.rows)
        vo = defaultdict(list); tp = defaultdict(list); fu = defaultdict(list); fg = defaultdict(list)
        for r in self.rows:
            ro, rr, tv, td = r["routing"], r["r"], r["Tv"], r["Td"]
            v = r["v"]
            vo[(ro, tv)].append(v["verify_only"]["us_med"])
            tp[(ro, rr, tv, td)].append(v["two_pass"]["us_med"])
            fu[(ro, rr, tv, td)].append(v["fused"]["us_med"])
            fg[(ro, rr, tv, td)].append(v["fused"]["GBs"])
        self._vo = {k: st.median(x) for k, x in vo.items()}
        self._vo_rel_spread = {k: (st.pstdev(x) / st.mean(x) if len(x) > 1 else 0.0) for k, x in vo.items()}
        self._tp = {k: st.median(x) for k, x in tp.items()}
        self._fu = {k: st.median(x) for k, x in fu.items()}
        self._fu_bw = {k: st.median(x) for k, x in fg.items()}
        self.routings = sorted({k[0] for k in self._vo})
        self.max_tokens = max(k[1] for k in self._vo)
        self.max_r = max(k[1] for k in self._tp)

    @classmethod
    def from_jsonl(cls, *paths):
        rows = []
        for p in paths:
            with open(p) as f:
                rows += [json.loads(l) for l in f if l.strip()]
        return cls(rows)

    def _chk(self, routing, tv, td=1, r=1):
        if routing not in self.routings:
            raise KeyError(f"routing {routing!r} not in table {self.routings}")
        if not (1 <= tv <= self.max_tokens and 1 <= td <= self.max_tokens and 1 <= r <= self.max_r):
            raise ValueError(f"outside measured grid: Tv={tv} Td={td} r={r} "
                             f"(max tokens {self.max_tokens}, max r {self.max_r})")

    def verify_us(self, routing, tv):
        self._chk(routing, tv); return self._vo[(routing, tv)]

    def two_pass_us(self, routing, r, tv, td):
        self._chk(routing, tv, td, r); return self._tp[(routing, r, tv, td)]

    def fused_us(self, routing, r, tv, td):
        self._chk(routing, tv, td, r); return self._fu[(routing, r, tv, td)]

    def fused_bw_pct(self, routing, r, tv, td):
        self._chk(routing, tv, td, r); return 100.0 * self._fu_bw[(routing, r, tv, td)] / self.peak

    def draft_pass_us(self, routing, r, td):
        """Standalone draft-only MoE layer pass = two_pass - verify_only, median over Tv (derived)."""
        self._chk(routing, 1, td, r)
        return st.median(self._tp[(routing, r, tv, td)] - self._vo[(routing, tv)]
                         for tv in range(1, self.max_tokens + 1))

    def measurement_rel_noise(self):
        """Median relative spread of repeated verify_only measurements (floor on cost-model error)."""
        return st.median(self._vo_rel_spread.values())

    def fingerprint(self):
        return f"{len(self.rows)}rows/{self.peak:.1f}GBs/{self.gpus[0]}"


@dataclass
class NorthCostModel:
    table: MicrobenchTable
    routing: str = "corr75"
    bw_eff: float = 0.70             # analytic: achieved/peak HBM BW; measured kernels reach ~0.68-0.71 of peak
    launch_us: float = 6.0           # SYNTHETIC per-task launch/tail overhead for non-measured kernels
    gather_penalty: float = 1.5      # SYNTHETIC: sparse KV gather costs 1.5x bytes vs contiguous
    index_fixed_us: float = 4.0      # SYNTHETIC per global layer index update
    index_per_tok_us: float = 0.5    # SYNTHETIC per new token per global layer
    ckpt_us: float = 3.0             # SYNTHETIC checkpoint save/commit constant
    accept_us: float = 5.0           # SYNTHETIC
    attn_bw: int = 30                # SYNTHETIC resource profile (percent)
    attn_sm: int = 30
    moe_sm: int = 65                 # SYNTHETIC SM demand of the measured MoE kernel (its bw demand IS measured)
    tile_b_speed: float = 0.85       # SYNTHETIC alternative attention tile: faster, hungrier
    name: str = "north-a100-v1"

    @property
    def bw_Bps(self):
        return self.table.peak * 1e9 * self.bw_eff

    def _bytes_us(self, nbytes):
        return nbytes / self.bw_Bps * 1e6

    def provenance_note(self):
        return (f"MoE expert layer measured/derived from microbench (one layer, routing={self.routing}, assumed identical "
                "for all 48 MoE layers); attention/LM-head analytic roofline; launch/index/ckpt/resource profiles synthetic.")

    def fingerprint(self):
        return f"{self.name}|{self.table.fingerprint()}|{self.routing}"

    def attn_group(self, layers, tokens, ctx, attn_budget, draft):
        """Attention branch of a layer group. Sparse budget applies only to GLOBAL layers of DRAFT passes."""
        an = 0.0; syn = 0.0
        for l in layers:
            w = self._bytes_us(C.ATTN_WEIGHT_BYTES)
            if C.is_global(l):
                if draft and attn_budget and attn_budget < ctx:
                    kv = self._bytes_us(attn_budget * C.KV_BYTES_PER_TOKEN_LAYER * self.gather_penalty)
                else:
                    kv = self._bytes_us(ctx * C.KV_BYTES_PER_TOKEN_LAYER)
            else:
                kv = self._bytes_us(min(ctx, C.SLIDING_WINDOW) * C.KV_BYTES_PER_TOKEN_LAYER)
            an += w + kv; syn += self.launch_us
        a = _v("tile_a", {"analytic": an, "synthetic": syn}, self.attn_bw, self.attn_sm)
        b = _v("tile_b", {"analytic": an * self.tile_b_speed, "synthetic": syn * self.tile_b_speed + 0.5 * len(layers)},
               self.attn_bw * 2, self.attn_sm * 2.3)
        return [a, b]

    def dense_group(self, tokens):
        t = self._bytes_us(C.DENSE_MLP_BYTES)
        return [_v("dense", {"analytic": t, "synthetic": self.launch_us}, 70, 60)]

    def moe_group(self, n_layers, tokens, r, mode, phase, n_spec_layers=None, spec_tokens=1):
        """MoE branch for n_layers MoE layers.
        phase=draft : standalone draft pass (derived).
        phase=verify: verify_only (mode two_pass) or fused / measured-two-pass pass that also carries the
        speculative next-cycle draft token in the first n_spec_layers layers (other layers verify_only)."""
        if n_layers == 0:
            return []
        ro = self.routing; t = self.table
        if phase == "draft":
            us = t.draft_pass_us(ro, r, tokens)
            return [_v("expert", {"derived": us * n_layers}, t.fused_bw_pct(ro, r, 1, tokens), self.moe_sm)]
        tv = tokens
        vo = t.verify_us(ro, tv)
        ns = n_layers if n_spec_layers is None else n_spec_layers
        if mode == "two_pass" or ns == 0:
            return [_v("expert", {"measured": vo * n_layers}, t.fused_bw_pct(ro, r, tv, spec_tokens), self.moe_sm)]
        if mode == "fused":
            per = t.fused_us(ro, r, tv, spec_tokens)
        elif mode == "spec_two_pass":
            per = t.two_pass_us(ro, r, tv, spec_tokens)
        else:
            raise ValueError(mode)
        return [_v("expert", {"measured": per * ns + vo * (n_layers - ns)},
                   t.fused_bw_pct(ro, r, tv, spec_tokens), self.moe_sm)]

    def lm_head(self, tokens):
        return [_v("lm_head", {"analytic": self._bytes_us(C.LM_HEAD_BYTES), "synthetic": self.launch_us}, 85, 80)]

    def index_update(self, n_global, tokens):
        return [_v("idx", {"synthetic": n_global * (self.index_fixed_us + self.index_per_tok_us * tokens)}, 20, 20)]

    def ckpt_save(self, tokens):
        by = tokens * C.KV_BYTES_PER_TOKEN_LAYER * 36
        return [_v("ckpt", {"analytic": self._bytes_us(by), "synthetic": self.ckpt_us}, 20, 10)]

    def commit(self):
        return [_v("commit", {"synthetic": self.ckpt_us}, 5, 5)]

    def accept(self):
        return [_v("accept", {"synthetic": self.accept_us}, 5, 5)]

    # memory in bytes
    def draft_scratch_bytes(self, gamma, n_active_layers, batch, attn_budget, ctx):
        kv = gamma * batch * n_active_layers * C.KV_BYTES_PER_TOKEN_LAYER
        idx = 0 if not attn_budget else ctx * 16 * len(C.GLOBAL_LAYERS) * batch   # SYNTHETIC 16 B/token/layer index
        return kv + idx

    def ckpt_bytes(self, tokens):
        return tokens * C.KV_BYTES_PER_TOKEN_LAYER * 36


class PerturbedCostModel:
    """Wraps a cost model, scaling durations by per-kind multipliers (systematic error) -> 'truth' model."""

    _KIND_OF = {"expert": "moe", "tile_a": "attn", "tile_b": "attn", "dense": "dense", "lm_head": "lm_head",
                "idx": "index", "ckpt": "ckpt", "commit": "misc", "accept": "misc"}

    def __init__(self, base, mult):
        self.base = base; self.mult = dict(mult)

    def __getattr__(self, n):
        return getattr(self.base, n)

    def fingerprint(self):
        return self.base.fingerprint() + "|perturbed|" + ",".join(f"{k}={v:.3f}" for k, v in sorted(self.mult.items()))

    def _s(self, vs):
        out = []
        for v in vs:
            m = self.mult.get(self._KIND_OF.get(v.name, "misc"), 1.0)
            out.append(Variant(v.name, v.us * m, v.bw, v.sm, tuple((k, x * m) for k, x in v.src_us)))
        return out

    def attn_group(self, *a, **k): return self._s(self.base.attn_group(*a, **k))
    def dense_group(self, *a, **k): return self._s(self.base.dense_group(*a, **k))
    def moe_group(self, *a, **k): return self._s(self.base.moe_group(*a, **k))
    def lm_head(self, *a, **k): return self._s(self.base.lm_head(*a, **k))
    def index_update(self, *a, **k): return self._s(self.base.index_update(*a, **k))
    def ckpt_save(self, *a, **k): return self._s(self.base.ckpt_save(*a, **k))
    def commit(self, *a, **k): return self._s(self.base.commit(*a, **k))
    def accept(self, *a, **k): return self._s(self.base.accept(*a, **k))


def draw_multipliers(rng: random.Random, sigmas: dict):
    return {k: math.exp(rng.gauss(0.0, s)) for k, s in sigmas.items()}


def default_sigmas(table: MicrobenchTable, scale: float = 1.0):
    """Relative-error sigmas per kind: moe uses measured repeat noise (floored at 5%); analytic kinds 25%;
    synthetic kinds 50%. PRIORS for the perturbation study, not measured model errors."""
    m = max(0.05, table.measurement_rel_noise())
    return {k: scale * s for k, s in {"moe": m, "dense": 0.25, "attn": 0.25, "lm_head": 0.25,
                                      "index": 0.5, "ckpt": 0.5, "misc": 0.5}.items()}
