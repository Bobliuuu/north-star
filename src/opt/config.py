"""Configuration space, workload buckets and architecture constants (from docs/NORTH_ARCH.md)."""
from __future__ import annotations

from dataclasses import dataclass, field

# --- North architecture facts (docs/NORTH_ARCH.md; not measured here) ---
N_LAYERS = 49                      # layer 0 dense, 1..48 MoE
GLOBAL_LAYERS = tuple(range(0, 49, 4))          # 13 global-attention layers
SLIDING_WINDOW = 4096
KV_BYTES_PER_TOKEN_LAYER = 2048    # K+V, 4 kv heads x 128 x 2 B
ATTN_WEIGHT_BYTES = 18_900_000 * 2     # ~18.9M params bf16 per layer (NORTH_ARCH, derived)
DENSE_MLP_BYTES = 3 * 2048 * 3072 * 2
LM_HEAD_BYTES = 262_144 * 2048 * 2
MODES = ("two_pass", "spec_two_pass", "fused")


def is_global(layer: int) -> bool:
    return layer % 4 == 0


@dataclass(frozen=True)
class Bucket:
    """Workload bucket. Policies are cached per bucket. `routing` selects which microbench routing
    mode (uniform / corr75 / identical) stands in for draft-vs-verify expert overlap: an ASSUMPTION
    until real overlap is measured on North traces."""
    workload: str = "code"
    ctx: int = 8192
    batch: int = 1
    routing: str = "corr75"

    @property
    def key(self) -> str:
        return f"{self.workload}|ctx{self.ctx}|b{self.batch}|{self.routing}"


@dataclass(frozen=True)
class Config:
    gamma: int                      # draft block length (tokens drafted per cycle)
    attn_budget: int                # keys attended by draft on GLOBAL layers; 0 = full
    r: int                          # draft expert budget (top-r of native top-8)
    skip: tuple = ()                # layers skipped in the DRAFT only (verifier never skips)
    mode: str = "two_pass"          # expert schedule: two_pass | spec_two_pass | fused

    @property
    def draft_key(self):
        """Key of the calibration row (acceptance does not depend on gamma or mode)."""
        return (self.attn_budget, self.r, tuple(sorted(self.skip)))

    def label(self) -> str:
        sk = "none" if not self.skip else f"{len(self.skip)}L"
        return f"g{self.gamma}/attn{self.attn_budget or 'full'}/r{self.r}/skip-{sk}/{self.mode}"


def layer_groups(n_groups: int, n_layers: int = N_LAYERS):
    """Contiguous layer groups (schedule granularity)."""
    base, extra = divmod(n_layers, n_groups)
    out, s = [], 0
    for g in range(n_groups):
        e = s + base + (1 if g < extra else 0)
        out.append(tuple(range(s, e)))
        s = e
    return out
