"""DraftConfig: the single knob set for draft-mode approximations."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

WEIGHT_CONVENTIONS = ("raw_sigmoid", "renorm", "mass_preserve")
ATTN_SELECTORS = ("oracle_topk", "lsa_flat", "lsa_tree", "subq_style")
EXPERT_POLICIES = ("topr", "eat")


@dataclass(frozen=True)
class LSAParams:
    """Latent-sparse-attention-style index (see docs/APPROX.md for what is/ isn't LSA).
    Defaults are scaled-down analogues of the paper's (pool 4, cluster block = window = 512, ~8 keys/bucket,
    node fanout 4, beam 2048); our contexts are 2k-32k so we use window=cluster block=128 by default."""
    pool_size: int = 4
    bucket_size: int = 8          # average keys per bucket (n_b)
    node_fanout: int = 4          # n_{b,l}
    beam_width: int = 64          # p_l
    start_nodes: int = 64         # lowest level with <= start_nodes nodes starts the beam
    cluster: str = "kmeans"       # "kmeans" (Lloyd, stand-in for Ward) | "position" (contiguous; paper's research option)
    kmeans_iters: int = 4
    head_agg: str = "mean_q"      # "mean_q": one index query per KV head = mean of its query heads (LSA sums heads)


@dataclass(frozen=True)
class EATParams:
    """Hyper-parameters as printed in arXiv 2609.33614 Table 1 (see docs/APPROX.md)."""
    ema_alpha: float = 0.95            # alpha (C_s EMA)
    w: tuple = (0.3, 0.3, 0.4)         # w1,w2,w3
    beta: float = 0.6
    a1: float = 0.5                    # alpha_1
    a2: float = 1.0                    # alpha_2
    tau_scale: float = 1.0             # STAND-IN for the undefined PPL feedback (constant multiplier on tau)
    tau_bounds: tuple | None = (0.1, 0.9)  # our reading of "10%/90% of the routing weight distribution": quantiles of g_i per token
    candidate_scope: str = "native_topk"   # "all" (paper: E over all experts) | "native_topk" (draft: subset of router's top-k)
    online_update: bool = False        # keep updating history from draft-time selections (not in paper)


@dataclass(frozen=True)
class DraftConfig:
    # attention budget (long-range keys per query beyond sink+window) on `attn_layers`; None = untouched
    attn_budget: int | None = None
    attn_selector: str = "oracle_topk"
    attn_layers: object = "global"     # "global" (13 layers) | "global_nope" (12) | tuple of layer ids
    sink: int = 4
    window: int = 128
    lsa: LSAParams = field(default_factory=LSAParams)
    subq_block: int = 64
    subq_cand_mult: int = 2
    attn_chunk: int = 512              # query chunk for the sequence-mode implementation
    # expert budget
    expert_r: int | None = None        # None = native top-k (policy "eat" ignores r except as cap when set)
    weight_convention: str = "raw_sigmoid"
    expert_policy: str = "topr"
    eat: EATParams = field(default_factory=EATParams)
    # layer skipping (never layer 0)
    skip_layers: frozenset = frozenset()

    def __post_init__(self):
        assert self.weight_convention in WEIGHT_CONVENTIONS, self.weight_convention
        assert self.attn_selector in ATTN_SELECTORS, self.attn_selector
        assert self.expert_policy in EXPERT_POLICIES, self.expert_policy
        object.__setattr__(self, "skip_layers", frozenset(self.skip_layers))
        assert 0 not in self.skip_layers, "layer 0 (dense + RoPE global) is never skipped"

    @property
    def attn_on(self):
        return self.attn_budget is not None

    @property
    def expert_on(self):
        return self.expert_r is not None or self.expert_policy == "eat"

    @property
    def is_identity(self):
        return not (self.attn_on or self.expert_on or self.skip_layers)

    def with_(self, **kw):
        return replace(self, **kw)

    def label(self):
        parts = []
        if self.attn_on:
            parts.append(f"attn[{self.attn_selector},k={self.attn_budget}]")
        if self.expert_policy == "eat":
            parts.append(f"eat[{self.weight_convention},s={self.eat.tau_scale},{self.eat.candidate_scope}]")
        elif self.expert_r is not None:
            parts.append(f"r{self.expert_r}[{self.weight_convention}]")
        if self.skip_layers:
            parts.append(f"skip{len(self.skip_layers)}")
        return "+".join(parts) or "full"

    def to_dict(self):
        d = {k: getattr(self, k) for k in ("attn_budget", "attn_selector", "attn_layers", "sink", "window", "expert_r",
                                           "weight_convention", "expert_policy")}
        d["skip_layers"] = sorted(self.skip_layers)
        d["label"] = self.label()
        return d


def attn_layer_ids(cfg_model, sel):
    """Global layers of the model config. 'global' = all full_attention layers (0,4,..,48 for North),
    'global_nope' = those without RoPE (excludes dense layer 0)."""
    glob = [i for i, t in enumerate(cfg_model.layer_types) if t == "full_attention"]
    if sel == "global":
        return glob
    if sel == "global_nope":
        return [i for i in glob if cfg_model.mlp_layer_types[i] != "dense"]
    return sorted(sel)
