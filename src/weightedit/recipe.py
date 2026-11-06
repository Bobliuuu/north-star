"""Bounded action space for weight edits. North proposes a JSON recipe; the harness validates it here."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, fields

KINDS = ("router_lowrank", "router_bias")
QUANTS = ("none", "nvfp4", "int4")


class RecipeError(ValueError):
    pass


@dataclass(frozen=True)
class Recipe:
    expert_r: int = 2
    kind: str = "router_lowrank"
    rank: int = 4
    layers: object = "all"          # "all" or list of MoE layer ids
    steps: int = 100
    lr: float = 1e-3
    window: int = 64
    batch: int = 2
    temperature: float = 1.0
    gain: bool = True
    seed: int = 0
    expert_quant: str = "none"   # draft-only fake-quantized expert weights (verifier unchanged)

    def validate(self, mc) -> "Recipe":
        """mc = HF model config. Raises RecipeError on any out-of-range value."""
        moe = moe_layers(mc)
        K = mc.num_experts_per_tok
        def need(cond, msg):
            if not cond:
                raise RecipeError(msg)
        need(isinstance(self.expert_r, int) and not isinstance(self.expert_r, bool) and 1 <= self.expert_r <= K - 1,
             f"expert_r must be int in 1..{K-1}")
        need(self.kind in KINDS, f"kind must be one of {KINDS}")
        need(isinstance(self.rank, int) and 1 <= self.rank <= 16, "rank in 1..16")
        need(isinstance(self.steps, int) and 1 <= self.steps <= 2000, "steps in 1..2000")
        need(isinstance(self.lr, (int, float)) and 1e-5 <= self.lr <= 1e-2, "lr in 1e-5..1e-2")
        need(isinstance(self.window, int) and 16 <= self.window <= 2048, "window in 16..2048")
        need(isinstance(self.batch, int) and 1 <= self.batch <= 16, "batch in 1..16")
        need(isinstance(self.temperature, (int, float)) and 0.5 <= self.temperature <= 2.0, "temperature in 0.5..2")
        need(isinstance(self.gain, bool), "gain is bool")
        need(isinstance(self.seed, int), "seed is int")
        need(self.expert_quant in QUANTS, f"expert_quant must be one of {QUANTS}")
        if self.layers != "all":
            need(isinstance(self.layers, (list, tuple)) and len(self.layers) > 0, "layers: 'all' or non-empty list")
            need(all(isinstance(i, int) and i in moe for i in self.layers), f"layers must be MoE layer ids {moe[:4]}..")
        return self

    def layer_ids(self, mc):
        moe = moe_layers(mc)
        return moe if self.layers == "all" else sorted(set(self.layers))

    def to_dict(self):
        d = asdict(self)
        if d["layers"] != "all":
            d["layers"] = sorted(d["layers"])
        return d

    @staticmethod
    def from_dict(d: dict) -> "Recipe":
        names = {f.name for f in fields(Recipe)}
        extra = set(d) - names
        if extra:
            raise RecipeError(f"unknown fields {sorted(extra)}")
        return Recipe(**d)

    @staticmethod
    def from_json(s: str) -> "Recipe":
        try:
            d = json.loads(s)
        except json.JSONDecodeError as e:
            raise RecipeError(f"bad json: {e}")
        if not isinstance(d, dict):
            raise RecipeError("recipe must be a JSON object")
        return Recipe.from_dict(d)


def moe_layers(mc):
    return [i for i, t in enumerate(mc.mlp_layer_types) if t != "dense"]


SCHEMA_TEXT = (
    "Recipe JSON fields (all optional, defaults shown): "
    '{"expert_r":2 (int 1..K-1), "kind":"router_lowrank"|"router_bias", "rank":4 (1..16), '
    '"layers":"all" or [MoE layer ids], "steps":100 (1..2000), "lr":1e-3 (1e-5..1e-2), "window":64 (16..2048), '
    '"batch":2 (1..16), "temperature":1.0 (0.5..2), "gain":true, "seed":0, "expert_quant":"none"|"nvfp4"|"int4"}'
)
