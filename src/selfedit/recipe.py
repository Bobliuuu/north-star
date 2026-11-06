from __future__ import annotations

import json
from dataclasses import dataclass, asdict, fields

from weightedit.recipe import moe_layers, RecipeError

from .quant import bits_per_weight

GROUPS = (16, 32, 64, 128)


@dataclass(frozen=True)
class QRecipe:
    int4_layers: tuple = ()
    group: int = 32

    def validate(self, mc, exact_k=None) -> "QRecipe":
        moe = set(moe_layers(mc))
        if self.group not in GROUPS:
            raise RecipeError(f"group must be in {GROUPS}")
        if mc.hidden_size % self.group or mc.intermediate_size % self.group:
            raise RecipeError(f"group {self.group} must divide hidden {mc.hidden_size} and intermediate {mc.intermediate_size}")
        ls = list(self.int4_layers)
        if not all(isinstance(i, int) and not isinstance(i, bool) for i in ls):
            raise RecipeError("int4_layers must be ints")
        if len(set(ls)) != len(ls):
            raise RecipeError("duplicate layers")
        if not all(i in moe for i in ls):
            raise RecipeError(f"layers must be MoE layer ids in {min(moe)}..{max(moe)}")
        if exact_k is not None and len(ls) != exact_k:
            raise RecipeError(f"need exactly {exact_k} layers, got {len(ls)}")
        return self

    def layers(self):
        return sorted(self.int4_layers)

    def bytes_saved(self, mc):
        """INFERRED fraction of the routed-expert byte stream saved vs an FP8 (8-bit) incumbent."""
        return len(self.int4_layers) / len(moe_layers(mc)) * (1 - bits_per_weight(self.group) / 8)

    def to_dict(self):
        return dict(int4_layers=self.layers(), group=self.group)

    @staticmethod
    def from_dict(d):
        names = {f.name for f in fields(QRecipe)}
        if not isinstance(d, dict) or set(d) - names:
            raise RecipeError(f"bad recipe object / unknown fields: {sorted(set(d) - names) if isinstance(d, dict) else type(d)}")
        return QRecipe(int4_layers=tuple(d.get("int4_layers", ())), group=d.get("group", 32))
