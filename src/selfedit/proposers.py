"""Layer-set proposers. All return (recipe_dict, meta). Only source='llm' (valid, no fallback) can be self-authored."""
from __future__ import annotations

import json
import random

from weightedit.loop import extract_json
from weightedit.recipe import moe_layers

from .recipe import QRecipe


def greedy_from_table(table: dict, K: int):
    """K least-sensitive layers (smallest single-layer KL)."""
    return sorted(sorted(table, key=lambda i: table[i])[:K])


def random_layers(mc, K, seed):
    return sorted(random.Random(seed).sample(moe_layers(mc), K))


def history_text(history, k=6):
    if not history:
        return "(none yet)"
    return "\n".join(json.dumps(dict(int4_layers=h["layers"], kl=round(h["kl"], 5), agree=round(h["agree"], 4),
                                     passed=h["passed"]), default=float) for h in history[-k:])


class LayerProposer:
    def __init__(self, gen_fn, mc, K, table=None, group=32, fallback_seed=1000):
        self.gen, self.mc, self.K, self.table, self.group, self.fb = gen_fn, mc, K, table, group, fallback_seed
        self.round = 0

    def prompt(self, history):
        moe = moe_layers(self.mc)
        t = ""
        if self.table is not None:
            t = ("Measured single-layer damage (KL in nats/token vs your original outputs when ONLY that layer is "
                 "rounded; smaller = safer): " + ", ".join(f"{i}:{self.table[i]:.4f}" for i in sorted(self.table)) + "\n")
        return ("You are North, a mixture-of-experts coding model. To cut memory bandwidth, the routed-expert weights "
                f"of some MoE layers will be rounded to int4 (group {self.group}). MoE layer ids: {moe[0]}..{moe[-1]} "
                f"({len(moe)} layers; layer 0 is dense and cannot be chosen). Choose EXACTLY {self.K} layers to round so "
                "that your output distribution changes as little as possible (KL to the original). Early and late "
                "layers are often more sensitive.\n" + t + f"Past rounds:\n{history_text(history)}\n"
                'Reply with ONE JSON object and nothing else: {"int4_layers":[...], "group":' + str(self.group) + "}")

    def propose(self, history):
        self.round += 1
        text = ""
        try:
            text = self.gen(self.prompt(history))
            d = extract_json(text)
            if d is None:
                raise ValueError("no json object in reply")
            QRecipe.from_dict(d).validate(self.mc, exact_k=self.K)
            return d, dict(source="llm", raw=text[-800:])
        except Exception as e:  # noqa: BLE001
            d = dict(int4_layers=random_layers(self.mc, self.K, self.fb + self.round), group=self.group)
            return d, dict(source="fallback", error=f"{type(e).__name__}: {e}", raw=text[-800:])
