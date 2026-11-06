"""Draft-mode approximations for North (cohere2_moe). See docs/APPROX.md.

Nothing here touches the verifier path: all approximations are installed by `hooks.DraftPatch`
(instance-level forward overrides that are removed on exit) and gated by an explicit `active` flag."""
from .config import DraftConfig, EATParams, LSAParams
from .hooks import DraftPatch

__all__ = ["DraftConfig", "EATParams", "LSAParams", "DraftPatch"]
