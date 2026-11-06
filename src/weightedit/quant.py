"""Fake-quantization of expert weights for the DRAFT path (simulates a 4-bit copy; verifier keeps original weights).
Groups of `group` along the input (last) dim. Formats:
  nvfp4 : e2m1 values {0,.5,1,1.5,2,3,4,6}, per-16 scale rounded to e4m3 (as in NVFP4; global scale ignored)
  int4  : symmetric int4 [-7,7], per-group absmax scale in the weight dtype
Effective bits/weight: nvfp4 = 4 + 8/16 = 4.5, int4g16 = 4 + 16/16 = 5 (fp16 scale) -> bytes fraction vs bf16."""
from __future__ import annotations

import torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
BITS = {"none": 16.0, "nvfp4": 4.5, "int4": 5.0}


def bytes_frac(fmt):
    return BITS[fmt] / 16.0


def fake_quant(w: torch.Tensor, fmt: str, group: int = 16) -> torch.Tensor:
    if fmt == "none":
        return w
    shp, dt = w.shape, w.dtype
    assert shp[-1] % group == 0, f"last dim {shp[-1]} not divisible by {group}"
    x = w.float().reshape(*shp[:-1], shp[-1] // group, group)
    amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    if fmt == "int4":
        scale = (amax / 7).to(torch.float16).float().clamp_min(1e-12)
        q = torch.clamp(torch.round(x / scale), -7, 7) * scale
    elif fmt == "nvfp4":
        scale = (amax / 6)
        if hasattr(torch, "float8_e4m3fn"):
            scale = scale.clamp(max=448.0).to(torch.float8_e4m3fn).float().clamp_min(1e-12)
        grid = E2M1.to(x.device)
        a = (x.abs() / scale).clamp(max=6.0)
        idx = (a.unsqueeze(-1) - grid).abs().argmin(-1)
        q = torch.sign(x) * grid[idx] * scale
    else:
        raise ValueError(fmt)
    return q.reshape(shp).to(dt)
