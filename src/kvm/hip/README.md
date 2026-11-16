# moe_fused for AMD Instinct (HIP)

`moe_fused_hip.cpp` ports `../moe_fused.cu` (one North MoE layer, fused draft/verify expert GEMM: H 2048, I 768, 128
experts, top-8, gated SiLU) to HIP with 4 wavefronts of 64. It comes verbatim from the sass-lane AMD-ISA lane
(magni-sass `feat/amd-isa` at `27e2040`, `amd/moe_fused_hip.cpp`, sha256 `3743dc3c...81be`). `MOE_PATH` selects the
matrix path: 0 rocWMMA, 1 hand MFMA intrinsics, 2 path 1 with K-permuted 16-byte loads (`global_load_dwordx4` /
`ds_read_b128`), 3 path 2 with the down tile staged in LDS.

`build.sh <out_dir> [gfx942|gfx950]` builds the shipped configuration, path 2 with the LLVM AMDGPU `iterative-ilp`
scheduler (`moe_fused_hip`), and the path 1 baseline (`moe_fused_hip_p1`). Each binary runs `selftest` (CPU fp64
reference) or `bench <out.jsonl> [reps]` (4 shapes, L2 and Infinity Cache flushed before each timed launch).

| chip | path 2 + iterative-ilp vs path 1 | decode shape (1,1,1) | source |
|---|---|---|---|
| MI300X (gfx942) | **1.344x** geomean of 4 shapes, 3 interleaved repeats, spread 0.039 | 125.0 -> 78.7 us | this file through `build.sh`, 2026-11-16 (traceDB `traces/sass-lane/mi300x/ship-hip-20261116`) |
| MI300X (gfx942) | 1.30-1.34x | 124.9 -> 77-79 us | AMD-ISA lane, `traces/sass-lane/mi300x/amdisa-w1-20261116` |
| MI350X (gfx950) | 1.27-1.28x | 91.7 -> 63.5 us | AMD-ISA lane, `traces/sass-lane/mi350x/amdisa-w1-20261116` |

Selftest passes for both builds (rel 1.1e-4 path 2, 4.0e-5 path 1; gate 2e-2). Why it wins, from the ISA: the
hot-loop waitcnts drop from 25 to 11, because the scheduler spends 17 more VGPRs keeping loads in flight, and occupancy
stays at 4 because LDS is the limit. The MI350X needs a ROCm 7 user space; the MI300X runs on the host ROCm.
