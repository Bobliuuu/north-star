#!/usr/bin/env bash
# Build north-star's moe_fused for AMD Instinct (HIP port, src/kvm/hip/moe_fused_hip.cpp).
#   src/kvm/hip/build.sh <out_dir> [arch]        arch: gfx942 (MI300X, default) or gfx950 (MI350X)
# Builds the shipped configuration and its baseline:
#   moe_fused_hip     path 2 (K-permuted 16-byte loads: global_load_dwordx4 / ds_read_b128) + LLVM AMDGPU
#                     iterative-ilp scheduler. sass-lane AMD-ISA lane (traceDB traces/sass-lane/{mi300x,mi350x}/amdisa-w1-20261116):
#                     1.30-1.34x on MI300X and 1.27-1.28x on MI350X over path 1, CPU fp64 reference passing.
#   moe_fused_hip_p1  path 1 (hand MFMA intrinsics, default scheduler): the baseline those numbers are against.
# Run: <bin> selftest   (CPU fp64 reference, prints JSON with "pass":1)   |   <bin> bench <out.jsonl> [reps]
set -euo pipefail
OUT=${1:?out dir}; ARCH=${2:-gfx942}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
CXX=${HIPCC:-hipcc}
RP=${ROCM_PATH:+-Wl,-rpath,$ROCM_PATH/lib}
$CXX -O3 -std=c++17 --offload-arch="$ARCH" $RP -DMOE_PATH=2 -mllvm -amdgpu-sched-strategy=iterative-ilp \
    "$HERE/moe_fused_hip.cpp" -o "$OUT/moe_fused_hip"
$CXX -O3 -std=c++17 --offload-arch="$ARCH" $RP -DMOE_PATH=1 "$HERE/moe_fused_hip.cpp" -o "$OUT/moe_fused_hip_p1"
echo "built $OUT/moe_fused_hip (path 2 + iterative-ilp) and $OUT/moe_fused_hip_p1 (path 1) for $ARCH"
