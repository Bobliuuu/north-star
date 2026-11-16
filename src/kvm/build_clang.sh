#!/usr/bin/env bash
# Build moe_fused with clang-CUDA (NVPTX) + NVIDIA ptxas instead of nvcc.
#   src/kvm/build_clang.sh <out_binary> [sm]        sm: 90 (default)
# sass-lane LLVM lane (traceDB traces/sass-lane/h100/llvm-w*-20261116, traces/sass-lane/llvm/README.md): on H100 the
# same source through clang 20 + ptxas beats the best nvcc build (both minBlocks 3, which moe_fused.cu's per-arch
# defaults select for sm_90). Re-measured from this script (modal/ab_clang_run.py, run cl1, traceDB
# traces/sass-lane/h100/ship-clang-20261116): 1.0145x time-weighted on the real-North-routing replay (spread 0.0003) and
# 1.014x on the kernel bench (blocks 1.010-1.020). On B200 and RTX PRO 6000 the lane saw a tie or a loss, so nvcc stays
# the default build everywhere; this is an opt-in for H100. clang_compat.h supplies the __ldcs/__ldcg overloads clang 20's CUDA headers lack (same PTX as nvcc).
set -euo pipefail
OUT=${1:?out binary}; SM=${2:-90}
HERE=$(cd "$(dirname "$0")" && pwd)
CLANG=${CLANG:-clang++-20}
CUDA=${CUDA_HOME:-/usr/local/cuda}
"$CLANG" -x cuda --cuda-path="$CUDA" --cuda-gpu-arch=sm_"$SM" -O3 -std=c++17 -I "$HERE" -I "$HERE/include" \
    -include "$HERE/clang_compat.h" -Xcuda-ptxas -v "$HERE/moe_fused.cu" -o "$OUT" \
    -L"$CUDA/lib64" -lcudart -ldl -lrt -pthread
