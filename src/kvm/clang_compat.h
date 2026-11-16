// Force-included (-include) in clang-CUDA builds only: CUDA cache-hint loads that clang 20's CUDA headers
// lack for vector types. Same PTX as nvcc's: ld.global.cs (evict-first streaming load).
#pragma once
#if defined(__clang__) && defined(__CUDA__)
#include <vector_types.h>
static __device__ __forceinline__ uint4 __ldcs(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
static __device__ __forceinline__ float4 __ldcg(const float4* p) {
  float4 r;
  asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w) : "l"(p));
  return r;
}
#endif
