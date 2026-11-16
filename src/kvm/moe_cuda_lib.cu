// Shared library around moe_fused.cu's kernel for North's decode path (harness/moe_cuda.py loads it with ctypes).
// Built with -DNS_NORTH_LAYOUT so the kernel reads North's HF expert tensors in place: gate_up [E, 2I, H], down [E, H, I].
// No torch C++ API: the launcher takes raw device pointers and the caller's CUDA stream, so it is CUDA-graph capturable.
#define NS_KERNEL_ONLY
#ifndef NS_NORTH_LAYOUT
#define NS_NORTH_LAYOUT
#endif
#include "moe_fused.cu"

extern "C" int ns_moe_dims(int* h, int* i, int* topk, int* sl) { *h = H; *i = I; *topk = TOPK; *sl = SL; return 0; }

extern "C" int ns_moe_launch(const void* x, const void* gate_up, const void* down, const void* tasks, const void* prow,
                             const void* pw, void* out, int ntasks, void* stream) {
    moe_kernel<<<ntasks, 128, 0, (cudaStream_t)stream>>>((const bf16*)x, (const bf16*)gate_up, nullptr, (const bf16*)down,
                                                         (const MoeTask*)tasks, (const int*)prow, (const float*)pw, (float*)out);
    return (int)cudaGetLastError();
}
