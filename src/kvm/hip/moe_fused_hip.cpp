// HIP port of north-star's moe_fused core (src/kvm/moe_fused.cu): one North MoE layer, fused draft/verify expert GEMM.
// Shapes: hidden 2048, expert intermediate 768, 128 experts, top-8, gated SiLU, Wd stored transposed [768][2048].
//
// The CUDA kernel uses 4 warps x 32 lanes and WMMA 16x16x16. AMD Instinct runs 64-lane wavefronts, so this port uses
// 4 wavefronts (256 threads), one 16-column tile per wavefront, the same tiling and the same shared-memory tiles.
//
// MOE_PATH selects the matrix path at compile time (one binary per path, so the ISA of each can be read on its own):
//   0  rocWMMA 16x16x16 bf16 fragments: the straight port, layouts chosen by the library
//   1  __builtin_amdgcn_mfma_f32_16x16x16bf16_1k with explicit lane layouts: the accumulators are written straight to
//      the h tile and the atomics (no fp32 scratch round trip through LDS)
//   2  path 1 + K-permuted wide loads in stage 1: each 16-lane group owns 8 consecutive k (16 bytes) of a 32-wide k
//      step and feeds two MFMAs from one global_load_dwordx4 and one ds_read_b128 (the dot product is order-free, so
//      the same k permutation on A and B leaves the result unchanged up to fp32 summation order)
//   3  path 2 + stage 2 Wd tile staged through LDS with 16-byte global loads (Wd is n-contiguous, MFMA wants k per lane)
// Knobs: MOE_WAVES_PER_EU (0 = none) -> __attribute__((amdgpu_waves_per_eu)), MOE_MINB -> __launch_bounds__ min blocks,
//        MOE_SGB = 1 -> sched_group_barrier interleave of MFMA with VMEM/DS reads in the stage-1 k loop,
//        MOE_SETPRIO = 1 -> s_setprio around the MFMA block.
//
// Modes:  moe_fused_hip selftest            (CPU fp32 reference on one routed shape; prints max rel error)
//         moe_fused_hip bench <out.jsonl>   (fused and two-pass timings over a small shape set, L2 flushed)
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <array>
#include <string>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#if MOE_PATH == 0
#include <rocwmma/rocwmma.hpp>
#endif

#ifndef MOE_PATH
#define MOE_PATH 1
#endif
#ifndef MOE_WAVES_PER_EU
#define MOE_WAVES_PER_EU 0
#endif
#ifndef MOE_MINB
#define MOE_MINB 1
#endif
#ifndef MOE_SGB
#define MOE_SGB 0
#endif
#ifndef MOE_SETPRIO
#define MOE_SETPRIO 0
#endif

typedef __hip_bfloat16 bf16;
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); exit(2);} } while (0)

constexpr int H = 2048, IM = 768, TOPK = 8;   // IM: expert intermediate (rocWMMA defines its own I)
constexpr int SL = 64, NS = IM / SL, MAXTOK = 32;
#ifndef MOE_XPAD
#define MOE_XPAD 8
#endif
#ifndef MOE_UNROLL1
#define MOE_UNROLL1 8   // paths 2/3: unroll of the 32-wide stage-1 k loop (KC/32 = 8 steps)
#endif
#define MOE_STR(x) #x
#define MOE_PRAGMA(x) _Pragma(MOE_STR(x))
#define MOE_PRAGMA_UNROLL1 MOE_PRAGMA(unroll MOE_UNROLL1)
constexpr int KC = 256, XPAD = MOE_XPAD, HPAD = 8, MT_SPLIT = 16;
constexpr int NW = 4, WS = 64, NT = NW * WS;   // 4 wavefronts of 64
constexpr int LDX = KC + XPAD, LDH = SL + HPAD;
constexpr size_t EXPERT_BYTES = (size_t)3 * IM * H * 2;
constexpr int KVM_NO_VAR = -1;

struct MoeTask { int32_t expert, slice, pair_off, pair_cnt, wait_var, wait_ver, pub_var, pub_ver; };

typedef short bf16x4 __attribute__((ext_vector_type(4)));
typedef short bf16x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float silu_mul(float g, float u) { return (g / (1.f + __expf(-g))) * u; }

#if MOE_PATH == 0
// ---------------------------------------------------------------- path 0: rocWMMA
template <int MT>
__device__ __forceinline__ void run_item(const bf16* __restrict__ x, const bf16* __restrict__ wg, const bf16* __restrict__ wu,
                                         const bf16* __restrict__ wd, const MoeTask t, const int* __restrict__ prow,
                                         const float* __restrict__ pw, float* __restrict__ out, bf16* xs, bf16* hs,
                                         float* scratch, int* srow, float* sw) {
    using namespace rocwmma;
    typedef rocwmma::bfloat16_t rb;
    const int tid = threadIdx.x, warp = tid / WS, lane = tid % WS;
    const int cnt = t.pair_cnt;
    if (tid < cnt) { srow[tid] = prow[t.pair_off + tid]; sw[tid] = pw[t.pair_off + tid]; }
    __syncthreads();
    const size_t wbase = ((size_t)t.expert * IM + (size_t)t.slice * SL) * H;
    fragment<accumulator, 16, 16, 16, float> ag[MT], au[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) { fill_fragment(ag[m], 0.f); fill_fragment(au[m], 0.f); }
    const rb* gp = reinterpret_cast<const rb*>(wg + wbase + (size_t)warp * 16 * H);
    const rb* up = reinterpret_cast<const rb*>(wu + wbase + (size_t)warp * 16 * H);
    const rb* xsr = reinterpret_cast<const rb*>(xs);
    for (int kc = 0; kc < H; kc += KC) {
        __syncthreads();
        for (int i = tid; i < MT * 16 * (KC / 8); i += NT) {
            int r = i / (KC / 8), c = (i % (KC / 8)) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (r < cnt) v = *reinterpret_cast<const uint4*>(x + (size_t)srow[r] * H + kc + c);
            *reinterpret_cast<uint4*>(xs + r * LDX + c) = v;
        }
        __syncthreads();
#pragma unroll 16
        for (int ks = 0; ks < KC / 16; ++ks) {
            fragment<matrix_b, 16, 16, 16, rb, col_major> bg, bu;
            load_matrix_sync(bg, gp + kc + ks * 16, H);
            load_matrix_sync(bu, up + kc + ks * 16, H);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                fragment<matrix_a, 16, 16, 16, rb, row_major> a;
                load_matrix_sync(a, xsr + m * 16 * LDX + ks * 16, LDX);
                mma_sync(ag[m], a, bg, ag[m]);
                mma_sync(au[m], a, bu, au[m]);
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
        for (int i = 0; i < ag[m].num_elements; ++i) ag[m].x[i] = silu_mul(ag[m].x[i], au[m].x[i]);
        float* sc = scratch + warp * 256;
        store_matrix_sync(sc, ag[m], 16, mem_row_major);
        for (int i = lane; i < 256; i += WS) hs[(m * 16 + i / 16) * LDH + warp * 16 + i % 16] = __float2bfloat16(sc[i]);
    }
    __syncthreads();
    const rb* dp = reinterpret_cast<const rb*>(wd + wbase);
    const rb* hsr = reinterpret_cast<const rb*>(hs);
    for (int nt = warp; nt < H / 16; nt += NW) {
        fragment<accumulator, 16, 16, 16, float> ac[MT];
#pragma unroll
        for (int m = 0; m < MT; ++m) fill_fragment(ac[m], 0.f);
#pragma unroll
        for (int ks = 0; ks < SL / 16; ++ks) {
            fragment<matrix_b, 16, 16, 16, rb, row_major> b;
            load_matrix_sync(b, dp + (size_t)ks * 16 * H + nt * 16, H);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                fragment<matrix_a, 16, 16, 16, rb, row_major> a;
                load_matrix_sync(a, hsr + m * 16 * LDH + ks * 16, LDH);
                mma_sync(ac[m], a, b, ac[m]);
            }
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            float* sc = scratch + warp * 256;
            store_matrix_sync(sc, ac[m], 16, mem_row_major);
            for (int i = lane; i < 256; i += WS) {
                int r = m * 16 + i / 16;
                if (r < cnt) atomicAdd(out + (size_t)srow[r] * H + nt * 16 + i % 16, sw[r] * sc[i]);
            }
        }
    }
}
#else
// ---------------------------------------------------------------- paths 1-3: MFMA intrinsics
// v_mfma_f32_16x16x16bf16_1k lane layouts (CDNA3 ISA guide, 16x16x16 bf16):
//   A[i][k]: lane = i + 16 * (k / 4), element k % 4     B[k][j]: lane = j + 16 * (k / 4), element k % 4
//   D[i][j]: lane = j + 16 * (i / 4), element i % 4
__device__ __forceinline__ f32x4 mfma16(bf16x4 a, bf16x4 b, f32x4 c) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}
__device__ __forceinline__ short bf_bits(bf16 v) { return *reinterpret_cast<short*>(&v); }

template <int MT>
__device__ __forceinline__ void run_item(const bf16* __restrict__ x, const bf16* __restrict__ wg, const bf16* __restrict__ wu,
                                         const bf16* __restrict__ wd, const MoeTask t, const int* __restrict__ prow,
                                         const float* __restrict__ pw, float* __restrict__ out, bf16* xs, bf16* hs,
                                         bf16* wst, int* srow, float* sw) {
    const int tid = threadIdx.x, warp = tid / WS, lane = tid % WS;
    const int l16 = lane & 15, g4 = lane >> 4;   // column/row within the tile, k group
    const int cnt = t.pair_cnt;
    if (tid < cnt) { srow[tid] = prow[t.pair_off + tid]; sw[tid] = pw[t.pair_off + tid]; }
    __syncthreads();
    const size_t wbase = ((size_t)t.expert * IM + (size_t)t.slice * SL) * H;
    f32x4 ag[MT], au[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) { ag[m] = f32x4{0, 0, 0, 0}; au[m] = f32x4{0, 0, 0, 0}; }
    // B operand of stage 1: gate/up weights, row n = intermediate column (l16 of this wave's tile), contiguous along k
    const bf16* gp = wg + wbase + (size_t)(warp * 16 + l16) * H;
    const bf16* up = wu + wbase + (size_t)(warp * 16 + l16) * H;
    for (int kc = 0; kc < H; kc += KC) {
        __syncthreads();
        for (int i = tid; i < MT * 16 * (KC / 8); i += NT) {
            int r = i / (KC / 8), c = (i % (KC / 8)) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (r < cnt) v = *reinterpret_cast<const uint4*>(x + (size_t)srow[r] * H + kc + c);
            *reinterpret_cast<uint4*>(xs + r * LDX + c) = v;
        }
        __syncthreads();
#if MOE_SETPRIO
        __builtin_amdgcn_s_setprio(1);
#endif
#if MOE_PATH == 1
#pragma unroll 16
        for (int ks = 0; ks < KC / 16; ++ks) {
            const int k0 = kc + ks * 16 + 4 * g4;
            bf16x4 bg = *reinterpret_cast<const bf16x4*>(gp + k0);
            bf16x4 bu = *reinterpret_cast<const bf16x4*>(up + k0);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                bf16x4 a = *reinterpret_cast<const bf16x4*>(xs + (m * 16 + l16) * LDX + ks * 16 + 4 * g4);
                ag[m] = mfma16(a, bg, ag[m]);
                au[m] = mfma16(a, bu, au[m]);
            }
#if MOE_SGB
            __builtin_amdgcn_sched_group_barrier(0x020, 2, 0);       // VMEM reads
            __builtin_amdgcn_sched_group_barrier(0x100, MT, 0);      // DS reads
            __builtin_amdgcn_sched_group_barrier(0x008, 2 * MT, 0);  // MFMA
#endif
        }
#else   // paths 2, 3: 32-wide k steps, lane group g4 owns k = 8*g4 .. 8*g4+7 of the step (one 16-byte load per operand)
MOE_PRAGMA_UNROLL1
        for (int ks = 0; ks < KC / 32; ++ks) {
            const int k0 = kc + ks * 32 + 8 * g4;
            bf16x8 bg = *reinterpret_cast<const bf16x8*>(gp + k0);
            bf16x8 bu = *reinterpret_cast<const bf16x8*>(up + k0);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                bf16x8 a = *reinterpret_cast<const bf16x8*>(xs + (m * 16 + l16) * LDX + ks * 32 + 8 * g4);
                ag[m] = mfma16(a.lo, bg.lo, ag[m]);
                au[m] = mfma16(a.lo, bu.lo, au[m]);
                ag[m] = mfma16(a.hi, bg.hi, ag[m]);
                au[m] = mfma16(a.hi, bu.hi, au[m]);
            }
#if MOE_SGB
            __builtin_amdgcn_sched_group_barrier(0x020, 2, 0);
            __builtin_amdgcn_sched_group_barrier(0x100, MT, 0);
            __builtin_amdgcn_sched_group_barrier(0x008, 4 * MT, 0);
#endif
        }
#endif
#if MOE_SETPRIO
        __builtin_amdgcn_s_setprio(0);
#endif
    }
    // h = bf16(silu(g) * u): D[i][j] lane = j + 16 * (i / 4) -> row 4*g4 + e, column l16 of this wave's tile
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int e = 0; e < 4; ++e)
            hs[(m * 16 + 4 * g4 + e) * LDH + warp * 16 + l16] = __float2bfloat16(silu_mul(ag[m][e], au[m][e]));
    __syncthreads();
    // stage 2: out[tok][n] += sw * sum_k h[tok][k] * Wd[k][n], k over the slice's 64 intermediate columns
    const bf16* dp = wd + wbase;
    for (int nt = warp; nt < H / 16; nt += NW) {
        f32x4 ac[MT];
#pragma unroll
        for (int m = 0; m < MT; ++m) ac[m] = f32x4{0, 0, 0, 0};
#if MOE_PATH == 3
        // stage the [64 k][16 n] Wd tile with 16-byte loads: 128 chunks of 8 n, two per lane, into this wave's LDS area
        bf16* ws = wst + warp * (SL * 16);
#pragma unroll
        for (int c = lane; c < SL * 2; c += WS) {
            const int k = c >> 1, n8 = (c & 1) * 8;
            *reinterpret_cast<uint4*>(ws + k * 16 + n8) = *reinterpret_cast<const uint4*>(dp + (size_t)k * H + nt * 16 + n8);
        }
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
#endif
#pragma unroll
        for (int ks = 0; ks < SL / 16; ++ks) {
            bf16x4 b;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int k = ks * 16 + 4 * g4 + e;
#if MOE_PATH == 3
                b[e] = bf_bits(ws[k * 16 + l16]);
#else
                b[e] = bf_bits(dp[(size_t)k * H + nt * 16 + l16]);
#endif
            }
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                bf16x4 a = *reinterpret_cast<const bf16x4*>(hs + (m * 16 + l16) * LDH + ks * 16 + 4 * g4);
                ac[m] = mfma16(a, b, ac[m]);
            }
        }
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int r = m * 16 + 4 * g4 + e;
                if (r < cnt) atomicAdd(out + (size_t)srow[r] * H + nt * 16 + l16, sw[r] * ac[m][e]);
            }
#if MOE_PATH == 3
        __builtin_amdgcn_wave_barrier();
#endif
    }
}
#endif

#if MOE_WAVES_PER_EU > 0
#define MOE_WPE __attribute__((amdgpu_waves_per_eu(MOE_WAVES_PER_EU)))
#else
#define MOE_WPE
#endif

__global__ void __launch_bounds__(NT, MOE_MINB) MOE_WPE moe_kernel(const bf16* __restrict__ x, const bf16* __restrict__ wg,
                                                  const bf16* __restrict__ wu, const bf16* __restrict__ wd,
                                                  const MoeTask* __restrict__ tasks, const int* __restrict__ prow,
                                                  const float* __restrict__ pw, float* __restrict__ out) {
    __shared__ __attribute__((aligned(16))) bf16 xs[MAXTOK * LDX];
    __shared__ __attribute__((aligned(16))) bf16 hs[MAXTOK * LDH];
#if MOE_PATH == 0
    __shared__ __attribute__((aligned(16))) float scratch[NW * 256];
#else
    __shared__ __attribute__((aligned(16))) bf16 scratch[MOE_PATH == 3 ? NW * SL * 16 : 8];
#endif
    __shared__ int srow[MAXTOK];
    __shared__ float sw[MAXTOK];
    const MoeTask t = tasks[blockIdx.x];
    if (t.pair_cnt > MT_SPLIT) run_item<2>(x, wg, wu, wd, t, prow, pw, out, xs, hs, scratch, srow, sw);
    else run_item<1>(x, wg, wu, wd, t, prow, pw, out, xs, hs, scratch, srow, sw);
}

__global__ void fill_bf16(bf16* p, size_t n, uint32_t seed, float scale) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t)i * 2654435761u ^ (uint32_t)(i >> 32) * 40503u ^ seed * 97531u;
        h ^= h >> 16; h *= 0x7feb352dU; h ^= h >> 15; h *= 0x846ca68bU; h ^= h >> 16;
        p[i] = __float2bfloat16((((h & 0xFFFFFF) / 16777216.f) * 2.f - 1.f) * scale);
    }
}

// ---------------- host ----------------
struct TokRoute { int row; int e[TOPK]; float w[TOPK]; };
struct WorkList { std::vector<MoeTask> tasks; std::vector<int> prow; std::vector<float> pw; int nexp = 0; };
static WorkList build_union(const std::vector<TokRoute>& V, const std::vector<TokRoute>& D, int r, int nE) {
    std::vector<std::vector<std::pair<int, float>>> per(nE);
    for (auto& t : V) for (int k = 0; k < TOPK; ++k) per[t.e[k]].push_back({t.row, t.w[k]});
    for (auto& t : D) for (int k = 0; k < r; ++k) per[t.e[k]].push_back({t.row, t.w[k]});
    WorkList wl;
    for (int e = 0; e < nE; ++e) {
        if (per[e].empty()) continue;
        wl.nexp++; int off = (int)wl.prow.size();
        for (auto& p : per[e]) { wl.prow.push_back(p.first); wl.pw.push_back(p.second); }
        for (int s = 0; s < NS; ++s) wl.tasks.push_back({e, s, off, (int)per[e].size(), KVM_NO_VAR, 0, KVM_NO_VAR, 0});
    }
    return wl;
}
struct DevWL { MoeTask* t = nullptr; int* r = nullptr; float* w = nullptr; int n = 0; };
static DevWL upload(const WorkList& wl) {
    DevWL d; d.n = (int)wl.tasks.size();
    CK(hipMalloc(&d.t, std::max<size_t>(1, d.n) * sizeof(MoeTask))); CK(hipMalloc(&d.r, std::max<size_t>(1, wl.prow.size()) * 4));
    CK(hipMalloc(&d.w, std::max<size_t>(1, wl.pw.size()) * 4));
    if (d.n) { CK(hipMemcpy(d.t, wl.tasks.data(), d.n * sizeof(MoeTask), hipMemcpyHostToDevice));
               CK(hipMemcpy(d.r, wl.prow.data(), wl.prow.size() * 4, hipMemcpyHostToDevice));
               CK(hipMemcpy(d.w, wl.pw.data(), wl.pw.size() * 4, hipMemcpyHostToDevice)); }
    return d;
}
struct Dev { bf16 *x, *wg, *wu, *wd; float* out; };
static void launch(const Dev& D, const DevWL& w, hipStream_t s) {
    if (w.n) moe_kernel<<<w.n, NT, 0, s>>>(D.x, D.wg, D.wu, D.wd, w.t, w.r, w.w, D.out);
}
static TokRoute make_route(std::mt19937& g, int row, int nE) {
    TokRoute t; t.row = row;
    std::vector<int> p(nE); std::iota(p.begin(), p.end(), 0); std::shuffle(p.begin(), p.end(), g);
    std::vector<float> lg(TOPK); std::normal_distribution<float> nd(1.0f, 1.0f);
    for (auto& v : lg) v = nd(g);
    std::sort(lg.begin(), lg.end(), std::greater<float>());
    for (int k = 0; k < TOPK; ++k) { t.e[k] = p[k]; t.w[k] = 1.f / (1.f + expf(-lg[k])); }
    return t;
}
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7FFF + ((u >> 16) & 1); return (uint16_t)(u >> 16); }

static int selftest() {
    const int E = 128, Tv = 12, Td = 12, r = 4;   // up to 24 tokens: exercises both the 1- and 2-row-tile paths
    Dev D; size_t wn = (size_t)E * IM * H;
    CK(hipMalloc(&D.wg, wn * 2)); CK(hipMalloc(&D.wu, wn * 2)); CK(hipMalloc(&D.wd, wn * 2));
    CK(hipMalloc(&D.x, (size_t)MAXTOK * H * 2)); CK(hipMalloc(&D.out, (size_t)MAXTOK * H * 4));
    fill_bf16<<<4096, 256>>>(D.wg, wn, 1, 0.03f); fill_bf16<<<4096, 256>>>(D.wu, wn, 2, 0.03f);
    fill_bf16<<<4096, 256>>>(D.wd, wn, 3, 0.03f); fill_bf16<<<64, 256>>>(D.x, (size_t)MAXTOK * H, 4, 1.0f);
    CK(hipDeviceSynchronize());
    std::mt19937 g(7);
    std::vector<TokRoute> V, Dr;
    for (int i = 0; i < Tv; ++i) V.push_back(make_route(g, i, E));
    for (int i = 0; i < Td; ++i) { TokRoute t = V[i % Tv]; t.row = Tv + i; Dr.push_back(t); }   // identical routing: 24 tokens on 8 experts
    WorkList wl = build_union(V, Dr, r, E); DevWL dw = upload(wl);
    CK(hipMemset(D.out, 0, (size_t)MAXTOK * H * 4)); launch(D, dw, 0); CK(hipDeviceSynchronize());
    const int T = Tv + Td;
    std::vector<float> o((size_t)T * H); CK(hipMemcpy(o.data(), D.out, o.size() * 4, hipMemcpyDeviceToHost));
    std::vector<uint16_t> hx((size_t)T * H), hg((size_t)IM * H), hu((size_t)IM * H), hd((size_t)IM * H);
    CK(hipMemcpy(hx.data(), D.x, hx.size() * 2, hipMemcpyDeviceToHost));
    std::vector<double> ref((size_t)T * H, 0.0);
    std::vector<int> used; for (auto& t : V) for (int k = 0; k < TOPK; ++k) used.push_back(t.e[k]);
    std::sort(used.begin(), used.end()); used.erase(std::unique(used.begin(), used.end()), used.end());
    for (int e : used) {
        CK(hipMemcpy(hg.data(), D.wg + (size_t)e * IM * H, hg.size() * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(hu.data(), D.wu + (size_t)e * IM * H, hu.size() * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(hd.data(), D.wd + (size_t)e * IM * H, hd.size() * 2, hipMemcpyDeviceToHost));
        auto apply = [&](const TokRoute& t, int lim) {
            for (int k = 0; k < lim; ++k) if (t.e[k] == e) {
                for (int j = 0; j < IM; ++j) {
                    double sg = 0, su = 0;
                    for (int c = 0; c < H; ++c) { double xv = bf2f(hx[(size_t)t.row * H + c]); sg += xv * bf2f(hg[(size_t)j * H + c]); su += xv * bf2f(hu[(size_t)j * H + c]); }
                    float hj = bf2f(f2bf((float)((sg / (1.0 + exp(-sg))) * su)));
                    for (int n = 0; n < H; ++n) ref[(size_t)t.row * H + n] += (double)t.w[k] * hj * bf2f(hd[(size_t)j * H + n]);
                }
            }
        };
        for (auto& t : V) apply(t, TOPK);
        for (auto& t : Dr) apply(t, r);
    }
    double maxabs = 0, maxerr = 0;
    for (size_t i = 0; i < ref.size(); ++i) { maxabs = std::max(maxabs, fabs(ref[i])); maxerr = std::max(maxerr, fabs(ref[i] - o[i])); }
    const double rel = maxerr / std::max(maxabs, 1e-30);
    printf("{\"selftest\":1,\"path\":%d,\"tokens\":%d,\"max_abs_err\":%.4e,\"max_abs_ref\":%.4e,\"rel\":%.4e,\"pass\":%d}\n",
           MOE_PATH, T, maxerr, maxabs, rel, rel < 2e-2 ? 1 : 0);
    return rel < 2e-2 ? 0 : 1;
}

static double median(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

static int bench(const std::string& outp, int reps) {
    const int E = 128;
    hipDeviceProp_t prop; CK(hipGetDeviceProperties(&prop, 0));
    Dev D; size_t wn = (size_t)E * IM * H;
    CK(hipMalloc(&D.wg, wn * 2)); CK(hipMalloc(&D.wu, wn * 2)); CK(hipMalloc(&D.wd, wn * 2));
    CK(hipMalloc(&D.x, (size_t)MAXTOK * H * 2)); CK(hipMalloc(&D.out, (size_t)MAXTOK * H * 4));
    fill_bf16<<<4096, 256>>>(D.wg, wn, 1, 0.03f); fill_bf16<<<4096, 256>>>(D.wu, wn, 2, 0.03f);
    fill_bf16<<<4096, 256>>>(D.wd, wn, 3, 0.03f); fill_bf16<<<64, 256>>>(D.x, (size_t)MAXTOK * H, 4, 1.0f);
    CK(hipDeviceSynchronize());
    size_t flush_n = (size_t)512 << 20; void* flush; CK(hipMalloc(&flush, flush_n));   // MI300X: 256 MB Infinity Cache
    hipStream_t st; CK(hipStreamCreate(&st));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    FILE* fo = fopen(outp.c_str(), "w");
    const int shapes[][3] = {{1, 1, 1}, {4, 4, 2}, {8, 8, 4}, {16, 16, 8}};
    for (auto& s : shapes) {
        std::mt19937 g(1000003u + 7919u * s[0] + 104729u * s[1]);
        std::vector<TokRoute> V, Dr;
        for (int i = 0; i < s[0]; ++i) V.push_back(make_route(g, i, E));
        for (int i = 0; i < s[1]; ++i) Dr.push_back(make_route(g, s[0] + i, E));
        WorkList wl = build_union(V, Dr, s[2], E); DevWL dw = upload(wl);
        std::vector<float> ts;
        for (int rep = -3; rep < reps; ++rep) {
            CK(hipMemsetAsync(flush, rep & 255, flush_n, st));
            CK(hipMemsetAsync(D.out, 0, (size_t)MAXTOK * H * 4, st));
            CK(hipEventRecord(e0, st)); launch(D, dw, st); CK(hipEventRecord(e1, st)); CK(hipStreamSynchronize(st));
            if (rep >= 0) { float ms; CK(hipEventElapsedTime(&ms, e0, e1)); ts.push_back(ms * 1000.f); }
        }
        const double med = median(ts), bytes = (double)wl.nexp * EXPERT_BYTES;
        fprintf(fo, "{\"gpu\":\"%s\",\"arch\":\"%s\",\"path\":%d,\"wpe\":%d,\"minb\":%d,\"sgb\":%d,\"setprio\":%d,\"Tv\":%d,\"Td\":%d,\"r\":%d,"
                    "\"experts\":%d,\"us_med\":%.2f,\"us_min\":%.2f,\"GBs\":%.1f}\n",
                prop.name, prop.gcnArchName, MOE_PATH, MOE_WAVES_PER_EU, MOE_MINB, MOE_SGB, MOE_SETPRIO, s[0], s[1], s[2], wl.nexp, med,
                *std::min_element(ts.begin(), ts.end()), bytes / (med * 1e-6) / 1e9);
        fflush(fo);
    }
    fclose(fo);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: moe_fused_hip selftest | bench <out.jsonl> [reps]\n"); return 1; }
    std::string mode = argv[1];
    if (mode == "selftest") return selftest();
    if (mode == "bench" && argc >= 3) return bench(argv[2], argc >= 4 ? atoi(argv[3]) : 30);
    return 1;
}
