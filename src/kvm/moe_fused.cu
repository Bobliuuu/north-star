// Northstar core microbenchmark: fused draft/verify MoE expert GEMM (one North MoE layer).
//
// Shapes (CohereLabs/North-Mini-Code-1.0 config.json): hidden 2048, expert intermediate 768, 128 experts, top-8,
// sigmoid router weights, norm_topk_prob=false, gated SiLU. Each expert: Wg,Wu [768][2048], Wd stored TRANSPOSED
// [768][2048] (row j = down-proj column j) so every weight slice is contiguous.
//
// Variants timed (all use the SAME kernel; only the work lists differ):
//   verify_only   : verify tokens, all 8 routed experts
//   two_pass      : verify launch, then draft launch (draft = top-r of its routed set)   [baseline]
//   fused         : ONE launch over the union of experts; a weight tile is loaded once and applied to every token
//                   (verify and draft) routed to that expert
//   *_graph       : the same, captured in a CUDA graph
//
// Work item = (expert, 64-wide slice of the intermediate dim). A block streams gate/up/down weights of that slice once
// and applies them to all tokens of the expert via tensor cores (WMMA bf16, fp32 accumulate). Output is accumulated
// with fp32 atomics (order nondeterministic -> compare with tolerance).
//
// Modes:  moe_fused bench <out.jsonl> [--full] [--seeds N] [--reps N]
//         moe_fused check <dir>      (reads python-generated tensors, writes out_fused.bin / out_two.bin)
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include "include/kvm_task_abi.h"

using namespace nvcuda;
typedef __nv_bfloat16 bf16;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2);} } while (0)

constexpr int H = 2048, I = 768, TOPK = 8;
constexpr int SL = 64;                 // intermediate columns per work item
constexpr int NS = I / SL;             // 12 slices per expert
constexpr int KC = 256;                // k chunk staged in smem for stage 1
constexpr int LDX = KC + 8, LDH = SL + 8;
constexpr int MAXTOK = 32;             // max tokens routed to one expert (Tv+Td <= 32)
constexpr size_t EXPERT_BYTES = (size_t)3 * I * H * 2;

// Task descriptor for one work item. Mirrors the ABI v2 intent (op, args, wait/publish versions); in this
// single-launch microbenchmark all items are independent so wait/pub are KVM_NO_VAR and the dependency fields are
// carried only to show the mapping (see DESIGN.md). Layout: 8 x int32.
struct MoeTask {
    int32_t expert, slice, pair_off, pair_cnt;
    int32_t wait_var, wait_ver, pub_var, pub_ver;
};

template <int MT>
__device__ __forceinline__ void run_item(const bf16* __restrict__ x, const bf16* __restrict__ wg, const bf16* __restrict__ wu,
                                         const bf16* __restrict__ wd, const MoeTask t, const int* __restrict__ prow,
                                         const float* __restrict__ pw, float* __restrict__ out, bf16* xs, bf16* hs,
                                         float* scratch, int* srow, float* sw) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int cnt = t.pair_cnt;
    if (tid < cnt) { srow[tid] = prow[t.pair_off + tid]; sw[tid] = pw[t.pair_off + tid]; }
    __syncthreads();
    const size_t wbase = ((size_t)t.expert * I + (size_t)t.slice * SL) * H;
    // ---- stage 1: gate/up for j-tile `warp` (16 intermediate cols), all tokens ----
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> ag[MT], au[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) { wmma::fill_fragment(ag[m], 0.f); wmma::fill_fragment(au[m], 0.f); }
    const bf16* gp = wg + wbase + (size_t)warp * 16 * H;
    const bf16* up = wu + wbase + (size_t)warp * 16 * H;
    for (int kc = 0; kc < H; kc += KC) {
        __syncthreads();
        for (int i = tid; i < MT * 16 * (KC / 8); i += blockDim.x) {
            int r = i / (KC / 8), c = (i % (KC / 8)) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (r < cnt) v = *reinterpret_cast<const uint4*>(x + (size_t)srow[r] * H + kc + c);
            *reinterpret_cast<uint4*>(xs + r * LDX + c) = v;
        }
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < KC / 16; ++ks) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> bg, bu;
            wmma::load_matrix_sync(bg, gp + kc + ks * 16, H);
            wmma::load_matrix_sync(bu, up + kc + ks * 16, H);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
                wmma::load_matrix_sync(a, xs + m * 16 * LDX + ks * 16, LDX);
                wmma::mma_sync(ag[m], a, bg, ag[m]);
                wmma::mma_sync(au[m], a, bu, au[m]);
            }
        }
    }
    // h = bf16( silu(g) * u )
#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
        for (int i = 0; i < ag[m].num_elements; ++i) {
            float g = ag[m].x[i];
            ag[m].x[i] = (g / (1.f + __expf(-g))) * au[m].x[i];
        }
        float* sc = scratch + warp * 256;
        wmma::store_matrix_sync(sc, ag[m], 16, wmma::mem_row_major);
        __syncwarp();
        for (int i = lane; i < 256; i += 32) hs[(m * 16 + i / 16) * LDH + warp * 16 + i % 16] = __float2bfloat16(sc[i]);
        __syncwarp();
    }
    __syncthreads();
    // ---- stage 2: partial down projection of this slice, accumulate into out with routing weight ----
    const bf16* dp = wd + wbase;
    for (int nt = warp; nt < H / 16; nt += 4) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> ac[MT];
#pragma unroll
        for (int m = 0; m < MT; ++m) wmma::fill_fragment(ac[m], 0.f);
#pragma unroll
        for (int ks = 0; ks < SL / 16; ++ks) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> b;
            wmma::load_matrix_sync(b, dp + (size_t)ks * 16 * H + nt * 16, H);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
                wmma::load_matrix_sync(a, hs + m * 16 * LDH + ks * 16, LDH);
                wmma::mma_sync(ac[m], a, b, ac[m]);
            }
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            float* sc = scratch + warp * 256;
            wmma::store_matrix_sync(sc, ac[m], 16, wmma::mem_row_major);
            __syncwarp();
            for (int i = lane; i < 256; i += 32) {
                int r = m * 16 + i / 16;
                if (r < cnt) atomicAdd(out + (size_t)srow[r] * H + nt * 16 + i % 16, sw[r] * sc[i]);
            }
            __syncwarp();
        }
    }
}

__global__ void __launch_bounds__(128) moe_kernel(const bf16* __restrict__ x, const bf16* __restrict__ wg,
                                                  const bf16* __restrict__ wu, const bf16* __restrict__ wd,
                                                  const MoeTask* __restrict__ tasks, const int* __restrict__ prow,
                                                  const float* __restrict__ pw, float* __restrict__ out) {
    __shared__ __align__(16) bf16 xs[MAXTOK * LDX];
    __shared__ __align__(16) bf16 hs[MAXTOK * LDH];
    __shared__ __align__(16) float scratch[4 * 256];
    __shared__ int srow[MAXTOK];
    __shared__ float sw[MAXTOK];
    const MoeTask t = tasks[blockIdx.x];
    if (t.pair_cnt > 16) run_item<2>(x, wg, wu, wd, t, prow, pw, out, xs, hs, scratch, srow, sw);
    else run_item<1>(x, wg, wu, wd, t, prow, pw, out, xs, hs, scratch, srow, sw);
}

__global__ void fill_bf16(bf16* p, size_t n, uint32_t seed, float scale) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t)i * 2654435761u ^ (uint32_t)(i >> 32) * 40503u ^ seed * 97531u;
        h ^= h >> 16; h *= 0x7feb352dU; h ^= h >> 15; h *= 0x846ca68bU; h ^= h >> 16;
        p[i] = __float2bfloat16((((h & 0xFFFFFF) / 16777216.f) * 2.f - 1.f) * scale);
    }
}
__global__ void read_kernel(const uint4* p, size_t n, unsigned* sink) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = __ldcs(p + i); acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) *sink = acc;
}

// ---------------- host: routing + work lists ----------------
struct TokRoute { int row; int e[TOPK]; float w[TOPK]; };   // experts sorted by router score desc

struct WorkList {
    std::vector<MoeTask> tasks; std::vector<int> prow; std::vector<float> pw; int nexp = 0, npairs = 0;
};
static WorkList build(const std::vector<TokRoute>& toks, int rlim, int nE) {
    std::vector<std::vector<std::pair<int, float>>> per(nE);
    for (auto& t : toks) for (int k = 0; k < rlim; ++k) per[t.e[k]].push_back({t.row, t.w[k]});
    WorkList wl;
    for (int e = 0; e < nE; ++e) {
        if (per[e].empty()) continue;
        wl.nexp++;
        int off = (int)wl.prow.size();
        for (auto& p : per[e]) { wl.prow.push_back(p.first); wl.pw.push_back(p.second); }
        wl.npairs += (int)per[e].size();
        for (int s = 0; s < NS; ++s) wl.tasks.push_back({e, s, off, (int)per[e].size(), KVM_NO_VAR, 0, KVM_NO_VAR, 0});
    }
    return wl;
}
// Per-launch device copy of a work list
struct DevWL { MoeTask* t = nullptr; int* r = nullptr; float* w = nullptr; int n = 0; };
static DevWL alloc_wl(int maxtasks, int maxpairs) {
    DevWL d; CK(cudaMalloc(&d.t, maxtasks * sizeof(MoeTask))); CK(cudaMalloc(&d.r, maxpairs * 4)); CK(cudaMalloc(&d.w, maxpairs * 4)); return d;
}
static void upload(DevWL& d, const WorkList& wl) {
    d.n = (int)wl.tasks.size();
    if (d.n) { CK(cudaMemcpy(d.t, wl.tasks.data(), d.n * sizeof(MoeTask), cudaMemcpyHostToDevice));
               CK(cudaMemcpy(d.r, wl.prow.data(), wl.prow.size() * 4, cudaMemcpyHostToDevice));
               CK(cudaMemcpy(d.w, wl.pw.data(), wl.pw.size() * 4, cudaMemcpyHostToDevice)); }
}
struct Dev { bf16 *x, *wg, *wu, *wd; float* out; };
static void launch(const Dev& D, const DevWL& w, cudaStream_t s) {
    if (w.n) moe_kernel<<<w.n, 128, 0, s>>>(D.x, D.wg, D.wu, D.wd, w.t, w.r, w.w, D.out);
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
// correlated draft: copy verify token's routed set, replace each slot w.p. 0.25 by a random expert not already in the set
static TokRoute make_corr(std::mt19937& g, const TokRoute& src, int row, int nE, float pr) {
    TokRoute t = src; t.row = row; std::uniform_real_distribution<float> u(0, 1);
    for (int k = 0; k < TOPK; ++k) if (u(g) < pr) {
        for (;;) { int c = g() % nE; bool dup = false; for (int j = 0; j < TOPK; ++j) dup |= (t.e[j] == c); if (!dup) { t.e[k] = c; break; } }
    }
    return t;
}

static void renorm_draft(std::vector<TokRoute>& D, int r) {   // draft weight convention "sum1": w_k / sum_{k<r} w_k
    for (auto& t : D) { float z = 0; for (int k = 0; k < r; ++k) z += t.w[k]; for (int k = 0; k < r; ++k) t.w[k] /= z; }
}
// weight bytes if the GEMM tile is only N columns wide (megakernel gemm-n8: tokens of an expert beyond N re-stream the weights)
static double bytes_tiled(const std::vector<TokRoute>& a, int ra, const std::vector<TokRoute>& b, int rb, int nE, int N) {
    std::vector<int> c(nE, 0);
    for (auto& t : a) for (int k = 0; k < ra; ++k) c[t.e[k]]++;
    for (auto& t : b) for (int k = 0; k < rb; ++k) c[t.e[k]]++;
    double n = 0; for (int e = 0; e < nE; ++e) n += (c[e] + N - 1) / N; return n * EXPERT_BYTES;
}
static double median(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

// ---------------- check mode ----------------
static std::vector<char> slurp(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(3); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); std::vector<char> b(n);
    if (fread(b.data(), 1, n, f) != (size_t)n) exit(3); fclose(f); return b;
}
template <class T> static T* to_dev(const std::vector<char>& b) { T* p; CK(cudaMalloc(&p, b.size())); CK(cudaMemcpy(p, b.data(), b.size(), cudaMemcpyHostToDevice)); return p; }

static int check_mode(const std::string& dir) {
    int E, Tv, Td, r, rn = 0; { FILE* f = fopen((dir + "/meta.txt").c_str(), "r"); if (!f || fscanf(f, "%d %d %d %d %d", &E, &Tv, &Td, &r, &rn) < 4) return 3; fclose(f); }
    auto bx = slurp(dir + "/x.bin"), bg = slurp(dir + "/wg.bin"), bu = slurp(dir + "/wu.bin"), bd = slurp(dir + "/wd.bin");
    auto bi = slurp(dir + "/idx.bin"), bw = slurp(dir + "/w.bin");
    int T = Tv + Td; const int* idx = (const int*)bi.data(); const float* rw = (const float*)bw.data();
    std::vector<TokRoute> V, D;
    for (int t = 0; t < T; ++t) { TokRoute tr; tr.row = t; for (int k = 0; k < TOPK; ++k) { tr.e[k] = idx[t * TOPK + k]; tr.w[k] = rw[t * TOPK + k]; } (t < Tv ? V : D).push_back(tr); }
    if (rn) renorm_draft(D, r);
    Dev D_; D_.x = to_dev<bf16>(bx); D_.wg = to_dev<bf16>(bg); D_.wu = to_dev<bf16>(bu); D_.wd = to_dev<bf16>(bd);
    CK(cudaMalloc(&D_.out, (size_t)T * H * 4));
    std::vector<TokRoute> all = V; all.insert(all.end(), D.begin(), D.end());
    // fused: union; verify uses 8 experts, draft uses top-r. Build one list with per-token limit.
    auto build_mixed = [&]() {
        std::vector<TokRoute> m = V; for (auto t : D) { for (int k = r; k < TOPK; ++k) t.w[k] = 0; m.push_back(t); }
        // zero-weight slots must not create work: build by hand
        std::vector<std::vector<std::pair<int, float>>> per(E);
        for (auto& t : V) for (int k = 0; k < TOPK; ++k) per[t.e[k]].push_back({t.row, t.w[k]});
        for (auto& t : D) for (int k = 0; k < r; ++k) per[t.e[k]].push_back({t.row, t.w[k]});
        WorkList wl;
        for (int e = 0; e < E; ++e) { if (per[e].empty()) continue; int off = wl.prow.size();
            for (auto& p : per[e]) { wl.prow.push_back(p.first); wl.pw.push_back(p.second); }
            for (int s = 0; s < NS; ++s) wl.tasks.push_back({e, s, off, (int)per[e].size(), KVM_NO_VAR, 0, KVM_NO_VAR, 0}); }
        return wl; };
    WorkList wf = build_mixed(), wv = build(V, TOPK, E), wdr = build(D, r, E);
    auto run = [&](std::vector<const WorkList*> ls, const char* name) {
        CK(cudaMemset(D_.out, 0, (size_t)T * H * 4));
        for (auto l : ls) { DevWL d = alloc_wl(std::max<int>(1, l->tasks.size()), std::max<int>(1, l->prow.size())); upload(d, *l); launch(D_, d, 0); CK(cudaDeviceSynchronize()); }
        std::vector<float> o((size_t)T * H); CK(cudaMemcpy(o.data(), D_.out, o.size() * 4, cudaMemcpyDeviceToHost));
        FILE* f = fopen((dir + "/" + name).c_str(), "wb"); fwrite(o.data(), 4, o.size(), f); fclose(f); };
    run({&wf}, "out_fused.bin"); run({&wv, &wdr}, "out_two.bin");
    printf("check: E=%d Tv=%d Td=%d r=%d fused_experts=%zu two_pass_experts=%d+%d\n", E, Tv, Td, r, wf.tasks.size() / NS, wv.nexp, wdr.nexp);
    return 0;
}

// ---------------- bench mode ----------------
struct Stat { double med, mn; };
int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage\n"); return 1; }
    std::string mode = argv[1];
    if (mode == "check") return check_mode(argv[2]);
    std::string outp = argv[2]; bool full = false; int seeds = 1, reps = 15, rn = 0;
    for (int i = 3; i < argc; ++i) { if (!strcmp(argv[i], "--full")) full = true; else if (!strcmp(argv[i], "--seeds")) seeds = atoi(argv[++i]); else if (!strcmp(argv[i], "--renorm")) rn = 1; else if (!strcmp(argv[i], "--reps")) reps = atoi(argv[++i]); }
    const int E = 128;
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int sm = prop.major * 10 + prop.minor;
    fprintf(stderr, "GPU %s sm_%d L2=%dMB\n", prop.name, sm, prop.l2CacheSize >> 20);
    Dev D; size_t wn = (size_t)E * I * H;
    CK(cudaMalloc(&D.wg, wn * 2)); CK(cudaMalloc(&D.wu, wn * 2)); CK(cudaMalloc(&D.wd, wn * 2));
    CK(cudaMalloc(&D.x, (size_t)MAXTOK * H * 2)); CK(cudaMalloc(&D.out, (size_t)MAXTOK * H * 4));
    fill_bf16<<<4096, 256>>>(D.wg, wn, 1, 0.03f); fill_bf16<<<4096, 256>>>(D.wu, wn, 2, 0.03f);
    fill_bf16<<<4096, 256>>>(D.wd, wn, 3, 0.03f); fill_bf16<<<64, 256>>>(D.x, (size_t)MAXTOK * H, 4, 1.0f);
    CK(cudaDeviceSynchronize());
    size_t flush_n = (size_t)256 << 20; void* flush; CK(cudaMalloc(&flush, flush_n));
    // peak streaming-read reference over the 1.2GB of weights
    double peak_gbs = 0;
    { unsigned* sink; CK(cudaMalloc(&sink, 4)); cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
      bf16* ws[3] = {D.wg, D.wu, D.wd};
      for (int rep = 0; rep < 6; ++rep) { CK(cudaMemset(flush, rep, flush_n)); cudaEventRecord(a);
        for (int k = 0; k < 3; ++k) read_kernel<<<prop.multiProcessorCount * 8, 256>>>((const uint4*)ws[k], wn * 2 / 16, sink);
        cudaEventRecord(b); CK(cudaEventSynchronize(b)); float ms; cudaEventElapsedTime(&ms, a, b); peak_gbs = std::max(peak_gbs, 3.0 * wn * 2 / (ms * 1e-3) / 1e9); } }
    fprintf(stderr, "streaming read peak ~ %.0f GB/s\n", peak_gbs);

    DevWL dv = alloc_wl(E * NS, E * MAXTOK), dd = alloc_wl(E * NS, E * MAXTOK), df = alloc_wl(E * NS, E * MAXTOK);
    cudaStream_t st; CK(cudaStreamCreate(&st));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    FILE* fo = fopen(outp.c_str(), "w");
    std::vector<int> TV, TD, RS;
    if (full) { for (int i = 1; i <= 16; ++i) { TV.push_back(i); TD.push_back(i); } for (int i = 1; i <= 8; ++i) RS.push_back(i); }
    else { TV = {1, 4, 8, 16}; TD = {1, 4, 8, 16}; RS = {1, 2, 4, 8}; }
    const char* names[] = {"verify_only", "two_pass", "fused", "two_pass_graph", "fused_graph"};
    int ncfg = 0;
    for (int routing = 0; routing < 3; ++routing)       // 0 = independent uniform, 1 = correlated (slot kept w.p. 0.75), 2 = draft routes identical to a verify token (overlap upper bound)
    for (int seed = 0; seed < seeds; ++seed)
    for (int tv : TV) for (int td : TD) for (int r : RS) {
        std::mt19937 g(1000003u * seed + 7919u * tv + 104729u * td + 13u * routing);
        std::vector<TokRoute> V, Dr;
        for (int i = 0; i < tv; ++i) V.push_back(make_route(g, i, E));
        for (int i = 0; i < td; ++i) Dr.push_back(routing == 0 ? make_route(g, tv + i, E) : routing == 1 ? make_corr(g, V[i % tv], tv + i, E, 0.25f) : make_corr(g, V[i % tv], tv + i, E, 0.f));
        if (rn) renorm_draft(Dr, r);
        // work lists: verify (8 experts), draft (top-r), fused (union)
        WorkList wv = build(V, TOPK, E), wdr = build(Dr, r, E);
        WorkList wf;
        { std::vector<std::vector<std::pair<int, float>>> per(E);
          for (auto& t : V) for (int k = 0; k < TOPK; ++k) per[t.e[k]].push_back({t.row, t.w[k]});
          for (auto& t : Dr) for (int k = 0; k < r; ++k) per[t.e[k]].push_back({t.row, t.w[k]});
          for (int e = 0; e < E; ++e) { if (per[e].empty()) continue; wf.nexp++; int off = wf.prow.size();
            for (auto& p : per[e]) { wf.prow.push_back(p.first); wf.pw.push_back(p.second); } wf.npairs += per[e].size();
            for (int s = 0; s < NS; ++s) wf.tasks.push_back({e, s, off, (int)per[e].size(), KVM_NO_VAR, 0, KVM_NO_VAR, 0}); } }
        upload(dv, wv); upload(dd, wdr); upload(df, wf);
        auto body = [&](int v, cudaStream_t s) {
            CK(cudaMemsetAsync(D.out, 0, (size_t)MAXTOK * H * 4, s));
            if (v == 0) launch(D, dv, s);
            else if (v == 1 || v == 3) { launch(D, dv, s); launch(D, dd, s); }
            else launch(D, df, s);
        };
        cudaGraphExec_t ge[5] = {0};
        for (int v = 3; v <= 4; ++v) {
            cudaGraph_t gr; CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal)); body(v, st); CK(cudaStreamEndCapture(st, &gr));
            CK(cudaGraphInstantiate(&ge[v], gr, 0)); cudaGraphDestroy(gr);
        }
        double med[5], mn[5];
        std::vector<float> outs[5];
        for (int v = 0; v < 5; ++v) {
            std::vector<float> ts;
            for (int rep = -2; rep < reps; ++rep) {
                CK(cudaMemsetAsync(flush, rep & 255, flush_n, st));                // evict L2
                CK(cudaEventRecord(e0, st));
                if (v >= 3) CK(cudaGraphLaunch(ge[v], st)); else body(v, st);
                CK(cudaEventRecord(e1, st)); CK(cudaStreamSynchronize(st));
                if (rep >= 0) { float ms; cudaEventElapsedTime(&ms, e0, e1); ts.push_back(ms * 1000.f); }
            }
            med[v] = median(ts); mn[v] = *std::min_element(ts.begin(), ts.end());
            if (v == 1 || v == 2) { outs[v].resize((size_t)(tv + td) * H); CK(cudaMemcpy(outs[v].data(), D.out, outs[v].size() * 4, cudaMemcpyDeviceToHost)); }
        }
        double maxdiff = 0, maxref = 0;
        for (size_t i = 0; i < outs[1].size(); ++i) { maxdiff = std::max<double>(maxdiff, fabs(outs[1][i] - outs[2][i])); maxref = std::max<double>(maxref, fabs(outs[1][i])); }
        for (int v = 3; v <= 4; ++v) cudaGraphExecDestroy(ge[v]);
        int inv = 0, tot = 0; { std::vector<char> inV(E, 0); for (auto& t : V) for (int k = 0; k < TOPK; ++k) inV[t.e[k]] = 1;
          for (auto& t : Dr) for (int k = 0; k < r; ++k) { tot++; inv += inV[t.e[k]]; } }
        int maxtok = 0; { std::vector<int> c(E, 0); for (auto& t : V) for (int k = 0; k < TOPK; ++k) maxtok = std::max(maxtok, ++c[t.e[k]]);
          for (auto& t : Dr) for (int k = 0; k < r; ++k) maxtok = std::max(maxtok, ++c[t.e[k]]); }
        double n8f = bytes_tiled(V, TOPK, Dr, r, E, 8), n8t = bytes_tiled(V, TOPK, {}, 0, E, 8) + bytes_tiled({}, 0, Dr, r, E, 8);
        double n16f = bytes_tiled(V, TOPK, Dr, r, E, 16), n16t = bytes_tiled(V, TOPK, {}, 0, E, 16) + bytes_tiled({}, 0, Dr, r, E, 16);
        int nv = wv.nexp, nd = wdr.nexp, nu = wf.nexp;
        double bytes[5] = {(double)nv * EXPERT_BYTES, (double)(nv + nd) * EXPERT_BYTES, (double)nu * EXPERT_BYTES, (double)(nv + nd) * EXPERT_BYTES, (double)nu * EXPERT_BYTES};
        fprintf(fo, "{\"gpu\":\"%s\",\"sm\":%d,\"dtype\":\"bf16\",\"routing\":\"%s\",\"seed\":%d,\"r\":%d,\"Tv\":%d,\"Td\":%d,"
                    "\"experts_verify\":%d,\"experts_draft\":%d,\"experts_union\":%d,\"pairs_verify\":%d,\"pairs_draft\":%d,\"peak_read_GBs\":%.1f,"
                    "\"renorm\":%d,\"draft_expert_overlap\":%.4f,\"max_tok_per_expert\":%d,\"wb_fused_n8\":%.0f,\"wb_two_n8\":%.0f,\"wb_fused_n16\":%.0f,\"wb_two_n16\":%.0f,\"fused_vs_two_maxabsdiff\":%.3e,\"out_maxabs\":%.3e,\"v\":{",
                prop.name, sm, routing == 0 ? "uniform" : routing == 1 ? "corr75" : "identical", seed, r, tv, td, nv, nd, nu, wv.npairs, wdr.npairs, peak_gbs, rn, tot ? (double)inv / tot : 0.0, maxtok, n8f, n8t, n16f, n16t, maxdiff, maxref);
        for (int v = 0; v < 5; ++v)
            fprintf(fo, "%s\"%s\":{\"us_med\":%.2f,\"us_min\":%.2f,\"weight_bytes\":%.0f,\"GBs\":%.1f}", v ? "," : "", names[v], med[v], mn[v], bytes[v], bytes[v] / (med[v] * 1e-6) / 1e9);
        fprintf(fo, "}}\n");
        if (++ncfg % 100 == 0) { fflush(fo); fprintf(stderr, "%d configs\n", ncfg); }
    }
    fclose(fo);
    fprintf(stderr, "done %d configs\n", ncfg);
    return 0;
}
