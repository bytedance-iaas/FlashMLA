// Copyright (c) 2025, FlashMLA.
// SM90 FP8 Native Sparse Prefill Attention - phase1_new.cuh
// Design: Native fp8 GMMA path (straightforward baseline)
//   QK GEMM: fp8 SS (E4M3 x E4M3 -> F32, k=32)
//   PV GEMM: fp8 RS (E4M3 x E4M3 -> F32, V physically transposed in smem)
//   Producer: loads fp8 KV via cp.async.cg, then transposes V
//   Q: consumer WG0 loads fp8 Q from gmem to fp8 smem
//
// Implementation notes:
//   - Per-tile GMMA commit+wait for strict accumulator ordering
//   - System-level memory fences after async operations for correctness
//   - Sequential producer: load K, then transpose V
//   - No pipeline overlap between stages

#pragma once

#include "config.h"
#include "flashmla_utils.h"
#include "../../helpers.h"
#include <cuda_fp8.h>

using namespace cute;

#include "../../../extension/sm90/dense_fp8/fp8_transpose_v.h"
#include "../../../extension/sm90/dense_fp8/utils.h"

namespace sm90 {
namespace fwd {

template <typename Kernel, typename TMAParamsT>
__global__ void sparse_attn_fwd_q8_new_kernel(
    __grid_constant__ const SparseAttnFwdQ8SM90NewParams params,
    __grid_constant__ const TMAParamsT tma_params);

template <int D_QK, bool HAVE_TOPK_LENGTH>
struct KernelTemplateQ8New {

    static constexpr int D_Q = D_QK;
    static constexpr int D_K = D_QK;
    static constexpr int D_V = 512;

    static constexpr int B_H    = 64;
    static constexpr int B_TOPK = 64;
    static constexpr int NUM_THREADS = 128 * 3;
    static constexpr float MAX_INIT_VAL = -1e30f;

    using fp8_t = cutlass::float_e4m3_t;

    // Barrier IDs (SM90: max 8 user NamedBarrier IDs)
    enum NamedBarriers : uint32_t {
        wg0_bunch_0_ready = 0,   // sM + sS0 ready
        wg1_bunch_0_ready = 1,   // sM + sS1 ready
        vt0_for_wg0 = 2,        // V[0] all transposed (prod+WG0)
        vt0_for_wg1 = 3,        // V[0] all transposed (prod+WG1)
        sL_ready = 4,            // post-loop only
        warpgroup0_sync = 5,     // post-loop only
        warpgroup1_sync = 6,     // post-loop only
        epilogue_sync = 7,       // pre-loop only (q_load_done)
    };
    static constexpr uint32_t q_load_done  = epilogue_sync;
    // Temporal reuse: IDs 5,6 double as vt1 signals (in-loop) and wg sync (post-loop)
    static constexpr uint32_t vt1_for_wg0  = warpgroup0_sync;
    static constexpr uint32_t vt1_for_wg1  = warpgroup1_sync;

    // ========================================================================
    // FP8 Smem Layouts
    // ========================================================================
    template<int NUM_TILES>
    using SmemLayoutQTiles_FP8 = decltype(coalesce(tile_to_shape(
        GMMA::Layout_K_SW64_Atom<fp8_t>{},
        Shape<Int<B_H>, Int<64*NUM_TILES>>{},
        Step<_1, _2>{}
    ), Shape<_1, _1>{}));

    template<int NUM_TILES>
    using SmemLayoutKTiles_FP8 = decltype(coalesce(tile_to_shape(
        GMMA::Layout_K_SW64_Atom<fp8_t>{},
        Shape<Int<B_TOPK>, Int<64*NUM_TILES>>{},
        Step<_1, _2>{}
    ), Shape<_1, _1>{}));

    template<int NUM_TILES>
    using SmemLayoutVtTiles_FP8 = decltype(coalesce(tile_to_shape(
        GMMA::Layout_K_SW64_Atom<fp8_t>{},
        Shape<Int<64*NUM_TILES>, Int<B_TOPK>>{},
        Step<_1, _2>{}
    ), Shape<_1, _1>{}));

    using SmemLayoutS_FP8 = decltype(coalesce(tile_to_shape(
        GMMA::Layout_K_SW64_Atom<fp8_t>{},
        Shape<Int<B_H>, Int<B_TOPK>>{}
    ), Shape<_1, _1>{}));

    template<int NUM_TILES>
    using SmemLayoutOTiles = decltype(coalesce(tile_to_shape(
        GMMA::Layout_K_SW128_Atom<bf16>{},
        Shape<Int<B_H>, Int<64*NUM_TILES>>{},
        Step<_1, _2>{}
    ), Shape<_1, _1>{}));

    using SmemLayoutQ = SmemLayoutQTiles_FP8<D_Q/64>;
    using SmemLayoutK = SmemLayoutKTiles_FP8<D_Q/64>;
    using SmemLayoutVt = SmemLayoutVtTiles_FP8<D_V/64>;
    using SmemLayoutHalfVt = SmemLayoutVtTiles_FP8<D_V/64/2>;
    using SmemLayoutS = SmemLayoutS_FP8;
    using SmemLayoutO = SmemLayoutOTiles<D_V/64>;

    using SmemTransposeV = SmemTransposeFp8_64x64<B_TOPK, D_V, SmemLayoutKTiles_FP8<D_V/64>>;

    // ========================================================================
    // FP8 GMMA atoms -- native E4M3, k=32
    // ========================================================================
    using TiledMMA_QK = decltype(make_tiled_mma(
        GMMA::MMA_64x64x32_F32E4M3E4M3_SS_TN<>{},
        Layout<Shape<_1, _1, _1>>{}
    ));

    using TiledMMA_PV_LocalP = decltype(make_tiled_mma(
        GMMA::MMA_64x256x32_F32E4M3E4M3_RS_TN<>{},
        Layout<Shape<_1, _1, _1>>{}
    ));

    using TiledMMA_PV_RemoteP = decltype(make_tiled_mma(
        GMMA::MMA_64x256x32_F32E4M3E4M3_SS_TN<>{},
        Layout<Shape<_1, _1, _1>>{}
    ));

    // ========================================================================
    // Shared Memory Plan
    // ========================================================================
    struct SharedMemoryPlan {
        union {
            array_aligned<fp8_t, cosize_v<SmemLayoutQ>> q;
            array_aligned<bf16, cosize_v<SmemLayoutO>> o;
        } q_o;
        array_aligned<fp8_t, cosize_v<SmemLayoutK>> k[2];
        array_aligned<fp8_t, cosize_v<SmemLayoutVt>> vt[2];
        // Padded sS stride (36B, avoids bank conflicts)
        array_aligned<fp8_t, 128 * 36> s[2];

        bool is_kv_valid[2][B_TOPK];
        float2 sM[32];
        float2 sL[64];
        float final_max_logits[64], final_lse[64];
        transac_bar_t bar_q, bar_k0_ready[2], bar_k1_ready[2], bar_is_kv_valid_ready;
        transac_bar_t bar_k0_free, bar_k1_free;
        transac_bar_t bar_vt_free[2];
    };

    struct TmaParams_t {
        CUtensorMap tensor_map_O;
    };

    // ========================================================================
    // devfunc -- straightforward baseline, no pipeline overlap
    // ========================================================================
    template<typename TMAParamType>
    static __device__ __forceinline__ void
    devfunc(const SparseAttnFwdQ8SM90NewParams& params, const TMAParamType& tma_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
        const int q_h_idx = blockIdx.x % (params.h_q / B_H);
        const int s_q_idx = blockIdx.x / (params.h_q / B_H);
        const int warpgroup_idx = cutlass::canonical_warp_group_idx();
        const int warp_idx = cutlass::canonical_warp_idx_sync();
        const int idx_in_warpgroup = threadIdx.x % 128;

        extern __shared__ char wksp_buf[];
        SharedMemoryPlan& plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);

        const float q_scale  = params.q_scale_ptr ? __ldg(params.q_scale_ptr) : params.q_scale;
        const float kv_scale = params.kv_scale_ptr ? __ldg(params.kv_scale_ptr) : params.kv_scale;
        const float qk_combined_scale_div_log2 = q_scale * kv_scale * params.sm_scale_div_log2;

        if (warp_idx == 0 && elect_one_sync()) {
            cute::prefetch_tma_descriptor(&tma_params.tensor_map_O);
            plan.bar_q.init(1);
            plan.bar_k0_free.init(128);
            plan.bar_k1_free.init(128);
            CUTE_UNROLL
            for (int i = 0; i < 2; ++i) {
                plan.bar_k0_ready[i].init(128);
                plan.bar_k1_ready[i].init(128);
            }
            plan.bar_is_kv_valid_ready.init(16);
            CUTE_UNROLL
            for (int i = 0; i < 2; ++i) {
                plan.bar_vt_free[i].init(256);
            }
            fence_barrier_init();
        }

        __syncthreads();
        const int topk_length = HAVE_TOPK_LENGTH ? __ldg(params.topk_length + s_q_idx) : params.topk;
        const int num_topk_blocks = HAVE_TOPK_LENGTH
            ? ku::ceil_div(topk_length, (int)B_TOPK)
            : (int)((unsigned int)params.topk / (unsigned int)B_TOPK);

        // ================================================================
        // Consumer WG0/WG1
        // ================================================================
        if (warpgroup_idx == 0 || warpgroup_idx == 1) {
            cutlass::arch::warpgroup_reg_alloc<216>();

            // Load Q (WG0 only)
            if (warpgroup_idx == 0) {
                const fp8_t* gQ = reinterpret_cast<const fp8_t*>(params.q)
                    + s_q_idx * (int64_t)params.stride_q_s_q
                    + q_h_idx * B_H * (int64_t)params.stride_q_h_q;

                constexpr int Q_GROUP_SIZE = 4;
                constexpr int Q_NUM_GROUPS = 128 / Q_GROUP_SIZE;
                constexpr int Q_ROWS_PER_GROUP = B_H / Q_NUM_GROUPS;
                int q_ig = idx_in_warpgroup % Q_GROUP_SIZE;
                int q_gg = idx_in_warpgroup / Q_GROUP_SIZE;
                constexpr int NUM_Q_TILES = D_Q / 64;
                int64_t q_cache_policy = createpolicy_evict_first();
                auto sQ_full = make_tensor(make_smem_ptr(plan.q_o.q.data()), SmemLayoutQ{});
                CUTE_UNROLL
                for (int lr = 0; lr < Q_ROWS_PER_GROUP; ++lr) {
                    int row = q_gg + lr * Q_NUM_GROUPS;
                    CUTE_UNROLL
                    for (int ti = 0; ti < NUM_Q_TILES; ++ti) {
                        int col = ti * 64 + q_ig * 16;
                        bool q_pred = (col + 16) <= D_Q;
                        cp_async_cacheglobal_l2_prefetch_256B(
                            gQ + row * (int64_t)params.stride_q_h_q + col,
                            &sQ_full(row, col),
                            q_pred,
                            q_cache_policy
                        );
                    }
                }
                asm volatile("cp.async.commit_group;\n" ::);
                asm volatile("cp.async.wait_group 0;\n" ::);
            }
            fence_view_async_shared();
            NamedBarrier::arrive_and_wait(256, q_load_done);

            // Register fragments
            float rM[2] = {MAX_INIT_VAL, MAX_INIT_VAL};
            float rL[2] = {0.0f, 0.0f};
            Tensor rO = partition_fragment_C(TiledMMA_PV_LocalP{}, Shape<Int<B_H>, Int<D_V/2>>{});
            Tensor rP = partition_fragment_C(TiledMMA_QK{}, Shape<Int<B_H>, Int<B_TOPK>>{});
            cute::fill(rO, 0.0f);

            using rP_fp8_layout_t = decltype(flash::convert_layout_acc_Aregs<TiledMMA_PV_LocalP>(
                partition_fragment_C(TiledMMA_QK{}, Shape<Int<B_H>, Int<B_TOPK>>{}).layout()));
            Tensor rP_fp8_local = make_tensor<fp8_t>(rP_fp8_layout_t{});

            bool cur_bar_wait_phase = 0;
            struct Warpgroup0 {};
            struct Warpgroup1 {};

            static constexpr int NUM_QK_TILES = D_Q / 64;

            auto qkt_gemm_one_tile = [&](auto wg_tag, int tile_idx, bool clear_accum) {
                constexpr bool IS_WG1 = std::is_same_v<decltype(wg_tag), Warpgroup1>;
                TiledMMA_QK tiled_mma_QK;
                Tensor sQ_tile = make_tensor(
                    make_smem_ptr(plan.q_o.q.data() + tile_idx * B_H * 64),
                    SmemLayoutQTiles_FP8<1>{}
                );
                Tensor sK_tile = make_tensor(
                    make_smem_ptr(plan.k[(int)IS_WG1].data() + tile_idx * B_TOPK * 64),
                    SmemLayoutKTiles_FP8<1>{}
                );
                gemm_ss(clear_accum, tiled_mma_QK, sQ_tile, sK_tile, rP, idx_in_warpgroup);
            };

            auto mask_rP = [&](auto wg_tag) {
                constexpr bool IS_WG1 = std::is_same_v<decltype(wg_tag), Warpgroup1>;
                plan.bar_is_kv_valid_ready.wait(cur_bar_wait_phase);
                CUTE_UNROLL
                for (int row_idx = 0; row_idx < 2; ++row_idx) {
                    CUTE_UNROLL
                    for (int i = row_idx * 2; i < size(rP); i += 4) {
                        int col = 8 * (i / 4) + (idx_in_warpgroup % 4) * 2;
                        if (!plan.is_kv_valid[IS_WG1][col])   rP(i) = -INFINITY;
                        if (!plan.is_kv_valid[IS_WG1][col+1]) rP(i+1) = -INFINITY;
                    }
                }
            };

            auto online_softmax_and_rescale_o = [&](auto wg_tag) {
                constexpr bool IS_WG1 = std::is_same_v<decltype(wg_tag), Warpgroup1>;
                const float scale = qk_combined_scale_div_log2;
                float r_sM[2];
                if constexpr (IS_WG1) {
                    *(float2*)r_sM = plan.sM[idx_in_warpgroup / 4];
                }
                float new_maxs[2];
                CUTE_UNROLL
                for (int row_idx = 0; row_idx < 2; ++row_idx) {
                    float cur_max = -INFINITY;
                    CUTE_UNROLL
                    for (int i = row_idx * 2; i < size(rP); i += 4) {
                        cur_max = max(cur_max, max(rP(i), rP(i+1)));
                    }
                    cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
                    cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));
                    cur_max *= scale;
                    new_maxs[row_idx] = max(IS_WG1 ? r_sM[row_idx] : rM[row_idx], cur_max);
                    float scale_for_o = exp2f(rM[row_idx] - new_maxs[row_idx]);
                    CUTE_UNROLL
                    for (int i = row_idx * 2; i < size(rO); i += 4) {
                        rO(i)   *= scale_for_o;
                        rO(i+1) *= scale_for_o;
                    }
                    float cur_sum = 0;
                    CUTE_UNROLL
                    for (int i = row_idx * 2; i < size(rP); i += 4) {
                        float p0 = exp2f(rP(i)   * scale - new_maxs[row_idx]);
                        float p1 = exp2f(rP(i+1) * scale - new_maxs[row_idx]);
                        rP(i)   = p0;
                        rP(i+1) = p1;
                        cur_sum += p0 + p1;
                    }
                    rL[row_idx] = rL[row_idx] * scale_for_o + cur_sum;
                }
                __syncwarp();
                if (idx_in_warpgroup % 4 == 0) {
                    plan.sM[idx_in_warpgroup / 4] = *(float2*)new_maxs;
                }
                rM[0] = new_maxs[0];
                rM[1] = new_maxs[1];

                flash::permute_Cregs_fp8(rP);
                Tensor rP_acc = make_tensor(rP.data(),
                    flash::convert_layout_acc_Aregs<TiledMMA_PV_LocalP>(rP.layout()));
                flash::convert_type_out(rP_acc, rP_fp8_local);
            };

            auto reduce_L = [&]() {
                rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 1);
                rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 2);
                rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 1);
                rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 2);
                if (idx_in_warpgroup % 4 == 0)
                    plan.sL[threadIdx.x / 4] = *(float2*)(rL);
                __threadfence_block();
                NamedBarrier::arrive_and_wait(256, NamedBarriers::sL_ready);
                float2 peer_L = plan.sL[(threadIdx.x / 4) ^ 32];
                rL[0] += peer_L.x;
                rL[1] += peer_L.y;
            };

            auto store_O = [&]() {
                float scale_factors[2];
                CUTE_UNROLL
                for (int i = 0; i < 2; ++i) {
                    float attn_sink = params.attn_sink == nullptr
                        ? -CUDART_INF_F
                        : params.attn_sink[q_h_idx * B_H + get_AorC_row_idx(i, idx_in_warpgroup)] * CUDART_L2E_F;
                    scale_factors[i] = kv_scale / (rL[i] + exp2f(attn_sink - rM[i]));
                    if (rL[i] == 0.0f)
                        scale_factors[i] = 0.0f;
                }

                Tensor sO_tile = make_tensor(
                    make_smem_ptr(plan.q_o.o.data() + warpgroup_idx * B_H * (D_V / 2)),
                    SmemLayoutOTiles<4>{}
                );
                bf16* stsm_addrs[4];
                int stsm_row = (idx_in_warpgroup / 32) * 16 + (idx_in_warpgroup % 16);
                CUTE_UNROLL
                for (int i = 0; i < 64 / 16; ++i) {
                    stsm_addrs[i] = &sO_tile(stsm_row, (idx_in_warpgroup % 32 / 16 * 8) + 16 * i);
                }
                bool s2g_pred = warp_idx % 4 == 0 && elect_one_sync();

                warpgroup_wait<0>();
                CUTE_UNROLL
                for (int tile_idx = 0; tile_idx < (D_V / 2) / 64; tile_idx += 1) {
                    constexpr int NUM_ELEMS_EACH_TILE = B_H * 64 / 128;
                    bf16 cur_rOb[NUM_ELEMS_EACH_TILE];
                    CUTE_UNROLL
                    for (int i = 0; i < NUM_ELEMS_EACH_TILE; ++i) {
                        cur_rOb[i] = (bf16)(rO(tile_idx * NUM_ELEMS_EACH_TILE + i) * scale_factors[i % 4 >= 2]);
                    }
                    CUTE_UNROLL
                    for (int i = 0; i < 64 / 16; ++i) {
                        SM90_U32x4_STSM_N::copy(
                            *reinterpret_cast<uint32_t*>(cur_rOb + i * 8 + 0),
                            *reinterpret_cast<uint32_t*>(cur_rOb + i * 8 + 2),
                            *reinterpret_cast<uint32_t*>(cur_rOb + i * 8 + 4),
                            *reinterpret_cast<uint32_t*>(cur_rOb + i * 8 + 6),
                            *reinterpret_cast<uint128_t*>(stsm_addrs[i] + tile_idx * (B_H * 64))
                        );
                    }
                    asm volatile("" ::: "memory");
                    NamedBarrier::arrive_and_wait(128,
                        warpgroup_idx ? NamedBarriers::warpgroup1_sync : NamedBarriers::warpgroup0_sync);
                    if (s2g_pred) {
                        int g_tile_idx = warpgroup_idx * 4 + tile_idx;
                        SM90_TMA_STORE_3D::copy(
                            &tma_params.tensor_map_O,
                            plan.q_o.o.data() + g_tile_idx * (B_H * 64),
                            g_tile_idx * 64,
                            q_h_idx * B_H,
                            s_q_idx
                        );
                    }
                }
                cute::tma_store_arrive();
            };

            // P save/load: no padding (stride=32, has bank conflicts)
            constexpr int kP_per_thread = 32;
            constexpr int kP_stride = 36;

            auto save_rP_fp8_to_sS = [&](fp8_t* sS_data) {
                uint32_t* dst = reinterpret_cast<uint32_t*>(sS_data + idx_in_warpgroup * kP_stride);
                uint32_t* src = reinterpret_cast<uint32_t*>(&rP_fp8_local(0));
                CUTE_UNROLL
                for (int i = 0; i < kP_per_thread / 4; i++) {
                    dst[i] = src[i];
                }
            };

            auto load_sS_to_rP = [&](fp8_t* sS_data) {
                uint32_t* src = reinterpret_cast<uint32_t*>(sS_data + idx_in_warpgroup * kP_stride);
                uint32_t* dst = reinterpret_cast<uint32_t*>(&rP_fp8_local(0));
                CUTE_UNROLL
                for (int i = 0; i < kP_per_thread / 4; i++) {
                    dst[i] = src[i];
                }
            };

            auto rescale_rO = [&](float scales[2]) {
                CUTE_UNROLL
                for (int row = 0; row < 2; ++row) {
                    CUTE_UNROLL
                    for (int i = row * 2; i < size(rO); i += 4) {
                        rO(i)   *= scales[row];
                        rO(i+1) *= scales[row];
                    }
                    rL[row] *= scales[row];
                }
            };

            // ============================================================
            // WG0 consumer loop
            // ============================================================
            if (warpgroup_idx == 0) {

                CUTE_NO_UNROLL
                for (int block_idx = 0; block_idx < num_topk_blocks; block_idx += 2) {
                    // --- QK GEMM: per-tile commit for strict ordering ---
                    plan.bar_k0_ready[0].wait(cur_bar_wait_phase);
                    fence_view_async_shared();
                    __threadfence_system();
                    qkt_gemm_one_tile(Warpgroup0{}, 0, true);
                    warpgroup_commit_batch();
                    warpgroup_wait<0>();
                    __threadfence_system();
                    CUTE_UNROLL
                    for (int tile = 1; tile < NUM_QK_TILES; ++tile) {
                        qkt_gemm_one_tile(Warpgroup0{}, tile, false);
                        warpgroup_commit_batch();
                        warpgroup_wait<0>();
                        __threadfence_system();
                    }
                    fence_view_async_shared();
                    plan.bar_k0_free.arrive();

                    // --- Softmax + P save ---
                    mask_rP(Warpgroup0{});
                    __syncwarp();
                    __threadfence_system();
                    online_softmax_and_rescale_o(Warpgroup0{});
                    __syncwarp();
                    __threadfence_system();
                    save_rP_fp8_to_sS(plan.s[0].data());
                    __threadfence_block();
                    NamedBarrier::arrive_and_wait(256, NamedBarriers::wg0_bunch_0_ready);

                    // --- PV-local: wait for V[0], RS GMMA left half ---
                    NamedBarrier::arrive_and_wait(256, vt0_for_wg0);
                    fence_view_async_shared();
                    __threadfence_system();
                    Tensor sVt0l = make_tensor(make_smem_ptr(plan.vt[0].data()), SmemLayoutHalfVt{});
                    gemm_rs(false, TiledMMA_PV_LocalP{}, rP_fp8_local, sVt0l, rO, idx_in_warpgroup);
                    warpgroup_commit_batch();
                    warpgroup_wait<0>();
                    __threadfence_system();
                    plan.bar_vt_free[0].arrive();

                    // --- PV-remote: wait for WG1's P, apply rescale, wait for V[1] ---
                    NamedBarrier::arrive_and_wait(256, NamedBarriers::wg1_bunch_0_ready);
                    fence_view_async_shared();
                    __threadfence_system();
                    float new_rM[2], scale_factors_arr[2];
                    *(float2*)new_rM = plan.sM[idx_in_warpgroup / 4];
                    CUTE_UNROLL
                    for (int i = 0; i < 2; ++i) {
                        scale_factors_arr[i] = exp2f(rM[i] - new_rM[i]);
                        rM[i] = new_rM[i];
                    }
                    rescale_rO(scale_factors_arr);
                    load_sS_to_rP(plan.s[1].data());

                    NamedBarrier::arrive_and_wait(256, vt1_for_wg0);
                    fence_view_async_shared();
                    __threadfence_system();
                    Tensor sVt1l = make_tensor(make_smem_ptr(plan.vt[1].data()), SmemLayoutHalfVt{});
                    gemm_rs(false, TiledMMA_PV_LocalP{}, rP_fp8_local, sVt1l, rO, idx_in_warpgroup);
                    warpgroup_commit_batch();
                    warpgroup_wait<0>();
                    __threadfence_system();
                    plan.bar_vt_free[1].arrive();

                    cur_bar_wait_phase ^= 1;
                }

                // Fix column permutation from fp8 V transpose
                {
                    int t1_bit0 = (threadIdx.x >> 2) & 1;
                    #pragma unroll
                    for (int g = 0; g < 32; g++) {
                        float a = rO(4*g + 0);
                        float b = rO(4*g + 1);
                        float c = rO(4*g + 2);
                        float d = rO(4*g + 3);
                        float send0 = t1_bit0 ? a : c;
                        float send1 = t1_bit0 ? b : d;
                        float recv0 = __shfl_xor_sync(0xFFFFFFFF, send0, 4);
                        float recv1 = __shfl_xor_sync(0xFFFFFFFF, send1, 4);
                        if (t1_bit0 == 0) {
                            rO(4*g + 2) = recv0;
                            rO(4*g + 3) = recv1;
                        } else {
                            rO(4*g + 0) = recv0;
                            rO(4*g + 1) = recv1;
                        }
                    }
                }

                reduce_L();
                store_O();

            } else {
                // ============================================================
                // WG1 consumer loop
                // ============================================================

                CUTE_NO_UNROLL
                for (int block_idx = 0; block_idx < num_topk_blocks; block_idx += 2) {
                    // --- QK GEMM: per-tile commit for strict ordering ---
                    plan.bar_k1_ready[0].wait(cur_bar_wait_phase);
                    fence_view_async_shared();
                    __threadfence_system();
                    qkt_gemm_one_tile(Warpgroup1{}, 0, true);
                    warpgroup_commit_batch();
                    warpgroup_wait<0>();
                    __threadfence_system();
                    CUTE_UNROLL
                    for (int tile = 1; tile < NUM_QK_TILES; ++tile) {
                        qkt_gemm_one_tile(Warpgroup1{}, tile, false);
                        warpgroup_commit_batch();
                        warpgroup_wait<0>();
                        __threadfence_system();
                    }
                    fence_view_async_shared();
                    plan.bar_k1_free.arrive();

                    // --- Softmax + P save ---
                    mask_rP(Warpgroup1{});
                    __syncwarp();
                    __threadfence_system();
                    NamedBarrier::arrive_and_wait(256, NamedBarriers::wg0_bunch_0_ready);
                    online_softmax_and_rescale_o(Warpgroup1{});
                    __syncwarp();
                    __threadfence_system();
                    save_rP_fp8_to_sS(plan.s[1].data());
                    __threadfence_block();
                    NamedBarrier::arrive_and_wait(256, NamedBarriers::wg1_bunch_0_ready);

                    // --- PV-local: wait for V[1], RS GMMA right half ---
                    NamedBarrier::arrive_and_wait(256, vt1_for_wg1);
                    fence_view_async_shared();
                    __threadfence_system();
                    Tensor sVt1r = make_tensor(
                        make_smem_ptr(plan.vt[1].data() + 256 * B_TOPK),
                        SmemLayoutHalfVt{}
                    );
                    gemm_rs(false, TiledMMA_PV_LocalP{}, rP_fp8_local, sVt1r, rO, idx_in_warpgroup);
                    warpgroup_commit_batch();
                    warpgroup_wait<0>();
                    __threadfence_system();
                    plan.bar_vt_free[1].arrive();

                    // --- PV-remote: load WG0's P, wait for V[0] ---
                    load_sS_to_rP(plan.s[0].data());
                    fence_view_async_shared();
                    __threadfence_system();
                    NamedBarrier::arrive_and_wait(256, vt0_for_wg1);
                    fence_view_async_shared();
                    __threadfence_system();
                    Tensor sVt0r = make_tensor(
                        make_smem_ptr(plan.vt[0].data() + 256 * B_TOPK),
                        SmemLayoutHalfVt{}
                    );
                    gemm_rs(false, TiledMMA_PV_LocalP{}, rP_fp8_local, sVt0r, rO, idx_in_warpgroup);
                    warpgroup_commit_batch();
                    warpgroup_wait<0>();
                    __threadfence_system();
                    plan.bar_vt_free[0].arrive();

                    cur_bar_wait_phase ^= 1;
                }

                // Fix column permutation (WG1)
                {
                    int t1_bit0 = (threadIdx.x >> 2) & 1;
                    #pragma unroll
                    for (int g = 0; g < 32; g++) {
                        float a = rO(4*g + 0);
                        float b = rO(4*g + 1);
                        float c = rO(4*g + 2);
                        float d = rO(4*g + 3);
                        float send0 = t1_bit0 ? a : c;
                        float send1 = t1_bit0 ? b : d;
                        float recv0 = __shfl_xor_sync(0xFFFFFFFF, send0, 4);
                        float recv1 = __shfl_xor_sync(0xFFFFFFFF, send1, 4);
                        if (t1_bit0 == 0) {
                            rO(4*g + 2) = recv0;
                            rO(4*g + 3) = recv1;
                        } else {
                            rO(4*g + 0) = recv0;
                            rO(4*g + 1) = recv1;
                        }
                    }
                }

                reduce_L();
                store_O();

                if (idx_in_warpgroup % 4 == 0) {
                    for (int row = 0; row < 2; ++row) {
                        int real_row = get_AorC_row_idx(row, idx_in_warpgroup);
                        bool is_no_valid_tokens = rL[row] == 0.0f;
                        plan.final_max_logits[real_row] = is_no_valid_tokens
                            ? -INFINITY : rM[row] * CUDART_LN2_F;
                        plan.final_lse[real_row] = is_no_valid_tokens
                            ? +INFINITY : logf(rL[row]) + rM[row] * CUDART_LN2_F;
                    }
                    asm volatile("" ::: "memory");
                }

                NamedBarrier::arrive_and_wait(128, NamedBarriers::warpgroup1_sync);
                if (idx_in_warpgroup == 0) {
                    int g_offset = s_q_idx * params.h_q + q_h_idx * B_H;
                    SM90_BULK_COPY_S2G::copy(plan.final_max_logits,
                                             params.max_logits + g_offset, B_H * sizeof(float));
                    SM90_BULK_COPY_S2G::copy(plan.final_lse,
                                             params.lse + g_offset, B_H * sizeof(float));
                    cute::tma_store_arrive();
                }
            }

        } else {
            // ================================================================
            // Producer WG2: sequential load K then transpose V
            // ================================================================
            cutlass::arch::warpgroup_reg_dealloc<72>();

            constexpr int GROUP_SIZE = 8, NUM_GROUPS = 128 / GROUP_SIZE;
            constexpr int NUM_ROWS_PER_GROUP = B_TOPK / NUM_GROUPS;
            int idx_in_group = idx_in_warpgroup % GROUP_SIZE;
            int group_idx = idx_in_warpgroup / GROUP_SIZE;
            int* gIndices = params.indices + s_q_idx * params.stride_indices_s_q;

            fp8_t* my_sK_base = &(make_tensor(make_smem_ptr(plan.k[0].data()), SmemLayoutKTiles_FP8<1>{})(group_idx, idx_in_group * 16));
            const fp8_t* my_gKV_base = reinterpret_cast<const fp8_t*>(params.kv) + idx_in_group * 16;

            int64_t token_indices[2][NUM_ROWS_PER_GROUP];
            bool is_token_valid[2][NUM_ROWS_PER_GROUP];

            auto load_token_indices = [&](int block_idx) {
                CUTE_UNROLL
                for (int buf_idx = 0; buf_idx < 2; ++buf_idx) {
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < NUM_ROWS_PER_GROUP; ++local_row) {
                        int offs = (block_idx + buf_idx) * B_TOPK + local_row * NUM_GROUPS + group_idx;
                        int t = __ldg(gIndices + offs);
                        token_indices[buf_idx][local_row] = t * (int64_t)params.stride_kv_s_kv;
                        bool is_cur_token_valid = t >= 0 && t < params.s_kv;
                        if constexpr (HAVE_TOPK_LENGTH) {
                            is_cur_token_valid &= offs < topk_length;
                        }
                        is_token_valid[buf_idx][local_row] = is_cur_token_valid;
                    }
                }
            };

            int64_t cache_policy = createpolicy_evict_first();

            auto copy_tiles = [&](int buf_idx, int smem_buf, int tile_start, int tile_end) {
                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_ROWS_PER_GROUP; ++local_row) {
                    int64_t token_index = token_indices[buf_idx][local_row];
                    CUTE_UNROLL
                    for (int tile_idx = tile_start; tile_idx < tile_end; ++tile_idx) {
                        cp_async_cacheglobal_l2_prefetch_256B(
                            my_gKV_base + token_index + tile_idx * 64,
                            my_sK_base + (smem_buf * cosize_v<SmemLayoutK> + tile_idx * (B_TOPK * 64) + local_row * NUM_GROUPS * 64),
                            is_token_valid[buf_idx][local_row],
                            cache_policy
                        );
                    }
                }
            };

            auto commit_to_mbar = [&](transac_bar_t& bar) {
                cutlass::arch::cpasync_barrier_arrive_noinc((uint64_t*)(&bar));
            };

            SmemTransposeV smem_transpose_v;
            using SmemLayoutTransposeV_t = typename SmemTransposeV::SmemLayoutTransposeV;
            using SmemLayoutTransposeVt_t = typename SmemTransposeV::SmemLayoutTransposeVt;

            // Individual tile transpose (no pair optimization)
            auto transpose_v_tile = [&](int smem_k_buf, int vt_buf, int tile_idx) {
                Tensor sV_src = as_position_independent_swizzle_tensor(make_tensor(
                    make_smem_ptr(plan.k[smem_k_buf].data()),
                    SmemLayoutTransposeV_t{}
                ));
                Tensor sVt_dst = as_position_independent_swizzle_tensor(make_tensor(
                    make_smem_ptr(plan.vt[vt_buf].data()),
                    SmemLayoutTransposeVt_t{}
                ));
                smem_transpose_v.transpose(
                    flatten(sV_src(_, 0, tile_idx)),
                    flatten(sVt_dst(_, 0, tile_idx))
                );
            };

            int cur_bar_wait_phase_prod = 1;

            CUTE_NO_UNROLL
            for (int block_idx = 0; block_idx < num_topk_blocks; block_idx += 2) {
                // Load indices at loop start (no prefetch)
                load_token_indices(block_idx);

                plan.bar_k0_free.wait(cur_bar_wait_phase_prod);
                plan.bar_k1_free.wait(cur_bar_wait_phase_prod);
                __threadfence_system();

                if (idx_in_group == 0) {
                    CUTE_UNROLL
                    for (int buf_idx = 0; buf_idx < 2; ++buf_idx)
                        CUTE_UNROLL
                        for (int local_row = 0; local_row < NUM_ROWS_PER_GROUP; ++local_row)
                            plan.is_kv_valid[buf_idx][local_row * NUM_GROUPS + group_idx]
                                = is_token_valid[buf_idx][local_row];
                    plan.bar_is_kv_valid_ready.arrive();
                }

                // Load ALL of K[0] in one shot
                copy_tiles(0, 0, 0, D_K / 64);
                commit_to_mbar(plan.bar_k0_ready[0]);
                asm volatile("cp.async.commit_group;\n" ::);

                // Load ALL of K[1] in one shot
                copy_tiles(1, 1, 0, D_K / 64);
                commit_to_mbar(plan.bar_k1_ready[0]);
                asm volatile("cp.async.commit_group;\n" ::);

                // Wait for all loads to complete
                asm volatile("cp.async.wait_group 0;\n" ::);
                fence_view_async_shared();
                __threadfence_system();
                NamedBarrier::arrive_and_wait(128, 8);

                // Transpose V[0]
                if (block_idx > 0) {
                    plan.bar_vt_free[0].wait(cur_bar_wait_phase_prod);
                }
                CUTE_UNROLL
                for (int j = 0; j < D_V / 64; ++j) {
                    transpose_v_tile(0, 0, j);
                }
                asm volatile("" ::: "memory");
                __threadfence_block();
                __threadfence_system();
                NamedBarrier::arrive(256, vt0_for_wg0);
                NamedBarrier::arrive(256, vt0_for_wg1);

                // Transpose V[1]
                if (block_idx > 0) {
                    plan.bar_vt_free[1].wait(cur_bar_wait_phase_prod);
                }
                CUTE_UNROLL
                for (int j = 0; j < D_V / 64; ++j) {
                    transpose_v_tile(1, 1, j);
                }
                asm volatile("" ::: "memory");
                __threadfence_block();
                __threadfence_system();
                NamedBarrier::arrive(256, vt1_for_wg0);
                NamedBarrier::arrive(256, vt1_for_wg1);

                cur_bar_wait_phase_prod ^= 1;
            }
        }

        cute::tma_store_wait<0>();
#else
        if (cute::thread0()) {
            CUTE_INVALID_CONTROL_PATH("This kernel only supports sm90");
        }
#endif
    }

    // ========================================================================
    // run() -- host-side launch
    // ========================================================================
    static void run(const SparseAttnFwdQ8SM90NewParams& params) {
        KU_ASSERT(params.h_kv == 1);
        KU_ASSERT(params.topk % (2 * B_TOPK) == 0);
        KU_ASSERT(params.topk > 0);
        KU_ASSERT(params.h_q % B_H == 0);

        CUtensorMap tensor_map_O;
        {
            uint64_t size[3] = {(uint64_t)D_V, (uint64_t)params.h_q, (uint64_t)params.s_q};
            uint64_t stride[2] = {D_V * sizeof(bf16), D_V * params.h_q * sizeof(bf16)};
            uint32_t box_size[3] = {64, B_H, 1};
            uint32_t elem_stride[3] = {1, 1, 1};
            CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
                &tensor_map_O,
                CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                3,
                params.out,
                size,
                stride,
                box_size,
                elem_stride,
                CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
                CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
                CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
            );
            KU_ASSERT(res == CUresult::CUDA_SUCCESS);
        }

        TmaParams_t tma_p = { tensor_map_O };

        auto kernel = &sparse_attn_fwd_q8_new_kernel<
            KernelTemplateQ8New<D_QK, HAVE_TOPK_LENGTH>, TmaParams_t>;

        constexpr size_t smem_size = sizeof(SharedMemoryPlan);
        KU_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

        cutlass::ClusterLaunchParams launch_params = {
            dim3((params.h_q / B_H) * params.s_q, 1, 1),
            dim3(NUM_THREADS, 1, 1),
            dim3(1, 1, 1),
            smem_size,
            params.stream
        };
        cutlass::launch_kernel_on_cluster(
            launch_params, (void*)kernel, params, tma_p
        );
        KU_CHECK_KERNEL_LAUNCH();
    }
};

// ============================================================================
// Global kernel entry point
// ============================================================================
template <typename Kernel, typename TMAParamsT>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, 1)
sparse_attn_fwd_q8_new_kernel(
    __grid_constant__ const SparseAttnFwdQ8SM90NewParams params,
    __grid_constant__ const TMAParamsT tma_params)
{
    Kernel::devfunc(params, tma_params);
}

// ============================================================================
// External dispatch function
// ============================================================================
template <int D_QK, bool HAVE_TOPK_LENGTH>
void run_fwd_phase1_q8_sm90_new_kernel(const SparseAttnFwdQ8SM90NewParams& params) {
    KernelTemplateQ8New<D_QK, HAVE_TOPK_LENGTH>::run(params);
}

}  // namespace fwd
}  // namespace sm90
