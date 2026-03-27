#pragma once

#include "config.h"

#include "flashmla_utils.h"
#include "../../helpers.h"

#include <cuda_fp8.h>

namespace sm90::fwd {

using namespace cute;

CUTE_DEVICE void st_global_cs_128(float f0, float f1, float f2, float f3, void *dst_ptr) {
    asm volatile("st.weak.global.cs.v4.f32 [%0], {%1, %2, %3, %4};\n"
                 :
                 : "l"(dst_ptr),
                   "f"(f0), "f"(f1), "f"(f2), "f"(f3)
                );
}

CUTE_DEVICE
float2 __shfl_xor_sync_float2(
    uint32_t mask, float2 value, int offset
) {
    float2 res;
    *reinterpret_cast<long long*>(&res) = __shfl_xor_sync(
        mask,
        *reinterpret_cast<long long*>(&value),
        offset
    );
    return res;
}

CUTE_DEVICE
void tma_bulk_reduce_add(void const* src_ptr, void* dst_ptr, int32_t store_bytes) {
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(src_ptr);
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;\n"
                     :
                     : "l"(dst_ptr), "r"(smem_int_ptr), "r"(store_bytes)
                     : "memory");
}

CUTE_DEVICE
float fp8_e4m3fn_to_float(uint8_t x) {
    const __half_raw hraw = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(x), __NV_E4M3);
    const __half h = *reinterpret_cast<const __half*>(&hraw);
    return __half2float(h);
}

CUTE_DEVICE
void fp8x4_e4m3fn_to_float(uint32_t packed, float& f0, float& f1, float& f2, float& f3) {
    f0 = fp8_e4m3fn_to_float(static_cast<uint8_t>(packed & 0xff));
    f1 = fp8_e4m3fn_to_float(static_cast<uint8_t>((packed >> 8) & 0xff));
    f2 = fp8_e4m3fn_to_float(static_cast<uint8_t>((packed >> 16) & 0xff));
    f3 = fp8_e4m3fn_to_float(static_cast<uint8_t>((packed >> 24) & 0xff));
}

static __global__ void sparse_attn_fwd_q8_direct_kernel(const SparseAttnFwdQ8Params params) {
    const int row = blockIdx.x;
    const int q_h_idx = row % params.h_q;
    const int s_q_idx = row / params.h_q;
    if (s_q_idx >= params.s_q) {
        return;
    }

    extern __shared__ float smem[];
    float* q_vec = smem;
    float* logits = q_vec + 576;
    float* weights = logits + params.topk;
    int* kv_base_stage = reinterpret_cast<int*>(weights + params.topk);

    __shared__ float s_max_logit;
    __shared__ float s_inv_denom;
    __shared__ float s_lse;
    __shared__ int s_has_valid;
    __shared__ int s_all_valid;
    __shared__ float s_reduce[8];

    const float q_scale = params.q_scale_ptr ? __ldg(params.q_scale_ptr) : params.q_scale;
    const float kv_scale = params.kv_scale_ptr ? __ldg(params.kv_scale_ptr) : params.kv_scale;
    const float q_prescale = q_scale * params.sm_scale * kv_scale;

    const int q_base = s_q_idx * params.stride_q_s_q + q_h_idx * params.stride_q_h_q;
    for (int d = threadIdx.x * 4; d < params.d_qk; d += blockDim.x * 4) {
        if (d + 3 < params.d_qk) {
            const uint32_t packed_q = __ldg(reinterpret_cast<const uint32_t*>(params.q + q_base + d));
            float q0, q1, q2, q3;
            fp8x4_e4m3fn_to_float(packed_q, q0, q1, q2, q3);
            q_vec[d + 0] = q0 * q_prescale;
            q_vec[d + 1] = q1 * q_prescale;
            q_vec[d + 2] = q2 * q_prescale;
            q_vec[d + 3] = q3 * q_prescale;
        } else {
            for (int d_tail = d; d_tail < params.d_qk; ++d_tail) {
                q_vec[d_tail] = fp8_e4m3fn_to_float(__ldg(params.q + q_base + d_tail)) * q_prescale;
            }
        }
    }
    __syncthreads();

    const int topk_length = params.topk_length ? __ldg(params.topk_length + s_q_idx) : params.topk;
    const int topk_valid_len = max(0, min(topk_length, params.topk));
    const int* idx_row = params.indices + s_q_idx * params.stride_indices_s_q;
    if (threadIdx.x == 0) {
        s_all_valid = (topk_valid_len == params.topk) ? 1 : 0;
    }
    __syncthreads();

    for (int t = threadIdx.x; t < topk_valid_len; t += blockDim.x) {
        const int token = __ldg(idx_row + t);
        const bool valid = (token >= 0) && (token < params.s_kv);
        if (!valid) {
            s_all_valid = 0;
        }
    }
    __syncthreads();

    for (int t = threadIdx.x; t < topk_valid_len; t += blockDim.x) {
        const int token = __ldg(idx_row + t);
        const bool valid = (token >= 0) && (token < params.s_kv);
        kv_base_stage[t] = valid ? (token * params.stride_kv_s_kv) : -1;
    }
    for (int t = topk_valid_len + threadIdx.x; t < params.topk; t += blockDim.x) {
        kv_base_stage[t] = -1;
    }
    __syncthreads();

    if (false && s_all_valid) {
        for (int t = threadIdx.x; t < params.topk; t += blockDim.x) {
            const int kv_base = kv_base_stage[t];
            const uint8_t* kv_row = params.kv + kv_base;

            float dot = 0.0f;
            int d = 0;
            for (; d + 3 < params.d_qk; d += 4) {
                const uint32_t packed_kv = __ldg(reinterpret_cast<const uint32_t*>(kv_row + d));
                float kv0, kv1, kv2, kv3;
                fp8x4_e4m3fn_to_float(packed_kv, kv0, kv1, kv2, kv3);
                dot = fmaf(q_vec[d + 0], kv0, dot);
                dot = fmaf(q_vec[d + 1], kv1, dot);
                dot = fmaf(q_vec[d + 2], kv2, dot);
                dot = fmaf(q_vec[d + 3], kv3, dot);
            }
            for (; d < params.d_qk; ++d) {
                const float kvd = fp8_e4m3fn_to_float(__ldg(kv_row + d));
                dot = fmaf(q_vec[d], kvd, dot);
            }
            logits[t] = dot;
        }
    } else {
        for (int t = threadIdx.x; t < params.topk; t += blockDim.x) {
            const int kv_base = kv_base_stage[t];
            if (kv_base < 0) {
                logits[t] = -CUDART_INF_F;
                continue;
            }
            const uint8_t* kv_row = params.kv + kv_base;

            float dot = 0.0f;
            int d = 0;
            for (; d + 3 < params.d_qk; d += 4) {
                const uint32_t packed_kv = __ldg(reinterpret_cast<const uint32_t*>(kv_row + d));
                float kv0, kv1, kv2, kv3;
                fp8x4_e4m3fn_to_float(packed_kv, kv0, kv1, kv2, kv3);
                dot = fmaf(q_vec[d + 0], kv0, dot);
                dot = fmaf(q_vec[d + 1], kv1, dot);
                dot = fmaf(q_vec[d + 2], kv2, dot);
                dot = fmaf(q_vec[d + 3], kv3, dot);
            }
            for (; d < params.d_qk; ++d) {
                const float kvd = fp8_e4m3fn_to_float(__ldg(kv_row + d));
                dot = fmaf(q_vec[d], kvd, dot);
            }
            logits[t] = dot;
        }
    }
    __syncthreads();

    float local_max = -CUDART_INF_F;
    for (int t = threadIdx.x; t < params.topk; t += blockDim.x) {
        local_max = max(local_max, logits[t]);
    }
    const unsigned full_mask = 0xffffffffu;
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;

    float warp_max = local_max;
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_max = max(warp_max, __shfl_down_sync(full_mask, warp_max, offset));
    }
    if (lane_id == 0) {
        s_reduce[warp_id] = warp_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_max = (lane_id < num_warps) ? s_reduce[lane_id] : -CUDART_INF_F;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_max = max(block_max, __shfl_down_sync(full_mask, block_max, offset));
        }
        if (lane_id == 0) {
            s_max_logit = block_max;
            s_has_valid = (block_max != -CUDART_INF_F) ? 1 : 0;
        }
    }
    __syncthreads();

    if (!s_has_valid) {
        for (int t = threadIdx.x; t < params.topk; t += blockDim.x) {
            weights[t] = 0.0f;
        }
        if (threadIdx.x == 0) {
            s_lse = CUDART_INF_F;
            s_inv_denom = 0.0f;
        }
    } else {
        float local_sum = 0.0f;
        for (int t = threadIdx.x; t < params.topk; t += blockDim.x) {
            const float logit = logits[t];
            const float w = (logit == -CUDART_INF_F) ? 0.0f : __expf(logit - s_max_logit);
            weights[t] = w;
            local_sum += w;
        }

        float warp_sum = local_sum;
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(full_mask, warp_sum, offset);
        }
        if (lane_id == 0) {
            s_reduce[warp_id] = warp_sum;
        }
        __syncthreads();

        if (warp_id == 0) {
            float block_sum = (lane_id < num_warps) ? s_reduce[lane_id] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1) {
                block_sum += __shfl_down_sync(full_mask, block_sum, offset);
            }
            if (lane_id == 0) {
                const float sum_exp = block_sum;
                const float attn_sink = params.attn_sink ? params.attn_sink[q_h_idx] : -CUDART_INF_F;
                const float sink_term = __expf(attn_sink - s_max_logit);
                const float denom = sum_exp + sink_term;
                s_inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;
                s_lse = logf(max(denom, 1e-30f)) + s_max_logit;
            }
        }
        __syncthreads();

        for (int t = threadIdx.x; t < params.topk; t += blockDim.x) {
            weights[t] *= (s_inv_denom * kv_scale);
        }
    }
    __syncthreads();

    const int out_base = (s_q_idx * params.h_q + q_h_idx) * params.d_v;
    if (!s_has_valid) {
        for (int dv_idx = threadIdx.x * 4; dv_idx < params.d_v; dv_idx += blockDim.x * 4) {
            if (dv_idx + 3 < params.d_v) {
                params.out[out_base + dv_idx + 0] = static_cast<bf16>(0.0f);
                params.out[out_base + dv_idx + 1] = static_cast<bf16>(0.0f);
                params.out[out_base + dv_idx + 2] = static_cast<bf16>(0.0f);
                params.out[out_base + dv_idx + 3] = static_cast<bf16>(0.0f);
            } else {
                for (int d_tail = dv_idx; d_tail < params.d_v; ++d_tail) {
                    params.out[out_base + d_tail] = static_cast<bf16>(0.0f);
                }
            }
        }
    } else if (false && s_all_valid) {
        for (int dv_idx = threadIdx.x * 4; dv_idx < params.d_v; dv_idx += blockDim.x * 4) {
            if (dv_idx + 3 < params.d_v) {
                float out0 = 0.0f;
                float out1 = 0.0f;
                float out2 = 0.0f;
                float out3 = 0.0f;
                int t = 0;
                for (; t + 1 < params.topk; t += 2) {
                    const float w0 = weights[t + 0];
                    if (w0 != 0.0f) {
                        const int kv_base0 = kv_base_stage[t + 0];
                        const uint8_t* kv_row0 = params.kv + kv_base0;
                        const uint32_t packed_kv0 = __ldg(reinterpret_cast<const uint32_t*>(kv_row0 + dv_idx));
                        float kv00, kv01, kv02, kv03;
                        fp8x4_e4m3fn_to_float(packed_kv0, kv00, kv01, kv02, kv03);
                        out0 = fmaf(w0, kv00, out0);
                        out1 = fmaf(w0, kv01, out1);
                        out2 = fmaf(w0, kv02, out2);
                        out3 = fmaf(w0, kv03, out3);
                    }

                    const float w1 = weights[t + 1];
                    if (w1 != 0.0f) {
                        const int kv_base1 = kv_base_stage[t + 1];
                        const uint8_t* kv_row1 = params.kv + kv_base1;
                        const uint32_t packed_kv1 = __ldg(reinterpret_cast<const uint32_t*>(kv_row1 + dv_idx));
                        float kv10, kv11, kv12, kv13;
                        fp8x4_e4m3fn_to_float(packed_kv1, kv10, kv11, kv12, kv13);
                        out0 = fmaf(w1, kv10, out0);
                        out1 = fmaf(w1, kv11, out1);
                        out2 = fmaf(w1, kv12, out2);
                        out3 = fmaf(w1, kv13, out3);
                    }
                }
                for (; t < params.topk; ++t) {
                    const float w = weights[t];
                    if (w == 0.0f) {
                        continue;
                    }
                    const int kv_base = kv_base_stage[t];
                    const uint8_t* kv_row = params.kv + kv_base;
                    const uint32_t packed_kv = __ldg(reinterpret_cast<const uint32_t*>(kv_row + dv_idx));
                    float kv0, kv1, kv2, kv3;
                    fp8x4_e4m3fn_to_float(packed_kv, kv0, kv1, kv2, kv3);
                    out0 = fmaf(w, kv0, out0);
                    out1 = fmaf(w, kv1, out1);
                    out2 = fmaf(w, kv2, out2);
                    out3 = fmaf(w, kv3, out3);
                }
                params.out[out_base + dv_idx + 0] = static_cast<bf16>(out0);
                params.out[out_base + dv_idx + 1] = static_cast<bf16>(out1);
                params.out[out_base + dv_idx + 2] = static_cast<bf16>(out2);
                params.out[out_base + dv_idx + 3] = static_cast<bf16>(out3);
            } else {
                for (int d_tail = dv_idx; d_tail < params.d_v; ++d_tail) {
                    float out_val = 0.0f;
                    for (int t = 0; t < params.topk; ++t) {
                        const float w = weights[t];
                        if (w == 0.0f) {
                            continue;
                        }
                        const int kv_base = kv_base_stage[t];
                        const uint8_t* kv_row = params.kv + kv_base;
                        const float kvv = fp8_e4m3fn_to_float(__ldg(kv_row + d_tail));
                        out_val = fmaf(w, kvv, out_val);
                    }
                    params.out[out_base + d_tail] = static_cast<bf16>(out_val);
                }
            }
        }
    } else {
        for (int dv_idx = threadIdx.x * 4; dv_idx < params.d_v; dv_idx += blockDim.x * 4) {
            if (dv_idx + 3 < params.d_v) {
                float out0 = 0.0f;
                float out1 = 0.0f;
                float out2 = 0.0f;
                float out3 = 0.0f;
                int t = 0;
                for (; t + 1 < params.topk; t += 2) {
                    const float w0 = weights[t + 0];
                    if (w0 != 0.0f) {
                        const int kv_base0 = kv_base_stage[t + 0];
                        if (kv_base0 >= 0) {
                            const uint8_t* kv_row0 = params.kv + kv_base0;
                            const uint32_t packed_kv0 = __ldg(reinterpret_cast<const uint32_t*>(kv_row0 + dv_idx));
                            float kv00, kv01, kv02, kv03;
                            fp8x4_e4m3fn_to_float(packed_kv0, kv00, kv01, kv02, kv03);
                            out0 = fmaf(w0, kv00, out0);
                            out1 = fmaf(w0, kv01, out1);
                            out2 = fmaf(w0, kv02, out2);
                            out3 = fmaf(w0, kv03, out3);
                        }
                    }

                    const float w1 = weights[t + 1];
                    if (w1 != 0.0f) {
                        const int kv_base1 = kv_base_stage[t + 1];
                        if (kv_base1 >= 0) {
                            const uint8_t* kv_row1 = params.kv + kv_base1;
                            const uint32_t packed_kv1 = __ldg(reinterpret_cast<const uint32_t*>(kv_row1 + dv_idx));
                            float kv10, kv11, kv12, kv13;
                            fp8x4_e4m3fn_to_float(packed_kv1, kv10, kv11, kv12, kv13);
                            out0 = fmaf(w1, kv10, out0);
                            out1 = fmaf(w1, kv11, out1);
                            out2 = fmaf(w1, kv12, out2);
                            out3 = fmaf(w1, kv13, out3);
                        }
                    }
                }
                for (; t < params.topk; ++t) {
                    const float w = weights[t];
                    if (w == 0.0f) {
                        continue;
                    }
                    const int kv_base = kv_base_stage[t];
                    if (kv_base < 0) {
                        continue;
                    }
                    const uint8_t* kv_row = params.kv + kv_base;
                    const uint32_t packed_kv = __ldg(reinterpret_cast<const uint32_t*>(kv_row + dv_idx));
                    float kv0, kv1, kv2, kv3;
                    fp8x4_e4m3fn_to_float(packed_kv, kv0, kv1, kv2, kv3);
                    out0 = fmaf(w, kv0, out0);
                    out1 = fmaf(w, kv1, out1);
                    out2 = fmaf(w, kv2, out2);
                    out3 = fmaf(w, kv3, out3);
                }
                params.out[out_base + dv_idx + 0] = static_cast<bf16>(out0);
                params.out[out_base + dv_idx + 1] = static_cast<bf16>(out1);
                params.out[out_base + dv_idx + 2] = static_cast<bf16>(out2);
                params.out[out_base + dv_idx + 3] = static_cast<bf16>(out3);
            } else {
                for (int d_tail = dv_idx; d_tail < params.d_v; ++d_tail) {
                    float out_val = 0.0f;
                    for (int t = 0; t < params.topk; ++t) {
                        const float w = weights[t];
                        if (w == 0.0f) {
                            continue;
                        }
                        const int kv_base = kv_base_stage[t];
                        if (kv_base < 0) {
                            continue;
                        }
                        const uint8_t* kv_row = params.kv + kv_base;
                        const float kvv = fp8_e4m3fn_to_float(__ldg(kv_row + d_tail));
                        out_val = fmaf(w, kvv, out_val);
                    }
                    params.out[out_base + d_tail] = static_cast<bf16>(out_val);
                }
            }
        }
    }

    if (threadIdx.x == 0) {
        params.max_logits[s_q_idx * params.h_q + q_h_idx] = s_max_logit;
        params.lse[s_q_idx * params.h_q + q_h_idx] = s_lse;
    }
}

template<int D_QK, bool HAVE_TOPK_LENGTH>
template<typename TMAParams>
__device__ void KernelTemplate<D_QK, HAVE_TOPK_LENGTH>::devfunc(const SparseAttnFwdParams &params, const TMAParams &tma_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    const int q_h_idx = blockIdx.x % (params.h_q/B_H);
    const int s_q_idx = blockIdx.x / (params.h_q/B_H);
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int idx_in_warpgroup = threadIdx.x % 128;

    // Define shared tensors
    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
    Tensor sQ = make_tensor(make_smem_ptr(plan.q_o.q.data()), SmemLayoutQ{});
    Tensor sO = make_tensor(make_smem_ptr(plan.q_o.o.data()), SmemLayoutO{});
    Tensor sS0 = make_tensor(make_smem_ptr(D_QK == 576 ? plan.k[0].data()+64*512 : plan.s[1].data()), SmemLayoutS{});    // Overlap with sK0's RoPE part for V3.2
    Tensor sS1 = make_tensor(make_smem_ptr(plan.s[0].data()), SmemLayoutS{});

    if (warp_idx == 0 && elect_one_sync()) {
        // Prefetch TMA descriptors
        cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_O);

        // Initialize barriers
        plan.bar_q.init(1);
        CUTE_UNROLL
        for (int i = 0; i < 2; ++i) {
            plan.bar_k0_free[i].init(128);
            plan.bar_k0_ready[i].init(128);
            plan.bar_k1_free[i].init(128);
            plan.bar_k1_ready[i].init(128);
        }
        plan.bar_is_kv_valid_ready.init(16);
        fence_barrier_init();
    }

    __syncthreads();
    const int topk_length = HAVE_TOPK_LENGTH ? __ldg(params.topk_length + s_q_idx) : params.topk;
    const int num_topk_blocks = HAVE_TOPK_LENGTH ? ku::ceil_div(topk_length, (int)B_TOPK) : (int)((unsigned int)params.topk/(unsigned int)B_TOPK);
    if (warpgroup_idx == 0 || warpgroup_idx == 1) {
        cutlass::arch::warpgroup_reg_alloc<216>();

        if (warp_idx == 0 && elect_one_sync()) {
            // Load Q
            Tensor gQ = flat_divide(
                tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx),
                Tile<Int<B_H>, Int<D_Q>>{}
            )(_, _, q_h_idx, _0{});
            launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q, TMA::CacheHintSm90::EVICT_FIRST);
            plan.bar_q.arrive_and_expect_tx(B_H*D_Q*sizeof(bf16));
        }

        float rM[2] = {MAX_INIT_VAL, MAX_INIT_VAL}; // Meaning: the `max_logits` used for O / rL calculation
        float rL[2] = {0.0f, 0.0f};
        Tensor rO = partition_fragment_C(TiledMMA_PV_LocalP{}, Shape<Int<B_H>, Int<D_V/2>>{});
        Tensor rP = partition_fragment_C(TiledMMA_QK{}, Shape<Int<B_H>, Int<B_TOPK>>{});
        Tensor rS = make_tensor<bf16>(partition_shape_A(TiledMMA_PV_LocalP{}, Shape<Int<B_H>, Int<B_TOPK>>{}));
        cute::fill(rO, 0.0f);

        // Wait for Q
        plan.bar_q.wait(0);

        bool cur_bar_wait_phase = 0;

        struct Warpgroup0 {};
        struct Warpgroup1 {};

        auto qkt_gemm_one_tile = [&](auto warpgroup_idx, int tile_idx, bool clear_accum) {
            constexpr bool IS_WG1 = std::is_same_v<decltype(warpgroup_idx), Warpgroup1>;
            TiledMMA tiled_mma_QK = TiledMMA_QK{};
            Tensor sQ_tile = flat_divide(sQ, Tile<Int<B_H>, Int<64>>{})(_, _, _0{}, tile_idx);
            Tensor sK_tile = make_tensor(make_smem_ptr(plan.k[(int)IS_WG1].data() + tile_idx*B_TOPK*64), SmemLayoutKTiles<1>{});
            gemm_ss(clear_accum, tiled_mma_QK, sQ_tile, sK_tile, rP, idx_in_warpgroup);
        };

        auto mask_rP = [&](auto warpgroup_idx) {
            constexpr bool IS_WG1 = std::is_same_v<decltype(warpgroup_idx), Warpgroup1>;
            plan.bar_is_kv_valid_ready.wait(cur_bar_wait_phase);
            CUTE_UNROLL
            for (int row_idx = 0; row_idx < 2; ++row_idx) {
                CUTE_UNROLL
                for (int i = row_idx*2; i < size(rP); i += 4) {
                    int col = 8*(i/4) + (idx_in_warpgroup%4)*2;
                    if (!plan.is_kv_valid[IS_WG1][col]) rP(i) = -INFINITY;
                    if (!plan.is_kv_valid[IS_WG1][col+1]) rP(i+1) = -INFINITY;
                }
            }
        };

        auto online_softmax_and_rescale_o = [&](auto warpgroup_idx) {
            plan.bar_is_kv_valid_ready.wait(cur_bar_wait_phase);
            constexpr bool IS_WG1 = std::is_same_v<decltype(warpgroup_idx), Warpgroup1>;
            const float scale = params.sm_scale_div_log2;
            float r_sM[2];
            if constexpr (IS_WG1) {
                *(float2*)r_sM = plan.sM[idx_in_warpgroup/4];
            }
            float new_maxs[2];
            CUTE_UNROLL
            for (int row_idx = 0; row_idx < 2; ++row_idx) {
                // Get rowwise max
                float cur_max = -INFINITY;
                CUTE_UNROLL
                for (int i = row_idx*2; i < size(rP); i += 4) {
                    cur_max = max(cur_max, max(rP(i), rP(i+1)));
                }
                cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
                cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));
                cur_max *= scale;

                // Get new max and scale
                // For WG1, old_max comes from sM (written by WG0); for WG0, old_max comes from rM (read by WG0 from sM in the last round)
                new_maxs[row_idx] = max(IS_WG1 ? r_sM[row_idx] : rM[row_idx], cur_max);

                // Scale O
                float scale_for_o = exp2f(rM[row_idx]-new_maxs[row_idx]);
                CUTE_UNROLL
                for (int i = row_idx*2; i < size(rO); i += 4) {
                    rO(i) *= scale_for_o;
                    rO(i+1) *= scale_for_o;
                }

                // Get rS
                float cur_sum = 0;
                CUTE_UNROLL
                for (int i = row_idx*2; i < size(rP); i += 4) {
                    rP(i) = exp2f(rP(i)*scale - new_maxs[row_idx]);
                    rP(i+1) = exp2f(rP(i+1)*scale - new_maxs[row_idx]);
                    rS(i) = (bf16)rP(i);
                    rS(i+1) = (bf16)rP(i+1);
                    cur_sum += rP(i) + rP(i+1);
                }
                rL[row_idx] = rL[row_idx]*scale_for_o + cur_sum;
            }
            __syncwarp();
            if (idx_in_warpgroup%4 == 0) {
                plan.sM[idx_in_warpgroup/4] = *(float2*)new_maxs;
            }
            rM[0] = new_maxs[0];
            rM[1] = new_maxs[1];
        };

        auto reduce_L = [&]() {
            // Reduce L
            // For example, thread 0 reduces with thread 1, 2, and 3, as well as thread 128, 129, 130, and 131
            rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 1);
            rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 2);
            rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 1);
            rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 2);
            if (idx_in_warpgroup%4 == 0)
                plan.sL[threadIdx.x/4] = *(float2*)(rL);
            NamedBarrier::arrive_and_wait(256, NamedBarriers::sL_ready);
            float2 peer_L = plan.sL[(threadIdx.x/4)^32];
            rL[0] += peer_L.x;
            rL[1] += peer_L.y;
        };

        auto store_O = [&]() {
            float scale_factors[2];
            CUTE_UNROLL
            for (int i = 0; i < 2; ++i) {
                float attn_sink = params.attn_sink == nullptr ? -CUDART_INF_F : params.attn_sink[q_h_idx*B_H + get_AorC_row_idx(i, idx_in_warpgroup)]*CUDART_L2E_F;
                scale_factors[i] = 1.0f / (rL[i] + exp2f(attn_sink - rM[i]));
                if (rL[i] == 0.0f)
                    scale_factors[i] = 0.0f;    // The output should be 0 whatever attn_sink is
            }

            Tensor sO = make_tensor(make_smem_ptr(plan.q_o.o.data() + warpgroup_idx*B_H*(D_V/2)), SmemLayoutOTiles<4>{});
            bf16* stsm_addrs[4];
            int stsm_row = (idx_in_warpgroup/32)*16 + (idx_in_warpgroup%16);
            CUTE_UNROLL
            for (int i = 0; i < 64/16; ++i) {
                stsm_addrs[i] = &sO(stsm_row, (idx_in_warpgroup%32/16*8) + 16*i);
            }
            bool s2g_pred = warp_idx%4 == 0 && elect_one_sync();

            warpgroup_wait<0>();
            CUTE_UNROLL
            for (int tile_idx = 0; tile_idx < (D_V/2)/64; tile_idx += 1) {
                // Convert
                constexpr int NUM_ELEMS_EACH_TILE = B_H*64 / 128;   // 64: tile size, 128: warpgroup size
                bf16 cur_rOb[NUM_ELEMS_EACH_TILE];
                CUTE_UNROLL
                for (int i = 0; i < NUM_ELEMS_EACH_TILE; ++i) {
                    cur_rOb[i] = (bf16)(rO(tile_idx*NUM_ELEMS_EACH_TILE + i) * scale_factors[i%4>=2]);
                }
                // R -> S
                CUTE_UNROLL
                for (int i = 0; i < 64/16; ++i) {
                    SM90_U32x4_STSM_N::copy(
                        *reinterpret_cast<uint32_t*>(cur_rOb + i*8 + 0),
                        *reinterpret_cast<uint32_t*>(cur_rOb + i*8 + 2),
                        *reinterpret_cast<uint32_t*>(cur_rOb + i*8 + 4),
                        *reinterpret_cast<uint32_t*>(cur_rOb + i*8 + 6),
                        *reinterpret_cast<uint128_t*>(stsm_addrs[i] + tile_idx*(B_H*64))
                    );
                }
                fence_view_async_shared();
                NamedBarrier::arrive_and_wait(128, warpgroup_idx ? NamedBarriers::warpgroup1_sync : NamedBarriers::warpgroup0_sync);
                // S -> G
                if (s2g_pred) {
                    int g_tile_idx = warpgroup_idx*4 + tile_idx;
                    SM90_TMA_STORE_3D::copy(
                        &tma_params.tensor_map_O,
                        plan.q_o.o.data() + g_tile_idx*(B_H*64),
                        g_tile_idx*64,
                        q_h_idx*B_H,
                        s_q_idx
                    );
                }
            }
            cute::tma_store_arrive();
        };


        if (warpgroup_idx == 0) {
            // Warpgroup 0

            auto pipelined_wait_and_qkt_gemm_l = [&]() __attribute__((always_inline)) {
                plan.bar_k0_ready[0].wait(cur_bar_wait_phase);
                qkt_gemm_one_tile(Warpgroup0{}, 0, true);
                qkt_gemm_one_tile(Warpgroup0{}, 1, false);
                qkt_gemm_one_tile(Warpgroup0{}, 2, false);
                qkt_gemm_one_tile(Warpgroup0{}, 3, false);
                warpgroup_commit_batch();
            };

            auto pipelined_wait_and_qkt_gemm_r = [&]() __attribute__((always_inline)) {
                plan.bar_k0_ready[1].wait(cur_bar_wait_phase);
                qkt_gemm_one_tile(Warpgroup0{}, 4, false);
                qkt_gemm_one_tile(Warpgroup0{}, 5, false);
                qkt_gemm_one_tile(Warpgroup0{}, 6, false);
                qkt_gemm_one_tile(Warpgroup0{}, 7, false);
                if constexpr (D_QK == 576) {
                    qkt_gemm_one_tile(Warpgroup0{}, 8, false);
                }
                warpgroup_commit_batch();
            };

            auto scale_rS = [&](float scales[2]) {
                CUTE_UNROLL
                for (int row = 0; row < 2; ++row) {
                    CUTE_UNROLL
                    for (int i = row*2; i < size(rP); i += 4) {
                        rS(i) = (bf16)(rP(i) * scales[row]);
                        rS(i+1) = (bf16)(rP(i+1) * scales[row]);
                    }
                }
            };

            auto rescale_rO = [&](float scales[2]) {
                CUTE_UNROLL
                for (int row = 0; row < 2; ++row) {
                    CUTE_UNROLL
                    for (int i = row*2; i < size(rO); i += 4) {
                        rO(i) *= scales[row];
                        rO(i+1) *= scales[row];
                    }
                    rL[row] *= scales[row];
                }
            };
            
            CUTE_NO_UNROLL
            for (int block_idx = 0; block_idx < num_topk_blocks; block_idx += 2) {
                Tensor sV0l = make_tensor(make_smem_ptr(plan.k[0].data()), SmemLayoutKTilesTransposed<4>{});
                Tensor sV1l = make_tensor(make_smem_ptr(plan.k[1].data()), SmemLayoutKTilesTransposed<4>{});

                if (block_idx == 0) {
                    // NOTE: We put this code here to avoid register spilling
                    pipelined_wait_and_qkt_gemm_l();
                    pipelined_wait_and_qkt_gemm_r();
                    warpgroup_wait<0>();
                }
                
                // Online softmax, inform WG1
                mask_rP(Warpgroup0{});
                
                
                online_softmax_and_rescale_o(Warpgroup0{});
                NamedBarrier::arrive(256, NamedBarriers::wg0_bunch_0_ready);

                // Issue rO0 += rS0 @ sV0l
                gemm_rs(false, TiledMMA_PV_LocalP{}, rS, sV0l, rO, idx_in_warpgroup);
                warpgroup_commit_batch();

                // Mark V0L as free
                warpgroup_wait<0>();
                plan.bar_k0_free[0].arrive();

                // Wait for new sM, scale rS, save, inform WG1
                NamedBarrier::arrive_and_wait(256, NamedBarriers::wg1_bunch_0_ready);
                float new_rM[2], scale_factors[2];
                *(float2*)new_rM = plan.sM[idx_in_warpgroup/4];
                CUTE_UNROLL
                for (int i = 0; i < 2; ++i) {
                    scale_factors[i] = exp2f(rM[i] - new_rM[i]);
                    rM[i] = new_rM[i];
                }
                scale_rS(scale_factors);
                save_rS_to_sS(rS, sS0, idx_in_warpgroup);
                fence_view_async_shared();
                NamedBarrier::arrive(256, NamedBarriers::wg0_s0_ready);

                // Wait for sS1
                NamedBarrier::arrive_and_wait(256, NamedBarriers::wg1_s1_ready);

                // Rescale rO0, Issue rO0 += sS1 @ sV1L
                rescale_rO(scale_factors);
                gemm_ss(false, TiledMMA_PV_RemoteP{}, sS1, sV1l, rO, idx_in_warpgroup);
                warpgroup_commit_batch();

                cur_bar_wait_phase ^= 1;

                if (block_idx+2 < num_topk_blocks) {
                    // Launch the next QK^T GEMM
                    pipelined_wait_and_qkt_gemm_l();

                    // Mark V1L as free
                    warpgroup_wait<1>();
                    plan.bar_k1_free[0].arrive();
                    pipelined_wait_and_qkt_gemm_r();

                    // Wait for rP0 = sQ @ sK0
                    warpgroup_wait<0>();
                } else {
                    // Mark V1L as free
                    warpgroup_wait<0>();
                    plan.bar_k1_free[0].arrive();
                }
            }

            reduce_L();
            store_O();
        } else {
            // Warpgroup 1

            auto pipelined_wait_and_qkt_gemm = [&]() __attribute__((always_inline)) {
                plan.bar_k1_ready[1].wait(cur_bar_wait_phase);
                qkt_gemm_one_tile(Warpgroup1{}, 4, true);
                qkt_gemm_one_tile(Warpgroup1{}, 5, false);
                qkt_gemm_one_tile(Warpgroup1{}, 6, false);
                qkt_gemm_one_tile(Warpgroup1{}, 7, false);
                if constexpr (D_QK == 576) {
                    qkt_gemm_one_tile(Warpgroup1{}, 8, false);
                }
                plan.bar_k1_ready[0].wait(cur_bar_wait_phase);
                qkt_gemm_one_tile(Warpgroup1{}, 0, false);
                qkt_gemm_one_tile(Warpgroup1{}, 1, false);
                qkt_gemm_one_tile(Warpgroup1{}, 2, false);
                qkt_gemm_one_tile(Warpgroup1{}, 3, false);
                warpgroup_commit_batch();
            };
            
            CUTE_NO_UNROLL
            for (int block_idx = 0; block_idx < num_topk_blocks; block_idx += 2) {
                Tensor sV0r = make_tensor(make_smem_ptr(plan.k[0].data()+64*256), SmemLayoutKTilesTransposed<4>{});
                Tensor sV1r = make_tensor(make_smem_ptr(plan.k[1].data()+64*256), SmemLayoutKTilesTransposed<4>{});

                // Issue rP1 = sQ @ sK1, and wait
                pipelined_wait_and_qkt_gemm();
                warpgroup_wait<0>();

                mask_rP(Warpgroup1{});


                // Wait for WG0 (for sM), online softmax, Notify WG0 (sM ready)
                NamedBarrier::arrive_and_wait(256, NamedBarriers::wg0_bunch_0_ready);
                online_softmax_and_rescale_o(Warpgroup1{});
                NamedBarrier::arrive(256, NamedBarriers::wg1_bunch_0_ready);


                // Issue rO1 += rS1 @ sV1R
                gemm_rs(false, TiledMMA_PV_LocalP{}, rS, sV1r, rO, idx_in_warpgroup);
                warpgroup_commit_batch();
                
                // Wait for WG0 (for sS0), Issue rO1 += rS0 @ sV0R
                save_rS_to_sS(rS, sS1, idx_in_warpgroup);   // Put it here is faster
                NamedBarrier::arrive_and_wait(256, NamedBarriers::wg0_s0_ready);
                gemm_ss(false, TiledMMA_PV_RemoteP{}, sS0, sV0r, rO, idx_in_warpgroup);
                warpgroup_commit_batch();
                
                // Save rS1, inform WG0
                fence_view_async_shared();
                NamedBarrier::arrive(256, NamedBarriers::wg1_s1_ready);

                // Wait for GEMM, and inform that sV1R is free
                warpgroup_wait<1>();
                plan.bar_k1_free[1].arrive();

                // Wait for GEMM, and inform that sV0R is free
                warpgroup_wait<0>();
                plan.bar_k0_free[1].arrive();

                cur_bar_wait_phase ^= 1;
            }

            reduce_L();
            store_O();

            // Save lse
            if (idx_in_warpgroup%4 == 0) {
                for (int row = 0; row < 2; ++row) {
                    int real_row = get_AorC_row_idx(row, idx_in_warpgroup);
                    bool is_no_valid_tokens = rL[row] == 0.0f;
                    plan.final_max_logits[real_row] = is_no_valid_tokens ? -INFINITY : rM[row]*CUDART_LN2_F;
                    plan.final_lse[real_row] = is_no_valid_tokens ? +INFINITY : logf(rL[row]) + rM[row]*CUDART_LN2_F;
                }
                fence_view_async_shared();
            }

            NamedBarrier::arrive_and_wait(128, NamedBarriers::warpgroup1_sync);
            if (idx_in_warpgroup == 0) {
                int g_offset = s_q_idx*params.h_q + q_h_idx*B_H;
                SM90_BULK_COPY_S2G::copy(plan.final_max_logits, params.max_logits + g_offset, B_H*sizeof(float));
                SM90_BULK_COPY_S2G::copy(plan.final_lse, params.lse + g_offset, B_H*sizeof(float));
                cute::tma_store_arrive();
            }
        }
    } else {
        // Producer warpgroup
        cutlass::arch::warpgroup_reg_dealloc<72>();

        constexpr int GROUP_SIZE = 8, NUM_GROUPS = 128/GROUP_SIZE;
        constexpr int NUM_ROWS_PER_GROUP = B_TOPK / NUM_GROUPS;
        int idx_in_group = idx_in_warpgroup % GROUP_SIZE;
        int group_idx = idx_in_warpgroup / GROUP_SIZE;
        int* gIndices = params.indices + s_q_idx*params.stride_indices_s_q;   // [topk]

        bf16* my_sKV_base = &(make_tensor(make_smem_ptr(plan.k[0].data()), SmemLayoutKTiles<1>{})(group_idx, idx_in_group*8));
        bf16* my_gKV_base = params.kv + idx_in_group*8;
        
        int64_t token_indices[2][NUM_ROWS_PER_GROUP];
        bool is_token_valid[2][NUM_ROWS_PER_GROUP];
        auto load_token_indices = [&](int block_idx) {
            CUTE_UNROLL
            for (int buf_idx = 0; buf_idx < 2; ++buf_idx) {
                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_ROWS_PER_GROUP; ++local_row) {
                    int offs = (block_idx+buf_idx)*B_TOPK + local_row*NUM_GROUPS + group_idx;
                    int t = __ldg(gIndices + offs);
                    token_indices[buf_idx][local_row] = t*(int64_t)params.stride_kv_s_kv;   // We mult it with params.stride_kv_s_kv here since it's faster
                    bool is_cur_token_valid = t >= 0 && t < params.s_kv;
                    if constexpr (HAVE_TOPK_LENGTH) {
                        is_cur_token_valid &= offs < topk_length;
                    }
                    is_token_valid[buf_idx][local_row] = is_cur_token_valid;
                }
            }
        };
        
        int64_t cache_policy = createpolicy_evict_last();
        auto copy_tiles = [&](int block_idx, int buf_idx, int tile_start, int tile_end) {
            // Copy some K/V tiles from global memory to shared memory
            // A tile has a shape of 64 (B_TOPK) x 64
            // `buf_idx` is the index of the shared memory buffer, 0 or 1
            // `tile_idx` is the index of the tile to load, from 0 to D_K/64-1 = 8
            CUTE_UNROLL
            for (int local_row = 0; local_row < NUM_ROWS_PER_GROUP; ++local_row) {
                int64_t token_index = token_indices[buf_idx][local_row];
                CUTE_UNROLL
                for (int tile_idx = tile_start; tile_idx < tile_end; ++tile_idx) {
                    cp_async_cacheglobal_l2_prefetch_256B(
                        my_gKV_base + token_index + tile_idx*64,
                        my_sKV_base + (buf_idx*B_TOPK*D_K + tile_idx*(B_TOPK*64) + local_row*NUM_GROUPS*64),
                        is_token_valid[buf_idx][local_row],
                        cache_policy
                    );
                }
            }
        };

        auto commit_to_mbar = [&](transac_bar_t &bar) {
            cutlass::arch::cpasync_barrier_arrive_noinc((uint64_t*)(&bar));
        };

        int cur_bar_wait_phase = 1;

        CUTE_NO_UNROLL
        for (int block_idx = 0; block_idx < num_topk_blocks; block_idx += 2) {
            load_token_indices(block_idx);

            // V0L
            plan.bar_k0_free[0].wait(cur_bar_wait_phase);
            copy_tiles(block_idx+0, 0, 0, 4);
            commit_to_mbar(plan.bar_k0_ready[0]);

            // V1R
            plan.bar_k1_free[1].wait(cur_bar_wait_phase);
            copy_tiles(block_idx+1, 1, 4, D_K/64);
            commit_to_mbar(plan.bar_k1_ready[1]);
            
            // V0R
            plan.bar_k0_free[1].wait(cur_bar_wait_phase);
            copy_tiles(block_idx+0, 0, 4, D_K/64);
            commit_to_mbar(plan.bar_k0_ready[1]);

            // V1L
            plan.bar_k1_free[0].wait(cur_bar_wait_phase);
            copy_tiles(block_idx+1, 1, 0, 4);
            commit_to_mbar(plan.bar_k1_ready[0]);

            // Valid mask
            // NOTE: V1R's finish implies maskings of the last round have finished
            if (idx_in_group == 0) {
                CUTE_UNROLL
                for (int buf_idx = 0; buf_idx < 2; ++buf_idx)
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < NUM_ROWS_PER_GROUP; ++local_row)
                        plan.is_kv_valid[buf_idx][local_row*NUM_GROUPS+group_idx] = is_token_valid[buf_idx][local_row];
                plan.bar_is_kv_valid_ready.arrive();
            }

            cur_bar_wait_phase ^= 1;
        }
    }


#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm90");
    }
#endif
}

template<typename Kernel, typename TMAParams>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, 1)
sparse_attn_fwd_kernel(__grid_constant__ const SparseAttnFwdParams params, __grid_constant__ const TMAParams tma_params) {
    Kernel::devfunc(params, tma_params);
}

template<int D_QK, bool HAVE_TOPK_LENGTH>
void KernelTemplate<D_QK, HAVE_TOPK_LENGTH>::run(const SparseAttnFwdParams &params) {
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk % (2*B_TOPK) == 0);   // To save some boundry checkings
    KU_ASSERT(params.topk > 0);
    KU_ASSERT(params.h_q % B_H == 0);

    auto shape_Q = make_shape(params.h_q, params.d_qk, params.s_q);
    auto tma_Q = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.q),
            make_layout(
                shape_Q,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q)
            )
        ),
        SmemLayoutQ{}
    );

    CUtensorMap tensor_map_O;
    {
        uint64_t size[3] = {D_V, (unsigned long)params.h_q, (unsigned long)params.s_q};
        uint64_t stride[2] = {D_V*sizeof(bf16), D_V*params.h_q*sizeof(bf16)};
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

    TmaParams<
        decltype(shape_Q), decltype(tma_Q)
    > tma_params = {
        shape_Q, tma_Q,
        tensor_map_O
    };
    auto kernel = &sparse_attn_fwd_kernel<KernelTemplate<D_QK, HAVE_TOPK_LENGTH>, decltype(tma_params)>;

    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    cutlass::ClusterLaunchParams launch_params = {
        dim3((params.h_q/B_H)*params.s_q, 1, 1),    // NOTE: We put s_q on the first dim since it can be larger than 65536 (the maximum size of griddim.y and griddim.z)
        dim3(NUM_THREADS, 1, 1),
        dim3(1, 1, 1),
        smem_size,
        params.stream
    }; 
    cutlass::launch_kernel_on_cluster(
        launch_params, (void*)kernel, params, tma_params
    );
    KU_CHECK_KERNEL_LAUNCH();
}

template<int D_QK, bool HAVE_TOPK_LENGTH>
void run_fwd_phase1_kernel(const SparseAttnFwdParams& params) {
    KernelTemplate<D_QK, HAVE_TOPK_LENGTH>::run(params);
}

template<int D_QK, bool HAVE_TOPK_LENGTH>
struct KernelTemplateQ8 {
    static void run(const SparseAttnFwdQ8Params& params) {
        constexpr int kThreads = 128;
        const int grid = params.s_q * params.h_q;
        const size_t smem_bytes = sizeof(float) * (576 + 2 * params.topk)
                                + sizeof(int) * params.topk;
        sparse_attn_fwd_q8_direct_kernel<<<grid, kThreads, smem_bytes, params.stream>>>(params);
        KU_CHECK_KERNEL_LAUNCH();
    }
};

template<int D_QK, bool HAVE_TOPK_LENGTH>
void run_fwd_phase1_q8_kernel(const SparseAttnFwdQ8Params& params) {
    KernelTemplateQ8<D_QK, HAVE_TOPK_LENGTH>::run(params);
}

}
