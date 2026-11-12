#pragma once

#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/epilogue/sm90_mega_gate.cuh>
#include <deep_gemm/layout/sm90_mega_gate.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

namespace deep_gemm {

template <uint32_t SHAPE_K, uint32_t kNumExperts, uint32_t kBlockTokens, uint32_t kNumTopk,
          bool kHasBias, bool kHasImageTokenMask, bool kHasMask,
          bool kHasFixRoutingMask, bool kHasForceRandom,
          bool kHasPhysicalMap, bool kHasUnmappedTopkIdx, uint32_t kNumExpertGroups = 1>
CUTLASS_GLOBAL void __launch_bounds__(layout::sm90_mega_gate::Config<kBlockTokens, kNumExperts>::kNumThreads, 1)
sm90_bf16_mega_gate_impl(const __grid_constant__ cute::TmaDescriptor tensor_map_x,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_weight,
                          const float* bias, const float* image_bias, const bool* image_token_mask,
                          const bool* mask, const bool* fix_routing_mask, const bool* force_random,
                          const layout::mega_gate::RoutingArgs routing_args, const uint32_t num_tokens,
                          float2* candidate_scores, int* candidate_indices) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    constexpr bool kSplitExperts = kNumExpertGroups > 1;
    using Config = layout::sm90_mega_gate::Config<kBlockTokens, kNumExperts, kSplitExperts>;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using WGMMA = typename mma::sm90::BF16MMASelector<Config::kExpertsPerGroup>::type;
    constexpr auto kMajor = cute::UMMA::Major::K;
    constexpr auto BLOCK_K = Config::kBlockK;
    constexpr auto kAtomK = Config::kSwizzle / sizeof(cutlass::bfloat16_t);
    constexpr auto kNumStages = Config::kNumStages;
    constexpr auto kNumMathThreads = Config::kNumMathThreads;
    constexpr auto kNumMathWarps = kNumMathThreads / 32;
    DG_STATIC_ASSERT(SHAPE_K % BLOCK_K == 0 and kNumStages >= 2, "Invalid gate pipeline");
    DG_STATIC_ASSERT(kBlockTokens <= 64 and kBlockTokens % 16 == 0, "Invalid token tile");
    DG_STATIC_ASSERT(not kSplitExperts or (kNumExperts == 128 and not kHasFixRoutingMask), "Invalid split-N gate");

    const auto lane_idx = ptx::get_lane_idx();
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto token_base = blockIdx.x * kBlockTokens;
    const auto expert_base = kSplitExperts ? blockIdx.y * kNumExperts : 0;
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto smem_x = reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer);
    auto smem_weight = reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + kNumStages * Config::kXStageBytes);
    auto full_barriers = reinterpret_cast<Barrier*>(smem_buffer + Config::kBarrierOffset);
    auto empty_barriers = full_barriers + kNumStages;

    if (warp_idx == kNumMathWarps and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_x);
        cute::prefetch_tma_descriptor(&tensor_map_weight);
        #pragma unroll
        for (uint32_t stage = 0; stage < kNumStages; ++ stage) {
            full_barriers[stage].init(1);
            empty_barriers[stage].init(kNumMathWarps);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();
    cudaGridDependencySynchronize();

    if (warp_idx >= kNumMathWarps) {
        cutlass::arch::warpgroup_reg_dealloc<32>();
        if (warp_idx == kNumMathWarps and cute::elect_one_sync()) {
            for (uint32_t k_block = 0; k_block < SHAPE_K / BLOCK_K; ++ k_block) {
                const auto stage = k_block % kNumStages;
                const auto phase = k_block / kNumStages & 1;
                empty_barriers[stage].wait(phase ^ 1);
                tma::copy<BLOCK_K, Config::kMmaTokens, Config::kSwizzle, cutlass::bfloat16_t>(
                    &tensor_map_x, &full_barriers[stage], smem_x + stage * Config::kMmaTokens * BLOCK_K,
                    k_block * BLOCK_K, token_base);
                #pragma unroll
                for (uint32_t group = 0; group < Config::kNumMathGroups; ++ group)
                    tma::copy<BLOCK_K, Config::kExpertsPerGroup, Config::kSwizzle, cutlass::bfloat16_t>(
                        &tensor_map_weight, &full_barriers[stage],
                        smem_weight + (stage * kNumExperts + group * Config::kExpertsPerGroup) * BLOCK_K,
                        k_block * BLOCK_K, expert_base + group * Config::kExpertsPerGroup);
                full_barriers[stage].arrive_and_expect_tx(Config::kStageBytes);
            }
        }
    } else {
        cutlass::arch::warpgroup_reg_alloc<(kSplitExperts ? 128 : (Config::kNumMathGroups == 1 ? 248 : 232))>();
        const auto math_group = warp_idx / 4;
        auto x_desc = mma::sm90::make_gmma_desc<kMajor, Config::kMmaTokens, kAtomK, Config::kSwizzle>(smem_x, 0, 0);
        auto weight_desc = mma::sm90::make_gmma_desc<kMajor, kNumExperts, kAtomK, Config::kSwizzle>(
            smem_weight, math_group * Config::kExpertsPerGroup, 0);
        const auto x_base = x_desc.reg32_[0], weight_base = weight_desc.reg32_[0];
        float accum[WGMMA::kNumAccum] = {0};

        for (uint32_t k_block = 0; k_block < SHAPE_K / BLOCK_K; ++ k_block) {
            const auto stage = k_block % kNumStages;
            const auto phase = k_block / kNumStages & 1;
            full_barriers[stage].wait(phase);
            #pragma unroll
            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                ptx::warpgroup_fence_operand(accum[i]);
            ptx::warpgroup_arrive();
            #pragma unroll
            for (uint32_t k = 0; k < BLOCK_K; k += WGMMA::K) {
                x_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<kMajor, Config::kMmaTokens, kAtomK, Config::kSwizzle, cutlass::bfloat16_t>(
                    x_base + stage * Config::kXStageBytes / 16, 0, k % kAtomK, k / kAtomK * Config::kMmaTokens * kAtomK);
                weight_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<kMajor, kNumExperts, kAtomK, Config::kSwizzle, cutlass::bfloat16_t>(
                    weight_base + stage * Config::kWeightStageBytes / 16, 0, k % kAtomK, k / kAtomK * kNumExperts * kAtomK);
                WGMMA::wgmma(x_desc, weight_desc, accum, 1);
            }
            ptx::warpgroup_commit_batch();
            #pragma unroll
            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                ptx::warpgroup_fence_operand(accum[i]);
            if constexpr (kSplitExperts) {
                // Keep the current MMA in flight while releasing the preceding
                // operand stage. The producer can refill it during the next MMA.
                ptx::warpgroup_wait<1>();
                if (k_block > 0 and lane_idx == 0)
                    empty_barriers[(k_block - 1) % kNumStages].arrive();
            } else {
                ptx::warpgroup_wait<0>();
                if (lane_idx == 0)
                    empty_barriers[stage].arrive();
            }
        }
        if constexpr (kSplitExperts) {
            ptx::warpgroup_wait<0>();
            if (lane_idx == 0)
                empty_barriers[(SHAPE_K / BLOCK_K - 1) % kNumStages].arrive();
        }

        // All consumers must finish their WGMMA reads before the operand storage
        // can become score storage. Every TMA has completed by the last full wait.
        cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);
        auto scores = reinterpret_cast<float*>(smem_buffer);
        const auto row = warp_idx % 4 * 16 + lane_idx / 4;
        const auto col = math_group * Config::kExpertsPerGroup + lane_idx % 4 * 2;
        #pragma unroll
        for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
            if (row < kBlockTokens)
                ptx::st_shared(reinterpret_cast<float2*>(scores + row * kNumExperts + col + i * 8),
                               make_float2(accum[i * 4], accum[i * 4 + 1]));
            if (row + 8 < kBlockTokens)
                ptx::st_shared(reinterpret_cast<float2*>(scores + (row + 8) * kNumExperts + col + i * 8),
                               make_float2(accum[i * 4 + 2], accum[i * 4 + 3]));
        }
        cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

        if constexpr (not kSplitExperts) {
            // One warp owns a token, including all optional routing controls.
            for (uint32_t token = warp_idx; token < kBlockTokens and token_base + token < num_tokens; token += kNumMathWarps) {
                epilogue::sm90_mega_gate::route_token_from_logits<kNumExperts, kNumTopk,
                    kHasBias, kHasImageTokenMask, kHasMask, kHasFixRoutingMask, kHasForceRandom,
                    kHasPhysicalMap, kHasUnmappedTopkIdx>(scores + token * kNumExperts,
                        bias, image_bias, image_token_mask, mask, fix_routing_mask, force_random, routing_args, token_base + token);
            }
        }
    }
    if constexpr (kSplitExperts) {
        cutlass::arch::NamedBarrier::sync(Config::kNumThreads, 1);
        if (warp_idx >= kNumMathWarps)
            cutlass::arch::warpgroup_reg_alloc<64>();
        const auto scores = reinterpret_cast<const float*>(smem_buffer);
        for (uint32_t token = warp_idx; token < kBlockTokens and token_base + token < num_tokens; token += Config::kNumThreads / 32) {
            epilogue::sm90_mega_gate::store_group_topk<kNumTopk, kNumExpertGroups, kHasBias, kHasImageTokenMask>(
                scores + token * kNumExperts, bias, image_bias, image_token_mask,
                candidate_scores, candidate_indices, token_base + token, blockIdx.y, routing_args.num_routed_experts);
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "SM90 MegaGate requires SM90 or later");
#endif
}

template <uint32_t kNumTopk, uint32_t kNumExpertGroups, bool kHasMask, bool kHasForceRandom,
          bool kHasPhysicalMap, bool kHasUnmappedTopkIdx>
CUTLASS_GLOBAL void __launch_bounds__(128)
sm90_bf16_mega_gate_merge(const float2* candidate_scores, const int* candidate_indices,
                          const bool* mask, const bool* force_random,
                          const layout::mega_gate::RoutingArgs routing_args, const uint32_t num_tokens) {
    using namespace epilogue::sm90_mega_gate;
    cudaGridDependencySynchronize();
    const auto token = blockIdx.x * 4 + threadIdx.x / 32;
    const auto lane_idx = ptx::get_lane_idx();
    if (token >= num_tokens)
        return;
    if (kHasMask and not mask[token]) {
        store_token_outputs<kNumTopk, kHasUnmappedTopkIdx>(routing_args, token, lane_idx, -1, 0.0f, -1);
    } else if (kHasForceRandom and force_random[token]) {
        store_random_topk_token<kNumTopk, kHasUnmappedTopkIdx, kHasPhysicalMap>(routing_args, token, lane_idx);
    } else {
        merge_group_topk<kNumTopk, kNumExpertGroups, kHasPhysicalMap, kHasUnmappedTopkIdx>(
            candidate_scores, candidate_indices, routing_args, token);
    }
}

// Decode batches have too few token tiles to occupy the GPU with WGMMA. Spread
// their dot products across experts, then route the resulting small logits matrix.
template <uint32_t SHAPE_K>
CUTLASS_GLOBAL void __launch_bounds__(128)
sm90_bf16_mega_gate_gemv(const cutlass::bfloat16_t* x, const cutlass::bfloat16_t* weight,
                        float* logits, const uint32_t num_experts) {
    cudaGridDependencySynchronize();
    const auto lane_idx = ptx::get_lane_idx();
    const auto expert = blockIdx.x * 4 + threadIdx.x / 32;
    const auto x_row = x + static_cast<uint64_t>(blockIdx.y) * SHAPE_K;
    const auto weight_row = weight + static_cast<uint64_t>(expert) * SHAPE_K;
    float accum[8] = {0};
    for (uint32_t k = lane_idx * 8; k < SHAPE_K; k += 256) {
        const auto x_values = *reinterpret_cast<const uint4*>(x_row + k);
        const auto w_values = *reinterpret_cast<const uint4*>(weight_row + k);
        #pragma unroll
        for (uint32_t i = 0; i < 8; ++ i)
            accum[i] = fmaf(static_cast<float>(reinterpret_cast<const cutlass::bfloat16_t*>(&x_values)[i]),
                            static_cast<float>(reinterpret_cast<const cutlass::bfloat16_t*>(&w_values)[i]), accum[i]);
    }
    auto sum = 0.0f;
    #pragma unroll
    for (uint32_t i = 0; i < 8; ++ i)
        sum += accum[i];
    sum = math::warp_reduce_sum(sum);
    if (lane_idx == 0)
        logits[blockIdx.y * num_experts + expert] = sum;
}

template <uint32_t kNumExperts, uint32_t kNumTopk,
          bool kHasBias, bool kHasImageTokenMask, bool kHasMask,
          bool kHasFixRoutingMask, bool kHasForceRandom,
          bool kHasPhysicalMap, bool kHasUnmappedTopkIdx>
CUTLASS_GLOBAL void __launch_bounds__(128)
sm90_bf16_mega_gate_route(const float* logits,
                         const float* bias, const float* image_bias, const bool* image_token_mask,
                         const bool* mask, const bool* fix_routing_mask, const bool* force_random,
                         const layout::mega_gate::RoutingArgs routing_args, const uint32_t num_tokens) {
    cudaGridDependencySynchronize();
    const auto token = blockIdx.x * 4 + threadIdx.x / 32;
    if (token < num_tokens)
        epilogue::sm90_mega_gate::route_token_from_logits<kNumExperts, kNumTopk,
            kHasBias, kHasImageTokenMask, kHasMask, kHasFixRoutingMask, kHasForceRandom,
            kHasPhysicalMap, kHasUnmappedTopkIdx>(logits + token * routing_args.num_routed_experts,
                bias, image_bias, image_token_mask, mask, fix_routing_mask, force_random, routing_args, token);
}

} // namespace deep_gemm
