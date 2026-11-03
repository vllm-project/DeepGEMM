#pragma once

#include <cmath>
#include <cstdint>

#include <curand_kernel.h>
#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/layout/mega_gate.cuh>
#include <deep_gemm/ptx/utils.cuh>

// Keep SM90 routing local so upstream SM100 helpers can be updated independently.
namespace deep_gemm::epilogue::sm90_mega_gate {

using namespace layout::mega_gate;

CUTLASS_DEVICE float sqrt_softplus(const float& value) {
    const auto softplus_precise = log1pf(expf(value));
    const auto softplus = value > 20.0f ? value : softplus_precise;
    return sqrtf(softplus);
}

template <uint32_t kNumValues>
CUTLASS_DEVICE void sqrt_softplus(float (&values)[kNumValues]) {
    float exponentials[kNumValues], softplus[kNumValues];
    #pragma unroll
    for (uint32_t value_idx = 0; value_idx < kNumValues; ++ value_idx)
        exponentials[value_idx] = expf(values[value_idx]);
    #pragma unroll
    for (uint32_t value_idx = 0; value_idx < kNumValues; ++ value_idx)
        softplus[value_idx] = log1pf(exponentials[value_idx]);
    #pragma unroll
    for (uint32_t value_idx = 0; value_idx < kNumValues; ++ value_idx)
        values[value_idx] = sqrtf(values[value_idx] > 20.0f ? values[value_idx] : softplus[value_idx]);
}

CUTLASS_DEVICE void warp_reduce_best(const float& score, int& expert_idx) {
    const auto best_score = math::warp_reduce_max(score);
    // Ties resolve to the smallest expert index
    const auto tied_expert_idx = score == best_score and expert_idx >= 0 ?
        static_cast<uint32_t>(expert_idx) : 0xffffffffu;
    expert_idx = static_cast<int>(__reduce_min_sync(0xffffffff, tied_expert_idx));
}

CUTLASS_DEVICE void compare_swap_best_first(float& lhs_score, int& lhs_expert_idx, float& rhs_score, int& rhs_expert_idx) {
    if (rhs_score > lhs_score or (rhs_score == lhs_score and rhs_expert_idx < lhs_expert_idx)) {
        const auto score = lhs_score;
        const auto expert_idx = lhs_expert_idx;
        lhs_score = rhs_score, lhs_expert_idx = rhs_expert_idx;
        rhs_score = score, rhs_expert_idx = expert_idx;
    }
}

CUTLASS_DEVICE uint32_t sort4_best_first(float& score_0, float& score_1, float& score_2, float& score_3) {
    int idx_0 = 0, idx_1 = 1, idx_2 = 2, idx_3 = 3;
    compare_swap_best_first(score_0, idx_0, score_1, idx_1);
    compare_swap_best_first(score_2, idx_2, score_3, idx_3);
    compare_swap_best_first(score_0, idx_0, score_2, idx_2);
    compare_swap_best_first(score_1, idx_1, score_3, idx_3);
    compare_swap_best_first(score_1, idx_1, score_2, idx_2);
    return static_cast<uint32_t>(idx_0 | idx_1 << 2 | idx_2 << 4 | idx_3 << 6);
}

template <uint32_t kNumExpertWaves, uint32_t kNumTopk>
CUTLASS_DEVICE void select_warp_topk(float (&scores)[kNumExpertWaves * kNumExpertsPerLaneVector], int& selected_expert_idx) {
    DG_STATIC_ASSERT(kNumExpertsPerLaneVector == 4, "The lane vector is sorted with a 4-element network");
    uint32_t permutations[kNumExpertWaves];
    #pragma unroll
    for (uint32_t expert_wave_idx = 0; expert_wave_idx < kNumExpertWaves; ++ expert_wave_idx) {
        const auto local_expert_base_idx = expert_wave_idx * kNumExpertsPerLaneVector;
        permutations[expert_wave_idx] = sort4_best_first(scores[local_expert_base_idx + 0],
                                                         scores[local_expert_base_idx + 1],
                                                         scores[local_expert_base_idx + 2],
                                                         scores[local_expert_base_idx + 3]);
    }

    // Every round, each lane offers the best unconsumed candidate of each wave (a 3-bit cursor into the
    // 2-bit permutation), the warp picks the best one and only the winning lane advances that cursor
    const auto lane_idx = ptx::get_lane_idx();
    uint32_t packed_cursors = 0;
    #pragma unroll
    for (uint32_t output_idx = 0; output_idx < kNumTopk; ++ output_idx) {
        auto best_score = -cute::numeric_limits<float>::infinity();
        int best_expert_idx = -1;
        uint32_t best_expert_wave_idx = 0;
        #pragma unroll
        for (uint32_t expert_wave_idx = 0; expert_wave_idx < kNumExpertWaves; ++ expert_wave_idx) {
            const auto cursor = packed_cursors >> (expert_wave_idx * 3) & 7;
            const auto local_expert_base_idx = expert_wave_idx * kNumExpertsPerLaneVector;
            const auto candidate_score = cursor == 4 ? -cute::numeric_limits<float>::infinity() :
                                         cursor & 2 ? (cursor & 1 ? scores[local_expert_base_idx + 3] : scores[local_expert_base_idx + 2]) :
                                                      (cursor & 1 ? scores[local_expert_base_idx + 1] : scores[local_expert_base_idx + 0]);
            const auto candidate_offset = permutations[expert_wave_idx] >> (cursor * 2) & 3;
            const auto candidate_expert_idx = static_cast<int>(expert_wave_idx * kNumExpertsPerWave +
                                                               lane_idx * kNumExpertsPerLaneVector + candidate_offset);
            if (candidate_score > best_score) {
                best_score = candidate_score;
                best_expert_idx = candidate_expert_idx;
                best_expert_wave_idx = expert_wave_idx;
            }
        }
        const auto local_best_expert_idx = best_expert_idx;
        warp_reduce_best(best_score, best_expert_idx);
        if (local_best_expert_idx == best_expert_idx)
            packed_cursors += 1u << (best_expert_wave_idx * 3);
        if (lane_idx == output_idx)
            selected_expert_idx = best_expert_idx;
    }
}

template <uint32_t kNumTopk, bool kHasUnmappedTopkIdx>
CUTLASS_DEVICE void store_token_outputs(const RoutingArgs& args, const uint32_t& global_token_idx, const uint32_t& lane_idx,
                                        const int64_t& physical_expert_idx, const float& weight, const int64_t& logical_expert_idx) {
    const auto num_physical_topk = kNumTopk + args.num_shared_experts;
    if (lane_idx < num_physical_topk) {
        const auto output_idx = global_token_idx * num_physical_topk + lane_idx;
        args.topk_idx[output_idx] = physical_expert_idx;
        args.topk_weights[output_idx] = weight;
    }
    if constexpr (kHasUnmappedTopkIdx) {
        if (lane_idx < kNumTopk)
            args.unmapped_topk_idx[global_token_idx * args.unmapped_topk_idx_stride + lane_idx] = logical_expert_idx;
    }
}

template <uint32_t kNumTopk, bool kHasUnmappedTopkIdx, bool kHasPhysicalMap>
CUTLASS_DEVICE void store_topk_token(const RoutingArgs& args, const int* routed_logical_count,
                                     const uint32_t& global_token_idx, const uint32_t& lane_idx,
                                     int selected_expert_idx, const float& selected_unbiased_score) {
    DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k slots exceed a warp");
    const auto num_physical_topk = kNumTopk + args.num_shared_experts;
    // The shared experts follow the routed ones and take the slots after the top-k
    if (lane_idx >= kNumTopk and lane_idx < num_physical_topk)
        selected_expert_idx = static_cast<int>(lane_idx + args.num_routed_experts - kNumTopk);

    auto physical_expert_idx = selected_expert_idx;
    if constexpr (kHasPhysicalMap) {
        if (lane_idx < num_physical_topk) {
            const auto logical_expert_idx = static_cast<uint32_t>(selected_expert_idx);
            // The SMEM cache only covers the routed experts
            const auto num_duplicates = static_cast<uint32_t>((lane_idx < kNumTopk ? routed_logical_count : args.logical_count)[logical_expert_idx]);
            DG_TRAP_ONLY_DEVICE_ASSERT(num_duplicates > 0 and num_duplicates <= args.num_duplicate_experts);
            // Spread the tokens of a logical expert over its duplicates, differently on every rank
            const auto duplicate_idx = (args.rank_idx + global_token_idx * 23333u) % num_duplicates;
            physical_expert_idx = args.to_physical_map[logical_expert_idx * args.num_duplicate_experts + duplicate_idx];
        }
    }

    // The smallest shuffle width covering the routed lanes, the other lanes hold zero
    constexpr uint32_t kSumWidth = kNumTopk <= 8 ? 8 : (kNumTopk <= 16 ? 16 : 32);
    const auto topk_sum = math::warp_reduce_sum<kSumWidth>(selected_unbiased_score) + 1e-20f;
    const auto selected_weight = lane_idx < kNumTopk ? selected_unbiased_score / topk_sum * args.routed_scaling_factor : 1.0f;
    store_token_outputs<kNumTopk, kHasUnmappedTopkIdx>(args, global_token_idx, lane_idx,
                                                       physical_expert_idx, selected_weight, selected_expert_idx);
}

template <uint32_t kNumTopk, bool kHasUnmappedTopkIdx, bool kHasPhysicalMap>
CUTLASS_DEVICE void store_random_topk_token(const RoutingArgs& args, const uint32_t& global_token_idx, const uint32_t& lane_idx) {
    const auto num_physical_topk = kNumTopk + args.num_shared_experts;
    auto num_physical_experts = args.num_routed_experts + args.num_shared_experts;
    if constexpr (kHasPhysicalMap) {
        auto local_count = 0u;
        for (uint32_t logical_expert_idx = lane_idx; logical_expert_idx < num_physical_experts; logical_expert_idx += 32)
            local_count += static_cast<uint32_t>(args.logical_count[logical_expert_idx]);
        num_physical_experts = __reduce_add_sync(0xffffffff, local_count);
    }
    DG_TRAP_ONLY_DEVICE_ASSERT(num_physical_experts >= num_physical_topk);

    // Sample `num_physical_topk` distinct experts: lane `i` draws from `n - i`, then ranks below every earlier draw
    // that is not larger shift up by one
    curandStatePhilox4_32_10_t rng_state;
    curand_init(args.rank_idx, global_token_idx * 32 + lane_idx, 0, &rng_state);
    const auto random_weight = curand_uniform(&rng_state);
    constexpr uint32_t kInvalidIdx = 0xffffffffu;
    auto candidate_idx = lane_idx < num_physical_topk ?
        curand(&rng_state) % (num_physical_experts - lane_idx) : kInvalidIdx;
    auto selected_expert_idx = kInvalidIdx;
    for (uint32_t output_idx = 0; output_idx < num_physical_topk; ++ output_idx) {
        const auto min_candidate_idx = __reduce_min_sync(0xffffffff, candidate_idx);
        const auto min_lane_idx = __reduce_min_sync(0xffffffff, candidate_idx == min_candidate_idx ? lane_idx : kInvalidIdx);
        if (candidate_idx != kInvalidIdx) {
            if (min_lane_idx == lane_idx) {
                selected_expert_idx = candidate_idx;
                candidate_idx = kInvalidIdx;
            } else if (candidate_idx >= min_candidate_idx and min_lane_idx < lane_idx) {
                ++ candidate_idx;
            }
        }
    }
    store_token_outputs<kNumTopk, kHasUnmappedTopkIdx>(args, global_token_idx, lane_idx,
                                                       static_cast<int64_t>(selected_expert_idx), random_weight, -1);
}

// Route one complete row of FP32 logits, in either shared or global memory.
// The caller assigns all 32 lanes of a warp to the same token.
template <uint32_t kNumExperts, uint32_t kNumTopk,
          bool kHasBias, bool kHasImageTokenMask, bool kHasMask,
          bool kHasFixRoutingMask, bool kHasForceRandom,
          bool kHasPhysicalMap, bool kHasUnmappedTopkIdx>
CUTLASS_DEVICE void route_token_from_logits(const float* row_scores,
                                             const float* bias, const float* image_bias, const bool* image_token_mask,
                                             const bool* mask, const bool* fix_routing_mask, const bool* force_random,
                                             const RoutingArgs& routing_args, const uint32_t global_token) {
    const auto lane_idx = ptx::get_lane_idx();
    constexpr auto kNumExpertWaves = kNumExperts / kNumExpertsPerWave;
    if (kHasMask and not mask[global_token]) {
        store_token_outputs<kNumTopk, kHasUnmappedTopkIdx>(routing_args, global_token, lane_idx, -1, 0.0f, -1);
        return;
    }
    if (kHasForceRandom and force_random[global_token]) {
        store_random_topk_token<kNumTopk, kHasUnmappedTopkIdx, kHasPhysicalMap>(routing_args, global_token, lane_idx);
        return;
    }

    int selected_expert = -1;
    if (kHasFixRoutingMask and fix_routing_mask[global_token]) {
        if (lane_idx < kNumTopk) {
            const auto fixed_expert = routing_args.unmapped_topk_idx[
                global_token * routing_args.unmapped_topk_idx_stride + lane_idx];
            DG_TRAP_ONLY_DEVICE_ASSERT(fixed_expert >= 0 and fixed_expert < routing_args.num_routed_experts);
            selected_expert = static_cast<int>(fixed_expert);
        }
    } else {
        const float* ranking_bias = kHasBias ? bias : nullptr;
        if constexpr (kHasImageTokenMask)
            ranking_bias = image_token_mask[global_token] ? image_bias : ranking_bias;
        float ranking[kNumExpertWaves * kNumExpertsPerLaneVector];
        #pragma unroll
        for (uint32_t wave = 0; wave < kNumExpertWaves; ++ wave) {
            const auto expert = wave * kNumExpertsPerWave + lane_idx * kNumExpertsPerLaneVector;
            auto& values = *reinterpret_cast<float (*)[kNumExpertsPerLaneVector]>(ranking + wave * kNumExpertsPerLaneVector);
            *reinterpret_cast<float4*>(values) = expert < routing_args.num_routed_experts ?
                *reinterpret_cast<const float4*>(row_scores + expert) : make_float4(0, 0, 0, 0);
            if constexpr (kHasBias or kHasImageTokenMask)
                sqrt_softplus(values);
            // Contiguous bias views may have a storage offset and only 4-byte
            // alignment. Scalar loads preserve the public API's acceptance of them.
            #pragma unroll
            for (uint32_t i = 0; i < kNumExpertsPerLaneVector; ++ i)
                values[i] = expert < routing_args.num_routed_experts ?
                    values[i] + (ranking_bias ? ranking_bias[expert + i] : 0.0f) : -cute::numeric_limits<float>::infinity();
        }
        select_warp_topk<kNumExpertWaves, kNumTopk>(ranking, selected_expert);
        if (lane_idx < kNumTopk)
            selected_expert = cute::min(cute::max(selected_expert, 0), static_cast<int>(routing_args.num_routed_experts) - 1);
    }
    const auto selected_score = lane_idx < kNumTopk ? sqrt_softplus(row_scores[selected_expert]) : 0.0f;
    store_topk_token<kNumTopk, kHasUnmappedTopkIdx, kHasPhysicalMap>(
        routing_args, routing_args.logical_count, global_token, lane_idx, selected_expert, selected_score);
}

} // namespace deep_gemm::epilogue::sm90_mega_gate
