#pragma once

#include <deep_gemm/epilogue/mega_gate.cuh>

namespace deep_gemm::epilogue::sm90_mega_gate {

using namespace epilogue::mega_gate;

// The union of each expert group's top-k contains the global top-k. Keep both
// ranking and unbiased scores: bias affects selection, but not normalization.
template <uint32_t kNumTopk, uint32_t kNumExpertGroups, bool kHasBias, bool kHasImageTokenMask>
CUTLASS_DEVICE void store_group_topk(const float* row_scores,
                                      const float* bias, const float* image_bias, const bool* image_token_mask,
                                      float2* candidate_scores, int* candidate_indices,
                                      const uint32_t global_token, const uint32_t expert_group,
                                      const uint32_t num_experts) {
    const auto lane_idx = ptx::get_lane_idx();
    const auto expert_base = expert_group * kNumExpertsPerWave;
    const float* ranking_bias = kHasBias ? bias : nullptr;
    if constexpr (kHasImageTokenMask)
        ranking_bias = image_token_mask[global_token] ? image_bias : ranking_bias;

    float ranking[kNumExpertsPerLaneVector];
    *reinterpret_cast<float4*>(ranking) = *reinterpret_cast<const float4*>(row_scores + lane_idx * kNumExpertsPerLaneVector);
    if constexpr (kHasBias or kHasImageTokenMask)
        sqrt_softplus(ranking);
    #pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLaneVector; ++ i) {
        const auto expert = expert_base + lane_idx * kNumExpertsPerLaneVector + i;
        ranking[i] = expert < num_experts ? ranking[i] + (ranking_bias ? ranking_bias[expert] : 0.0f) :
                     -cute::numeric_limits<float>::infinity();
    }
    int selected_expert = -1;
    select_warp_topk<1, kNumTopk>(ranking, selected_expert);
    if (lane_idx < kNumTopk) {
        const auto valid = selected_expert >= 0 and expert_base + selected_expert < num_experts;
        const auto expert = valid ? static_cast<int>(expert_base) + selected_expert : 0x7fffffff;
        const auto score = valid ? sqrt_softplus(row_scores[selected_expert]) : 0.0f;
        const auto rank = valid ? ((kHasBias or kHasImageTokenMask) ? score : row_scores[selected_expert]) +
                                  (ranking_bias ? ranking_bias[expert] : 0.0f) : -cute::numeric_limits<float>::infinity();
        const auto offset = (global_token * kNumExpertGroups + expert_group) * kNumTopk + lane_idx;
        candidate_scores[offset] = make_float2(rank, score);
        candidate_indices[offset] = expert;
    }
}

template <uint32_t kNumTopk, uint32_t kNumExpertGroups,
          bool kHasPhysicalMap, bool kHasUnmappedTopkIdx>
CUTLASS_DEVICE void merge_group_topk(const float2* candidate_scores, const int* candidate_indices,
                                      const RoutingArgs& routing_args, const uint32_t global_token) {
    const auto lane_idx = ptx::get_lane_idx();
    constexpr auto kNumCandidates = kNumTopk * kNumExpertGroups;
    constexpr auto kNumWaves = math::constexpr_ceil_div(kNumCandidates, 32u);
    float ranking[kNumWaves], scores[kNumWaves];
    int experts[kNumWaves];
    #pragma unroll
    for (uint32_t wave = 0; wave < kNumWaves; ++ wave) {
        const auto slot = wave * 32 + lane_idx;
        const auto offset = global_token * kNumCandidates + slot;
        const auto values = slot < kNumCandidates ? candidate_scores[offset] :
                            make_float2(-cute::numeric_limits<float>::infinity(), 0.0f);
        ranking[wave] = values.x;
        scores[wave] = values.y;
        experts[wave] = slot < kNumCandidates ? candidate_indices[offset] : 0x7fffffff;
    }

    int selected_expert = -1;
    float selected_score = 0.0f;
    #pragma unroll
    for (uint32_t output_idx = 0; output_idx < kNumTopk; ++ output_idx) {
        auto best_rank = -cute::numeric_limits<float>::infinity();
        int best_expert = 0x7fffffff;
        float best_score = 0.0f;
        #pragma unroll
        for (uint32_t wave = 0; wave < kNumWaves; ++ wave) {
            if (ranking[wave] > best_rank or (ranking[wave] == best_rank and experts[wave] < best_expert)) {
                best_rank = ranking[wave];
                best_expert = experts[wave];
                best_score = scores[wave];
            }
        }
        const auto local_expert = best_expert;
        warp_reduce_best(best_rank, best_expert);
        const auto winner = __reduce_min_sync(0xffffffff, local_expert == best_expert ? lane_idx : 32u);
        best_score = __shfl_sync(0xffffffff, best_score, winner);
        if (lane_idx == output_idx) {
            selected_expert = best_expert;
            selected_score = best_score;
        }
        #pragma unroll
        for (uint32_t wave = 0; wave < kNumWaves; ++ wave) {
            if (experts[wave] == best_expert) {
                ranking[wave] = -cute::numeric_limits<float>::infinity();
                experts[wave] = 0x7fffffff;
            }
        }
    }
    // Invalid candidates must never reach the physical-map lookup.
    if (lane_idx < kNumTopk)
        selected_expert = cute::min(cute::max(selected_expert, 0), static_cast<int>(routing_args.num_routed_experts) - 1);
    store_topk_token<kNumTopk, kHasUnmappedTopkIdx, kHasPhysicalMap>(
        routing_args, routing_args.logical_count, global_token, lane_idx, selected_expert, selected_score);
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
    float selected_score = 0.0f;
    if (kHasFixRoutingMask and fix_routing_mask[global_token]) {
        if (lane_idx < kNumTopk) {
            const auto fixed_expert = routing_args.unmapped_topk_idx[
                global_token * routing_args.unmapped_topk_idx_stride + lane_idx];
            DG_TRAP_ONLY_DEVICE_ASSERT(fixed_expert >= 0 and fixed_expert < routing_args.num_routed_experts);
            selected_expert = static_cast<int>(fixed_expert);
            selected_score = sqrt_softplus(row_scores[selected_expert]);
        }
    } else {
        const float* ranking_bias = kHasBias ? bias : nullptr;
        if constexpr (kHasImageTokenMask)
            ranking_bias = image_token_mask[global_token] ? image_bias : ranking_bias;
        constexpr auto kNumValues = kNumExpertWaves * kNumExpertsPerLaneVector;
        float scores[kNumValues], bias_values[kNumValues], ranking[kNumValues];
        // Issue all independent loads before scoring any expert wave.
        #pragma unroll
        for (uint32_t wave = 0; wave < kNumExpertWaves; ++ wave) {
            const auto expert = wave * kNumExpertsPerWave + lane_idx * kNumExpertsPerLaneVector;
            *reinterpret_cast<float4*>(scores + wave * kNumExpertsPerLaneVector) = expert < routing_args.num_routed_experts ?
                *reinterpret_cast<const float4*>(row_scores + expert) : make_float4(0, 0, 0, 0);
        }
        // Bias views need only 4-byte alignment, including views with a storage offset.
        #pragma unroll
        for (uint32_t i = 0; i < kNumValues; ++ i) {
            const auto expert = i / kNumExpertsPerLaneVector * kNumExpertsPerWave +
                                lane_idx * kNumExpertsPerLaneVector + i % kNumExpertsPerLaneVector;
            bias_values[i] = expert < routing_args.num_routed_experts and ranking_bias ? ranking_bias[expert] : 0.0f;
        }
        if constexpr (kHasBias or kHasImageTokenMask)
            sqrt_softplus(scores);
        #pragma unroll
        for (uint32_t i = 0; i < kNumValues; ++ i) {
            const auto expert = i / kNumExpertsPerLaneVector * kNumExpertsPerWave +
                                lane_idx * kNumExpertsPerLaneVector + i % kNumExpertsPerLaneVector;
            ranking[i] = expert < routing_args.num_routed_experts ? scores[i] + bias_values[i] :
                         -cute::numeric_limits<float>::infinity();
        }
        select_warp_topk<kNumExpertWaves, kNumTopk>(ranking, selected_expert);
        if (lane_idx < kNumTopk)
            selected_expert = cute::min(cute::max(selected_expert, 0), static_cast<int>(routing_args.num_routed_experts) - 1);
        // Keep the unbiased score in registers instead of reloading the winning logit.
        selected_score = select_warp_expert_value<kNumExpertWaves>(scores, selected_expert);
        if constexpr (not kHasBias and not kHasImageTokenMask)
            selected_score = lane_idx < kNumTopk ? sqrt_softplus(selected_score) : 0.0f;
    }
    store_topk_token<kNumTopk, kHasUnmappedTopkIdx, kHasPhysicalMap>(
        routing_args, routing_args.logical_count, global_token, lane_idx, selected_expert, selected_score);
}

} // namespace deep_gemm::epilogue::sm90_mega_gate
