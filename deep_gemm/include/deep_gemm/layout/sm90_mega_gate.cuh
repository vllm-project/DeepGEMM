#pragma once

#include <deep_gemm/layout/mega_gate.cuh>

namespace deep_gemm::layout::sm90_mega_gate {

// The operands and the logits have disjoint lifetimes. Reuse the TMA pipeline
// storage for the complete row of logits, keeping routing entirely inside a CTA.
template <uint32_t kBlockTokens, uint32_t kNumExperts>
struct Config {
    static constexpr uint32_t kBlockK = 64;
    static constexpr uint32_t kMmaTokens = 64;
    static constexpr uint32_t kSwizzle = 128;
    static constexpr uint32_t kNumMathGroups = math::constexpr_ceil_div(kNumExperts, 256u);
    static constexpr uint32_t kExpertsPerGroup = kNumExperts / kNumMathGroups;
    static constexpr uint32_t kNumMathThreads = kNumMathGroups * 128;
    static constexpr uint32_t kNumThreads = kNumMathThreads + 128;
    // Even short token tiles need all 64 WGMMA rows initialized: the unused
    // rows must not overlap another stage that TMA is concurrently writing.
    static constexpr uint32_t kXStageBytes = kMmaTokens * kBlockK * sizeof(cutlass::bfloat16_t);
    static constexpr uint32_t kWeightStageBytes = kNumExperts * kBlockK * sizeof(cutlass::bfloat16_t);
    static constexpr uint32_t kStageBytes = kXStageBytes + kWeightStageBytes;
    static constexpr uint32_t kMaxStages = (232448u - 1024u) / kStageBytes;
    static constexpr uint32_t kNumStages = kMaxStages < 4 ? kMaxStages : 4;
    static constexpr uint32_t kPipelineBytes = kNumStages * kStageBytes;
    static constexpr uint32_t kScoreBytes = kBlockTokens * kNumExperts * sizeof(float);
    static constexpr uint32_t kBarrierOffset = kPipelineBytes > kScoreBytes ? kPipelineBytes : kScoreBytes;
    static constexpr uint32_t kSmemBytes = kBarrierOffset + 2 * kNumStages * sizeof(cutlass::arch::ClusterTransactionBarrier);
};

} // namespace deep_gemm::layout::sm90_mega_gate
