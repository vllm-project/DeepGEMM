#pragma once

#include <cute/atom/copy_traits_sm100.hpp>

#include <deep_gemm/epilogue/mega_gate.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>

namespace deep_gemm::epilogue::mega_gate {

template <bool kScoreOnStore, uint32_t kScoreTokenStride,
          uint32_t UMMA_M, uint32_t kNumMmaCtas, uint32_t kNumGateWarps>
CUTLASS_DEVICE void sm100_store_tmem_scores_to_global(float* scores, const uint32_t& tmem_base_addr,
                                                      const uint32_t& effective_umma_n,
                                                      const uint32_t& expert_base_idx,
                                                      const uint32_t& mma_cta_rank,
                                                      const uint32_t& gate_warp_idx) {
    // NOTES: with `UMMA_M == 128` on a CTA pair, the 64 experts of a CTA are spread over the two lane halves,
    //        so warps 0/1 and 2/3 of a warpgroup see the same experts for the two token halves
    constexpr uint32_t kNumTmemFragmentRows = 8;
    constexpr bool kPartitionedTokenPaths = UMMA_M == 128 and kNumMmaCtas == 2;
    constexpr uint32_t kNumExpertsPerCta = UMMA_M / kNumMmaCtas;
    const auto warp_idx_in_wg = gate_warp_idx % 4;
    const auto wg_idx = gate_warp_idx / 4;
    constexpr uint32_t kTokenStep = kNumGateWarps / 4 * kNumTmemFragmentRows;
    const auto num_token_cols = kPartitionedTokenPaths ? effective_umma_n / 2 : effective_umma_n;
    const auto token_base_idx = kPartitionedTokenPaths ? warp_idx_in_wg / 2 * num_token_cols : 0u;
    const auto expert_atom_idx = kPartitionedTokenPaths ? warp_idx_in_wg % 2 : warp_idx_in_wg;
    const auto global_expert_idx = expert_base_idx + mma_cta_rank * kNumExpertsPerCta +
                                   expert_atom_idx * 32 + ptx::get_lane_idx();

    // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits
    for (auto token_idx = wg_idx * kNumTmemFragmentRows; token_idx < num_token_cols; token_idx += kTokenStep) {
        uint32_t values[kNumTmemFragmentRows];
        ptx::tmem_load_32dp32b<kNumTmemFragmentRows>(tmem_base_addr + token_idx, values);
        cutlass::arch::fence_view_async_tmem_load();
        const auto row_base_ptr = scores + static_cast<uint64_t>(token_base_idx + token_idx) * kScoreTokenStride +
                                  global_expert_idx;
        #pragma unroll
        for (uint32_t row_idx = 0; row_idx < kNumTmemFragmentRows; ++ row_idx) {
            auto score = __uint_as_float(values[row_idx]);
            if constexpr (kScoreOnStore)
                score = sqrt_softplus(score);
            row_base_ptr[row_idx * kScoreTokenStride] = score;
        }
    }
}

} // namespace deep_gemm::epilogue::mega_gate
