#pragma once

#include <format>

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../../runtime/runtime.hpp"
#include "../heuristics/mega_moe.hpp"

namespace deep_gemm {

static void sm120_fp8_fp4_mega_moe(
    const torch::stable::Tensor& output,
    const torch::stable::Tensor& l1_acts, const torch::stable::Tensor& l1_acts_sf,
    const torch::stable::Tensor& l1_weights, const torch::stable::Tensor& l2_weights,
    const torch::stable::Tensor& l1_weight_scales, const torch::stable::Tensor& l2_weight_scales,
    const std::optional<torch::stable::Tensor>& cumulative_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int rank_idx, const int num_max_tokens_per_rank,
    const int num_experts_per_rank, const int num_tokens, const int num_tokens_for_block_m, const int num_topk,
    const int hidden, const int intermediate,
    const float activation_clamp, const float activation_alpha, const float activation_beta,
    const bool fast_math
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_ring_tokens = static_cast<int>(l1_acts.size(0));
    const auto num_sf_ring_tokens = static_cast<int>(l1_acts_sf.size(0));
    DG_HOST_ASSERT(num_tokens_for_block_m >= 0);
    const auto [cluster_size, block_m, store_block_m_l1, store_block_m_l2, block_k, num_epilogue_threads] =
        get_block_config_for_mega_moe(num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
                                     num_tokens_for_block_m, MmaKind::MXFP8FP4);
    const auto num_ctas_per_sm = get_num_ctas_per_sm_for_sm120_mega_moe(block_m);
    const auto num_threads = block_m == 16 ? 128 : 256;
    const auto num_ctas = std::min({get_num_sms_for_mega_moe() * num_ctas_per_sm,
                                    num_ring_tokens / block_m, num_sf_ring_tokens / 128});
    DG_HOST_ASSERT(num_ctas > 0);
    const auto num_ring_slots = std::min(num_ctas, num_ring_tokens / (block_m * (num_ranks > 1 ? 2 : 1)));
    DG_HOST_ASSERT(num_ring_slots > 0);
    const auto buffers = layout::MegaMoEBuffer(
        reinterpret_cast<void*>(sym_buffer_ptrs[rank_idx]), hidden, intermediate,
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
        num_ring_tokens, num_sf_ring_tokens, true);
    const auto kernel = jit->compile("sm120_fp8_fp4_mega_moe", std::format(R"(
#include <deep_gemm/impls/sm120_fp8_fp4_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_fp8_fp4_mega_moe_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)", hidden, intermediate, num_experts, num_topk, num_ctas, num_ring_slots, num_ranks, block_m, num_ctas_per_sm,
        fast_math ? "true" : "false"));
    int* stats_ptr = cumulative_recv_stats.has_value() ? cumulative_recv_stats->mutable_data_ptr<int>() : nullptr;
    jit->launch(kernel, {
        .grid_dim = dim3(num_ctas, 1, 1),
        .block_dim = dim3(num_threads, 1, 1),
        .cooperative = true,
    }, output.mutable_data_ptr(),
       l1_weights.const_data_ptr(), l2_weights.const_data_ptr(),
       l1_weight_scales.const_data_ptr(), l2_weight_scales.const_data_ptr(),
       l1_weight_scales.stride(0), l1_weight_scales.stride(2),
       l2_weight_scales.stride(0), l2_weight_scales.stride(2),
       stats_ptr, static_cast<uint32_t>(num_tokens),
       activation_clamp, activation_alpha, activation_beta,
       buffers, layout::SymBuffer<>(sym_buffer_ptrs, rank_idx));
}

}
