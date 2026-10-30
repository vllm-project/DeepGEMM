#pragma once

#include <format>
#include <torch/csrc/stable/ops.h>
#include "../../utils/torch_compat.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../heuristics/mega_moe.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

static void sm90_fp8_fp4_mega_moe(
    const torch::stable::Tensor& y,
    const torch::stable::Tensor& l1_acts, const torch::stable::Tensor& l1_acts_sf,
    const torch::stable::Tensor& l2_acts, const torch::stable::Tensor& l2_acts_sf,
    const torch::stable::Tensor& shared_l1_acts, const torch::stable::Tensor& shared_l1_acts_sf,
    const torch::stable::Tensor& shared_l2_acts, const torch::stable::Tensor& shared_l2_acts_sf,
    const torch::stable::Tensor& l1_weights, const torch::stable::Tensor& l2_weights,
    const torch::stable::Tensor& l1_weights_sf, const torch::stable::Tensor& l2_weights_sf,
    const torch::stable::Tensor& shared_l1_weights, const torch::stable::Tensor& shared_l2_weights,
    const torch::stable::Tensor& shared_l1_weights_sf, const torch::stable::Tensor& shared_l2_weights_sf,
    const std::optional<torch::stable::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_shared_experts,
    const int& num_tokens, const int& num_tokens_for_block_m, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& clamp_after_silu,
    const float& activation_alpha,
    const float& activation_beta,
    const bool& fast_math
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_ring_tokens = static_cast<int>(l1_acts.size(0));
    const auto num_sf_ring_tokens = static_cast<int>(l1_acts_sf.size(0));
    const auto shared_intermediate_hidden = intermediate_hidden * num_shared_experts;

    // Heuristics
    const auto config = get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens_for_block_m, num_topk, hidden, intermediate_hidden,
        num_ring_tokens, num_sf_ring_tokens,
        MmaKind::MXFP8FP4, num_shared_experts);

    // Make tensormap
    // NOTES: SF are int32 words per 128 K: activations hold FP32 scales, routed weights the `transform_fp4_sf_for_sm90_mega_moe` encoding
    constexpr int kGranK = 32;
    const int sf_smem_outer_dim = config.block_k / (kGranK * 4);
    const auto acts_desc = [&](const torch::stable::Tensor& t, const int& shape_k, const int& shape_m,
                               const int& block_k, const int& block_m, const int& swizzle_mode) {
        return make_tma_2d_desc(t, shape_k, shape_m, block_k, block_m, static_cast<int>(t.stride(-2)), swizzle_mode);
    };
    const auto sf_desc = [&](const torch::stable::Tensor& sf, const int& shape_mn, const int& shape_k,
                             const int& block_mn, const int& num_groups) {
        // NOTES: FP32 activation SF have one scale per 128 K, routed weight SF one `transform_fp4_sf_for_sm90_mega_moe` word per 128 K
        const bool is_fp32 = sf.scalar_type() == torch::headeronly::ScalarType::Float;
        return make_tma_sf_desc(cute::UMMA::Major::MN, sf, shape_mn, shape_k, block_mn, is_fp32 ? kGranK * 4 : kGranK, num_groups, 0, 0, false,
                                sf_smem_outer_dim, static_cast<int>(sf.stride(-1)));
    };
    // NOTES: packed FP4 weights are loaded as raw bytes, 64 bytes per 128-element K block
    const auto fp4_weights_desc = [&](const torch::stable::Tensor& w) {
        return make_tma_weights_3d_desc(torch_compat::view_dtype(w, torch::headeronly::ScalarType::Float8_e4m3fn),
                                        config.block_k / 2, config.load_block_n, config.swizzle_weights_mode);
    };
    // NOTES: L1 output and L2 activations are the same tensor; the post-SwiGLU output has half the N width
    // (`BLOCK_N / 2` per input tile), so the swizzle mode is also halved (128 -> 64)
    const auto tensor_map_l1_acts = acts_desc(l1_acts, hidden, config.num_ring_tokens, config.block_k, config.load_block_m, config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf = sf_desc(l1_acts_sf, config.num_sf_ring_tokens, hidden, config.sf_block_m, 1);
    const auto tensor_map_l1_weights = fp4_weights_desc(l1_weights);
    const auto tensor_map_l1_weights_sf = sf_desc(l1_weights_sf, intermediate_hidden * 2, hidden, config.block_n, num_experts_per_rank);
    const auto tensor_map_l1_output = acts_desc(l2_acts, intermediate_hidden, config.num_ring_tokens, config.block_n / 2, config.block_m, config.swizzle_acts_mode / 2);
    const auto tensor_map_l2_acts = acts_desc(l2_acts, intermediate_hidden, config.num_ring_tokens, config.block_k, config.load_block_m, config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf = sf_desc(l2_acts_sf, config.num_sf_ring_tokens, intermediate_hidden, config.sf_block_m, 1);
    const auto tensor_map_l2_weights = fp4_weights_desc(l2_weights);
    const auto tensor_map_l2_weights_sf = sf_desc(l2_weights_sf, hidden, intermediate_hidden, config.block_n, num_experts_per_rank);

    // Without shared experts, the routed tensormaps stand in
    // NOTES: FP8 shared weights use 128B swizzling, as tokens do
    const bool with_shared = num_shared_experts > 0;
    const auto tensor_map_shared_l1_acts = with_shared ? acts_desc(
        shared_l1_acts, hidden, num_max_tokens_per_rank, config.block_k, config.load_block_m, config.swizzle_acts_mode) : tensor_map_l1_acts;
    const auto tensor_map_shared_l1_acts_sf = with_shared ? sf_desc(
        shared_l1_acts_sf, static_cast<int>(shared_l1_acts_sf.size(0)), hidden, config.sf_block_m, 1) : tensor_map_l1_acts_sf;
    const auto tensor_map_shared_l1_weights = with_shared ? make_tma_weights_2d_desc(
        shared_l1_weights, config.block_k, config.load_block_n, config.swizzle_acts_mode) : tensor_map_l1_weights;
    const auto tensor_map_shared_l1_weights_sf = with_shared ? sf_desc(
        shared_l1_weights_sf, shared_intermediate_hidden * 2, hidden, config.block_n, 1) : tensor_map_l1_weights_sf;
    const auto tensor_map_shared_l1_output = with_shared ? acts_desc(
        shared_l2_acts, shared_intermediate_hidden, num_max_tokens_per_rank, config.block_n / 2, config.block_m, config.swizzle_acts_mode / 2) : tensor_map_l1_output;
    const auto tensor_map_shared_l2_acts = with_shared ? acts_desc(
        shared_l2_acts, shared_intermediate_hidden, num_max_tokens_per_rank, config.block_k, config.load_block_m, config.swizzle_acts_mode) : tensor_map_l2_acts;
    const auto tensor_map_shared_l2_acts_sf = with_shared ? sf_desc(
        shared_l2_acts_sf, static_cast<int>(shared_l2_acts_sf.size(0)), shared_intermediate_hidden, config.sf_block_m, 1) : tensor_map_l2_acts_sf;
    const auto tensor_map_shared_l2_weights = with_shared ? make_tma_weights_2d_desc(
        shared_l2_weights, config.block_k, config.load_block_n, config.swizzle_acts_mode) : tensor_map_l2_weights;
    const auto tensor_map_shared_l2_weights_sf = with_shared ? sf_desc(
        shared_l2_weights_sf, hidden, shared_intermediate_hidden, config.block_n, 1) : tensor_map_l2_weights_sf;

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->mutable_data_ptr<int>();

    const auto num_sms = get_num_sms_for_mega_moe();

    // Compile
    const auto kernel = jit->compile("sm90_fp8_fp4_mega_moe", std::format(R"(
#define DG_NO_DEVICE_PRINTF
// NOTES: `__builtin_unreachable` after the trap avoids ptxas spills in the math warps
#define DG_DEVICE_ASSERT(cond) do {{ if (not (cond)) {{ asm("trap;"); __builtin_unreachable(); }} }} while (0)
#include <deep_gemm/impls/sm90_fp8_fp4_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_fp4_mega_moe_impl<
        {},
        {}, {},
        {}, {},
        {},
        {}, {}, {},
        {},
        {},
        {},
        {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {},
        {},
        {},
        {},
        {}
    >);
}};
)", num_max_tokens_per_rank,
        hidden, intermediate_hidden,
        num_experts, num_shared_experts,
        num_topk,
        config.block_m, config.block_n, config.block_k,
        config.store_block_m_l2,
        config.sf_block_m,
        config.num_ring_tokens,
        config.num_sf_ring_tokens,
        config.num_stages,
        config.num_bytes_per_pull,
        config.num_dispatch_threads, config.num_non_epilogue_threads, config.num_epilogue_threads,
        num_sms, num_ranks,
        to_string(activation_clamp),
        clamp_after_silu ? "true" : "false",
        to_string(activation_alpha),
        to_string(activation_beta),
        fast_math ? "true" : "false",
        config.pingpong ? "true" : "false"));

    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = config.smem_size,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads, 1, 1),
            .cluster_dim = dim3(2, 1, 1),
        },
        y.mutable_data_ptr(),
        cumulative_local_expert_recv_stats_ptr,
        num_tokens,
        layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        tensor_map_l1_acts,
        tensor_map_l1_acts_sf,
        tensor_map_l1_weights,
        tensor_map_l1_weights_sf,
        tensor_map_l1_output,
        tensor_map_l2_acts,
        tensor_map_l2_acts_sf,
        tensor_map_l2_weights,
        tensor_map_l2_weights_sf,
        tensor_map_shared_l1_acts,
        tensor_map_shared_l1_acts_sf,
        tensor_map_shared_l1_weights,
        tensor_map_shared_l1_weights_sf,
        tensor_map_shared_l1_output,
        tensor_map_shared_l2_acts,
        tensor_map_shared_l2_acts_sf,
        tensor_map_shared_l2_weights,
        tensor_map_shared_l2_weights_sf);
}

} // namespace deep_gemm
