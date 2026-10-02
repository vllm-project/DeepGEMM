#pragma once

#include <cstdio>
#include <format>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include "../../utils/torch_compat.hpp"

#include <deep_gemm/layout/mega_gate.cuh>

#include "../../runtime/jit.hpp"
#include "../../utils/exception.hpp"
#include "../heuristics/mega_gate.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

namespace mega_gate_layout = layout::mega_gate;

static void sm100_bf16_mega_gate(const torch::stable::Tensor& x, const torch::stable::Tensor& weight,
                                 const std::optional<torch::stable::Tensor>& bias,
                                 const std::optional<torch::stable::Tensor>& image_bias,
                                 const std::optional<torch::stable::Tensor>& image_token_mask,
                                 const std::optional<torch::stable::Tensor>& mask,
                                 const std::optional<torch::stable::Tensor>& fix_routing_mask,
                                 const std::optional<torch::stable::Tensor>& to_physical_map,
                                 const std::optional<torch::stable::Tensor>& logical_count,
                                 const torch::stable::Tensor& topk_idx,
                                 const std::optional<torch::stable::Tensor>& unmapped_topk_idx,
                                 const torch::stable::Tensor& topk_weights,
                                 const std::optional<torch::stable::Tensor>& force_random,
                                 const int& num_tokens, const int& hidden,
                                 const int& num_routed_experts, const int& num_topk,
                                 const int& num_shared_experts, const float& routed_scaling_factor,
                                 const int& ep_rank, const int& scoring_type,
                                 const SM100BF16MegaGateConfig& config,
                                 const torch::stable::Tensor& scratch,
                                 const torch::stable::Tensor& score_barriers) {
    const auto num_duplicate_experts = to_physical_map.has_value() ? static_cast<int>(to_physical_map->size(1)) : 0;
    const auto unmapped_topk_idx_stride = unmapped_topk_idx.has_value() ? unmapped_topk_idx->stride(0) : 0;
    const auto num_aligned_experts = align(num_routed_experts, static_cast<int>(mega_gate_layout::kExpertAlignment));
    const auto experts_per_group = num_aligned_experts / config.num_expert_groups;
    const auto expert_mma = std::min(experts_per_group, 256);
    const auto load_block_m = config.block_tokens / config.num_mma_ctas;
    const auto load_block_n = expert_mma / config.num_mma_ctas;
    const auto tensor_map_x = make_tma_a_desc(cute::UMMA::Major::K, x, num_tokens, hidden,
                                              load_block_m, mega_gate_layout::BLOCK_K,
                                              static_cast<int>(x.stride(0)), 1, 128);
    const auto tensor_map_weight = make_tma_b_desc(cute::UMMA::Major::K, weight, num_routed_experts, hidden,
                                                   load_block_n, mega_gate_layout::BLOCK_K,
                                                   static_cast<int>(weight.stride(0)), 1, 128);

    if (deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
        printf("Mega gate: T=%d, H=%d, E=%d, block_tokens=%d, mma_ctas=%d, split_k=%d, expert_groups=%d, "
               "stages=%d, gate_warpgroups=%d, threads=%d, shared memory=%d\n",
               num_tokens, hidden, num_routed_experts, config.block_tokens, config.num_mma_ctas,
               config.num_split_k, config.num_expert_groups, config.num_stages,
               config.num_gate_warpgroups,
               mega_gate_layout::kNumNonEpilogueThreads + config.num_gate_warpgroups * 128,
               config.smem_size);
    }

    // Compile
    const auto kernel = jit->compile("sm100_bf16_mega_gate", std::format(R"(
#include <deep_gemm/impls/sm100_bf16_mega_gate.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_mega_gate_impl<
        {}, {},
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {},
        {}, {}, {}, {}, {}, {}, {}, {}
    >);
}};
)",
        hidden, num_aligned_experts,
        config.block_tokens,
        config.num_stages,
        config.num_gate_warpgroups * 128,
        config.num_mma_ctas, config.num_split_k, config.num_expert_groups,
        config.num_launch_sms,
        num_topk, scoring_type,
        num_routed_experts == num_aligned_experts,
        (mask ? mask->mutable_data_ptr() : nullptr) != nullptr,
        (unmapped_topk_idx ? unmapped_topk_idx->mutable_data_ptr() : nullptr) != nullptr,
        (to_physical_map ? to_physical_map->mutable_data_ptr() : nullptr) != nullptr,
        (bias ? bias->mutable_data_ptr() : nullptr) != nullptr,
        (image_token_mask ? image_token_mask->mutable_data_ptr() : nullptr) != nullptr,
        (fix_routing_mask ? fix_routing_mask->mutable_data_ptr() : nullptr) != nullptr,
        (force_random ? force_random->mutable_data_ptr() : nullptr) != nullptr));

    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = config.smem_size,
            .grid_dim = dim3(config.num_launch_sms, 1, 1),
            .block_dim = dim3(mega_gate_layout::kNumNonEpilogueThreads + config.num_gate_warpgroups * 128, 1, 1),
            .cluster_dim = dim3(config.num_mma_ctas, 1, 1),
        },
        tensor_map_x, tensor_map_weight,
        bias ? bias->mutable_data_ptr() : nullptr,
        image_bias ? image_bias->mutable_data_ptr() : nullptr,
        image_token_mask ? image_token_mask->mutable_data_ptr() : nullptr,
        mask ? mask->mutable_data_ptr() : nullptr,
        to_physical_map ? to_physical_map->mutable_data_ptr() : nullptr,
        logical_count ? logical_count->mutable_data_ptr() : nullptr,
        topk_idx.mutable_data_ptr(),
        unmapped_topk_idx ? unmapped_topk_idx->mutable_data_ptr() : nullptr,
        topk_weights.mutable_data_ptr(),
        scratch.mutable_data_ptr(), score_barriers.mutable_data_ptr(),
        static_cast<uint32_t>(num_tokens),
        static_cast<uint32_t>(num_routed_experts),
        static_cast<uint32_t>(num_shared_experts),
        static_cast<uint32_t>(num_duplicate_experts),
        routed_scaling_factor, static_cast<int>(ep_rank),
        unmapped_topk_idx_stride,
        fix_routing_mask ? fix_routing_mask->mutable_data_ptr() : nullptr,
        force_random ? force_random->mutable_data_ptr() : nullptr);
}

} // namespace deep_gemm
