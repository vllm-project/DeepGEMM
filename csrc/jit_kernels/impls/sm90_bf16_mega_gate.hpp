#pragma once

#include <format>
#include <torch/csrc/stable/ops.h>

#include "../../runtime/jit.hpp"
#include "../../runtime/runtime.hpp"
#include "../heuristics/sm90_mega_gate.hpp"
#include "runtime_utils.hpp"
#include "smxx_cublaslt.hpp"

namespace deep_gemm {

static void sm90_bf16_mega_gate(const torch::stable::Tensor& x, const torch::stable::Tensor& weight,
                                const std::optional<torch::stable::Tensor>& bias,
                                const std::optional<torch::stable::Tensor>& image_bias,
                                const std::optional<torch::stable::Tensor>& image_token_mask,
                                const std::optional<torch::stable::Tensor>& mask,
                                const std::optional<torch::stable::Tensor>& fix_routing_mask,
                                const std::optional<torch::stable::Tensor>& force_random,
                                const layout::mega_gate::RoutingArgs& routing_args,
                                const int num_tokens, const int hidden,
                                const int num_routed_experts, const int num_topk) {
    const auto num_aligned_experts = align(num_routed_experts, 128);
    // Match the TMA path's alignment checks before issuing vector loads in GEMV.
    DG_HOST_ASSERT(reinterpret_cast<std::uintptr_t>(x.mutable_data_ptr()) % 16u == 0u);
    DG_HOST_ASSERT(reinterpret_cast<std::uintptr_t>(weight.mutable_data_ptr()) % 16u == 0u);
    const auto config = get_sm90_mega_gate_config(num_tokens, hidden, num_routed_experts,
                                                 runtime->get_num_sms(), runtime->is_cublaslt_available(),
                                                 fix_routing_mask.has_value());
    if (config.use_gemv or config.use_cublas) {
        const auto logits = torch::stable::new_empty(x, {num_tokens, num_routed_experts}, torch::headeronly::ScalarType::Float);
        if (config.use_gemv) {
            const auto gemv_kernel = jit->compile("sm90_bf16_mega_gate_gemv", std::format(R"(
#include <deep_gemm/impls/sm90_bf16_mega_gate.cuh>
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm90_bf16_mega_gate_gemv<{}>);
}};
)", hidden));
            jit->launch(gemv_kernel, {
                .grid_dim = dim3(config.num_launch_ctas, 1, 1),
                .block_dim = dim3(128, 1, 1),
            }, x.mutable_data_ptr(), weight.mutable_data_ptr(), logits.mutable_data_ptr(),
                static_cast<uint32_t>(num_routed_experts), static_cast<uint32_t>(num_tokens));
        } else {
            cublaslt_gemm(x, weight, logits, num_tokens, num_routed_experts, hidden,
                          cute::UMMA::Major::K, cute::UMMA::Major::K, false);
        }
        const auto gate_kernel = jit->compile("sm90_bf16_mega_gate_route", std::format(R"(
#include <deep_gemm/impls/sm90_bf16_mega_gate.cuh>
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm90_bf16_mega_gate_route<
        {}, {}, {}, {}, {}, {}, {}, {}, {}
    >);
}};
)", num_aligned_experts, num_topk, bias.has_value(), image_token_mask.has_value(), mask.has_value(),
            fix_routing_mask.has_value(), force_random.has_value(),
            routing_args.to_physical_map != nullptr, routing_args.unmapped_topk_idx != nullptr));
        jit->launch(gate_kernel, {
            .grid_dim = dim3(config.num_route_ctas, 1, 1),
            .block_dim = dim3(128, 1, 1),
        }, logits.mutable_data_ptr(),
            bias ? bias->mutable_data_ptr() : nullptr,
            image_bias ? image_bias->mutable_data_ptr() : nullptr,
            image_token_mask ? image_token_mask->mutable_data_ptr() : nullptr,
            mask ? mask->mutable_data_ptr() : nullptr,
            fix_routing_mask ? fix_routing_mask->mutable_data_ptr() : nullptr,
            force_random ? force_random->mutable_data_ptr() : nullptr,
            routing_args, static_cast<uint32_t>(num_tokens));
        return;
    }
    const auto split_experts = config.num_expert_groups > 1;
    const auto tensor_map_x = make_tma_a_desc(cute::UMMA::Major::K, x, num_tokens, hidden,
                                              config.mma_tokens, config.block_k, static_cast<int>(x.stride(0)), 1, config.swizzle);
    const auto tensor_map_weight = make_tma_b_desc(cute::UMMA::Major::K, weight, num_routed_experts, hidden,
                                                   config.tile_experts / config.num_math_groups, config.block_k,
                                                   static_cast<int>(weight.stride(0)), 1, config.swizzle);
    const auto kernel = jit->compile(split_experts ? "sm90_bf16_mega_gate_split_n" : "sm90_bf16_mega_gate", std::format(R"(
#include <deep_gemm/impls/sm90_bf16_mega_gate.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_bf16_mega_gate_impl<
        {}, {}, {}, {}, {},
        {}, {}, {}, {}, {}, {}, {}, {}
    >);
}};
)", hidden, config.tile_experts, config.block_tokens, config.num_stages, num_topk,
        bias.has_value(), image_token_mask.has_value(), mask.has_value(),
        fix_routing_mask.has_value(), force_random.has_value(),
        routing_args.to_physical_map != nullptr, routing_args.unmapped_topk_idx != nullptr, config.num_expert_groups));

    torch::stable::Tensor candidate_scores, candidate_indices;
    if (split_experts) {
        candidate_scores = torch::stable::new_empty(x, {num_tokens, config.num_expert_groups * num_topk, 2}, torch::headeronly::ScalarType::Float);
        candidate_indices = torch::stable::new_empty(x, {num_tokens, config.num_expert_groups * num_topk}, torch::headeronly::ScalarType::Int);
    }

    jit->launch(kernel, {
        .num_smem_bytes = config.smem_size,
        .grid_dim = dim3(config.num_launch_ctas, 1, 1),
        .block_dim = dim3(config.num_threads, 1, 1),
    }, tensor_map_x, tensor_map_weight,
        bias ? bias->mutable_data_ptr() : nullptr,
        image_bias ? image_bias->mutable_data_ptr() : nullptr,
        image_token_mask ? image_token_mask->mutable_data_ptr() : nullptr,
        mask ? mask->mutable_data_ptr() : nullptr,
        fix_routing_mask ? fix_routing_mask->mutable_data_ptr() : nullptr,
        force_random ? force_random->mutable_data_ptr() : nullptr,
        routing_args, static_cast<uint32_t>(num_tokens),
        split_experts ? candidate_scores.mutable_data_ptr() : nullptr,
        split_experts ? candidate_indices.mutable_data_ptr() : nullptr);

    if (split_experts) {
        const auto merge_kernel = jit->compile("sm90_bf16_mega_gate_merge", std::format(R"(
#include <deep_gemm/impls/sm90_bf16_mega_gate.cuh>
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm90_bf16_mega_gate_merge<{}, {}, {}, {}, {}, {}>);
}};
)", num_topk, config.num_expert_groups, mask.has_value(), force_random.has_value(),
            routing_args.to_physical_map != nullptr, routing_args.unmapped_topk_idx != nullptr));
        jit->launch(merge_kernel, {
            .grid_dim = dim3(config.num_route_ctas, 1, 1),
            .block_dim = dim3(128, 1, 1),
        }, candidate_scores.mutable_data_ptr(), candidate_indices.mutable_data_ptr(),
            mask ? mask->mutable_data_ptr() : nullptr,
            force_random ? force_random->mutable_data_ptr() : nullptr,
            routing_args, static_cast<uint32_t>(num_tokens));
    }
}

} // namespace deep_gemm
