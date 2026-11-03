#pragma once

#include <format>
#include <torch/csrc/stable/ops.h>

#include <deep_gemm/layout/sm90_mega_gate.cuh>

#include "../../runtime/jit.hpp"
#include "../../runtime/runtime.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

template <uint32_t kBlockTokens>
static int get_sm90_mega_gate_smem_size(const int num_aligned_experts) {
    using namespace layout::sm90_mega_gate;
    switch (num_aligned_experts) {
        case 128: return Config<kBlockTokens, 128>::kSmemBytes;
        case 256: return Config<kBlockTokens, 256>::kSmemBytes;
        case 384: return Config<kBlockTokens, 384>::kSmemBytes;
        case 512: return Config<kBlockTokens, 512>::kSmemBytes;
        default: DG_HOST_UNREACHABLE("Invalid number of gate experts");
    }
}

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
    constexpr int block_k = 64, swizzle = 128;
    const auto num_aligned_experts = align(num_routed_experts, 128);
    // Match the TMA path's alignment checks before issuing vector loads in GEMV.
    DG_HOST_ASSERT(reinterpret_cast<std::uintptr_t>(x.mutable_data_ptr()) % 16u == 0u);
    DG_HOST_ASSERT(reinterpret_cast<std::uintptr_t>(weight.mutable_data_ptr()) % 16u == 0u);
    if (num_tokens <= 16 and not heuristics_runtime->get_deterministic_algorithms()) {
        const auto logits = torch::stable::new_empty(x, {num_tokens, num_routed_experts}, torch::headeronly::ScalarType::Float);
        const auto gemv_kernel = jit->compile("sm90_bf16_mega_gate_gemv", std::format(R"(
#include <deep_gemm/impls/sm90_bf16_mega_gate.cuh>
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm90_bf16_mega_gate_gemv<{}>);
}};
)", hidden));
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
        jit->launch(gemv_kernel, {
            .grid_dim = dim3(num_routed_experts / 4, num_tokens, 1),
            .block_dim = dim3(128, 1, 1),
        }, x.mutable_data_ptr(), weight.mutable_data_ptr(), logits.mutable_data_ptr(), static_cast<uint32_t>(num_routed_experts));
        jit->launch(gate_kernel, {
            .grid_dim = dim3(ceil_div(num_tokens, 4), 1, 1),
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
    // Smaller tiles expose more CTAs for short batches. Both variants accumulate
    // the complete K dimension in the same order, including deterministic mode.
    const auto block_tokens = num_tokens <= runtime->get_num_sms() * 16 ? 16 : 64;
    const auto num_math_groups = ceil_div(num_aligned_experts, 256);
    const auto smem_size = block_tokens == 16 ? get_sm90_mega_gate_smem_size<16>(num_aligned_experts) :
                                              get_sm90_mega_gate_smem_size<64>(num_aligned_experts);
    const auto tensor_map_x = make_tma_a_desc(cute::UMMA::Major::K, x, num_tokens, hidden,
                                              64, block_k, static_cast<int>(x.stride(0)), 1, swizzle);
    const auto tensor_map_weight = make_tma_b_desc(cute::UMMA::Major::K, weight, num_routed_experts, hidden,
                                                   num_aligned_experts / num_math_groups, block_k, static_cast<int>(weight.stride(0)), 1, swizzle);
    const auto kernel = jit->compile("sm90_bf16_mega_gate", std::format(R"(
#include <deep_gemm/impls/sm90_bf16_mega_gate.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_bf16_mega_gate_impl<
        {}, {}, {}, {},
        {}, {}, {}, {}, {}, {}, {}
    >);
}};
)", hidden, num_aligned_experts, block_tokens, num_topk,
        bias.has_value(), image_token_mask.has_value(), mask.has_value(),
        fix_routing_mask.has_value(), force_random.has_value(),
        routing_args.to_physical_map != nullptr, routing_args.unmapped_topk_idx != nullptr));

    jit->launch(kernel, {
        .num_smem_bytes = smem_size,
        .grid_dim = dim3(ceil_div(num_tokens, block_tokens), 1, 1),
        .block_dim = dim3((num_math_groups + 1) * 128, 1, 1),
    }, tensor_map_x, tensor_map_weight,
        bias ? bias->mutable_data_ptr() : nullptr,
        image_bias ? image_bias->mutable_data_ptr() : nullptr,
        image_token_mask ? image_token_mask->mutable_data_ptr() : nullptr,
        mask ? mask->mutable_data_ptr() : nullptr,
        fix_routing_mask ? fix_routing_mask->mutable_data_ptr() : nullptr,
        force_random ? force_random->mutable_data_ptr() : nullptr,
        routing_args, static_cast<uint32_t>(num_tokens));
}

} // namespace deep_gemm
