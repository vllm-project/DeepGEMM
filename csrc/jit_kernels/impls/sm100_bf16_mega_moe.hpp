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

// TRT-LLM BF16 weights: [num_experts, K / 64, N, 64]
static CUtensorMap make_trtllm_bf16_l1_tma_desc(const torch::stable::Tensor& weights, const int& load_block_n) {
    DG_HOST_ASSERT(weights.dim() == 4 and weights.size(3) == 64 and weights.is_contiguous());
    const auto rows = static_cast<cuuint64_t>(weights.size(2));
    CUtensorMap result;
    const cuuint64_t dims[3] = {64, rows, static_cast<cuuint64_t>(weights.size(0) * weights.size(1))};
    const cuuint64_t strides[2] = {128, rows * 128};
    const cuuint32_t box[3] = {64, static_cast<cuuint32_t>(load_block_n), 1};
    const cuuint32_t element_strides[3] = {1, 1, 1};
    DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuTensorMapEncodeTiled(
        &result, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, weights.mutable_data_ptr(), dims,
        strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return result;
}

// Also undoes TRT's 32-row L2 shuffle
static CUtensorMap make_trtllm_bf16_l2_tma_desc(const torch::stable::Tensor& weights, const int& load_block_n) {
    DG_HOST_ASSERT(weights.dim() == 4 and weights.size(3) == 64 and weights.is_contiguous());
    DG_HOST_ASSERT(weights.size(2) % 32 == 0 and load_block_n % 32 == 0);
    const auto rows = static_cast<cuuint64_t>(weights.size(2));
    CUtensorMap result;
    const cuuint64_t dims[5] = {64, 4, 8, rows / 32, static_cast<cuuint64_t>(weights.size(0) * weights.size(1))};
    const cuuint64_t strides[4] = {8 * 128, 128, 32 * 128, rows * 128};
    const cuuint32_t box[5] = {64, 4, 8, static_cast<cuuint32_t>(load_block_n / 32), 1};
    const cuuint32_t element_strides[5] = {1, 1, 1, 1, 1};
    DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuTensorMapEncodeTiled(
        &result, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, weights.mutable_data_ptr(), dims,
        strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return result;
}

template <bool use_trtllm_weights>
static void sm100_bf16_mega_moe(
    const torch::stable::Tensor& y,
    const torch::stable::Tensor& l1_acts, const torch::stable::Tensor& l2_acts,
    const torch::stable::Tensor& shared_l1_acts, const torch::stable::Tensor& shared_l2_acts,
    const torch::stable::Tensor& l1_weights, const torch::stable::Tensor& l2_weights,
    const torch::stable::Tensor& shared_l1_weights, const torch::stable::Tensor& shared_l2_weights,
    const torch::stable::Tensor& sm_locality_domains,
    const std::optional<torch::stable::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_shared_experts,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& clamp_after_silu,
    const float& activation_alpha,
    const float& activation_beta,
    const bool& fast_math,
    const void* post_expert_norm_gamma, const float& post_expert_norm_eps,
    const float* shared_expert_weights
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_ring_tokens = static_cast<int>(l1_acts.size(0));
    const auto shared_intermediate_hidden = intermediate_hidden * num_shared_experts;

    // Heuristics
    const auto config = get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden,
        num_ring_tokens, 0, MmaKind::BF16);
    // The BF16 kernel has a single store block size for both layers
    DG_HOST_ASSERT(config.store_block_m_l1 == config.store_block_m_l2);
    if constexpr (use_trtllm_weights) {
        DG_HOST_ASSERT(num_shared_experts == 0);
        DG_HOST_ASSERT(config.block_k == 64 and config.load_block_n % 32 == 0 and config.swizzle_weights_mode == 128 and
                       "Unsupported tile configuration for TRT-LLM BF16 weight layout");
    }

    // Make tensormap
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, config.num_ring_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l1_weights = use_trtllm_weights ?
        make_trtllm_bf16_l1_tma_desc(l1_weights, config.load_block_n) :
        make_tma_weights_3d_desc(l1_weights, config.block_k, config.load_block_n, config.swizzle_weights_mode);
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, config.num_ring_tokens,
                                                       config.block_n / 2, config.store_block_m_l1,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       config.swizzle_acts_mode);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, config.num_ring_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l2_weights = use_trtllm_weights ?
        make_trtllm_bf16_l2_tma_desc(l2_weights, config.load_block_n) :
        make_tma_weights_3d_desc(l2_weights, config.block_k, config.load_block_n, config.swizzle_weights_mode);

    const auto tensor_map_shared_l1_acts = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l1_acts,
        hidden, num_max_tokens_per_rank,
        config.block_k, config.load_block_m,
        static_cast<int>(shared_l1_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l1_acts;
    const auto tensor_map_shared_l1_weights = num_shared_experts > 0 ?
        make_tma_weights_2d_desc(shared_l1_weights, config.block_k, config.load_block_n, config.swizzle_weights_mode) : tensor_map_l1_weights;
    const auto tensor_map_shared_l1_output = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_acts,
        shared_intermediate_hidden, num_max_tokens_per_rank,
        config.block_n / 2, config.store_block_m_l1,
        static_cast<int>(shared_l2_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l1_output;
    const auto tensor_map_shared_l2_acts = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_acts,
        shared_intermediate_hidden, num_max_tokens_per_rank,
        config.block_k, config.load_block_m,
        static_cast<int>(shared_l2_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l2_acts;
    const auto tensor_map_shared_l2_weights = num_shared_experts > 0 ?
        make_tma_weights_2d_desc(shared_l2_weights, config.block_k, config.load_block_n, config.swizzle_weights_mode) : tensor_map_l2_weights;

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->mutable_data_ptr<int>();

    const auto num_sms = get_num_sms_for_mega_moe();

    // Compile
    const auto kernel = jit->compile("sm100_bf16_mega_moe", std::format(R"(
#include <deep_gemm/impls/sm100_bf16_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_mega_moe_impl<
        {},
        {}, {},
        {}, {},
        {}, {}, {},
        {},
        {},
        {},
        {},
        {},
        {}, {}, {},
        {}, {},
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
        config.store_block_m_l1,
        config.num_ring_tokens,
        config.num_stages,
        config.num_bytes_per_pull,
        config.num_dispatch_threads, config.num_non_epilogue_threads, config.num_epilogue_threads,
        num_sms, num_ranks,
        to_string(activation_clamp),
        clamp_after_silu ? "true" : "false",
        to_string(activation_alpha),
        to_string(activation_beta),
        fast_math ? "true" : "false",
        post_expert_norm_gamma != nullptr ? "true" : "false",
        shared_expert_weights != nullptr ? "true" : "false",
        use_trtllm_weights ? "true" : "false"));

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
        shared_expert_weights,
        post_expert_norm_gamma, post_expert_norm_eps,
        layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        tensor_map_l1_acts,
        tensor_map_l1_weights,
        tensor_map_l1_output,
        tensor_map_l2_acts,
        tensor_map_l2_weights,
        tensor_map_shared_l1_acts,
        tensor_map_shared_l1_weights,
        tensor_map_shared_l1_output,
        tensor_map_shared_l2_acts,
        tensor_map_shared_l2_weights,
        sm_locality_domains.mutable_data_ptr<uint8_t>());
}

} // namespace deep_gemm
