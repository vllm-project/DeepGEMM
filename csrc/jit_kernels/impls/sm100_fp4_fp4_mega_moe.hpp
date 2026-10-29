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

// TRT layout reader adapted from sgl-project/DeepGEMM cbe1727eacd9ad704795584ce1421dd1f5a8d67e.
// MIT licensed; see the retained LICENSE.
// Undo TRT's L2 shuffle: GMEM row 32*i3 + 8*i1 + i2 -> SMEM row 32*i3 + 4*i2 + i1.
static CUtensorMap make_trtllm_fp4_l2_tma_desc(
    const torch::stable::Tensor& weights, const int inner_bytes, const int rows) {
    DG_HOST_ASSERT(weights.is_contiguous() and rows % 128 == 0);
    DG_HOST_ASSERT(inner_bytes % 128 == 0);
    CUtensorMap result;
    const cuuint64_t dims[4] = {static_cast<cuuint64_t>(inner_bytes), 4, 8,
                                static_cast<cuuint64_t>(rows / 32)};
    const cuuint64_t strides[3] = {8ull * inner_bytes,
                                   static_cast<cuuint64_t>(inner_bytes), 32ull * inner_bytes};
    const cuuint32_t box[4] = {128, 4, 8, 4};
    const cuuint32_t element_strides[4] = {1, 1, 1, 1};
    DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuTensorMapEncodeTiled(
        &result, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, weights.mutable_data_ptr(), dims,
        strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return result;
}

// TRT SFs: [row block][k chunk][128 words]. L2 also undoes the 32-row shuffle.
static CUtensorMap make_trtllm_fp4_sf_tma_desc(
    const torch::stable::Tensor& sf, const bool& is_l1,
    const int& num_row_blocks, const int& num_k_chunks,
    const int& box_k_chunks, const int& box_row_blocks) {
    DG_HOST_ASSERT(num_row_blocks > 0 and num_k_chunks > 0);
    DG_HOST_ASSERT(box_k_chunks <= num_k_chunks and box_row_blocks <= num_row_blocks);
    CUtensorMap result;
    const auto row_block_stride = static_cast<cuuint64_t>(num_k_chunks) * 512ull;
    const cuuint32_t element_strides[5] = {1, 1, 1, 1, 1};
    if (is_l1) {
        const cuuint64_t dims[3] = {128, static_cast<cuuint64_t>(num_k_chunks),
                                    static_cast<cuuint64_t>(num_row_blocks)};
        const cuuint64_t strides[2] = {512ull, row_block_stride};
        const cuuint32_t box[3] = {128, static_cast<cuuint32_t>(box_k_chunks),
                                   static_cast<cuuint32_t>(box_row_blocks)};
        DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuTensorMapEncodeTiled(
            &result, CU_TENSOR_MAP_DATA_TYPE_UINT32, 3, sf.mutable_data_ptr(), dims,
            strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    } else {
        const cuuint64_t dims[5] = {4, 4, 8, static_cast<cuuint64_t>(num_k_chunks),
                                    static_cast<cuuint64_t>(num_row_blocks)};
        const cuuint64_t strides[4] = {128ull, 16ull, 512ull, row_block_stride};
        const cuuint32_t box[5] = {4, 4, 8, static_cast<cuuint32_t>(box_k_chunks),
                                   static_cast<cuuint32_t>(box_row_blocks)};
        DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuTensorMapEncodeTiled(
            &result, CU_TENSOR_MAP_DATA_TYPE_UINT32, 5, sf.mutable_data_ptr(), dims,
            strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }
    return result;
}


template <bool use_trtllm_weights>
static void sm100_fp4_fp4_mega_moe(
    const torch::stable::Tensor& y,
    const torch::stable::Tensor& l1_acts, const torch::stable::Tensor& l1_acts_sf,
    const torch::stable::Tensor& l2_acts, const torch::stable::Tensor& l2_acts_sf,
    // BF16 shared expert tensors: `shared_l1_acts` is the caller-provided BF16 `x`,
    // `shared_l2_acts` is the BF16 intermediate view sliced from the symmetric buffer
    const torch::stable::Tensor& shared_l1_acts, const torch::stable::Tensor& shared_l2_acts,
    const torch::stable::Tensor& shared_l1_weights, const torch::stable::Tensor& shared_l2_weights,
    const torch::stable::Tensor& l1_weights, const torch::stable::Tensor& l2_weights,
    const torch::stable::Tensor& l1_weights_sf, const torch::stable::Tensor& l2_weights_sf,
    const torch::stable::Tensor& sm_locality_domains,
    const std::optional<torch::stable::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_shared_experts,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math,
    const void* l1_alphas, const void* l2_alphas, const void* a2_scales,
    const void* post_expert_norm_gamma, const float& post_expert_norm_eps
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_ring_tokens = static_cast<int>(l1_acts.size(0));
    const auto num_sf_ring_tokens = static_cast<int>(l1_acts_sf.size(0));
    const auto shared_intermediate_hidden = intermediate_hidden * num_shared_experts;

    // Heuristics
    const auto config = get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden,
        num_ring_tokens, num_sf_ring_tokens,
        MmaKind::NVFP4);
    // The NVFP4 kernel has a single store block size for both layers
    DG_HOST_ASSERT(config.store_block_m_l1 == config.store_block_m_l2);
    if constexpr (use_trtllm_weights) {
        // The shuffled TMA boxes and L1 output packing below use exactly 128 rows.
        DG_HOST_ASSERT(num_shared_experts == 0);
        DG_HOST_ASSERT(config.block_n == 128 and config.load_block_n == 128 and
                       config.sf_block_n == 128 and config.swizzle_weights_mode == 128 and
                       config.block_k % 256 == 0 and
                       "Unsupported tile configuration for TRT-LLM NVFP4 weight layout");
    }

    // View all packed-FP4 tensors as raw bytes: TMA moves plain byte rows, so the
    // K dimensions below are all `elements / 2`
    constexpr int kGranK = 16;
    const auto l1_weights_bytes = torch_compat::view_dtype(l1_weights, torch::headeronly::ScalarType::Byte);
    const auto l2_weights_bytes = torch_compat::view_dtype(l2_weights, torch::headeronly::ScalarType::Byte);
    const int block_k_bytes = config.block_k / 2;
    const int sf_smem_outer_dim = config.block_k / (kGranK * 4);

    // Make tensormap
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden / 2, config.num_ring_tokens,
                                                     block_k_bytes, config.load_block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        config.num_sf_ring_tokens, hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0, 0, false,
                                                        sf_smem_outer_dim, static_cast<int>(l1_acts_sf.stride(-1)));
    const auto tensor_map_l1_weights = make_tma_2d_desc(l1_weights_bytes,
                                                        hidden / 2, num_experts_per_rank * intermediate_hidden * 2,
                                                        block_k_bytes, config.load_block_n,
                                                        static_cast<int>(l1_weights_bytes.stride(-2)),
                                                        config.swizzle_weights_mode);
    const auto tensor_map_l1_weights_sf = use_trtllm_weights ? make_trtllm_fp4_sf_tma_desc(
        l1_weights_sf, true, num_experts_per_rank * intermediate_hidden * 2 / 128,
        hidden / 64, sf_smem_outer_dim, config.block_n / 128) : make_tma_sf_desc(cute::UMMA::Major::MN, l1_weights_sf,
                                                           intermediate_hidden * 2, hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0, 0, false,
                                                           sf_smem_outer_dim, static_cast<int>(l1_weights_sf.stride(-1)));
    // NOTES: L1 output and L2 activations are essentially the same tensor.
    // Post-SwiGLU output has half the N width (`BLOCK_N / 2` elements = `BLOCK_N / 4`
    // bytes per input tile); the store box is small, so no swizzling is applied
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden / 2, config.num_ring_tokens,
                                                       config.block_n / 4, config.store_block_m_l1,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       0);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden / 2, config.num_ring_tokens,
                                                     block_k_bytes, config.load_block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        config.num_sf_ring_tokens, intermediate_hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0, 0, false,
                                                        sf_smem_outer_dim, static_cast<int>(l2_acts_sf.stride(-1)));
    const auto tensor_map_l2_weights = use_trtllm_weights ? make_trtllm_fp4_l2_tma_desc(
        l2_weights_bytes, intermediate_hidden / 2, num_experts_per_rank * hidden) : make_tma_2d_desc(l2_weights_bytes,
                                                        intermediate_hidden / 2, num_experts_per_rank * hidden,
                                                        block_k_bytes, config.load_block_n,
                                                        static_cast<int>(l2_weights_bytes.stride(-2)),
                                                        config.swizzle_weights_mode);
    const auto tensor_map_l2_weights_sf = use_trtllm_weights ? make_trtllm_fp4_sf_tma_desc(
        l2_weights_sf, false, num_experts_per_rank * hidden / 128,
        intermediate_hidden / 64, sf_smem_outer_dim, config.block_n / 128) : make_tma_sf_desc(cute::UMMA::Major::MN, l2_weights_sf,
                                                           hidden, intermediate_hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0, 0, false,
                                                           sf_smem_outer_dim, static_cast<int>(l2_weights_sf.stride(-1)));

    // BF16 shared expert descriptors: same 128B tiles hold `block_k / 4` BF16 elements.
    // With no shared experts they fall back to routed maps so the kernel params stay valid
    const int shared_block_k = config.block_k / 4;
    // NOTES: the caller's BF16 `x` may hold fewer rows than the token-aligned buffer, so bound
    // the descriptor by its real row count and let TMA zero-fill the last partial block
    const auto tensor_map_shared_l1_acts = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l1_acts,
        hidden, static_cast<int>(shared_l1_acts.size(0)),
        shared_block_k, config.load_block_m,
        static_cast<int>(shared_l1_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l1_acts;
    const auto tensor_map_shared_l1_weights = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l1_weights,
        hidden, shared_intermediate_hidden * 2,
        shared_block_k, config.load_block_n,
        static_cast<int>(shared_l1_weights.stride(-2)),
        config.swizzle_weights_mode) : tensor_map_l1_weights;
    const auto tensor_map_shared_l1_output = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_acts,
        shared_intermediate_hidden, num_max_tokens_per_rank,
        config.block_n / 2, config.store_block_m_l1,
        static_cast<int>(shared_l2_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l1_output;
    const auto tensor_map_shared_l2_acts = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_acts,
        shared_intermediate_hidden, num_max_tokens_per_rank,
        shared_block_k, config.load_block_m,
        static_cast<int>(shared_l2_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l2_acts;
    const auto tensor_map_shared_l2_weights = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_weights,
        shared_intermediate_hidden, hidden,
        shared_block_k, config.load_block_n,
        static_cast<int>(shared_l2_weights.stride(-2)),
        config.swizzle_weights_mode) : tensor_map_l2_weights;

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->mutable_data_ptr<int>();

    const auto num_sms = get_num_sms_for_mega_moe();

    // Compile
    const auto kernel = jit->compile("sm100_fp4_fp4_mega_moe", std::format(R"(
#include <deep_gemm/impls/sm100_fp4_fp4_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp4_fp4_mega_moe_impl<
        {},
        {}, {},
        {}, {},
        {},
        {}, {}, {},
        {},
        {}, {},
        {},
        {},
        {},
        {},
        {}, {}, {},
        {}, {},
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
        config.sf_block_m, config.sf_block_n,
        config.num_ring_tokens,
        config.num_sf_ring_tokens,
        config.num_stages,
        config.num_bytes_per_pull,
        config.num_dispatch_threads, config.num_non_epilogue_threads, config.num_epilogue_threads,
        num_sms, num_ranks,
        to_string(activation_clamp),
        fast_math ? "true" : "false",
        post_expert_norm_gamma != nullptr ? "true" : "false",
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
        l1_alphas, l2_alphas, a2_scales,
        post_expert_norm_gamma, post_expert_norm_eps,
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
        tensor_map_shared_l1_weights,
        tensor_map_shared_l1_output,
        tensor_map_shared_l2_acts,
        tensor_map_shared_l2_weights,
        sm_locality_domains.mutable_data_ptr<uint8_t>());
}

} // namespace deep_gemm
