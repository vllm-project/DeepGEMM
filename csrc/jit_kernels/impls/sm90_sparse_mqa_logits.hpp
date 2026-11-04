#pragma once

#include <deep_gemm/layout/sm90_sparse_mqa_logits.cuh>

#include "runtime_utils.hpp"

namespace deep_gemm {

static void launch_sm90_sparse_mqa_logits_metadata(
    const torch::stable::Tensor& context_lens, const torch::stable::Tensor& block_table,
    const torch::stable::Tensor& sparse_indices, const torch::stable::Tensor& valid_sparse_tokens,
    const torch::stable::Tensor& metadata, const int page_kv, const int sparse_block_kv
) {
    using namespace layout::sm90_sparse_mqa_logits;
    const int num_rows = sparse_indices.size(0);
    const int num_sparse_blocks = sparse_indices.size(1);
    const int64_t num_metadata_blocks = static_cast<int64_t>(num_rows) * num_sparse_blocks;
    const auto kernel = jit->compile("sm90_paged_sparse_mqa_logits_metadata", std::format(R"(
#include <deep_gemm/impls/sm90_fp8_paged_sparse_mqa_logits.cuh>
using namespace deep_gemm;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_paged_sparse_mqa_logits_metadata<{}, {}>);
}};
)", page_kv, sparse_block_kv));
    jit->launch(kernel, {.grid_dim = dim3(ceil_div(num_metadata_blocks, int64_t(kNumThreads)), 1, 1),
                         .block_dim = dim3(kNumThreads, 1, 1)},
                num_rows, num_sparse_blocks, static_cast<int>(block_table.size(1)), block_table.stride(0),
                context_lens.mutable_data_ptr(), block_table.mutable_data_ptr(), sparse_indices.mutable_data_ptr(),
                valid_sparse_tokens.mutable_data_ptr(), metadata.mutable_data_ptr());
}

static void launch_sm90_paged_sparse_mqa_logits(
    const torch::stable::Tensor& q, const torch::stable::Tensor& kv_cache,
    const torch::stable::Tensor& weights, const torch::stable::Tensor& metadata,
    const torch::stable::Tensor& logits, const int sparse_block_kv
) {
    using namespace layout::sm90_sparse_mqa_logits;
    const int num_sparse_blocks = metadata.size(1) - 1;
    const int logits_width = num_sparse_blocks * sparse_block_kv;
    const bool is_cache_aligned = kv_cache.stride(0) % 16 == 0 and
                                 reinterpret_cast<uintptr_t>(kv_cache.mutable_data_ptr()) % 16 == 0;
    const auto kernel = jit->compile("sm90_fp8_paged_sparse_mqa_logits", std::format(R"(
#include <deep_gemm/impls/sm90_fp8_paged_sparse_mqa_logits.cuh>
using namespace deep_gemm;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_paged_sparse_mqa_logits<{}, {}, {}>);
}};
)", kv_cache.size(1), sparse_block_kv, is_cache_aligned ? "true" : "false"));
    jit->launch(kernel, {.grid_dim = dim3(q.size(0) * ceil_div(logits_width, kTileTokens), 1, 1),
                         .block_dim = dim3(kNumThreads, 1, 1)},
                q.mutable_data_ptr(), kv_cache.mutable_data_ptr(), weights.mutable_data_ptr(),
                metadata.mutable_data_ptr(), logits.mutable_data_ptr(), num_sparse_blocks,
                static_cast<int>(kv_cache.size(0)), kv_cache.stride(0), weights.stride(0),
                static_cast<int>(logits.stride(0)));
}

} // namespace deep_gemm
