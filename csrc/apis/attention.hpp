#pragma once

#include <format>
#include <map>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include "../utils/torch_compat.hpp"

#include "../utils/compatibility.hpp"

#include "../jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp"
#include "../jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm100_mqa_logits.hpp"
#include "../jit_kernels/impls/sm100_sparse_mqa_logits.hpp"
#include "../jit_kernels/impls/sm90_fp8_mqa_logits.hpp"

#include "layout.hpp"
#include "../torch_library_utils.hpp"
#include "sm120_dispatch.hpp"

namespace deep_gemm::attention {

static void fp8_gemm_nt_skip_head_mid(const std::pair<torch::stable::Tensor, torch::stable::Tensor>& a,
                                      const std::pair<torch::stable::Tensor, torch::stable::Tensor>& b,
                                      const torch::stable::Tensor& d,
                                      const std::tuple<int, int, int>& head_splits,
                                      std::optional<std::tuple<int, int, int>> recipe,
                                      const std::string& compiled_dims,
                                      const bool& disable_ue8m0_cast) {
    // Shape must be `[M, K] @ [N, K].T`
    const auto major_a = get_major_type_ab(a.first);
    const auto major_b = get_major_type_ab(b.first);
    if (fp8_fp4_requires_k_major(a.first, b.first)) {
        DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
        DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    }

    // D must be N-major
    check_major_type_cd(d);

    // Type and shape checks
    const auto [m , k ] = get_shape<2>(a.first);
    const auto [n , k_] = get_shape<2>(b.first);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0);
    DG_HOST_ASSERT(a.first.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn);
    DG_HOST_ASSERT(b.first.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn);
    DG_HOST_ASSERT(d.scalar_type() == torch::headeronly::ScalarType::BFloat16 or d.scalar_type() == torch::headeronly::ScalarType::Float);

    // Check head splits and N
    const auto [left, mid, right] = head_splits;
    DG_HOST_ASSERT(n % (left + right) == 0 and n_ == n + n / (left + right) * mid);

    // Do nothing if the problem is empty
    if (m == 0)
        return;

    // Transform SFA and SFB into compute-required layout
    const auto [sfa, sfb, gran_k_a, gran_k_b] = layout::transform_sf_pair_into_required_layout(
        a.second, b.second, m, n, k, recipe, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, disable_ue8m0_cast);
    DG_HOST_ASSERT(gran_k_a == 128 and gran_k_b == 128);

    // Dispatch into different implements
    const auto arch_major = jit->device.get_arch_major();
    const auto epilogue_type = std::format("epilogue::transform::EpilogueHeadSplits<{}, {}, {}>", left, mid, right);
    if (arch_major == 9 and sfa.scalar_type() == torch::headeronly::ScalarType::Float and std::get<1>(recipe.value()) != 1) {
        const auto major_sfb = get_major_type_ab(sfb);
        sm90_fp8_gemm_1d2d(a.first, sfa, b.first, sfb, std::nullopt, d, m, n, k, major_a, major_b, major_sfb, compiled_dims, epilogue_type);
    } else if (arch_major == 10 and sfa.scalar_type() == torch::headeronly::ScalarType::Int) {
        // NOTES: Only granularity 128 and FP8 are exposed in the API
        sm100_fp8_fp4_gemm_1d1d(a.first, sfa, b.first, sfb, std::nullopt, d, m, n, k,
                                128, 128, major_a, major_b, compiled_dims, epilogue_type);
    } else if (arch_major == 12 and sfa.scalar_type() == torch::headeronly::ScalarType::Int) {
        DG_HOST_ASSERT(sfb.scalar_type() == torch::headeronly::ScalarType::Int);
        sm120_fp8_fp4_gemm_1d1d(a.first, sfa, b.first, sfb, std::nullopt, d, m, n, k,
                                128, 128, major_a, major_b, compiled_dims, epilogue_type);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture or scaling factor types");
    }
}

static torch::stable::Tensor fp8_fp4_mqa_logits(const std::tuple<torch::stable::Tensor, std::optional<torch::stable::Tensor>>& q,
                                        const std::tuple<torch::stable::Tensor, torch::stable::Tensor>& kv,
                                        const torch::stable::Tensor& weights,
                                        const torch::stable::Tensor& cu_seq_len_k_start,
                                        const torch::stable::Tensor& cu_seq_len_k_end,
                                        const bool& clean_logits,
                                        const int& max_seqlen_k,
                                        const torch::headeronly::ScalarType& logits_dtype,
                                        const std::optional<torch::stable::Tensor>& schedule_meta = std::nullopt) {
    const auto [q_fp, q_sf] = q;
    const auto [kv_fp, kv_sf] = kv;
    const auto qk_dtype = q_fp.scalar_type();
    const bool is_fp4 = qk_dtype == kPackedFP4;
    const bool is_mx_sf = q_sf.has_value();
    DG_HOST_ASSERT(not is_fp4 or is_mx_sf);

    const auto arch_major = jit->device.get_arch_major();
    // Check Q
    const auto [seq_len, num_heads, head_dim] = get_logical_shape<3>(q_fp);
    DG_HOST_ASSERT((not is_fp4 and head_dim == 32) or head_dim == 64 or head_dim == 128);
    DG_HOST_ASSERT(q_fp.is_contiguous());
    DG_HOST_ASSERT(q_fp.scalar_type() == (is_fp4 ? kPackedFP4 : torch::headeronly::ScalarType::Float8_e4m3fn));

    // Check SF Q
    if (is_mx_sf) {
        DG_HOST_ASSERT(arch_major == 10 or (arch_major == 12 and is_fp4));
        DG_HOST_ASSERT(q_sf.has_value());
        auto [_seq_len, _num_heads] = get_shape<2>(q_sf.value());
        DG_HOST_ASSERT(seq_len == _seq_len and num_heads == _num_heads);
        DG_HOST_ASSERT(q_sf.value().is_contiguous());
        DG_HOST_ASSERT(q_sf.value().scalar_type() == torch::headeronly::ScalarType::Int);
    }

    // Check KV
    const auto [seq_len_kv, _head_dim] = get_logical_shape<2>(kv_fp);
    DG_HOST_ASSERT(head_dim == _head_dim);
    DG_HOST_ASSERT(kv_fp.is_contiguous());
    DG_HOST_ASSERT(kv_fp.scalar_type() == (is_fp4 ? kPackedFP4 : torch::headeronly::ScalarType::Float8_e4m3fn));

    // Check SF KV
    auto [_seq_len_kv] = get_shape<1>(kv_sf);
    DG_HOST_ASSERT(seq_len_kv == _seq_len_kv);
    DG_HOST_ASSERT(kv_sf.is_contiguous());
    DG_HOST_ASSERT(kv_sf.scalar_type() == (is_mx_sf ? torch::headeronly::ScalarType::Int : torch::headeronly::ScalarType::Float));

    // Check weights
    auto [_seq_len, _num_heads] = get_shape<2>(weights);
    DG_HOST_ASSERT(seq_len == _seq_len and num_heads == _num_heads);
    DG_HOST_ASSERT(weights.stride(1) == 1);
    DG_HOST_ASSERT(weights.scalar_type() == torch::headeronly::ScalarType::Float or (arch_major == 10 and weights.scalar_type() == torch::headeronly::ScalarType::BFloat16));
    DG_HOST_ASSERT(weights.scalar_type() != torch::headeronly::ScalarType::BFloat16 or logits_dtype == torch::headeronly::ScalarType::BFloat16);

    // Check cu_seq_len_k_start
    DG_HOST_ASSERT(cu_seq_len_k_start.size(0) == seq_len);
    DG_HOST_ASSERT(cu_seq_len_k_start.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_start.scalar_type() == torch::headeronly::ScalarType::Int);

    // Check cu_seq_len_k_end
    DG_HOST_ASSERT(cu_seq_len_k_end.size(0) == seq_len);
    DG_HOST_ASSERT(cu_seq_len_k_end.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_end.scalar_type() == torch::headeronly::ScalarType::Int);

    // Allocate output
    DG_HOST_ASSERT(num_heads > 0 and num_heads <= 128 and num_heads % 4 == 0);
    // SM120a: 2 groups x 64 KV rows = 128; SM90/SM100 use 256
    const int block_kv = (arch_major == 12) ? sm120::kMqaBlockKv : 256;
    const int block_q = 128 / num_heads;

    torch::stable::Tensor logits;
    int aligned_seq_len = align(seq_len, block_q), stride_logits;
    // Logits row stride must be 1024-byte aligned
    const int stride_logits_alignment = 1024 / static_cast<int>(torch_compat::element_size(logits_dtype));
    if (max_seqlen_k == 0) {
        stride_logits = align(seq_len_kv + block_kv, stride_logits_alignment);
        logits = torch::stable::new_empty(q_fp, {aligned_seq_len, stride_logits}, logits_dtype);
        logits = torch_compat::narrow(torch_compat::narrow(logits, 0, 0, seq_len), 1, 0, seq_len_kv);
    } else {
        stride_logits = align(align(max_seqlen_k, block_kv), stride_logits_alignment);
        logits = torch::stable::new_empty(q_fp, {aligned_seq_len, stride_logits}, logits_dtype);
        logits = torch_compat::narrow(torch_compat::narrow(logits, 0, 0, seq_len), 1, 0, max_seqlen_k);
        DG_HOST_ASSERT(not clean_logits);
    }

    // Dispatch implementation
    // NOTES: the logits cleaning is fused into the kernels
    if (arch_major == 10) {
        DG_HOST_ASSERT(qk_dtype == torch::headeronly::ScalarType::Float8_e4m3fn or qk_dtype == kPackedFP4);
        sm100_mqa_logits(q_fp, q_sf, kv_fp, kv_sf, weights, cu_seq_len_k_start, cu_seq_len_k_end, logits, logits_dtype,
                         seq_len, seq_len_kv, max_seqlen_k, stride_logits, num_heads, head_dim, block_q, block_kv,
                         clean_logits, is_mx_sf, qk_dtype, schedule_meta);
    } else if (arch_major == 9) {
        DG_HOST_ASSERT(not schedule_meta.has_value());
        DG_HOST_ASSERT(not is_mx_sf);
        DG_HOST_ASSERT(qk_dtype == torch::headeronly::ScalarType::Float8_e4m3fn);
        DG_HOST_ASSERT(num_heads == 32 or num_heads == 64);
        DG_HOST_ASSERT(weights.scalar_type() == torch::headeronly::ScalarType::Float);
        sm90_fp8_mqa_logits(q_fp, kv_fp, kv_sf, weights, cu_seq_len_k_start, cu_seq_len_k_end, logits, logits_dtype,
                            seq_len, seq_len_kv, max_seqlen_k, stride_logits, num_heads, head_dim, block_q, block_kv,
                            clean_logits);
    } else if (arch_major == 12) {
        // NOTES: the SM120 kernels have no fused logits cleaning, and this branch of upstream
        //        has dropped the standalone `smxx_clean_logits` kernel, so cleaning is refused.
        DG_HOST_ASSERT(not clean_logits);
        DG_HOST_ASSERT(not schedule_meta.has_value());
        DG_HOST_ASSERT(qk_dtype == torch::headeronly::ScalarType::Float8_e4m3fn or qk_dtype == kPackedFP4);
        DG_HOST_ASSERT(num_heads == 16 or num_heads == 32 or num_heads == 64);
        DG_HOST_ASSERT(weights.scalar_type() == torch::headeronly::ScalarType::Float);
        sm120_mqa_logits(q_fp, q_sf, kv_fp, kv_sf, weights, cu_seq_len_k_start, cu_seq_len_k_end, logits, logits_dtype,
                         seq_len, seq_len_kv, max_seqlen_k, stride_logits, num_heads, head_dim, block_q, block_kv,
                         is_mx_sf, qk_dtype);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
    return logits;
}

static torch::stable::Tensor get_mqa_logits_metadata(const torch::stable::Tensor& cu_seq_len_k_start,
                                             const torch::stable::Tensor& cu_seq_len_k_end,
                                             const int& num_kv_tokens, const int& num_heads) {
    DG_HOST_ASSERT(jit->device.get_arch_major() == 10);
    const int num_q_tokens = static_cast<int>(cu_seq_len_k_start.size(0));
    DG_HOST_ASSERT(num_q_tokens > 0 and cu_seq_len_k_end.size(0) == num_q_tokens);
    DG_HOST_ASSERT(cu_seq_len_k_start.is_cuda() and cu_seq_len_k_end.is_cuda());
    DG_HOST_ASSERT(cu_seq_len_k_start.is_contiguous() and cu_seq_len_k_end.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_start.scalar_type() == torch::headeronly::ScalarType::Int and cu_seq_len_k_end.scalar_type() == torch::headeronly::ScalarType::Int);
    DG_HOST_ASSERT(num_kv_tokens > 0);
    DG_HOST_ASSERT(num_heads > 0 and num_heads <= 128 and num_heads % 4 == 0);

    constexpr int split_kv = 256;
    const int block_q = 128 / num_heads;
    const int num_sms = runtime->get_num_sms();
    const int required_words = get_mqa_logits_metadata_num_words(num_q_tokens, block_q, num_sms);
    const auto schedule_meta = torch::stable::new_empty(cu_seq_len_k_start, {required_words});

    sm100_mqa_logits_metadata(cu_seq_len_k_start, cu_seq_len_k_end, schedule_meta,
                              num_q_tokens, num_kv_tokens, block_q, split_kv, num_sms);
    return schedule_meta;
}

static const torch::stable::Tensor& get_sparse_mqa_logits_workspace(const torch::stable::Tensor& reference,
                                                            const int num_q_tokens) {
    using namespace layout::sparse_mqa_logits;
    constexpr int kNumMaxQTokens = 1 << 20;
    constexpr int64_t kNumWorkspaceBytes =
        sizeof(WorkspaceState) + static_cast<int64_t>(kNumMaxQTokens) * sizeof(QBlockInfo);
    DG_HOST_ASSERT(num_q_tokens <= kNumMaxQTokens);
    const auto stream = torch_compat::stream_key(reference);

    static std::map<torch_compat::StreamKey, torch::stable::Tensor> workspaces;
    auto& workspace = workspaces[stream];
    if (not workspace.defined()) {
        // Warm up each stream before capture so one-time zeroing is not replayed with the graph.
        DG_HOST_ASSERT(not torch_compat::is_capturing(stream.second));
        workspace = torch::stable::new_zeros(reference, {kNumWorkspaceBytes}, torch::headeronly::ScalarType::Byte);
    }
    return workspace;
}

// Each sparse-index row starts with the valid blocks inferred from its KV length. This prefix must
// contain unique, strictly increasing absolute block indices within the corresponding KV range.
// With unaligned ks, block i starts at i * sparse_block_kv + ks % sparse_block_kv.
static torch::stable::Tensor get_sparse_mqa_logits_metadata(const torch::stable::Tensor& cu_seq_len_k_start,
                                                    const torch::stable::Tensor& cu_seq_len_k_end,
                                                    const int& num_kv_tokens,
                                                    const torch::stable::Tensor& sparse_kv_block_indices,
                                                    const torch::headeronly::ScalarType& qk_dtype,
                                                    const int& sparse_block_kv,
                                                    const bool& use_unaligned_ks) {
    DG_HOST_ASSERT(jit->device.get_arch_major() == 10);
    const auto [num_q_tokens, num_max_sparse_blocks] = get_shape<2>(sparse_kv_block_indices);
    DG_HOST_ASSERT(num_q_tokens > 0 and num_kv_tokens >= 0);
    DG_HOST_ASSERT(cu_seq_len_k_start.dim() == 1 and cu_seq_len_k_start.size(0) == num_q_tokens);
    DG_HOST_ASSERT(cu_seq_len_k_end.dim() == 1 and cu_seq_len_k_end.size(0) == num_q_tokens);
    DG_HOST_ASSERT(sparse_kv_block_indices.scalar_type() == torch::headeronly::ScalarType::Int and sparse_kv_block_indices.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_start.scalar_type() == torch::headeronly::ScalarType::Int and cu_seq_len_k_start.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_end.scalar_type() == torch::headeronly::ScalarType::Int and cu_seq_len_k_end.is_contiguous());

    const int split_kv = get_sparse_mqa_split_kv(qk_dtype);
    auto metadata = torch::stable::new_empty(sparse_kv_block_indices, {get_num_metadata_bytes(num_q_tokens, num_max_sparse_blocks, sparse_block_kv,
                                                         split_kv, false, runtime->get_num_sms())}, torch::headeronly::ScalarType::Byte);
    const auto& workspace = get_sparse_mqa_logits_workspace(metadata, num_q_tokens);
    launch_sm100_sparse_mqa_logits_metadata(false, use_unaligned_ks, 1, num_kv_tokens, 0,
                                            sparse_kv_block_indices, metadata, workspace, split_kv, sparse_block_kv,
                                            cu_seq_len_k_start.mutable_data_ptr<int>(), cu_seq_len_k_end.mutable_data_ptr<int>(),
                                            nullptr, nullptr, nullptr);
    return metadata;
}

// Queries belonging to one request must be consecutive. Each sparse-index row starts with unique,
// strictly increasing logical block indices within its context length. Paired queries must also
// have identical block-table rows.
static torch::stable::Tensor get_paged_sparse_mqa_logits_metadata(const torch::stable::Tensor& context_lens,
                                                          const torch::stable::Tensor& block_table,
                                                          const torch::stable::Tensor& indices,
                                                          const int& page_kv,
                                                          const torch::stable::Tensor& sparse_kv_block_indices,
                                                          const torch::headeronly::ScalarType& qk_dtype,
                                                          const int& sparse_block_kv) {
    DG_HOST_ASSERT(jit->device.get_arch_major() == 10);
    const auto [num_q_tokens, num_max_sparse_blocks] = get_shape<2>(sparse_kv_block_indices);
    DG_HOST_ASSERT(num_q_tokens > 0);
    DG_HOST_ASSERT(context_lens.numel() == num_q_tokens and context_lens.scalar_type() == torch::headeronly::ScalarType::Int and context_lens.is_contiguous());
    DG_HOST_ASSERT(block_table.dim() == 2 and block_table.size(0) == num_q_tokens and block_table.size(1) > 0 and
                   block_table.scalar_type() == torch::headeronly::ScalarType::Int and block_table.stride(1) == 1);
    DG_HOST_ASSERT(block_table.stride(0) <= std::numeric_limits<uint32_t>::max());
    DG_HOST_ASSERT(indices.dim() == 1 and indices.size(0) == num_q_tokens and
                   indices.scalar_type() == torch::headeronly::ScalarType::Int and indices.is_contiguous());
    DG_HOST_ASSERT(sparse_block_kv == 8 or sparse_block_kv == 16);
    DG_HOST_ASSERT(page_kv > 0 and page_kv % sparse_block_kv == 0);
    DG_HOST_ASSERT(sparse_kv_block_indices.scalar_type() == torch::headeronly::ScalarType::Int and sparse_kv_block_indices.is_contiguous());

    const int split_kv = get_sparse_mqa_split_kv(qk_dtype);
    auto metadata = torch::stable::new_empty(sparse_kv_block_indices, {get_num_metadata_bytes(num_q_tokens, num_max_sparse_blocks, sparse_block_kv,
                                                         split_kv, true, runtime->get_num_sms())}, torch::headeronly::ScalarType::Byte);
    const auto& workspace = get_sparse_mqa_logits_workspace(metadata, num_q_tokens);
    launch_sm100_sparse_mqa_logits_metadata(true, false, page_kv, 0,
                                            static_cast<int>(block_table.stride(0)), sparse_kv_block_indices,
                                            metadata, workspace, split_kv, sparse_block_kv, nullptr, nullptr,
                                            context_lens.mutable_data_ptr<int>(), block_table.mutable_data_ptr<int>(), indices.mutable_data_ptr<int>());
    return metadata;
}

// Skip metadata header validation to avoid synchronizing the stream
static torch::stable::Tensor fp8_fp4_sparse_mqa_logits(const std::tuple<torch::stable::Tensor, std::optional<torch::stable::Tensor>>& q,
                                               const std::tuple<torch::stable::Tensor, torch::stable::Tensor>& kv,
                                               const torch::stable::Tensor& weights,
                                               const torch::stable::Tensor& metadata,
                                               const int& num_max_sparse_blocks,
                                               const int& sparse_block_kv,
                                               const bool& use_unaligned_ks) {
    using namespace layout::sparse_mqa_logits;
    const auto [q_fp, q_sf_optional] = q;
    const auto [kv_fp, kv_sf] = kv;
    DG_HOST_ASSERT(jit->device.get_arch_major() == 10 and q_sf_optional.has_value());
    DG_HOST_ASSERT(num_max_sparse_blocks > 0 and num_max_sparse_blocks % 4 == 0 and num_max_sparse_blocks <= 4096);
    DG_HOST_ASSERT(sparse_block_kv == 8 or sparse_block_kv == 16);
    const auto& q_sf = q_sf_optional.value();
    const auto qk_dtype = q_fp.scalar_type();
    const bool is_fp4 = qk_dtype == kPackedFP4;

    const auto [num_q_tokens, num_heads, head_dim] = get_logical_shape<3>(q_fp);
    DG_HOST_ASSERT(num_q_tokens > 0 and num_heads == kNumHeads and head_dim == kHeadDim);
    DG_HOST_ASSERT((is_fp4 or qk_dtype == torch::headeronly::ScalarType::Float8_e4m3fn) and q_fp.is_contiguous());
    const auto [_num_q_tokens_sf, _num_heads_sf] = get_shape<2>(q_sf);
    DG_HOST_ASSERT(_num_q_tokens_sf == num_q_tokens and _num_heads_sf == num_heads);
    DG_HOST_ASSERT(q_sf.scalar_type() == torch::headeronly::ScalarType::Int and q_sf.is_contiguous());

    const auto [num_kv_tokens, kv_head_dim] = get_logical_shape<2>(kv_fp);
    DG_HOST_ASSERT(num_kv_tokens > 0 and kv_head_dim == head_dim and kv_fp.scalar_type() == qk_dtype and kv_fp.is_contiguous());
    const auto [_num_kv_tokens_sf] = get_shape<1>(kv_sf);
    DG_HOST_ASSERT(_num_kv_tokens_sf == num_kv_tokens and kv_sf.scalar_type() == torch::headeronly::ScalarType::Int and kv_sf.is_contiguous());

    const auto [_num_q_tokens_weights, _num_heads_weights] = get_shape<2>(weights);
    DG_HOST_ASSERT(_num_q_tokens_weights == num_q_tokens and _num_heads_weights == num_heads);
    DG_HOST_ASSERT(weights.scalar_type() == torch::headeronly::ScalarType::BFloat16 and weights.stride(1) == 1);

    const int num_output_tokens = num_max_sparse_blocks * sparse_block_kv;
    const int logits_stride = align(num_output_tokens, 1024 / static_cast<int>(sizeof(nv_bfloat16)));
    auto logits = torch::stable::new_empty(q_fp, {align<int>(num_q_tokens, kBlockQ), logits_stride}, torch::headeronly::ScalarType::BFloat16);
    logits = torch_compat::narrow(torch_compat::narrow(logits, 0, 0, num_q_tokens), 1, 0, num_output_tokens);
    launch_sm100_sparse_mqa_logits(false, use_unaligned_ks, sparse_block_kv,
                                   q_fp, q_sf, kv_fp, kv_sf, weights, metadata, logits);
    return logits;
}

// Skip metadata header validation to avoid synchronizing the stream
static torch::stable::Tensor fp8_fp4_paged_sparse_mqa_logits(const std::tuple<torch::stable::Tensor, std::optional<torch::stable::Tensor>>& q,
                                                     const torch::stable::Tensor& fused_kv_cache,
                                                     const torch::stable::Tensor& weights,
                                                     const torch::stable::Tensor& metadata,
                                                     const int& num_max_sparse_blocks,
                                                     const int& sparse_block_kv) {
    using namespace layout::sparse_mqa_logits;
    const auto [q_fp, q_sf_optional] = q;
    DG_HOST_ASSERT(jit->device.get_arch_major() == 10 and q_sf_optional.has_value());
    DG_HOST_ASSERT(num_max_sparse_blocks > 0 and num_max_sparse_blocks % 4 == 0 and num_max_sparse_blocks <= 4096);
    DG_HOST_ASSERT(sparse_block_kv == 8 or sparse_block_kv == 16);
    const auto& q_sf = q_sf_optional.value();
    const auto qk_dtype = q_fp.scalar_type();
    const bool is_fp4 = qk_dtype == kPackedFP4;

    const auto [num_q_tokens, next_n, num_heads, head_dim] = get_logical_shape<4>(q_fp);
    DG_HOST_ASSERT(num_q_tokens > 0 and next_n == 1 and num_heads == kNumHeads and head_dim == kHeadDim);
    DG_HOST_ASSERT((is_fp4 or qk_dtype == torch::headeronly::ScalarType::Float8_e4m3fn) and q_fp.is_contiguous());
    const auto [_num_q_tokens_sf, _next_n_sf, _num_heads_sf] = get_shape<3>(q_sf);
    DG_HOST_ASSERT(_num_q_tokens_sf == num_q_tokens and _next_n_sf == 1 and _num_heads_sf == num_heads);
    DG_HOST_ASSERT(q_sf.scalar_type() == torch::headeronly::ScalarType::Int and q_sf.is_contiguous());

    const auto [num_kv_pages, page_kv, num_kv_heads, head_dim_with_sf] = get_shape<4>(fused_kv_cache);
    DG_HOST_ASSERT(num_kv_pages > 0 and page_kv > 0 and page_kv % sparse_block_kv == 0 and num_kv_heads == 1 and
                   head_dim_with_sf == (is_fp4 ? head_dim / 2 : head_dim) + static_cast<int>(sizeof(int)));
    DG_HOST_ASSERT(fused_kv_cache.scalar_type() == torch::headeronly::ScalarType::Byte and fused_kv_cache.stride(1) == head_dim_with_sf and
                   fused_kv_cache.stride(3) == 1 and fused_kv_cache.stride(0) <= std::numeric_limits<int>::max() and
                   fused_kv_cache.stride(0) % 512 == 0);

    const auto [_num_q_tokens_weights, _num_heads_weights] = get_shape<2>(weights);
    DG_HOST_ASSERT(_num_q_tokens_weights == num_q_tokens and _num_heads_weights == num_heads);
    DG_HOST_ASSERT(weights.scalar_type() == torch::headeronly::ScalarType::BFloat16 and weights.stride(1) == 1);

    const int num_output_tokens = num_max_sparse_blocks * sparse_block_kv;
    const int logits_stride = align(num_output_tokens, 1024 / static_cast<int>(sizeof(nv_bfloat16)));
    auto logits = torch::stable::new_empty(q_fp, {align<int>(num_q_tokens, kBlockQ), logits_stride}, torch::headeronly::ScalarType::BFloat16);
    logits = torch_compat::narrow(torch_compat::narrow(logits, 0, 0, num_q_tokens), 1, 0, num_output_tokens);
    launch_sm100_sparse_mqa_logits(true, false, sparse_block_kv, q_fp, q_sf, fused_kv_cache, torch::stable::Tensor(),
                                   weights, metadata, logits);
    return logits;
}

static torch::stable::Tensor get_paged_mqa_logits_metadata(const torch::stable::Tensor& context_lens, int block_kv, int num_sms, const std::optional<torch::stable::Tensor>& indices) {
    // NOTES: Only 2D context lens is supported for now
    DG_HOST_ASSERT(context_lens.dim() == 2);
    const bool is_context_lens_2d = true;
    const int batch_size = context_lens.size(0);
    const int next_n = context_lens.size(1);
    const bool is_varlen = indices.has_value();
    DG_HOST_ASSERT(context_lens.scalar_type() == torch::headeronly::ScalarType::Int);
    DG_HOST_ASSERT(context_lens.is_contiguous());

    // Create metadata tensor
    auto schedule_metadata = torch::stable::new_empty(context_lens, {num_sms + 1, 2});

    // Dispatch implementation
    const auto arch_major = jit->device.get_arch_major();
    if (arch_major == 12) {
        DG_HOST_ASSERT(block_kv == 32 or block_kv == 64);
        DG_HOST_ASSERT(not is_varlen or (next_n == 1 and indices.value().dim() == 1 and
                                         indices.value().size(0) == batch_size and
                                         indices.value().is_contiguous() and
                                         indices.value().scalar_type() == torch::headeronly::ScalarType::Int));
        const int next_n_atom = (is_varlen or next_n >= 2) ? 2 : 1;
        sm120_paged_mqa_logits_metadata(context_lens, schedule_metadata, batch_size, next_n, block_kv,
                                        num_sms, is_context_lens_2d, (next_n + next_n_atom - 1) / next_n_atom,
                                        is_varlen, is_varlen ? indices.value().mutable_data_ptr<int>() : nullptr);
    } else if (is_varlen) {
        const auto& indices_tensor = indices.value();
        DG_HOST_ASSERT(arch_major == 10 and next_n == 1 and (block_kv == 32 or block_kv == 64 or block_kv == 128));
        DG_HOST_ASSERT(indices_tensor.dim() == 1 and indices_tensor.size(0) == batch_size);
        DG_HOST_ASSERT(indices_tensor.is_contiguous());
        DG_HOST_ASSERT(indices_tensor.scalar_type() == torch::headeronly::ScalarType::Int);
        sm100_paged_mqa_logits_metadata(context_lens, schedule_metadata, batch_size, batch_size * next_n, next_n, num_sms, is_context_lens_2d, true, indices_tensor.mutable_data_ptr<int>());
    } else if (arch_major == 10) {
        DG_HOST_ASSERT(block_kv == 32 or block_kv == 64 or block_kv == 128);
        sm100_paged_mqa_logits_metadata(context_lens, schedule_metadata, batch_size, batch_size * next_n, next_n, num_sms, is_context_lens_2d, false, nullptr);
    } else if (arch_major == 9) {
        DG_HOST_ASSERT(block_kv == 32 or block_kv == 64);
        // SM90 always schedules 64-row compute tiles. A 32-row page is paired
        // with the following physical page inside each compute tile.
        sm90_paged_mqa_logits_metadata(context_lens, schedule_metadata, batch_size, next_n,
                                       64, num_sms, is_context_lens_2d, 1, false, nullptr);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    return schedule_metadata;
}

static torch::stable::Tensor fp8_fp4_paged_mqa_logits(const std::tuple<torch::stable::Tensor, std::optional<torch::stable::Tensor>>& q,
                                              const torch::stable::Tensor& fused_kv_cache,
                                              const torch::stable::Tensor& weights,
                                              const torch::stable::Tensor& context_lens,
                                              const torch::stable::Tensor& block_table,
                                              const torch::stable::Tensor& schedule_meta,
                                              const int& max_context_len,
                                              const bool& clean_logits,
                                              const torch::headeronly::ScalarType& logits_dtype,
                                              const std::optional<torch::stable::Tensor>& indices) {
    const auto [q_fp, q_sf] = q;
    const auto qk_dtype = q_fp.scalar_type();
    const bool is_fp4 = qk_dtype == kPackedFP4;
    const bool is_mx_sf = q_sf.has_value();
    DG_HOST_ASSERT(not is_fp4 or is_mx_sf);
    // NOTES: cleaning is not supported for paged logits
    DG_HOST_ASSERT(not clean_logits);

    torch::stable::Tensor kv_cache, kv_cache_sf;
    int kv_cache_stride_bytes;
    int block_table_stride = block_table.stride(0);
    int num_sms = runtime->get_num_sms();
    const auto arch_major = jit->device.get_arch_major();

    // Check Q
    const auto [batch_size, next_n, num_heads, head_dim] = get_logical_shape<4>(q_fp);
    DG_HOST_ASSERT((not is_fp4 and head_dim == 32) or head_dim == 64 or head_dim == 128);
    DG_HOST_ASSERT(q_fp.is_contiguous());
    DG_HOST_ASSERT(q_fp.scalar_type() == (is_fp4 ? kPackedFP4 : torch::headeronly::ScalarType::Float8_e4m3fn));
    DG_HOST_ASSERT(next_n >= 1);

    // Check SF Q
    if (is_mx_sf) {
        DG_HOST_ASSERT(arch_major == 10 or (arch_major == 12 and is_fp4));
        DG_HOST_ASSERT(q_sf.has_value());
        auto [_batch_size, _next_n, _num_heads] = get_shape<3>(q_sf.value());
        DG_HOST_ASSERT(batch_size == _batch_size and next_n == _next_n and num_heads == _num_heads);
        DG_HOST_ASSERT(q_sf.value().is_contiguous());
        DG_HOST_ASSERT(q_sf.value().scalar_type() == torch::headeronly::ScalarType::Int);
    }

    // Check fused KV cache
    const auto [num_kv_blocks, block_kv, num_heads_kv, head_dim_with_sf] = get_shape<4>(fused_kv_cache);
    DG_HOST_ASSERT((arch_major == 10 and (block_kv == 32 or block_kv == 64 or block_kv == 128)) or
                   (arch_major == 9 and (block_kv == 32 or block_kv == 64)) or
                   (arch_major == 12 and (block_kv == 32 or block_kv == 64)));
    const int kv_head_dim = is_fp4 ? head_dim / 2 : head_dim;
    const int sf_bytes = static_cast<int>(is_mx_sf ? sizeof(int) : sizeof(float));
    DG_HOST_ASSERT(num_heads_kv == 1 and head_dim_with_sf == kv_head_dim + sf_bytes);
    DG_HOST_ASSERT(fused_kv_cache.stride(1) == head_dim_with_sf and fused_kv_cache.stride(3) == 1);
    DG_HOST_ASSERT(fused_kv_cache.scalar_type() == torch::headeronly::ScalarType::Byte);

    // Derive KV values and SF tensor
    kv_cache_stride_bytes = fused_kv_cache.stride(0);
    DG_HOST_ASSERT(kv_cache_stride_bytes % sf_bytes == 0);
    kv_cache = torch::stable::from_blob(
        fused_kv_cache.mutable_data_ptr(),
        {num_kv_blocks, block_kv, kv_head_dim},
        {kv_cache_stride_bytes, kv_head_dim, 1},
        fused_kv_cache.device(), is_fp4 ? kPackedFP4 : torch::headeronly::ScalarType::Float8_e4m3fn
    );
    kv_cache_sf = torch::stable::from_blob(
        fused_kv_cache.mutable_data_ptr<uint8_t>() + block_kv * kv_head_dim,
        {num_kv_blocks, block_kv},
        {kv_cache_stride_bytes / sf_bytes, 1},
        fused_kv_cache.device(), is_mx_sf ? torch::headeronly::ScalarType::Int : torch::headeronly::ScalarType::Float
    );

    // Check weights
    auto [_batch_size_next_n, _num_heads] = get_shape<2>(weights);
    DG_HOST_ASSERT(_batch_size_next_n == batch_size * next_n and _num_heads == num_heads);
    DG_HOST_ASSERT(weights.stride(1) == 1);
    DG_HOST_ASSERT(weights.scalar_type() == torch::headeronly::ScalarType::Float or (arch_major == 10 and weights.scalar_type() == torch::headeronly::ScalarType::BFloat16));
    DG_HOST_ASSERT(weights.scalar_type() != torch::headeronly::ScalarType::BFloat16 or logits_dtype == torch::headeronly::ScalarType::BFloat16);

    // Check block table
    auto [_batch_size, _max_block_len] = get_shape<2>(block_table);
    DG_HOST_ASSERT(_batch_size == batch_size);
    DG_HOST_ASSERT(block_table.stride(1) == 1);
    DG_HOST_ASSERT(block_table.scalar_type() == torch::headeronly::ScalarType::Int);

    // Check indices
    const bool is_varlen = indices.has_value();
    const auto indices_tensor = indices.value_or(torch::stable::Tensor());
    if (is_varlen) {
        DG_HOST_ASSERT((arch_major == 10 or arch_major == 12) and next_n == 1);
        DG_HOST_ASSERT(indices_tensor.dim() == 1 and indices_tensor.size(0) == batch_size);
        DG_HOST_ASSERT(indices_tensor.is_contiguous());
        DG_HOST_ASSERT(indices_tensor.scalar_type() == torch::headeronly::ScalarType::Int);
    }

    // Check schedule metadata. SM90 next_n=4 uses one scheduler entry per
    // two-CTA cluster rather than one entry per SM.
    auto [_schedule_meta_size, _meta_info_size] = get_shape<2>(schedule_meta);
    const int num_kv_multicast = (arch_major == 9 and next_n == 4) ? 2 : 1;
    DG_HOST_ASSERT(_schedule_meta_size == num_sms / num_kv_multicast + 1 and _meta_info_size == 2);
    DG_HOST_ASSERT(schedule_meta.is_contiguous());
    DG_HOST_ASSERT(schedule_meta.scalar_type() == torch::headeronly::ScalarType::Int);

    // Check context lengths
    // NOTES: Only 2D context lens is supported for now
    DG_HOST_ASSERT(context_lens.dim() == 2);
    const bool is_context_lens_2d = true;
    const auto [__batch_size, _next_n] = get_shape<2>(context_lens);
    DG_HOST_ASSERT(batch_size == __batch_size and next_n == _next_n);
    DG_HOST_ASSERT(context_lens.is_contiguous());
    DG_HOST_ASSERT(context_lens.scalar_type() == torch::headeronly::ScalarType::Int);

    // Allocate output
    DG_HOST_ASSERT(logits_dtype == torch::headeronly::ScalarType::Float or logits_dtype == torch::headeronly::ScalarType::BFloat16);
    // SM120a: 2 groups x 64 KV rows = 128; SM90/SM100 use 256
    const int split_kv = (arch_major == 12) ? sm120::kPagedSplitKv : 256;
    // Logits row stride must be 1024-byte aligned
    const int stride_logits_alignment = 1024 / static_cast<int>(torch_compat::element_size(logits_dtype));
    const auto aligned_max_context_len = align(align(max_context_len, split_kv), stride_logits_alignment);
    auto logits = torch::stable::new_empty(q_fp, {batch_size * next_n, aligned_max_context_len}, logits_dtype);
    logits = torch_compat::slice(logits, -1, 0, max_context_len);

    // Dispatch implementation
    if (arch_major == 10) {
        DG_HOST_ASSERT(qk_dtype == torch::headeronly::ScalarType::Float8_e4m3fn or qk_dtype == kPackedFP4);
        constexpr int splits_per_chunk = 16;
        sm100_paged_mqa_logits(q_fp, q_sf, kv_cache, kv_cache_sf, weights, context_lens, logits, block_table, indices_tensor, schedule_meta,
                               logits_dtype, batch_size, batch_size * next_n, next_n, num_heads, head_dim, num_kv_blocks, block_kv, is_context_lens_2d,
                               is_varlen, aligned_max_context_len, block_table_stride, num_sms, split_kv, splits_per_chunk,
                               is_mx_sf, qk_dtype);
    } else if (arch_major == 9) {
        DG_HOST_ASSERT(not is_mx_sf);
        DG_HOST_ASSERT(qk_dtype == torch::headeronly::ScalarType::Float8_e4m3fn);
        DG_HOST_ASSERT(num_heads == 32 or num_heads == 64);
        DG_HOST_ASSERT(weights.scalar_type() == torch::headeronly::ScalarType::Float);
        sm90_fp8_paged_mqa_logits(q_fp, kv_cache, kv_cache_sf, weights, context_lens, logits, block_table, indices_tensor, schedule_meta,
                                  logits_dtype, batch_size, next_n, num_heads, head_dim, num_kv_blocks, block_kv, is_context_lens_2d,
                                  is_varlen, aligned_max_context_len, block_table_stride, num_sms, split_kv);
    } else if (arch_major == 12) {
        DG_HOST_ASSERT(qk_dtype == torch::headeronly::ScalarType::Float8_e4m3fn or qk_dtype == kPackedFP4);
        DG_HOST_ASSERT(num_heads == 16 or num_heads == 32 or num_heads == 64);
        DG_HOST_ASSERT(weights.scalar_type() == torch::headeronly::ScalarType::Float);
        sm120_paged_mqa_logits(q_fp, q_sf, kv_cache, kv_cache_sf, weights, context_lens, logits, block_table, indices_tensor, schedule_meta,
                               logits_dtype, batch_size, batch_size * next_n, next_n, num_heads, head_dim, num_kv_blocks, block_kv, is_context_lens_2d,
                               is_varlen, aligned_max_context_len, block_table_stride, num_sms, split_kv,
                               is_mx_sf, qk_dtype);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
    return logits;
}


// Legacy API wrappers
static torch::stable::Tensor fp8_mqa_logits(const torch::stable::Tensor& q,
                                    const std::tuple<torch::stable::Tensor, torch::stable::Tensor>& kv,
                                    const torch::stable::Tensor& weights,
                                    const torch::stable::Tensor& cu_seq_len_k_start,
                                    const torch::stable::Tensor& cu_seq_len_k_end,
                                    const bool& clean_logits,
                                    const int& max_seqlen_k) {
    return fp8_fp4_mqa_logits(std::make_tuple(q, std::nullopt), kv, weights,
                              cu_seq_len_k_start, cu_seq_len_k_end,
                              clean_logits, max_seqlen_k, torch::headeronly::ScalarType::Float);
}

static torch::stable::Tensor fp8_paged_mqa_logits(const torch::stable::Tensor& q,
                                          const torch::stable::Tensor& fused_kv_cache,
                                          const torch::stable::Tensor& weights,
                                          const torch::stable::Tensor& context_lens,
                                          const torch::stable::Tensor& block_table,
                                          const torch::stable::Tensor& schedule_meta,
                                          const int& max_context_len,
                                          const bool& clean_logits,
                                          const std::optional<torch::stable::Tensor>& indices) {
    return fp8_fp4_paged_mqa_logits(std::make_tuple(q, std::nullopt), fused_kv_cache, weights,
                                    context_lens, block_table, schedule_meta,
                                    max_context_len, clean_logits, torch::headeronly::ScalarType::Float, indices);
}
}  // namespace deep_gemm::attention

namespace deep_gemm::torch_registration {
using namespace deep_gemm::torch_utils;

static void fp8_gemm_nt_skip_head_mid(
    const torch::stable::Tensor& a, const torch::stable::Tensor& sfa,
    const torch::stable::Tensor& b, const torch::stable::Tensor& sfb,
    const torch::stable::Tensor& d,
    const std::vector<int64_t>& head_splits,
    const std::optional<std::vector<int64_t>>& recipe,
    const std::string& compiled_dims,
    const bool& disable_ue8m0_cast) {
    attention::fp8_gemm_nt_skip_head_mid(
        {a, sfa}, {b, sfb}, d,
        list_to_tuple3(head_splits),
        list_to_recipe3(recipe),
        compiled_dims, disable_ue8m0_cast);
}

static torch::stable::Tensor fp8_fp4_mqa_logits(
    const torch::stable::Tensor& q, const std::optional<torch::stable::Tensor>& q_sf,
    const torch::stable::Tensor& kv, const torch::stable::Tensor& kv_sf,
    const torch::stable::Tensor& weights,
    const torch::stable::Tensor& cu_seq_len_k_start,
    const torch::stable::Tensor& cu_seq_len_k_end,
    const bool& clean_logits,
    const int64_t& max_seqlen_k,
    torch::headeronly::ScalarType logits_dtype,
    const std::optional<torch::stable::Tensor>& schedule_meta) {
    return attention::fp8_fp4_mqa_logits(
        std::make_tuple(q, q_sf),
        std::make_tuple(kv, kv_sf),
        weights, cu_seq_len_k_start, cu_seq_len_k_end,
        clean_logits, static_cast<int>(max_seqlen_k),
        logits_dtype, schedule_meta);
}

static torch::stable::Tensor get_paged_mqa_logits_metadata(
    const torch::stable::Tensor& context_lens, const int64_t& block_kv,
    const int64_t& num_sms, const std::optional<torch::stable::Tensor>& indices) {
    return attention::get_paged_mqa_logits_metadata(
        context_lens, static_cast<int>(block_kv),
        static_cast<int>(num_sms), indices);
}

static torch::stable::Tensor fp8_fp4_paged_mqa_logits(
    const torch::stable::Tensor& q, const std::optional<torch::stable::Tensor>& q_sf,
    const torch::stable::Tensor& kv_cache,
    const torch::stable::Tensor& weights,
    const torch::stable::Tensor& context_lens,
    const torch::stable::Tensor& block_table,
    const torch::stable::Tensor& schedule_meta,
    const int64_t& max_context_len,
    const bool& clean_logits,
    torch::headeronly::ScalarType logits_dtype,
    const std::optional<torch::stable::Tensor>& indices) {
    return attention::fp8_fp4_paged_mqa_logits(
        std::make_tuple(q, q_sf),
        kv_cache, weights, context_lens, block_table, schedule_meta,
        static_cast<int>(max_context_len), clean_logits,
        logits_dtype, indices);
}

static torch::stable::Tensor fp8_mqa_logits(
    const torch::stable::Tensor& q,
    const torch::stable::Tensor& kv, const torch::stable::Tensor& kv_sf,
    const torch::stable::Tensor& weights,
    const torch::stable::Tensor& cu_seq_len_k_start,
    const torch::stable::Tensor& cu_seq_len_k_end,
    const bool& clean_logits,
    const int64_t& max_seqlen_k) {
    return attention::fp8_mqa_logits(
        q, std::make_tuple(kv, kv_sf), weights,
        cu_seq_len_k_start, cu_seq_len_k_end,
        clean_logits, static_cast<int>(max_seqlen_k));
}

static torch::stable::Tensor fp8_paged_mqa_logits(
    const torch::stable::Tensor& q,
    const torch::stable::Tensor& kv_cache,
    const torch::stable::Tensor& weights,
    const torch::stable::Tensor& context_lens,
    const torch::stable::Tensor& block_table,
    const torch::stable::Tensor& schedule_meta,
    const int64_t& max_context_len,
    const bool& clean_logits,
    const std::optional<torch::stable::Tensor>& indices) {
    return attention::fp8_paged_mqa_logits(
        q, kv_cache, weights,
        context_lens, block_table, schedule_meta,
        static_cast<int>(max_context_len), clean_logits, indices);
}

static torch::stable::Tensor get_mqa_logits_metadata(
    const torch::stable::Tensor& cu_seq_len_k_start,
    const torch::stable::Tensor& cu_seq_len_k_end,
    const int64_t& num_kv_tokens,
    const int64_t& num_heads) {
    return attention::get_mqa_logits_metadata(
        cu_seq_len_k_start, cu_seq_len_k_end, num_kv_tokens, num_heads);
}

static torch::stable::Tensor get_sparse_mqa_logits_metadata(
    const torch::stable::Tensor& cu_seq_len_k_start,
    const torch::stable::Tensor& cu_seq_len_k_end,
    const int64_t& num_kv_tokens,
    const torch::stable::Tensor& sparse_kv_block_indices,
    const torch::headeronly::ScalarType& qk_dtype,
    const int64_t& sparse_block_kv,
    const bool& use_unaligned_ks) {
    return attention::get_sparse_mqa_logits_metadata(
        cu_seq_len_k_start, cu_seq_len_k_end, num_kv_tokens, sparse_kv_block_indices, qk_dtype, sparse_block_kv, use_unaligned_ks);
}

static torch::stable::Tensor get_paged_sparse_mqa_logits_metadata(
    const torch::stable::Tensor& context_lens,
    const torch::stable::Tensor& block_table,
    const torch::stable::Tensor& indices,
    const int64_t& page_kv,
    const torch::stable::Tensor& sparse_kv_block_indices,
    const torch::headeronly::ScalarType& qk_dtype,
    const int64_t& sparse_block_kv) {
    return attention::get_paged_sparse_mqa_logits_metadata(
        context_lens, block_table, indices, page_kv, sparse_kv_block_indices, qk_dtype, sparse_block_kv);
}

static torch::stable::Tensor fp8_fp4_sparse_mqa_logits(
    const torch::stable::Tensor& q,
    const std::optional<torch::stable::Tensor>& q_sf,
    const torch::stable::Tensor& kv,
    const torch::stable::Tensor& kv_sf,
    const torch::stable::Tensor& weights,
    const torch::stable::Tensor& metadata,
    const int64_t& num_max_sparse_blocks,
    const int64_t& sparse_block_kv,
    const bool& use_unaligned_ks) {
    return attention::fp8_fp4_sparse_mqa_logits(
        {q, q_sf}, {kv, kv_sf}, weights, metadata, num_max_sparse_blocks, sparse_block_kv, use_unaligned_ks);
}

static torch::stable::Tensor fp8_fp4_paged_sparse_mqa_logits(
    const torch::stable::Tensor& q,
    const std::optional<torch::stable::Tensor>& q_sf,
    const torch::stable::Tensor& fused_kv_cache,
    const torch::stable::Tensor& weights,
    const torch::stable::Tensor& metadata,
    const int64_t& num_max_sparse_blocks,
    const int64_t& sparse_block_kv) {
    return attention::fp8_fp4_paged_sparse_mqa_logits(
        {q, q_sf}, fused_kv_cache, weights, metadata, num_max_sparse_blocks, sparse_block_kv);
}
} // namespace deep_gemm::torch_registration

STABLE_TORCH_LIBRARY_FRAGMENT(deep_gemm, m) {
    m.def(
        "fp8_gemm_nt_skip_head_mid(Tensor a, Tensor sfa, Tensor b, Tensor sfb, Tensor(d!) d, int[3] head_splits, int[3]? recipe=None, str compiled_dims='nk', bool disable_ue8m0_cast=False) -> ()");
    m.def(
        "fp8_fp4_mqa_logits(Tensor q, Tensor? q_sf, Tensor kv, Tensor kv_sf, Tensor weights, Tensor cu_seq_len_k_start, Tensor cu_seq_len_k_end, bool clean_logits=True, int max_seqlen_k=0, ScalarType logits_dtype=float, Tensor? schedule_meta=None) -> Tensor");
    m.def("get_mqa_logits_metadata(Tensor cu_seq_len_k_start, Tensor cu_seq_len_k_end, int num_kv_tokens, int num_heads) -> Tensor");
    m.def("get_sparse_mqa_logits_metadata(Tensor cu_seq_len_k_start, Tensor cu_seq_len_k_end, int num_kv_tokens, Tensor sparse_kv_block_indices, ScalarType qk_dtype, int sparse_block_kv, bool use_unaligned_ks=False) -> Tensor");
    m.def("get_paged_sparse_mqa_logits_metadata(Tensor context_lens, Tensor block_table, Tensor indices, int page_kv, Tensor sparse_kv_block_indices, ScalarType qk_dtype, int sparse_block_kv) -> Tensor");
    m.def("fp8_fp4_sparse_mqa_logits(Tensor q, Tensor? q_sf, Tensor kv, Tensor kv_sf, Tensor weights, Tensor metadata, int num_max_sparse_blocks, int sparse_block_kv, bool use_unaligned_ks=False) -> Tensor");
    m.def("fp8_fp4_paged_sparse_mqa_logits(Tensor q, Tensor? q_sf, Tensor kv_cache, Tensor weights, Tensor metadata, int num_max_sparse_blocks, int sparse_block_kv) -> Tensor");
    m.def(
        "get_paged_mqa_logits_metadata(Tensor context_lens, int block_kv, int num_sms, Tensor? indices=None) -> Tensor");
    m.def(
        "fp8_fp4_paged_mqa_logits(Tensor q, Tensor? q_sf, Tensor kv_cache, Tensor weights, Tensor context_lens, Tensor block_table, Tensor schedule_meta, int max_context_len, bool clean_logits=False, ScalarType logits_dtype=float, Tensor? indices=None) -> Tensor");
    // Legacy API
    m.def(
        "fp8_mqa_logits(Tensor q, Tensor kv, Tensor kv_sf, Tensor weights, Tensor cu_seq_len_k_start, Tensor cu_seq_len_k_end, bool clean_logits=True, int max_seqlen_k=0) -> Tensor");
    m.def(
        "fp8_paged_mqa_logits(Tensor q, Tensor kv_cache, Tensor weights, Tensor context_lens, Tensor block_table, Tensor schedule_meta, int max_context_len, bool clean_logits=False, Tensor? indices=None) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(deep_gemm, CUDA, m) {
    using namespace deep_gemm::torch_registration;

    m.impl("fp8_gemm_nt_skip_head_mid", TORCH_BOX(&fp8_gemm_nt_skip_head_mid));
    m.impl("fp8_fp4_mqa_logits", TORCH_BOX(&fp8_fp4_mqa_logits));
    m.impl("get_mqa_logits_metadata", TORCH_BOX(&deep_gemm::torch_registration::get_mqa_logits_metadata));
    m.impl("get_sparse_mqa_logits_metadata", TORCH_BOX(&deep_gemm::torch_registration::get_sparse_mqa_logits_metadata));
    m.impl("get_paged_sparse_mqa_logits_metadata", TORCH_BOX(&deep_gemm::torch_registration::get_paged_sparse_mqa_logits_metadata));
    m.impl("fp8_fp4_sparse_mqa_logits", TORCH_BOX(&deep_gemm::torch_registration::fp8_fp4_sparse_mqa_logits));
    m.impl("fp8_fp4_paged_sparse_mqa_logits", TORCH_BOX(&deep_gemm::torch_registration::fp8_fp4_paged_sparse_mqa_logits));
    m.impl("get_paged_mqa_logits_metadata", TORCH_BOX(&get_paged_mqa_logits_metadata));
    m.impl("fp8_fp4_paged_mqa_logits", TORCH_BOX(&fp8_fp4_paged_mqa_logits));
    // Legacy API
    m.impl("fp8_mqa_logits", TORCH_BOX(&fp8_mqa_logits));
    m.impl("fp8_paged_mqa_logits", TORCH_BOX(&fp8_paged_mqa_logits));
}
