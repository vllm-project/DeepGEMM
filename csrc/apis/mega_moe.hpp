#pragma once

#include <torch/library.h>
#include "../torch_library_utils.hpp"

#include <cmath>
#include <limits>
#include <tuple>
#include <string>

#include <deep_gemm/common/types.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>

#include "../runtime/runtime.hpp"
#include "../jit_kernels/impls/sm100_bf16_mega_moe.hpp"
#include "../jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp"
#include "../jit_kernels/impls/sm100_fp8_fp4_mega_moe_situ.hpp"

namespace deep_gemm::mega {

static int checked_int(const int64_t value) {
    // Preserve pybind behavior by rejecting values that do not fit exactly in a C++ int.
    DG_HOST_ASSERT(value >= std::numeric_limits<int>::min());
    DG_HOST_ASSERT(value <= std::numeric_limits<int>::max());
    return static_cast<int>(value);
}

static int get_token_alignment_for_mega_moe() {
    return layout::kLCMCandidateBlockM;
}

static int get_block_m_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const std::string& mma_type) {
    DG_HOST_ASSERT(num_tokens >= 0);
    const auto mma_kind = parse_mma_kind(mma_type);
    const auto [cluster_size, block_m, store_block_m, block_k, num_epilogue_threads] =
        get_block_config_for_mega_moe(num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens, mma_kind);
    return block_m;
}

struct SymmBufferLayoutInfo {
    int64_t num_bytes = 0;
    int64_t input_token_base = 0;
    int64_t input_sf_base = 0;
    int64_t input_topk_idx_base = 0;
    int64_t input_topk_weights_base = 0;
    int64_t shared_l1_sf_base = 0;
    int64_t shared_l2_token_base = 0;
    int64_t shared_l2_sf_base = 0;
    int64_t l1_token_base = 0;
    int64_t l1_sf_base = 0;
    int64_t l2_token_base = 0;
    int64_t l2_sf_base = 0;
    bool with_sf = false;
    int num_max_tokens_per_rank = 0;
    int num_topk = 0;
    int hidden = 0;
    int intermediate_hidden = 0;
    int num_shared_experts = 0;
    int shared_intermediate_hidden = 0;
    int num_ring_tokens = 0;
    int num_sf_ring_tokens = 0;

    // Flatten into a plain `int[]` so it can cross the TORCH_LIBRARY boundary
    std::vector<int64_t> to_int_list() const {
        return {
            num_bytes, input_token_base, input_sf_base, input_topk_idx_base, input_topk_weights_base,
            shared_l1_sf_base, shared_l2_token_base, shared_l2_sf_base,
            l1_token_base, l1_sf_base, l2_token_base, l2_sf_base,
            static_cast<int64_t>(with_sf),
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden, num_shared_experts, shared_intermediate_hidden,
            num_ring_tokens, num_sf_ring_tokens,
        };
    }

    static SymmBufferLayoutInfo from_int_list(const std::vector<int64_t>& values) {
        DG_HOST_ASSERT(static_cast<int64_t>(values.size()) == 21);
        SymmBufferLayoutInfo info;
        info.num_bytes = values[0];
        info.input_token_base = values[1];
        info.input_sf_base = values[2];
        info.input_topk_idx_base = values[3];
        info.input_topk_weights_base = values[4];
        info.shared_l1_sf_base = values[5];
        info.shared_l2_token_base = values[6];
        info.shared_l2_sf_base = values[7];
        info.l1_token_base = values[8];
        info.l1_sf_base = values[9];
        info.l2_token_base = values[10];
        info.l2_sf_base = values[11];
        // `with_sf` is a bool, encoded as 0/1 since the list is all `int64_t`.
        info.with_sf = values[12] != 0;
        info.num_max_tokens_per_rank = checked_int(values[13]);
        info.num_topk = checked_int(values[14]);
        info.hidden = checked_int(values[15]);
        info.intermediate_hidden = checked_int(values[16]);
        info.num_shared_experts = checked_int(values[17]);
        info.shared_intermediate_hidden = checked_int(values[18]);
        info.num_ring_tokens = checked_int(values[19]);
        info.num_sf_ring_tokens = checked_int(values[20]);
        return info;
    }
};

static SymmBufferLayoutInfo build_symm_buffer_layout(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const std::string& mma_type, const std::string& activation,
    const int& num_shared_experts = 0) {
    DG_HOST_ASSERT(num_experts % num_ranks == 0);
    DG_HOST_ASSERT(activation == "swiglu" or activation == "situ");
    DG_HOST_ASSERT(num_shared_experts >= 0);

    // Ring capacity: worst-case live pool blocks over all candidate BLOCK_M; mirrors the kernel assert.
    // TODO: we temporarily assume the SM count is consistent with the runtime value
    const auto num_sms = runtime->get_num_sms();
    const auto num_experts_per_rank = num_experts / num_ranks;
    const auto num_active_topk = std::min(num_topk, num_experts_per_rank);
    const auto num_max_routed_tokens = num_max_tokens_per_rank * num_ranks * num_active_topk;

    // Shared
    const int shared_intermediate_hidden = intermediate_hidden * num_shared_experts;

    // Iterate all block candidates to get the maximum ring size
    int num_ring_tokens = 0;
    for (const auto& block_m: layout::kCandidateBlockM) {
        const auto num_pool_blocks = ceil_div(num_max_routed_tokens, block_m) + num_experts_per_rank;
        const auto num_live_pool_blocks = sched::get_num_max_live_pool_blocks(
            num_pool_blocks, num_sms, hidden, intermediate_hidden);
        num_ring_tokens = std::max(num_ring_tokens, num_live_pool_blocks * block_m);
    }
    num_ring_tokens = math::align(num_ring_tokens, layout::kLCMCandidateBlockM);

    // Parse MMA type
    const auto mma_kind = parse_mma_kind(mma_type);
    const auto with_sf = is_mma_with_sf(mma_kind);

    // Compute num_sf_ring_tokens (max across all candidate block sizes)
    int num_sf_ring_tokens = 0;
    if (with_sf) {
        for (auto block_m: layout::kCandidateBlockM) {
            num_sf_ring_tokens = std::max(
                num_sf_ring_tokens,
                layout::get_num_sf_ring_tokens(num_ring_tokens, block_m));
        }
    }

    // All buffers
    const auto mega_buffer = layout::MegaMoEBuffer(
        nullptr, hidden, intermediate_hidden,
        num_ranks, num_experts, num_max_tokens_per_rank,
        num_topk, num_ring_tokens, num_sf_ring_tokens, with_sf,
        num_shared_experts
    );

    // Check SF buffer requirements
    if (with_sf) {
        DG_HOST_ASSERT(hidden % 128 == 0 and intermediate_hidden % 128 == 0);
        DG_HOST_ASSERT(shared_intermediate_hidden % 128 == 0);
        DG_HOST_ASSERT(num_sf_ring_tokens % 4 == 0);
    }

    SymmBufferLayoutInfo layout_info;
    layout_info.num_bytes = mega_buffer.get_num_bytes();
    layout_info.input_token_base = reinterpret_cast<int64_t>(mega_buffer.input_token_buffer.base);
    layout_info.input_sf_base = reinterpret_cast<int64_t>(mega_buffer.input_sf_buffer.base);
    layout_info.input_topk_idx_base = reinterpret_cast<int64_t>(mega_buffer.input_topk_idx_buffer.base);
    layout_info.input_topk_weights_base = reinterpret_cast<int64_t>(mega_buffer.input_topk_weights_buffer.base);
    layout_info.shared_l1_sf_base = reinterpret_cast<int64_t>(mega_buffer.shared_l1_sf_buffer.base);
    layout_info.shared_l2_token_base = reinterpret_cast<int64_t>(mega_buffer.shared_l2_token_buffer.base);
    layout_info.shared_l2_sf_base = reinterpret_cast<int64_t>(mega_buffer.shared_l2_sf_buffer.base);
    layout_info.l1_token_base = reinterpret_cast<int64_t>(mega_buffer.l1_token_buffer.base);
    layout_info.l1_sf_base = reinterpret_cast<int64_t>(mega_buffer.l1_sf_buffer.base);
    layout_info.l2_token_base = reinterpret_cast<int64_t>(mega_buffer.l2_token_buffer.base);
    layout_info.l2_sf_base = reinterpret_cast<int64_t>(mega_buffer.l2_sf_buffer.base);
    layout_info.with_sf = with_sf;
    layout_info.num_max_tokens_per_rank = num_max_tokens_per_rank;
    layout_info.num_topk = num_topk;
    layout_info.hidden = hidden;
    layout_info.intermediate_hidden = intermediate_hidden;
    layout_info.num_shared_experts = num_shared_experts;
    layout_info.shared_intermediate_hidden = shared_intermediate_hidden;
    layout_info.num_ring_tokens = num_ring_tokens;
    layout_info.num_sf_ring_tokens = num_sf_ring_tokens;
    return layout_info;
}

static std::tuple<int64_t, std::vector<int64_t>> get_symm_buffer_size_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const std::string& mma_type, const std::string& activation,
    const int& num_shared_experts = 0) {
    const auto layout_info = build_symm_buffer_layout(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden, mma_type, activation, num_shared_experts);
    return std::make_tuple(layout_info.num_bytes, layout_info.to_int_list());
}

using SymmBufferSlice = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                   at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                   at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

static SymmBufferSlice slice_symm_buffer_from_layout(
    const torch::Tensor& buffer, const SymmBufferLayoutInfo& layout_info) {
        auto x = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), layout_info.input_token_base),
            {layout_info.num_max_tokens_per_rank, layout_info.hidden},
            torch::TensorOptions().dtype(layout_info.with_sf ? torch::kFloat8_e4m3fn : torch::kBFloat16).device(buffer.device()));
        auto x_sf = layout_info.with_sf ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), layout_info.input_sf_base),
            {layout_info.num_max_tokens_per_rank, layout_info.hidden / 128},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        auto topk_idx = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), layout_info.input_topk_idx_base),
            {layout_info.num_max_tokens_per_rank, layout_info.num_topk},
            torch::TensorOptions().dtype(torch::kInt64).device(buffer.device()));
        auto topk_weights = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), layout_info.input_topk_weights_base),
            {layout_info.num_max_tokens_per_rank, layout_info.num_topk},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));

        auto shared_l1_acts = x;
        auto shared_l1_acts_sf = (layout_info.with_sf and layout_info.num_shared_experts > 0) ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), layout_info.shared_l1_sf_base),
            {layout::get_num_max_shared_sf_tokens(layout_info.num_max_tokens_per_rank), layout_info.hidden / 128},
            {1, layout::get_num_max_shared_sf_tokens(layout_info.num_max_tokens_per_rank)},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        auto shared_l2_acts = layout_info.num_shared_experts > 0 ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), layout_info.shared_l2_token_base),
            {layout_info.num_max_tokens_per_rank, layout_info.shared_intermediate_hidden},
            torch::TensorOptions().dtype(layout_info.with_sf ? torch::kFloat8_e4m3fn : torch::kBFloat16).device(buffer.device())) : torch::Tensor();
        auto shared_l2_acts_sf = (layout_info.with_sf and layout_info.num_shared_experts > 0) ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), layout_info.shared_l2_sf_base),
            {layout::get_num_max_shared_sf_tokens(layout_info.num_max_tokens_per_rank), layout_info.shared_intermediate_hidden / 128},
            {1, layout::get_num_max_shared_sf_tokens(layout_info.num_max_tokens_per_rank)},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();

        auto l1_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), layout_info.l1_token_base),
            {layout_info.num_ring_tokens, layout_info.hidden},
            torch::TensorOptions().dtype(layout_info.with_sf ? torch::kFloat8_e4m3fn : torch::kBFloat16).device(buffer.device()));
        auto l1_acts_sf = layout_info.with_sf ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), layout_info.l1_sf_base),
            {layout_info.num_sf_ring_tokens, layout_info.hidden / 128},
            {1, layout_info.num_sf_ring_tokens},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        auto l2_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), layout_info.l2_token_base),
            {layout_info.num_ring_tokens, layout_info.intermediate_hidden},
            torch::TensorOptions().dtype(layout_info.with_sf ? torch::kFloat8_e4m3fn : torch::kBFloat16).device(buffer.device()));
        auto l2_acts_sf = layout_info.with_sf ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), layout_info.l2_sf_base),
            {layout_info.num_sf_ring_tokens, layout_info.intermediate_hidden / 128},
            {1, layout_info.num_sf_ring_tokens},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        return std::make_tuple(x, x_sf, topk_idx, topk_weights,
                               shared_l1_acts, shared_l1_acts_sf, shared_l2_acts, shared_l2_acts_sf,
                               l1_acts, l1_acts_sf, l2_acts, l2_acts_sf);
}

static void fp8_fp4_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& shared_l1_weights_tuple_opt,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& shared_l2_weights_tuple_opt,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math,
    const float& activation_alpha,
    const float& activation_beta
) {
    const auto [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto [l2_weights, l2_weights_sf] = l2_weights_tuple;

    // Config checks
    const auto num_tokens = static_cast<int>(y.size(0));
    const auto [rm, rn, rk] = recipe;
    DG_HOST_ASSERT(rm == 1 and rn == 1 and rk == 32);
    DG_HOST_ASSERT(activation == "swiglu" or activation == "situ");
    DG_HOST_ASSERT(shared_l1_weights_tuple_opt.has_value() == shared_l2_weights_tuple_opt.has_value());

    // Activation checks
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);
    DG_HOST_ASSERT(std::isfinite(activation_alpha));
    DG_HOST_ASSERT(std::isfinite(activation_beta));
    if (activation == "situ") {
        DG_HOST_ASSERT(activation_alpha > 0);
        // Zero represents FlashInfer's optional, unset situ_linear_beta.
        DG_HOST_ASSERT(activation_beta >= 0);
    }

    // Tensor checks
    DG_HOST_ASSERT(get_major_type_ab(l1_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(l2_weights) == cute::UMMA::Major::K);
    const auto arch_major = jit->device.get_arch_major();
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] =
        check_grouped_ab_fp8_fp4(l1_weights, cute::UMMA::Major::K, arch_major);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] =
        check_grouped_ab_fp8_fp4(l2_weights, cute::UMMA::Major::K, arch_major);
    const auto weight_dtype = l1_weights.scalar_type();
    DG_HOST_ASSERT(weight_dtype == torch::kFloat8_e4m3fn or weight_dtype == kPackedFP4);
    DG_HOST_ASSERT(l2_weights.scalar_type() == weight_dtype);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());

    // Check weight SF layout for UE8M0 packing, MN-major, and TMA alignment
    constexpr int kGranMN = 1, kGranK = 32;
    check_sf_layout(l1_weights_sf, intermediate_hidden * 2, hidden, kGranMN, kGranK,
                    num_experts_per_rank, true, false, torch::kInt);
    check_sf_layout(l2_weights_sf, hidden, intermediate_hidden, kGranMN, kGranK,
                    num_experts_per_rank, true, false, torch::kInt);

    int num_shared_experts = 0, shared_intermediate_hidden = 0;
    torch::Tensor shared_l1_weights, shared_l1_weights_sf, shared_l2_weights, shared_l2_weights_sf;
    if (shared_l1_weights_tuple_opt.has_value()) {
        std::tie(shared_l1_weights, shared_l1_weights_sf) = shared_l1_weights_tuple_opt.value();
        std::tie(shared_l2_weights, shared_l2_weights_sf) = shared_l2_weights_tuple_opt.value();
        shared_intermediate_hidden = static_cast<int>(shared_l2_weights.size(1));
        num_shared_experts = shared_intermediate_hidden / intermediate_hidden;

        DG_HOST_ASSERT(shared_intermediate_hidden % intermediate_hidden == 0);
        DG_HOST_ASSERT(shared_l1_weights.dim() == 2 and shared_l2_weights.dim() == 2);
        DG_HOST_ASSERT(shared_l1_weights.size(0) == shared_intermediate_hidden * 2);
        DG_HOST_ASSERT(shared_l1_weights.size(1) == hidden);
        DG_HOST_ASSERT(shared_l2_weights.size(0) == hidden);
        DG_HOST_ASSERT(shared_l1_weights.scalar_type() == torch::kFloat8_e4m3fn);
        DG_HOST_ASSERT(shared_l2_weights.scalar_type() == torch::kFloat8_e4m3fn);
        DG_HOST_ASSERT(shared_l1_weights.is_contiguous() and shared_l2_weights.is_contiguous());
        DG_HOST_ASSERT(get_major_type_ab(shared_l1_weights) == cute::UMMA::Major::K);
        DG_HOST_ASSERT(get_major_type_ab(shared_l2_weights) == cute::UMMA::Major::K);
        check_sf_layout(shared_l1_weights_sf, shared_intermediate_hidden * 2, hidden, kGranMN, kGranK,
                        std::nullopt, true, false, torch::kInt);
        check_sf_layout(shared_l2_weights_sf, hidden, shared_intermediate_hidden, kGranMN, kGranK,
                        std::nullopt, true, false, torch::kInt);
    }

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    // Check buffer bytes
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto layout_info = build_symm_buffer_layout(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        weight_dtype == torch::kFloat8_e4m3fn ? "fp8xfp8" : "fp8xfp4",
        activation, num_shared_experts
    );
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(layout_info.num_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    // Already registered tensors
    const auto [x, x_sf, topk_idx, topk_weights,
                shared_l1_acts, shared_l1_acts_sf, shared_l2_acts, shared_l2_acts_sf,
                l1_acts, l1_acts_sf, l2_acts, l2_acts_sf] =
        slice_symm_buffer_from_layout(sym_buffer, layout_info);

    // Dispatch into different architectures
    if (arch_major == 10) {
        const auto launch = [&](const auto kernel, const auto&... activation_args) {
            kernel(y,
                   l1_acts, l1_acts_sf,
                   l2_acts, l2_acts_sf,
                   shared_l1_acts, shared_l1_acts_sf,
                   shared_l2_acts, shared_l2_acts_sf,
                   l1_weights, l2_weights,
                   l1_weights_sf, l2_weights_sf,
                   shared_l1_weights, shared_l2_weights,
                   shared_l1_weights_sf, shared_l2_weights_sf,
                   cumulative_local_expert_recv_stats,
                   sym_buffer_ptrs,
                   rank_idx, num_max_tokens_per_rank,
                   num_experts_per_rank,
                   num_shared_experts,
                   num_tokens, num_topk,
                   hidden, intermediate_hidden,
                   activation_args...);
        };
        if (activation == "situ") {
            launch(sm100_fp8_fp4_mega_moe_situ,
                   activation_alpha, activation_beta, fast_math);
        } else {
            launch(sm100_fp8_fp4_mega_moe,
                   activation_clamp, activation_alpha, activation_beta, fast_math);
        }
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Zero the entire symmetric buffer for debug mode
    // NOTES: caller must re-copy inputs into the buffer before each kernel call
    if (deep_jit::get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

static void bf16_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_weights,
    const torch::Tensor& l2_weights,
    const std::optional<torch::Tensor>& shared_l1_weights_opt,
    const std::optional<torch::Tensor>& shared_l2_weights_opt,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math,
    const float& activation_alpha,
    const float& activation_beta
) {
    // Config checks
    const auto num_tokens = static_cast<int>(y.size(0));
    DG_HOST_ASSERT(activation == "swiglu");
    DG_HOST_ASSERT(shared_l1_weights_opt.has_value() == shared_l2_weights_opt.has_value());

    // Activation checks
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);
    DG_HOST_ASSERT(std::isfinite(activation_alpha));
    DG_HOST_ASSERT(std::isfinite(activation_beta));

    // Tensor checks
    DG_HOST_ASSERT(get_major_type_ab(l1_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(l2_weights) == cute::UMMA::Major::K);
    const auto arch_major = jit->device.get_arch_major();
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] = get_shape<3>(l1_weights);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] = get_shape<3>(l2_weights);
    DG_HOST_ASSERT(l1_weights.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(l2_weights.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());

    int num_shared_experts = 0, shared_intermediate_hidden = 0;
    torch::Tensor shared_l1_weights, shared_l2_weights;
    if (shared_l1_weights_opt.has_value()) {
        shared_l1_weights = shared_l1_weights_opt.value();
        shared_l2_weights = shared_l2_weights_opt.value();
        shared_intermediate_hidden = static_cast<int>(shared_l2_weights.size(1));
        num_shared_experts = shared_intermediate_hidden / intermediate_hidden;

        DG_HOST_ASSERT(shared_intermediate_hidden % intermediate_hidden == 0);
        DG_HOST_ASSERT(shared_l1_weights.dim() == 2 and shared_l2_weights.dim() == 2);
        DG_HOST_ASSERT(shared_l1_weights.size(0) == shared_intermediate_hidden * 2);
        DG_HOST_ASSERT(shared_l1_weights.size(1) == hidden);
        DG_HOST_ASSERT(shared_l2_weights.size(0) == hidden);
        DG_HOST_ASSERT(shared_l1_weights.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(shared_l2_weights.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(shared_l1_weights.is_contiguous() and shared_l2_weights.is_contiguous());
        DG_HOST_ASSERT(get_major_type_ab(shared_l1_weights) == cute::UMMA::Major::K);
        DG_HOST_ASSERT(get_major_type_ab(shared_l2_weights) == cute::UMMA::Major::K);
    }

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    // Check buffer bytes
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto layout_info = build_symm_buffer_layout(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        "bf16xbf16", activation, num_shared_experts
    );
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(layout_info.num_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    // Already registered tensors
    const auto [x, _x_sf, topk_idx, topk_weights,
                shared_l1_acts, _shared_l1_acts_sf, shared_l2_acts, _shared_l2_acts_sf,
                l1_acts, _l1_acts_sf, l2_acts, _l2_acts_sf] =
        slice_symm_buffer_from_layout(sym_buffer, layout_info);

    // Dispatch into different architectures
    if (arch_major == 10) {
        sm100_bf16_mega_moe(y,
                            l1_acts, l2_acts,
                            shared_l1_acts, shared_l2_acts,
                            l1_weights, l2_weights,
                            shared_l1_weights, shared_l2_weights,
                            cumulative_local_expert_recv_stats,
                            sym_buffer_ptrs,
                            rank_idx, num_max_tokens_per_rank,
                            num_experts_per_rank,
                            num_shared_experts,
                            num_tokens, num_topk,
                            hidden, intermediate_hidden,
                            activation_clamp, activation_alpha, activation_beta,
                            fast_math);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Zero the entire symmetric buffer for debug mode
    // NOTES: caller must re-copy inputs into the buffer before each kernel call
    if (deep_jit::get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

} // namespace deep_gemm::mega

namespace deep_gemm::torch_registration {

static std::tuple<int, int, int> checked_recipe(const std::vector<int64_t>& recipe) {
    DG_HOST_ASSERT(recipe.size() == 3);
    return {mega::checked_int(recipe[0]), mega::checked_int(recipe[1]), mega::checked_int(recipe[2])};
}

static std::optional<std::tuple<torch::Tensor, torch::Tensor>> make_optional_tensor_pair(
    const std::optional<torch::Tensor>& value,
    const std::optional<torch::Tensor>& scale_factor) {
    // Preserve pybind behavior: an optional tensor tuple must contain both tensors or neither.
    DG_HOST_ASSERT(value.has_value() == scale_factor.has_value());
    if (not value.has_value())
        return std::nullopt;
    return std::make_tuple(*value, *scale_factor);
}

static int64_t get_token_alignment_for_mega_moe() {
    return mega::get_token_alignment_for_mega_moe();
}

static int64_t get_block_m_for_mega_moe(
    const int64_t& num_ranks,
    const int64_t& num_experts,
    const int64_t& num_max_tokens_per_rank,
    const int64_t& num_tokens,
    const int64_t& num_topk,
    const std::string& mma_type) {
    return mega::get_block_m_for_mega_moe(
        mega::checked_int(num_ranks), mega::checked_int(num_experts),
        mega::checked_int(num_max_tokens_per_rank), mega::checked_int(num_tokens),
        mega::checked_int(num_topk), mma_type);
}

static void fp8_fp4_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_weights_tuple,
    const torch::Tensor& l1_weights_tuple_sf,
    const torch::Tensor& l2_weights_tuple,
    const torch::Tensor& l2_weights_tuple_sf,
    const std::optional<torch::Tensor>& shared_l1_weights_tuple_opt,
    const std::optional<torch::Tensor>& shared_l1_weights_tuple_opt_sf,
    const std::optional<torch::Tensor>& shared_l2_weights_tuple_opt,
    const std::optional<torch::Tensor>& shared_l2_weights_tuple_opt_sf,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int64_t& rank_idx,
    const int64_t& num_max_tokens_per_rank,
    const int64_t& num_experts,
    const int64_t& num_topk,
    const std::vector<int64_t>& recipe,
    const std::string& activation,
    const std::optional<double>& activation_clamp_opt,
    const bool& fast_math,
    const double& activation_alpha,
    const double& activation_beta) {
    mega::fp8_fp4_mega_moe(
        y, {l1_weights_tuple, l1_weights_tuple_sf}, {l2_weights_tuple, l2_weights_tuple_sf},
        make_optional_tensor_pair(shared_l1_weights_tuple_opt, shared_l1_weights_tuple_opt_sf),
        make_optional_tensor_pair(shared_l2_weights_tuple_opt, shared_l2_weights_tuple_opt_sf),
        cumulative_local_expert_recv_stats, sym_buffer, sym_buffer_ptrs,
        mega::checked_int(rank_idx), mega::checked_int(num_max_tokens_per_rank),
        mega::checked_int(num_experts), mega::checked_int(num_topk), checked_recipe(recipe),
        activation, activation_clamp_opt, fast_math,
        static_cast<float>(activation_alpha), static_cast<float>(activation_beta));
}

static void bf16_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_weights,
    const torch::Tensor& l2_weights,
    const std::optional<torch::Tensor>& shared_l1_weights_opt,
    const std::optional<torch::Tensor>& shared_l2_weights_opt,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int64_t& rank_idx,
    const int64_t& num_max_tokens_per_rank,
    const int64_t& num_experts,
    const int64_t& num_topk,
    const std::string& activation,
    const std::optional<double>& activation_clamp_opt,
    const bool& fast_math,
    const double& activation_alpha,
    const double& activation_beta) {
    mega::bf16_mega_moe(
        y, l1_weights, l2_weights, shared_l1_weights_opt, shared_l2_weights_opt,
        cumulative_local_expert_recv_stats, sym_buffer, sym_buffer_ptrs,
        mega::checked_int(rank_idx), mega::checked_int(num_max_tokens_per_rank),
        mega::checked_int(num_experts), mega::checked_int(num_topk), activation,
        activation_clamp_opt, fast_math,
        static_cast<float>(activation_alpha), static_cast<float>(activation_beta));
}

static std::tuple<int64_t, std::vector<int64_t>> get_symm_buffer_size_for_mega_moe(
    int64_t num_ranks, int64_t num_experts, int64_t num_max_tokens_per_rank,
    int64_t num_topk, int64_t hidden, int64_t intermediate_hidden,
    const std::string& mma_type, const std::string& activation, int64_t num_shared_experts) {
    return mega::get_symm_buffer_size_for_mega_moe(
        mega::checked_int(num_ranks), mega::checked_int(num_experts),
        mega::checked_int(num_max_tokens_per_rank), mega::checked_int(num_topk),
        mega::checked_int(hidden), mega::checked_int(intermediate_hidden),
        mma_type, activation, mega::checked_int(num_shared_experts));
}
static mega::SymmBufferSlice _slice_symm_buffer_for_mega_moe(
    const torch::Tensor& buffer,
    const std::vector<int64_t>& layout_info) {
    return mega::slice_symm_buffer_from_layout(
        buffer, mega::SymmBufferLayoutInfo::from_int_list(layout_info));
}
} // namespace deep_gemm::torch_registration

TORCH_LIBRARY_FRAGMENT(deep_gemm, m) {
    m.def("get_token_alignment_for_mega_moe() -> int");
    m.def("get_block_m_for_mega_moe(int num_ranks, int num_experts, int num_max_tokens_per_rank, int num_tokens, int num_topk, str mma_type) -> int");

    m.def("get_symm_buffer_size_for_mega_moe(int num_ranks, int num_experts, int num_max_tokens_per_rank, int num_topk, int hidden, int intermediate_hidden, str mma_type, str activation, int num_shared_experts) -> (int, int[])",
          TORCH_FN(deep_gemm::torch_registration::get_symm_buffer_size_for_mega_moe));
    m.def("_slice_symm_buffer_for_mega_moe(Tensor(a) buffer, int[] layout_info) -> (Tensor(a), Tensor(a), Tensor(a), Tensor(a), Tensor(a), Tensor(a), Tensor(a), Tensor(a), Tensor(a), Tensor(a), Tensor(a), Tensor(a))");

    m.def("fp8_fp4_mega_moe(Tensor(a!) y, Tensor l1_weights_tuple, Tensor l1_weights_tuple_sf, Tensor l2_weights_tuple, Tensor l2_weights_tuple_sf, Tensor? shared_l1_weights_tuple_opt, Tensor? shared_l1_weights_tuple_opt_sf, Tensor? shared_l2_weights_tuple_opt, Tensor? shared_l2_weights_tuple_opt_sf, Tensor(b!)? cumulative_local_expert_recv_stats, Tensor(c!) sym_buffer, int[] sym_buffer_ptrs, int rank_idx, int num_max_tokens_per_rank, int num_experts, int num_topk, int[3] recipe, str activation, float? activation_clamp_opt, bool fast_math, float activation_alpha, float activation_beta) -> ()");
    m.def("bf16_mega_moe(Tensor(a!) y, Tensor l1_weights, Tensor l2_weights, Tensor? shared_l1_weights_opt, Tensor? shared_l2_weights_opt, Tensor(b!)? cumulative_local_expert_recv_stats, Tensor(c!) sym_buffer, int[] sym_buffer_ptrs, int rank_idx, int num_max_tokens_per_rank, int num_experts, int num_topk, str activation, float? activation_clamp_opt, bool fast_math, float activation_alpha, float activation_beta) -> ()");
}

TORCH_LIBRARY_IMPL(deep_gemm, CatchAll, m) {
    m.impl("get_token_alignment_for_mega_moe", TORCH_FN(deep_gemm::torch_registration::get_token_alignment_for_mega_moe));
    m.impl("get_block_m_for_mega_moe", TORCH_FN(deep_gemm::torch_registration::get_block_m_for_mega_moe));
}

TORCH_LIBRARY_IMPL(deep_gemm, CUDA, m) {
    m.impl("_slice_symm_buffer_for_mega_moe", TORCH_FN(deep_gemm::torch_registration::_slice_symm_buffer_for_mega_moe));

    m.impl("fp8_fp4_mega_moe", TORCH_FN(deep_gemm::torch_registration::fp8_fp4_mega_moe));
    m.impl("bf16_mega_moe", TORCH_FN(deep_gemm::torch_registration::bf16_mega_moe));
}
