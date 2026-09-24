#pragma once

#include <torch/library.h>
#include "../torch_library_utils.hpp"

#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

#include <deep_gemm/common/types.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/layout/nvfp4_mega_moe.cuh>

#include "../runtime/runtime.hpp"
#include "../jit_kernels/impls/sm100_nvfp4_mega_moe.hpp"

namespace deep_gemm::nvfp4_mega {

static int checked_int(const int64_t value) {
    // Preserve pybind behavior by rejecting values that do not fit exactly in a C++ int.
    DG_HOST_ASSERT(value >= std::numeric_limits<int>::min());
    DG_HOST_ASSERT(value <= std::numeric_limits<int>::max());
    return static_cast<int>(value);
}

static int get_token_alignment_for_nvfp4_mega_moe() {
    return layout::kLCMCandidateBlockM;
}

static int get_block_m_for_nvfp4_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk) {
    DG_HOST_ASSERT(num_tokens >= 0);
    const auto [cluster_size, block_m, store_block_m, block_k, num_epilogue_threads] =
        get_block_config_for_nvfp4_mega_moe(num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens);
    return block_m;
}

struct SymmBufferLayoutInfo {
    int64_t num_bytes = 0;
    int64_t input_token_base = 0;
    int64_t input_sf_base = 0;
    int64_t input_topk_idx_base = 0;
    int64_t input_topk_weights_base = 0;
    int64_t shared_l1_token_base = 0;
    int64_t shared_l1_sf_base = 0;
    int64_t shared_l2_token_base = 0;
    int64_t shared_l2_sf_base = 0;
    int64_t l1_token_base = 0;
    int64_t l1_sf_base = 0;
    int64_t l2_token_base = 0;
    int64_t l2_sf_base = 0;
    bool shared_with_sf = false;
    int num_max_tokens_per_rank = 0;
    int num_topk = 0;
    int hidden = 0;
    int intermediate_hidden = 0;
    int num_shared_experts = 0;
    int shared_intermediate_hidden = 0;
    int num_ring_tokens = 0;
    int num_sf_ring_tokens = 0;

    std::vector<int64_t> to_int_list() const {
        return {
            num_bytes, input_token_base, input_sf_base, input_topk_idx_base,
            input_topk_weights_base, shared_l1_token_base, shared_l1_sf_base,
            shared_l2_token_base, shared_l2_sf_base, l1_token_base, l1_sf_base,
            l2_token_base, l2_sf_base, static_cast<int64_t>(shared_with_sf),
            num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden,
            num_shared_experts, shared_intermediate_hidden, num_ring_tokens,
            num_sf_ring_tokens,
        };
    }

    static SymmBufferLayoutInfo from_int_list(const std::vector<int64_t>& values) {
        DG_HOST_ASSERT(static_cast<int64_t>(values.size()) == 22);
        SymmBufferLayoutInfo info;
        info.num_bytes = values[0];
        info.input_token_base = values[1];
        info.input_sf_base = values[2];
        info.input_topk_idx_base = values[3];
        info.input_topk_weights_base = values[4];
        info.shared_l1_token_base = values[5];
        info.shared_l1_sf_base = values[6];
        info.shared_l2_token_base = values[7];
        info.shared_l2_sf_base = values[8];
        info.l1_token_base = values[9];
        info.l1_sf_base = values[10];
        info.l2_token_base = values[11];
        info.l2_sf_base = values[12];
        info.shared_with_sf = values[13] != 0;
        info.num_max_tokens_per_rank = checked_int(values[14]);
        info.num_topk = checked_int(values[15]);
        info.hidden = checked_int(values[16]);
        info.intermediate_hidden = checked_int(values[17]);
        info.num_shared_experts = checked_int(values[18]);
        info.shared_intermediate_hidden = checked_int(values[19]);
        info.num_ring_tokens = checked_int(values[20]);
        info.num_sf_ring_tokens = checked_int(values[21]);
        return info;
    }
};

static SymmBufferLayoutInfo build_symm_buffer_layout(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_shared_experts = 0, const bool shared_bf16 = false) {
    DG_HOST_ASSERT(num_experts % num_ranks == 0);
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

    const bool shared_with_sf = not shared_bf16;
    // Compute num_sf_ring_tokens (max across all candidate block sizes)
    int num_sf_ring_tokens = 0;
    for (auto block_m: layout::kCandidateBlockM) {
        num_sf_ring_tokens = std::max(
            num_sf_ring_tokens,
            layout::get_num_sf_ring_tokens(num_ring_tokens, block_m));
    }

    // All buffers
    const auto mega_buffer = layout::NVFP4MegaMoEBuffer(
        nullptr, hidden, intermediate_hidden,
        num_ranks, num_experts, num_max_tokens_per_rank,
        num_topk, num_ring_tokens, num_sf_ring_tokens,
        num_shared_experts, shared_bf16
    );

    // Check SF buffer requirements
    DG_HOST_ASSERT(hidden % 128 == 0 and intermediate_hidden % 128 == 0);
    DG_HOST_ASSERT(shared_intermediate_hidden % 128 == 0);
    DG_HOST_ASSERT(num_sf_ring_tokens % 4 == 0);

    SymmBufferLayoutInfo layout_info;
    layout_info.num_bytes = mega_buffer.get_num_bytes();
    layout_info.input_token_base = reinterpret_cast<int64_t>(mega_buffer.input_token_buffer.base);
    layout_info.input_sf_base = reinterpret_cast<int64_t>(mega_buffer.input_sf_buffer.base);
    layout_info.input_topk_idx_base = reinterpret_cast<int64_t>(mega_buffer.input_topk_idx_buffer.base);
    layout_info.input_topk_weights_base = reinterpret_cast<int64_t>(mega_buffer.input_topk_weights_buffer.base);
    layout_info.shared_l1_token_base = reinterpret_cast<int64_t>(mega_buffer.shared_l1_token_buffer.base);
    layout_info.shared_l1_sf_base = reinterpret_cast<int64_t>(mega_buffer.shared_l1_sf_buffer.base);
    layout_info.shared_l2_token_base = reinterpret_cast<int64_t>(mega_buffer.shared_l2_token_buffer.base);
    layout_info.shared_l2_sf_base = reinterpret_cast<int64_t>(mega_buffer.shared_l2_sf_buffer.base);
    layout_info.l1_token_base = reinterpret_cast<int64_t>(mega_buffer.l1_token_buffer.base);
    layout_info.l1_sf_base = reinterpret_cast<int64_t>(mega_buffer.l1_sf_buffer.base);
    layout_info.l2_token_base = reinterpret_cast<int64_t>(mega_buffer.l2_token_buffer.base);
    layout_info.l2_sf_base = reinterpret_cast<int64_t>(mega_buffer.l2_sf_buffer.base);
    layout_info.shared_with_sf = shared_with_sf;
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

static std::tuple<int64_t, std::vector<int64_t>> get_symm_buffer_size_for_nvfp4_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_shared_experts = 0, const bool shared_bf16 = false) {
    const auto layout_info = build_symm_buffer_layout(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden, num_shared_experts, shared_bf16);
    return std::make_tuple(layout_info.num_bytes, layout_info.to_int_list());
}

using SymmBufferSlice = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                   torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                   torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

static SymmBufferSlice slice_symm_buffer_from_layout(
    const torch::Tensor& buffer, const SymmBufferLayoutInfo& layout_info) {
        // `x_sf` is K-major, while `l1_acts_sf` and `l2_acts_sf` are M-major.
        auto x = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), layout_info.input_token_base),
            {layout_info.num_max_tokens_per_rank, layout_info.hidden / 2},
            torch::TensorOptions().dtype(kPackedFP4).device(buffer.device()));
        auto x_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), layout_info.input_sf_base),
            {layout_info.num_max_tokens_per_rank, layout_info.hidden / 64},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device()));
        auto topk_idx = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), layout_info.input_topk_idx_base),
            {layout_info.num_max_tokens_per_rank, layout_info.num_topk},
            torch::TensorOptions().dtype(torch::kInt64).device(buffer.device()));
        auto topk_weights = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), layout_info.input_topk_weights_base),
            {layout_info.num_max_tokens_per_rank, layout_info.num_topk},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));

        auto shared_l1_acts = layout_info.num_shared_experts > 0 ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), layout_info.shared_l1_token_base),
            {layout_info.num_max_tokens_per_rank, layout_info.hidden},
            torch::TensorOptions().dtype(layout_info.shared_with_sf ? torch::kFloat8_e4m3fn : torch::kBFloat16).device(buffer.device())) : x;
        auto shared_l1_acts_sf = (layout_info.shared_with_sf and layout_info.num_shared_experts > 0) ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), layout_info.shared_l1_sf_base),
            {layout::get_num_max_shared_sf_tokens(layout_info.num_max_tokens_per_rank), layout_info.hidden / 128},
            {1, layout::get_num_max_shared_sf_tokens(layout_info.num_max_tokens_per_rank)},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        auto shared_l2_acts = layout_info.num_shared_experts > 0 ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), layout_info.shared_l2_token_base),
            {layout_info.num_max_tokens_per_rank, layout_info.shared_intermediate_hidden},
            torch::TensorOptions().dtype(layout_info.shared_with_sf ? torch::kFloat8_e4m3fn : torch::kBFloat16).device(buffer.device())) : torch::Tensor();
        auto shared_l2_acts_sf = (layout_info.shared_with_sf and layout_info.num_shared_experts > 0) ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), layout_info.shared_l2_sf_base),
            {layout::get_num_max_shared_sf_tokens(layout_info.num_max_tokens_per_rank), layout_info.shared_intermediate_hidden / 128},
            {1, layout::get_num_max_shared_sf_tokens(layout_info.num_max_tokens_per_rank)},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();

        auto l1_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), layout_info.l1_token_base),
            {layout_info.num_ring_tokens, layout_info.hidden / 2},
            torch::TensorOptions().dtype(kPackedFP4).device(buffer.device()));
        auto l1_acts_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), layout_info.l1_sf_base),
            {layout_info.num_sf_ring_tokens, layout_info.hidden / 64},
            {1, layout_info.num_sf_ring_tokens},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device()));
        auto l2_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), layout_info.l2_token_base),
            {layout_info.num_ring_tokens, layout_info.intermediate_hidden / 2},
            torch::TensorOptions().dtype(kPackedFP4).device(buffer.device()));
        auto l2_acts_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), layout_info.l2_sf_base),
            {layout_info.num_sf_ring_tokens, layout_info.intermediate_hidden / 64},
            {1, layout_info.num_sf_ring_tokens},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device()));
        return std::make_tuple(x, x_sf, topk_idx, topk_weights,
                               shared_l1_acts, shared_l1_acts_sf, shared_l2_acts, shared_l2_acts_sf,
                               l1_acts, l1_acts_sf, l2_acts, l2_acts_sf);
}

static void nvfp4_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<std::tuple<torch::Tensor, std::optional<torch::Tensor>>>& shared_l1_weights_tuple_opt,
    const std::optional<std::tuple<torch::Tensor, std::optional<torch::Tensor>>>& shared_l2_weights_tuple_opt,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math,
    const float& activation_alpha,
    const float& activation_beta,
    const std::optional<torch::Tensor>& l1_alpha_opt,
    const std::optional<torch::Tensor>& l2_alpha_opt,
    const float& l2_activation_scale
) {
    const auto [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto [l2_weights, l2_weights_sf] = l2_weights_tuple;

    // Config checks
    const auto num_tokens = static_cast<int>(y.size(0));
    DG_HOST_ASSERT(std::isfinite(l2_activation_scale) and l2_activation_scale > 0);
    DG_HOST_ASSERT(shared_l1_weights_tuple_opt.has_value() == shared_l2_weights_tuple_opt.has_value());

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
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] =
        check_grouped_ab_fp8_fp4(l1_weights, cute::UMMA::Major::K, arch_major);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] =
        check_grouped_ab_fp8_fp4(l2_weights, cute::UMMA::Major::K, arch_major);
    const auto weight_dtype = l1_weights.scalar_type();
    DG_HOST_ASSERT(weight_dtype == kPackedFP4);
    DG_HOST_ASSERT(l2_weights.scalar_type() == weight_dtype);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());

    DG_HOST_ASSERT(y.dim() == 2 and y.size(1) == hidden);
    DG_HOST_ASSERT(y.scalar_type() == torch::kBFloat16 and y.is_contiguous() and y.is_cuda());
    DG_HOST_ASSERT(hidden % 128 == 0 and intermediate_hidden % 128 == 0);
    for (const auto& alpha: {l1_alpha_opt, l2_alpha_opt}) {
        if (alpha.has_value()) {
            DG_HOST_ASSERT(alpha->scalar_type() == torch::kFloat and alpha->is_contiguous());
            DG_HOST_ASSERT(alpha->dim() == 1 and alpha->numel() == num_experts_per_rank);
            DG_HOST_ASSERT(alpha->device() == y.device());
        }
    }

    // Packed E4M3 block-scale bytes, MN-major and TMA aligned.
    constexpr int kGranMN = 1;
    constexpr int kGranK = 16;
    check_sf_layout(l1_weights_sf, intermediate_hidden * 2, hidden, kGranMN, kGranK,
                    num_experts_per_rank, true, false, torch::kInt);
    check_sf_layout(l2_weights_sf, hidden, intermediate_hidden, kGranMN, kGranK,
                    num_experts_per_rank, true, false, torch::kInt);

    int num_shared_experts = 0, shared_intermediate_hidden = 0;
    bool shared_bf16 = false;
    torch::Tensor shared_l1_weights, shared_l1_weights_sf, shared_l2_weights, shared_l2_weights_sf;
    if (shared_l1_weights_tuple_opt.has_value()) {
        const auto& [w1, sf1] = shared_l1_weights_tuple_opt.value();
        const auto& [w2, sf2] = shared_l2_weights_tuple_opt.value();
        shared_l1_weights = w1, shared_l2_weights = w2;
        shared_l1_weights_sf = sf1.value_or(torch::Tensor());
        shared_l2_weights_sf = sf2.value_or(torch::Tensor());
        shared_bf16 = shared_l1_weights.scalar_type() == torch::kBFloat16;
        shared_intermediate_hidden = static_cast<int>(shared_l2_weights.size(1));
        num_shared_experts = shared_intermediate_hidden / intermediate_hidden;

        DG_HOST_ASSERT(shared_intermediate_hidden % intermediate_hidden == 0);
        DG_HOST_ASSERT(shared_l1_weights.dim() == 2 and shared_l2_weights.dim() == 2);
        DG_HOST_ASSERT(shared_l1_weights.size(0) == shared_intermediate_hidden * 2);
        DG_HOST_ASSERT(shared_l1_weights.size(1) == hidden);
        DG_HOST_ASSERT(shared_l2_weights.size(0) == hidden);
        DG_HOST_ASSERT(shared_bf16 or shared_l1_weights.scalar_type() == torch::kFloat8_e4m3fn);
        DG_HOST_ASSERT(shared_l2_weights.scalar_type() == shared_l1_weights.scalar_type());
        DG_HOST_ASSERT(shared_l1_weights.device() == y.device() and shared_l2_weights.device() == y.device());
        DG_HOST_ASSERT(shared_l1_weights.is_contiguous() and shared_l2_weights.is_contiguous());
        DG_HOST_ASSERT(get_major_type_ab(shared_l1_weights) == cute::UMMA::Major::K);
        DG_HOST_ASSERT(get_major_type_ab(shared_l2_weights) == cute::UMMA::Major::K);
        if (shared_bf16) {
            DG_HOST_ASSERT(not sf1.has_value() and not sf2.has_value());
        } else {
            DG_HOST_ASSERT(sf1.has_value() and sf2.has_value());
            check_sf_layout(shared_l1_weights_sf, shared_intermediate_hidden * 2, hidden, kGranMN, 32,
                            std::nullopt, true, false, torch::kInt);
            check_sf_layout(shared_l2_weights_sf, hidden, shared_intermediate_hidden, kGranMN, 32,
                            std::nullopt, true, false, torch::kInt);
        }
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
        num_shared_experts, shared_bf16
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
        sm100_nvfp4_mega_moe(y,
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
                   activation_clamp, activation_alpha, activation_beta, fast_math,
                   l1_alpha_opt, l2_alpha_opt, l2_activation_scale);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Zero the entire symmetric buffer for debug mode
    // NOTES: caller must re-copy inputs into the buffer before each kernel call
    if (deep_jit::get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

} // namespace deep_gemm::nvfp4_mega

namespace deep_gemm::torch_registration {

static int64_t get_token_alignment_for_nvfp4_mega_moe() {
    return nvfp4_mega::get_token_alignment_for_nvfp4_mega_moe();
}

static int64_t get_block_m_for_nvfp4_mega_moe(
    int64_t num_ranks, int64_t num_experts, int64_t num_max_tokens_per_rank,
    int64_t num_tokens, int64_t num_topk) {
    return nvfp4_mega::get_block_m_for_nvfp4_mega_moe(
        nvfp4_mega::checked_int(num_ranks), nvfp4_mega::checked_int(num_experts),
        nvfp4_mega::checked_int(num_max_tokens_per_rank),
        nvfp4_mega::checked_int(num_tokens), nvfp4_mega::checked_int(num_topk));
}

static std::tuple<int64_t, std::vector<int64_t>> get_symm_buffer_size_for_nvfp4_mega_moe(
    int64_t num_ranks, int64_t num_experts, int64_t num_max_tokens_per_rank,
    int64_t num_topk, int64_t hidden, int64_t intermediate_hidden,
    int64_t num_shared_experts, bool shared_bf16) {
    return nvfp4_mega::get_symm_buffer_size_for_nvfp4_mega_moe(
        nvfp4_mega::checked_int(num_ranks), nvfp4_mega::checked_int(num_experts),
        nvfp4_mega::checked_int(num_max_tokens_per_rank),
        nvfp4_mega::checked_int(num_topk), nvfp4_mega::checked_int(hidden),
        nvfp4_mega::checked_int(intermediate_hidden),
        nvfp4_mega::checked_int(num_shared_experts), shared_bf16);
}

static nvfp4_mega::SymmBufferSlice _slice_symm_buffer_for_nvfp4_mega_moe(
    const torch::Tensor& buffer, const std::vector<int64_t>& layout_info) {
    return nvfp4_mega::slice_symm_buffer_from_layout(
        buffer, nvfp4_mega::SymmBufferLayoutInfo::from_int_list(layout_info));
}

static void nvfp4_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_weights, const torch::Tensor& l1_weights_sf,
    const torch::Tensor& l2_weights, const torch::Tensor& l2_weights_sf,
    const std::optional<torch::Tensor>& shared_l1_weights,
    const std::optional<torch::Tensor>& shared_l1_weights_sf,
    const std::optional<torch::Tensor>& shared_l2_weights,
    const std::optional<torch::Tensor>& shared_l2_weights_sf,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer, const std::vector<int64_t>& sym_buffer_ptrs,
    int64_t rank_idx, int64_t num_max_tokens_per_rank,
    int64_t num_experts, int64_t num_topk,
    const std::optional<double>& activation_clamp, bool fast_math,
    double activation_alpha, double activation_beta,
    const std::optional<torch::Tensor>& l1_alpha,
    const std::optional<torch::Tensor>& l2_alpha,
    double l2_activation_scale) {
    DG_HOST_ASSERT(shared_l1_weights.has_value() == shared_l2_weights.has_value());
    std::optional<std::tuple<torch::Tensor, std::optional<torch::Tensor>>> shared_l1;
    std::optional<std::tuple<torch::Tensor, std::optional<torch::Tensor>>> shared_l2;
    if (shared_l1_weights.has_value()) {
        shared_l1 = std::make_tuple(*shared_l1_weights, shared_l1_weights_sf);
        shared_l2 = std::make_tuple(*shared_l2_weights, shared_l2_weights_sf);
    }
    nvfp4_mega::nvfp4_mega_moe(
        y, std::make_tuple(l1_weights, l1_weights_sf),
        std::make_tuple(l2_weights, l2_weights_sf), shared_l1, shared_l2,
        cumulative_local_expert_recv_stats, sym_buffer, sym_buffer_ptrs,
        nvfp4_mega::checked_int(rank_idx),
        nvfp4_mega::checked_int(num_max_tokens_per_rank),
        nvfp4_mega::checked_int(num_experts), nvfp4_mega::checked_int(num_topk),
        activation_clamp.has_value()
            ? std::optional<float>(static_cast<float>(*activation_clamp))
            : std::nullopt,
        fast_math, static_cast<float>(activation_alpha),
        static_cast<float>(activation_beta), l1_alpha, l2_alpha,
        static_cast<float>(l2_activation_scale));
}

} // namespace deep_gemm::torch_registration

TORCH_LIBRARY_FRAGMENT(deep_gemm, m) {
    m.def("get_token_alignment_for_nvfp4_mega_moe() -> int");
    m.def("get_block_m_for_nvfp4_mega_moe(int num_ranks, int num_experts, int num_max_tokens_per_rank, int num_tokens, int num_topk) -> int");
    m.def("get_symm_buffer_size_for_nvfp4_mega_moe(int num_ranks, int num_experts, int num_max_tokens_per_rank, int num_topk, int hidden, int intermediate_hidden, int num_shared_experts=0, bool shared_bf16=False) -> (int, int[])",
          TORCH_FN(deep_gemm::torch_registration::get_symm_buffer_size_for_nvfp4_mega_moe));
    m.def("_slice_symm_buffer_for_nvfp4_mega_moe(Tensor(a) buffer, int[] layout_info) -> (Tensor(a), Tensor(a), Tensor(a), Tensor(a), Tensor(a), Tensor(a), Tensor(a), Tensor(a), Tensor(a), Tensor(a), Tensor(a), Tensor(a))");
    m.def("nvfp4_mega_moe(Tensor(a!) y, Tensor l1_weights_tuple, Tensor l1_weights_tuple_sf, Tensor l2_weights_tuple, Tensor l2_weights_tuple_sf, Tensor? shared_l1_weights_tuple_opt, Tensor? shared_l1_weights_tuple_opt_sf, Tensor? shared_l2_weights_tuple_opt, Tensor? shared_l2_weights_tuple_opt_sf, Tensor(b!)? cumulative_local_expert_recv_stats, Tensor(c!) sym_buffer, int[] sym_buffer_ptrs, int rank_idx, int num_max_tokens_per_rank, int num_experts, int num_topk, float? activation_clamp_opt, bool fast_math, float activation_alpha, float activation_beta, Tensor? l1_alpha=None, Tensor? l2_alpha=None, float l2_activation_scale=1.0) -> ()");
}

TORCH_LIBRARY_IMPL(deep_gemm, CatchAll, m) {
    m.impl("get_token_alignment_for_nvfp4_mega_moe", TORCH_FN(deep_gemm::torch_registration::get_token_alignment_for_nvfp4_mega_moe));
    m.impl("get_block_m_for_nvfp4_mega_moe", TORCH_FN(deep_gemm::torch_registration::get_block_m_for_nvfp4_mega_moe));
}

TORCH_LIBRARY_IMPL(deep_gemm, CUDA, m) {
    m.impl("_slice_symm_buffer_for_nvfp4_mega_moe", TORCH_FN(deep_gemm::torch_registration::_slice_symm_buffer_for_nvfp4_mega_moe));
    m.impl("nvfp4_mega_moe", TORCH_FN(deep_gemm::torch_registration::nvfp4_mega_moe));
}
