#pragma once

#include <cmath>
#include <functional>
#include <limits>
#include <pybind11/functional.h>

#include <deep_gemm/common/types.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/layout/nvfp4_mega_moe.cuh>

#include "../runtime/runtime.hpp"
#include "../jit_kernels/impls/sm100_nvfp4_mega_moe.hpp"

namespace deep_gemm::nvfp4_mega {

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

static std::tuple<int64_t, std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                                    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                                    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                                    torch::Tensor>(const torch::Tensor&)>>
get_symm_buffer_size_for_nvfp4_mega_moe(
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
    constexpr int packed_sf_k = 64;  // Four per-16 E4M3 scale bytes per int32.

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

    // Slice function: creates tensor views from the raw buffer.
    // NOTES: `x_sf` is K-major, while `l1_acts_sf` and `l2_acts_sf` are M-major
    auto slice_input_buffers = [=](const torch::Tensor& buffer) {
        auto x = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_token_buffer.base)),
            {num_max_tokens_per_rank, hidden / 2},
            torch::TensorOptions().dtype(kPackedFP4).device(buffer.device()));
        auto x_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_sf_buffer.base)),
            {num_max_tokens_per_rank, hidden / packed_sf_k},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device()));
        auto topk_idx = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_topk_idx_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kInt64).device(buffer.device()));
        auto topk_weights = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_topk_weights_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));

        auto shared_l1_acts = num_shared_experts > 0 ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.shared_l1_token_buffer.base)),
            {num_max_tokens_per_rank, hidden},
            torch::TensorOptions().dtype(shared_with_sf ? torch::kFloat8_e4m3fn : torch::kBFloat16).device(buffer.device())) : x;
        auto shared_l1_acts_sf = (shared_with_sf and num_shared_experts > 0) ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.shared_l1_sf_buffer.base)),
            {layout::get_num_max_shared_sf_tokens(num_max_tokens_per_rank), hidden / 128},
            {1, layout::get_num_max_shared_sf_tokens(num_max_tokens_per_rank)},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        auto shared_l2_acts = num_shared_experts > 0 ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.shared_l2_token_buffer.base)),
            {num_max_tokens_per_rank, shared_intermediate_hidden},
            torch::TensorOptions().dtype(shared_with_sf ? torch::kFloat8_e4m3fn : torch::kBFloat16).device(buffer.device())) : torch::Tensor();
        auto shared_l2_acts_sf = (shared_with_sf and num_shared_experts > 0) ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.shared_l2_sf_buffer.base)),
            {layout::get_num_max_shared_sf_tokens(num_max_tokens_per_rank), shared_intermediate_hidden / 128},
            {1, layout::get_num_max_shared_sf_tokens(num_max_tokens_per_rank)},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();

        auto l1_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l1_token_buffer.base)),
            {num_ring_tokens, hidden / 2},
            torch::TensorOptions().dtype(kPackedFP4).device(buffer.device()));
        auto l1_acts_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l1_sf_buffer.base)),
            {num_sf_ring_tokens, hidden / packed_sf_k},
            {1, num_sf_ring_tokens},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device()));
        auto l2_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l2_token_buffer.base)),
            {num_ring_tokens, intermediate_hidden / 2},
            torch::TensorOptions().dtype(kPackedFP4).device(buffer.device()));
        auto l2_acts_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l2_sf_buffer.base)),
            {num_sf_ring_tokens, intermediate_hidden / packed_sf_k},
            {1, num_sf_ring_tokens},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device()));
        auto x_scales = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_x_scales_buffer.base)),
            {num_max_tokens_per_rank},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
        return std::make_tuple(x, x_sf, topk_idx, topk_weights,
                               shared_l1_acts, shared_l1_acts_sf, shared_l2_acts, shared_l2_acts_sf,
                               l1_acts, l1_acts_sf, l2_acts, l2_acts_sf, x_scales);
    };
    return {mega_buffer.get_num_bytes(), slice_input_buffers};
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
    const float& l2_activation_scale,
    const bool& use_x_scales
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
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_nvfp4_mega_moe(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        num_shared_experts, shared_bf16
    );
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    // Already registered tensors
    const auto [x, x_sf, topk_idx, topk_weights,
                shared_l1_acts, shared_l1_acts_sf, shared_l2_acts, shared_l2_acts_sf,
                l1_acts, l1_acts_sf, l2_acts, l2_acts_sf, x_scales] = slice(sym_buffer);

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
                   l1_alpha_opt, l2_alpha_opt, l2_activation_scale, use_x_scales);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Zero the entire symmetric buffer for debug mode
    // NOTES: caller must re-copy inputs into the buffer before each kernel call
    if (deep_jit::get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

static void register_apis(pybind11::module_& m) {
    m.def("get_token_alignment_for_nvfp4_mega_moe", &get_token_alignment_for_nvfp4_mega_moe);
    m.def("get_block_m_for_nvfp4_mega_moe", &get_block_m_for_nvfp4_mega_moe);
    m.def("get_symm_buffer_size_for_nvfp4_mega_moe", &get_symm_buffer_size_for_nvfp4_mega_moe,
          pybind11::arg("num_ranks"), pybind11::arg("num_experts"),
          pybind11::arg("num_max_tokens_per_rank"), pybind11::arg("num_topk"),
          pybind11::arg("hidden"), pybind11::arg("intermediate_hidden"),
          pybind11::arg("num_shared_experts") = 0, pybind11::arg("shared_bf16") = false);
    m.def("nvfp4_mega_moe", &nvfp4_mega_moe,
          pybind11::arg("y"), pybind11::arg("l1_weights"), pybind11::arg("l2_weights"),
          pybind11::arg("shared_l1_weights"), pybind11::arg("shared_l2_weights"),
          pybind11::arg("cumulative_local_expert_recv_stats"), pybind11::arg("sym_buffer"),
          pybind11::arg("sym_buffer_ptrs"), pybind11::arg("rank_idx"),
          pybind11::arg("num_max_tokens_per_rank"), pybind11::arg("num_experts"), pybind11::arg("num_topk"),
          pybind11::arg("activation_clamp"),
          pybind11::arg("fast_math"), pybind11::arg("activation_alpha"), pybind11::arg("activation_beta"),
          pybind11::arg("l1_alpha") = pybind11::none(), pybind11::arg("l2_alpha") = pybind11::none(),
          pybind11::arg("l2_activation_scale") = 1.0f, pybind11::arg("use_x_scales") = false);
}

} // namespace deep_gemm::nvfp4_mega
