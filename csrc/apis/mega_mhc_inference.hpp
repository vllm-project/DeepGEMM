#pragma once

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include "../utils/torch_compat.hpp"
#include "../torch_library_utils.hpp"

#include "../utils/compatibility.hpp"

#include "../jit_kernels/impls/sm100_mega_mhc_inference.hpp"
#include "../utils/layout.hpp"

namespace deep_gemm::mega_mhc {

namespace mhc_layout = layout::mega_mhc;

/**
 * Execute shifted mHC, generate the next mixes, and apply RMSNorm
 * with optional FP8 E4M3 casting.
 *
 * Args:
 *     x                              [T,H]                             BF16
 *     residual                       [T,hc_mult,H]                     BF16
 *     shifted_prev_mix               [T,hc_mult,1]                     FP32
 *     post_mix                       [T,hc_mult,1]                     FP32
 *     comb_res_mix                   [T,hc_mult,hc_mult]               FP32
 *     fn                             [hc_mult*(hc_mult+2),hc_mult*H]   FP32
 *     mix_scales                     [3]                               FP32
 *     mix_bases                      [hc_mult*(hc_mult+2)]             FP32
 *     hc_mult                        int
 *     hc_norm_eps                    float
 *     hc_pre_eps                     float
 *     hc_post_scale                  float
 *     sinkhorn_eps                   float
 *     num_sinkhorn_iters             int
 *     rmsnorm_weight                 [H]                               BF16
 *     rmsnorm_eps                    float
 *     rmsnorm_scale                  float
 *     new_residual                   [T,hc_mult,H]                     BF16
 *     new_prev_mix                   [T,hc_mult,1]                     FP32
 *     new_post_mix                   [T,hc_mult,1]                     FP32
 *     new_comb_res_mix               [T,hc_mult,hc_mult]               FP32
 *     y_bf16                         [T,H] BF16 | None
 *     y_fp8                          [T,H] E4M3 | None
 *     y_gemm_sf                      [T,H/128] int32 | None
 *     y_routed_sf                    [T,H/128] int32 | None
 *     y_shared_sf                    [T,H/128] int32 | None
 *     shared_sf_block_m              int
 *
 * Returns:
 *     None
 *
 * Notes:
 *     Split-K is selected per invocation: 16 with deterministic algorithms;
 *     otherwise a wave estimate sets the split budget.
 *     FP8 gran_k is 32, with four UE8M0 scaling factors packed into each int32.
 *     y_bf16 and y_fp8 are independent optional outputs; at least one is required.
 *     With y_fp8, supply exactly either y_gemm_sf alone, or y_routed_sf and
 *     y_shared_sf together. y_gemm_sf is TMA-aligned column-major,
 *     y_routed_sf is contiguous row-major, and y_shared_sf is the Mega MoE
 *     shared-expert layout and requires shared_sf_block_m > 0.
 *     Split barriers are allocated per stream and must be initialized before
 *     CUDA Graph capture; calls sharing them, including replays of graphs
 *     captured on that stream, must execute in order. The kernel runs on the configured
 *     SM count with a cooperative launch to guarantee grid-wide progress.
 */
static void mega_mhc(
    const torch::stable::Tensor& x,
    const torch::stable::Tensor& residual,
    const torch::stable::Tensor& shifted_prev_mix,
    const torch::stable::Tensor& post_mix,
    const torch::stable::Tensor& comb_res_mix,
    const torch::stable::Tensor& fn,
    const torch::stable::Tensor& mix_scales,
    const torch::stable::Tensor& mix_bases,
    const int& hc_mult,
    const float& hc_norm_eps,
    const float& hc_pre_eps,
    const float& hc_post_scale,
    const float& sinkhorn_eps,
    const int& num_sinkhorn_iters,
    const torch::stable::Tensor& rmsnorm_weight,
    const float& rmsnorm_eps,
    const float& rmsnorm_scale,
    const torch::stable::Tensor& new_residual,
    const torch::stable::Tensor& new_prev_mix,
    const torch::stable::Tensor& new_post_mix,
    const torch::stable::Tensor& new_comb_res_mix,
    const std::optional<torch::stable::Tensor>& y_bf16,
    const std::optional<torch::stable::Tensor>& y_fp8,
    const std::optional<torch::stable::Tensor>& y_gemm_sf,
    const std::optional<torch::stable::Tensor>& y_routed_sf,
    const std::optional<torch::stable::Tensor>& y_shared_sf,
    const int& shared_sf_block_m
) {
    // FP8 uses either GEMM SF or routed and shared SF together.
    DG_HOST_ASSERT(y_routed_sf.has_value() == y_shared_sf.has_value());
    DG_HOST_ASSERT(not (y_gemm_sf.has_value() and y_routed_sf.has_value()));
    DG_HOST_ASSERT(y_bf16.has_value() or y_fp8.has_value());
    DG_HOST_ASSERT(y_fp8.has_value() == (y_gemm_sf.has_value() or y_routed_sf.has_value()));
    DG_HOST_ASSERT(y_shared_sf.has_value() ? shared_sf_block_m > 0 : shared_sf_block_m == 0);

    const auto [num_tokens, hidden] = get_shape<2>(x);
    const auto [residual_tokens, residual_num_routes, residual_hidden] = get_shape<3>(residual);
    const auto [post_mix_tokens, post_mix_num_routes, post_mix_width] = get_shape<3>(post_mix);
    const auto [comb_mix_tokens, comb_mix_rows, comb_mix_cols] = get_shape<3>(comb_res_mix);
    const auto [fn_num_outputs, fn_hidden] = get_shape<2>(fn);
    const auto [num_mix_scales] = get_shape<1>(mix_scales);
    const auto [num_mix_bases] = get_shape<1>(mix_bases);
    const auto [rmsnorm_hidden] = get_shape<1>(rmsnorm_weight);

    if (num_tokens == 0)
        return;

    const auto num_routes = hc_mult;
    DG_HOST_ASSERT(num_routes == mhc_layout::kNumRoutes);
    const auto num_hc_outputs = mhc_layout::kNumHCOutputs;
    DG_HOST_ASSERT(num_tokens <= static_cast<int>(mhc_layout::kNumMaxTokens));
    DG_HOST_ASSERT(hidden > 0 and hidden % (mhc_layout::kDefaultNumSplits * mhc_layout::BLOCK_K) == 0);
    DG_HOST_ASSERT(num_sinkhorn_iters >= 1);
    DG_HOST_ASSERT(num_tokens == residual_tokens and num_tokens == post_mix_tokens and num_tokens == comb_mix_tokens);
    DG_HOST_ASSERT(num_routes == residual_num_routes and num_routes == post_mix_num_routes and
                   num_routes == comb_mix_rows and num_routes == comb_mix_cols);
    DG_HOST_ASSERT(hidden == residual_hidden and post_mix_width == 1);
    DG_HOST_ASSERT(num_hc_outputs == fn_num_outputs and fn_hidden == num_routes * hidden);
    DG_HOST_ASSERT(num_mix_scales == 3 and num_mix_bases == num_hc_outputs);
    DG_HOST_ASSERT(rmsnorm_hidden == hidden);
    DG_HOST_ASSERT(new_residual.sizes().equals(residual.sizes()));
    DG_HOST_ASSERT(new_post_mix.sizes().equals(post_mix.sizes()));
    DG_HOST_ASSERT(new_comb_res_mix.sizes().equals(comb_res_mix.sizes()));

    const auto [prev_tokens, prev_num_routes, prev_width] = get_shape<3>(shifted_prev_mix);
    DG_HOST_ASSERT(prev_tokens == num_tokens and prev_num_routes == num_routes and prev_width == 1);
    DG_HOST_ASSERT(new_prev_mix.sizes().equals(shifted_prev_mix.sizes()));

    const auto device = x.device();
    const auto check_dense = [&](const torch::stable::Tensor& tensor, const torch::headeronly::ScalarType& dtype) {
        DG_HOST_ASSERT(tensor.scalar_type() == dtype);
        DG_HOST_ASSERT(tensor.is_contiguous());
        DG_HOST_ASSERT(tensor.device() == device);
    };

    check_dense(x, torch::headeronly::ScalarType::BFloat16);
    check_dense(residual, torch::headeronly::ScalarType::BFloat16);
    check_dense(post_mix, torch::headeronly::ScalarType::Float);
    check_dense(comb_res_mix, torch::headeronly::ScalarType::Float);
    check_dense(fn, torch::headeronly::ScalarType::Float);
    check_dense(mix_scales, torch::headeronly::ScalarType::Float);
    check_dense(mix_bases, torch::headeronly::ScalarType::Float);
    check_dense(rmsnorm_weight, torch::headeronly::ScalarType::BFloat16);
    check_dense(new_residual, torch::headeronly::ScalarType::BFloat16);
    check_dense(new_post_mix, torch::headeronly::ScalarType::Float);
    check_dense(new_comb_res_mix, torch::headeronly::ScalarType::Float);
    check_dense(shifted_prev_mix, torch::headeronly::ScalarType::Float);
    check_dense(new_prev_mix, torch::headeronly::ScalarType::Float);

    const auto check_norm_output = [&](const torch::stable::Tensor& tensor, const torch::headeronly::ScalarType& dtype) {
        const auto [output_tokens, output_hidden] = get_shape<2>(tensor);
        DG_HOST_ASSERT(output_tokens == num_tokens and output_hidden == hidden);
        check_dense(tensor, dtype);
    };

    if (y_bf16.has_value())
        check_norm_output(y_bf16.value(), torch::headeronly::ScalarType::BFloat16);
    if (y_fp8.has_value())
        check_norm_output(y_fp8.value(), torch::headeronly::ScalarType::Float8_e4m3fn);

    const auto check_sf = [&](const torch::stable::Tensor& tensor) {
        const auto [sf_tokens, num_sf_words] = get_shape<2>(tensor);
        DG_HOST_ASSERT(sf_tokens == num_tokens and
                       num_sf_words == hidden / static_cast<int>(mhc_layout::kHiddenPerSFWord));
        DG_HOST_ASSERT(tensor.scalar_type() == torch::headeronly::ScalarType::Int);
        DG_HOST_ASSERT(tensor.device() == device);
    };

    if (y_gemm_sf.has_value()) {
        check_sf(y_gemm_sf.value());
        DG_HOST_ASSERT(y_gemm_sf->stride(0) == 1);
        DG_HOST_ASSERT(y_gemm_sf->stride(1) >= num_tokens);
        DG_HOST_ASSERT(y_gemm_sf->stride(1) % get_tma_aligned_size(1, sizeof(uint32_t)) == 0);
    }
    if (y_routed_sf.has_value()) {
        check_sf(y_routed_sf.value());
        DG_HOST_ASSERT(y_routed_sf->is_contiguous());
    }
    if (y_shared_sf.has_value()) {
        check_sf(y_shared_sf.value());
        const auto num_required_rows = ceil_div<int64_t>(num_tokens, shared_sf_block_m) * align<int64_t>(shared_sf_block_m, 128);
        DG_HOST_ASSERT(y_shared_sf->stride(0) == 1);
        DG_HOST_ASSERT(y_shared_sf->stride(1) >= num_required_rows);
        const auto num_required_elements = y_shared_sf->storage_offset() + num_required_rows +
                                           (hidden / mhc_layout::kHiddenPerSFWord - 1) * y_shared_sf->stride(1);
        const auto num_required_bytes = static_cast<uint64_t>(num_required_elements) * y_shared_sf->element_size();
        DG_HOST_ASSERT(num_required_bytes <= torch_compat::storage_nbytes(*y_shared_sf));
    }

    DG_HOST_ASSERT(jit->device.get_arch_major() == 10);
    // The cooperative launch guarantees progress for the split barriers with any grid that fits the device
    // NOTES: upstream requires the full SM count; the fork keeps reduced counts (e.g. vLLM's DBO compute SMs)
    DG_HOST_ASSERT(runtime->get_num_sms() <= jit->device.get_num_sms());
    const auto num_sms = runtime->get_num_sms();
    const auto num_m_blocks = mhc_layout::Workspace<>::get_num_m_blocks(static_cast<uint32_t>(num_tokens));
    const auto num_splits = get_num_splits(num_m_blocks, static_cast<uint32_t>(hidden), static_cast<uint32_t>(num_sms));
    const auto num_mhc_scratch_bytes = mhc_layout::Workspace<>::get_num_scratch_bytes(num_m_blocks, num_splits);

    const bool needs_bf16_scratch = not y_bf16.has_value();
    const auto num_bf16_scratch_bytes = needs_bf16_scratch ?
        static_cast<int64_t>(num_tokens) * hidden * static_cast<int64_t>(x.element_size()) : int64_t{0};
    const auto num_scratch_bytes = static_cast<int64_t>(num_mhc_scratch_bytes) + num_bf16_scratch_bytes;
    const auto scratch = torch::stable::new_empty(
        x, {num_scratch_bytes}, torch::headeronly::ScalarType::Byte);
    const auto& split_barriers = get_split_barriers(x);

    // Shifted FP8 is derived from rounded BF16, so FP8-only calls borrow the scratch tail for that numerical boundary.
    const auto y_bf16_storage = needs_bf16_scratch ? torch::stable::view(
        torch::stable::view(torch_compat::narrow(scratch, 0, num_mhc_scratch_bytes, num_bf16_scratch_bytes),
                                torch::headeronly::ScalarType::BFloat16), {num_tokens, hidden})
        : y_bf16.value();

    // GEMM and routed SF differ only in strides and share one primary writer.
    const auto& y_primary_sf = y_routed_sf.has_value() ? y_routed_sf : y_gemm_sf;

    sm100_mega_mhc(
        x, residual, post_mix, comb_res_mix, shifted_prev_mix,
        fn, mix_scales, mix_bases,
        hc_norm_eps, hc_pre_eps, hc_post_scale, sinkhorn_eps, num_sinkhorn_iters,
        rmsnorm_weight, rmsnorm_eps,
        new_residual, new_prev_mix, new_post_mix, new_comb_res_mix,
        y_bf16_storage, y_bf16.has_value(), scratch, split_barriers,
        y_fp8, y_primary_sf, y_shared_sf,
        shared_sf_block_m, rmsnorm_scale, num_tokens, hidden, num_splits, num_sms);
}

} // namespace deep_gemm::mega_mhc

namespace deep_gemm::torch_registration {
using namespace deep_gemm::torch_utils;

static void mega_mhc(
    const torch::stable::Tensor& x,
    const torch::stable::Tensor& residual,
    const torch::stable::Tensor& shifted_prev_mix,
    const torch::stable::Tensor& post_mix,
    const torch::stable::Tensor& comb_res_mix,
    const torch::stable::Tensor& fn,
    const torch::stable::Tensor& mix_scales,
    const torch::stable::Tensor& mix_bases,
    const int64_t& hc_mult,
    const double& hc_norm_eps,
    const double& hc_pre_eps,
    const double& hc_post_scale,
    const double& sinkhorn_eps,
    const int64_t& num_sinkhorn_iters,
    const torch::stable::Tensor& rmsnorm_weight,
    const double& rmsnorm_eps,
    const double& rmsnorm_scale,
    const torch::stable::Tensor& new_residual,
    const torch::stable::Tensor& new_prev_mix,
    const torch::stable::Tensor& new_post_mix,
    const torch::stable::Tensor& new_comb_res_mix,
    const std::optional<torch::stable::Tensor>& y_bf16,
    const std::optional<torch::stable::Tensor>& y_fp8,
    const std::optional<torch::stable::Tensor>& y_gemm_sf,
    const std::optional<torch::stable::Tensor>& y_routed_sf,
    const std::optional<torch::stable::Tensor>& y_shared_sf,
    const int64_t& shared_sf_block_m) {
    mega_mhc::mega_mhc(
        x, residual, shifted_prev_mix, post_mix, comb_res_mix, fn, mix_scales, mix_bases, hc_mult, hc_norm_eps, hc_pre_eps, hc_post_scale, sinkhorn_eps, num_sinkhorn_iters, rmsnorm_weight, rmsnorm_eps, rmsnorm_scale, new_residual, new_prev_mix, new_post_mix, new_comb_res_mix, y_bf16, y_fp8, y_gemm_sf, y_routed_sf, y_shared_sf, shared_sf_block_m);
}
} // namespace deep_gemm::torch_registration

STABLE_TORCH_LIBRARY_FRAGMENT(deep_gemm, m) {
    m.def("mega_mhc(Tensor x, Tensor residual, Tensor shifted_prev_mix, Tensor post_mix, Tensor comb_res_mix, Tensor fn, Tensor mix_scales, Tensor mix_bases, int hc_mult, float hc_norm_eps, float hc_pre_eps, float hc_post_scale, float sinkhorn_eps, int num_sinkhorn_iters, Tensor rmsnorm_weight, float rmsnorm_eps, float rmsnorm_scale, Tensor(a!) new_residual, Tensor(b!) new_prev_mix, Tensor(c!) new_post_mix, Tensor(d!) new_comb_res_mix, Tensor(e!)? y_bf16=None, Tensor(f!)? y_fp8=None, Tensor(g!)? y_gemm_sf=None, Tensor(h!)? y_routed_sf=None, Tensor(i!)? y_shared_sf=None, int shared_sf_block_m=0) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(deep_gemm, CUDA, m) {
    m.impl("mega_mhc", TORCH_BOX(&deep_gemm::torch_registration::mega_mhc));
}
