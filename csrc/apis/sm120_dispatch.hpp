#pragma once

// All SM120-specific host-side flow lives here, NOT in csrc/apis/*.hpp.
//
// Rationale (see AI/2026-09-12-sm120-port-design.md sections 5.1 and 5.3): upstream never
// opens a file named sm120_*, so logic placed here is immune to rebase conflicts. The api
// headers keep only 1-3 line dispatch arms that call into this namespace, which is the
// cheapest merge there is: if upstream restructures an if/else chain, you re-add one line
// instead of re-porting ~111 lines of real logic.
//
// SM120 facts encoded here, stated once, so nobody has to re-derive them from five
// upstream diffs:
//   * SM120 MMA consumes K-major operands only -> operands are coerced up front.
//   * The AB-swap decision must PRECEDE the single SF transform, so the fp8/fp4 NT flow
//     owns its whole pipeline rather than reusing the shared one.
//   * Paged logits use 2 groups x 64 KV rows = 128, where SM90/SM100 use 256.
//
// UNVALIDATED: no sm120 hardware was available while porting; none of this has been run.

#include <torch/python.h>

#include "../jit_kernels/impls/sm120_fp8_fp4_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm120_bf16_gemm.hpp"
#include "../jit_kernels/impls/sm120_mqa_logits.hpp"
#include "../jit_kernels/impls/sm120_bmk_bnk_mn.hpp"
#include "../jit_kernels/impls/sm120_tf32_hc_prenorm_gemm.hpp"
#include "layout.hpp"

namespace deep_gemm::sm120 {

// SM120a: 2 groups x 64 KV rows = 128; SM90/SM100 use 256.
constexpr int kPagedSplitKv = 128;
constexpr int kMqaBlockKv   = 128;

// Repack a packed-FP4 tensor (two 4-bit codes per int8) from MN-major into K-major.
// File-local on purpose: this is its only consumer, so upstream-owned `csrc/utils/math.hpp`
// (where nv_dev put it) stays untouched.
static torch::Tensor fp4_repack_to_k_major(const torch::Tensor& a, int logical_mn) {
    DG_HOST_ASSERT(a.scalar_type() == kPackedFP4);
    const int ndim = a.dim();
    DG_HOST_ASSERT(ndim == 2 or ndim == 3);
    const int mn_packed = a.size(-2);
    const int k = a.size(-1);
    DG_HOST_ASSERT(mn_packed * 2 == logical_mn and k % 2 == 0);

    auto lo = a.bitwise_and(0x0F);
    auto hi = a.to(torch::kByte).bitwise_right_shift(4).to(torch::kInt8).bitwise_and(0x0F);

    auto shape_full = a.sizes().vec();
    shape_full[ndim - 2] = logical_mn;
    auto codes = torch::empty(shape_full, a.options());
    using S = torch::indexing::Slice;
    codes.index_put_({torch::indexing::Ellipsis, S(0, torch::indexing::None, 2), S()}, lo);
    codes.index_put_({torch::indexing::Ellipsis, S(1, torch::indexing::None, 2), S()}, hi);

    auto shape_view = shape_full;
    shape_view[ndim - 1] = k / 2;
    shape_view.push_back(2);
    auto codes2 = codes.view(shape_view);
    auto result = codes2.select(-1, 0).bitwise_and(0x0F)
                  .bitwise_or(codes2.select(-1, 1).bitwise_and(0x0F).to(torch::kByte)
                              .bitwise_left_shift(4).to(torch::kInt8));
    return result.contiguous();
}

// SM120 MMA consumes K-major operands: repack packed-FP4 (needs `logical_mn`) or copy
static torch::Tensor to_k_major(const torch::Tensor& t, const cute::UMMA::Major& major,
                                const int& logical_mn) {
    if (major == cute::UMMA::Major::K)
        return t;
    return t.scalar_type() == kPackedFP4 ? fp4_repack_to_k_major(t, logical_mn) : t.contiguous();
}

// SM120: AB-swap decision must precede the single SF transform, so it owns its own flow
static void fp8_fp4_gemm_nt(const std::pair<torch::Tensor, torch::Tensor>& a,
                            const std::pair<torch::Tensor, torch::Tensor>& b,
                            const torch::Tensor& d,
                            const std::optional<torch::Tensor>& c,
                            const std::optional<std::tuple<int, int, int>>& recipe,
                            const std::optional<std::tuple<int, int>>& recipe_a,
                            const std::optional<std::tuple<int, int>>& recipe_b,
                            const std::string& compiled_dims,
                            const bool& disable_ue8m0_cast,
                            const cute::UMMA::Major& major_a,
                            const cute::UMMA::Major& major_b,
                            const int& m, const int& n, const int& k) {
    // Force K-major operands
    const auto a_data = to_k_major(a.first, major_a, m);
    const auto b_data = to_k_major(b.first, major_b, n);
    constexpr auto k_major = cute::UMMA::Major::K;

    const bool is_mixed_fp4 = (a_data.scalar_type() != b_data.scalar_type()) and
                              (a_data.scalar_type() == kPackedFP4 or b_data.scalar_type() == kPackedFP4);
    DG_HOST_ASSERT(!is_mixed_fp4 or k % 128 == 0);

    // AB-swap for small-M decode: swap A↔B so small M becomes N (BN=16).
    // K-major B has N as TMA outer dim — no minimum size restriction.
    constexpr int kSwapAbMMax = 16;
    const bool swap_ab = (m >= 1 and m <= kSwapAbMMax
        and d.stride(-1) == 1 and !is_mixed_fp4 and !c.has_value());

    DG_HOST_ASSERT(recipe_a.has_value() == recipe_b.has_value());
    DG_HOST_ASSERT(not recipe.has_value() or not recipe_a.has_value());

    std::optional<std::tuple<int, int, int>> eff_recipe = std::nullopt;
    std::optional<std::tuple<int, int>> eff_recipe_a, eff_recipe_b;
    if (recipe_a.has_value()) {
        eff_recipe_a = swap_ab ? recipe_b : recipe_a;
        eff_recipe_b = swap_ab ? recipe_a : recipe_b;
    } else if (swap_ab) {
        const auto [ga, gb, gk] = recipe.value_or(
            get_default_recipe(a.second.scalar_type(), b.second.scalar_type()));
        eff_recipe_a = std::make_tuple(gb, gk);
        eff_recipe_b = std::make_tuple(ga, gk);
    } else {
        eff_recipe = recipe;
    }

    const auto& sf_a_raw = swap_ab ? b.second : a.second;
    const auto& sf_b_raw = swap_ab ? a.second : b.second;
    const int eff_m = swap_ab ? n : m;
    const int eff_n = swap_ab ? m : n;

    const auto [sfa, sfb, gran_k_a, gran_k_b] = layout::transform_sf_pair_into_required_layout(
        sf_a_raw, sf_b_raw, eff_m, eff_n, k, eff_recipe,
        eff_recipe_a, eff_recipe_b, std::nullopt, std::nullopt, disable_ue8m0_cast);
    DG_HOST_ASSERT(sfa.scalar_type() == torch::kInt and sfb.scalar_type() == torch::kInt);

    if (swap_ab) {
        sm120_fp8_fp4_gemm_1d1d(b_data, sfa, a_data, sfb, std::nullopt, d,
                                eff_m, eff_n, k, gran_k_a, gran_k_b,
                                k_major, k_major, compiled_dims,
                                std::nullopt, true);
    } else {
        sm120_fp8_fp4_gemm_1d1d(a_data, sfa, b_data, sfb, c, d, m, n, k, gran_k_a, gran_k_b,
                                k_major, k_major, compiled_dims);
    }
}

// Batched (`bmk,bnk->bmn`) AB-swap bound. Declared *after* `fp8_fp4_gemm_nt` on purpose:
// that function carries its own, tighter, function-local `kSwapAbMMax = 16`, and declaring
// this one later keeps it out of scope there rather than shadowing it.
constexpr int kSwapAbMMax = 32;

// AB-swap for small-M decode: BLOCK_M >= 64 wastes lanes at M <= 32, so swap A<->B to
// put the small dim on N (BLOCK_N 16/32). Done before the SF transform; the kernel
// writes back to the caller's buffer via runtime stride_cd_m/n (see sm120_fp8_fp4_bmm).
// Excluded when accumulating (c): swapped strides break the batched epilogue.
static bool bmm_swap_ab_eligible(const int& m,
                                 const cute::UMMA::Major& major_a,
                                 const cute::UMMA::Major& major_b,
                                 const torch::Tensor& d,
                                 const bool& with_accumulation) {
    return jit->device.get_arch_major() == 12 and m >= 1 and m <= kSwapAbMMax
        and major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K
        and d.stride(-1) == 1
        and not with_accumulation;
}

// The swapped batched flow. Takes the operands in the caller's *unswapped* order and performs
// the swap internally, because the swap has to happen before the single SF transform.
static void fp8_fp4_bmm_swapped(const torch::Tensor& a, const torch::Tensor& sfa,
                                const torch::Tensor& b, const torch::Tensor& sfb,
                                const std::optional<torch::Tensor>& c,
                                const torch::Tensor& d,
                                const int& batch_size,
                                const int& m, const int& n, const int& k,
                                const cute::UMMA::Major& major_a,
                                const cute::UMMA::Major& major_b,
                                const std::string& compiled_dims,
                                const std::optional<std::tuple<int, int, int>>& recipe) {
    // Swap per-tensor granularities to match the swapped operands; else asymmetric
    // recipes like (1,128,128) trip the SF layout shape check.
    const auto eff_recipe = recipe.has_value()
        ? recipe.value()
        : get_default_recipe(sfa.scalar_type(), sfb.scalar_type());
    const auto& [ga, gb, gk] = eff_recipe;
    std::optional<std::tuple<int, int, int>> swap_recipe = std::nullopt;
    std::optional<std::tuple<int, int>> swap_recipe_a = std::make_tuple(gb, gk);
    std::optional<std::tuple<int, int>> swap_recipe_b = std::make_tuple(ga, gk);
    const auto [transformed_sfa_swap, transformed_sfb_swap, gran_k_a_swap, gran_k_b_swap]
        = layout::transform_sf_pair_into_required_layout(
            sfb, sfa, /*m=*/n, /*n=*/m, k, swap_recipe,
            swap_recipe_a, swap_recipe_b, batch_size, batch_size, false);
    sm120_fp8_fp4_bmm(
        b, transformed_sfa_swap, a, transformed_sfb_swap, c, d,
        batch_size, /*m=*/n, /*n=*/m, k,
        gran_k_a_swap, gran_k_b_swap,
        major_b, major_a, compiled_dims,
        /*swap_ab=*/true);
}

} // namespace deep_gemm::sm120
