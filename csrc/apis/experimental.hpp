#pragma once

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include "../utils/torch_compat.hpp"

#include "../experimental/sm100_bf16_gemm_nt_with_add_norm_n1024_k2816.hpp"
#include "../experimental/sm100_bf16_gemm_nt_with_bias_add_norm_n1024_k1024.hpp"
#include "../experimental/sm100_bf16_gemm_nt_with_swiglu_n5632_k1024.hpp"
#include "../experimental/sm100_bf16_newton_schulz.hpp"
#include "../experimental/sm100_fp8_gemm_nt_segmented_k256.hpp"

#include "../experimental/v41f_qkv_lora.hpp"
#include "../experimental/v41f_wv_o_proj.hpp"

// Experimental kernels without stability guarantees
// NOTES: ops are registered with an `experimental_` prefix, and `deep_gemm/_C.py` exposes them as `deep_gemm.experimental.X`
namespace deep_gemm::torch_registration {

static void experimental_fp8_gemm_nt_segmented_k256(const torch::stable::Tensor& a, const torch::stable::Tensor& sfa,
                                                    const torch::stable::Tensor& b, const torch::stable::Tensor& sfb,
                                                    const torch::stable::Tensor& d) {
    experimental::fp8_gemm_nt_segmented_k256({a, sfa}, {b, sfb}, d);
}

static void experimental_batched_bf16_newton_schulz(const torch::stable::Tensor& x, const torch::stable::Tensor& out,
                                                    const double& a, const double& b, const double& c) {
    experimental::batched_bf16_newton_schulz(x, out, static_cast<float>(a), static_cast<float>(b), static_cast<float>(c));
}

static void experimental_v41f_qkv_lora(const torch::stable::Tensor& x, const torch::stable::Tensor& x_sf,
                                       const torch::stable::Tensor& qkv_weight, const torch::stable::Tensor& qkv_weight_sf,
                                       const torch::stable::Tensor& q_b_weight, const torch::stable::Tensor& q_b_weight_sf,
                                       const torch::stable::Tensor& q_norm_weight, const torch::stable::Tensor& kv_norm_weight,
                                       const torch::stable::Tensor& positions, const torch::stable::Tensor& cos_sin,
                                       const torch::stable::Tensor& slots,
                                       const torch::stable::Tensor& q, const torch::stable::Tensor& cache,
                                       const double& eps) {
    // NOTES: the kernel takes a 32-bit `eps`
    experimental::v41f_qkv_lora({x, x_sf}, {qkv_weight, qkv_weight_sf}, {q_b_weight, q_b_weight_sf},
                                q_norm_weight, kv_norm_weight, positions, cos_sin, slots, q, cache,
                                static_cast<float>(eps));
}

static void experimental_v41f_wv_o_proj(const torch::stable::Tensor& x, const torch::stable::Tensor& x_sf,
                                        const torch::stable::Tensor& wv_weight, const torch::stable::Tensor& wv_weight_sf,
                                        const torch::stable::Tensor& o_weight, const torch::stable::Tensor& o_weight_sf,
                                        const torch::stable::Tensor& y) {
    experimental::v41f_wv_o_proj({x, x_sf}, {wv_weight, wv_weight_sf}, {o_weight, o_weight_sf}, y);
}

} // namespace deep_gemm::torch_registration

STABLE_TORCH_LIBRARY_FRAGMENT(deep_gemm, m) {
    // FP8 GEMM with segmented A [M, S, 256] and prepacked per-32 UE8M0 scales.
    m.def("experimental_fp8_gemm_nt_segmented_k256(Tensor a, Tensor sfa, Tensor b, Tensor sfb, Tensor(d!) d) -> ()");

    // torch: d = (F.silu(a @ b[:2816].T) * (a @ b[2816:].T)).bfloat16()
    m.def("experimental_bf16_gemm_nt_with_swiglu_n5632_k1024(Tensor a, Tensor b, Tensor(d!) d) -> ()");

    // torch: residual += a @ b.T (bf16); d = (residual.float() * rsqrt(residual.float().pow(2).mean(-1, keepdim=True) + 1e-6) * gamma).bfloat16()
    m.def("experimental_bf16_gemm_nt_with_add_norm_n1024_k2816(Tensor a, Tensor b, Tensor gamma, Tensor(residual!) residual, Tensor(d!) d) -> ()");

    // torch: residual += a @ b.T + bias (bf16); d = (residual.float() * rsqrt(residual.float().pow(2).mean(-1, keepdim=True) + 1e-6) * gamma).bfloat16()
    m.def("experimental_bf16_gemm_nt_with_bias_add_norm_n1024_k1024(Tensor a, Tensor b, Tensor bias, Tensor gamma, Tensor(residual!) residual, Tensor(d!) d) -> ()");

    m.def("experimental_batched_bf16_newton_schulz(Tensor x, Tensor(out!) out, float a, float b, float c) -> ()");

    m.def("experimental_v41f_qkv_lora(Tensor x, Tensor x_sf, Tensor qkv_weight, Tensor qkv_weight_sf, Tensor q_b_weight, Tensor q_b_weight_sf, Tensor q_norm_weight, Tensor kv_norm_weight, Tensor positions, Tensor cos_sin, Tensor slots, Tensor(q!) q, Tensor(cache!) cache, float eps) -> ()");
    m.def("experimental_v41f_wv_o_proj(Tensor x, Tensor x_sf, Tensor wv_weight, Tensor wv_weight_sf, Tensor o_weight, Tensor o_weight_sf, Tensor(y!) y) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(deep_gemm, CUDA, m) {
    m.impl("experimental_fp8_gemm_nt_segmented_k256",
           TORCH_BOX(&deep_gemm::torch_registration::experimental_fp8_gemm_nt_segmented_k256));
    m.impl("experimental_bf16_gemm_nt_with_swiglu_n5632_k1024",
           TORCH_BOX(&deep_gemm::experimental::bf16_gemm_nt_with_swiglu_n5632_k1024));
    m.impl("experimental_bf16_gemm_nt_with_add_norm_n1024_k2816",
           TORCH_BOX(&deep_gemm::experimental::bf16_gemm_nt_with_add_norm_n1024_k2816));
    m.impl("experimental_bf16_gemm_nt_with_bias_add_norm_n1024_k1024",
           TORCH_BOX(&deep_gemm::experimental::bf16_gemm_nt_with_bias_add_norm_n1024_k1024));
    m.impl("experimental_batched_bf16_newton_schulz",
           TORCH_BOX(&deep_gemm::torch_registration::experimental_batched_bf16_newton_schulz));
    m.impl("experimental_v41f_qkv_lora",
           TORCH_BOX(&deep_gemm::torch_registration::experimental_v41f_qkv_lora));
    m.impl("experimental_v41f_wv_o_proj",
           TORCH_BOX(&deep_gemm::torch_registration::experimental_v41f_wv_o_proj));
}
