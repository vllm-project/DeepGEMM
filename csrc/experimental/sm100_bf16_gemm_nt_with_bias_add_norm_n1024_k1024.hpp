#pragma once

#include "sm100_bf16_gemm_nt_with_add_norm_n1024_k2816.hpp"

namespace deep_gemm::experimental {

// torch: residual += a @ b.T + bias (bf16); d = (residual.float() * rsqrt(residual.float().pow(2).mean(-1, keepdim=True) + 1e-6) * gamma).bfloat16()
static void bf16_gemm_nt_with_bias_add_norm_n1024_k1024(
    const torch::stable::Tensor& a, const torch::stable::Tensor& b, const torch::stable::Tensor& bias,
    const torch::stable::Tensor& gamma, const torch::stable::Tensor& residual, const torch::stable::Tensor& d) {
    launch_bf16_gemm_nt_with_add_norm<1024, true>(a, b, bias, gamma, residual, d);
}

} // namespace deep_gemm::experimental
