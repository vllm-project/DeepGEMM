#pragma once

#include <cutlass/bfloat16.h>
#include <deep_gemm/epilogue/transform.cuh>

namespace deep_gemm {

template <typename cd_dtype_t, uint32_t kSplitKFactor, bool kWithAccumulation,
          typename epilogue_type_t = epilogue::transform::EpilogueIdentity>
__global__ void sm120_split_k_reduce_impl(
    cd_dtype_t* __restrict__ gmem_d,
    const float* __restrict__ workspace,
    uint32_t shape_m, uint32_t shape_n,
    int stride_cd_m, int stride_cd_n) {
    cudaGridDependencySynchronize();

    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = shape_m * shape_n;
    if (idx >= total)
        return;

    const uint32_t row = idx / shape_n;
    const uint32_t col = idx % shape_n;
    const uint32_t ws_stride = shape_m * shape_n;

    float sum = workspace[idx];
    #pragma unroll
    for (uint32_t s = 1; s < kSplitKFactor; ++s)
        sum += workspace[s * ws_stride + idx];

    const auto physical_col = epilogue_type_t::template apply_index_n<1>(col);
    const auto d_idx = static_cast<int64_t>(row) * stride_cd_m + static_cast<int64_t>(physical_col) * stride_cd_n;
    if constexpr (kWithAccumulation)
        sum += static_cast<float>(gmem_d[d_idx]);
    gmem_d[d_idx] = cd_dtype_t(sum);
}

} // namespace deep_gemm
