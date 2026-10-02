#pragma once

#include <algorithm>
#include <cstdint>
#include <format>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include "../../utils/torch_compat.hpp"

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "../../utils/layout.hpp"
#include "../heuristics/runtime.hpp"

namespace deep_gemm {

// DeepGEMM-specific int64 sequence [0, end), compiled through the existing JIT.
static torch::stable::Tensor deep_gemm_arange_int64(int64_t end, torch::stable::Device device) {
    STD_TORCH_CHECK(end >= 0, "deep_gemm_arange_int64 requires a non-negative end");
    STD_TORCH_CHECK(device.is_cuda(), "deep_gemm_arange_int64 requires a CUDA device");
    auto output = torch::stable::empty(
        {end}, torch::headeronly::ScalarType::Long, torch::headeronly::Layout::Strided, device);
    if (end == 0)
        return output;

    constexpr int threads_per_block = 256;
    constexpr int max_blocks = 4096;
    const auto required_blocks = 1 + (end - 1) / threads_per_block;
    const auto blocks = static_cast<unsigned int>(std::min<int64_t>(required_blocks, max_blocks));
    const auto kernel = jit->compile("deep_gemm_arange_int64", R"(
#include <cstdint>

extern "C" __global__ void deep_gemm_arange_int64_kernel(std::int64_t* output, std::int64_t end) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = index; i < end; i += stride)
        output[i] = i;
}
)");
    jit->launch(
        kernel, {
            .stream = reinterpret_cast<CUstream>(torch_compat::current_stream(output)),
            .grid_dim = dim3(blocks, 1, 1),
            .block_dim = dim3(threads_per_block, 1, 1),
        },
        output.mutable_data_ptr<int64_t>(), end);
    return output;
}

class PackFP32IntoUE8M0Runtime final {
public:
    struct Args {
        int num_groups, mn, sf_k, packed_sf_k, gran_k, k_alignment;
        bool use_psum_layout;
        int block_mn, block_packed_sf_k;
        void *sf, *out, *grouped_layout;

        deep_jit::cuda::LaunchOptions options;
    };

    static void compile_and_launch(const std::string& tag, const Args& args) {
        const auto kernel = jit->compile(tag, std::format(R"(
#include <deep_gemm/impls/smxx_layout.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&pack_fp32_into_ue8m0<
        {}, {}, {}, {}, {}, {}
    >);
}};
)", args.num_groups, args.options.block_dim->x, args.block_mn, args.block_packed_sf_k,
    "true", args.use_psum_layout ? "true" : "false"));

        // Launch
        jit->launch(
            kernel, args.options,
            args.sf, args.out, args.grouped_layout, args.mn, args.sf_k, args.packed_sf_k,
            args.gran_k, args.k_alignment
        );
    }
};

static std::tuple<int, int, int, int, int, torch::stable::Tensor> preprocess_sf(const torch::stable::Tensor& sf) {
    // NOTES: for the extreme performance, you may rewrite/fuse this function in CUDA
    const auto dim = sf.dim();
    DG_HOST_ASSERT(dim == 2 or dim == 3);
    DG_HOST_ASSERT(sf.scalar_type() == torch::headeronly::ScalarType::Float);
    const auto batched_sf = dim == 2 ? torch::stable::unsqueeze(sf, 0) : sf;

    const auto [num_sf_batches, mn, sf_k] = get_shape<3>(batched_sf);
    const auto tma_aligned_mn = get_tma_aligned_size(mn, static_cast<int>(sf.element_size()));
    return {dim, num_sf_batches, mn, sf_k, tma_aligned_mn, batched_sf};
}

static torch::stable::Tensor get_mn_major_tma_aligned_tensor(const torch::stable::Tensor& sf) {
    const auto [dim, num_sf_batches, mn, sf_k, tma_aligned_mn, batched_sf] = preprocess_sf(sf);

    // The last kernel already gives a column-major TMA aligned layout
    if ((batched_sf.stride(0) == tma_aligned_mn * sf_k or dim == 2) and batched_sf.stride(1) == 1 and batched_sf.stride(2) == tma_aligned_mn)
        return (dim == 2) ? torch::stable::squeeze(batched_sf, 0) : batched_sf;

    const auto out = torch_compat::empty_strided(
        {num_sf_batches, mn, sf_k},
        {tma_aligned_mn * sf_k, 1, tma_aligned_mn},
        batched_sf.scalar_type(),
        batched_sf.device());

    if (not batched_sf.is_contiguous()) {
        // Fallback to PyTorch's slow copy if not contiguous
        // ReSharper disable once CppExpressionWithoutSideEffects
        torch_compat::copy_(out, batched_sf);
    } else {
        constexpr int block_mn = 64;
        constexpr int block_sf_k = 128;
        constexpr int num_threads = 512;
        const auto smem_sf_k = sf_k < block_sf_k ? sf_k : block_sf_k;
        const auto smem_size = block_mn * (smem_sf_k + (1 - smem_sf_k % 2)) * static_cast<int>(sizeof(float));

        // Compile
        const auto kernel = jit->compile("transpose_fp32", std::format(R"(
#include <deep_gemm/impls/smxx_layout.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&transpose_fp32<
        {}, {}, {}, {}
    >);
}};
)", num_threads, block_mn, sf_k, block_sf_k));

        // Launch
        jit->launch(
            kernel, {
                .num_smem_bytes = smem_size,
                .grid_dim = dim3(ceil_div(mn, block_mn) * ceil_div(sf_k, block_sf_k), num_sf_batches, 1),
                .block_dim = dim3(num_threads, 1, 1),
            },
            batched_sf.mutable_data_ptr(), out.mutable_data_ptr(), static_cast<uint32_t>(mn)
        );
    }
    return (dim == 2) ? torch::stable::squeeze(out, 0) : out;
}

static torch::stable::Tensor get_mn_major_tma_aligned_packed_ue8m0_tensor_torch(const torch::stable::Tensor& sf) {
    const auto sf_reshaped = (sf.dim() == 2) ? torch::stable::unsqueeze(sf, 0) : sf;

    // First, convert into UE8M0 `uint8_t`
    const auto ue8m0_tensor = torch::stable::to(torch_compat::bitwise_right_shift(torch_compat::view_dtype(sf_reshaped, torch::headeronly::ScalarType::Int), 23), torch::headeronly::ScalarType::Byte);

    // Second, make padded packed tensors
    const auto [num_sf_batches, mn, k] = get_shape<3>(sf_reshaped);
    const auto aligned_mn = get_tma_aligned_size(mn, 4);
    const auto aligned_k  = align(k, 4);

    auto padded = torch::stable::new_zeros(sf, {num_sf_batches, aligned_mn, aligned_k}, torch::headeronly::ScalarType::Byte);
    // ReSharper disable once CppExpressionWithoutSideEffects
    torch_compat::copy_(torch_compat::slice(torch_compat::slice(padded, 1, 0, mn), 2, 0, k), ue8m0_tensor);
    padded = torch::stable::view(torch_compat::view_dtype(torch::stable::view(padded, {-1}), torch::headeronly::ScalarType::Int), {num_sf_batches, aligned_mn, aligned_k / 4});

    // Finally, transpose
    auto out = torch_compat::empty_strided(
        {num_sf_batches, aligned_mn, aligned_k / 4},
        {aligned_mn * (aligned_k / 4), 1, aligned_mn},
        torch::headeronly::ScalarType::Int,
        sf.device());
    out = torch_compat::slice(torch_compat::copy_(out, padded), 1, 0, mn);
    return (sf.dim() == 2) ? torch::stable::squeeze(out, 0) : out;
}

static torch::stable::Tensor get_mn_major_tma_aligned_packed_ue8m0_tensor(const torch::stable::Tensor& sf,
                                                                  const std::optional<torch::stable::Tensor>& psum_layout = std::nullopt) {
    const auto [dim, num_sf_batches, mn, sf_k, tma_aligned_mn, batched_sf] = preprocess_sf(sf);
    const auto packed_sf_k = ceil_div(sf_k, 4);
    const auto out = torch_compat::empty_strided(
        {num_sf_batches, mn, packed_sf_k},
        {packed_sf_k * tma_aligned_mn, 1, tma_aligned_mn},
        torch::headeronly::ScalarType::Int,
        batched_sf.device());

    // PSUM layout (always 2D contiguous) lets the pack kernel skip uninitialized MN gap rows
    const auto use_psum_layout = psum_layout.has_value();
    if (use_psum_layout) {
        DG_HOST_ASSERT(num_sf_batches == 1 and batched_sf.is_contiguous());
        DG_HOST_ASSERT(psum_layout->scalar_type() == torch::headeronly::ScalarType::Int and psum_layout->is_contiguous());
        DG_HOST_ASSERT(psum_layout->numel() > 0);
    }
    const auto m_alignment = use_psum_layout ? heuristics_runtime->get_mk_alignment_for_contiguous_layout() : 0;
    const auto num_psum_groups = use_psum_layout ? static_cast<int>(psum_layout->numel()) : 1;

    if (batched_sf.is_contiguous()) {
        if ((mn * sf_k) % 4 != 0 and num_sf_batches > 1)
            return get_mn_major_tma_aligned_packed_ue8m0_tensor_torch(sf);

        constexpr int block_mn = 48;
        constexpr int block_sf_k = 128;
        constexpr int num_threads = 512;
        const auto psum_smem_elems = use_psum_layout ? align(num_psum_groups * 2, 4) : 0;
        const auto smem_sf_k = sf_k < block_sf_k ? sf_k : block_sf_k + 1;
        const auto smem_size = block_mn * smem_sf_k * 4 + psum_smem_elems * 4;

        // Compile
        const auto kernel = jit->compile("transpose_and_pack_fp32_into_ue8m0", std::format(R"(
#include <deep_gemm/impls/smxx_layout.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&transpose_and_pack_fp32_into_ue8m0<
        {}, {}, {}, {}, {}, {}
    >);
}};
)", num_threads, block_mn, sf_k, block_sf_k, num_psum_groups, use_psum_layout ? "true" : "false"));

        // Launch
        jit->launch(
            kernel, {
                .num_smem_bytes = smem_size,
                .grid_dim = dim3(ceil_div(mn, block_mn) * ceil_div(sf_k, block_sf_k), num_sf_batches, 1),
                .block_dim = dim3(num_threads, 1, 1),
            },
            batched_sf.mutable_data_ptr(), out.mutable_data_ptr(), static_cast<uint32_t>(mn),
            use_psum_layout ? psum_layout->mutable_data_ptr() : nullptr, static_cast<uint32_t>(m_alignment)
        );
    } else {
        DG_HOST_ASSERT(not use_psum_layout);
        if (mn % 4 != 0 or num_sf_batches > 1)
            return get_mn_major_tma_aligned_packed_ue8m0_tensor_torch(sf);
        DG_HOST_ASSERT(batched_sf.stride(1) == 1 and batched_sf.stride(2) == mn);

        constexpr int block_mn = 128;
        constexpr int block_packed_sf_k = 16;
        constexpr int num_threads = 512;

        // Compile and launch
        PackFP32IntoUE8M0Runtime::compile_and_launch("pack_fp32_into_ue8m0", {
            .num_groups = 1,
            .mn = mn,
            .sf_k = sf_k,
            .packed_sf_k = packed_sf_k,
            .gran_k = 128,
            .k_alignment = 128,
            .use_psum_layout = false,
            .block_mn = block_mn,
            .block_packed_sf_k = block_packed_sf_k,
            .sf = batched_sf.mutable_data_ptr(),
            .out = out.mutable_data_ptr(),
            .grouped_layout = nullptr,
            .options = {
                .grid_dim = dim3(ceil_div(mn, block_mn), ceil_div(packed_sf_k, block_packed_sf_k), 1),
                .block_dim = dim3(num_threads, 1, 1),
            }
        });
    }
    return (dim == 2) ? torch::stable::squeeze(out, 0) : out;
}

static torch::stable::Tensor get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(const torch::stable::Tensor& sf,
                                                                            const torch::stable::Tensor& grouped_layout,
                                                                            const std::optional<std::vector<int>>& ks_cpu,
                                                                            const int gran_k,
                                                                            const int k_alignment,
                                                                            const bool& use_psum_layout) {
    // SM120 reaches this packer through `csrc/apis/layout.hpp`'s FP32 k-grouped path.
    const auto arch_major = jit->device.get_arch_major();
    DG_HOST_ASSERT((arch_major == 10 or arch_major == 12) and (gran_k == 32 or gran_k == 128) and k_alignment % 128 == 0);
    const auto [sf_k, mn] = get_shape<2>(sf);
    const auto num_groups = static_cast<int>(grouped_layout.numel());

    DG_HOST_ASSERT(sf.is_contiguous());
    DG_HOST_ASSERT(num_groups <= 128 and mn % 4 == 0);
    DG_HOST_ASSERT(grouped_layout.is_contiguous() and grouped_layout.scalar_type() == torch::headeronly::ScalarType::Int);

    const auto has_synced_ks = ks_cpu.has_value() and not ks_cpu.value().empty();
    if (has_synced_ks) {
        DG_HOST_ASSERT(static_cast<int>(ks_cpu.value().size()) == num_groups);
    } else {
        DG_HOST_ASSERT(use_psum_layout);
    }

    int packed_sf_k = 0;
    if (has_synced_ks) {
        int ref_sf_k = 0;
        for (const auto k: ks_cpu.value()) {
            DG_HOST_ASSERT(k % k_alignment == 0 and k % gran_k == 0);
            const auto group_sf_k = k / gran_k;
            ref_sf_k += group_sf_k;
            packed_sf_k += ceil_div(group_sf_k, 4);
        }
        DG_HOST_ASSERT(ref_sf_k == sf_k);
    } else {
        // Exact group sizes are read from the psum layout by the pack kernel.
        // This upper bound allows three tail-padding slots per group.
        packed_sf_k = (sf_k + num_groups * 3) / 4;
    }
    if (packed_sf_k == 0)
        return torch::stable::new_empty(sf, {0, mn}, torch::headeronly::ScalarType::Int);

    const auto out = torch::stable::new_empty(sf, {packed_sf_k, mn}, torch::headeronly::ScalarType::Int);

    constexpr int block_mn = 128;
    constexpr int block_packed_sf_k = 16;
    constexpr int num_threads = 512;

    // Compile and launch
    PackFP32IntoUE8M0Runtime::compile_and_launch("pack_fp32_into_ue8m0", {
        .num_groups = num_groups,
        .mn = mn,
        .sf_k = sf_k,
        .packed_sf_k = packed_sf_k,
        .gran_k = gran_k,
        .k_alignment = k_alignment,
        .use_psum_layout = use_psum_layout,
        .block_mn = block_mn,
        .block_packed_sf_k = block_packed_sf_k,
        .sf = sf.mutable_data_ptr(),
        .out = out.mutable_data_ptr(),
        .grouped_layout = grouped_layout.mutable_data_ptr(),
        .options = {
            .grid_dim = dim3(ceil_div(mn, block_mn), ceil_div(packed_sf_k, block_packed_sf_k), 1),
            .block_dim = dim3(num_threads, 1, 1),
        }
    });
    return out;
}

// Validate a user-provided, already packed UE8M0 (`int32`) SF tensor.
static torch::stable::Tensor check_k_grouped_packed_ue8m0_tensor(const torch::stable::Tensor& sf,
                                                         const torch::stable::Tensor& grouped_layout,
                                                         const std::optional<std::vector<int>>& ks_cpu,
                                                         const int gran_k,
                                                         const int k_alignment,
                                                         const bool& use_psum_layout) {
    DG_HOST_ASSERT(sf.scalar_type() == torch::headeronly::ScalarType::Int);
    DG_HOST_ASSERT(gran_k == 32);
    DG_HOST_ASSERT(sf.dim() == 2);
    DG_HOST_ASSERT(sf.is_contiguous());

    const auto [packed_sf_k, mn] = get_shape<2>(sf);
    const auto num_groups = static_cast<int>(grouped_layout.numel());
    DG_HOST_ASSERT(mn % 4 == 0);
    DG_HOST_ASSERT(sf.stride(0) == mn and sf.stride(1) == 1);
    DG_HOST_ASSERT(grouped_layout.is_contiguous() and grouped_layout.scalar_type() == torch::headeronly::ScalarType::Int);
    DG_HOST_ASSERT(packed_sf_k > 0);

    const auto has_synced_ks = ks_cpu.has_value() and not ks_cpu.value().empty();
    if (has_synced_ks) {
        DG_HOST_ASSERT(static_cast<int>(ks_cpu.value().size()) == num_groups);
        int required_packed_sf_k = 0;
        for (const auto k: ks_cpu.value()) {
            DG_HOST_ASSERT(k % k_alignment == 0);
            required_packed_sf_k += ceil_div(k, gran_k * 4);
        }
        DG_HOST_ASSERT(packed_sf_k >= required_packed_sf_k);
    } else {
        DG_HOST_ASSERT(use_psum_layout);
    }
    return sf;
}

} // namespace deep_gemm
