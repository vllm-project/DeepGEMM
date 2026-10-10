#pragma once

#if 0 // Legacy ATen implementation retained below for merge context; stable implementation follows.

#include <c10/util/accumulate.h>
#include <c10/util/strides.h>
#include <torch/all.h>
#include <torch/library.h>

#include "../jit_kernels/impls/sm100_locality_domain.hpp"
#include "../runtime/runtime.hpp"
#include "../utils/exception.hpp"
#include "../utils/layout.hpp"
#include "../utils/math.hpp"

namespace deep_gemm::locality_domain {

/// Localized allocations

static int get_num_locality_domains() {
    return LocalityDomainAllocator::get_num_locality_domains();
}

static size_t get_granularity() {
    return LocalityDomainAllocator::get_granularity();
}

static bool is_localization_available() {
    return mlopart::is_available();
}

static void release(void* ptr) {
    locality_domain_allocator.release(ptr);
}

// `torch.empty(sizes, dtype=dtype)` in the locality domain `domain_idx`
static torch::Tensor empty(const at::IntArrayRef& sizes, const torch::ScalarType& dtype, const int& domain_idx) {
    auto& allocator = locality_domain_allocator;
    const auto num_bytes = align<size_t>(c10::multiply_integers(sizes) * c10::elementSize(dtype), allocator.get_granularity());
    const auto ptr = allocator.reserve(num_bytes);
    allocator.map(ptr, num_bytes, domain_idx);
    return torch::from_blob(ptr, sizes, release, torch::dtype(dtype).device(torch::kCUDA));
}

// `torch.empty((num_domains, *sizes), dtype=dtype)`, with slice `d` in domain `d`
static torch::Tensor empty_per_domain(const at::IntArrayRef& sizes, const torch::ScalarType& dtype) {
    auto& allocator = locality_domain_allocator;
    const int64_t elem_size = static_cast<int64_t>(c10::elementSize(dtype)), num_domains = get_num_locality_domains();
    const int64_t num_slice_bytes = align<int64_t>(c10::multiply_integers(sizes) * elem_size, static_cast<int64_t>(allocator.get_granularity()));
    const auto ptr = static_cast<char*>(allocator.reserve(num_slice_bytes * num_domains));
    for (int d = 0; d < num_domains; ++ d)
        allocator.map(ptr + d * num_slice_bytes, num_slice_bytes, d);

    std::vector<int64_t> full_sizes = {num_domains}, strides = {num_slice_bytes / elem_size};
    full_sizes.insert(full_sizes.end(), sizes.begin(), sizes.end());
    const auto slice_strides = c10::contiguous_strides(sizes);
    strides.insert(strides.end(), slice_strides.begin(), slice_strides.end());
    return torch::from_blob(ptr, full_sizes, strides, release, torch::dtype(dtype).device(torch::kCUDA));
}

/// Locality domain of the SMs

static constexpr int kNumSMsPerTPC = 2;
static constexpr int kNumLinesPerChunk = kNumLocalityDomainProbeHops;
static constexpr int kNumWordsPerLine = kNumLocalityDomainProbeLineBytes / sizeof(int);
static constexpr int kNumProbeChunksPerSM = 8;
// Coprime with the line count, so that the chain visits every line
static constexpr int kChainLineStride = 7;
static constexpr float kMinFarNearRatio = 1.25f;
static constexpr int kNumMaxProbeAttempts = 5;

static void flush_l2() {
    torch::zeros({256 << 20}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
}

// One chunk of word indices: word 0 of line `l` points to line `(l + kChainLineStride) % kNumLinesPerChunk`
static torch::Tensor get_probe_chain() {
    const auto options = torch::dtype(torch::kInt).device(torch::kCUDA);
    const auto chain = torch::zeros({kNumLinesPerChunk, kNumWordsPerLine}, options);
    chain.select(1, 0).copy_((torch::arange(kNumLinesPerChunk, options) + kChainLineStride) % kNumLinesPerChunk * kNumWordsPerLine);
    return chain.view(-1);
}

// `(num_domains, num_sms, kNumProbeChunksPerSM, kNumLocalityDomainProbeChunkBytes)` -> `(num_domains, num_sms)`
static torch::Tensor probe(const torch::Tensor& buf) {
    buf.view(torch::kInt).view({buf.size(0), -1, kNumLocalityDomainProbeChunkBytes / 4}).copy_(get_probe_chain());
    const auto latency = torch::zeros({buf.size(0), buf.size(1), buf.size(2)}, buf.options().dtype(torch::kInt16));
    flush_l2();
    for (int d = 0; d < buf.size(0); ++ d)
        sm100_locality_domain_probe_chase(buf[d], latency[d]);
    DG_HOST_ASSERT(latency.gt(0).all().item<bool>());
    return std::get<0>(latency.to(torch::kFloat).median(-1));
}

// Make an attempt
static torch::Tensor try_probe_sm_locality_domains(const torch::Tensor& buf) {
    const auto [latency, order] = probe(buf).sort(0);
    const auto sm_domain = order[0].to(torch::kUInt8);
    const auto tpcs = sm_domain.view({-1, kNumSMsPerTPC});
    const auto clear = latency[1].ge(latency[0] * kMinFarNearRatio).all().item<bool>();
    const auto consistent = torch::equal(tpcs, tpcs.select(1, 0).unsqueeze(1).expand_as(tpcs));
    return clear and consistent ? sm_domain : torch::Tensor();
}

// Without localization, an arbitrary SM mapping is functionally correct
static torch::Tensor get_even_sm_locality_domains() {
    const auto tpc_idx = torch::arange(jit->device.get_num_sms() / kNumSMsPerTPC, torch::dtype(torch::kInt).device(torch::kCUDA));
    return (tpc_idx % get_num_locality_domains()).to(torch::kUInt8).repeat_interleave(kNumSMsPerTPC);
}

// Probe multiple times; fallback when unstable
static torch::Tensor probe_sm_locality_domains() {
    const auto num_sms = jit->device.get_num_sms();
    const auto buf = empty_per_domain({num_sms, kNumProbeChunksPerSM, kNumLocalityDomainProbeChunkBytes}, torch::kUInt8);
    for (int attempt = 0; attempt < kNumMaxProbeAttempts; ++ attempt) {
        if (const auto sm_domain = try_probe_sm_locality_domains(buf); sm_domain.defined())
            return sm_domain;
    }
    TORCH_WARN("The SM locality domain probe found no stable mapping; using an even one");
    return get_even_sm_locality_domains();
}

static torch::Tensor get_sm_locality_domains() {
    if (not runtime->sm_locality_domains.defined())
        runtime->sm_locality_domains = is_localization_available() ? probe_sm_locality_domains() : get_even_sm_locality_domains();
    return runtime->sm_locality_domains;
}

static torch::Tensor balance_sm_locality_domains(const torch::Tensor& sm_domain) {
    const auto tpcs = sm_domain.view({-1, kNumSMsPerTPC});
    const auto tpc_domain = tpcs.select(1, 0);
    DG_HOST_ASSERT(tpcs.size(0) % 2 == 0);
    const int64_t num_tpcs = tpcs.size(0), num_domain_1 = tpc_domain.sum().item<int64_t>();
    const uint8_t larger = num_domain_1 * 2 > num_tpcs;
    const auto surplus = (tpc_domain == larger).nonzero().flatten().slice(0, num_tpcs / 2);
    const auto balanced = sm_domain.clone();
    balanced.view({-1, kNumSMsPerTPC}).index_fill_(0, surplus, larger ^ 1);
    return balanced;
}

static torch::Tensor get_balanced_sm_locality_domains() {
    if (not runtime->balanced_sm_locality_domains.defined())
        runtime->balanced_sm_locality_domains = balance_sm_locality_domains(get_sm_locality_domains());
    return runtime->balanced_sm_locality_domains;
}

static void release_mlopart() {
    get_balanced_sm_locality_domains();
    mlopart::release();
}

}  // namespace deep_gemm::locality_domain

// NOTES: exposed to Python as `deep_gemm.locality_domain.X` by the `_C` facade
TORCH_LIBRARY_FRAGMENT(deep_gemm, m) {
    using namespace deep_gemm::locality_domain;
    m.def("locality_domain_get_num_locality_domains() -> int", []() {
        return static_cast<int64_t>(get_num_locality_domains());
    });
    m.def("locality_domain_get_granularity() -> int", []() {
        return static_cast<int64_t>(get_granularity());
    });
    m.def("locality_domain_is_localization_available() -> bool", &is_localization_available);
    m.def("locality_domain_empty(int[] sizes, ScalarType dtype, int domain_idx) -> Tensor",
          [](const std::vector<int64_t>& sizes, const c10::ScalarType& dtype, const int64_t& domain_idx) {
        return empty(sizes, dtype, static_cast<int>(domain_idx));
    });
    m.def("locality_domain_empty_per_domain(int[] sizes, ScalarType dtype) -> Tensor",
          [](const std::vector<int64_t>& sizes, const c10::ScalarType& dtype) {
        return empty_per_domain(sizes, dtype);
    });
    m.def("locality_domain_is_localized(Tensor t) -> bool", [](const torch::Tensor& t) {
        return deep_gemm::is_localized(t);
    });
    m.def("locality_domain_get_sm_locality_domains() -> Tensor", &get_sm_locality_domains);
    m.def("locality_domain_get_balanced_sm_locality_domains() -> Tensor", &get_balanced_sm_locality_domains);
    m.def("locality_domain_release_mlopart() -> ()", &release_mlopart);
}
#endif

#include <vector>

#include <cuda_runtime.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>

#include "../jit_kernels/impls/sm100_locality_domain.hpp"
#include "../runtime/runtime.hpp"
#include "../utils/exception.hpp"
#include "../utils/layout.hpp"
#include "../utils/math.hpp"

namespace deep_gemm::locality_domain {

static int64_t get_num_locality_domains() {
    return LocalityDomainAllocator::get_num_locality_domains();
}

static int64_t get_granularity() {
    return static_cast<int64_t>(LocalityDomainAllocator::get_granularity());
}

static bool is_localization_available() {
    return mlopart::is_available();
}

static torch::stable::Device current_cuda_device() {
    int device_index = 0;
    DG_CUDA_RUNTIME_CHECK(cudaGetDevice(&device_index));
    return torch::stable::Device(torch::headeronly::kCUDA, device_index);
}

static int64_t numel(const std::vector<int64_t>& sizes) {
    int64_t value = 1;
    for (const auto size : sizes) {
        DG_HOST_ASSERT(size >= 0);
        value *= size;
    }
    return value;
}

static std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& sizes) {
    std::vector<int64_t> strides(sizes.size());
    int64_t stride = 1;
    for (size_t i = sizes.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= sizes[i];
    }
    return strides;
}

// PyTorch 2.10's stable from_blob has no deleter callback. The process-wide allocator
// therefore owns localized reservations until its destructor releases them at exit.
static torch::stable::Tensor empty(const std::vector<int64_t>& sizes,
                                   const torch::headeronly::ScalarType& dtype,
                                   const int64_t& domain_idx) {
    auto& allocator = locality_domain_allocator;
    const auto num_bytes = align<size_t>(
        static_cast<size_t>(numel(sizes)) * torch_compat::element_size(dtype),
        allocator.get_granularity());
    auto* ptr = allocator.reserve(num_bytes);
    allocator.map(ptr, num_bytes, static_cast<int>(domain_idx));
    return torch::stable::from_blob(
        ptr, sizes, contiguous_strides(sizes), current_cuda_device(), dtype);
}

static torch::stable::Tensor empty_per_domain(const std::vector<int64_t>& sizes,
                                              const torch::headeronly::ScalarType& dtype) {
    auto& allocator = locality_domain_allocator;
    const int64_t elem_size = static_cast<int64_t>(torch_compat::element_size(dtype));
    const int64_t num_domains = get_num_locality_domains();
    const int64_t num_slice_bytes = align<int64_t>(
        numel(sizes) * elem_size, static_cast<int64_t>(allocator.get_granularity()));
    auto* ptr = static_cast<char*>(allocator.reserve(num_slice_bytes * num_domains));
    for (int d = 0; d < num_domains; ++d)
        allocator.map(ptr + d * num_slice_bytes, num_slice_bytes, d);

    std::vector<int64_t> full_sizes = {num_domains};
    full_sizes.insert(full_sizes.end(), sizes.begin(), sizes.end());
    std::vector<int64_t> strides = {num_slice_bytes / elem_size};
    const auto slice_strides = contiguous_strides(sizes);
    strides.insert(strides.end(), slice_strides.begin(), slice_strides.end());
    return torch::stable::from_blob(ptr, full_sizes, strides, current_cuda_device(), dtype);
}

static bool is_localized_op(const torch::stable::Tensor& tensor) {
    return deep_gemm::is_localized(tensor);
}

static void probe_chase(const torch::stable::Tensor& buffer,
                        const torch::stable::Tensor& latency) {
    DG_HOST_ASSERT(buffer.size(0) == latency.size(0));
    for (int64_t domain = 0; domain < buffer.size(0); ++domain)
        sm100_locality_domain_probe_chase(
            torch::stable::select(buffer, 0, domain),
            torch::stable::select(latency, 0, domain));
}

static void set_sm_locality_domains(const torch::stable::Tensor& sm_domains,
                                    const torch::stable::Tensor& balanced_sm_domains) {
    runtime->sm_locality_domains = sm_domains;
    runtime->balanced_sm_locality_domains = balanced_sm_domains;
}

static torch::stable::Tensor get_sm_locality_domains() {
    DG_HOST_ASSERT(runtime->sm_locality_domains.defined());
    return runtime->sm_locality_domains;
}

static torch::stable::Tensor get_balanced_sm_locality_domains() {
    DG_HOST_ASSERT(runtime->balanced_sm_locality_domains.defined());
    return runtime->balanced_sm_locality_domains;
}

static void release_mlopart() {
    mlopart::release();
}

} // namespace deep_gemm::locality_domain

STABLE_TORCH_LIBRARY_FRAGMENT(deep_gemm, m) {
    m.def("locality_domain_get_num_locality_domains() -> int");
    m.def("locality_domain_get_granularity() -> int");
    m.def("locality_domain_is_localization_available() -> bool");
    m.def("locality_domain_empty(int[] sizes, ScalarType dtype, int domain_idx) -> Tensor");
    m.def("locality_domain_empty_per_domain(int[] sizes, ScalarType dtype) -> Tensor");
    m.def("locality_domain_is_localized(Tensor tensor) -> bool");
    m.def("locality_domain_probe_chase(Tensor buffer, Tensor(a!) latency) -> ()");
    m.def("locality_domain_set_sm_locality_domains(Tensor sm_domains, Tensor balanced_sm_domains) -> ()");
    m.def("locality_domain_get_sm_locality_domains() -> Tensor");
    m.def("locality_domain_get_balanced_sm_locality_domains() -> Tensor");
    m.def("locality_domain_release_mlopart() -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(deep_gemm, CompositeExplicitAutograd, m) {
    using namespace deep_gemm::locality_domain;
    m.impl("locality_domain_get_num_locality_domains", TORCH_BOX(&get_num_locality_domains));
    m.impl("locality_domain_get_granularity", TORCH_BOX(&get_granularity));
    m.impl("locality_domain_is_localization_available", TORCH_BOX(&is_localization_available));
    // These operators have no tensor argument from which the dispatcher could
    // infer CUDA, even though they allocate or return CUDA tensors.
    m.impl("locality_domain_empty", TORCH_BOX(&empty));
    m.impl("locality_domain_empty_per_domain", TORCH_BOX(&empty_per_domain));
    m.impl("locality_domain_get_sm_locality_domains", TORCH_BOX(&get_sm_locality_domains));
    m.impl("locality_domain_get_balanced_sm_locality_domains", TORCH_BOX(&get_balanced_sm_locality_domains));
    m.impl("locality_domain_release_mlopart", TORCH_BOX(&release_mlopart));
}

STABLE_TORCH_LIBRARY_IMPL(deep_gemm, CUDA, m) {
    using namespace deep_gemm::locality_domain;
    m.impl("locality_domain_is_localized", TORCH_BOX(&is_localized_op));
    m.impl("locality_domain_probe_chase", TORCH_BOX(&probe_chase));
    m.impl("locality_domain_set_sm_locality_domains", TORCH_BOX(&set_sm_locality_domains));
}
