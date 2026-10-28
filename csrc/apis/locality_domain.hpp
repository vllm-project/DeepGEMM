#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>

#include "../jit_kernels/impls/sm100_locality_domain.hpp"
#include "../jit_kernels/impls/smxx_layout.hpp"
#include "../runtime/runtime.hpp"
#include "../utils/exception.hpp"
#include "../utils/layout.hpp"
#include "../utils/math.hpp"

namespace deep_gemm::locality_domain {

// Compatibility helpers for locality operations without stable 2.11 C++ APIs.
namespace stable_ops {

using Tensor = torch::stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;

static torch::stable::Device current_cuda_device() {
    return torch::stable::Device(
        torch::headeronly::kCUDA, torch::stable::accelerator::getCurrentDeviceIndex());
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
        stride *= std::max<int64_t>(sizes[i], 1);
    }
    return strides;
}

static Tensor zeros(const std::vector<int64_t>& sizes, ScalarType dtype, torch::stable::Device device) {
    return torch::stable::full(sizes, 0.0, dtype, std::nullopt, device);
}

static Tensor binary(const char* op, const Tensor& self, const Tensor& other) {
    std::array<StableIValue, 2> stack{
        torch::stable::detail::from(self), torch::stable::detail::from(other)};
    torch_compat::detail::call_dispatcher(op, "Tensor", stack.data());
    return torch::stable::detail::to<Tensor>(stack[0]);
}

static Tensor binary_scalar(const char* op, const Tensor& self, double value) {
    const auto other = torch::stable::full({}, value, self.scalar_type(), std::nullopt, self.device());
    return binary(op, self, other);
}

static Tensor roll(const Tensor& self, int64_t shift, int64_t dim) {
    std::array<StableIValue, 3> stack{
        torch::stable::detail::from(self), torch::stable::detail::from(std::vector<int64_t>{shift}),
        torch::stable::detail::from(std::vector<int64_t>{dim})};
    torch_compat::detail::call_dispatcher("aten::roll", "", stack.data());
    return torch::stable::detail::to<Tensor>(stack[0]);
}

static Tensor repeat_interleave(const Tensor& self, int64_t repeats) {
    std::array<StableIValue, 4> stack{
        torch::stable::detail::from(self), torch::stable::detail::from(repeats),
        torch::stable::detail::from(std::optional<int64_t>{}),
        torch::stable::detail::from(std::optional<int64_t>(self.numel() * repeats))};
    torch_compat::detail::call_dispatcher("aten::repeat_interleave", "self_int", stack.data());
    return torch::stable::detail::to<Tensor>(stack[0]);
}

static bool all(const Tensor& self) {
    std::array<StableIValue, 1> stack{torch::stable::detail::from(self)};
    torch_compat::detail::call_dispatcher("aten::all", "", stack.data());
    const auto result = torch::stable::detail::to<Tensor>(stack[0]);
    bool value = false;
    TORCH_ERROR_CODE_CHECK(aoti_torch_item_bool(result.get(), &value));
    return value;
}

static bool equal(const Tensor& self, const Tensor& other) {
    std::array<StableIValue, 2> stack{
        torch::stable::detail::from(self), torch::stable::detail::from(other)};
    torch_compat::detail::call_dispatcher("aten::equal", "", stack.data());
    return torch::stable::detail::to<bool>(stack[0]);
}

static Tensor median(const Tensor& self, int64_t dim) {
    std::array<StableIValue, 3> stack{
        torch::stable::detail::from(self), torch::stable::detail::from(dim),
        torch::stable::detail::from(false)};
    torch_compat::detail::call_dispatcher("aten::median", "dim", stack.data());
    const auto values = torch::stable::detail::to<Tensor>(stack[0]);
    (void)torch::stable::detail::to<Tensor>(stack[1]);
    return values;
}

static std::pair<Tensor, Tensor> sort(const Tensor& self, int64_t dim) {
    std::array<StableIValue, 3> stack{
        torch::stable::detail::from(self), torch::stable::detail::from(dim),
        torch::stable::detail::from(false)};
    torch_compat::detail::call_dispatcher("aten::sort", "", stack.data());
    return {torch::stable::detail::to<Tensor>(stack[0]), torch::stable::detail::to<Tensor>(stack[1])};
}

static Tensor nonzero(const Tensor& self) {
    std::array<StableIValue, 1> stack{torch::stable::detail::from(self)};
    torch_compat::detail::call_dispatcher("aten::nonzero", "", stack.data());
    return torch::stable::detail::to<Tensor>(stack[0]);
}

static Tensor expand(const Tensor& self, const std::vector<int64_t>& sizes) {
    std::array<StableIValue, 3> stack{
        torch::stable::detail::from(self), torch::stable::detail::from(sizes),
        torch::stable::detail::from(false)};
    torch_compat::detail::call_dispatcher("aten::expand", "", stack.data());
    return torch::stable::detail::to<Tensor>(stack[0]);
}

static void index_fill_(const Tensor& self, int64_t dim, const Tensor& index, uint8_t value) {
    const auto fill = torch::stable::full({}, value, self.scalar_type(), std::nullopt, self.device());
    std::array<StableIValue, 4> stack{
        torch::stable::detail::from(self), torch::stable::detail::from(dim),
        torch::stable::detail::from(index), torch::stable::detail::from(fill)};
    torch_compat::detail::call_dispatcher("aten::index_fill_", "int_Tensor", stack.data());
    (void)torch::stable::detail::to<Tensor>(stack[0]);
}

}  // namespace stable_ops

static int64_t get_num_locality_domains() {
    return LocalityDomainAllocator::get_num_locality_domains();
}

static int64_t get_granularity() {
    return static_cast<int64_t>(LocalityDomainAllocator::get_granularity());
}

static bool is_localization_available() {
    return mlopart::is_available();
}

static void release(void* ptr) {
    locality_domain_allocator.release(ptr);
}

static torch::stable::Tensor empty(const std::vector<int64_t>& sizes,
                                   const torch::headeronly::ScalarType& dtype,
                                   const int64_t& domain_idx) {
    auto& allocator = locality_domain_allocator;
    const auto num_bytes = align<size_t>(
        static_cast<size_t>(stable_ops::numel(sizes)) * torch_compat::element_size(dtype),
        allocator.get_granularity());
    const auto ptr = allocator.reserve(num_bytes);
    allocator.map(ptr, num_bytes, static_cast<int>(domain_idx));
    return torch::stable::from_blob(
        ptr, sizes, stable_ops::contiguous_strides(sizes), stable_ops::current_cuda_device(), dtype, release);
}

static torch::stable::Tensor empty_per_domain(const std::vector<int64_t>& sizes,
                                              const torch::headeronly::ScalarType& dtype) {
    auto& allocator = locality_domain_allocator;
    const int64_t elem_size = static_cast<int64_t>(torch_compat::element_size(dtype));
    const int64_t num_domains = get_num_locality_domains();
    const int64_t num_slice_bytes = align<int64_t>(
        stable_ops::numel(sizes) * elem_size, static_cast<int64_t>(allocator.get_granularity()));
    auto* ptr = static_cast<char*>(allocator.reserve(num_slice_bytes * num_domains));
    for (int d = 0; d < num_domains; ++d)
        allocator.map(ptr + d * num_slice_bytes, num_slice_bytes, d);

    std::vector<int64_t> full_sizes = {num_domains};
    full_sizes.insert(full_sizes.end(), sizes.begin(), sizes.end());
    std::vector<int64_t> strides = {num_slice_bytes / elem_size};
    const auto slice_strides = stable_ops::contiguous_strides(sizes);
    strides.insert(strides.end(), slice_strides.begin(), slice_strides.end());
    return torch::stable::from_blob(ptr, full_sizes, strides, stable_ops::current_cuda_device(), dtype, release);
}

static bool is_localized_op(const torch::stable::Tensor& tensor) {
    return deep_gemm::is_localized(tensor);
}

static constexpr int kNumSMsPerTPC = 2;
static constexpr int kNumLinesPerChunk = kNumLocalityDomainProbeHops;
static constexpr int kNumWordsPerLine = kNumLocalityDomainProbeLineBytes / sizeof(int);
static constexpr int kNumProbeChunksPerSM = 8;
static constexpr int kChainLineStride = 7;
static constexpr float kMinFarNearRatio = 1.25f;
static constexpr int kNumMaxProbeAttempts = 5;

static void flush_l2() {
    stable_ops::zeros({256 << 20}, torch::headeronly::ScalarType::Byte, stable_ops::current_cuda_device());
}

// One chunk of word indices: line l points to line (l + kChainLineStride) % kNumLinesPerChunk.
static torch::stable::Tensor get_probe_chain() {
    const auto device = stable_ops::current_cuda_device();
    const auto chain = stable_ops::zeros(
        {kNumLinesPerChunk, kNumWordsPerLine}, torch::headeronly::ScalarType::Int, device);
    const auto line = torch::stable::to(
        deep_gemm::deep_gemm_arange_int64(kNumLinesPerChunk, device),
        torch::headeronly::ScalarType::Int);
    const auto next_line = stable_ops::roll(line, -kChainLineStride, 0);
    const auto offsets = stable_ops::binary_scalar("aten::mul", next_line, kNumWordsPerLine);
    torch_compat::copy_(torch::stable::select(chain, 1, 0), offsets);
    return torch::stable::flatten(chain);
}

// (num_domains, num_sms, probe_chunks, probe_chunk_bytes) -> (num_domains, num_sms).
static torch::stable::Tensor probe(const torch::stable::Tensor& buf) {
    const auto words = torch::stable::view(
        torch::stable::view(buf, torch::headeronly::ScalarType::Int),
        {buf.size(0), -1, kNumLocalityDomainProbeChunkBytes / 4});
    torch_compat::copy_(words, get_probe_chain());
    const auto latency = stable_ops::zeros(
        {buf.size(0), buf.size(1), buf.size(2)}, torch::headeronly::ScalarType::Short, buf.device());
    flush_l2();
    for (int64_t d = 0; d < buf.size(0); ++d)
        sm100_locality_domain_probe_chase(
            torch::stable::select(buf, 0, d), torch::stable::select(latency, 0, d));
    DG_HOST_ASSERT(stable_ops::all(stable_ops::binary_scalar("aten::gt", latency, 0)));
    return stable_ops::median(torch::stable::to(latency, torch::headeronly::ScalarType::Float), -1);
}

static torch::stable::Tensor try_probe_sm_locality_domains(const torch::stable::Tensor& buf) {
    const auto [latency, order] = stable_ops::sort(probe(buf), 0);
    const auto sm_domain = torch::stable::to(
        torch::stable::select(order, 0, 0), torch::headeronly::ScalarType::Byte);
    const auto tpcs = torch::stable::view(sm_domain, {-1, kNumSMsPerTPC});
    const auto threshold = stable_ops::binary_scalar(
        "aten::mul", torch::stable::select(latency, 0, 0), kMinFarNearRatio);
    const auto clear = stable_ops::all(stable_ops::binary(
        "aten::ge", torch::stable::select(latency, 0, 1), threshold));
    const auto tpc_first = torch::stable::unsqueeze(torch::stable::select(tpcs, 1, 0), 1);
    const auto consistent = stable_ops::equal(
        tpcs, stable_ops::expand(tpc_first, {tpcs.size(0), tpcs.size(1)}));
    return clear and consistent ? sm_domain : torch::stable::Tensor();
}

// Without localization, an arbitrary SM mapping is functionally correct.
static torch::stable::Tensor get_even_sm_locality_domains() {
    const auto tpc_idx = torch::stable::to(
        deep_gemm::deep_gemm_arange_int64(
            jit->device.get_num_sms() / kNumSMsPerTPC, stable_ops::current_cuda_device()),
        torch::headeronly::ScalarType::Int);
    const auto domain = stable_ops::binary_scalar("aten::remainder", tpc_idx, get_num_locality_domains());
    return stable_ops::repeat_interleave(
        torch::stable::to(domain, torch::headeronly::ScalarType::Byte), kNumSMsPerTPC);
}

static torch::stable::Tensor probe_sm_locality_domains() {
    const auto num_sms = jit->device.get_num_sms();
    const auto buf = empty_per_domain(
        {num_sms, kNumProbeChunksPerSM, kNumLocalityDomainProbeChunkBytes},
        torch::headeronly::ScalarType::Byte);
    for (int attempt = 0; attempt < kNumMaxProbeAttempts; ++attempt) {
        if (const auto sm_domain = try_probe_sm_locality_domains(buf); sm_domain.defined())
            return sm_domain;
    }
    aoti_torch_warn(__func__, __FILE__, static_cast<uint32_t>(__LINE__),
                    "The SM locality domain probe found no stable mapping; using an even one");
    return get_even_sm_locality_domains();
}

static torch::stable::Tensor get_sm_locality_domains() {
    if (not runtime->sm_locality_domains.defined())
        runtime->sm_locality_domains = is_localization_available()
            ? probe_sm_locality_domains() : get_even_sm_locality_domains();
    return runtime->sm_locality_domains;
}

static torch::stable::Tensor balance_sm_locality_domains(const torch::stable::Tensor& sm_domain) {
    const auto tpcs = torch::stable::view(sm_domain, {-1, kNumSMsPerTPC});
    const auto tpc_domain = torch::stable::select(tpcs, 1, 0);
    DG_HOST_ASSERT(tpcs.size(0) % 2 == 0);
    const int64_t num_tpcs = tpcs.size(0);
    int64_t num_domain_1 = 0;
    const auto domain_1_sum = torch::stable::sum(tpc_domain);
    TORCH_ERROR_CODE_CHECK(aoti_torch_item_int64(domain_1_sum.get(), &num_domain_1));
    const uint8_t larger = num_domain_1 * 2 > num_tpcs;
    const auto matches = torch::stable::flatten(
        stable_ops::nonzero(stable_ops::binary_scalar("aten::eq", tpc_domain, larger)));
    const auto surplus = torch_compat::slice(matches, 0, num_tpcs / 2, matches.numel());
    const auto balanced = torch::stable::clone(sm_domain);
    stable_ops::index_fill_(torch::stable::view(balanced, {-1, kNumSMsPerTPC}), 0, surplus, larger ^ 1);
    return balanced;
}

static torch::stable::Tensor get_balanced_sm_locality_domains() {
    if (not runtime->balanced_sm_locality_domains.defined())
        runtime->balanced_sm_locality_domains = balance_sm_locality_domains(get_sm_locality_domains());
    return runtime->balanced_sm_locality_domains;
}

static void release_mlopart() {
    get_balanced_sm_locality_domains();
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
}
