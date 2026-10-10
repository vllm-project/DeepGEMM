#pragma once

// ATen ops with no curated `torch::stable::ops` wrapper yet, called via `torch_call_dispatcher`.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cuda_runtime.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/version.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/shim_utils.h>

namespace deep_gemm::torch_compat {

namespace detail {

inline void call_dispatcher(const char* op_name, const char* overload_name, StableIValue* stack) {
    TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(op_name, overload_name, stack, TORCH_ABI_VERSION));
}

// `torch_call_dispatcher` can't marshal `Scalar`-typed args; wrap in a 0-dim tensor instead.
inline torch::stable::Tensor scalar_tensor_like(const torch::stable::Tensor& self, double value) {
    return torch::stable::full({}, value, self.scalar_type(), std::nullopt, self.device());
}

} // namespace detail

// `aten::bitwise_and.Tensor(Tensor self, Tensor other) -> Tensor`
// TODO(torch 2.14): Remove this local wrapper and call torch::stable::bitwise_and directly.
inline torch::stable::Tensor bitwise_and(const torch::stable::Tensor& self, const torch::stable::Tensor& other) {
    std::array<StableIValue, 2> stack{
        torch::stable::detail::from(self), torch::stable::detail::from(other)};
    detail::call_dispatcher("aten::bitwise_and", "Tensor", stack.data());
    return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// `aten::bitwise_and.Scalar(Tensor self, Scalar other) -> Tensor`, via the `.Tensor` overload
inline torch::stable::Tensor bitwise_and(const torch::stable::Tensor& self, int64_t other) {
    return bitwise_and(self, detail::scalar_tensor_like(self, static_cast<double>(other)));
}

// `aten::bitwise_or.Tensor(Tensor self, Tensor other) -> Tensor`
// TODO(torch 2.14): Remove this local wrapper and call torch::stable::bitwise_or directly.
inline torch::stable::Tensor bitwise_or(const torch::stable::Tensor& self, const torch::stable::Tensor& other) {
    std::array<StableIValue, 2> stack{
        torch::stable::detail::from(self), torch::stable::detail::from(other)};
    detail::call_dispatcher("aten::bitwise_or", "Tensor", stack.data());
    return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// `aten::bitwise_or.Scalar(Tensor self, Scalar other) -> Tensor`, via the `.Tensor` overload
inline torch::stable::Tensor bitwise_or(const torch::stable::Tensor& self, int64_t other) {
    return bitwise_or(self, detail::scalar_tensor_like(self, static_cast<double>(other)));
}

// `aten::bitwise_left_shift.Tensor(Tensor self, Tensor other) -> Tensor`
// TODO(torch 2.14): Remove this local wrapper and call torch::stable::bitwise_left_shift directly.
inline torch::stable::Tensor bitwise_left_shift(const torch::stable::Tensor& self, const torch::stable::Tensor& other) {
    std::array<StableIValue, 2> stack{
        torch::stable::detail::from(self), torch::stable::detail::from(other)};
    detail::call_dispatcher("aten::bitwise_left_shift", "Tensor", stack.data());
    return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// `aten::bitwise_left_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor`, via the `.Tensor` overload
inline torch::stable::Tensor bitwise_left_shift(const torch::stable::Tensor& self, int64_t other) {
    return bitwise_left_shift(self, detail::scalar_tensor_like(self, static_cast<double>(other)));
}

// `aten::bitwise_right_shift.Tensor(Tensor self, Tensor other) -> Tensor`
// TODO(torch 2.14): Remove this local wrapper and call torch::stable::bitwise_right_shift directly.
inline torch::stable::Tensor bitwise_right_shift(const torch::stable::Tensor& self, const torch::stable::Tensor& other) {
    std::array<StableIValue, 2> stack{
        torch::stable::detail::from(self), torch::stable::detail::from(other)};
    detail::call_dispatcher("aten::bitwise_right_shift", "Tensor", stack.data());
    return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// `aten::bitwise_right_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor`, via the `.Tensor` overload
inline torch::stable::Tensor bitwise_right_shift(const torch::stable::Tensor& self, int64_t other) {
    return bitwise_right_shift(self, detail::scalar_tensor_like(self, static_cast<double>(other)));
}

// `aten::index_select(Tensor self, int dim, Tensor index) -> Tensor`
// TODO(torch 2.14): Remove this local wrapper and call torch::stable::index_select directly.
inline torch::stable::Tensor index_select(const torch::stable::Tensor& self, int64_t dim,
                                          const torch::stable::Tensor& index) {
    std::array<StableIValue, 3> stack{
        torch::stable::detail::from(self), torch::stable::detail::from(dim),
        torch::stable::detail::from(index)};
    detail::call_dispatcher("aten::index_select", "", stack.data());
    return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// `aten::floor_divide(Tensor self, Tensor other) -> Tensor`
// TODO(torch 2.14): Remove this local wrapper and call torch::stable::floor_divide directly.
inline torch::stable::Tensor floor_divide(const torch::stable::Tensor& self, const torch::stable::Tensor& other) {
    std::array<StableIValue, 2> stack{
        torch::stable::detail::from(self), torch::stable::detail::from(other)};
    detail::call_dispatcher("aten::floor_divide", "", stack.data());
    return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// `aten::floor_divide.Scalar(Tensor self, Scalar other) -> Tensor`, via the `.Tensor` overload
inline torch::stable::Tensor floor_divide(const torch::stable::Tensor& self, int64_t other) {
    return floor_divide(self, detail::scalar_tensor_like(self, static_cast<double>(other)));
}

// `Tensor::nbytes()` has no stable ABI equivalent; reconstruct from `numel()` and `element_size()`.
inline size_t nbytes(const torch::stable::Tensor& self) {
    return static_cast<size_t>(self.numel()) * self.element_size();
}

// `aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)`
// TODO(torch 2.14): Remove this local wrapper and call torch::stable::permute directly.
inline torch::stable::Tensor permute(const torch::stable::Tensor& self, std::vector<int64_t> dims) {
    std::array<StableIValue, 2> stack{
        torch::stable::detail::from(self), torch::stable::detail::from(dims)};
    detail::call_dispatcher("aten::permute", "", stack.data());
    return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// Call empty_strided through the stable dispatcher because no stable C++ wrapper is available.
inline torch::stable::Tensor empty_strided(const std::vector<int64_t>& size,
                                           const std::vector<int64_t>& stride,
                                           std::optional<torch::headeronly::ScalarType> dtype,
                                           std::optional<torch::stable::Device> device) {
    std::array<StableIValue, 6> stack{
        torch::stable::detail::from(size),
        torch::stable::detail::from(stride),
        torch::stable::detail::from(dtype),
        torch::stable::detail::from(std::nullopt),  // layout
        torch::stable::detail::from(device),
        torch::stable::detail::from(std::nullopt)};  // pin_memory
    detail::call_dispatcher("aten::empty_strided", "", stack.data());
    return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// `aten::view.dtype`: a different overload than the shape-based `view` in `torch::stable::ops`.
// TODO(torch 2.14): Remove this local wrapper and call torch::stable::view(self, dtype) directly.
inline torch::stable::Tensor view_dtype(const torch::stable::Tensor& self, torch::headeronly::ScalarType dtype) {
    std::array<StableIValue, 2> stack{
        torch::stable::detail::from(self), torch::stable::detail::from(dtype)};
    detail::call_dispatcher("aten::view", "dtype", stack.data());
    return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// `c10::elementSize(ScalarType)` has no stable ABI equivalent; the C shim exposes the same table.
inline size_t element_size(torch::headeronly::ScalarType dtype) {
    const int32_t shim_dtype = torch::stable::detail::to<int32_t>(torch::stable::detail::from(dtype));
    return aoti_torch_dtype_element_size(shim_dtype);
}

// Get the current CUDA stream through the stable-compatible AOTI shim.
inline cudaStream_t current_stream(const torch::stable::Tensor& tensor) {
    void* stream = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(tensor.get_device_index(), &stream));
    return static_cast<cudaStream_t>(stream);
}

using StreamKey = std::pair<int32_t, cudaStream_t>;

// Combine the device and stream so runtime state is keyed to the active CUDA stream.
inline StreamKey stream_key(const torch::stable::Tensor& tensor) {
    return {tensor.get_device_index(), current_stream(tensor)};
}

// Query CUDA graph capture state without relying on unstable PyTorch APIs.
inline bool is_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status;
    const auto error = cudaStreamIsCapturing(stream, &status);
    STD_TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
    return status != cudaStreamCaptureStatusNone;
}

// Adapt the stable API's mutable Tensor& parameter to const handles and temporaries.
inline torch::stable::Tensor zero_(torch::stable::Tensor self) {
    return torch::stable::zero_(self);
}

// Use the dispatcher for slice because the stable header API does not expose this overload here.
inline torch::stable::Tensor slice(const torch::stable::Tensor& self, int64_t dim,
                                  int64_t start, int64_t end, int64_t step = 1) {
    std::array<StableIValue, 5> stack{
        torch::stable::detail::from(self), torch::stable::detail::from(dim),
        torch::stable::detail::from(std::optional<int64_t>(start)),
        torch::stable::detail::from(std::optional<int64_t>(end)), torch::stable::detail::from(step)};
    detail::call_dispatcher("aten::slice", "Tensor", stack.data());
    return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// Provide a contiguous-stride convenience form; callers with custom strides use the stable API directly.
inline torch::stable::Tensor from_blob(void* data, std::vector<int64_t> sizes,
                                     torch::stable::Device device, torch::headeronly::ScalarType dtype) {
    std::vector<int64_t> strides(sizes.size());
    int64_t stride = 1;
    for (size_t i = sizes.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= std::max<int64_t>(sizes[i], 1);
    }
    return torch::stable::from_blob(data, sizes, strides, device, dtype);
}

// Convert a stable PyTorch dtype to the CUDA datatype expected by cuBLASLt.
inline cudaDataType scalar_type_to_cuda_data_type(torch::headeronly::ScalarType type) {
    switch (type) {
        case torch::headeronly::ScalarType::Float: return CUDA_R_32F;
        case torch::headeronly::ScalarType::Half: return CUDA_R_16F;
        case torch::headeronly::ScalarType::BFloat16: return CUDA_R_16BF;
        case torch::headeronly::ScalarType::Float8_e4m3fn: return CUDA_R_8F_E4M3;
        default: STD_TORCH_CHECK(false, "Unsupported cuBLASLt scalar type");
    }
}

// Adapt the stable API's mutable Tensor& parameter to const handles and temporaries.
inline torch::stable::Tensor copy_(torch::stable::Tensor self, const torch::stable::Tensor& src) {
    return torch::stable::copy_(self, src);
}

// Adapt the stable API's mutable Tensor& parameter to const handles and temporaries.
inline torch::stable::Tensor narrow(torch::stable::Tensor self, int64_t dim, int64_t start, int64_t length) {
    return torch::stable::narrow(self, dim, start, length);
}

// Query allocated storage size, which can differ from the tensor's logical byte count.
inline uint64_t storage_nbytes(const torch::stable::Tensor& self) {
    int64_t bytes = 0;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_storage_size(self.get(), &bytes));
    return static_cast<uint64_t>(bytes);
}
}
