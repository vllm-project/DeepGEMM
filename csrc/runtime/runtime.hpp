#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>

#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include "../utils/torch_compat.hpp"
#include <cublasLt.h>

#include <deep_jit/utils/env.hpp>
#include <deep_jit/utils/lazy.hpp>

#include "../utils/exception.hpp"
#include "jit.hpp"
#include "locality_domain.hpp"

namespace deep_gemm {

class Runtime {
public:
    int num_sms = 0, tc_util = 0;

    // cuBLASLt utils
    // cuBLAS will select worse heuristics when workspace > 16 MiB with shape, e.g. m=128, n=7168, k=16384
    static constexpr size_t kCublasLtWorkspaceSize = 16 * 1024 * 1024;
    // cuBLASLt may use the workspace asynchronously, so concurrent streams cannot share it.
    std::map<torch_compat::StreamKey, torch::stable::Tensor> cublaslt_workspaces;

    // Create the cuBLASLt handle ourselves
    cublasLtHandle_t cublaslt_handle;
    bool use_pytorch_managed_cublaslt_handle;
    bool use_temp_cublaslt_workspace;

    // The locality domain of every SM, probed lazily
    torch::stable::Tensor sm_locality_domains;
    // We need to balance the SMs across locality domains to provide simpler load balancing
    torch::stable::Tensor balanced_sm_locality_domains;

    explicit Runtime() {
        // Whether to use PyTorch cuBLASLt
        // By default, we don't use it,
        // as `at::cuda::getCurrentCUDABlasLtHandle` has large CPU overhead with some PyTorch versions
        use_pytorch_managed_cublaslt_handle = deep_jit::get_env<int>("DG_USE_PYTORCH_CUBLASLT_HANDLE", 0) > 0;
        // Whether to create workspace tensor on each call instead of holding one.
        // Enabled by compute-sanitizer tests, which trigger `cudaErrorCudartUnloading`
        // when the workspace tensor is destructed after CUDA driver shutdown.
        use_temp_cublaslt_workspace = deep_jit::get_env<int>("DG_USE_TEMP_CUBLASLT_WORKSPACE", 0) > 0;

        if (not use_pytorch_managed_cublaslt_handle)
            DG_CUBLASLT_CHECK(cublasLtCreate(&cublaslt_handle));
    }

    ~Runtime() noexcept(false) {
        if (not use_pytorch_managed_cublaslt_handle)
            DG_CUBLASLT_CHECK(cublasLtDestroy(cublaslt_handle));
    }

    cublasLtHandle_t get_cublaslt_handle() const {
        if (use_pytorch_managed_cublaslt_handle) {
            void* handle = nullptr;
            TORCH_ERROR_CODE_CHECK(torch_get_current_cuda_blas_handle(&handle));
            return reinterpret_cast<cublasLtHandle_t>(handle);
        }

        // Self-managed handle
        return cublaslt_handle;
    }

    torch::stable::Tensor get_cublaslt_workspace(const torch::stable::Tensor& reference) {
        const auto stream = torch_compat::stream_key(reference);
        if (use_temp_cublaslt_workspace)
            return torch::stable::new_empty(reference, {kCublasLtWorkspaceSize}, torch::headeronly::ScalarType::Byte);

        auto& workspace = cublaslt_workspaces[stream];
        if (not workspace.defined())
            workspace = torch::stable::new_empty(reference, {kCublasLtWorkspaceSize}, torch::headeronly::ScalarType::Byte);
        return workspace;
    }

    void set_num_sms(const int& new_num_sms) {
        DG_HOST_ASSERT(0 < new_num_sms and new_num_sms <= jit->device.get_num_sms());
        num_sms = new_num_sms;
    }

    int get_num_sms() {
        if (num_sms == 0)
            num_sms = jit->device.get_num_sms();
        return num_sms;
    }

    bool is_cublaslt_available() {
        return get_num_sms() == jit->device.get_num_sms();
    }

    void set_tc_util(const int& new_tc_util) {
        DG_HOST_ASSERT(0 <= new_tc_util and new_tc_util <= 100);
        tc_util = new_tc_util;
    }

    int get_tc_util() const {
        return tc_util == 0 ? 100 : tc_util;
    }

};

inline auto runtime = deep_jit::LazyInit<Runtime>([](){ return std::make_shared<Runtime>(); });

}  // namespace deep_gemm
