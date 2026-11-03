#pragma once

#include <torch/csrc/stable/version.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <deep_gemm/common/exception.cuh>

// The convenience wrappers require 2.14 headers but dispatch through the 2.11 stable ABI.
DG_STATIC_ASSERT(TORCH_ABI_VERSION >= TORCH_VERSION_2_14_0,
                 "Building DeepGEMM requires PyTorch 2.14 or newer");
DG_STATIC_ASSERT(TORCH_FEATURE_VERSION >= TORCH_VERSION_2_11_0,
                 "DeepGEMM requires the PyTorch 2.11 stable ABI or newer");
DG_STATIC_ASSERT(CUDA_VERSION >= 12090, "DeepGEMM requires CUDA Driver API 12.9 or newer");
DG_STATIC_ASSERT(CUDART_VERSION >= 12090, "DeepGEMM requires CUDA Runtime API 12.9 or newer");
