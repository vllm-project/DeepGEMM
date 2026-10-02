#pragma once

#include <torch/csrc/stable/version.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <deep_gemm/common/exception.cuh>

DG_STATIC_ASSERT(TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0,
                 "DeepGEMM requires PyTorch 2.10 or newer");
DG_STATIC_ASSERT(CUDA_VERSION >= 12090, "DeepGEMM requires CUDA Driver API 12.9 or newer");
DG_STATIC_ASSERT(CUDART_VERSION >= 12090, "DeepGEMM requires CUDA Runtime API 12.9 or newer");
