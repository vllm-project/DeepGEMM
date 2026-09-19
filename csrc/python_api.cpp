#include <memory>

#include <torch/all.h>
#include <torch/custom_class.h>
#include <torch/library.h>
#include "utils/registration.h"

#include <deep_jit/backend/cuda/backend.hpp>

#include "runtime/runtime.hpp"
#include "apis/config.hpp"
#include "apis/attention.hpp"
#include "apis/einsum.hpp"
#include "apis/hyperconnection.hpp"
#include "apis/gemm.hpp"
#include "apis/layout.hpp"
#include "apis/mega_moe.hpp"
#include "apis/mega_mhc.hpp"
#include "apis/mega_gate.hpp"

namespace deep_gemm {
// Preserve the opaque runtime handle exposed by DeepJIT's former Python binding.
struct JitRuntimeHandle : torch::CustomClassHolder {
    std::shared_ptr<deep_jit::Runtime<deep_jit::CUDA>> value;
    JitRuntimeHandle() : value(jit.get()) {}
};
}

TORCH_LIBRARY(deep_gemm, m) {
    // Register JIT objects
    m.class_<deep_gemm::JitRuntimeHandle>("Runtime");
    m.def("get_jit() -> __torch__.torch.classes.deep_gemm.Runtime", []() { return c10::make_intrusive<deep_gemm::JitRuntimeHandle>(); });
}

REGISTER_EXTENSION(_C_extension)
