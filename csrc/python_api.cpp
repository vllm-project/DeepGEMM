#include <memory>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include "utils/torch_compat.hpp"
#include "utils/registration.h"

#include <deep_jit/backend/cuda/backend.hpp>

#include "runtime/runtime.hpp"
#include "apis/config.hpp"
#include "apis/epilogue_class.hpp"
#include "apis/attention.hpp"
#include "apis/einsum.hpp"
#include "apis/hyperconnection.hpp"
#include "apis/gemm.hpp"
#include "apis/layout.hpp"
#include "apis/locality_domain.hpp"
#include "apis/mega_moe.hpp"
#include "apis/nvfp4_mega_moe.hpp"
#include "apis/mega_mhc.hpp"
#include "apis/mega_gate.hpp"

namespace deep_gemm {
static void ensure_jit() { (void)jit.get(); }
}
STABLE_TORCH_LIBRARY(deep_gemm, m) {
    m.def("_ensure_jit() -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(deep_gemm, CompositeExplicitAutograd, m) {
    m.impl("_ensure_jit", TORCH_BOX(&deep_gemm::ensure_jit));
}

REGISTER_EXTENSION(_C_extension)
