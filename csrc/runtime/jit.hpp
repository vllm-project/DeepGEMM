#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include <cutlass/version.h>
#include <deep_jit/backend/cuda/backend.hpp>

namespace deep_gemm {

inline deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>> jit(nullptr);

inline void init_jit(const std::string& library_root_path) {
    const auto library_root = std::filesystem::absolute(library_root_path);
    const auto include_dir = library_root / "include";
    const auto config = deep_jit::Config(
        library_root,
        "DG",
        "cutlass-" + std::to_string(CUTLASS_VERSION),
        {include_dir},
        {"deep_gemm/"});

    jit = deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>>([config] {
        auto runtime = std::make_shared<deep_jit::Runtime<deep_jit::CUDA>>(config);

        // Use the SM100 family target while the pinned CUTLASS lacks SM107 guards.
        // Keep the physical capability unchanged for kernel dispatch. Remove this
        // override once native SM107 JIT compilation and execution are validated:
        // https://github.com/deepseek-ai/DeepGEMM/issues/461
        const auto [major, minor] = runtime->device.get_arch_pair();
        if (major == 10 and minor == 7) {
            runtime->default_compiler_options.arch = "100f";
            // SM107 has the K=64 MXF8F6F4 MMA; the flag is part of the JIT cache key.
            runtime->default_compiler_options.nvcc_flags->emplace_back("-DDG_SM107_MXF8F6F4_K64=1");
        }

        runtime->default_compiler_options.nvcc_flags->
            emplace_back("--diag-suppress=39,161,174,177,186,940");
        runtime->default_compiler_options.nvcc_flags->
            emplace_back("--compiler-options=-Wno-deprecated-declarations,-Wno-abi");
        return runtime;
    });
}

}  // namespace deep_gemm
