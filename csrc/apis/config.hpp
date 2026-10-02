#pragma once

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include "../utils/torch_compat.hpp"

#include "../jit_kernels/heuristics/runtime.hpp"
#include "../runtime/runtime.hpp"

namespace deep_gemm::config {
static void init(const std::string& library_root_path) {
        init_jit(library_root_path);
}
static void set_num_sms(const int64_t& new_num_sms) {
        runtime->set_num_sms(new_num_sms);
}
static int64_t get_num_sms() {
       return int64_t(runtime->get_num_sms());
}
static void set_tc_util(const int64_t& new_tc_util) {
        runtime->set_tc_util(new_tc_util);
}
static int64_t get_tc_util() {
        return int64_t(runtime->get_tc_util());
}
static void set_pdl(const bool& new_enable_pdl) {
        jit->default_launch_options.enable_pdl = new_enable_pdl;
}
static bool get_pdl() {
        return *jit->default_launch_options.enable_pdl;
}
static void use_deterministic_algorithms(const bool enabled) {
        heuristics_runtime->use_deterministic_algorithms(enabled);
}
static void set_ignore_compile_dims(const bool& new_value) {
        heuristics_runtime->set_ignore_compile_dims(new_value);
}
static void set_block_size_multiple_of(const std::vector<int64_t>& new_value) {
        DG_HOST_ASSERT(new_value.size() == 1 or new_value.size() == 2);
        heuristics_runtime->set_block_size_multiple_of(new_value[0], new_value.size() == 1 ? new_value[0] : new_value[1]);
}
}

STABLE_TORCH_LIBRARY_FRAGMENT(deep_gemm, m) {
    m.def("init(str library_root_path) -> ()");
    m.def("set_num_sms(int new_num_sms) -> ()");
    m.def("get_num_sms() -> int");
    m.def("set_tc_util(int new_tc_util) -> ()");
    m.def("get_tc_util() -> int");
    m.def("set_pdl(bool new_enable_pdl) -> ()");
    m.def("get_pdl() -> bool");
    m.def("use_deterministic_algorithms(bool enabled) -> ()");
    m.def("set_ignore_compile_dims(bool new_value) -> ()");
    m.def("set_block_size_multiple_of(int[] new_value) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(deep_gemm, CompositeExplicitAutograd, m) {
    m.impl("init", TORCH_BOX(&deep_gemm::config::init));
    m.impl("set_num_sms", TORCH_BOX(&deep_gemm::config::set_num_sms));
    m.impl("get_num_sms", TORCH_BOX(&deep_gemm::config::get_num_sms));
    m.impl("set_tc_util", TORCH_BOX(&deep_gemm::config::set_tc_util));
    m.impl("get_tc_util", TORCH_BOX(&deep_gemm::config::get_tc_util));
    m.impl("set_pdl", TORCH_BOX(&deep_gemm::config::set_pdl));
    m.impl("get_pdl", TORCH_BOX(&deep_gemm::config::get_pdl));
    m.impl("use_deterministic_algorithms", TORCH_BOX(&deep_gemm::config::use_deterministic_algorithms));
    m.impl("set_ignore_compile_dims", TORCH_BOX(&deep_gemm::config::set_ignore_compile_dims));
    m.impl("set_block_size_multiple_of", TORCH_BOX(&deep_gemm::config::set_block_size_multiple_of));
}
