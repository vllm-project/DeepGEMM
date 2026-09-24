#pragma once

#include <torch/library.h>

#include "../jit_kernels/heuristics/runtime.hpp"
#include "../runtime/runtime.hpp"

TORCH_LIBRARY_FRAGMENT(deep_gemm, m) {
    using namespace deep_gemm;
    m.def("init(str library_root_path) -> ()", [&](const std::string& library_root_path) {
        init_jit(library_root_path);
    });
    m.def("set_num_sms(int new_num_sms) -> ()", [&](const int64_t& new_num_sms) {
        runtime->set_num_sms(new_num_sms);
    });
    m.def("get_num_sms() -> int", [&]() {
       return int64_t(runtime->get_num_sms());
    });
    m.def("set_tc_util(int new_tc_util) -> ()", [&](const int64_t& new_tc_util) {
        runtime->set_tc_util(new_tc_util);
    });
    m.def("get_tc_util() -> int", [&]() {
        return int64_t(runtime->get_tc_util());
    });
    m.def("set_pdl(bool new_enable_pdl) -> ()", [](const bool& new_enable_pdl) {
        jit->default_launch_options.enable_pdl = new_enable_pdl;
    });
    m.def("get_pdl() -> bool", []() {
        return *jit->default_launch_options.enable_pdl;
    });
    m.def("use_deterministic_algorithms(bool enabled) -> ()", [&](const bool enabled) {
        heuristics_runtime->use_deterministic_algorithms(enabled);
    });
    m.def("set_ignore_compile_dims(bool new_value) -> ()", [&](const bool& new_value) {
        heuristics_runtime->set_ignore_compile_dims(new_value);
    });
    m.def("set_block_size_multiple_of(int[] new_value) -> ()", [&](const std::vector<int64_t>& new_value) {
        DG_HOST_ASSERT(new_value.size() == 1 or new_value.size() == 2);
        heuristics_runtime->set_block_size_multiple_of(new_value[0], new_value.size() == 1 ? new_value[0] : new_value[1]);
    });
}
