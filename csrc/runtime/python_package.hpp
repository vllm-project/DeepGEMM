#pragma once

#include <string>

namespace deep_gemm::python_package {

inline std::string package_name = "deep_gemm";

inline void set_package_name(const std::string& name) {
    package_name = name;
}

inline std::string module_name(const char* relative_name) {
    return package_name + "." + relative_name;
}

}  // namespace deep_gemm::python_package
