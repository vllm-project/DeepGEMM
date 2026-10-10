#pragma once

#include <optional>
#include <string>

#include <deep_gemm/common/types.cuh>

#include "epilogue_class.hpp"

namespace deep_gemm {

// SM120 kernels only instantiate the identity `epilogue::operators` implementation;
// custom epilogue classes are rejected by the APIs before reaching them
static std::string get_default_epilogue_type(const std::optional<std::string>& epilogue_type) {
    return epilogue_type.value_or(IdentityEpilogue().get_epilogue_operator_type());
}

// The operator type to instantiate the kernel with, and the operator's runtime state to launch it with
struct EpilogueInput {
    std::string type = IdentityEpilogue().get_epilogue_operator_type();
    EpilogueOperatorArgs args;
};

static EpilogueInput make_epilogue_input(const int& m, const int& n,
                                         const std::optional<std::string>& epilogue_type = std::nullopt) {
    return {get_default_epilogue_type(epilogue_type), {}};
}

} // namespace deep_gemm
