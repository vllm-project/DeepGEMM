#pragma once

#include <cstdint>
#include <memory>
#include <optional>

#include <torch/csrc/stable/ops.h>

#include "../jit_kernels/impls/epilogue_class.hpp"

namespace deep_gemm::epilogue_class {

enum Kind : int64_t {
    None = 0,
    Identity = 1,
    Alpha = 2,
    FP8Quantization = 3,
    BF16StochasticRounding = 4,
};

static std::shared_ptr<EpilogueClass> unwrap(
    const int64_t& kind,
    const std::optional<double>& alpha,
    const std::optional<torch::stable::Tensor>& sfd) {
    switch (kind) {
        case None:
            DG_HOST_ASSERT(not alpha.has_value() and not sfd.has_value());
            return nullptr;
        case Identity:
            DG_HOST_ASSERT(not alpha.has_value() and not sfd.has_value());
            return std::make_shared<IdentityEpilogue>();
        case Alpha:
            DG_HOST_ASSERT(alpha.has_value() and not sfd.has_value());
            return std::make_shared<AlphaEpilogue>(static_cast<float>(*alpha));
        case FP8Quantization:
            DG_HOST_ASSERT(not alpha.has_value() and sfd.has_value());
            return std::make_shared<FP8QuantizationEpilogue>(*sfd);
        case BF16StochasticRounding:
            DG_HOST_ASSERT(not alpha.has_value() and not sfd.has_value());
            return std::make_shared<BF16StochasticRoundingEpilogue>();
        default:
            DG_HOST_UNREACHABLE("Unknown epilogue kind");
    }
}

} // namespace deep_gemm::epilogue_class
