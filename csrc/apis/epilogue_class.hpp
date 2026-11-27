#pragma once

#include <cstdint>
#include <memory>
#include <optional>

#include <torch/csrc/stable/ops.h>

#include "../jit_kernels/impls/epilogue_class.hpp"

namespace deep_gemm::epilogue_class {

// The user-facing epilogue classes (`deep_gemm.epilogue.X` in `deep_gemm/_C.py`) cross the TORCH_LIBRARY
// boundary as `(kind, alpha, sfd, aux)`; `alpha` also carries the FP8 clamp value, `aux` the clamped count
// NOTES: `alpha` and the FP8 `(d, sfd)` output pair are the shorthand for `Alpha` and unclamped `FP8CastWithClamp`
enum Kind : int64_t {
    None = 0,
    Identity = 1,
    Alpha = 2,
    FP8CastWithClamp = 3,
    BF16StochasticRounding = 4,
    FP8StochasticRounding = 5,
};

static std::shared_ptr<EpilogueClass> unwrap(
    const int64_t& kind,
    const std::optional<double>& alpha,
    const std::optional<torch::stable::Tensor>& sfd,
    const std::optional<torch::stable::Tensor>& aux) {
    DG_HOST_ASSERT(kind == FP8CastWithClamp or not aux.has_value());
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
        case FP8CastWithClamp:
            DG_HOST_ASSERT(sfd.has_value());
            return std::make_shared<FP8CastWithClampEpilogue>(
                *sfd, alpha.has_value() ? std::optional<float>(static_cast<float>(*alpha)) : std::nullopt, aux);
        case BF16StochasticRounding:
            DG_HOST_ASSERT(not alpha.has_value() and not sfd.has_value());
            return std::make_shared<BF16StochasticRoundingEpilogue>();
        case FP8StochasticRounding:
            DG_HOST_ASSERT(not alpha.has_value() and sfd.has_value());
            return std::make_shared<FP8StochasticRoundingEpilogue>(*sfd);
        default:
            DG_HOST_UNREACHABLE("Unknown epilogue kind");
    }
}

} // namespace deep_gemm::epilogue_class
