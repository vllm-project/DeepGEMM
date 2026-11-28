#pragma once

#include <format>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include <map>

#include <torch/csrc/stable/ops.h>
#include "../../utils/torch_compat.hpp"

#include <deep_gemm/common/types.cuh>

#include "../../runtime/jit.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/layout.hpp"

namespace deep_gemm {

// The GEMM epilogue type: selects the device-side `epilogue::operators` implementation,
// carries its runtime state, and validates the GEMM output against it
// NOTES: every SM100 GEMM launches with exactly one epilogue class; they do not compose
class EpilogueClass {
public:
    virtual ~EpilogueClass() = default;

    virtual std::string get_epilogue_operator_type() const = 0;

    virtual EpilogueOperatorArgs make_epilogue_operator_args(const int& m, const int& n) const { return {}; }

    virtual deep_jit::cuda::CompilerOptions compiler_options() const { return {}; }

    virtual void check(const std::optional<torch::stable::Tensor>& c, const torch::stable::Tensor& d) const {
        DG_HOST_ASSERT(d.scalar_type() == torch::headeronly::ScalarType::BFloat16 or
                       d.scalar_type() == torch::headeronly::ScalarType::Float);
    }

    // The BLAS linear-combination coefficient, when this epilogue is a plain store cuBLASLt
    // can compute directly; `nullopt` when cuBLASLt cannot handle it
    virtual std::optional<float> get_alpha() const { return std::nullopt; }

    // The SFD this epilogue quantizes into, if any (an empty problem zeroes it)
    virtual std::optional<torch::stable::Tensor> output_sfd() const { return std::nullopt; }
};

class IdentityEpilogue final: public EpilogueClass {
public:
    std::string get_epilogue_operator_type() const override {
        return "epilogue::operators::Identity";
    }

    std::optional<float> get_alpha() const override { return 1.0f; }
};

class AlphaEpilogue final: public EpilogueClass {
    float alpha;

public:
    explicit AlphaEpilogue(const float& alpha): alpha(alpha) {}

    std::string get_epilogue_operator_type() const override {
        return "epilogue::operators::ScaleByAlpha";
    }

    EpilogueOperatorArgs make_epilogue_operator_args(const int&, const int&) const override {
        return {.alpha = alpha};
    }

    std::optional<float> get_alpha() const override { return alpha; }
};

class FP8CastWithClampEpilogue final: public EpilogueClass {
    torch::stable::Tensor sfd;
    std::optional<float> clamp_value;
    std::optional<torch::stable::Tensor> clamped_count;
    bool is_sfd_128x4;

    static uint32_t* get_clamp_count_ready(const torch::stable::Tensor& reference) {
        const auto stream = torch_compat::stream_key(reference);
        static std::map<torch_compat::StreamKey, torch::stable::Tensor> ready_by_stream;
        auto& ready = ready_by_stream[stream];
        if (not ready.defined()) {
            // Initialize each stream once, before graph capture.
            DG_HOST_ASSERT(not torch_compat::is_capturing(stream.second));
            ready = torch::stable::new_zeros(reference, {2}, torch::headeronly::ScalarType::UInt32);
        }
        return static_cast<uint32_t*>(ready.mutable_data_ptr());
    }

public:
    // `is_sfd_128x4` selects the `(ceil_div(m, 128), num_words, 128)` tiled SFD over the MN-major one
    explicit FP8CastWithClampEpilogue(const torch::stable::Tensor& sfd,
                                      const std::optional<float>& clamp_value = std::nullopt,
                                      const std::optional<torch::stable::Tensor>& clamped_count = std::nullopt,
                                      const bool& is_sfd_128x4 = false):
        sfd(sfd), clamp_value(clamp_value), clamped_count(clamped_count), is_sfd_128x4(is_sfd_128x4) {
        DG_HOST_ASSERT(not clamp_value.has_value() or clamp_value.value() > 0.0f);
        if (clamped_count.has_value())
            DG_HOST_ASSERT(clamped_count->scalar_type() == torch::headeronly::ScalarType::UInt32 and clamped_count->numel() == 1);
    }

    std::string get_epilogue_operator_type() const override {
        return "epilogue::operators::CastToFP8WithClamp";
    }

    EpilogueOperatorArgs make_epilogue_operator_args(const int& m, const int& n) const override {
        return {.sfd = static_cast<uint32_t*>(sfd.mutable_data_ptr()),
                .sfd_stride = static_cast<uint32_t>(is_sfd_128x4 ? sfd.stride(0) : sfd.stride(-1)),
                .shape_m = static_cast<uint32_t>(m), .shape_n = static_cast<uint32_t>(n),
                .is_sfd_128x4 = is_sfd_128x4,
                .clamp_value = clamp_value.value_or(0.0f),
                .clamped_count = clamped_count.has_value() ? static_cast<uint32_t*>(clamped_count->mutable_data_ptr()) : nullptr,
                .clamp_count_ready = clamped_count.has_value() ? get_clamp_count_ready(sfd) : nullptr};
    }

    void check(const std::optional<torch::stable::Tensor>& c, const torch::stable::Tensor& d) const override {
        DG_HOST_ASSERT(not c.has_value() and d.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn and
                       "FP8 quantization requires a direct E4M3 output");
        DG_HOST_ASSERT(sfd.device() == d.device());
        if (clamped_count.has_value())
            DG_HOST_ASSERT(clamped_count->device() == d.device());
        // The 128x4 tiled SFD is validated by its producer (`fp8_einsum`)
        if (not is_sfd_128x4) {
            const auto [num_groups, m, n] = get_shape<3>(d);
            DG_HOST_ASSERT(n % 128 == 0);
            check_sf_layout(sfd, m, num_groups * n, 1, 32, std::nullopt, true, false, torch::headeronly::ScalarType::Int);
        }
    }

    std::optional<torch::stable::Tensor> output_sfd() const override { return sfd; }
};

class BiasEpilogue final: public EpilogueClass {
    torch::stable::Tensor bias;

public:
    explicit BiasEpilogue(const torch::stable::Tensor& bias): bias(bias) {}

    std::string get_epilogue_operator_type() const override {
        return std::format("epilogue::operators::AddBias<{}>",
                           bias.scalar_type() == torch::headeronly::ScalarType::Float ? "float" : "cutlass::bfloat16_t");
    }

    EpilogueOperatorArgs make_epilogue_operator_args(const int&, const int& n) const override {
        return {.shape_n = static_cast<uint32_t>(n), .bias = bias.const_data_ptr()};
    }

    void check(const std::optional<torch::stable::Tensor>& c, const torch::stable::Tensor& d) const override {
        EpilogueClass::check(c, d);
        DG_HOST_ASSERT(jit->device.get_arch_major() == 10 and "Bias GEMM requires SM100");
        DG_HOST_ASSERT(d.dim() == 2 and bias.dim() == 2 and bias.size(0) == 1 and bias.size(1) == d.size(1));
        DG_HOST_ASSERT(bias.device() == d.device() and bias.scalar_type() == d.scalar_type());
        DG_HOST_ASSERT(bias.is_contiguous() and "Bias must be contiguous");
    }
};

class BF16StochasticRoundingEpilogue final: public EpilogueClass {
public:
    std::string get_epilogue_operator_type() const override {
        return "epilogue::operators::StochasticRoundToBF16";
    }

    // `cvt.rs` requires an architecture-specific target (e.g., `sm_100a`), not a family one
    deep_jit::cuda::CompilerOptions compiler_options() const override {
        return {.arch = jit->device.get_arch(false)};
    }

    void check(const std::optional<torch::stable::Tensor>& c, const torch::stable::Tensor& d) const override {
        DG_HOST_ASSERT(not c.has_value() and d.scalar_type() == torch::headeronly::ScalarType::BFloat16 and
                       "Stochastic rounding requires a direct BF16 output");
    }
};

class FP8StochasticRoundingEpilogue final: public EpilogueClass {
    torch::stable::Tensor sfd;

public:
    explicit FP8StochasticRoundingEpilogue(const torch::stable::Tensor& sfd): sfd(sfd) {}

    std::string get_epilogue_operator_type() const override {
        return "epilogue::operators::StochasticRoundToFP8";
    }

    EpilogueOperatorArgs make_epilogue_operator_args(const int& m, const int& n) const override {
        return {.sfd = static_cast<uint32_t*>(sfd.mutable_data_ptr()),
                .sfd_stride = static_cast<uint32_t>(sfd.stride(-1)),
                .shape_m = static_cast<uint32_t>(m), .shape_n = static_cast<uint32_t>(n)};
    }

    // `cvt.rs` requires an architecture-specific target (e.g., `sm_100a`), not a family one
    deep_jit::cuda::CompilerOptions compiler_options() const override {
        return {.arch = jit->device.get_arch(false)};
    }

    void check(const std::optional<torch::stable::Tensor>& c, const torch::stable::Tensor& d) const override {
        DG_HOST_ASSERT(not c.has_value() and d.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn and
                       "FP8 stochastic rounding requires a direct E4M3 output");
        const auto [num_groups, m, n] = get_shape<3>(d);
        DG_HOST_ASSERT(n % 128 == 0 and sfd.device() == d.device());
        check_sf_layout(sfd, m, num_groups * n, 1, 32, std::nullopt, true, false, torch::headeronly::ScalarType::Int);
    }

    std::optional<torch::stable::Tensor> output_sfd() const override { return sfd; }
};

// Resolve the epilogue-related API inputs into the single epilogue class of an SM100 GEMM,
// validated against the GEMM output: the user class, the standalone `alpha` (an API argument
// for cuBLASLt signature parity), bias, or the FP8 `(d, sfd)` output pair; all exclusive
// NOTES: a `[1, N]` C distinct from D is a bias on SM100 only; other architectures accumulate it as a normal C
static std::shared_ptr<EpilogueClass> resolve_epilogue_class(const std::shared_ptr<EpilogueClass>& epilogue_class,
                                                             const std::optional<torch::stable::Tensor>& c,
                                                             const torch::stable::Tensor& d,
                                                             const std::optional<float>& alpha = std::nullopt,
                                                             const std::optional<torch::stable::Tensor>& sfd = std::nullopt) {
    const bool with_bias = jit->device.get_arch_major() == 10 and
                           c.has_value() and d.dim() == 2 and c->dim() == 2 and
                           c->size(0) == 1 and c->const_data_ptr() != d.const_data_ptr();
    DG_HOST_ASSERT((epilogue_class != nullptr) + alpha.has_value() + sfd.has_value() + with_bias <= 1 and
                   "The epilogue class, `alpha`, the FP8 output pair and bias are exclusive");

    std::shared_ptr<EpilogueClass> resolved = epilogue_class;
    if (with_bias)
        resolved = std::make_shared<BiasEpilogue>(c.value());
    else if (alpha.has_value())
        resolved = std::make_shared<AlphaEpilogue>(alpha.value());
    else if (sfd.has_value())
        resolved = std::make_shared<FP8CastWithClampEpilogue>(sfd.value());
    else if (resolved == nullptr)
        resolved = std::make_shared<IdentityEpilogue>();
    resolved->check(c, d);
    return resolved;
}

} // namespace deep_gemm
