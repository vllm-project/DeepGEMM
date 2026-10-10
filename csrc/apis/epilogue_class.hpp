#pragma once

#include <memory>
#include <optional>

#include <torch/custom_class.h>
#include <torch/library.h>

#include "../jit_kernels/impls/epilogue_class.hpp"

namespace deep_gemm::epilogue_class {

// TORCH_LIBRARY carrier of the epilogue classes: `deep_gemm.epilogue.X(...)` builds one through the
// factory ops below, and the GEMM APIs take it as their optional `epilogue` argument
// NOTES: `alpha` and the FP8 `(d, sfd)` output pair are the shorthand for `Alpha` and `FP8Quantization`
struct Epilogue: torch::CustomClassHolder {
    std::shared_ptr<EpilogueClass> value;

    explicit Epilogue(std::shared_ptr<EpilogueClass> value): value(std::move(value)) {}
};

using EpilogueArg = std::optional<c10::intrusive_ptr<Epilogue>>;

static std::shared_ptr<EpilogueClass> unwrap(const EpilogueArg& epilogue) {
    return epilogue.has_value() ? epilogue.value()->value : nullptr;
}

template <typename epilogue_class_t, typename... Args>
static c10::intrusive_ptr<Epilogue> make(const Args&... args) {
    return c10::make_intrusive<Epilogue>(std::make_shared<epilogue_class_t>(args...));
}

} // namespace deep_gemm::epilogue_class

// NOTES: the class must be registered before any schema refers to it, so every API header
//        taking an epilogue includes this one first
TORCH_LIBRARY_FRAGMENT(deep_gemm, m) {
    using namespace deep_gemm;
    using namespace deep_gemm::epilogue_class;

    m.class_<Epilogue>("Epilogue");
    m.def("epilogue_identity() -> __torch__.torch.classes.deep_gemm.Epilogue", []() {
        return make<IdentityEpilogue>();
    });
    m.def("epilogue_alpha(float alpha) -> __torch__.torch.classes.deep_gemm.Epilogue", [](const double& alpha) {
        return make<AlphaEpilogue>(static_cast<float>(alpha));
    });
    m.def("epilogue_fp8_quantization(Tensor sfd) -> __torch__.torch.classes.deep_gemm.Epilogue", [](const torch::Tensor& sfd) {
        return make<FP8QuantizationEpilogue>(sfd);
    });
    m.def("epilogue_bf16_stochastic_rounding() -> __torch__.torch.classes.deep_gemm.Epilogue", []() {
        return make<BF16StochasticRoundingEpilogue>();
    });
}
