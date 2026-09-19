#pragma once

// DeepGEMM enters DeepJIT through TORCH_LIBRARY kernels. The dispatcher releases
// Python's GIL before invoking these kernels, so DeepJIT must not manage it again.
// Keep this adapter ahead of DeepJIT's pybind-based helper on the include path.
namespace deep_jit {
class GilScopedRelease {
public:
    GilScopedRelease() = default;
    GilScopedRelease(const GilScopedRelease&) = delete;
    GilScopedRelease& operator=(const GilScopedRelease&) = delete;
    GilScopedRelease(GilScopedRelease&&) = delete;
    GilScopedRelease& operator=(GilScopedRelease&&) = delete;
};
} // namespace deep_jit
