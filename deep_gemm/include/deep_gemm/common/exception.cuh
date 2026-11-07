#pragma once

#include <cuda/std/cstdint>
#include <deep_gemm/common/compile.cuh>

#ifdef __CLION_IDE__

CUTLASS_HOST_DEVICE void host_device_printf(const char* format, ...) {
#ifdef _MSC_VER
    __debugbreak();
#else
    asm volatile("trap;");
#endif
}

#define printf host_device_printf
#endif

#if defined(__CUDA_ARCH__) || defined(__CUDACC_RTC__)
#define DG_TRAP() __trap()
#elif defined(_MSC_VER)
#define DG_TRAP() __debugbreak()
#else
#define DG_TRAP() asm("trap;")
#endif

#ifndef DG_DEVICE_ASSERT
#define DG_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) { \
        printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond); \
        DG_TRAP(); \
    } \
} while (0)
#endif

#ifndef DG_TRAP_ONLY_DEVICE_ASSERT
#define DG_TRAP_ONLY_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) \
        DG_TRAP(); \
} while (0)
#endif

#ifndef DG_STATIC_ASSERT
#define DG_STATIC_ASSERT(cond, ...) static_assert(cond, __VA_ARGS__)
#endif

#ifndef DG_UNIFIED_ASSERT
#ifdef DG_IN_CUDA_COMPILATION
#define DG_UNIFIED_ASSERT(cond) DG_DEVICE_ASSERT(cond)
#else
#define DG_UNIFIED_ASSERT(cond) DG_HOST_ASSERT(cond)
#endif
#endif
