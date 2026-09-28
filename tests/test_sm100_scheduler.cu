// nvcc -std=c++17 -arch=sm_100f -I<deep_gemm>/include \
//   -I/usr/local/cuda/include/cccl tests/test_sm100_scheduler.cu -o /tmp/test_scheduler
#include <cuda_runtime.h>
#include <deep_gemm/scheduler/gemm.cuh>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(cudaError_t error) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
        std::exit(1);
    }
}

template <unsigned Multicast, bool MulticastOnA>
__global__ void visit_tiles(int* layout, int* visited, int* counts, int* errors) {
    deep_gemm::sched::Scheduler<deep_gemm::GemmType::MGroupedContiguous,
                               128, 128, 4, Multicast, MulticastOnA, 8> scheduler(8192, 512, 128, layout);
    unsigned m, n;
    int count = 0;
    while (scheduler.template get_next_block<true>(m, n)) {
        if (scheduler.current_iter != count)
            atomicAdd(errors, 1);
        atomicAdd(visited + m * 4 + n, 1);
        ++ count;
    }
    if (scheduler.current_iter != count)
        atomicAdd(errors, 1);
    counts[blockIdx.x] = count;
}

template <unsigned Multicast, bool MulticastOnA>
static void test() {
    int *layout, *visited, *counts, *errors;
    check(cudaMalloc(&layout, 8192 * sizeof(int)));
    check(cudaMalloc(&visited, 256 * sizeof(int)));
    check(cudaMalloc(&counts, 8 * sizeof(int)));
    check(cudaMalloc(&errors, sizeof(int)));
    for (int pattern = 0; pattern < 7; ++ pattern) {
        std::vector<int> ids(8192, -1), actual(256), cta_counts(8);
        for (int m = 0; m < 64; ++ m) {
            const bool valid = pattern == 1 or
                               (pattern == 2 and m >= 32) or
                               (pattern == 3 and m < 32) or
                               (pattern == 4 and m % 3 == 1) or
                               (pattern == 5 and m == 16) or
                               (pattern == 6 and m == 17);
            if (valid)
                for (int row = m * 128; row < (m + 1) * 128; ++ row)
                    ids[row] = m % 4;
        }
        check(cudaMemcpy(layout, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice));
        check(cudaMemset(visited, 0, 256 * sizeof(int)));
        check(cudaMemset(errors, 0, sizeof(int)));
        visit_tiles<Multicast, MulticastOnA><<<8, 1>>>(layout, visited, counts, errors);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        check(cudaMemcpy(actual.data(), visited, 256 * sizeof(int), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(cta_counts.data(), counts, 8 * sizeof(int), cudaMemcpyDeviceToHost));
        int phase_errors;
        check(cudaMemcpy(&phase_errors, errors, sizeof(int), cudaMemcpyDeviceToHost));
        if (phase_errors != 0)
            std::abort();
        for (int m = 0; m < 64; ++ m) {
            bool active = ids[m * 128] >= 0;
            if constexpr (Multicast == 2 and not MulticastOnA)
                active |= ids[(m ^ 1) * 128] >= 0;
            for (int n = 0; n < 4; ++ n)
                if (actual[m * 4 + n] != static_cast<int>(active))
                    std::abort();
        }
        if constexpr (Multicast == 2)
            for (int cta = 0; cta < 8; cta += 2)
                if (cta_counts[cta] != cta_counts[cta + 1])
                    std::abort();
    }
    check(cudaFree(layout)); check(cudaFree(visited));
    check(cudaFree(counts)); check(cudaFree(errors));
}

int main() {
    test<1, true>(); test<1, false>();
    test<2, true>(); test<2, false>();
    std::puts("SM100 scheduler: 28 coverage, peer-agreement and phase cases passed");
}
