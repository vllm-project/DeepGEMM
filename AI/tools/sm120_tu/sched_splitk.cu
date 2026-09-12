// Scheduler must accept a trailing kSplitKFactor and expose split-K state.
#include <deep_gemm/scheduler/gemm.cuh>
using namespace deep_gemm;
static void f(){
    using S = sched::Scheduler<GemmType::Normal, 128, 128, 1, 1, false, 148,
                               true, 128u, 128u,
                               sched::get_num_1d_blocks_per_group<GemmType::Normal,128,128,148,false>(),
                               /*kSplitKFactor=*/4>;
    static_assert(sizeof(S) > 0);
}
