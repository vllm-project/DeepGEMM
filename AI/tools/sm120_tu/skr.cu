#include <deep_gemm/impls/sm120_split_k_reduce.cuh>
using namespace deep_gemm;
static void f(){auto p=reinterpret_cast<void*>(&sm120_split_k_reduce_impl<cutlass::bfloat16_t,2>);(void)p;}
