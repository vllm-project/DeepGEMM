#include <deep_gemm/impls/sm120_tf32_hc_prenorm_gemm.cuh>
using namespace deep_gemm;
static void f(){auto p=reinterpret_cast<void*>(&sm120_tf32_hc_prenorm_gemm_impl<
 128,128, 128,64,64, 1, 3, 256,128>);(void)p;}
