#include <deep_gemm/impls/sm120_bmk_bnk_mn.cuh>
using namespace deep_gemm;
static void f(){auto p=reinterpret_cast<void*>(&sm120_bmn_bnk_mn_gemm_impl<
 0,128,128, 128,64,64, 1, 128, 3, 128,256>);(void)p;}
