#include <deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh>
using namespace deep_gemm;
static void f(){auto p=reinterpret_cast<void*>(&sm120_fp8_fp4_gemm_1d1d_impl<
 0,4096,7168, 32,32, 1, 128,128,128, 128,128, 128, 3, 128,256, 148,
 GemmType::Normal,false, cutlass::bfloat16_t, epilogue::transform::EpilogueIdentity,
 true,false,false, true,false, 128, 1>);(void)p;}
