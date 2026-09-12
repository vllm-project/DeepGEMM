#include <deep_gemm/impls/sm120_fp4_mqa_logits.cuh>
using namespace deep_gemm;
static void f(){auto p=reinterpret_cast<void*>(&sm120_fp4_mqa_logits<
 32,128, false, 64,128, 2,4, 148, 128,256, float>);(void)p;}
