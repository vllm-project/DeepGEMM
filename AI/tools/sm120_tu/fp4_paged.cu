#include <deep_gemm/impls/sm120_fp4_paged_mqa_logits.cuh>
using namespace deep_gemm;
static void f(){auto p=reinterpret_cast<void*>(&sm120_fp4_paged_mqa_logits<
 2,32, 128,64, false,false, 2,4, 128, 128,256, float>);(void)p;}
