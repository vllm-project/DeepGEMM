#include <deep_gemm/impls/sm120_fp8_mqa_logits.cuh>
using namespace deep_gemm;
// Reachable tuple, derived from the host launcher (not chosen freely):
//   kNumHeads=16          -> csrc/apis/attention.hpp arch-12 arm allows {16,32,64}
//   kHeadDim=128          -> sm120_mqa_logits.hpp asserts {32,64,128} for FP8
//   kIsCompressedLogits=1 -> `max_seqlen_k > 0`; every arch-12 test row is compressed
//   BLOCK_Q=8             -> `128 / num_heads` = 128/16 (csrc/apis/attention.hpp:148)
//   BLOCK_KV=128          -> `sm120::kMqaBlockKv` (csrc/apis/sm120_dispatch.hpp:33)
//   kNumQStages=2, kNumKVStages=3 -> csrc/jit_kernels/impls/sm120_mqa_logits.hpp:89
//   kNumTMAThreads=128, kNumMathThreads=256 -> same, lines 88 and 90
//   logits_dtype=float    -> one of the two arch-12 test dtypes
// kNumSMs=148 is this host's `runtime->get_num_sms()`; the real sm120 value is unknown here.
static void f(){auto p=reinterpret_cast<void*>(&sm120_fp8_mqa_logits<
 16,128, true, 8,128, 2,3, 148, 128,256, float>);(void)p;}
