# NVFP4 MegaMoE with MXFP8 or BF16 shared experts

`deep_gemm.nvfp4_mega_moe` runs dispatch, routed gate/up GEMM, SwiGLU,
activation quantization, down GEMM, shared experts, and combine in one persistent
SM100/SM103 kernel. Its implementation and launcher are separate from the
MXFP8 and SiTU implementations:

- `deep_gemm/include/deep_gemm/impls/sm100_nvfp4_mega_moe.cuh`
- `csrc/jit_kernels/impls/sm100_nvfp4_mega_moe.hpp`

The host API, buffer layout, tuning heuristics, PTX helper, Python buffer/API,
and quantization utilities also live in dedicated NVFP4 files. The existing
MegaMoE API, heuristics, layout, Python module, and math/PTX helpers remain
unchanged from upstream. Integration only adds native registration and Python
exports, so upstream rebases do not have to merge NVFP4 branches into those files.

Routed GEMMs use native `tcgen05.mma.kind::mxf4nvf4.block_scale.block16`:
packed E2M1 operands and E4M3 block scales. Shared GEMMs use MXFP8 E4M3 operands
and UE8M0 scales per 32 elements by default. `shared_dtype=torch.bfloat16`
selects native BF16 shared GEMMs with BF16 input, weights, and intermediate
activations, without shared-expert FP8 quantization. Output is BF16. Shared
experts use their own input, so they do not inherit routed-input FP4 error.

## Preparation and API

`deep_gemm.nvfp4_mega_moe` calls its own native `_C.nvfp4_mega_moe` entry
point, with a fixed per-16 NVFP4 recipe and SwiGLU activation. It does not
route through `fp8_fp4_mega_moe`. The existing FP8/FP4 API retains its per-32
recipe and original arguments; NVFP4 global scales and BF16 shared weights
are specific to the new entry point.

Create `deep_gemm.NVFP4SymmBuffer(..., num_shared_experts=1)` and populate:

```python
buffer = deep_gemm.NVFP4SymmBuffer(
    group, num_experts, max_tokens_per_rank, topk, hidden, intermediate_hidden,
    num_shared_experts=1,
    shared_dtype=torch.bfloat16,  # torch.float8_e4m3fn selects MXFP8 (the default)
)
```

| View | Logical contents | Dtype and shape |
| --- | --- | --- |
| `x` | Routed input, two E2M1 values per byte, low nibble first | int8 `[capacity, H/2]` |
| `x_sf` | Four E4M3 block-scale bytes per int32 | int32 `[capacity, H/64]`, row major |
| `shared_l1_acts` | Independently quantized MXFP8 input | float8_e4m3fn `[capacity, H]` |
| `shared_l1_acts_sf` | Per-32 UE8M0 input scales | int32, MN major with padded UTCCP rows |
| `topk_idx` | Global expert indices; `-1` masks a route | int64 `[capacity, topk]` |
| `topk_weights` | Routing weights | float32 `[capacity, topk]` |

For BF16 shared experts, pass `shared_dtype=torch.bfloat16` to `NVFP4SymmBuffer`.
`shared_l1_acts` and `shared_l2_acts` then have dtype BF16 and logical shapes
`[capacity, H]` and `[capacity, I*num_shared_experts]`; their SF views are
`None`. Copy the original BF16 input into `shared_l1_acts`. Pass BF16 weight
tensors directly to `transform_weights_for_mega_moe` and `nvfp4_mega_moe`
(not `(weight, scale)` pairs). Do not quantize shared weights or intermediate
activations. MXFP8 remains the default and its API is unchanged.

Only the first `M` input and routing rows are consumed. Capacity is rounded up
by `NVFP4SymmBuffer`; `M` comes from the output tensor. Shared SF row packing depends
on `get_block_m_for_nvfp4_mega_moe(...)`. For logical row `r`, with `i = r % block_m`,
the physical row is
`(r // block_m) * align(block_m, 128) + (i // 128) * 128 + (i % 32) * 4 + (i % 128) // 32`.
Repack shared input SF if the selected block M changes.

Routed weights are pairs `(packed_values, packed_scales)` with logical shapes
`[local_experts, 2*I, H]` and `[local_experts, H, I]`. Values have half the
logical final dimension. Scales have final dimension `K/64` and MN-major
strides `(N*K/64, 1, N)`. Shared weights are the corresponding 2D MXFP8 pairs,
with shared intermediate width `I * num_shared_experts` and scale width `K/128`.
Call `transform_weights_for_mega_moe(w1, w2)` on each pair of weight tensors;
it interleaves gate/up and packs SF rows for UTCCP.

```python
routed_w1, routed_w2 = deep_gemm.transform_weights_for_mega_moe(w1, w2)
shared_w1, shared_w2 = deep_gemm.transform_weights_for_mega_moe(sw1, sw2)
deep_gemm.nvfp4_mega_moe(
    y, routed_w1, routed_w2, buffer,
    shared_l1_weights=shared_w1,
    shared_l2_weights=shared_w2,
    activation_clamp=7.0, activation_alpha=1.702, activation_beta=1.0,
    l1_alpha=l1_global_products, l2_alpha=l2_global_products,
    l2_activation_scale=mid_global_scale,
)
```

NVFP4 dequantization is `E2M1_value * E4M3_block_scale * global_scale`.
The optional `l1_alpha` and `l2_alpha` are contiguous CUDA float32 vectors,
one entry per local expert, containing activation-global × weight-global
dequantization scales for each GEMM. They multiply FP32 accumulators before
BF16 rounding. `l2_activation_scale` controls the fused intermediate quantizer
and must also be included in `l2_alpha`. All default to one. Input global
scaling must be consistent across ranks when using per-expert L1 products.

`per_token_cast_to_nvfp4` and `cast_back_from_nvfp4` are PyTorch reference
helpers in `deep_gemm.utils.nvfp4` for preparation and testing.
The block quantizer rounds E4M3 scales
to nearest, clamps them to `[2^-9, 448]`, and rounds/saturates E2M1 values.
Global scales should be calibrated to avoid block-scale underflow/saturation.

SwiGLU is `gate * sigmoid(alpha * gate) * (up + beta)`, with the existing
MegaMoE BF16 rounding and clamp semantics. Routing weights multiply its output
before the routed intermediate is quantized. Shared output enters combine with
unit weight. This is a layer kernel; it does not compute routing scores or
load model checkpoints. NVFP4 SiTU is not implemented.

## Verification and timing

`tests/test_nvfp4_mega_moe.py` provides correctness checks and optional CUDA
event timing. Generated measurements and benchmark artifacts are not checked
into the repository. Multi-node runs require a common NVLink fabric domain
and an extension built for the allocated machines.

The benchmark defaults to MiniMax M3's H=6144, I=3072, 128 routed experts,
top-4, one shared expert, SwiGLU alpha=1.702, beta=1, clamp=7, and routing
weights normalized to sum to 2. Dimensions, alpha, clamp, and routing factor
follow the [published model configuration](https://huggingface.co/MiniMaxAI/MiniMax-M3-MXFP8/blob/main/config.json);
the benchmark explicitly sets beta to 1.
Inputs and weights are synthetic, seeded BF16 tensors quantized to the requested
formats. These are layer microbenchmarks, not end-to-end model evaluations.

```bash
PYTHONPATH=. python tests/test_nvfp4_mega_moe.py --benchmark \
  --output benchmarks/minimax_m3_nvfp4_ep1.json
PYTHONPATH=. python tests/test_nvfp4_mega_moe.py --num-processes 4 --benchmark \
  --output benchmarks/minimax_m3_nvfp4_ep4.json
```

The default sweep is M=1,2,4,...,8192 **per rank**. The timed interval contains
one complete fused kernel, including dispatch and combine. Weight preparation,
input quantization/copies, buffer allocation, and routing are excluded. CUDA
graphs remove host launch gaps; external CUDA events bracket each invocation.
A 256 MiB cache flush precedes each measurement outside its event interval.
Three measured graph replays of ten invocations produce 30 samples per point.
`--warm-cache` disables the flush. JSON retains all samples, reference errors,
per-rank results, source hashes, and software versions.

Correctness uses decoded FP4/FP8 operands and independent FP32 GEMMs, reproducing
BF16 boundaries, quantization, activation, routing, and shared output addition.
It checks expert counts, finite outputs, repeated launches, and relative L2 error
below 0.005. `--masked`, `--shared-experts`, and global-scale options exercise
additional cases. `--skip-check` is for repeated performance tuning only.

`DG_NVFP4_MOE_BLOCK_M` and `DG_NVFP4_MOE_BLOCK_K` override tile selection for
tuning. Supported M tiles are 16, 32, 64, 128, 192, 240; K is 128 or 256.
They affect NVFP4 only, and the M override is reflected by the shared-SF
layout query. `--num-sms` controls the existing DeepGEMM SM limit.

The BF16 shared specialization uses half the routed K tile per pipeline stage,
so BF16 storage does not reduce the number of routed pipeline stages. Shared
GEMMs still cover the complete K dimension. For A/B experiments,
`DG_NVFP4_MOE_SHARED_FULL_K=1` restores the larger shared K tile and storage
footprint; it does not change arithmetic precision or the layer operation.

The test harness supports `--shared-dtype bf16`, `--routing-scale`, and
multi-node launches using the repository's `init_dist` convention: `WORLD_SIZE`
is the number of nodes, `RANK` is the node index, and `--num-processes` is the
number of local GPU workers. JSON records the resulting global world size,
rank/host/GPU identities, source hashes, and Slurm job ID.
