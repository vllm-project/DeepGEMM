# SM90 Mega Gate

`bf16_mega_gate` fuses the BF16 gate GEMM, `sqrtsoftplus` scoring and MoE top-k routing on
SM90 (including H20) and SM100 with the same interface:

```python
# x: contiguous BF16 [num_tokens, hidden]
# weight: contiguous BF16 [num_routed_experts, hidden]
topk_idx, topk_weights = deep_gemm.bf16_mega_gate(
    x, weight, num_topk=6, use_shared_as_routed=False, num_shared_experts=0,
    routed_scaling_factor=1.5, ep_rank=0, bias=bias,
)
```

The hidden size must be a positive multiple of 256, the routed expert count a multiple
of 4 up to 512, and the total selected expert count (including shared experts) at most
32. Outputs are INT64 indices and FP32 weights. Bias affects ranking only; weights use
the unbiased scores, normalized over the selected routed experts and multiplied by
`routed_scaling_factor`. During score-based routing, shared experts receive weight 1.
Masked tokens receive indices of -1 and weights of 0.

Both architectures support image bias, fixed/random routing, physical expert mapping,
strided rows in `unmapped_topk_idx`, and preallocated `out=(topk_idx, topk_weights)`.
Warm up the intended variants on the capture stream before using CUDA Graphs.
The SM90 implementation uses TMA/WGMMA and keeps logits in shared memory. For at least
32768 tokens and more than 128 routed experts, it splits the expert dimension into
128-expert tiles. Each CTA computes its tile's GEMM and local top-k, then a second
kernel selects the global top-k from those candidates. This is exact selection:
every global top-k expert must be in its tile's local top-k. Equal ranking scores
prefer the smaller expert index, and unbiased scores are normalized only after the
global selection.

The Split-N path uses three operand stages and overlaps consecutive WGMMA groups.
It allocates per-call candidate storage of
`num_tokens * ceil(num_routed_experts / 128) * num_topk * 12` bytes; the complete
logits matrix is never written to global memory. Candidate storage belongs to the
call's CUDA stream and is captured with CUDA Graphs, without cross-CTA spin barriers
or shared global workspace.

Outside deterministic mode, batches of up to 16 tokens use GEMV plus routing;
batches of 17–32767 tokens use cuBLASLt plus routing when the full device SM count
is available. This avoids the fused gate's overhead at smaller sizes. Large
fixed-routing batches, a restricted SM budget at smaller sizes, and deterministic
mode retain the original unsplit WGMMA kernel. Deterministic mode keeps the same
accumulation order across batch sizes. The 32768-row crossover is based on H20
measurements with hidden size 5120, 384 experts and top-6; other shapes and GPUs
may have different crossovers. See
[`test_sm90_mega_gate.py`](../tests/test_sm90_mega_gate.py) for correctness checks and
benchmarks. It reuses the API contract and reference helpers from the upstream
[`test_mega_gate.py`](../tests/test_mega_gate.py).

Run the SM90 tests and benchmarks from the repository root:

```sh
PYTHONPATH=. .venv/bin/python tests/test_sm90_mega_gate.py
```

For an alternating comparison against the original unsplit path, install
FlashInfer's CUPTI timing dependencies and run:

```sh
PYTHONPATH=. .venv/bin/python benchmarks/bench_sm90_mega_gate.py --output gate.json
```

The benchmark uses CUDA Graphs with cold L2 and checks correctness before timing.
Pass `--image-bias` for image routing, or `--vllm-reference` when vLLM is installed
to include cuBLASLt plus its DeepSeek-V4 top-k kernel (384 experts, top-6).
Sizes are actual gate input rows, not request context lengths or tokens summed
across data-parallel ranks. Kernel speedups do not establish full-model speedups.

SM90 routing helpers live in `epilogue/sm90_mega_gate.cuh` to keep the upstream
SM100 implementation unchanged. When upstream routing semantics change, compare
its `epilogue/sm100_mega_gate.cuh` helpers with the SM90 counterparts.
