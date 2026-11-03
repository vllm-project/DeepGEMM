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
The SM90 fused kernel uses TMA/WGMMA and keeps logits in shared memory without global
scratch or cross-CTA barriers. Batches of up to 16 tokens use a separate GEMV and routing
kernel to expose more parallelism. Deterministic mode always uses the fused WGMMA path
with the same accumulation order across batch sizes. See
[`test_sm90_mega_gate.py`](../tests/test_sm90_mega_gate.py) for correctness checks and
benchmarks. It reuses the API contract and reference helpers from the upstream
[`test_mega_gate.py`](../tests/test_mega_gate.py).

Run the SM90 tests and benchmarks from the repository root:

```sh
PYTHONPATH=. python tests/test_sm90_mega_gate.py
```

SM90 routing helpers live in `epilogue/sm90_mega_gate.cuh` to keep the upstream
SM100 implementation unchanged. When upstream routing semantics change, compare
its `epilogue/sm100_mega_gate.cuh` helpers with the SM90 counterparts.
