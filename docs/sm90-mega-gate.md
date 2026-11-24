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
On SM90, batches of up to 16 tokens use GEMV plus routing by default. Larger
batches use cuBLASLt plus routing when all device SMs are available. H100/H200 have
no upper token threshold for this path. On H20 (78 SMs), batches of at least 32768
tokens with more than 128 experts retain Split-N unless fixed routing is requested.

Deterministic mode uses TMA/WGMMA. With a reduced `set_num_sms()` budget, batches
above 16 tokens also use this path. Each logit is accumulated over the complete
hidden dimension in a fixed order. Expert tiles
run on separate CTAs for deterministic routing with more than 128 experts, including
small batches. The same Split-N path handles large batches with a reduced SM budget.
Fixed routing uses an unsplit tile. Split-N selects each tile's local top-k, then
merges those candidates into the global top-k. Invalid candidate indices are clamped
before expert mapping, including rows whose logits are all NaN.

Custom kernels use grid-stride scheduling when SMs are restricted. Neighboring
Split-N tasks share the same input tile. Each call owns its candidate storage, so
concurrent streams and CUDA Graph replay do not share mutable workspace.

Launch choices live in `csrc/jit_kernels/heuristics/sm90_mega_gate.hpp` and can be
inspected with `DG_PRINT_CONFIGS=1`. SM90 scoring, tie-breaking, expert mapping
and output helpers stay in `epilogue/sm90_mega_gate.cuh`. The SM100 implementation
is unchanged to keep future upstream merges independent of the SM90 kernels.

Use an environment with PyTorch and the build dependencies installed. For a source
checkout, build the local extension before running tests from the repository root:

```sh
bash develop.sh
PYTHONPATH=. python tests/test_sm90_mega_gate.py
```

After installing a wheel with `install.sh`, run the tests against that installation
without prepending the source checkout to `PYTHONPATH`:

```sh
python tests/test_sm90_mega_gate.py
```

[`test_sm90_mega_gate.py`](../tests/test_sm90_mega_gate.py) covers the SM90 API,
boundaries, CUDA Graphs, fallback paths, independent FP32/FP64 references,
non-finite input regressions, SM budgets and bitwise batch invariance. It reuses
the existing input and reference utilities from `test_mega_gate.py` without
changing that file. Its benchmark times the complete call with CUDA events and
CUDA Graphs using warm L2.
