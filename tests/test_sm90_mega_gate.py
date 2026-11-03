import torch
import torch.nn.functional as F

import deep_gemm
from deep_gemm.testing import (
    assert_bitwise_equal,
    bench_kineto,
    count_bytes,
    get_arch_major,
    test_filter,
)
from test_mega_gate import (
    NUM_TOPK,
    check_case,
    check_correctness,
    check_topk_weights,
    enumerate_cases,
    make_inputs,
    make_outputs,
    run_mega_gate,
    run_reference_gemm,
    test_mega_gate_api_contract as _test_mega_gate_api_contract,
)


@test_filter(lambda: get_arch_major() == 9)
@torch.no_grad()
def test_sm90_mega_gate() -> None:
    torch.manual_seed(0)

    print('Testing BF16 GEMM + MoE top-k gate:')
    for num_tokens, hidden, num_routed_experts in enumerate_cases():
        inputs = make_inputs(num_tokens, hidden, num_routed_experts)
        outputs = make_outputs(inputs)
        weight_diff = check_correctness(inputs, outputs)

        kernel_names = (('sm90_bf16_mega_gate_gemv', 'sm90_bf16_mega_gate_route')
                        if num_tokens <= 16 else 'mega_gate')
        fused_t = bench_kineto(
            lambda: run_mega_gate(inputs, outputs), kernel_names, suppress_kineto_output=True)
        if isinstance(fused_t, tuple):
            fused_t = sum(fused_t)
        num_flops = 2 * num_tokens * num_routed_experts * hidden
        num_bytes = count_bytes(*inputs.values(), *outputs.values())
        print(f' > Perf (m={num_tokens:6}, n={num_routed_experts:4}, k={hidden:4}, topk={NUM_TOPK}): '
              f'{fused_t * 1e6:7.1f} us | '
              f'{num_flops / fused_t / 1e12:4.0f} TFLOPS | '
              f'{num_bytes / fused_t / 1e9:4.0f} GB/s | '
              f'diff {weight_diff:.1e}')
    print()


def check_small_batch_routing_controls() -> None:
    num_tokens, hidden, num_routed_experts = 16, 4096, 384
    inputs = make_inputs(num_tokens, hidden, num_routed_experts, use_shared_as_routed=True, with_image_bias=True)
    outputs = make_outputs(inputs, use_shared_as_routed=True, with_unmapped_topk_idx=True)
    logits = run_reference_gemm(inputs)

    token_idx = torch.arange(num_tokens, device='cuda')
    mask, fix_routing_mask, force_random = token_idx % 11 != 0, token_idx % 3 == 0, token_idx % 5 == 0
    fixed_topk_idx = torch.randint(
        0, num_routed_experts, (num_tokens, NUM_TOPK), dtype=torch.int64, device='cuda')
    unmapped_topk_idx = outputs['unmapped_topk_idx']
    unmapped_topk_idx.copy_(fixed_topk_idx)

    fused_idx, fused_weights = run_mega_gate(
        inputs, outputs, True, mask=mask, fix_routing_mask=fix_routing_mask, force_random=force_random)
    check_topk_weights(inputs, outputs, logits, mask, force_random, fix_routing_mask)

    active_force_random = force_random & mask
    fixed = fix_routing_mask & mask & ~force_random
    assert torch.equal(unmapped_topk_idx[fixed], fixed_topk_idx[fixed])
    random_idx, random_weights = fused_idx[active_force_random], fused_weights[active_force_random]
    assert torch.all((random_idx >= 0) & (random_idx < inputs['logical_count'].sum())).item()
    assert torch.all(random_weights > 0).item()
    sorted_random_idx = random_idx.sort(dim=1).values
    assert torch.all(sorted_random_idx[:, 1:] != sorted_random_idx[:, :-1]).item()
    assert torch.all(unmapped_topk_idx[active_force_random] == -1).item()


@test_filter(lambda: get_arch_major() == 9)
@torch.no_grad()
def test_sm90_mega_gate_api_contract() -> None:
    # Reuse upstream contract checks, bypassing only their SM100 architecture filter.
    _test_mega_gate_api_contract.__wrapped__()
    check_small_batch_routing_controls()


@test_filter(lambda: get_arch_major() in (9, 10))
@torch.no_grad()
def test_mega_gate_boundaries() -> None:
    torch.manual_seed(3)
    # Expert padding, partial token tiles, smallest hidden size and all top-k lanes.
    for num_tokens, num_experts, num_topk in ((1, 4, 1), (3, 4, 4), (17, 132, 32),
                                            (65, 260, 32), (129, 388, 32), (257, 512, 32)):
        check_case(num_tokens, 256, num_experts, num_topk=num_topk,
                   with_image_bias=True, with_mask=True, with_unmapped_topk_idx=True)
    check_case(33, 256, 128, num_topk=32, with_bias=False, with_physical_map=False)
    check_case(33, 256, 260, num_topk=6, with_bias=False, with_image_bias=True,
               with_mask=True, with_physical_map=False)

    for num_tokens in (3, 17):
        inputs = make_inputs(num_tokens, 256, 132, with_image_bias=True)
        for key in ('bias', 'image_bias'):
            inputs[key] = torch.randn((133,), dtype=torch.float32, device='cuda')[1:]
            assert inputs[key].is_contiguous() and inputs[key].data_ptr() % 16 != 0
        check_correctness(inputs, make_outputs(inputs, with_unmapped_topk_idx=True))

    # Exercise both shared slots and a nonzero EP rank, including duplicate selection.
    inputs = make_inputs(33, 256, 256, use_shared_as_routed=True, num_shared_experts=2)
    logical = torch.empty((33, 8), dtype=torch.int64, device='cuda')
    idx, weights = deep_gemm.bf16_mega_gate(
        **inputs, num_topk=8, use_shared_as_routed=True, num_shared_experts=2,
        routed_scaling_factor=0.75, ep_rank=7, unmapped_topk_idx=logical)
    all_logical = torch.cat((logical, torch.tensor([256, 257], device='cuda').expand(33, -1)), dim=1)
    duplicates = (7 + torch.arange(33, device='cuda')[:, None] * 23333) % inputs['logical_count'][all_logical]
    assert torch.equal(idx, inputs['to_physical_map'][all_logical, duplicates].long())
    scores = F.softplus(run_reference_gemm(inputs)).sqrt().gather(1, logical)
    torch.testing.assert_close(weights[:, :8], scores / (scores.sum(dim=1, keepdim=True) + 1e-20) * 0.75)
    assert torch.all(weights[:, 8:] == 1).item()


@test_filter(lambda: get_arch_major() in (9, 10))
@torch.no_grad()
def test_mega_gate_cuda_graph() -> None:
    torch.manual_seed(4)
    # Independent captures on multiple streams catch shared workspace/barrier reuse.
    jobs = []
    for num_tokens, num_experts in ((3, 132), (17, 132), (129, 388)):
        inputs = make_inputs(num_tokens, 256, num_experts, with_image_bias=True, with_mask=True)
        outputs = make_outputs(inputs, with_unmapped_topk_idx=True)
        storage = torch.full((num_tokens, NUM_TOPK + 3), -123, dtype=torch.int64, device='cuda')
        outputs['unmapped_topk_idx'] = storage[:, :NUM_TOPK]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run_mega_gate(inputs, outputs)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = run_mega_gate(inputs, outputs)
        assert result[0].data_ptr() == outputs['topk_idx'].data_ptr()
        assert result[1].data_ptr() == outputs['topk_weights'].data_ptr()
        jobs.append((inputs, outputs, storage, stream, graph))

    for _ in range(3):
        for inputs, outputs, storage, stream, graph in jobs:
            with torch.cuda.stream(stream):
                inputs['x'].normal_()
                inputs['weight'].normal_(std=256 ** -0.5)
                inputs['mask'].logical_not_()
                inputs['image_token_mask'].logical_not_()
                graph.replay()
        for inputs, outputs, storage, stream, graph in jobs:
            torch.cuda.current_stream().wait_stream(stream)
            check_topk_weights(inputs, outputs, run_reference_gemm(inputs), inputs['mask'])
            assert torch.all(storage[:, NUM_TOPK:] == -123).item()


@test_filter(lambda: get_arch_major() == 9)
@torch.no_grad()
def test_sm90_mega_gate_batch_invariance() -> None:
    torch.manual_seed(5)
    deep_gemm.use_deterministic_algorithms(True)
    try:
        # Cross the 16/64-token tile boundary without changing any dot product.
        for num_experts in (128, 384, 512):
            inputs = make_inputs(16, 256, num_experts, with_physical_map=False)
            outputs = make_outputs(inputs, with_unmapped_topk_idx=True)
            run_mega_gate(inputs, outputs)
            large_inputs = {**inputs, 'x': inputs['x'].repeat(256, 1)}
            large_outputs = make_outputs(large_inputs, with_unmapped_topk_idx=True)
            run_mega_gate(large_inputs, large_outputs)
            for name in outputs:
                assert_bitwise_equal(large_outputs[name], outputs[name].repeat(256, 1), name)
    finally:
        deep_gemm.use_deterministic_algorithms(False)


if __name__ == '__main__':
    test_sm90_mega_gate_api_contract()
    test_mega_gate_boundaries()
    test_mega_gate_cuda_graph()
    test_sm90_mega_gate_batch_invariance()
    test_sm90_mega_gate()
