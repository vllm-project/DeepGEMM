from contextlib import contextmanager
import json
from math import prod
from pathlib import Path
import tempfile

import torch

import deep_gemm
from deep_gemm.testing import (
    assert_bitwise_equal,
    count_bytes,
    get_arch_major,
    test_filter,
)
from test_mega_gate import (
    NUM_TOPK,
    check_case,
    check_correctness,
    check_routing_controls,
    check_topk_weights,
    enumerate_cases,
    make_inputs,
    make_outputs,
    run_mega_gate,
    test_mega_gate_api_contract,
    test_mega_gate_boundaries,
    test_mega_gate_cuda_graph,
)


@contextmanager
def limited_sms(num_sms: int):
    previous = deep_gemm.get_num_sms()
    deep_gemm.set_num_sms(num_sms)
    try:
        yield
    finally:
        deep_gemm.set_num_sms(previous)


@test_filter(lambda: get_arch_major() == 9)
@torch.no_grad()
def test_sm90_mega_gate() -> None:
    torch.manual_seed(0)

    print('Testing BF16 GEMM + MoE top-k gate (CUDA Graph, warm L2):')
    for num_tokens, hidden, num_routed_experts in enumerate_cases():
        inputs = make_inputs(num_tokens, hidden, num_routed_experts)
        outputs = make_outputs(inputs)
        weight_diff = check_correctness(inputs, outputs)

        # CUDA events over a captured call include GEMM and routing on every path.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run_mega_gate(inputs, outputs)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run_mega_gate(inputs, outputs)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(30):
            graph.replay()
        end.record()
        end.synchronize()
        fused_t = start.elapsed_time(end) / 30 / 1000
        num_flops = 2 * num_tokens * num_routed_experts * hidden
        num_bytes = count_bytes(*inputs.values(), *outputs.values())
        print(f' > Perf (m={num_tokens:6}, n={num_routed_experts:4}, k={hidden:4}, topk={NUM_TOPK}): '
              f'{fused_t * 1e6:7.1f} us | '
              f'{num_flops / fused_t / 1e12:4.0f} TFLOPS | '
              f'{num_bytes / fused_t / 1e9:4.0f} GB/s | '
              f'diff {weight_diff:.1e}')
    print()


@test_filter(lambda: get_arch_major() == 9)
@torch.no_grad()
def test_sm90_mega_gate_split_n() -> None:
    with limited_sms(max(1, deep_gemm.get_num_sms() // 2)):
        torch.manual_seed(6)
        # The last expert tile may have fewer valid experts than its local top-k.
        for num_experts in (132, 260, 388, 512):
            check_case(32769, 256, num_experts, num_topk=32, with_image_bias=True,
                       with_mask=True, with_unmapped_topk_idx=True)
        check_case(32768, 5120, 384, with_bias=False, with_physical_map=False)
        check_case(32769, 5120, 384, with_image_bias=True, with_mask=True, with_unmapped_topk_idx=True)
        check_routing_controls(32769)

        inputs = make_inputs(32769, 256, 384, with_physical_map=False)
        inputs['x'].zero_()
        inputs['bias'].zero_()
        outputs = make_outputs(inputs)
        expected = torch.tensor([0, 127, 128, 255, 256, 383], device='cuda')
        inputs['bias'][expected] = 1
        idx, weights = run_mega_gate(inputs, outputs)
        assert torch.equal(idx, expected.expand_as(idx))
        torch.testing.assert_close(weights, torch.full_like(weights, 1.5 / NUM_TOPK))

        # Equal rankings across all expert groups must prefer the smallest IDs.
        inputs['bias'].zero_()
        idx, _ = run_mega_gate(inputs, outputs)
        assert torch.equal(idx, torch.arange(NUM_TOPK, device='cuda').expand_as(idx))

        for key in ('bias', 'image_bias'):
            inputs[key] = torch.randn((385,), device='cuda')[1:]
        inputs['image_token_mask'] = torch.arange(32769, device='cuda') % 2 == 0
        inputs['x'].normal_()
        check_correctness(inputs, outputs)


@test_filter(lambda: get_arch_major() == 9)
@torch.no_grad()
def test_sm90_mega_gate_batch_invariance() -> None:
    torch.manual_seed(5)
    deep_gemm.use_deterministic_algorithms(True)
    try:
        # Cross the tile boundary and force persistent waves, with a full K reduction.
        for num_experts, hidden in ((128, 256), (384, 5120), (512, 7168)):
            for fixed_routing in (False, True):
                inputs = make_inputs(16, hidden, num_experts, with_physical_map=False)
                outputs = make_outputs(inputs, with_unmapped_topk_idx=True)
                fixed_indices = torch.arange(NUM_TOPK, device='cuda').expand(16, -1)
                outputs['unmapped_topk_idx'].copy_(fixed_indices)
                overrides = {'fix_routing_mask': torch.ones(16, dtype=torch.bool, device='cuda')} if fixed_routing else {}
                run_mega_gate(inputs, outputs, **overrides)
                large_inputs = {**inputs, 'x': inputs['x'].repeat(256, 1)}
                large_outputs = make_outputs(large_inputs, with_unmapped_topk_idx=True)
                large_outputs['unmapped_topk_idx'].copy_(fixed_indices.repeat(256, 1))
                large_overrides = {key: value.repeat(256) for key, value in overrides.items()}
                run_mega_gate(large_inputs, large_outputs, **large_overrides)
                for name in outputs:
                    assert_bitwise_equal(large_outputs[name], outputs[name].repeat(256, 1), name)
    finally:
        deep_gemm.use_deterministic_algorithms(False)


@test_filter(lambda: get_arch_major() == 9)
@torch.no_grad()
def test_sm90_mega_gate_nonfinite() -> None:
    torch.manual_seed(7)
    # Force Split-N at its real threshold, independently of device-specific dispatch.
    with limited_sms(max(1, deep_gemm.get_num_sms() // 2)):
        for num_experts in (132, 384, 512):
            for with_physical_map in (False, True):
                for with_bias in (False, True):
                    inputs = make_inputs(32769, 256, num_experts, with_bias=with_bias,
                                         with_physical_map=with_physical_map)
                    bad_rows = torch.tensor([5, 16, 32768], device='cuda')
                    inputs['x'][bad_rows] = torch.tensor(
                        [float('nan'), float('inf'), -float('inf')], dtype=torch.bfloat16, device='cuda')[:, None]
                    outputs = make_outputs(inputs, with_unmapped_topk_idx=True)
                    run_mega_gate(inputs, outputs)
                    logical = outputs['unmapped_topk_idx']
                    assert torch.all((logical >= 0) & (logical < num_experts)).item()
                    physical_limit = inputs['logical_count'].sum() if with_physical_map else num_experts
                    assert torch.all((outputs['topk_idx'] >= 0) & (outputs['topk_idx'] < physical_limit)).item()
                    assert torch.all(outputs['topk_weights'][bad_rows] == 0).item()
                    assert torch.isfinite(outputs['topk_weights']).all().item()
                    good_inputs = {**inputs, 'x': inputs['x'][:5]}
                    good_outputs = {key: value[:5] for key, value in outputs.items()}
                    logits = good_inputs['x'].double() @ good_inputs['weight'].double().T
                    check_topk_weights(good_inputs, good_outputs, logits.float())


@test_filter(lambda: get_arch_major() == 9)
@torch.no_grad()
def test_sm90_mega_gate_independent_reference() -> None:
    torch.manual_seed(8)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for deterministic in (False, True):
            deep_gemm.use_deterministic_algorithms(deterministic)
            cases = ((1, 5120, 384), (16, 7168, 512), (129, 4096, 256), (8192, 5120, 384)) if deterministic else (
                (17, 4096, 256), (1024, 5120, 384), (32768, 5120, 128),
                (32768, 5120, 384), (65536, 7168, 512))
            for num_tokens, hidden, num_experts in cases:
                inputs = make_inputs(num_tokens, hidden, num_experts, with_image_bias=True)
                outputs = make_outputs(inputs, with_unmapped_topk_idx=True)
                run_mega_gate(inputs, outputs)
                logits = inputs['x'].float() @ inputs['weight'].float().T
                check_topk_weights(inputs, outputs, logits, allow_numerical_ties=True)
                # Check rows across the batch in FP64 as well, independently of cuBLASLt BF16 GEMM.
                rows = torch.linspace(0, num_tokens - 1, min(num_tokens, 32), device='cuda').long()
                sampled_inputs = {**inputs, 'x': inputs['x'][rows], 'image_token_mask': inputs['image_token_mask'][rows]}
                sampled_outputs = {key: value[rows] for key, value in outputs.items()}
                reference = sampled_inputs['x'].double() @ inputs['weight'].double().T
                check_topk_weights(sampled_inputs, sampled_outputs, reference.float(), allow_numerical_ties=True)
    finally:
        deep_gemm.use_deterministic_algorithms(False)
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


@test_filter(lambda: get_arch_major() == 9)
@torch.no_grad()
def test_sm90_mega_gate_sm_budget() -> None:
    torch.manual_seed(9)
    original_sms = deep_gemm.get_num_sms()
    for num_sms in (1, max(1, original_sms // 2)):
        with limited_sms(num_sms):
            # GEMV, fused WGMMA, and Split-N + merge, including persistent iterations.
            for num_tokens in (16, 129, 32769):
                inputs = make_inputs(num_tokens, 256, 384, with_image_bias=True)
                outputs = make_outputs(inputs, with_unmapped_topk_idx=True)
                run_mega_gate(inputs, outputs)
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as profile:
                    run_mega_gate(inputs, outputs)
                    torch.cuda.synchronize()
                with tempfile.TemporaryDirectory() as directory:
                    trace = Path(directory) / 'gate.json'
                    profile.export_chrome_trace(str(trace))
                    kernels = [event for event in json.loads(trace.read_text())['traceEvents']
                               if event.get('cat') == 'kernel' and 'sm90_bf16_mega_gate' in event['name']]
                assert kernels, 'No Mega Gate launches recorded'
                assert all(prod(event['args']['grid']) <= num_sms for event in kernels), kernels
                logits = inputs['x'].double() @ inputs['weight'].double().T
                check_topk_weights(inputs, outputs, logits.float())


@test_filter(lambda: get_arch_major() == 9)
@torch.no_grad()
def test_sm90_mega_gate_fallback_cuda_graph() -> None:
    original_pdl = deep_gemm.get_pdl()
    try:
        with limited_sms(max(1, deep_gemm.get_num_sms() // 2)):
            for pdl in (False, True):
                deep_gemm.set_pdl(pdl)
                test_mega_gate_cuda_graph()
    finally:
        deep_gemm.set_pdl(original_pdl)


if __name__ == '__main__':
    test_mega_gate_api_contract()
    test_mega_gate_boundaries()
    test_sm90_mega_gate_split_n()
    test_mega_gate_cuda_graph()
    test_sm90_mega_gate_batch_invariance()
    test_sm90_mega_gate_nonfinite()
    test_sm90_mega_gate_independent_reference()
    test_sm90_mega_gate_sm_budget()
    test_sm90_mega_gate_fallback_cuda_graph()
    test_sm90_mega_gate()
