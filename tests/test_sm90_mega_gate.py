from contextlib import contextmanager
import json
from math import prod
from pathlib import Path
import tempfile
from typing import Any

import torch
import torch.nn.functional as F

import deep_gemm
from deep_gemm.testing import (
    assert_bitwise_equal,
    calc_diff,
    count_bytes,
    get_arch_major,
    test_filter as _test_filter,
)
from test_mega_gate import (
    GATE_CONFIGS,
    NUM_TOPK,
    ROUTED_SCALING_FACTOR,
    check_case,
    check_correctness,
    enumerate_cases,
    make_inputs,
    make_outputs,
    run_mega_gate,
    run_reference_gemm,
)


def check_topk_weights(inputs: dict[str, Any], outputs: dict[str, torch.Tensor | None], logits: torch.Tensor,
                       mask: torch.Tensor | None = None,
                       force_random: torch.Tensor | None = None,
                       fix_routing_mask: torch.Tensor | None = None,
                       num_topk: int = NUM_TOPK,
                       allow_numerical_ties: bool = False) -> float:
    fused_idx, fused_weights = outputs['topk_idx'], outputs['topk_weights']
    num_tokens, num_routed_experts = logits.shape
    active = mask if mask is not None else torch.ones((num_tokens,), dtype=torch.bool, device='cuda')
    score_based = active if force_random is None else active & ~force_random
    weight_diff = 0.0

    if score_based.any():
        scores = F.softplus(logits).sqrt()
        logical_topk_idx = outputs['unmapped_topk_idx']
        if logical_topk_idx is None:
            logical_topk_idx = fused_idx[:, :num_topk]
            to_physical_map = inputs['to_physical_map']
            if to_physical_map is not None:
                logical_topk_idx = torch.where(
                    logical_topk_idx < to_physical_map.shape[0],
                    logical_topk_idx, logical_topk_idx - to_physical_map.shape[0])

        selected_idx = logical_topk_idx[score_based]
        assert torch.all((selected_idx >= 0) & (selected_idx < num_routed_experts)).item()

        ranking_bias = (inputs['bias'] if inputs['bias'] is not None
                        else torch.zeros((num_routed_experts,), dtype=torch.float32, device='cuda'))
        if inputs['image_token_mask'] is not None:
            ranking_bias = torch.where(
                inputs['image_token_mask'][:, None], inputs['image_bias'], ranking_bias)
        ranking = scores + ranking_bias

        def reference_weights(row_mask: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
            topk_scores = scores[row_mask].gather(1, topk_idx)
            return topk_scores / (topk_scores.sum(dim=1, keepdim=True) + 1e-20) * ROUTED_SCALING_FACTOR

        weight_diff = float(calc_diff(fused_weights[score_based, :num_topk],
                                      reference_weights(score_based, selected_idx)))
        assert weight_diff < 1e-11, f'{weight_diff=}'

        ranked = score_based if fix_routing_mask is None else score_based & ~fix_routing_mask
        if ranked.any():
            ranked_idx = logical_topk_idx[ranked]
            sorted_idx = ranked_idx.sort(dim=1).values
            assert torch.all(sorted_idx[:, 1:] != sorted_idx[:, :-1]).item()

            selected_ranking = ranking[ranked].gather(1, ranked_idx)
            unselected_max = ranking[ranked].scatter(1, ranked_idx, float('-inf')).max(dim=1).values
            crossing = unselected_max - selected_ranking.min(dim=1).values
            tolerance = 1e-5 + 1e-5 * ranking[ranked].abs().amax(dim=1)
            assert torch.all(crossing <= tolerance).item(), \
                f'top-k selection beyond numerical ties: {crossing.max().item()=}'

            ref_weights = reference_weights(ranked, ranking[ranked].topk(num_topk, dim=1).indices)
            # Independent GEMMs may exchange experts whose biased scores differ
            # only by rounding. Their unbiased weights need not be equal. Every
            # selected expert's weight and the ranking bound were checked above.
            stable = crossing < -tolerance if allow_numerical_ties else torch.ones_like(crossing, dtype=torch.bool)
            if stable.any():
                ranked_diff = float(calc_diff(fused_weights[ranked, :num_topk][stable].sort(dim=1).values,
                                              ref_weights[stable].sort(dim=1).values))
                assert ranked_diff < 1e-11, f'{ranked_diff=}'
                weight_diff = max(weight_diff, ranked_diff)

        if fused_weights.shape[1] > num_topk:
            assert torch.all(fused_weights[score_based, num_topk:] == 1.0).item()

    if (~active).any():
        assert torch.all(fused_idx[~active] == -1).item()
        assert torch.all(fused_weights[~active] == 0).item()
        if outputs['unmapped_topk_idx'] is not None:
            assert torch.all(outputs['unmapped_topk_idx'][~active] == -1).item()

    return weight_diff



def check_routing_controls(num_tokens: int = 512) -> None:
    hidden, num_routed_experts = 4096, 384
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

    # Omitting fixed routing exercises the split-N merge's random/masked paths.
    run_mega_gate(inputs, outputs, True, mask=mask, force_random=force_random)
    check_topk_weights(inputs, outputs, logits, mask, force_random)
    assert torch.all(unmapped_topk_idx[active_force_random] == -1).item()
    assert torch.equal(outputs['topk_idx'][active_force_random], random_idx)
    assert torch.equal(outputs['topk_weights'][active_force_random], random_weights)



@_test_filter(lambda: get_arch_major() == 9)
@torch.no_grad()
def test_sm90_mega_gate_api_contract() -> None:
    torch.manual_seed(2)

    for num_routed_experts, num_shared_experts, num_topk in GATE_CONFIGS:
        can_route_shared = (
            num_topk % num_shared_experts == 0 and
            num_routed_experts % (num_topk // num_shared_experts) == 0)
        for use_shared_as_routed in ((False, True) if can_route_shared else (False,)):
            check_case(16, 4096, num_routed_experts, use_shared_as_routed=use_shared_as_routed,
                       num_topk=num_topk, num_shared_experts=num_shared_experts)

    check_case(16, 4096, 128, with_bias=False, with_physical_map=False)
    check_case(129, 4352, 260, with_image_bias=True, with_mask=True, with_unmapped_topk_idx=True)

    inputs = make_inputs(128, 4096, 384, with_image_bias=True)
    inputs['bias'].fill_(1e8)
    inputs['image_bias'].fill_(-1e8)
    check_correctness(inputs, make_outputs(inputs))

    check_routing_controls()
    check_routing_controls(16)

    num_topk = 8
    inputs = make_inputs(8, 4096, 128, with_bias=False, with_physical_map=False)
    inputs['x'].zero_()
    inputs['weight'].zero_()
    topk_idx, _ = run_mega_gate(inputs, make_outputs(inputs, num_topk=num_topk), num_topk=num_topk)
    assert torch.equal(topk_idx, torch.arange(num_topk, dtype=torch.int64, device='cuda').expand(8, -1))

    inputs = make_inputs(16, 4096, 128, with_physical_map=False)
    outputs = make_outputs(inputs, with_unmapped_topk_idx=True, num_topk=num_topk)
    unmapped_storage = torch.empty((16, num_topk + 3), dtype=torch.int64, device='cuda')
    outputs['unmapped_topk_idx'] = unmapped_storage[:, :num_topk]
    assert outputs['unmapped_topk_idx'].stride() == (num_topk + 3, 1)
    check_correctness(inputs, outputs, num_topk=num_topk)

    inputs = make_inputs(32, 4096, 128, with_physical_map=False)
    outputs = make_outputs(inputs, with_unmapped_topk_idx=True, num_topk=num_topk)
    force_random = torch.ones((32,), dtype=torch.bool, device='cuda')
    topk_idx, topk_weights = run_mega_gate(inputs, outputs, num_topk=num_topk, force_random=force_random)
    assert torch.all((topk_idx >= 0) & (topk_idx < 128)).item()
    assert torch.all(topk_weights > 0).item()
    sorted_topk_idx = topk_idx.sort(dim=1).values
    assert torch.all(sorted_topk_idx[:, 1:] != sorted_topk_idx[:, :-1]).item()
    assert torch.all(outputs['unmapped_topk_idx'] == -1).item()

    inputs = make_inputs(0, 4096, 128, with_physical_map=False)
    topk_idx, topk_weights = run_mega_gate(inputs, make_outputs(inputs, num_topk=num_topk), num_topk=num_topk)
    assert topk_idx.shape == (0, num_topk) and topk_weights.shape == (0, num_topk)

    # Deterministic mode never splits K (batch invariance)
    deep_gemm.use_deterministic_algorithms(True)
    try:
        for num_tokens, hidden, num_routed_experts in ((16, 4096, 256), (8192, 5120, 384)):
            check_case(num_tokens, hidden, num_routed_experts, with_unmapped_topk_idx=True)
    finally:
        deep_gemm.use_deterministic_algorithms(False)



@_test_filter(lambda: get_arch_major() == 9)
@torch.no_grad()
def test_sm90_mega_gate_boundaries() -> None:
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



@_test_filter(lambda: get_arch_major() == 9)
@torch.no_grad()
def test_sm90_mega_gate_cuda_graph() -> None:
    torch.manual_seed(4)
    # Independent captures on multiple streams catch shared workspace/barrier reuse.
    jobs = []
    for num_tokens, num_experts in ((3, 132), (17, 132), (129, 388), (32769, 260), (32769, 384)):
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



@contextmanager
def limited_sms(num_sms: int):
    previous = deep_gemm.get_num_sms()
    deep_gemm.set_num_sms(num_sms)
    try:
        yield
    finally:
        deep_gemm.set_num_sms(previous)


@_test_filter(lambda: get_arch_major() == 9)
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


@_test_filter(lambda: get_arch_major() == 9)
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


@_test_filter(lambda: get_arch_major() == 9)
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


@_test_filter(lambda: get_arch_major() == 9)
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


@_test_filter(lambda: get_arch_major() == 9)
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


@_test_filter(lambda: get_arch_major() == 9)
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


@_test_filter(lambda: get_arch_major() == 9)
@torch.no_grad()
def test_sm90_mega_gate_fallback_cuda_graph() -> None:
    original_pdl = deep_gemm.get_pdl()
    try:
        with limited_sms(max(1, deep_gemm.get_num_sms() // 2)):
            for pdl in (False, True):
                deep_gemm.set_pdl(pdl)
                test_sm90_mega_gate_cuda_graph()
    finally:
        deep_gemm.set_pdl(original_pdl)


if __name__ == '__main__':
    test_sm90_mega_gate_api_contract()
    test_sm90_mega_gate_boundaries()
    test_sm90_mega_gate_split_n()
    test_sm90_mega_gate_cuda_graph()
    test_sm90_mega_gate_batch_invariance()
    test_sm90_mega_gate_nonfinite()
    test_sm90_mega_gate_independent_reference()
    test_sm90_mega_gate_sm_budget()
    test_sm90_mega_gate_fallback_cuda_graph()
    test_sm90_mega_gate()
