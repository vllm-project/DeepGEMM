"""Opt-in regressions for wide flattened MegaMoE and Triton indices."""

import os

import pytest
import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.utils.dist import init_dist
from deep_gemm.utils import per_token_cast_to_fp8
from test_mega_moe import _to_mn_major_sf


pytestmark = [
    pytest.mark.skipif(os.getenv('DG_SM120_MOE_LARGE_INDEX') != '1',
                       reason='Set DG_SM120_MOE_LARGE_INDEX=1; requires at least 40 GiB free GPU memory'),
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
                       reason='SM120 required'),
]


@pytest.fixture(autouse=True)
def require_large_index_memory():
    torch.cuda.empty_cache()
    if torch.cuda.mem_get_info()[0] < 40 * 1024 ** 3:
        pytest.skip('Large-index regression requires at least 40 GiB free GPU memory')


def test_sm120_mega_moe_large_output_index():
    previous_device = torch.get_default_device()
    _, _, group = init_dist(0, 1)
    buffer = None
    try:
        hidden, intermediate = 32768, 128
        num_tokens = (1 << 32) // hidden + 1

        def zero_weights(rows, columns):
            packed = torch.zeros((1, rows, columns // 2), dtype=torch.int8, device='cuda')
            scales = torch.full((1, rows, columns // 128), 0x7f7f7f7f, dtype=torch.int32, device='cuda')
            return packed, _to_mn_major_sf(scales)

        weights = deep_gemm.transform_weights_for_mega_moe(
            zero_weights(intermediate * 2, hidden), zero_weights(hidden, intermediate))
        buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            group, 1, num_tokens, 1, hidden, intermediate, mma_type='fp8xfp4')
        buffer.x.zero_()
        buffer.x_sf.fill_(0x7f7f7f7f)
        buffer.topk_idx.zero_()
        buffer.topk_weights.fill_(1)
        output = torch.full((num_tokens, hidden), float('nan'), dtype=torch.bfloat16, device='cuda')
        assert output.numel() > 1 << 32
        deep_gemm.fp8_fp4_mega_moe(output, *weights, buffer)
        nonzero_elements = torch.count_nonzero(output).item()
        print(f'Output elements={output.numel()}, nonzero or unwritten elements={nonzero_elements}')
        assert nonzero_elements == 0, f'{nonzero_elements} output elements are nonzero or remain poisoned'
    finally:
        torch.cuda.synchronize()
        if buffer is not None:
            buffer.destroy()
        dist.destroy_process_group()
        torch.set_default_device(previous_device)


def test_sm120_mega_moe_triton_large_index():
    pytest.importorskip('triton')
    from bench_mega_moe_triton import combine_routed, dispatch_quantized, swiglu_quantized

    hidden, intermediate, num_rows = 32768, 16384, 65537
    activations = torch.empty((num_rows, hidden), dtype=torch.float8_e4m3fn, device='cuda')
    activations.view(torch.uint8).zero_()
    scales = torch.full((num_rows, hidden // 128), 0x7f7f7f7f, dtype=torch.int32, device='cuda')
    token_indices = torch.arange(num_rows, dtype=torch.int64, device='cuda')
    routed = torch.empty_like(activations)
    routed.view(torch.uint8).fill_(0xff)
    routed_scales = torch.full((hidden // 128, num_rows), -1, dtype=torch.int32, device='cuda').t()
    assert routed.numel() > 1 << 31
    dispatch_quantized(activations, scales, token_indices, routed, routed_scales)
    assert torch.count_nonzero(routed.view(torch.uint8)).item() == 0
    assert (routed_scales == 0x7f7f7f7f).all().item()

    gate_up = torch.zeros((num_rows, intermediate * 2), dtype=torch.bfloat16, device='cuda')
    routing_weights = torch.ones(num_rows, dtype=torch.float32, device='cuda')
    quantized = torch.empty((num_rows, intermediate), dtype=torch.float8_e4m3fn, device='cuda')
    quantized.view(torch.uint8).fill_(0xff)
    quantized_scales = torch.full((intermediate // 128, num_rows), -1, dtype=torch.int32, device='cuda').t()
    assert gate_up.numel() > 1 << 31
    swiglu_quantized(gate_up, routing_weights, quantized, quantized_scales, 10)
    assert torch.count_nonzero(quantized.view(torch.uint8)).item() == 0
    _, expected_scales = per_token_cast_to_fp8(torch.zeros((1, 128), dtype=torch.bfloat16, device='cuda'),
                                               use_ue8m0=True, use_packed_ue8m0=True, gran_k=32)
    assert (quantized_scales == expected_scales.flatten()[0]).all().item()

    output = torch.full_like(gate_up, float('nan'))
    combine_routed(gate_up, token_indices, output, 1)
    assert torch.count_nonzero(output).item() == 0
    print(f'Triton dispatch/activation/combine passed with {output.numel()} elements (>2**31)')
