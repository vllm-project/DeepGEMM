"""Opt-in DeepSeek checkpoint layer validation with synthetic hidden states, not model evaluation."""

import functools
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.testing import calc_diff
from deep_gemm.utils import cast_back_from_fp4, per_token_cast_to_fp8
from deep_gemm.utils.dist import init_dist
from test_mega_moe import _apply_activation, _calc_token_diff, _emu_gemm, _hi_expert, _moe_reference, _to_mn_major_sf


pytestmark = [
    pytest.mark.skipif(not os.getenv('DG_SM120_MOE_CHECKPOINT'), reason='Set DG_SM120_MOE_CHECKPOINT'),
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
                       reason='SM120 required'),
]


def _checkpoint_expert(activations, activation_scales, l1_weights, l2_weights, routing_weights, args):
    def gemm(values, scales, weights):
        packed, exponents = weights
        unscaled = cast_back_from_fp4(packed, torch.ones_like(exponents, dtype=torch.float32), gran_k=32)
        weight_scales = torch.exp2(exponents.float() - 127)
        return _emu_gemm(values.float(), scales, unscaled, weight_scales, 32)

    gate_up = gemm(activations, activation_scales, l1_weights).bfloat16().float()
    intermediate, scales = per_token_cast_to_fp8(
        _apply_activation(gate_up, routing_weights, args), use_ue8m0=True, gran_k=32)
    return gemm(intermediate, scales, l2_weights).bfloat16()


@pytest.fixture(scope='module')
def checkpoint_group():
    previous_device = torch.get_default_device()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    _, _, group = init_dist(0, 1)
    try:
        yield group
    finally:
        dist.destroy_process_group()
        torch.set_default_device(previous_device)
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


@pytest.fixture(scope='module', params=(1, 8))
def checkpoint_case(request, checkpoint_group):
    safe_open = pytest.importorskip('safetensors').safe_open
    checkpoint_path = Path(os.environ['DG_SM120_MOE_CHECKPOINT'])
    config = json.loads((checkpoint_path / 'config.json').read_text())
    assert config['quantization_config']['expert_dtype'] == 'fp4'
    assert config['quantization_config']['scale_fmt'] == 'ue8m0'
    text_config = config.get('text_config', config)
    assert text_config['scoring_func'] == 'sqrtsoftplus' and text_config['norm_topk_prob']
    layer_idx = int(os.getenv('DG_SM120_MOE_LAYER', '0'))
    prefix = f'layers.{layer_idx}.ffn'
    weight_map = json.loads((checkpoint_path / 'model.safetensors.index.json').read_text())['weight_map']

    def load_tensor(name):
        key = f'{prefix}.{name}'
        with safe_open(str(checkpoint_path / weight_map[key]), framework='pt', device='cuda') as shard:
            return shard.get_tensor(key)

    num_tokens, hidden = request.param, text_config['hidden_size']
    intermediate, num_topk = text_config['moe_intermediate_size'], text_config['num_experts_per_tok']
    generator = torch.Generator(device='cuda').manual_seed(20261112)
    hidden_states = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda', generator=generator)
    router_logits = (hidden_states @ load_tensor('gate.weight').T).float()
    scores = torch.nn.functional.softplus(router_logits).sqrt()
    global_routes = torch.topk(scores + load_tensor('gate.bias'), num_topk, sorted=False).indices
    routing_weights = scores.gather(1, global_routes)
    routing_weights = routing_weights / routing_weights.sum(-1, keepdim=True) * text_config['routed_scaling_factor']
    expert_ids = global_routes.unique(sorted=True)
    routes = torch.searchsorted(expert_ids, global_routes).long()

    def load_expert_weights(names, suffix):
        return torch.stack([
            torch.cat([load_tensor(f'experts.{expert_idx}.{name}.{suffix}') for name in names], dim=0)
            for expert_idx in expert_ids.tolist()
        ])

    l1_data, l2_data = load_expert_weights(('w1', 'w3'), 'weight'), load_expert_weights(('w2',), 'weight')
    l1_exponents = load_expert_weights(('w1', 'w3'), 'scale').view(torch.uint8)
    l2_exponents = load_expert_weights(('w2',), 'scale').view(torch.uint8)
    assert l1_data.dtype == l2_data.dtype == torch.int8
    assert l1_data.shape == (expert_ids.numel(), intermediate * 2, hidden // 2)
    assert l1_exponents.shape == (expert_ids.numel(), intermediate * 2, hidden // 32)
    assert l2_data.shape == (expert_ids.numel(), hidden, intermediate // 2)
    assert l2_exponents.shape == (expert_ids.numel(), hidden, intermediate // 32)
    assert (l1_exponents < 255).all() and (l2_exponents < 255).all()
    raw_weights = (l1_data, l1_exponents), (l2_data, l2_exponents)
    transformed_weights = deep_gemm.transform_weights_for_mega_moe(
        (l1_data, _to_mn_major_sf(l1_exponents.contiguous().view(torch.int32))),
        (l2_data, _to_mn_major_sf(l2_exponents.contiguous().view(torch.int32))))
    quantized, scales = per_token_cast_to_fp8(hidden_states, use_ue8m0=True, gran_k=32)
    packed_quantized, packed_scales = per_token_cast_to_fp8(
        hidden_states, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    activation_args = SimpleNamespace(activation='swiglu', activation_clamp=text_config['swiglu_limit'],
                                     activation_alpha=1.0, activation_beta=0.0)
    expected = _moe_reference(functools.partial(_checkpoint_expert, args=activation_args),
                              (quantized, scales), routes, routing_weights, *raw_weights, None, torch.float).bfloat16()
    high_precision = _moe_reference(functools.partial(_hi_expert, args=activation_args),
                                    (hidden_states.double(),), routes, routing_weights, *raw_weights, None, torch.double)

    shared_data = torch.cat([load_tensor(f'shared_experts.{name}.weight') for name in ('w1', 'w3')])
    shared_scales = torch.cat([load_tensor(f'shared_experts.{name}.scale').view(torch.uint8)
                              for name in ('w1', 'w3')]).repeat_interleave(32, dim=0)
    shared_down = load_tensor('shared_experts.w2.weight')
    shared_down_scales = load_tensor('shared_experts.w2.scale').view(torch.uint8).repeat_interleave(32, dim=0)
    shared_gate_up = torch.empty((num_tokens, intermediate * 2), dtype=torch.bfloat16, device='cuda')
    deep_gemm.fp8_gemm_nt((packed_quantized, _to_mn_major_sf(packed_scales)),
                         (shared_data, _to_mn_major_sf(shared_scales.contiguous().view(torch.int32))),
                         shared_gate_up, recipe=(1, 1, 32))
    shared_gate_up_reference = _emu_gemm(quantized.float(), scales, shared_data.float(),
                                        torch.exp2(shared_scales.float() - 127), 32).bfloat16().float()
    shared_activations = _apply_activation(shared_gate_up.float(), torch.ones(num_tokens), activation_args)
    shared_quantized, shared_packed_scales = per_token_cast_to_fp8(
        shared_activations, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    shared_output = torch.empty_like(hidden_states)
    deep_gemm.fp8_gemm_nt((shared_quantized, _to_mn_major_sf(shared_packed_scales)),
                         (shared_down, _to_mn_major_sf(shared_down_scales.contiguous().view(torch.int32))),
                         shared_output, recipe=(1, 1, 32))
    shared_reference_quantized, shared_reference_scales = per_token_cast_to_fp8(
        _apply_activation(shared_gate_up_reference, torch.ones(num_tokens), activation_args), use_ue8m0=True, gran_k=32)
    shared_expected = _emu_gemm(shared_reference_quantized.float(), shared_reference_scales, shared_down.float(),
                                torch.exp2(shared_down_scales.float() - 127), 32).bfloat16()
    assert calc_diff(shared_output, shared_expected) < 1e-5
    assert _calc_token_diff(shared_output, shared_expected).max().item() < 1e-4
    print(f'Checkpoint layer={layer_idx}, tokens={num_tokens}, loaded experts={expert_ids.tolist()}, '
          f'shared calc_diff={calc_diff(shared_output, shared_expected):.3e}')
    return SimpleNamespace(group=checkpoint_group, hidden=hidden, intermediate=intermediate, num_tokens=num_tokens,
                           num_experts=expert_ids.numel(), num_topk=num_topk, routes=routes, routing_weights=routing_weights,
                           quantized=packed_quantized, scales=packed_scales, weights=transformed_weights,
                           expected=expected, high_precision=high_precision, shared_output=shared_output,
                           shared_expected=shared_expected, activation_args=activation_args)


@pytest.mark.parametrize('block_m', (16, 32, 64))
def test_sm120_checkpoint_layer(checkpoint_case, block_m):
    case = checkpoint_case
    selection_tokens = {16: 0, 32: math.ceil(11 * case.num_experts / case.num_topk),
                        64: math.ceil(25 * case.num_experts / case.num_topk)}[block_m]
    assert deep_gemm.get_block_m_for_mega_moe(1, case.num_experts, case.num_tokens,
                                            selection_tokens, case.num_topk, 'fp8xfp4') == block_m
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        case.group, case.num_experts, case.num_tokens, case.num_topk, case.hidden, case.intermediate, mma_type='fp8xfp4')
    buffer.x[:case.num_tokens].copy_(case.quantized)
    buffer.x_sf[:case.num_tokens].copy_(case.scales)
    buffer.topk_idx[:case.num_tokens].copy_(case.routes)
    buffer.topk_weights[:case.num_tokens].copy_(case.routing_weights)
    output = torch.empty_like(case.expected)

    def launch():
        deep_gemm.fp8_fp4_mega_moe(output, *case.weights, buffer, activation_clamp=case.activation_args.activation_clamp,
                                  num_tokens_for_block_m=selection_tokens)

    try:
        launch()
        eager_output = output.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        for _ in range(3):
            output.fill_(float('nan'))
            graph.replay()
            assert torch.equal(output, eager_output)
        difference = calc_diff(output, case.expected)
        token_difference = _calc_token_diff(output, case.expected).max().item()
        high_precision_difference = calc_diff(output, case.high_precision)
        print(f'BLOCK_M={block_m}, calc_diff={difference:.3e}, max_token_diff={token_difference:.3e}, '
              f'hi_diff={high_precision_difference:.3e}')
        assert difference < 1e-5 and token_difference < 1e-4 and high_precision_difference < 2e-3
        combined = (output.float() + case.shared_output.float()).bfloat16()
        combined_reference = (case.expected.float() + case.shared_expected.float()).bfloat16()
        assert calc_diff(combined, combined_reference) < 1e-5
        assert _calc_token_diff(combined, combined_reference).max().item() < 1e-4
    finally:
        torch.cuda.synchronize()
        buffer.destroy()
