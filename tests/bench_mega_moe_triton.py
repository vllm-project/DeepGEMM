import torch
import triton
import triton.language as tl


@triton.jit
def _dispatch_quantized(activations, scales, token_indices, output, output_scales,
                        kNumRows: tl.constexpr, kHidden: tl.constexpr,
                        kInputScaleStrideRow: tl.constexpr, kInputScaleStrideCol: tl.constexpr,
                        kOutputScaleStrideCol: tl.constexpr, kBlockSize: tl.constexpr):
    program = tl.program_id(0)
    if kNumRows * kHidden > 0x7fffffff:
        program = program.to(tl.int64)
    offsets = program * kBlockSize + tl.arange(0, kBlockSize)
    rows, columns = offsets // kHidden, offsets % kHidden
    valid = rows < kNumRows
    source_rows = tl.load(token_indices + rows, valid, 0)
    values = tl.load(activations + source_rows * kHidden + columns, valid, 0)
    tl.store(output + offsets, values, valid)
    scale_mask = valid & (columns % 128 == 0)
    packed_scales = tl.load(scales + source_rows * kInputScaleStrideRow +
                           columns // 128 * kInputScaleStrideCol, scale_mask, 0)
    tl.store(output_scales + rows + columns // 128 * kOutputScaleStrideCol, packed_scales, scale_mask)


@triton.jit
def _swiglu_quantized(gate_up, routing_weights, output, output_scales,
                      kNumRows: tl.constexpr, kIntermediate: tl.constexpr, kOutputScaleStrideCol: tl.constexpr,
                      kClamp: tl.constexpr):
    row = tl.program_id(0)
    if kNumRows * kIntermediate * 2 > 0x7fffffff:
        row = row.to(tl.int64)
    channel_start = tl.program_id(1) * 128
    channels = channel_start + tl.arange(0, 128)
    gate = tl.load(gate_up + row * kIntermediate * 2 + channels).to(tl.float32)
    up = tl.load(gate_up + row * kIntermediate * 2 + kIntermediate + channels).to(tl.float32)
    if kClamp is not None:
        gate = tl.minimum(gate, kClamp)
        up = tl.minimum(tl.maximum(up, -kClamp), kClamp)
    weight = tl.load(routing_weights + row)
    values = tl.div_rn(gate, 1.0 + tl.exp(-gate)) * up * weight
    grouped = values.reshape((4, 32))
    amax = tl.maximum(tl.max(tl.abs(grouped), axis=1), 1e-4)
    rounded_exponents = (amax.to(tl.uint32, bitcast=True) + 0x7fffff - (0x60 << 16)) >> 23
    exponents = tl.maximum(rounded_exponents, 113) - 8
    inverse_scales = ((254 - exponents) << 23).to(tl.float32, bitcast=True)
    quantized = (grouped * inverse_scales[:, None]).reshape((128,))
    tl.store(output + row * kIntermediate + channels, quantized)
    packed_scale = tl.sum(exponents << (tl.arange(0, 4) * 8), axis=0)
    tl.store(output_scales + row + channel_start // 128 * kOutputScaleStrideCol, packed_scale)


@triton.jit
def _combine_routed(expert_outputs, route_rows, output, kNumTokens: tl.constexpr,
                     kHidden: tl.constexpr, kTopK: tl.constexpr, kBlockSize: tl.constexpr):
    program = tl.program_id(0)
    if kNumTokens * kHidden > 0x7fffffff:
        program = program.to(tl.int64)
    offsets = program * kBlockSize + tl.arange(0, kBlockSize)
    token_indices, channels = offsets // kHidden, offsets % kHidden
    valid = token_indices < kNumTokens
    values = tl.full((kBlockSize,), 0, tl.float32)
    for topk_idx in tl.static_range(kTopK):
        rows = tl.load(route_rows + token_indices * kTopK + topk_idx, valid, 0)
        values += tl.load(expert_outputs + rows * kHidden + channels, valid, 0).to(tl.float32)
    tl.store(output + offsets, values, valid)


def dispatch_quantized(activations, scales, token_indices, output, output_scales):
    num_rows, hidden = output.shape
    _dispatch_quantized[(triton.cdiv(num_rows * hidden, 256),)](
        activations.view(torch.uint8), scales, token_indices, output.view(torch.uint8), output_scales, num_rows, hidden,
        scales.stride(0), scales.stride(1), output_scales.stride(1), 256)


def swiglu_quantized(gate_up, routing_weights, output, output_scales, clamp):
    num_rows, intermediate = output.shape
    _swiglu_quantized[(num_rows, intermediate // 128)](
        gate_up, routing_weights, output, output_scales, num_rows, intermediate, output_scales.stride(1), clamp,
        num_warps=4, enable_fp_fusion=False)


def combine_routed(expert_outputs, route_rows, output, num_topk):
    num_tokens, hidden = output.shape
    _combine_routed[(triton.cdiv(num_tokens * hidden, 256),)](
        expert_outputs, route_rows, output, num_tokens, hidden, num_topk, 256)
